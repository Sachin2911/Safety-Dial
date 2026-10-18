"""Isolated CPU audit worker and bounded fallback accounting for a main episode.

Only an explicitly requested frozen job is opened. The managed parent enforces the
worker process deadline; this module never launches processes or calls a policy.
Fallback counters are reported lower bounds, never independently verified results.
"""

import hashlib
import json
import math
import os
import time
from pathlib import Path

from helpers import maniguard_main_protocol as protocol_api
from helpers.maniguard_main_dispatch import WORK_COUNTS, _atomic, _check_signed, _signed
from helpers.maniguard_phase2_optim import record_hash

ROOT = Path(__file__).resolve().parents[2]
REQUEST_SCHEMA = 'maniguard-main-audit-request-v1'
RESULT_SCHEMA = 'maniguard-main-audit-result-v1'
LOWER_SCHEMA = 'maniguard-main-audit-lower-bounds-v1'
LEDGER_READ_BYTES = 1 << 20  # Bound fallback parsing independently of episode length.
SOURCE_PATH = 'experiments/helpers/maniguard_main_audit_worker.py'
SCRIPT_PATH = 'experiments/scripts/maniguard_main_audit.py'
READER_PATH = 'experiments/helpers/maniguard_main_results.py'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _mapping(value):
    if not isinstance(value, dict):
        return {}
    return {name: number for name, number in value.items()
            if name in WORK_COUNTS and type(number) is int and number >= 0}


def lower_bound_snapshot(output, job):
    """Read at most one bounded ledger; every value remains explicitly unaudited.

    Do not scan the possibly huge operation journal synchronously after a killed
    audit. The reader worker owns full replay; a missing/stale ledger is unknown.
    """
    result = {'schema': LOWER_SCHEMA, 'job_sha256': record_hash(job), 'costs': {},
              'cost_breakdown': {name: {} for name in ('slot', 'startup', 'failed_work')},
              'unknown_cost_fields': list(WORK_COUNTS), 'independently_audited': False,
              'source': None, 'issues': []}
    path = Path(output) / 'main_costs.json'
    try:
        with path.open('rb') as stream:
            data = stream.read(LEDGER_READ_BYTES + 1)
        if len(data) > LEDGER_READ_BYTES:
            raise ValueError('Child ledger exceeds bounded fallback read size')
        ledger = json.loads(data)
        _check_signed(ledger)
        if ledger.get('schema') != 'maniguard-main-costs-v1' or ledger.get('job_sha256') != record_hash(job):
            raise ValueError('Fallback ledger belongs to a different episode')
        result['source'] = {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest()}
        result['costs'] = _mapping(ledger.get('counts'))
        elapsed = ledger.get('elapsed_seconds')
        if type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0:
            result['costs']['elapsed_seconds'] = elapsed
        categories = ledger.get('by_category')
        if isinstance(categories, dict):
            for category in result['cost_breakdown']:
                result['cost_breakdown'][category] = _mapping(categories.get(category))
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        result['issues'].append(type(error).__name__ + ': ' + str(error))
    return _signed(result)


def _inside(path, root):
    path = (root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Audit request path escapes the bound study root')
    return path


def run_request(request_path, *, root=ROOT, reader=None, validator=None, clock=time.time):
    """Audit one exact request; parent must separately supervise this process."""
    root = Path(root).resolve(strict=True)
    request_path = _inside(request_path, root)
    request = json.loads(request_path.read_text())
    _check_signed(request)
    keys = {'schema', 'protocol_path', 'job', 'output', 'process_receipt_path',
            'result_path', 'lower_bound_path', 'deadline', 'root', 'config_dir', 'sha256'}
    if set(request) != keys or request['schema'] != REQUEST_SCHEMA or request['root'] != str(root):
        raise ValueError('Unsupported exact CPU audit request')
    deadline = request['deadline']
    if type(deadline) not in (int, float) or not math.isfinite(deadline) or clock() >= deadline:
        raise RuntimeError('Audit deadline already expired')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Main audit worker must explicitly hide GPUs')
    protocol_path = _inside(request['protocol_path'], root)
    protocol = json.loads(protocol_path.read_text())
    config_dir = _inside(request['config_dir'], root)
    validate = validator or protocol_api.validate_main_protocol
    validate(protocol, root=root, config_dir=config_dir, require_frozen=True)
    authorization = protocol_api.authorize_job(protocol, request['job'], root=root, config_dir=config_dir)
    if authorization != {'authorized': True, 'protocol_sha256': protocol['sha256'],
                         'episode_id': request['job']['episode_id'], 'job_sha256': record_hash(request['job'])}:
        raise ValueError('Audit request differs from the exact frozen grid')
    refs = {str((root / item['path']).resolve()): item for item in protocol['spec']['implementation_files']}
    for relative in (SOURCE_PATH, SCRIPT_PATH, READER_PATH):
        path = root / relative
        reference = refs.get(str(path.resolve()))
        if reference is None or protocol_api._file(reference, root) != path.resolve():
            raise ValueError('Audit worker/script/reader is not frozen: ' + relative)
    if refs[str((root / SOURCE_PATH).resolve())]['sha256'] != _sha(__file__):
        raise ValueError('Running audit worker bytes differ from the frozen source')
    output = _inside(request['output'], root)
    result_path = _inside(request['result_path'], root)
    lower_path = _inside(request['lower_bound_path'], root)
    if result_path.exists() or lower_path.exists():
        raise ValueError('Audit artifacts cannot be reused or overwritten')
    process_path = _inside(request['process_receipt_path'], root)
    process = json.loads(process_path.read_text())
    _check_signed(process)
    if (process.get('schema') != 'maniguard-main-process-v1'
            or process.get('protocol_sha256') != protocol['sha256']
            or process.get('job_sha256') != record_hash(request['job'])
            or process.get('output') != str(output)):
        raise ValueError('Audit process receipt differs from its exact episode')
    _atomic(lower_path, lower_bound_snapshot(output, request['job']))
    if reader is None:
        from helpers.maniguard_main_results import read_main_episode
        reader = read_main_episode
    row = reader(output, protocol, request['job'], process_receipt=process, root=root, config_dir=config_dir)
    if clock() >= deadline:
        raise RuntimeError('Audit deadline expired before result publication')
    result = _signed({'schema': RESULT_SCHEMA, 'protocol_sha256': protocol['sha256'],
                      'job_sha256': record_hash(request['job']), 'output': str(output),
                      'request_sha256': request['sha256'], 'audit_worker_sha256': _sha(__file__),
                      'audit_reader_sha256': refs[str((root / READER_PATH).resolve())]['sha256'], 'row': row})
    _atomic(result_path, result)
    return result
