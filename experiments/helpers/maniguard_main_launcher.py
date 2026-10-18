"""Managed main-study parent wiring, with exact claims and an independent child audit.

The CLI has no synthetic backend flag. The child must pass its pure capability
preflight before any service operation. Until a reviewed pre-advance physics driver
is available, that gate deliberately refuses real execution. Injected dependencies
in unit tests exercise orchestration only and provide no performance evidence.
"""

import fcntl
import hashlib
import json
import math
import os
import secrets
import subprocess
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from pathlib import Path

from helpers import maniguard_main_audit_worker as audit_worker
from helpers import maniguard_main_protocol as protocol_api
from helpers import maniguard_phase2_services as services
from helpers.maniguard_main_dispatch import (
    WORK_COUNTS,
    DispatchPersistenceError,
    MainDispatch,
    _atomic,
    _check_signed,
    _counts,
    _evidence,
    _known,
    _signed,
)
from helpers.maniguard_phase2_optim import record_hash

RUNTIME_SCHEMA = 'maniguard-main-runtime-contract-v1'
CLAIM_SCHEMA = 'maniguard-main-live-claim-v1'
REQUEST_SCHEMA = 'maniguard-main-episode-request-v1'
PROCESS_SCHEMA = 'maniguard-main-process-v1'
MAIN_PROFILES = {'phase2_learned', 'phase2_identity', 'phase2_fixed', 'phase2_contextual',
                 'aac_joint20', 'aac_fixed8_joint20', 'benchmark_original'}
PARENT_SOURCES = (
    'experiments/helpers/maniguard_main_launcher.py',
    'experiments/helpers/maniguard_main_dispatch.py',
    'experiments/helpers/maniguard_main_protocol.py',
    'experiments/helpers/maniguard_main_episode.py',
    'experiments/helpers/maniguard_main_results.py',
    audit_worker.SOURCE_PATH,
    audit_worker.SCRIPT_PATH,
    'experiments/helpers/maniguard_phase2_services.py',
    'experiments/scripts/maniguard_main_episode.py',
    'experiments/scripts/maniguard_main_run.py',
    *services.SERVER_SOURCES,
)


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _zero():
    return dict.fromkeys(WORK_COUNTS, 0)


def _hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_runtime_contract(protocol, reference, *, root):
    """All numerical allowances originate in a previously frozen source record."""
    if reference not in protocol['spec']['implementation_files']:
        raise ValueError('Runtime contract is not an exact frozen implementation binding')
    value = json.loads(protocol_api._file(reference, root).read_text())
    _check_signed(value)
    expected = {'schema', 'supported_profiles', 'episode_timeout_seconds', 'policy_startup_timeout_seconds',
                'termination_grace_seconds', 'cleanup_reserve_seconds', 'audit_timeout_seconds',
                'audit_cleanup_reserve_seconds', 'per_episode_startup_counts',
                'per_episode_failed_work_counts', 'backend', 'sha256'}
    if set(value) != expected or value['schema'] != RUNTIME_SCHEMA:
        raise ValueError('Unsupported explicit main runtime contract')
    profiles = value['supported_profiles']
    if (not isinstance(profiles, list) or not profiles or len(set(profiles)) != len(profiles)
            or not set(profiles) <= MAIN_PROFILES):
        raise ValueError('Runtime contract has unsupported or duplicate profiles')
    for key in ('episode_timeout_seconds', 'policy_startup_timeout_seconds',
                'termination_grace_seconds', 'cleanup_reserve_seconds',
                'audit_timeout_seconds', 'audit_cleanup_reserve_seconds'):
        if not _positive(value[key]):
            raise ValueError('Every runtime timeout must be explicit, finite and positive')
    if value['cleanup_reserve_seconds'] < 3 * value['termination_grace_seconds']:
        raise ValueError('Cleanup reserve must cover natural-exit, TERM and KILL grace')
    if value['audit_cleanup_reserve_seconds'] < 3 * value['termination_grace_seconds']:
        raise ValueError('Audit cleanup reserve must cover natural-exit, TERM and KILL grace')
    for key in ('per_episode_startup_counts', 'per_episode_failed_work_counts'):
        _counts(value[key])
        if value[key]['policy_control_steps'] != 0:
            raise ValueError('Startup/failure allowances cannot create extra policy actions')
    if not isinstance(value['backend'], dict):
        raise TypeError('Runtime driver provenance must be explicit')
    return value


def verify_parent_sources(protocol, *, root):
    refs = {str((Path(root) / item['path']).resolve()): item
            for item in protocol['spec']['implementation_files']}
    for name in PARENT_SOURCES:
        path = (Path(root) / name).resolve()
        ref = refs.get(str(path))
        if ref is None or protocol_api._file(ref, root) != path:
            raise ValueError('Executed parent/child/audit source is not frozen: ' + name)


def expected_handshake(job, protocol, *, root):
    refs = {str((Path(root) / item['path']).resolve()): item
            for item in protocol['spec']['implementation_files']}
    hashes = {}
    for name in services.SERVER_SOURCES:
        ref = refs.get(str((Path(root) / name).resolve()))
        if ref is None:
            raise ValueError('Policy-server implementation is not frozen: ' + name)
        protocol_api._file(ref, root)
        hashes[name] = ref['sha256']
    checkpoint = job['checkpoint']
    return {'serve_config': checkpoint['config'], 'checkpoint_revision': checkpoint['revision'],
            'checkpoint': str((Path(root) / 'data/maniguard/policies' / job['family'] /
                               checkpoint['step']).resolve()),
            'episode_seed_reset': 'study_episode_id', 'rng_isolation': 'per_episode',
            'server_source_hashes': hashes}


def verify_handshake(metadata, expected):
    if not isinstance(metadata, dict) or any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError('Actual policy handshake differs from the exact frozen job/server pins')
    provenance = metadata.get('checkpoint_provenance', {})
    if (provenance.get('verification') != 'all_declared_files_sha256_and_available_upstream_digests'
            or not isinstance(provenance.get('receipt_sha256'), str)
            or len(provenance['receipt_sha256']) != 64
            or type(provenance.get('verified_files')) is not int or provenance['verified_files'] <= 0
            or type(provenance.get('verified_bytes')) is not int or provenance['verified_bytes'] <= 0):
        raise ValueError('Actual policy checkpoint lacks complete byte-verification evidence')


class MainLiveDispatch(MainDispatch):
    """Distinct live state/evidence namespace; no synthetic callback loop is accepted."""

    schema = 'maniguard-main-dispatch-live-v1'
    execution_kind = 'managed_main_child'
    evidence_kind = 'main_audited'
    supported_profiles = MAIN_PROFILES

    def write_evidence(self, path, value):
        self._usable()
        path = Path(path)
        if not path.resolve().is_relative_to(self.directory.resolve()):
            raise ValueError('Parent evidence escapes its canonical dispatch directory')
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError('Durable parent evidence cannot be overwritten: ' + str(path))
            _atomic(path, value)
        except BaseException as error:
            self.persistence_failure = {'error_type': type(error).__name__, 'error': str(error),
                                        'last_committed_sha256': self._saved_sha}
            try:
                diagnostic = self.directory / ('persistence-failure-' + str(time.time_ns()) + '.json')
                _atomic(diagnostic, _signed({**self.persistence_failure,
                    'uncommitted_state': _evidence(self.state), 'artifact_path': str(path),
                    'uncommitted_artifact': _evidence(value), 'durability_uncertain': True}))
            except BaseException as diagnostic_error:  # noqa: BLE001 - preserve primary failure
                self.persistence_failure['diagnostic_error'] = str(diagnostic_error)
            if not isinstance(error, Exception):
                raise
            raise DispatchPersistenceError('Parent evidence persistence failed: ' + str(error)) from error

    def claim_with_allowances(self, job, contract):
        self._ready()
        startup = _counts(contract['per_episode_startup_counts'])
        failed = _counts(contract['per_episode_failed_work_counts'])
        slot = _counts(job['resource_envelope']['successful_work'])
        total = {key: slot[key] + startup[key] + failed[key] for key in WORK_COUNTS}
        issues = self._budget_issues(total)
        issues += self._budget_issues(startup, 'startup') + self._budget_issues(failed, 'failed_work')
        if issues:
            raise RuntimeError('; '.join(dict.fromkeys(issues)))
        claim = self.claim(job)
        events = {}
        for category, allowance in (('startup', startup), ('failed_work', failed)):
            event_id = job['episode_id'] + '/' + category
            if event_id in self.state['extra_events']:
                raise ValueError('Episode allowance already exists')
            events[category] = event_id
            self.state['extra_events'][event_id] = {
                'event_id': event_id, 'category': category, 'status': 'reserved',
                'reservation': allowance, 'known_counts': _zero(), 'unknown_counts': [], 'raw_counts': None}
        self.state['claims'][job['episode_id']]['allowance_events'] = events
        self._persist()
        return {**claim, 'allowance_events': events}

    def record_audited(self, claim, row):
        """Pay each physical count once; missing split evidence remains unresolved."""
        self._usable()
        row = _evidence(row)
        if not isinstance(row, dict):
            row = {'valid': False, 'costs': {}, 'cost_issues': ['Main reader returned no mapping']}
        raw = deepcopy(row)
        uncertain = row.get('unknown_cost_fields') or []
        if (not isinstance(uncertain, list) or any(name not in WORK_COUNTS for name in uncertain)
                or row.get('cost_issues') or row.get('missing_cost_fields')):
            uncertain = list(WORK_COUNTS)
        split = row.get('cost_breakdown')
        split_valid = isinstance(split, dict) and set(split) == {'slot', 'startup', 'failed_work'}
        if split_valid:
            known_parts = {part: _known(split[part]) for part in split}
            total, missing = _known(row.get('costs'))
            split_valid = not missing and not any(unknown for _, unknown in known_parts.values())
            split_valid = split_valid and all(total[key] == sum(part[0][key] for part in known_parts.values())
                                              for key in WORK_COUNTS)
        if not split_valid:
            # No invented attribution: retain the only available lower bound in
            # slot accounting, while every unallocated extra category stays unknown.
            lower, _ = _known(row.get('costs'))
            if isinstance(split, dict):
                parts = [_known(split.get(part))[0] for part in ('slot', 'startup', 'failed_work')]
                for key in WORK_COUNTS:
                    if any(key in part for part in parts):
                        lower[key] = max(lower.get(key, 0), sum(part.get(key, 0) for part in parts))
            split = {'slot': lower, 'startup': {}, 'failed_work': {}}
            uncertain = list(WORK_COUNTS)
            row['valid'] = False
            row['cost_issues'] = list(row.get('cost_issues') or []) + ['Missing or inconsistent cost breakdown']
        for part in ('startup', 'failed_work'):
            self.settle_extra(claim['allowance_events'][part], split[part],
                              issues=[] if split_valid else ['Cost attribution unresolved'],
                              unknown_fields=uncertain)
        exact = all(row.get(key) == value for key, value in claim['job'].items())
        reader_name = 'experiments/helpers/maniguard_main_results.py'
        reader_refs = [item for item in self.protocol['spec']['implementation_files']
                       if (self.root / item['path']).resolve() == (self.root / reader_name).resolve()]
        reader_hash = reader_refs[0]['sha256'] if len(reader_refs) == 1 else None
        valid = (row.get('valid') is True and exact and row.get('full_episode') is True
                 and row.get('short_check') is False and row.get('audit_reader_sha256') == reader_hash
                 and row.get('protocol_sha256') == self.protocol['sha256']
                 and row.get('output') == claim['output'] and split_valid and not uncertain)
        slot_costs = {**split['slot'], 'elapsed_seconds': (row.get('costs') or {}).get('elapsed_seconds')}
        retained = {**row, 'evidence_kind': self.evidence_kind, 'job': claim['job'], 'valid': valid,
                    'costs': slot_costs, 'audited_main_row': raw, 'unknown_cost_fields': uncertain}
        return self.record_result(claim['job']['episode_id'], retained)

    def report(self):
        value = super().report()
        value.update(execution_kind=self.execution_kind, physical_evidence_audited=False,
                     performance_results=False, live_execution_authorized=False,
                     note='Dispatch accounting is not a performance analysis; each child still requires exact authorization.')
        return value


@contextmanager
def inherited_execution_lease(dispatch, claim):
    """Borrow the existing reentrant machine lock and pass its open description."""
    path = dispatch.root / 'runs/maniguard/setup/execution.lock'
    with services.execution_lease(path):
        entry = services._EXECUTION_LEASES[str(path.resolve())]
        fd = os.dup(entry['stream'].fileno())
        nonce = secrets.token_hex(32)
        metadata = {'pid': os.getpid(), 'acquired_at': dispatch._now(),
                    'nonce': nonce, 'job_sha256': claim['job_sha256']}
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            os.ftruncate(fd, 0)
            os.write(fd, (json.dumps(metadata, sort_keys=True) + '\n').encode())
            os.fsync(fd)
            yield {'lease_path': str(path.resolve()), 'lease_fd': fd, 'lease_nonce': nonce,
                   'parent_pid': os.getpid()}
        finally:
            os.close(fd)


def verify_parent_binding(request, *, parent_pid=None):
    """Child-side verification before connection/import/output; not a substitute for authorize_job."""
    binding = request['parent_binding']
    actual_parent = os.getppid() if parent_pid is None else parent_pid
    if binding['parent_pid'] != actual_parent:
        raise ValueError('Main child is not a direct child of the claiming parent')
    claim_path = Path(binding['claim_path']).resolve(strict=True)
    dispatch_root = Path(binding['dispatch_root']).resolve(strict=True)
    if not claim_path.is_relative_to(dispatch_root):
        raise ValueError('Claim evidence is outside the bound dispatch directory')
    claim = json.loads(claim_path.read_text())
    _check_signed(claim)
    if (claim.get('schema') != CLAIM_SCHEMA or claim['sha256'] != binding['claim_sha256']
            or claim['parent_pid'] != actual_parent or claim['nonce'] != binding['lease_nonce']
            or claim['job'] != request['job'] or claim['job_sha256'] != record_hash(request['job'])
            or claim['output'] != request['output']
            or claim['reservation'] != request['reservation']['counts']):
        raise ValueError('Immutable claim differs from the exact child request')
    state_path = Path(claim['dispatch_state_path']).resolve(strict=True)
    if state_path != dispatch_root / 'state.json':
        raise ValueError('Claim uses a different durable state registry')
    state = json.loads(state_path.read_text())
    _check_signed(state)
    study_root = Path(state['study_root']).resolve(strict=True)
    canonical = (study_root / 'runs/maniguard/main-dispatch' / state['study_id']).resolve()
    if (dispatch_root != canonical or not canonical.is_relative_to(study_root)
            or state.get('schema') != MainLiveDispatch.schema
            or state.get('study_id') != state['protocol'].get('study_id')
            or state.get('protocol_sha256') != state['protocol'].get('sha256')):
        raise ValueError('Child claim registry is not the canonical live study directory')
    _check_signed(state['protocol'])
    anchor = json.loads((canonical / 'binding.json').read_text())
    _check_signed(anchor)
    binding_keys = ('schema', 'study_root', 'study_id', 'protocol_sha256', 'grid_sha256',
                    'prior_sha256', 'implementation_sha256', 'execution_kind')
    if any(anchor.get(key) != state.get(key) for key in binding_keys):
        raise ValueError('Child claim state differs from its immutable study anchor')
    expected_claim = canonical / 'attempts' / record_hash(request['job']) / 'claim.json'
    episodes = (canonical / 'episodes').resolve()
    expected_output = (episodes / request['job']['episode_id']).resolve()
    if (claim_path != expected_claim or not expected_output.is_relative_to(episodes)
            or claim['output'] != str(expected_output)):
        raise ValueError('Child claim or output differs from its canonical frozen-job path')
    current = state['claims'].get(request['job']['episode_id'], {})
    if (state.get('execution_kind') != 'managed_main_child' or current.get('status') != 'claimed'
            or current.get('job_sha256') != claim['job_sha256'] or current.get('output') != claim['output']
            or current.get('live_claim_sha256') != claim['sha256']
            or state.get('protocol_sha256') != claim['protocol_sha256']
            or current.get('reservation') != request['reservation']['counts']):
        raise ValueError('Child lacks a currently claimed live durable slot')
    for category, key in (('startup', 'startup_counts'), ('failed_work', 'failed_work_counts')):
        event_id = current.get('allowance_events', {}).get(category)
        event = state['extra_events'].get(event_id, {})
        if (event.get('status') != 'reserved' or event.get('category') != category
                or event.get('reservation') != request['reservation'][key]):
            raise ValueError('Child category allowance differs from its durable reservation')
    fd = binding['lease_fd']
    if type(fd) is not int or fd < 0:
        raise ValueError('Child needs an inherited machine execution lease descriptor')
    lease_path = Path(binding['lease_path']).resolve(strict=True)
    expected_path = Path(state['study_root']) / 'runs/maniguard/setup/execution.lock'
    descriptor, expected = os.fstat(fd), lease_path.stat()
    if lease_path != expected_path or (descriptor.st_dev, descriptor.st_ino) != (expected.st_dev, expected.st_ino):
        raise ValueError('Inherited descriptor does not identify the canonical execution lease')
    parent_fd = Path('/proc') / str(actual_parent) / 'fd' / str(fd)
    parent_stat = parent_fd.stat()
    if (parent_stat.st_dev, parent_stat.st_ino) != (expected.st_dev, expected.st_ino):
        raise ValueError('Lease descriptor is not held by the claiming parent')
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with lease_path.open('r') as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(probe, fcntl.LOCK_UN)
            raise ValueError('Inherited lease is not exclusively held')
    metadata = json.loads(os.pread(fd, 65536, 0))
    if (metadata.get('pid') != actual_parent or metadata.get('nonce') != binding['lease_nonce']
            or metadata.get('job_sha256') != claim['job_sha256']):
        raise ValueError('Inherited execution lease metadata differs from the claim')
    return claim


def _remaining_seconds(dispatch):
    budget = dispatch.protocol['spec']['budget']['allocated_gpu_hours_max'] * 3600
    used = dispatch.snapshot()['allocated_gpu_seconds']
    return (budget - used) / dispatch.prior['reserved_gpus']


def _require_time(dispatch, contract, *, before_service):
    needed = (contract['episode_timeout_seconds'] + contract['cleanup_reserve_seconds']
              + contract['audit_timeout_seconds'] + contract['audit_cleanup_reserve_seconds'])
    if before_service:
        needed += contract['policy_startup_timeout_seconds']
    if _remaining_seconds(dispatch) < needed:
        raise RuntimeError('Remaining continuous GPU reservation cannot fund the complete next attempt')


def _process_ok(process):
    return (process.get('returncode') == 0 and process.get('process_group_remaining') is False
            and not any(process.get(key) for key in ('error', 'cleanup_error', 'timed_out', 'interrupted',
                                                     'orphaned_descendants')))



def _failed_audit_row(claim, protocol, reason, *, elapsed=0):
    lower = audit_worker.lower_bound_snapshot(claim['output'], claim['job'])
    return {'valid': False, 'output': claim['output'], 'protocol_sha256': protocol['sha256'],
            'costs': {**lower['costs'], 'elapsed_seconds': elapsed},
            'cost_breakdown': lower['cost_breakdown'], 'unknown_cost_fields': list(WORK_COUNTS),
            'missing_cost_fields': list(WORK_COUNTS),
            'cost_issues': ['Independent audit incomplete; ledger counters are unaudited lower bounds'],
            'invalid_reasons': [reason], 'outcome': None, 'fallback_evidence': lower}


def _audit_in_worker(dispatch, claim, attempt, protocol_path, config_dir, contract, *,
                     python, runner, wall):
    """Run full replay off the parent thread, retaining uncertainty on every failure."""
    job, protocol, root = claim['job'], dispatch.protocol, dispatch.root
    try:
        cleanup = contract['audit_cleanup_reserve_seconds']
        timeout = contract['audit_timeout_seconds']
        if _remaining_seconds(dispatch) < timeout + cleanup:
            raise RuntimeError('Remaining GPU reservation cannot fund independent audit and cleanup')
        request_path, result_path = attempt / 'audit_request.json', attempt / 'audit_result.json'
        request = _signed({'schema': audit_worker.REQUEST_SCHEMA, 'root': str(root),
            'config_dir': str(config_dir), 'protocol_path': str(protocol_path), 'job': job,
            'output': claim['output'], 'process_receipt_path': str(attempt / 'main_process.json'),
            'result_path': str(result_path), 'lower_bound_path': str(attempt / 'audit_lower_bound.json'),
            'deadline': wall() + timeout})
        dispatch.write_evidence(request_path, request)
        command = [str(python), str(root / audit_worker.SCRIPT_PATH), '--request', str(request_path)]

        def started(record):
            dispatch.state['claims'][job['episode_id']]['audit_process'] = _evidence(record)
            dispatch._persist()

        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                           OPENBLAS_NUM_THREADS='1')
        process = runner(command, attempt / 'audit.log', timeout, env=environment,
                         termination_grace=contract['termination_grace_seconds'], on_started=started)
        if dispatch.persistence_failure is not None:
            raise DispatchPersistenceError('Audit process evidence failed to persist')
        dispatch.write_evidence(attempt / 'audit_process.json', _signed({
            'schema': 'maniguard-main-audit-process-v1', 'request_sha256': request['sha256'],
            'process': process}))
        if (not _process_ok(process) or process.get('command') != command
                or wall() >= request['deadline']):
            raise RuntimeError('Independent audit worker failed or exceeded its deadline')
        # Result summaries are small. Never parse an unbounded child artifact on
        # the parent thread after the worker's deadline has ended.
        with result_path.open('rb') as stream:
            raw = stream.read((4 << 20) + 1)
        if len(raw) > 4 << 20:
            raise ValueError('Independent audit result exceeds bounded summary size')
        result = json.loads(raw)
        _check_signed(result)
        expected = {'schema': audit_worker.RESULT_SCHEMA, 'protocol_sha256': protocol['sha256'],
                    'job_sha256': record_hash(job), 'output': claim['output'],
                    'request_sha256': request['sha256'],
                    'audit_worker_sha256': _hash_file(root / audit_worker.SOURCE_PATH),
                    'audit_reader_sha256': _hash_file(root / audit_worker.READER_PATH)}
        if set(result) != {*expected, 'row', 'sha256'} or any(
                result.get(key) != value for key, value in expected.items()):
            raise ValueError('Independent audit result differs from its exact request/source')
        if not isinstance(result['row'], dict):
            raise TypeError('Independent audit result is not a row mapping')
        if _remaining_seconds(dispatch) <= 0:
            raise RuntimeError('GPU reservation expired before independent audit acceptance')
        return result['row']
    except Exception as error:
        if dispatch.persistence_failure is not None:
            raise
        return _failed_audit_row(claim, protocol,
            f'Independent audit failure: {type(error).__name__}: {error}',
            elapsed=max(0.0, wall() - claim['claimed_at']))


def run_main(protocol_path, prior_path, runtime_reference, *, root, config_dir=None, resume=False,
             max_jobs=None, port=18000, python=sys.executable, service=None, runner=None, reader=None,
             preflight=None, validator=None, source_verifier=verify_parent_sources, wall=time.time,
             managed_check=None, audit_runner=None):
    """Run exact main jobs under one managed parent; dependencies are injectable for tests.

    Never retry a claim. The default child preflight currently refuses unsupported
    physics enforcement before claims, service switches or process launches.
    """
    root = Path(root).resolve(strict=True)
    config_dir = Path(config_dir) if config_dir is not None else root / 'configs/maniguard/phase2'
    protocol_path, prior_path = Path(protocol_path).resolve(strict=True), Path(prior_path).resolve(strict=True)
    protocol, prior = json.loads(protocol_path.read_text()), json.loads(prior_path.read_text())
    validate = validator or protocol_api.validate_main_protocol
    validate(protocol, root=root, config_dir=config_dir, require_frozen=True)
    contract = load_runtime_contract(protocol, runtime_reference, root=root)
    source_verifier(protocol, root=root)
    if managed_check is None:
        if os.environ.get('SUPERVISOR_ENABLED') != '1' or not os.environ.get('SUPERVISOR_PROCESS_NAME'):
            raise RuntimeError('Main execution parent must run as a supervisor-managed service')
    else:
        managed_check()
    if max_jobs is not None and (type(max_jobs) is not int or max_jobs <= 0):
        raise ValueError('max_jobs must be a positive stop-after count, never extra slots')
    if preflight is None:
        from helpers.maniguard_main_episode import preflight_backend
        preflight = preflight_backend
    from helpers import maniguard_main_results as audit_module
    jobs = protocol_api.compile_jobs(protocol)
    for job in jobs:
        if job['resource_contract']['profile'] not in audit_module.SUPPORTED_PROFILES:
            raise RuntimeError('Independent execution replay is not implemented for this controller profile')
        if job['resource_contract']['profile'] not in contract['supported_profiles']:
            raise ValueError('Frozen runtime contract does not support every declared controller')
        preflight(protocol, job, contract, root=root, config_dir=config_dir)
    if 'maniguard_main_episode.py' not in services.EPISODE_SCRIPTS:
        raise RuntimeError('Machine process scanner does not yet recognize the main child entry point')
    service = service or services.PolicyService(root=root, port=port,
                                                lease_path=root / 'runs/maniguard/setup/execution.lock')
    runner = runner or services.run_fresh_episode
    audit_runner = audit_runner or services.run_fresh_episode
    with MainLiveDispatch.open(protocol, prior, root=root, config_dir=config_dir, resume=resume,
                               clock=wall, validator=validator) as dispatch:
        completed = 0
        previous_family = None
        while max_jobs is None or completed < max_jobs:
            pending = dispatch.pending_jobs()
            if not pending:
                break
            job = next((item for item in pending if item['family'] == previous_family), pending[0])
            validate(protocol, root=root, config_dir=config_dir, require_frozen=True)
            source_verifier(protocol, root=root)
            load_runtime_contract(protocol, runtime_reference, root=root)
            preflight(protocol, job, contract, root=root, config_dir=config_dir)
            _require_time(dispatch, contract, before_service=True)
            claim = dispatch.claim_with_allowances(job, contract)
            attempt = dispatch.directory / 'attempts' / record_hash(job)
            process = None
            child_started = False
            runner_invoked = False
            try:
                with inherited_execution_lease(dispatch, claim) as lease:
                    if services.active_episode_processes():
                        raise RuntimeError('An existing manipulation episode prevents main dispatch')
                    expected = expected_handshake(job, protocol, root=root)
                    # Detect registry drift before ensure_family could switch a checkpoint.
                    if service.expected_metadata(job['family']) != expected:
                        raise ValueError('Policy service registry differs from the frozen job')
                    service_record = service.ensure_family(job['family'], attempt / 'service',
                        startup_timeout=contract['policy_startup_timeout_seconds'])
                    verify_handshake(service_record.get('metadata'), expected)
                    dispatch.write_evidence(attempt / 'main_service.json', _signed(service_record))
                    validate(protocol, root=root, config_dir=config_dir, require_frozen=True)
                    source_verifier(protocol, root=root)
                    _require_time(dispatch, contract, before_service=False)
                    live_claim = _signed({'schema': CLAIM_SCHEMA, 'protocol_sha256': protocol['sha256'],
                        'job': job, 'job_sha256': record_hash(job), 'output': claim['output'],
                        'reservation': claim['reservation'], 'dispatch_state_path': str(dispatch.path),
                        'parent_pid': lease['parent_pid'], 'nonce': lease['lease_nonce'],
                        'claimed_at': claim['claimed_at']})
                    claim_path = attempt / 'claim.json'
                    dispatch.write_evidence(claim_path, live_claim)
                    dispatch.state['claims'][job['episode_id']]['live_claim_sha256'] = live_claim['sha256']
                    dispatch._persist()
                    handshake_path = attempt / 'handshake.json'
                    dispatch.write_evidence(handshake_path, service_record['metadata'])
                    request_path = attempt / 'request.json'
                    command = [str(python), str(root / 'experiments/scripts/maniguard_main_episode.py'),
                               '--request', str(request_path)]
                    request = _signed({'schema': REQUEST_SCHEMA, 'command': command, 'protocol_path': str(protocol_path),
                        'runtime_contract': runtime_reference, 'job': job, 'output': claim['output'],
                        'parent_binding': {**lease, 'dispatch_root': str(dispatch.directory),
                            'claim_path': str(claim_path), 'claim_sha256': live_claim['sha256']},
                        'reservation': {'counts': claim['reservation'],
                            'startup_counts': contract['per_episode_startup_counts'],
                            'failed_work_counts': contract['per_episode_failed_work_counts'],
                            'deadline': min(wall() + contract['episode_timeout_seconds'],
                                claim['claimed_at'] + contract['policy_startup_timeout_seconds']
                                + contract['episode_timeout_seconds'])},
                        'policy': {'host': '127.0.0.1', 'port': port, 'handshake_path': str(handshake_path),
                                   'handshake_sha256': _hash_file(handshake_path)}})
                    dispatch.write_evidence(request_path, request)

                    def on_started(record, job_id=job['episode_id']):
                        nonlocal child_started
                        child_started = True
                        dispatch.state['claims'][job_id]['child_process'] = _evidence(record)
                        dispatch._persist()

                    child_timeout = request['reservation']['deadline'] - wall()
                    if child_timeout <= 0:
                        raise TimeoutError('Child deadline expired before process launch')
                    runner_invoked = True
                    process = runner(command, attempt / 'child.log', child_timeout,
                        popen=partial(subprocess.Popen, pass_fds=(lease['lease_fd'],)),
                        termination_grace=contract['termination_grace_seconds'], on_started=on_started)
                    if dispatch.persistence_failure is not None:
                        raise DispatchPersistenceError('Child process evidence failed to persist')
                    process_record = _signed({'schema': PROCESS_SCHEMA, 'protocol_sha256': protocol['sha256'],
                        'job_sha256': record_hash(job), 'output': claim['output'], 'process': process})
                    dispatch.state['claims'][job['episode_id']]['backend_receipt'] = process_record
                    dispatch._persist()
                    dispatch.write_evidence(attempt / 'main_process.json', process_record)
                    # Store parent evidence next to child output only when the child created it.
                    if Path(claim['output']).is_dir():
                        dispatch.write_evidence(Path(claim['output']) / 'main_process.json', process_record)
                    if reader is None:
                        row = _audit_in_worker(dispatch, claim, attempt, protocol_path, config_dir,
                            contract, python=python, runner=audit_runner, wall=wall)
                    else:
                        # Injectable synthetic reader only; production always uses
                        # the isolated CPU worker and a process-level deadline.
                        row = reader(claim['output'], protocol, job, process_receipt=process_record,
                                     root=root, config_dir=config_dir)
                    if not _process_ok(process):
                        row = {**row, 'valid': False,
                               'invalid_reasons': list(row.get('invalid_reasons') or []) + ['Main child process failed']}
                    result = dispatch.record_audited(claim, row)
            except BaseException as error:
                if dispatch.persistence_failure is not None:
                    raise
                if dispatch.state['claims'][job['episode_id']]['status'] == 'claimed':
                    # A started/unknown child may have spent anything in its reserved
                    # envelope. Preserve uncertainty instead of inventing zero work.
                    known_zero = not runner_invoked and not child_started and process is None
                    costs = {**(_zero() if known_zero else {}),
                             'elapsed_seconds': max(0.0, wall() - claim['claimed_at'])}
                    failure = {'valid': False, 'costs': costs, 'output': claim['output'],
                        'protocol_sha256': protocol['sha256'],
                        'invalid_reasons': [f'Parent/child/audit failure: {type(error).__name__}: {error}'],
                        'cost_issues': [] if known_zero else ['Unknown work after child dispatch'],
                        'missing_cost_fields': [] if known_zero else list(WORK_COUNTS),
                        'cost_breakdown': {part: _zero() for part in ('slot', 'startup', 'failed_work')}
                                          if known_zero else None}
                    if not known_zero:
                        failure = _failed_audit_row(claim, protocol, failure['invalid_reasons'][0],
                                                   elapsed=costs['elapsed_seconds'])
                    result = dispatch.record_audited(claim, failure)
                if not isinstance(error, Exception):
                    raise
                result = dispatch.report()
            completed += 1
            previous_family = job['family']
            if result['blocked_reasons']:
                return result
        return dispatch.report()
