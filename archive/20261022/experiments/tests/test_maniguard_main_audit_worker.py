"""CPU audit process body and bounded fallback tests; no subprocesses or real episodes."""

import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_maniguard_main_dispatch import counts, rebind
from test_maniguard_main_dispatch import fixture as base_fixture

from helpers import maniguard_main_audit_worker as api
from helpers import maniguard_main_protocol as protocol_api
from helpers.maniguard_main_dispatch import _signed
from helpers.maniguard_phase2_optim import record_hash


@pytest.fixture
def audit_fixture(tmp_path, monkeypatch):
    value = base_fixture.__wrapped__(tmp_path, monkeypatch)
    root = value['root']
    for name in (api.SOURCE_PATH, api.SCRIPT_PATH, api.READER_PATH):
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(api.ROOT / name, destination)
        value['protocol']['spec']['implementation_files'].append(protocol_api.source_reference(destination, root=root))
    jobs = protocol_api.compile_jobs(value['protocol'])
    value['protocol']['grid_sha256'] = record_hash(jobs)
    rebind(value)
    job = protocol_api.compile_jobs(value['protocol'])[0]
    protocol_path = root / 'protocol.json'
    protocol_path.write_text(json.dumps(value['protocol']))
    output = root / 'episode'
    output.mkdir()
    ledger = _signed({'schema': 'maniguard-main-costs-v1', 'job_sha256': record_hash(job),
        'counts': counts(policy_queries=2, physics_steps=7), 'elapsed_seconds': 2,
        'by_category': {'slot': counts(policy_queries=2, physics_steps=6),
                        'startup': counts(physics_steps=1), 'failed_work': counts()}})
    (output / 'main_costs.json').write_text(json.dumps(ledger))
    process_path = root / 'process.json'
    process_path.write_text(json.dumps(_signed({'schema': 'maniguard-main-process-v1',
        'protocol_sha256': value['protocol']['sha256'], 'job_sha256': record_hash(job),
        'output': str(output), 'process': {'returncode': 0}})))
    request = _signed({'schema': api.REQUEST_SCHEMA, 'root': str(root),
        'config_dir': str(root / 'configs/maniguard/phase2'), 'protocol_path': str(protocol_path),
        'job': job, 'output': str(output), 'process_receipt_path': str(process_path),
        'result_path': str(root / 'audit-result.json'), 'lower_bound_path': str(root / 'lower.json'),
        'deadline': 110.0})
    request_path = root / 'request.json'
    request_path.write_text(json.dumps(request))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    value.update(job=job, output=output, request=request, request_path=request_path,
                 reader=lambda *_args, **_kwargs: {'valid': True, 'costs': counts(policy_queries=2)})
    return value


def run(value, **kwargs):
    return api.run_request(value['request_path'], root=value['root'],
                           **{'reader': value['reader'], 'clock': value['clock'], **kwargs})


def test_worker_persists_lower_bound_before_reader_and_exact_result(audit_fixture):
    value = audit_fixture

    def reader(output, protocol, job, **kwargs):
        lower = json.loads(Path(value['request']['lower_bound_path']).read_text())
        assert lower['costs']['policy_queries'] == 2
        assert lower['unknown_cost_fields'] == list(api.WORK_COUNTS)
        assert not lower['independently_audited']
        assert output == value['output'] and job == value['job']
        assert kwargs['process_receipt']['protocol_sha256'] == protocol['sha256']
        return {'valid': True, 'costs': counts(policy_queries=2)}

    result = run(value, reader=reader)
    assert result['request_sha256'] == value['request']['sha256']
    assert result['row']['valid'] is True
    assert json.loads(Path(value['request']['result_path']).read_text()) == result


def test_audit_failure_preserves_lower_bound_without_valid_result(audit_fixture):
    def reader(*_args, **_kwargs):
        raise ValueError('Synthetic audit failed')

    with pytest.raises(ValueError, match='audit failed'):
        run(audit_fixture, reader=reader)
    assert not Path(audit_fixture['request']['result_path']).exists()
    assert json.loads(Path(audit_fixture['request']['lower_bound_path']).read_text())['costs']['physics_steps'] == 7


def test_expired_deadline_and_visible_gpu_rejected_before_reader(audit_fixture, monkeypatch):
    value = audit_fixture
    value['clock'].value = 111
    with pytest.raises(RuntimeError, match='deadline'):
        run(value, reader=lambda *_args, **_kwargs: pytest.fail('Do not audit'))
    value['clock'].value = 100
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    with pytest.raises(RuntimeError, match='hide GPUs'):
        run(value, reader=lambda *_args, **_kwargs: pytest.fail('Do not audit'))
    assert not Path(value['request']['lower_bound_path']).exists()


def test_reader_that_finishes_late_cannot_publish_success(audit_fixture):
    def slow(*_args, **_kwargs):
        audit_fixture['clock'].value = 111
        return {'valid': True}

    with pytest.raises(RuntimeError, match='before result publication'):
        run(audit_fixture, reader=slow)
    assert not Path(audit_fixture['request']['result_path']).exists()
    assert Path(audit_fixture['request']['lower_bound_path']).is_file()


def test_undeclared_case_and_changed_source_never_call_reader(audit_fixture):
    request = {**audit_fixture['request'], 'job': {**audit_fixture['job'], 'scene': 'undeclared/base'}}
    audit_fixture['request_path'].write_text(json.dumps(_signed(request)))
    with pytest.raises(ValueError, match='exact member'):
        run(audit_fixture, reader=lambda *_args, **_kwargs: pytest.fail('Do not audit'))
    audit_fixture['request_path'].write_text(json.dumps(audit_fixture['request']))
    (audit_fixture['root'] / api.SOURCE_PATH).write_text('tampered')
    with pytest.raises(ValueError, match='source changed'):
        run(audit_fixture, reader=lambda *_args, **_kwargs: pytest.fail('Do not audit'))


def test_bounded_fallback_does_not_scan_a_large_journal(audit_fixture, monkeypatch):
    value = audit_fixture
    (value['output'] / 'main_operations.jsonl').write_text('unreadable enormous journal')
    original = Path.open

    def reject_journal(path, *args, **kwargs):
        if path.name == 'main_operations.jsonl':
            pytest.fail('Fallback must not scan the unbounded operation journal')
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', reject_journal)
    result = api.lower_bound_snapshot(value['output'], value['job'])
    assert result['costs']['policy_queries'] == 2
    assert result['cost_breakdown']['startup']['physics_steps'] == 1
    assert result['unknown_cost_fields'] == list(api.WORK_COUNTS)


def test_oversize_or_wrong_job_ledger_stays_unknown_without_zero_filling(audit_fixture, monkeypatch):
    value = audit_fixture
    monkeypatch.setattr(api, 'LEDGER_READ_BYTES', 8)
    result = api.lower_bound_snapshot(value['output'], value['job'])
    assert result['costs'] == {} and result['issues']
    monkeypatch.setattr(api, 'LEDGER_READ_BYTES', 1 << 20)
    result = api.lower_bound_snapshot(value['output'], {**value['job'], 'seed': 999})
    assert result['costs'] == {} and result['issues']
