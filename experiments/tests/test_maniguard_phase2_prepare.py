"""Saved-record correction must not discard costs or repeat physical work."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/maniguard_phase2_prepare.py"
SPEC = importlib.util.spec_from_file_location("phase2_prepare_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_reaudit_retains_original_verdict_and_physical_costs(tmp_path, monkeypatch):
    job = {"job_id": "short-example", "baseline_executor": False}
    original = {"job": job, "status": "invalid", "returncode": 0,
                "protocol_sha256": "protocol", "output": str(tmp_path / "episode"),
                "elapsed_seconds": 52.0, "ledger": {"physics_steps": 278},
                "evidence": {"valid": False, "invalid_reasons": ["reader interpretation"]}}
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(original))
    monkeypatch.setattr(MODULE, "audit_attempt", lambda record: {"valid": True})

    def forbid_physics(*args, **kwargs):
        pytest.fail("Saved-record reanalysis must never launch an episode")

    monkeypatch.setattr(MODULE, "run_fresh_episode", forbid_physics)
    result = MODULE.reaudit_attempt(path, job, "protocol")
    assert result["status"] == "valid"
    assert result["elapsed_seconds"] == original["elapsed_seconds"]
    assert result["ledger"] == original["ledger"]
    assert result["audit_history"][0]["evidence"] == original["evidence"]
    assert result["audit_history"][0]["status"] == "invalid"
    assert result["reaudit"]["physics_rerun"] is False
    assert json.loads(path.read_text()) == result


@pytest.mark.parametrize("change", [{"status": "started"}, {"job": {}},
                                    {"protocol_sha256": "different"}])
def test_reaudit_rejects_unresolved_or_mismatched_attempts(tmp_path, monkeypatch, change):
    job = {"job_id": "example"}
    record = {"job": job, "status": "invalid", "returncode": 0,
              "protocol_sha256": "protocol", **change}
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(record))
    before = path.read_bytes()
    monkeypatch.setattr(MODULE, "audit_attempt", lambda _: pytest.fail("Not a terminal match"))
    with pytest.raises(ValueError, match="matching terminal"):
        MODULE.reaudit_attempt(path, job, "protocol")
    assert path.read_bytes() == before


def test_frozen_preparation_checksum_rejects_changed_assignments(tmp_path):
    real = SCRIPT.parents[2] / "configs/maniguard/phase2/preparation.json"
    protocol = MODULE.load_preparation(real)
    protocol["short_jobs"][0]["seed"] += 1
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="checksum"):
        MODULE.load_preparation(changed)


@pytest.mark.parametrize('process_status', [
    {}, {'timed_out': True}, {'interrupted': True},
    {'orphaned_descendants': True, 'error': 'descendants'},
])
def test_preparation_holds_lease_claims_before_spawn_and_retains_failure(
        tmp_path, monkeypatch, process_status):
    from contextlib import contextmanager

    output = tmp_path / 'output'
    reservation = tmp_path / 'reservation.json'
    reservation.write_text(json.dumps({'allocated_gpus': 1,
                                      'start_unix_seconds': MODULE.time.time() - 1}))
    monkeypatch.setattr(MODULE.sys, 'argv', [str(SCRIPT), '--jobs', 'short-jar-identity',
                        '--output', str(output), '--reservation', str(reservation)])
    monkeypatch.setattr(MODULE, 'load_model', lambda _: (MODULE.identity_genome(), {}))
    monkeypatch.setattr(MODULE, 'active_episode_processes', list)
    monkeypatch.setattr(MODULE, 'audit_attempt', lambda _: {'valid': True})
    lease = {'held': False}

    @contextmanager
    def acquire():
        assert not lease['held']
        lease['held'] = True
        try:
            yield
        finally:
            lease['held'] = False

    receipt_path = output / 'attempts/short-jar-identity.json'

    def run(command, log_path, timeout, *, env, on_started):
        assert lease['held']
        claim = json.loads(receipt_path.read_text())
        assert claim['status'] == 'started'
        assert claim['command'] == command
        on_started({'pid': 1234, 'returncode': None})
        assert json.loads(receipt_path.read_text())['process']['pid'] == 1234
        return {'pid': 1234, 'returncode': 0, 'process_elapsed_seconds': 12.5,
                'process_group_remaining': False, **process_status}

    monkeypatch.setattr(MODULE, 'execution_lease', acquire)
    monkeypatch.setattr(MODULE, 'run_fresh_episode', run)
    if process_status:
        with pytest.raises(RuntimeError, match='attempt/costs preserved'):
            MODULE.main()
    else:
        MODULE.main()
    assert not lease['held']
    receipt = json.loads(receipt_path.read_text())
    assert receipt['status'] == ('invalid' if process_status else 'valid')
    assert receipt['elapsed_seconds'] == 12.5
    assert receipt['process']['pid'] == 1234
    # Reanalysis may correct an outcome reader; it cannot erase a process failure.
    reaudit = MODULE.reaudit_attempt(receipt_path, receipt['job'], receipt['protocol_sha256'])
    assert reaudit['status'] == receipt['status']
    with pytest.raises(ValueError, match='Already claimed'):
        MODULE.main()


def test_active_episode_blocks_preparation_before_claim(tmp_path, monkeypatch):
    output = tmp_path / 'output'
    reservation = tmp_path / 'reservation.json'
    reservation.write_text(json.dumps({'allocated_gpus': 1,
                                      'start_unix_seconds': MODULE.time.time() - 1}))
    monkeypatch.setattr(MODULE.sys, 'argv', [str(SCRIPT), '--jobs', 'short-jar-identity',
                        '--output', str(output), '--reservation', str(reservation)])
    monkeypatch.setattr(MODULE, 'load_model', lambda _: (MODULE.identity_genome(), {}))
    monkeypatch.setattr(MODULE, 'active_episode_processes', lambda: [{'pid': 1234}])
    monkeypatch.setattr(MODULE, 'run_fresh_episode', lambda *a, **k: pytest.fail('must not spawn'))
    with pytest.raises(RuntimeError, match='already active'):
        MODULE._main()
    assert not list((output / 'attempts').glob('*.json'))


def test_malformed_terminal_ledger_does_not_hide_exit_or_elapsed_time(tmp_path, monkeypatch):
    from contextlib import nullcontext

    output = tmp_path / 'output'
    reservation = tmp_path / 'reservation.json'
    reservation.write_text(json.dumps({'allocated_gpus': 1,
                                      'start_unix_seconds': MODULE.time.time() - 1}))
    monkeypatch.setattr(MODULE.sys, 'argv', [str(SCRIPT), '--jobs', 'short-jar-identity',
                        '--output', str(output), '--reservation', str(reservation)])
    monkeypatch.setattr(MODULE, 'load_model', lambda _: (MODULE.identity_genome(), {}))
    monkeypatch.setattr(MODULE, 'execution_lease', nullcontext)
    monkeypatch.setattr(MODULE, 'active_episode_processes', list)
    monkeypatch.setattr(MODULE, 'audit_attempt', lambda _: {'valid': True})

    def run(command, log_path, timeout, *, env, on_started):
        episode = output / 'short-jar-identity'
        episode.mkdir()
        (episode / 'ledger.json').write_text('{incomplete')
        on_started({'pid': 1234})
        return {'pid': 1234, 'returncode': 0, 'process_elapsed_seconds': 12.5,
                'process_group_remaining': False}

    monkeypatch.setattr(MODULE, 'run_fresh_episode', run)
    with pytest.raises(RuntimeError, match='attempt/costs preserved'):
        MODULE.main()
    receipt = json.loads((output / 'attempts/short-jar-identity.json').read_text())
    assert receipt['status'] == 'invalid'
    assert receipt['returncode'] == 0
    assert receipt['elapsed_seconds'] == 12.5
    assert receipt['ledger_read_error']['type'] == 'JSONDecodeError'
