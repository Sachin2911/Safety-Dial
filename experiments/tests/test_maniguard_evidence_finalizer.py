"""Terminal preservation orchestration tests; no simulator or provider calls."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import finalize_maniguard_evidence as target


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def process_record(process):
    return {'pid': process.pid,
            'start_ticks': (Path('/proc') / str(process.pid) / 'stat').read_text().split()[21]}


def test_live_identity_rejects_wrong_command_and_ignores_reused_pid():
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    try:
        record = process_record(process)
        assert target.live(record, 'time.sleep')
        with pytest.raises(ValueError, match='command does not match'):
            target.live(record, 'wrong-command')
        assert not target.live({**record, 'start_ticks': '0'}, 'time.sleep')
    finally:
        process.terminate()
        process.wait()


def test_terminal_status_does_not_override_live_trainer(tmp_path, monkeypatch):
    write(tmp_path / 'run/status.json', {'status': 'completed'})
    monkeypatch.setattr(target, 'live', lambda *args: True)
    assert not target.require_terminal(tmp_path, 'run', {})


def test_missing_trainer_with_running_state_requires_recovery(tmp_path, monkeypatch):
    write(tmp_path / 'run/status.json', {'status': 'running'})
    monkeypatch.setattr(target, 'live', lambda *args: False)
    with pytest.raises(ValueError, match='without a terminal status'):
        target.require_terminal(tmp_path, 'run', {})


def test_mirror_stop_waits_for_verified_boundary(tmp_path):
    script = ('import time\n'
              'while True: time.sleep(0.1)\n'
              '# mirror_maniguard_v3_evidence.py')
    process = subprocess.Popen([sys.executable, '-c', script])
    state = tmp_path / 'state.json'
    write(state, {'pending': {'unverified': True}})
    try:
        record = process_record(process)
        with pytest.raises(TimeoutError, match='left running'):
            target.stop_mirror(record, state, timeout=0.05)
        assert process.poll() is None
        assert (Path('/proc') / str(process.pid) / 'stat').read_text().split()[2] != 'T'
        write(state, {'pending': None})
        target.stop_mirror(record, state)
        process.wait(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_secret_scan_includes_binary_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv('HF_TOKEN', 'synthetic-sensitive-value')
    (tmp_path / 'checkpoint.bin').write_bytes(b'\0synthetic-sensitive-value\0')
    with pytest.raises(ValueError, match='Known credential'):
        target.check_secrets(tmp_path)


def fixture(tmp_path, complete):
    project = tmp_path / 'project'
    upstream = tmp_path / 'checkout/maniguard/eval/benchmark.py'
    upstream.parent.mkdir(parents=True)
    upstream.write_text('# unchanged\n')
    write(project / 'run/status.json', {
        'status': 'completed' if complete else 'budget_exhausted',
        'generation': 1 if complete else 0,
        'jobs': {str(i): {'valid': True} for i in range(7 if complete else 2)},
        'validation': {}, 'active': None})
    write(project / 'run/config.json', {
        'search': {'generations': 1, 'population': 4, 'tasks': [0]},
        'validation': {'tasks': [3], 'policy_seeds': [1]}})
    write(project / 'service-.json', {'pid': 99999999})
    # The legacy seal API derives service names from run-* names.
    project.joinpath('run').rename(project / 'run-test')
    (project / 'service-.json').rename(project / 'service-test.json')
    write(project / 'run-test/generated/evaluator-source.json', {
        'checkout': str(upstream.parents[2]),
        'upstream_sha256': hashlib.sha256(upstream.read_bytes()).hexdigest()})
    write(project / 'run-test/episodes/failed/outcome.json', {'valid': False})
    mirror = tmp_path / 'mirror'
    write(mirror / 'state.json', {'pending': None})
    write(mirror / 'receipts/snapshot.json', {'verification_complete': True})
    write(mirror / 'snapshots/snapshot/index.json', {'files': {}})
    return project, mirror


@pytest.mark.parametrize('complete', [False, True])
def test_downloaded_seal_verified_and_upload_retry_does_not_reseal(tmp_path, monkeypatch, complete):
    project, mirror = fixture(tmp_path, complete)
    output = tmp_path / 'output'
    monkeypatch.setattr(target, 'live', lambda *args: False)

    def audit(project, run, output, incomplete):
        write(output, {'final': True, 'status': 'completed' if complete else 'budget_exhausted',
                       'valid_replayed_episodes': 7 if complete else 2, 'declared_episodes': 7,
                       'complete_schedule': complete})
    monkeypatch.setattr(target, 'audit', audit)
    calls = []

    def upload(release, verification, name, receipt_path):
        calls.append(name)
        if len(calls) == 1:
            raise TimeoutError('synthetic network interruption')
        shutil.copytree(release, verification / 'prefix')
        return {'prefix': 'prefix', 'complete_schedule': complete,
                'private': True, 'verification_complete': True}
    monkeypatch.setattr(target, 'upload_and_verify', upload)
    with pytest.raises(TimeoutError, match='network interruption'):
        target.finalize(project, 'run-test', mirror, output, {}, {}, {})
    seal_before = (project / 'remote-seal.json').read_bytes()
    result = target.finalize(project, 'run-test', mirror, output, {}, {}, {})
    assert (project / 'remote-seal.json').read_bytes() == seal_before
    assert result['independent_seal_verification']['local_verified']
    assert result['complete_schedule'] == complete
    assert not result['deletion_requested'] and not result['provider_absence_confirmed']
    assert (output / 'sealed-download/run-test/episodes/failed/outcome.json').exists()
    assert (output / 'sealed-download/supervision/private-mirror-closeout/receipts/snapshot.json')\
        .exists()


def test_sealed_project_mutation_blocks_retry(tmp_path, monkeypatch):
    project, mirror = fixture(tmp_path, False)
    output = tmp_path / 'output'
    output.mkdir()
    write(output / 'seal-identity.json', {'remote_seal_sha256': 'wrong'})
    write(project / 'remote-seal.json', {'files': {}})
    monkeypatch.setattr(target, 'live', lambda *args: False)
    with pytest.raises(ValueError, match='seal changed'):
        target.finalize(project, 'run-test', mirror, output, {}, {}, {})
