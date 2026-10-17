"""Service, clock and process tests use fakes; no service or simulator starts."""

import fcntl
import json
import os
import signal
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.maniguard_phase2_optim import record_hash
from helpers.maniguard_phase2_services import (
    PRIOR_COUNTS,
    PRIOR_SCHEMA,
    PolicyService,
    ReservationBudget,
    ServiceFailure,
    active_episode_processes,
    atomic_json,
    execution_lease,
    file_hash,
    probe_policy_metadata,
    run_fresh_episode,
)


class Clock:
    def __init__(self, value=200.0):
        self.value = value

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def accounting(tmp_path, *, timestamp=200.0, start=100.0, counts=None):
    evidence = tmp_path / 'evidence.json'
    evidence.write_text('{"audited": true}\n')
    prior = {'schema': PRIOR_SCHEMA, 'timestamp': timestamp, 'reservation_started_at': start,
             'reserved_gpus': 1, 'wall_seconds': timestamp - start,
             'allocated_gpu_seconds': timestamp - start,
             'counts': {key: (counts or {}).get(key, 0) for key in PRIOR_COUNTS},
             'sources': [{'path': str(evidence), 'sha256': file_hash(evidence)}]}
    prior['sha256'] = record_hash(prior)
    return prior


def service_fake(tmp_path, *, probes, active=None, clock=None, command_error=False):
    clock = clock or Clock()
    commands = []
    expected = {'serve_config': 'test-config', 'checkpoint_revision': 'revision',
                'rng_isolation': 'per_episode', 'server_source_hashes': {'server': 'abc'}}
    sequence = iter(probes)

    def probe(timeout):
        assert 0 < timeout <= 30
        value = next(sequence)
        if isinstance(value, BaseException):
            raise value
        return expected if value == 'matching' else value

    def runner(command, **kwargs):
        assert 0 < kwargs['timeout'] <= 30
        commands.append(command)
        clock.advance(1)
        return SimpleNamespace(returncode=int(command_error), stdout='', stderr='fake failure')

    config = tmp_path / 'policy.conf'
    config.write_text('original-config\n')
    service = PolicyService(config_path=config, probe=probe, runner=runner,
                            scan=lambda: active or [], monotonic=clock, wall=clock,
                            sleep=clock.advance, lease_path=tmp_path / 'execution.lock')
    service.expected_metadata = lambda family: expected
    return service, commands, config, clock


def test_matching_handshake_reuses_without_any_service_mutation(tmp_path):
    service, commands, config, _ = service_fake(tmp_path, probes=['matching'])
    result = service.ensure_family('jar', tmp_path / 'audit')
    assert result['status'] == 'reused' and result['service_changed'] is False
    assert commands == [] and config.read_text() == 'original-config\n'
    assert json.loads((tmp_path / 'audit/service.json').read_text()) == result


def test_wrong_handshake_and_active_episode_fail_before_stop_or_config_write(tmp_path):
    service, commands, config, _ = service_fake(
        tmp_path, probes=[{'serve_config': 'wrong'}], active=[{'pid': 123}])
    with pytest.raises(ServiceFailure, match='while manipulation episodes'):
        service.ensure_family('jar', tmp_path / 'audit')
    record = json.loads((tmp_path / 'audit/service.json').read_text())
    assert record['status'] == 'failed' and record['active_episodes'] == [{'pid': 123}]
    assert commands == [] and config.read_text() == 'original-config\n'


def test_idle_switch_archives_previous_config_and_polls_exact_handshake(tmp_path):
    service, commands, config, clock = service_fake(
        tmp_path, probes=[ConnectionError('not ready'), {'checkpoint_revision': 'stale'}, 'matching'])
    before = clock()
    record = service.ensure_family('stack', tmp_path / 'audit', startup_timeout=60)
    assert commands == [['supervisorctl', 'stop', 'maniguard_policy'],
                        ['supervisorctl', 'reread'],
                        ['supervisorctl', 'update', 'maniguard_policy'],
                        ['supervisorctl', 'start', 'maniguard_policy']]
    assert '--family stack --phase2 --parallel-clients' in config.read_text()
    assert 'stdout_logfile=/dev/stdout' in config.read_text()
    assert (tmp_path / 'audit/previous-policy.conf').read_text() == 'original-config\n'
    assert record['status'] == 'started' and record['service_changed'] is True
    assert record['elapsed_seconds'] == clock() - before == 6
    assert record['metadata'] == record['expected']


def test_failed_supervisor_operation_is_durable_and_does_not_replace_config(tmp_path):
    service, commands, config, _ = service_fake(tmp_path, probes=[{}], command_error=True)
    with pytest.raises(ServiceFailure, match='Supervisor operation failed'):
        service.ensure_family('jar', tmp_path / 'audit')
    assert len(commands) == 1 and config.read_text() == 'original-config\n'
    record = json.loads((tmp_path / 'audit/service.json').read_text())
    assert record['commands'][0]['returncode'] == 1
    assert record['status'] == 'failed' and record['elapsed_seconds'] == 1


def test_service_deadline_includes_polling_and_saves_timeout(tmp_path):
    service, _, _, clock = service_fake(tmp_path, probes=[{}, {}, {}, {}, {}])
    before = clock()
    with pytest.raises(ServiceFailure, match='deadline expired'):
        service.ensure_family('jar', tmp_path / 'audit', startup_timeout=7)
    record = json.loads((tmp_path / 'audit/service.json').read_text())
    assert record['status'] == 'failed' and clock() - before == 7


def test_process_scan_identifies_legacy_and_phase2_but_not_unrelated_python(tmp_path):
    entries = {1: ['/python', '/workspace/maniguard_episode.py', '--scene', 'a'],
               2: ['/python', '/workspace/maniguard_phase2_episode.py'],
               3: ['/python', 'analysis.py', 'maniguard_episode.py.txt']}
    for pid, args in entries.items():
        directory = tmp_path / str(pid)
        directory.mkdir()
        (directory / 'cmdline').write_bytes(b'\0'.join(value.encode() for value in args))
        (directory / 'environ').write_text('DO_NOT_READ_SECRET')
    (tmp_path / 'self').mkdir()
    assert {row['pid'] for row in active_episode_processes(tmp_path)} == {1, 2}


def test_handshake_probe_bounds_an_external_client_and_ignores_prior_stdout():
    captured = []

    def runner(command, **kwargs):
        captured.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout='client startup\n{"revision":"ok"}\n', stderr='')

    assert probe_policy_metadata(18000, 2.5, runner=runner) == {'revision': 'ok'}
    assert captured[0][1]['timeout'] == 2.5
    assert captured[0][0][-1] == '18000'


def test_reservation_counts_one_continuous_gpu_clock_and_all_attempted_work(tmp_path):
    clock = Clock(230)
    prior = accounting(tmp_path, counts={'policy_control_steps': 50, 'policy_queries': 10})
    envelope = {'gpu_hours_ceiling': 1, 'policy_actions_max': 1000,
                'exhaustion_safe_successful_queries_max': 1000}
    budget = ReservationBudget(prior, envelope, wall=clock)
    snapshot = budget.check({'policy_control_steps': 20, 'policy_queries': 5,
                             'elapsed_seconds': 500, 'encoder_seconds': 500})
    assert snapshot['wall_seconds'] == snapshot['allocated_gpu_seconds'] == 130
    assert snapshot['counts']['policy_control_steps'] == 70
    assert snapshot['counts']['policy_queries'] == 15
    clock.advance(500)  # Counts idle/service/restart time exactly once, not durations above.
    assert budget.snapshot()['allocated_gpu_seconds'] == 630
    with pytest.raises(RuntimeError, match='Policy-action ceiling'):
        budget.check(next_actions=951)
    with pytest.raises(RuntimeError, match='Policy-query ceiling'):
        budget.check(next_queries=991)
    clock.advance(3000)
    with pytest.raises(RuntimeError, match='GPU-hour ceiling'):
        budget.check()


def test_prior_evidence_and_full_counter_set_are_required(tmp_path):
    prior = accounting(tmp_path)
    envelope = {'gpu_hours_ceiling': 1, 'policy_actions_max': 1000,
                'exhaustion_safe_successful_queries_max': 1000}
    del prior['counts']['encoder_images']
    prior['sha256'] = record_hash({key: value for key, value in prior.items() if key != 'sha256'})
    with pytest.raises(ValueError, match='every nonnegative attempted-work'):
        ReservationBudget(prior, envelope, wall=Clock())
    prior = accounting(tmp_path)
    Path(prior['sources'][0]['path']).write_text('changed')
    with pytest.raises(ValueError, match='source changed'):
        ReservationBudget(prior, envelope, wall=Clock())


def test_timeout_terminates_fresh_process_group_and_records_actual_elapsed(tmp_path):
    clock = Clock()
    signals, started = [], []

    class Process:
        pid = 4321
        returncode = None

        def wait(self, timeout):
            clock.advance(timeout)
            raise subprocess.TimeoutExpired('fake', timeout)

        def poll(self):
            return self.returncode

    process = Process()
    alive = [True]

    def popen(command, **kwargs):
        assert kwargs['start_new_session'] is True
        assert kwargs['stdin'] == subprocess.DEVNULL
        return process

    def killpg(pid, sig):
        signals.append((pid, sig))
        if sig == signal.SIGKILL:
            alive[0] = False
            process.returncode = -9

    record = run_fresh_episode(['fake', 'episode'], tmp_path / 'episode.log', 20,
                               popen=popen, killpg=killpg, group_alive=lambda pid: alive[0],
                               wall=clock, monotonic=clock, sleep=clock.advance,
                               on_started=started.append)

    assert record['timed_out'] and record['returncode'] == -9
    assert record['process_elapsed_seconds'] == pytest.approx(25)
    assert signals == [(4321, signal.SIGTERM), (4321, signal.SIGKILL)]
    assert started[0]['pid'] == 4321 and started[0]['returncode'] is None


def test_process_start_failure_and_interruption_remain_accountable(tmp_path):
    def missing(*args, **kwargs):
        raise FileNotFoundError('fake executable')

    record = run_fresh_episode(['missing'], tmp_path / 'missing.log', 10, popen=missing)
    assert record['pid'] is None and record['returncode'] is None
    assert 'FileNotFoundError' in record['error']
    assert record['process_elapsed_seconds'] >= 0


def test_execution_lease_is_reentrant_and_blocks_other_file_descriptions(tmp_path):
    path = tmp_path / 'execution.lock'
    with (execution_lease(path), execution_lease(path), path.open('a+') as other,
          pytest.raises(BlockingIOError)):
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with path.open('a+') as other:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (pytest.raises(RuntimeError, match='owns the manipulation execution lease'),
              execution_lease(path)):
            pytest.fail('Contended lease must never yield')
    with execution_lease(path):
        assert json.loads(path.read_text())['pid'] > 0


def test_exited_leader_does_not_leave_descendants_running(tmp_path):
    clock, signals, alive = Clock(), [], [True]
    process = SimpleNamespace(pid=5678, poll=lambda: 0, wait=lambda timeout: 0)

    def killpg(pid, sig):
        signals.append((pid, sig))
        alive[0] = False

    record = run_fresh_episode(['fake'], tmp_path / 'orphan.log', 20,
                               popen=lambda *args, **kwargs: process,
                               killpg=killpg, group_alive=lambda pid: alive[0],
                               wall=clock, monotonic=clock, sleep=clock.advance)
    assert record['returncode'] == 0 and record['orphaned_descendants'] is True
    assert record['descendants_observed_after_leader_exit'] is True
    assert record['transient_descendants'] is False
    assert record['descendant_exit_grace_seconds'] == pytest.approx(5)
    assert record['process_elapsed_seconds'] == pytest.approx(5)
    assert 'descendants remained' in record['error']
    assert record['process_group_remaining'] is False
    assert signals == [(5678, signal.SIGTERM)]


def test_eof_cleanup_descendants_can_exit_naturally_without_invalidation(tmp_path):
    clock, signals = Clock(), []
    started = clock()
    process = SimpleNamespace(pid=5678, poll=lambda: 0, wait=lambda timeout: 0)
    record = run_fresh_episode(
        ['fake'], tmp_path / 'natural.log', 20,
        popen=lambda *args, **kwargs: process,
        killpg=lambda pid, sig: signals.append((pid, sig)),
        group_alive=lambda pid: clock() - started < .25,
        wall=clock, monotonic=clock, sleep=clock.advance,
    )
    assert record['returncode'] == 0 and not record.get('error')
    assert record['descendants_observed_after_leader_exit'] is True
    assert record['transient_descendants'] is True and not record.get('orphaned_descendants')
    assert record['process_group_remaining'] is False
    assert .25 <= record['descendant_exit_grace_seconds'] <= .35
    assert record['process_elapsed_seconds'] == record['descendant_exit_grace_seconds']
    assert signals == []


def test_persistent_orphan_reaches_kill_after_two_bounded_graces(tmp_path):
    clock, signals, alive = Clock(), [], [True]
    process = SimpleNamespace(pid=5678, poll=lambda: 0, wait=lambda timeout: 0)

    def killpg(pid, sig):
        signals.append((pid, sig))
        if sig == signal.SIGKILL:
            alive[0] = False

    record = run_fresh_episode(
        ['fake'], tmp_path / 'persistent.log', 20,
        popen=lambda *args, **kwargs: process, killpg=killpg,
        group_alive=lambda pid: alive[0], wall=clock, monotonic=clock, sleep=clock.advance,
        termination_grace=2,
    )
    assert record['returncode'] == 0 and record['orphaned_descendants']
    assert record['error'] and not record['transient_descendants']
    assert record['descendant_exit_grace_seconds'] == pytest.approx(2)
    assert record['process_elapsed_seconds'] == pytest.approx(4)
    assert record['process_group_remaining'] is False
    assert signals == [(5678, signal.SIGTERM), (5678, signal.SIGKILL)]


def test_group_surviving_kill_still_invalid_after_natural_grace(tmp_path):
    clock, signals = Clock(), []
    process = SimpleNamespace(pid=5678, poll=lambda: 0, wait=lambda timeout: 0)
    record = run_fresh_episode(
        ['fake'], tmp_path / 'remaining.log', 20,
        popen=lambda *args, **kwargs: process,
        killpg=lambda pid, sig: signals.append((pid, sig)),
        group_alive=lambda pid: True, wall=clock, monotonic=clock, sleep=clock.advance,
        termination_grace=2,
    )
    assert record['returncode'] == 0 and record['orphaned_descendants']
    assert record['process_group_remaining'] is True
    assert 'did not exit after SIGKILL' in record['error']
    assert record['process_elapsed_seconds'] == pytest.approx(6)
    assert signals == [(5678, signal.SIGTERM), (5678, signal.SIGKILL)]


def test_interrupt_still_cleans_group_and_returns_auditable_record(tmp_path):
    alive, signals = [True], []

    def interrupted(timeout):
        raise KeyboardInterrupt('supervisor stop')

    def killpg(pid, sig):
        signals.append((pid, sig))
        alive[0] = False

    process = SimpleNamespace(pid=5678, wait=interrupted,
                              poll=lambda: None if alive[0] else -15)
    record = run_fresh_episode(['fake'], tmp_path / 'interrupted.log', 20,
                               popen=lambda *args, **kwargs: process,
                               killpg=killpg, group_alive=lambda pid: alive[0])
    assert record['interrupted'] and record['returncode'] == -15
    assert record['process_group_remaining'] is False
    assert signals == [(5678, signal.SIGTERM)]


def test_atomic_record_flushes_data_and_directory_before_return(tmp_path, monkeypatch):
    modes = []
    original = os.fsync

    def synced(fd):
        modes.append(os.fstat(fd).st_mode)
        original(fd)

    monkeypatch.setattr(os, 'fsync', synced)
    path = tmp_path / 'attempt.json'
    atomic_json(path, {'claimed': True})
    assert json.loads(path.read_text()) == {'claimed': True}
    assert len(modes) == 2 and stat.S_ISREG(modes[0]) and stat.S_ISDIR(modes[1])


def test_managed_policy_capture_preserves_failure_and_platform_stdout(tmp_path):
    import shlex
    import subprocess

    from helpers.maniguard_phase2_services import render_policy_config

    root = tmp_path / 'root with spaces'
    wrapper = root / 'scripts/maniguard/serve.sh'
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text('#!/bin/sh\necho policy-startup-fixture >&2\nexit 7\n')
    wrapper.chmod(0o755)
    output = tmp_path / 'policy transcript.log'
    config = render_policy_config('clutter', root=root, log_path=output)
    command = next(line.removeprefix('command=') for line in config.splitlines()
                   if line.startswith('command='))
    result = subprocess.run(shlex.split(command), text=True, capture_output=True, check=False)
    assert result.returncode == 7
    assert result.stdout == output.read_text() == 'policy-startup-fixture\n'
    assert 'stdout_logfile=/dev/stdout' in config
    assert 'stdout_logfile_maxbytes=0' in config
