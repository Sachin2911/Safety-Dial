"""Concurrent scheduling and rental accounting without models or simulator access."""

import json
import sys
import threading
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_maniguard_phase2_train import FakeCoordinator

from helpers.maniguard_pilot_parallel import AllocationClock, run_parallel, validate_routes

ROUTES = [{'gpu': index, 'family': family, 'port': 18100 + index, 'slots': 1}
          for index, family in enumerate(['jar', 'stack', 'clutter'])]


def budget():
    prior = {'timestamp': 100.0, 'reserved_gpus': 1, 'allocated_gpu_seconds': 10.0,
             'counts': {'policy_control_steps': 20, 'policy_queries': 5}}
    allocations = [{'instance_id': 'old', 'gpus': 1, 'started_at': 90.0, 'ended_at': None},
                   {'instance_id': 'new', 'gpus': 4, 'started_at': 110.0, 'ended_at': None}]
    envelope = {'gpu_hours_ceiling': 1, 'policy_actions_max': 1000,
                'exhaustion_safe_successful_queries_max': 1000}
    return AllocationClock(prior, allocations, envelope, clock=lambda: 120.0)


def test_overlapping_allocations_count_setup_and_idle_without_refunding_old_gpu():
    value = budget()
    snap = value.snapshot({'policy_control_steps': 30})
    assert snap['allocated_gpu_seconds'] == 10 + 20 + 4 * 10
    assert snap['counts']['policy_control_steps'] == 50
    assert value.remaining_wall_seconds() == (3600 - 70) / 5
    value.allocations[0]['ended_at'] = 115
    assert value.snapshot()['allocated_gpu_seconds'] == 65


@pytest.mark.parametrize('mutation', ['missing_old', 'duplicate', 'future_end', 'negative_gpus'])
def test_rental_history_must_cover_prior_and_only_observed_allocation(mutation):
    original = budget()
    allocations = deepcopy(original.allocations)
    if mutation == 'missing_old':
        allocations.pop(0)
    elif mutation == 'duplicate':
        allocations.append(deepcopy(allocations[0]))
    elif mutation == 'future_end':
        allocations[0]['ended_at'] = 121
    else:
        allocations[1]['gpus'] = -1
    with pytest.raises(ValueError):
        AllocationClock(original.prior, allocations, original.envelope, clock=lambda: 120)


def test_budget_reserves_all_inflight_horizons_before_launch():
    value = budget()
    value.check(next_actions=980)
    with pytest.raises(RuntimeError, match='concurrent attempts'):
        value.check(next_actions=981)
    value.clock = lambda: 1000
    with pytest.raises(RuntimeError, match='GPU-hour'):
        value.check()


@pytest.mark.parametrize('mutation', ['missing_family', 'shared_port', 'shared_gpu', 'excess_slots'])
def test_routes_require_independent_fixed_family_servers(mutation):
    routes = deepcopy(ROUTES)
    if mutation == 'missing_family':
        routes.pop()
    elif mutation == 'shared_port':
        routes[1]['port'] = routes[0]['port']
    elif mutation == 'shared_gpu':
        routes[1]['gpu'] = routes[0]['gpu']
    else:
        routes[1]['slots'] = 3
    with pytest.raises(ValueError):
        validate_routes(routes, ['jar', 'stack', 'clutter'])


def harness(tmp_path):
    coordinator = FakeCoordinator({'families': ['jar', 'stack', 'clutter']})
    state = tmp_path / 'coordinator.json'
    coordinator.save(state)

    def accept(job, slot, output, value, error):
        return {'output': str(output), 'valid': error is None and value is True,
                'costs': {'policy_control_steps': 8, 'policy_queries': 2}, 'outcome': None}

    return coordinator, state, accept


def test_three_family_workers_launch_after_claim_and_parent_records_once(tmp_path):
    coordinator, state, accept = harness(tmp_path)
    barrier = threading.Barrier(3, timeout=5)
    started, lock = [], threading.Lock()

    def execute(job, slot, output):
        saved = json.loads(state.read_text())
        assert saved['claims'][job['job_id']] == str(output)
        assert slot['family'] == job['family']
        with lock:
            started.append(job['job_id'])
        if job['job_id'] != 'job-2':
            barrier.wait()
        return True

    result = run_parallel(coordinator, ROUTES, state, lambda job: str(tmp_path / job['job_id']),
                          execute, accept, budget())
    assert result['status'] == 'complete'
    assert set(started) == {'job-0', 'job-1', 'job-2', 'job-3'}
    assert len(started) == 4 and result['cost_totals']['policy_control_steps'] == 32
    assert not result['in_flight_claims']


def test_failure_drains_already_running_work_and_does_not_launch_next_job(tmp_path):
    coordinator, state, accept = harness(tmp_path)
    barrier = threading.Barrier(3, timeout=5)
    failed_processed = threading.Event()
    started = []

    def execute(job, slot, output):
        started.append(job['job_id'])
        barrier.wait()
        if job['job_id'] == 'job-0':
            raise RuntimeError('paid synthetic failure')
        assert failed_processed.wait(5)
        return True

    def record(job, slot, output, value, error):
        if error is not None:
            failed_processed.set()
        return accept(job, slot, output, value, error)

    result = run_parallel(coordinator, ROUTES, state, lambda job: str(tmp_path / job['job_id']),
                          execute, record, budget())
    assert result['status'] == 'blocked'
    assert set(started) == {'job-0', 'job-1', 'job-3'}
    assert result['physical_attempts_recorded'] == 3
    assert not result['in_flight_claims']


def test_takeover_rejects_claims_without_starting_workers(tmp_path):
    coordinator, state, accept = harness(tmp_path)
    coordinator.mark_started('job-0', tmp_path / 'old-attempt')
    with pytest.raises(ValueError, match='Reconcile'):
        run_parallel(coordinator, ROUTES, state, lambda job: str(tmp_path / job['job_id']),
                     lambda *_: pytest.fail('No automatic retry'), accept, budget())


def test_pause_launch_limit_preserves_remaining_slots_without_repeats(tmp_path):
    coordinator, state, accept = harness(tmp_path)
    launched = []

    def execute(job, *_):
        launched.append(job['job_id'])
        return True

    args = (coordinator, ROUTES, state, lambda job: str(tmp_path / job['job_id']),
            execute, accept, budget())
    result = run_parallel(*args, max_new_jobs=2)
    assert result['physical_attempts_recorded'] == 2
    assert result['status'] != 'complete' and not result['in_flight_claims']
    result = run_parallel(*args)
    assert result['status'] == 'complete' and len(set(launched)) == len(launched) == 4


def test_real_coordinator_parallel_completion_preserves_optimizer_updates(tmp_path, monkeypatch):
    from test_maniguard_phase2_coordinator import measured_row

    import helpers.maniguard_phase2_coordinator as module
    from helpers.maniguard_phase2_optim import record_hash
    from helpers.maniguard_phase2_protocol import build_pilot

    protocol = build_pilot()
    protocol['status'] = 'frozen'
    protocol['sha256'] = record_hash({k: v for k, v in protocol.items() if k != 'sha256'})
    monkeypatch.setattr(module, 'validate_protocol', lambda *_args, **_kwargs: {'ready_to_run': True})
    serial, parallel = module.PilotCoordinator(protocol), module.PilotCoordinator(protocol)
    # Complete the first search block and first PPO update on each coordinator,
    # retaining the original intervening stages and exact slot identities.
    stop_stage = next(i for i, stage in enumerate(serial.stage_plan) if stage['kind'] == 'ppo') + 1
    count = 0
    while serial.stage_index < stop_stage:
        for job in serial.next_jobs():
            output = str(tmp_path / 'episodes' / job['job_id'])
            serial.mark_started(job['job_id'], output)
            serial.record_result(job['job_id'], measured_row(job, output))
            count += 1
    value = budget()
    value.envelope['policy_actions_max'] = 1000000
    value.envelope['exhaustion_safe_successful_queries_max'] = 1000000
    result = run_parallel(parallel, ROUTES, tmp_path / 'parallel.json',
        lambda job: str(tmp_path / 'episodes' / job['job_id']), lambda *_: True,
        lambda job, slot, output, result, error: measured_row(job, output), value,
        max_new_jobs=count)
    assert not result['blocked_reasons'] and parallel.stage_index == serial.stage_index
    assert parallel.nominees == serial.nominees
    for name in serial.learners:
        assert serial.learners[name].state_dict() == parallel.learners[name].state_dict()
    for name in serial.searches:
        assert serial.searches[name].state_dict() == parallel.searches[name].state_dict()
