"""Concurrent lane dispatch reproduces the frozen serial optimizer and selection state."""

import json
import random
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_maniguard_phase2_coordinator import frozen_protocol, measured_row  # noqa: F401
from test_maniguard_pilot_parallel import budget

from helpers.maniguard_phase2_coordinator import PilotCoordinator
from helpers.maniguard_pilot_lanes import LaneCoordinator, lane_key
from helpers.maniguard_pilot_parallel import run_parallel

ROUTES = [{'gpu': index, 'family': family, 'port': 18100 + index, 'slots': 1}
          for index, family in enumerate(['jar', 'jar', 'stack', 'stack', 'clutter'])]


def utility(job):
    """Deterministic outcomes that depend on the dispatched genome and family."""
    return (1, 0, -2)[(int(job['actor_sha256'][:8], 16) + len(job['family'])) % 3]


def row_for(job, base):
    return measured_row(job, base / 'episodes' / job['job_id'], utility=utility(job))


def serial_reference(protocol, base):
    coordinator = PilotCoordinator(protocol)
    while coordinator.status not in ('blocked', 'complete'):
        for job in coordinator.next_jobs():
            coordinator.mark_started(job['job_id'], base / 'episodes' / job['job_id'])
            coordinator.record_result(job['job_id'], row_for(job, base))
    assert coordinator.status == 'complete'
    return coordinator


def drive(lanes, base, seed, *, stop_after=None, inflight=None):
    """Claim and finish random subsets across lanes, in random completion order."""
    rng = random.Random(seed)
    inflight = {} if inflight is None else inflight
    finished = 0
    while lanes.status not in ('blocked', 'complete'):
        pending = lanes.next_jobs()
        for job in rng.sample(pending, min(len(pending), rng.randint(1, 7))):
            lanes.mark_started(job['job_id'], base / 'episodes' / job['job_id'])
            inflight[job['job_id']] = job
        assert inflight, 'Lane dispatch stalled with nothing in flight'
        for job_id in rng.sample(sorted(inflight), min(len(inflight), rng.randint(1, 5))):
            lanes.record_result(job_id, row_for(inflight.pop(job_id), base))
            finished += 1
            if stop_after is not None and finished >= stop_after:
                return inflight
    return inflight


def assert_equivalent(serial, lanes):
    assert lanes.status == 'complete'
    state = lanes.canonical_state()
    reference = serial.state_dict()
    for key in ('stage_plan', 'stage_index', 'history', 'nominees', 'selected',
                'reference_metrics', 'searches', 'learners', 'claims', 'blocked_reasons'):
        assert state[key] == reference[key], key
    by_job = lambda ledger: sorted(ledger, key=lambda entry: entry['job_id'])
    assert by_job(state['ledger']) == by_job(reference['ledger'])
    restored = PilotCoordinator.from_state_dict(json.loads(json.dumps(state)))
    assert restored.report() == serial.report()


@pytest.fixture
def serial(frozen_protocol, tmp_path):  # noqa: F811
    return serial_reference(frozen_protocol, tmp_path)


def test_lane_keys_split_runs_and_shared_references(frozen_protocol):  # noqa: F811
    coordinator = PilotCoordinator(frozen_protocol)
    keys = {lane_key(stage) for stage in coordinator.stage_plan}
    assert keys == {f'{method}/s{seed}' for method in ('evolution', 'random', 'ppo')
                    for seed in frozen_protocol['optimizer_seeds']} | {'references'}


@pytest.mark.parametrize('seed', [0, 1])
def test_random_concurrent_interleavings_match_frozen_serial_state(
        frozen_protocol, serial, tmp_path, seed):  # noqa: F811
    lanes = LaneCoordinator(PilotCoordinator(frozen_protocol))
    drive(lanes, tmp_path, seed)
    assert_equivalent(serial, lanes)
    order = lanes.completion_order
    assert order != sorted(order), 'The test must exercise a non-serial block order'


def test_lane_state_round_trips_with_inflight_claims(frozen_protocol, serial, tmp_path):  # noqa: F811
    lanes = LaneCoordinator(PilotCoordinator(frozen_protocol))
    inflight = drive(lanes, tmp_path, 7, stop_after=150)
    assert inflight and lanes.report()['in_flight_claims']
    path = tmp_path / 'lanes.json'
    lanes.save(path)
    restored = LaneCoordinator.load(path)
    assert restored.state_dict() == lanes.state_dict()
    assert {job['job_id'] for job in restored.next_jobs()}.isdisjoint(inflight)
    drive(restored, tmp_path, 8, inflight=inflight)
    assert_equivalent(serial, restored)
    restored.save(path)
    with pytest.raises(ValueError, match='changed'):
        lanes.save(path)


def test_takeover_from_partial_frozen_block_preserves_recorded_results(
        frozen_protocol, serial, tmp_path):  # noqa: F811
    base = PilotCoordinator(frozen_protocol)
    first = base.next_jobs()
    for job in [job for job in first if job['family'] == 'clutter']:
        base.mark_started(job['job_id'], tmp_path / 'episodes' / job['job_id'])
        base.record_result(job['job_id'], row_for(job, tmp_path))
    assert base.stage_index == 0 and len(base.active_results) == 8
    lanes = LaneCoordinator(PilotCoordinator.from_state_dict(base.state_dict()))
    pending = lanes.next_jobs()
    assert not {job['job_id'] for job in pending} & set(base.active_results)
    assert len({lane_key(lanes.base.stage_plan[lanes._active_stage_of(job['job_id'])])
                for job in pending}) == 7
    drive(lanes, tmp_path, 3)
    assert_equivalent(serial, lanes)


def test_takeover_refuses_claims_or_blocks(frozen_protocol, tmp_path):  # noqa: F811
    base = PilotCoordinator(frozen_protocol)
    job = base.next_jobs()[0]
    base.mark_started(job['job_id'], tmp_path / 'one')
    with pytest.raises(ValueError, match='in-flight'):
        LaneCoordinator(base)


def test_invalid_attempt_blocks_every_lane_but_drains_inflight(frozen_protocol, tmp_path):  # noqa: F811
    lanes = LaneCoordinator(PilotCoordinator(frozen_protocol))
    jobs = lanes.next_jobs()
    lane_jobs = {}
    for job in jobs:
        lane_jobs.setdefault(lanes._active_stage_of(job['job_id']), job)
    first, second = list(lane_jobs.values())[:2]
    for job in (first, second):
        lanes.mark_started(job['job_id'], tmp_path / 'episodes' / job['job_id'])
    bad = row_for(first, tmp_path)
    bad['valid'] = False
    assert lanes.record_result(first['job_id'], bad)['status'] == 'blocked'
    assert lanes.next_jobs() == []
    report = lanes.record_result(second['job_id'], row_for(second, tmp_path))
    assert report['status'] == 'blocked' and report['physical_attempts_recorded'] == 2
    assert report['accepted_episode_slots'] == 1 and not report['in_flight_claims']
    assert lanes.completion_order == []


def test_critical_lanes_are_offered_first(frozen_protocol):  # noqa: F811
    lanes = LaneCoordinator(PilotCoordinator(frozen_protocol))
    jobs = lanes.next_jobs()
    assert jobs[0]['method'] == 'ppo' and {job['method'] for job in jobs[:6]} == {'ppo'}
    assert jobs[-1]['stage'] == 'reference_validation'


def test_run_parallel_drives_lanes_without_duplicate_claims(frozen_protocol, tmp_path):  # noqa: F811
    lanes = LaneCoordinator(PilotCoordinator(frozen_protocol))
    state = tmp_path / 'lanes.json'
    lanes.save(state)
    seen, lock = [], threading.Lock()

    def execute(job, slot, output):
        saved = json.loads(state.read_text())
        assert job['job_id'] in saved['claims'] and slot['family'] == job['family']
        with lock:
            seen.append(job['job_id'])
        return True

    value = budget()
    value.envelope.update(policy_actions_max=10**7, exhaustion_safe_successful_queries_max=10**7)
    report = run_parallel(lanes, ROUTES, state, lambda job: str(tmp_path / 'episodes' / job['job_id']),
                          execute, lambda job, slot, output, result, error: row_for(job, tmp_path),
                          value, max_new_jobs=60)
    assert len(seen) == len(set(seen)) == 60 and not report['in_flight_claims']
    assert report['dispatch'] == 'concurrent-lanes' and report['accepted_episode_slots'] == 60
    assert LaneCoordinator.load(state).state_dict() == lanes.state_dict()


def test_priority_reorders_dispatch_only(frozen_protocol, serial, tmp_path):  # noqa: F811
    with pytest.raises(ValueError, match='priority'):
        LaneCoordinator(PilotCoordinator(frozen_protocol), priority=['unknown'])
    order = ['references', 'random/s17', 'evolution/s17']
    lanes = LaneCoordinator(PilotCoordinator(frozen_protocol), priority=order)
    jobs = lanes.next_jobs()
    assert {job['stage'] for job in jobs[:18]} == {'reference_validation'}
    assert {job['method'] for job in jobs[18:42]} == {'random'}
    assert {job['method'] for job in jobs[42:66]} == {'evolution'}
    assert jobs[66]['optimizer_seed'] == 29 or jobs[66]['method'] == 'ppo'
    path = tmp_path / 'lanes.json'
    lanes.save(path)
    assert LaneCoordinator.load(path, priority=order).next_jobs() == jobs
    drive(lanes, tmp_path, 11)
    assert_equivalent(serial, lanes)


def test_voided_attempt_releases_claim_and_reruns_same_job(frozen_protocol, tmp_path):  # noqa: F811
    lanes = LaneCoordinator(PilotCoordinator(frozen_protocol))
    state = tmp_path / 'lanes.json'
    lanes.save(state)
    launches, voided = [], []

    def execute(job, slot, output):
        launches.append(job['job_id'])
        return True

    def accept(job, slot, output, result, error):
        if not voided:
            voided.append(job['job_id'])
            return None
        return row_for(job, tmp_path)

    def release(job_id):
        del lanes.base.claims[job_id]
        lanes.base.revision += 1

    value = budget()
    value.envelope.update(policy_actions_max=10**7, exhaustion_safe_successful_queries_max=10**7)
    output = lambda job: str(tmp_path / 'episodes' / job['job_id'])
    report = run_parallel(lanes, ROUTES, state, output, execute, accept, value,
                          max_new_jobs=12, release=release)
    assert launches.count(voided[0]) == 2 and not report['in_flight_claims']
    assert [entry['job_id'] for entry in lanes.base.ledger].count(voided[0]) == 1
    assert report['accepted_episode_slots'] == 11 and report['status'] != 'blocked'
    with pytest.raises(RuntimeError, match='release policy'):
        run_parallel(LaneCoordinator(PilotCoordinator(frozen_protocol)), ROUTES, tmp_path / 'x.json',
                     output, execute, lambda *args: None, value, max_new_jobs=1)

