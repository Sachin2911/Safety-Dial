"""Synthetic audited-row fixtures for scheduling; no simulator or subprocess execution."""

import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import helpers.maniguard_phase2_coordinator as coordinator_module
from helpers.maniguard_phase2_coordinator import PilotCoordinator
from helpers.maniguard_phase2_model import decode, forward, log_prob
from helpers.maniguard_phase2_optim import record_hash
from helpers.maniguard_phase2_protocol import build_pilot


@pytest.fixture
def frozen_protocol(monkeypatch):
    # Freeze-evidence semantics are tested by protocol tests. Isolate the coordinator
    # state machine from unrelated artifact preparation using an explicit validator stub.
    protocol = build_pilot()
    protocol['status'] = 'frozen'
    protocol['sha256'] = record_hash({k: v for k, v in protocol.items() if k != 'sha256'})

    def validated(value, *args, **kwargs):
        assert value['status'] == 'frozen'
        assert kwargs['require_frozen'] is True
        return {'ready_to_run': True}

    monkeypatch.setattr(coordinator_module, 'validate_protocol', validated)
    return protocol


def measured_row(job, output, *, utility=1, queries=1):
    """Provide the result-reader contract and a genuine NumPy recurrent trajectory."""
    rng = np.random.default_rng(job['sampling_seed'])
    features = rng.normal(size=(2, 64))
    hidden = None
    actions, logs = [], []
    for feature in features:
        logits, hidden = forward(job['genome'], feature, hidden)
        choices = decode(logits, rng if job['stochastic'] else None)
        actions.append(choices.tolist())
        logs.append(log_prob(logits, choices))
    violation = utility == -2
    complete = utility == 1
    row = {key: deepcopy(job[key]) for key in ('family', 'scene', 'seed', 'sampling_seed',
           'actor_sha256', 'mode', 'fixed_choices', 'stochastic', 'purpose')}
    output = str(Path(output).resolve())
    row.update(output=output, episode_id=output, action_horizon=job['max_steps'],
               full_episode=True, short_check=False, valid=True, invalid_reasons=[], steps=8,
               outcome={'valid': True, 'C_req': complete, 'S_req': complete,
                        'V': violation, 'U': utility},
               costs={'policy_queries': queries, 'physics_steps': 48,
                      'policy_control_steps': 8, 'elapsed_seconds': 1.0},
               missing_cost_fields=[], cost_issues=[], observed_work={'policy_control_steps': 8},
               ppo_eligible=job['stochastic'] and job['purpose'] == 'train',
               trajectory={'features': features.tolist(), 'actions': actions, 'old_log_probs': logs,
                           'actor_sha256': job['actor_sha256'], 'family': job['family'],
                           'terminal_utility': utility, 'episode_id': output})
    return row


def record_job(coordinator, job, base, **kwargs):
    output = base / job['job_id']
    coordinator.mark_started(job['job_id'], output)
    row = measured_row(job, output, **kwargs)
    report = coordinator.record_result(job['job_id'], row)
    assert report['status'] != 'blocked', report['blocked_reasons']
    return row


def complete_block(coordinator, base):
    jobs = coordinator.next_jobs()
    for job in jobs:
        record_job(coordinator, job, base)
    return jobs


def test_unfrozen_protocol_cannot_create_training_coordinator():
    with pytest.raises(ValueError, match='freeze incomplete'):
        PilotCoordinator(build_pilot())


def test_claimed_job_is_not_redispatched_and_partial_state_resumes_exactly(
        frozen_protocol, tmp_path):
    first = PilotCoordinator(frozen_protocol)
    jobs = first.next_jobs()
    claimed = jobs[0]
    output = tmp_path / claimed['job_id']
    first.mark_started(claimed['job_id'], output)
    path = tmp_path / 'state.json'
    first.save(path)
    second = PilotCoordinator.load(path)
    assert first.state_dict() == second.state_dict()
    assert claimed['job_id'] not in {job['job_id'] for job in second.next_jobs()}
    assert first.next_jobs() == second.next_jobs()
    row = measured_row(claimed, output)
    assert first.record_result(claimed['job_id'], row) == second.record_result(claimed['job_id'], row)
    assert first.state_dict() == second.state_dict()
    # Finish the same stage on both copies, including one exact learner/search transition.
    for job in first.next_jobs():
        record_job(first, job, tmp_path)
        record_job(second, job, tmp_path)
    assert first.state_dict() == second.state_dict()
    assert first.next_jobs() == second.next_jobs()
    second.save(path)
    with pytest.raises(ValueError, match='reload first'):
        first.save(path)


def test_invalid_attempt_keeps_costs_blocks_scoring_and_rejects_duplicate(
        frozen_protocol, tmp_path):
    coordinator = PilotCoordinator(frozen_protocol)
    jobs = coordinator.next_jobs()
    job = jobs[0]
    output = tmp_path / job['job_id']
    coordinator.mark_started(job['job_id'], output)
    row = measured_row(job, output)
    row.update(valid=False, outcome=None, trajectory=None, ppo_eligible=False,
               invalid_reasons=['Native simulator failed after recorded actions'],
               missing_cost_fields=['encoder_seconds'], cost_issues=['missing encoder timing'])
    report = coordinator.record_result(job['job_id'], row)
    assert report['status'] == 'blocked'
    assert report['physical_attempts_recorded'] == 1
    assert report['accepted_episode_slots'] == 0
    assert report['cost_totals']['physics_steps'] == 48
    assert report['incomplete_cost_records'] == [str(output.resolve())]
    assert report['completed_stages'] == 0 and coordinator.next_jobs() == []
    assert not report['in_flight_claims']
    with pytest.raises(ValueError, match='duplicate'):
        coordinator.record_result(job['job_id'], row)
    assert coordinator.report()['cost_totals']['physics_steps'] == 48


def test_wrong_actor_or_unclaimed_attempt_is_never_scored_but_costs_survive(
        frozen_protocol, tmp_path):
    coordinator = PilotCoordinator(frozen_protocol)
    job = coordinator.next_jobs()[0]
    output = tmp_path / job['job_id']
    coordinator.mark_started(job['job_id'], output)
    row = measured_row(job, output)
    row['actor_sha256'] = '0' * 64
    report = coordinator.record_result(job['job_id'], row)
    assert report['status'] == 'blocked' and report['accepted_episode_slots'] == 0
    assert 'Result differs from dispatched actor_sha256' in report['blocked_reasons']
    unexpected = measured_row(job, tmp_path / 'unassigned-attempt')
    report = coordinator.record_result('not-a-dispatched-job', unexpected)
    assert report['physical_attempts_recorded'] == 2
    assert report['cost_totals']['policy_control_steps'] == 16


def test_inflight_valid_work_drains_after_failure_without_optimizer_update(
        frozen_protocol, tmp_path):
    coordinator = PilotCoordinator(frozen_protocol)
    jobs = coordinator.next_jobs()
    for job in jobs[:2]:
        coordinator.mark_started(job['job_id'], tmp_path / job['job_id'])
    failed = measured_row(jobs[0], tmp_path / jobs[0]['job_id'])
    failed['valid'] = False
    coordinator.record_result(jobs[0]['job_id'], failed)
    valid = measured_row(jobs[1], tmp_path / jobs[1]['job_id'])
    report = coordinator.record_result(jobs[1]['job_id'], valid)
    assert report['status'] == 'blocked'
    assert report['physical_attempts_recorded'] == 2
    assert report['accepted_episode_slots'] == 1
    assert report['completed_stages'] == 0
    assert all(search.round_index == 0 for search in coordinator.searches.values())
    assert all(learner.episodes_seen == 0 for learner in coordinator.learners.values())


def test_complete_synthetic_pilot_updates_paired_panels_and_selects_with_fixed_ties(
        frozen_protocol, tmp_path):
    coordinator = PilotCoordinator(frozen_protocol)
    training_jobs = []
    while coordinator.status == 'training':
        jobs = coordinator.next_jobs()
        assert jobs and len({job['actor_sha256'] for job in jobs}) >= 1
        if jobs[0]['method'] == 'ppo':
            assert len(jobs) == 3
            assert len({job['actor_sha256'] for job in jobs}) == 1
            assert [job['family'] for job in jobs] == ['jar', 'stack', 'clutter']
        for job in jobs:
            training_jobs.append(job)
            utility = 1 if job.get('candidate_slot', 0) == 0 else 0
            record_job(coordinator, job, tmp_path, utility=utility)
    assert len(training_jobs) == 432
    assert coordinator.status == 'validation' and len(coordinator.nominees) == 6
    for search in coordinator.searches.values():
        assert search.round_index == 3
        assert [row['panel'] for row in search.archive] == ['panel-0', 'panel-1', 'panel-2']
        assert len(search.archive[-1]['candidates']) == 8
    for learner in coordinator.learners.values():
        assert learner.episodes_seen == 72 and learner.updates == 24
        assert set(learner.checkpoints) == {'48', '72'}
    saved_nominees = deepcopy(coordinator.nominees)
    validation_order = [case['family'] + '/' + case['scene']
                        for case in frozen_protocol['validation_cases']]
    while coordinator.status == 'validation':
        jobs = coordinator.next_jobs()
        for job in jobs:
            utility, queries = 1, 2
            if job['stage'] == 'finalist_validation':
                if job['method'] == 'evolution':
                    # Finalist zero: two successes, one violation, three safe unfinished.
                    # Finalist one: all safe unfinished. Both mean U=0; lower V wins.
                    case = validation_order.index(job['family'] + '/' + job['scene'])
                    utility = ((1, 1, -2, 0, 0, 0)[case] if job['finalist'] == 0 else 0)
                    queries = 1 if job['finalist'] == 0 else 9
                elif job['method'] == 'random':
                    queries = 3 if job['finalist'] == 0 else 1
                # PPO ties every metric and must select earlier nominated index zero.
            record_job(coordinator, job, tmp_path, utility=utility, queries=queries)
    assert coordinator.status == 'complete' and coordinator.next_jobs() == []
    assert coordinator.nominees == saved_nominees  # Validation never mutates a candidate.
    assert len(coordinator.ledger) == 522
    assert len(coordinator.selected) == 6
    for key, selected in coordinator.selected.items():
        assert selected['finalist'] == (0 if key.startswith('ppo/') else 1)
    assert set(coordinator.reference_metrics) == {'identity', 'fixed', 'contextual'}
    report = coordinator.report()
    assert report['cost_totals']['policy_control_steps'] == 522 * 8
    assert report['accepted_episode_slots'] == 522
    restored = PilotCoordinator.from_state_dict(json.loads(json.dumps(coordinator.state_dict())))
    assert restored.state_dict() == coordinator.state_dict()
    assert restored.report() == report
