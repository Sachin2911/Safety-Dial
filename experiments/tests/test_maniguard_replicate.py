"""Replicate-diagnostic scheduling: durable claims, no replay, blocking, analysis shape."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.maniguard_replicate import (
    DiagnosticBudget,
    ReplicateCoordinator,
    analyze,
    compile_jobs,
    load_replicate_protocol,
    wilson,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def protocol():
    return load_replicate_protocol(ROOT / 'configs/maniguard/phase2/replicate-v1.json')


def row_for(job, output, *, safe=True, valid=True):
    return {'output': str(Path(output).resolve()), 'episode_id': str(Path(output).resolve()),
            'family': job['family'], 'scene': job['scene'], 'seed': job['seed'],
            'actor_sha256': job['actor_sha256'], 'valid': valid, 'invalid_reasons': [],
            'outcome': {'valid': True, 'C_req': safe, 'S_req': safe, 'V': False, 'U': int(safe)},
            'costs': {'policy_control_steps': 100, 'policy_queries': 10, 'elapsed_seconds': 5.0},
            'cost_issues': [], 'missing_cost_fields': [], 'steps': 100}


def test_jobs_cover_every_cell_once_in_replicate_rounds(protocol):
    jobs = compile_jobs(protocol)
    cells = protocol['replicates_per_cell'] * len(protocol['controllers']) * len(protocol['cases'])
    assert len(jobs) == cells == 192 and len({job['job_id'] for job in jobs}) == cells
    first_round = jobs[:24]
    assert len({(job['slot'], job['scene']) for job in first_round}) == 24
    assert all(job['replicate'] == 0 for job in first_round)


def test_claims_persist_and_results_bind(protocol, tmp_path):
    coordinator = ReplicateCoordinator(protocol)
    state = tmp_path / 'coordinator.json'
    coordinator.save(state)
    job = coordinator.next_jobs()[0]
    coordinator.mark_started(job['job_id'], tmp_path / 'a')
    coordinator.save(state)
    resumed = ReplicateCoordinator.load(protocol, state)
    assert job['job_id'] in resumed.report()['in_flight_claims']
    assert job['job_id'] not in {item['job_id'] for item in resumed.next_jobs()}
    with pytest.raises(ValueError, match='already claimed'):
        resumed.mark_started(job['job_id'], tmp_path / 'b')
    report = resumed.record_result(job['job_id'], row_for(job, tmp_path / 'a'))
    assert report['accepted_episode_slots'] == 1 and report['status'] == 'running'
    with pytest.raises(ValueError, match='already recorded'):
        resumed.record_result(job['job_id'], row_for(job, tmp_path / 'a'))
    resumed.save(state)
    with pytest.raises(ValueError, match='changed'):
        coordinator.save(state)


def test_invalid_or_misrouted_attempt_blocks_and_keeps_costs(protocol, tmp_path):
    coordinator = ReplicateCoordinator(protocol)
    first, second = coordinator.next_jobs()[:2]
    coordinator.mark_started(first['job_id'], tmp_path / 'a')
    coordinator.mark_started(second['job_id'], tmp_path / 'b')
    report = coordinator.record_result(first['job_id'], row_for(first, tmp_path / 'a', valid=False))
    assert report['status'] == 'blocked' and coordinator.next_jobs() == []
    wrong = row_for(second, tmp_path / 'b')
    wrong['actor_sha256'] = 'other'
    report = coordinator.record_result(second['job_id'], wrong)
    assert report['physical_attempts_recorded'] == 2 and report['accepted_episode_slots'] == 0
    assert report['cost_totals']['policy_control_steps'] == 200


def test_budget_enforces_gpu_hours_and_absolute_stop(protocol):
    envelope = dict(protocol['resource_envelope'], absolute_stop_utc='2099-01-01T00:00:00Z')
    budget = DiagnosticBudget(envelope, 1000.0, clock=lambda: 1000.0 + 3600)
    assert budget.remaining_wall_seconds() == pytest.approx(3 * 3600)
    with pytest.raises(RuntimeError, match='policy-action'):
        budget.check(next_actions=envelope['policy_actions_max'] + 1)
    late = DiagnosticBudget(envelope, 1000.0, clock=lambda: 1000.0 + 4 * 3600)
    with pytest.raises(RuntimeError, match='GPU-hour'):
        late.check()


def test_analysis_reports_partial_and_complete_results(protocol, tmp_path):
    coordinator = ReplicateCoordinator(protocol)
    for index, job in enumerate(coordinator.next_jobs()[:48]):
        coordinator.mark_started(job['job_id'], tmp_path / str(index))
        coordinator.record_result(job['job_id'], row_for(job, tmp_path / str(index), safe=job['slot'] % 2 == 0))
    result = analyze(coordinator, pilot_vectors={'random': list(range(8))})
    assert result['complete'] is False and result['episodes_analyzed'] == 48
    assert set(result['controllers']) == {str(slot) for slot in range(8)}
    assert result['controllers']['0']['safe_completion'] == 1.0
    assert result['controller_homogeneity']['p_value'] < 0.01
    assert result['required_episodes_per_arm_for_80pct_power']['0.2'] > 10
    assert wilson(5, 10)[0] < 0.5 < wilson(5, 10)[1]
    json.dumps(result)
