"""Check optimizer comparison scope, split isolation and measured terminal utility."""

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_maniguard_results import write_fixture

from helpers.maniguard_search import (
    audit_generation,
    restore_search,
    search_state,
    terminal_utility,
    validate_protocol,
)
from helpers.unisafe_search import AdapterSearch

SPLITS = {'families': {'jar': {'train': ['task_0019'], 'test': ['task_0023']}}}
PROTOCOL = {'family': 'jar', 'base_adapter': {'kind': 'release', 'horizon': 8},
            'domain': {'release_delay': [0, 2, 4]},
            'methods': ['evolution', 'random', 'policy_gradient'], 'population': 4,
            'generations': 2, 'search_seeds': [17],
            'training_episodes': [{'scene': 'task_0019/base', 'seed': 0}],
            'objective': 'task_completion_minus_two_violations'}


def test_training_protocol_accepted_and_reserved_scene_rejected():
    validate_protocol(PROTOCOL, SPLITS)
    changed = deepcopy(PROTOCOL)
    changed['training_episodes'][0]['scene'] = 'task_0023/base'
    with pytest.raises(ValueError, match='training split'):
        validate_protocol(changed, SPLITS)


def test_handback_search_keeps_guard_trigger_fixed():
    changed = deepcopy(PROTOCOL)
    changed['base_adapter'] = {'kind': 'handback', 'privileged': True}
    changed['domain'] = {'handback_hold_steps': [0, 2, 4]}
    validate_protocol(changed, SPLITS)
    changed['domain']['lift_margin'] = [0.001, 0.1]
    with pytest.raises(ValueError, match='guard trigger'):
        validate_protocol(changed, SPLITS)


@pytest.mark.parametrize('domain', [
    {'privileged': [False, True]}, {'horizon': [0, 8]}, {'release_delay': []},
])
def test_invalid_or_information_changing_domains_rejected(domain):
    changed = deepcopy(PROTOCOL)
    changed['domain'] = domain
    with pytest.raises(ValueError):
        validate_protocol(changed, SPLITS)


def test_missing_ordinary_learning_comparison_rejected():
    changed = deepcopy(PROTOCOL)
    changed['methods'] = ['evolution', 'random']
    with pytest.raises(ValueError, match='policy gradient'):
        validate_protocol(changed, SPLITS)


@pytest.mark.parametrize(('success', 'violation', 'expected'), [
    (True, False, 1), (False, False, 0), (True, True, -1), (False, True, -2),
])
def test_terminal_utility_preserves_unsafe_outcomes(success, violation, expected):
    assert terminal_utility({'safe_completion': success and not violation,
                             'unsafe_completion': success and violation,
                             'violation': violation}) == expected


@pytest.mark.parametrize('method', ['evolution', 'random', 'policy_gradient'])
def test_saved_optimizer_resumes_identical_draws_and_updates(method):
    optimizer = AdapterSearch(method, {'release_delay': [0, 2, 4], 'min_closing_steps': [1, 3]}, 4, 71)
    genes = optimizer.ask()
    optimizer.tell(genes, [0, 1, -1, 0])
    restored = restore_search(search_state(optimizer))
    next_genes = optimizer.ask()
    assert next_genes == restored.ask()
    optimizer.tell(next_genes, [1, 0, -2, 0])
    restored.tell(next_genes, [1, 0, -2, 0])
    assert search_state(optimizer) == search_state(restored)


@pytest.mark.parametrize('workers', [0, 3, True, 1.5])
def test_unsupported_search_concurrency_rejected(workers):
    changed = deepcopy(PROTOCOL)
    changed['workers'] = workers
    with pytest.raises(ValueError, match='workers'):
        validate_protocol(changed, SPLITS)


@pytest.mark.parametrize('extra', [{'max_steps': 32}, {'family': 'lid'}, {'adapter': {}}])
def test_training_episode_cannot_override_task_or_horizon(extra):
    changed = deepcopy(PROTOCOL)
    changed['training_episodes'][0].update(extra)
    with pytest.raises(ValueError, match='only scene and seed'):
        validate_protocol(changed, SPLITS)


def write_generation(path):
    # Interleaved labels ensure candidate fitness is not inferred from row position.
    conditions = [(1, 1, True, True), (0, 0, True, False),
                  (1, 0, False, False), (0, 1, True, False)]
    jobs, processes = [], []
    for index, (candidate, seed, success, violation) in enumerate(conditions):
        adapter = {'kind': 'release', 'release_delay': [0, 4][candidate]}
        jobs.append({'family': 'jar', 'scene': 'task_0000/base', 'seed': seed,
                     'label': f'candidate-{candidate}', 'adapter': adapter})
        episode = path / f'episode-{index:04d}'
        files = write_fixture(episode, success=success, violation=violation)
        job = files['job.json'] | {'seed': seed, 'adapter': adapter}
        (episode / 'job.json').write_text(json.dumps(job))
        ledger = files['ledger.json'] | {
            'physics_steps': 12, 'physics_seconds': 0.1, 'env_step_attempts': 2,
            'env_steps_returned': 2, 'policy_queries': 1, 'policy_query_seconds': 0.01,
            'elapsed_seconds': 1.0,
        }
        (episode / 'ledger.json').write_text(json.dumps(ledger))
        processes.append({'index': index, 'returncode': 0, 'elapsed_seconds': 1.1})
    plan = {'split': 'train', 'genes': [[0], [1]], 'jobs': jobs}
    (path / 'plan.json').write_text(json.dumps(plan))
    (path / 'processes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in processes))
    (path / 'complete.json').write_text(json.dumps({'episodes': len(jobs)}))
    return plan


def test_generation_uses_physical_outcomes_instead_of_cached_success_claims(tmp_path):
    plan = write_generation(tmp_path)
    (tmp_path / 'evidence.jsonl').write_text(json.dumps({'safe_completion': True}) + '\n')
    audit = audit_generation(tmp_path, plan)
    assert audit['fitness'] == [1.0, -0.5]
    assert audit['candidate_outcomes'][0]['safe_completion'] == 2
    assert audit['candidate_outcomes'][1]['unsafe_completion'] == 1
    assert audit['candidate_outcomes'][1]['safe_unfinished'] == 1
    assert audit['batch_report']['recorded_cost_totals']['physics_steps'] == 48
    assert audit['batch_report']['recorded_cost_totals']['policy_queries'] == 4


def test_generation_cannot_update_from_a_different_saved_plan(tmp_path):
    plan = write_generation(tmp_path)
    changed = deepcopy(plan)
    changed['jobs'][0]['adapter']['release_delay'] = 8
    with pytest.raises(ValueError, match='differs from the proposed'):
        audit_generation(tmp_path, changed)


def test_generation_requires_the_batch_to_finish(tmp_path):
    plan = write_generation(tmp_path)
    (tmp_path / 'complete.json').unlink()
    with pytest.raises(ValueError, match='completion marker'):
        audit_generation(tmp_path, plan)


def test_generation_rejects_corrupt_monitor_despite_complete_marker(tmp_path):
    plan = write_generation(tmp_path)
    monitor = tmp_path / 'episode-0000/episode/task_0000/base_ltl.json'
    value = json.loads(monitor.read_text())
    value['violated'] = False  # Raw episode result records an actual violation.
    monitor.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='completion marker'):
        audit_generation(tmp_path, plan)


def test_generation_rejects_missing_interaction_costs(tmp_path):
    plan = write_generation(tmp_path)
    ledger = tmp_path / 'episode-0000/ledger.json'
    value = json.loads(ledger.read_text())
    del value['policy_queries']
    ledger.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='cost accounting'):
        audit_generation(tmp_path, plan)


@pytest.mark.parametrize('seed', [0, 2])
def test_candidates_require_the_same_distinct_scene_seed_pairs(tmp_path, seed):
    plan = write_generation(tmp_path)
    plan['jobs'][3]['seed'] = seed
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    job_path = tmp_path / 'episode-0003/job.json'
    job = json.loads(job_path.read_text())
    job['seed'] = seed
    job_path.write_text(json.dumps(job))
    with pytest.raises(ValueError, match='duplicate|same scene/seed'):
        audit_generation(tmp_path, plan)
