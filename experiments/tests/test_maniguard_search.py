"""Check optimizer comparison scope, split isolation and measured terminal utility."""

import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.maniguard_search import (
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
