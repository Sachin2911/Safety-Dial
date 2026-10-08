"""Checks for query-free sensitivity arithmetic; no experiment outcomes are simulated."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evo_readiness_sizing_review import normal_power


def test_zero_effect_gives_nominal_alpha():
    result = normal_power(effect=0, seed_variance=.01, episode_variance=.35,
                          interaction_variance=.35, seeds=10, episodes=320)
    assert result['approximate_two_sided_power'] == pytest.approx(.025)
    assert result['variance_of_mean'] == pytest.approx(.01 / 10 + .35 / 320 + .35 / 3200)


def test_more_seeds_improve_power_but_more_episodes_cannot_remove_seed_variance():
    params = dict(effect=.1, seed_variance=.04, episode_variance=.1, interaction_variance=.35)
    base = normal_power(**params, seeds=10, episodes=320)
    more = normal_power(**params, seeds=20, episodes=320)
    roots = normal_power(**params, seeds=10, episodes=100000000)
    assert more['approximate_two_sided_power'] > base['approximate_two_sided_power']
    assert roots['variance_of_mean'] >= .004


@pytest.mark.parametrize('value', [-1, float('nan'), float('inf')])
def test_invalid_variance_is_rejected(value):
    with pytest.raises(ValueError):
        normal_power(effect=.1, seed_variance=value, episode_variance=.1,
                     interaction_variance=.1, seeds=10, episodes=320)
