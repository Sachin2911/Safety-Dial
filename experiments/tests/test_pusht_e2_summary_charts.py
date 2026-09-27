"""Focused saved-summary arithmetic; no model, environment or real experiment input."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    'e2_charts', Path(__file__).resolve().parents[1] / 'scripts/pusht_e2_summary_charts.py')
charts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(charts)


def metric(accepted=10, bad=2, unknown=1):
    return {'n': 20, 'n_accepted': accepted, 'n_false_safe': bad, 'n_accepted_censored': unknown,
            'fsa': None if not accepted or unknown else bad/accepted,
            'acceptance_rate': accepted/20, 'fsa_lower': bad/accepted if accepted else None,
            'fsa_upper': (bad+unknown)/accepted if accepted else None}


def test_unknown_acceptance_retains_interval_without_point_or_ci():
    saved = metric()
    saved['fsa_imagined_ci'] = {'lo': .1, 'hi': .3, 'point': None}
    result = charts.metric_numbers(saved)
    assert result['fsa_point'] is None
    assert result['fsa_lower'] == .2 and result['fsa_upper'] == .3
    assert 'fsa_imagined_ci' not in result
    saved['fsa'] = .25
    with pytest.raises(ValueError, match='point'):
        charts.metric_numbers(saved)


def test_zero_acceptance_remains_undefined_and_invalid_counts_fail():
    d = charts.metric_numbers(metric(0, 0, 0))
    assert d['point_status'] == 'undefined_zero_acceptance' and d['fsa_lower'] is None
    with pytest.raises(ValueError, match='counts'):
        charts.metric_numbers(metric(1, 1, 1))


def test_saved_bound_uses_exact_integer_ratios_not_rounded_percentages():
    base = {'known_false_safe': 756, 'accepted_unresolved': 49, 'accepted': 3139}
    adapted = {'known_false_safe': 620, 'accepted_unresolved': 50, 'accepted': 2950}
    result = charts.relative_outer_bound(base, adapted)
    assert result['upper_exact_fraction'] == '42857/237475'
    assert result['lower_exact_fraction'] == '12707/223020'
    assert result['upper'] < .25 and result['lower'] > 0
    with pytest.raises(ValueError, match='positive baseline'):
        charts.relative_outer_bound(dict(base, known_false_safe=0), adapted)


def goal(*, steps=250, coverage=.1, early_verified=False):
    early = steps < 250
    return {'root_id': 'r1', 'episode': 1, 'case_sha256': 'abc', 'requested_steps': 250,
            'executed_steps': steps, 'valid': True, 'arena_exit': False, 'truncated': False,
            'terminated': early, 'censored_future': early, 'outcome_complete': not early or early_verified,
            'completed_on_verified_goal': early_verified, 'final_coverage': coverage}


def test_full_horizon_is_not_reported_as_goal_success():
    result = charts.goal_completion([goal(coverage=0)])
    assert result['counts']['full_planned_horizon'] == 1
    assert result['counts']['verified_early_goal'] == 0
    assert 'not goal-success' in result['interpretation']


def test_unverified_early_terminal_remains_incomplete():
    result = charts.goal_completion([goal(steps=100, coverage=.8)])
    assert result['counts']['early_unverified_terminal'] == 1
    assert result['counts']['full_planned_horizon'] == 0
    with pytest.raises(ValueError, match='Unverified early goal'):
        charts.goal_completion([goal(steps=100, coverage=.8, early_verified=True)])


def test_duplicate_goal_cases_fail():
    with pytest.raises(ValueError, match='Duplicate'):
        charts.goal_completion([goal(), goal()])


def test_malformed_acceptance_rate_or_bounds_fail():
    m = metric()
    m['acceptance_rate'] = .9
    with pytest.raises(ValueError, match='rate mismatch'):
        charts.metric_numbers(m)
    m = metric()
    m['fsa_upper'] = .9
    with pytest.raises(ValueError, match='endpoints'):
        charts.metric_numbers(m)
