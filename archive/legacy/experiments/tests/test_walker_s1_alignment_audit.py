"""Saved-row timing checks; no environment, model or real data use."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    's1_alignment', Path(__file__).resolve().parents[1] / 'scripts/walker_s1_alignment_audit.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def columns():
    q = np.zeros((3, 9))
    q[:, 0] = [0, .024, .048]
    q[:, 1] = [1.2, .9, .7]
    return {'qpos': q, 'qvel': np.zeros((3, 9)), 'action': np.zeros((3, 6)),
            'x_velocity': np.array([3., 3., -4.]), 'cost': np.array([1, 1, 0]),
            'healthy': np.array([1, 1, 0]), 'terminated': np.array([0, 1, 0]),
            'truncated': np.zeros(3), 'episode_idx': np.zeros(3, dtype=int),
            'step_idx': np.arange(3)}


def rows(c, termination_on=True):
    return audit.window_rows(c, episode=0, offset=100, start=0, stop=3, length=3,
                             termination_on=termination_on)


def test_pre_action_health_differs_from_transition_termination():
    out = rows(columns())
    assert out[1]['healthy'] == 1 and out[1]['terminated'] == 1
    assert all(r['healthy_matches'] for r in out)
    assert out[1]['termination_matches'] is True
    assert out[0]['frame_velocity_from_preceding_transition'] is None
    assert out[1]['frame_velocity_from_preceding_transition'] == 3.


def test_final_successor_is_unavailable_not_safe_or_zero_error():
    out = rows(columns())
    assert out[-1]['expected_x_velocity'] is None
    assert out[-1]['velocity_matches'] is None
    assert out[-1]['termination_matches'] is None
    assert audit.summarize(out)['termination']['unavailable'] == 1


def test_probe_disabled_termination_does_not_become_health_label():
    c = columns()
    c['terminated'][:] = 0
    out = rows(c, termination_on=False)
    assert out[-1]['healthy'] == 0 and out[-1]['termination_matches'] is True
    assert audit.summarize(out)['termination']['mismatches'] == 0


def test_health_and_cost_use_strict_signed_boundaries():
    q = np.zeros((4, 9))
    q[:, 1] = [.8, 2., 1.2, 1.2]
    q[:, 2] = [0, 0, -1, 1]
    assert not audit.healthy(q).any()
    c = columns()
    c['x_velocity'][-1] = audit.SPEED_LIMIT
    assert rows(c)[-1]['expected_cost'] == 0
    c['x_velocity'][-1] = -10
    assert rows(c)[-1]['cost_matches'] is True


def test_shifted_health_and_wrong_velocity_are_visible_mismatches():
    c = columns()
    c['healthy'][1] = 0
    c['x_velocity'][0] = 3.5
    counts = audit.summarize(rows(c))
    assert counts['healthy']['mismatches'] == 1
    assert counts['velocity']['mismatches'] == 1


def test_extra_successor_supports_window_end_without_crossing_episode():
    c = columns()
    out = audit.window_rows(c, episode=0, offset=100, start=0, stop=2, length=3,
                            termination_on=True)
    assert len(out) == 2 and out[-1]['velocity_matches'] is True
    assert out[-1]['expected_terminated'] == 1
    c['episode_idx'][-1] = 1
    with pytest.raises(ValueError, match='crosses'):
        audit.window_rows(c, episode=0, offset=100, start=0, stop=2, length=3,
                          termination_on=True)


def test_nonfinite_and_nonbinary_values_fail_closed():
    c = columns()
    c['qpos'][0, 1] = np.nan
    with pytest.raises(ValueError, match='Nonfinite'):
        rows(c)
    c = columns()
    c['healthy'][0] = 2
    with pytest.raises(ValueError, match='Nonbinary'):
        rows(c)
