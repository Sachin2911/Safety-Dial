"""Regression for the observed downward-prediction/upward-motion guard failure."""

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))
from helpers.maniguard_adaptation import AdapterConfig, ExecutionAdapter, Frame
from helpers.maniguard_runtime import arm_position_limits


def failure_case():
    # Published physical boundary, not a synthetic target constructed from the fix.
    path = ROOT / 'docs/research/research_notes/ManiGuard fruits Jar guard lift prediction mismatch.json'
    case = json.loads(path.read_text())
    proposal = next(row for row in case['boundary_records']
                    if row['type'] == 'proposal' and row['step'] == case['proposal_step'])
    values = {key: np.asarray(value) if isinstance(value, list) else value
              for key, value in proposal['frame'].items()}
    bounds = np.asarray(case['joint_limits_radians_from_longfinger_usda'])
    frame = Frame(**values, joint_lower_limits=bounds[:, 0], joint_upper_limits=bounds[:, 1],
                  joint_has_limits=np.ones(7, dtype=bool))
    return np.asarray(proposal['original'], dtype=np.float32), frame


@pytest.mark.parametrize('kind', ['prerequisite', 'handback'])
def test_recorded_open_jar_lift_is_blocked_with_controller_limits(kind):
    proposal, frame = failure_case()
    saved = proposal.copy()
    legacy = ExecutionAdapter(AdapterConfig(kind=kind, privileged=True))
    corrected = ExecutionAdapter(AdapterConfig(kind=kind, privileged=True, guard_joint_limits=True))
    old_action, old_event = legacy.act(proposal, frame)
    action, event = corrected.act(proposal, frame)
    assert not old_event['guard_blocked'] and old_event['predicted_lift_linearized'] < 0
    np.testing.assert_array_equal(old_action, proposal)
    assert event['guard_blocked'] and event['predicted_lift_linearized'] > 0.014
    assert event['predicted_lift_requested'] < -0.005
    assert event['guard_prediction_mode'] == 'controller_joint_limits'
    np.testing.assert_array_equal(action[:7], frame.joints)
    assert action[-1] == proposal[-1]
    np.testing.assert_array_equal(proposal, saved)


def test_unbounded_joint_is_not_clipped_in_the_prediction():
    proposal, frame = failure_case()
    mask = frame.joint_has_limits.copy()
    mask[1] = False
    frame = replace(frame, joint_has_limits=mask)
    adapter = ExecutionAdapter(AdapterConfig(kind='prerequisite', privileged=True,
                                             guard_joint_limits=True))
    action, event = adapter.act(proposal, frame)
    assert not event['guard_blocked']
    np.testing.assert_array_equal(event['guard_prediction_joint_target'], proposal[:7])
    np.testing.assert_array_equal(action, proposal)


def test_confirmed_prerequisite_preserves_original_action_even_if_target_is_clipped():
    proposal, frame = failure_case()
    adapter = ExecutionAdapter(AdapterConfig(kind='prerequisite', privileged=True,
                                             guard_joint_limits=True))
    action, event = adapter.act(proposal, replace(frame, prerequisite=True))
    assert event['predicted_lift_linearized'] > 0.014 and not event['guard_blocked']
    np.testing.assert_array_equal(action, proposal)


def test_feasible_target_gives_the_same_decision_as_the_original_estimator():
    _, frame = failure_case()
    proposal = np.r_[frame.joints, -1.0]
    proposal[2] += 0.01
    original = ExecutionAdapter(AdapterConfig(kind='prerequisite', privileged=True))
    corrected = ExecutionAdapter(AdapterConfig(kind='prerequisite', privileged=True,
                                               guard_joint_limits=True))
    before, old = original.act(proposal, frame)
    after, new = corrected.act(proposal, frame)
    assert old['guard_blocked'] and new['guard_blocked']
    assert old['predicted_lift_linearized'] == new['predicted_lift_linearized']
    np.testing.assert_array_equal(before, after)


def test_corrected_guard_requires_live_limits_but_legacy_records_remain_valid():
    proposal, frame = failure_case()
    missing = replace(frame, joint_lower_limits=None, joint_upper_limits=None, joint_has_limits=None)
    legacy = ExecutionAdapter(AdapterConfig(kind='prerequisite', privileged=True))
    _, event = legacy.act(proposal, missing)
    assert not event['guard_blocked']
    corrected = ExecutionAdapter(AdapterConfig(kind='prerequisite', privileged=True,
                                               guard_joint_limits=True))
    with pytest.raises(ValueError, match='live arm position limits'):
        corrected.act(proposal, missing)


@pytest.mark.parametrize('change', [
    {'joint_upper_limits': None},
    {'joint_has_limits': np.ones(7)},
    {'joint_lower_limits': np.zeros(6)},
    {'joint_lower_limits': np.full(7, float('nan'))},
    {'joint_lower_limits': np.full(7, 100)},
])
def test_malformed_position_limits_fail_before_action(change):
    proposal, frame = failure_case()
    adapter = ExecutionAdapter(AdapterConfig(kind='prerequisite', privileged=True,
                                             guard_joint_limits=True))
    with pytest.raises(ValueError, match='[Pp]osition limits'):
        adapter.act(proposal, replace(frame, **change))


@pytest.mark.parametrize('config', [
    {'kind': 'identity', 'guard_joint_limits': True},
    {'kind': 'prerequisite', 'privileged': True, 'guard_joint_limits': 'yes'},
    {'kind': 'prerequisite', 'privileged': True, 'guard_joint_limits': 1},
])
def test_prediction_mode_is_explicit_and_guard_only(config):
    with pytest.raises(ValueError):
        AdapterConfig(**config)


def test_runtime_selects_controller_limits_in_arm_index_order_and_copies_them():
    lower = np.arange(9, dtype=float) - 20
    upper = np.arange(9, dtype=float) + 20
    mask = np.array([True, False, True, True, False, True, True, True, True])
    robot = SimpleNamespace(control_limits={'position': (lower, upper), 'has_limit': mask})
    indices = np.array([7, 6, 5, 4, 3, 2, 1])
    actual = arm_position_limits(robot, indices)
    for returned, source in zip(actual, (lower, upper, mask)):
        np.testing.assert_array_equal(returned, source[indices])
        assert not np.shares_memory(returned, source)
    assert actual[2].dtype == bool
