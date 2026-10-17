"""Causal feature-contract checks, with no learned encoder or simulator execution."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.maniguard_adaptation import Frame
from helpers.maniguard_phase2_features import (
    combine_features,
    finite_vector,
    project,
    projection_matrices,
    robot_features,
)


def frame(**kwargs):
    return Frame(np.zeros(7), np.array([.01, .01]), np.zeros(3), **kwargs)


def test_initial_features_do_not_invent_unobserved_history():
    state = frame()
    proposal = np.r_[np.full(7, .05), -1.0]
    actual = robot_features(state, proposal)
    np.testing.assert_array_equal(actual[9:19], np.zeros(10))
    np.testing.assert_array_equal(actual[26:33], np.zeros(7))
    np.testing.assert_allclose(actual[19:26], .25)
    assert actual.shape == (40,) and np.isfinite(actual).all()
    assert actual[33] == -1 and actual[34] == -1


def test_tracking_uses_applied_command_and_measured_joints_not_previous_proposal():
    proposal = np.r_[np.full(7, .1), 1.0]
    old_proposal = np.r_[np.full(7, .9), -1.0]
    applied = np.r_[np.full(7, .04), -1.0]
    actual = robot_features(frame(), proposal, last_executed=applied, last_proposal=old_proposal)
    np.testing.assert_allclose(actual[12:19], .2)
    np.testing.assert_array_equal(actual[26:33], np.full(7, -1.0))
    assert actual[33] == 1 and actual[34] == -1
    np.testing.assert_array_equal(applied, np.r_[np.full(7, .04), -1.0])


def test_requested_motion_uses_controller_feasible_targets_without_changing_proposal():
    bounds = {"joint_lower_limits": np.full(7, -.1), "joint_upper_limits": np.full(7, .1),
              "joint_has_limits": np.array([True, False, True, False, True, False, True])}
    proposal = np.r_[np.full(7, .3), -1.0]
    actual = robot_features(frame(**bounds), proposal)
    np.testing.assert_allclose(actual[19:26], [.5, 1, .5, 1, .5, 1, .5])
    np.testing.assert_array_equal(proposal, np.r_[np.full(7, .3), -1.0])


def test_end_effector_delta_uses_previous_decision_observation():
    actual = robot_features(frame(), np.zeros(8), last_eef=np.array([-.01, .02, 0.0]))
    np.testing.assert_allclose(actual[9:12], [.2, -.4, 0])


@pytest.mark.parametrize("field", ["prerequisite", "target_engaged"])
def test_privileged_false_labels_are_rejected_as_well_as_true(field):
    for value in (False, True):
        with pytest.raises(ValueError, match="Privileged"):
            robot_features(frame(**{field: value}), np.zeros(8))


def test_fixed_projection_is_reproducible_normalized_and_does_not_fit_observations():
    visual, text = projection_matrices()
    repeated_visual, repeated_text = projection_matrices()
    np.testing.assert_array_equal(visual, repeated_visual)
    np.testing.assert_array_equal(text, repeated_text)
    np.testing.assert_allclose(visual.T @ visual, np.eye(8), atol=1e-6)
    assert not np.array_equal(visual, text)
    embedding = np.random.default_rng(2).normal(size=512).astype(np.float32)
    expected = project(embedding, visual)
    np.testing.assert_allclose(project(embedding * 3, visual), expected, atol=1e-7)
    assert expected.shape == (8,) and np.max(np.abs(expected)) <= 1
    with pytest.raises(ValueError, match="Zero"):
        project(np.zeros(512), visual)


def test_visual_and_task_context_remain_separate_and_combination_copies_inputs():
    inputs = [np.zeros(40), np.ones(8) * .1, np.ones(8) * .2, np.ones(8) * .3]
    actual = combine_features(*inputs)
    assert actual.shape == (64,)
    np.testing.assert_allclose(actual[40:48], .1)
    np.testing.assert_allclose(actual[48:56], .2)
    np.testing.assert_allclose(actual[56:64], .3)
    inputs[0][:] = 1
    np.testing.assert_array_equal(actual[:40], np.zeros(40))


@pytest.mark.parametrize("value", [np.ones(8, dtype=bool), np.array(["1"] * 8),
                                  np.full(8, 1 + 2j), np.full(8, np.nan)])
def test_feature_validation_rejects_non_real_and_nonfinite_arrays(value):
    with pytest.raises(ValueError):
        finite_vector(value, (8,), "fixture")


@pytest.mark.parametrize("kwargs", [{"since_query": -1}, {"opening_steps": -1},
                                    {"choices": [0, 3, 0, 0]}, {"choices": [0, 0, 0]}])
def test_feature_history_rejects_invalid_counter_and_choice_domains(kwargs):
    with pytest.raises(ValueError):
        robot_features(frame(), np.zeros(8), **kwargs)
