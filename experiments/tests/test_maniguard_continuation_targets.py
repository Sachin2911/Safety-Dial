"""Synthetic source-contract checks; no real runtime or physical validation."""

import ast
import copy
from pathlib import Path

import numpy as np
import pytest

from helpers.maniguard_continuation_residual import BoundedActionResidual
from helpers.maniguard_continuation_targets import ARM_PRESET, NativeJointResidualAdapter


def test_supported_preset_matches_primary_checkout_without_importing_simulator():
    root = Path(__file__).resolve().parents[2]
    source = root / "third_party/ManiGuard/maniguard/envs/frozen_task_runtime.py"
    tree = ast.parse(source.read_text())
    entries = [value for node in ast.walk(tree) if isinstance(node, ast.Dict)
               for key, value in zip(node.keys, node.values, strict=True)
               if isinstance(key, ast.Constant) and key.value == "joint_position_raw"]
    assert len(entries) == 1
    preset = ast.literal_eval(entries[0])
    assert preset["arm_0"] == ARM_PRESET
    assert preset["gripper_0"] == snapshot()["gripper_controller"]


def snapshot():
    return {
        "schema": "maniguard-raw-joint-runtime-snapshot-v1",
        "evidence_origin": "synthetic_engineering_fixture",
        "action_normalize": False, "arm_action_indices": list(range(7)),
        "gripper_action_indices": [7], "arm_controller": dict(ARM_PRESET),
        "gripper_controller": {"name": "MultiFingerGripperController", "mode": "binary",
                               "command_input_limits": None},
        "gripper_convention": "+1_open_-1_close", "joint_has_limits": [True] * 7,
        "joint_lower_limits": [-2.] * 7, "joint_upper_limits": [2.] * 7,
        "runtime_evidence_sha256": "a" * 64,
        "source_sha256": {"joint_controller": "b" * 64, "controller_base": "c" * 64,
                          "controller_preset": "d" * 64},
    }


def adapter(weights=None, contract=None):
    contract = snapshot() if contract is None else contract
    residual = BoundedActionResidual(
        np.zeros((8, 49)) if weights is None else weights,
        contract["joint_lower_limits"], contract["joint_upper_limits"])
    return NativeJointResidualAdapter(contract, residual)


def native():
    actions = np.tile(np.r_[np.linspace(-1.8, 1.8, 7), 1.], (8, 1))
    actions[:, 0] = 3.
    actions[::2, 7] = -1.
    return actions


def test_zero_residual_preserves_out_of_range_native_environment_commands_exactly():
    actions = native()
    result = adapter().propose(actions, np.ones(44), np.zeros(5))
    np.testing.assert_array_equal(result["proposed_environment_prefix"], actions)
    np.testing.assert_array_equal(result["native_joint_target_prefix"][:, 0], np.full(8, 2.))
    np.testing.assert_array_equal(
        result["residual_joint_target_prefix"], result["native_joint_target_prefix"])
    assert result["native_clipped_coordinates"][:, 0].all()
    assert result["runtime_evidence_origin"] == "synthetic_engineering_fixture"
    assert result["runtime_evidence_independently_verified"] is False
    assert result["executed_environment_prefix"] is result["achieved_joint_positions"] is None
    assert result["dispatch_authorized"] is False and result["new_executed_controls"] == 0


def test_inward_residual_moves_effective_target_where_raw_command_correction_cannot():
    weights = np.zeros((8, 49))
    weights[0, 0] = -1.
    controller = adapter(weights)
    actions = native()
    result = controller.propose(actions, np.eye(44)[0], np.zeros(5))
    delta = result["effective_arm_target_delta"][:, 0]
    assert np.all(delta < 0) and np.all(delta >= -.05)
    # A naive 0.05-radian correction of raw target 3 still clips to 2.
    np.testing.assert_array_equal(np.clip(actions[:, 0] - .05, -2., 2.), np.full(8, 2.))
    assert np.all(result["residual_joint_target_prefix"][:, 0] < 2.)
    assert np.all(result["environment_input_arm_delta"][:, 0] < -1.)
    np.testing.assert_array_equal(
        controller.canonical_targets(result["proposed_environment_prefix"]),
        result["residual_joint_target_prefix"])
    np.testing.assert_array_equal(result["proposed_environment_prefix"][:, 1:], actions[:, 1:])


def test_outward_saturated_residual_preserves_raw_input_when_effective_target_is_unchanged():
    weights = np.zeros((8, 49))
    weights[0, 0] = 1.
    actions = native()
    result = adapter(weights).propose(actions, np.eye(44)[0], np.zeros(5))
    np.testing.assert_array_equal(result["proposed_environment_prefix"], actions)
    assert not result["effective_arm_target_delta"].any()


@pytest.mark.parametrize("key,value", [
    ("motor_type", "velocity"), ("use_delta_commands", True), ("use_impedances", True),
    ("command_input_limits", "default"), ("command_output_limits", "default"),
])
def test_different_controller_equations_require_a_different_adapter(key, value):
    contract = snapshot()
    contract["arm_controller"][key] = value
    with pytest.raises(ValueError, match="controller preset"):
        adapter(contract=contract)


@pytest.mark.parametrize("key,value", [
    ("action_normalize", True), ("arm_action_indices", [6, 5, 4, 3, 2, 1, 0]),
    ("gripper_action_indices", [0]),
])
def test_normalized_or_reordered_environment_commands_are_rejected(key, value):
    contract = snapshot()
    contract[key] = value
    with pytest.raises(ValueError, match="native action contract"):
        adapter(contract=contract)


def test_unlimited_joints_and_unpinned_evidence_are_rejected():
    contract = snapshot()
    contract["joint_has_limits"][0] = False
    with pytest.raises(ValueError, match="confirmed finite position limits"):
        adapter(contract=contract)
    contract = snapshot()
    contract["runtime_evidence_sha256"] = "unknown"
    with pytest.raises(ValueError, match="source/evidence hashes"):
        adapter(contract=contract)


def test_different_runtime_evidence_invalidates_memory_even_with_equal_joint_bounds():
    weights = np.zeros((8, 49))
    weights[:, 0] = 1.
    first = adapter(weights)
    first.propose(native(), np.eye(44)[0], np.zeros(5))
    state = first.state_dict()
    second = adapter(weights)
    second.load_state_dict(copy.deepcopy(state))
    expected = first.propose(native(), -np.eye(44)[0], np.zeros(5))
    actual = second.propose(native(), -np.eye(44)[0], np.zeros(5))
    np.testing.assert_array_equal(actual["proposed_environment_prefix"],
                                  expected["proposed_environment_prefix"])
    contract = snapshot()
    contract["runtime_evidence_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="different runtime contract"):
        adapter(weights, contract).load_state_dict(state)


def test_bounds_mismatch_is_rejected_and_invalid_commands_do_not_change_memory():
    residual = BoundedActionResidual(np.zeros((8, 49)), np.full(7, -1.), np.ones(7))
    with pytest.raises(ValueError, match="joint bounds differ"):
        NativeJointResidualAdapter(snapshot(), residual)
    controller = adapter()
    state = controller.state_dict()
    actions = native()
    actions[0, 7] = 0.
    with pytest.raises(ValueError, match="raw absolute commands"):
        controller.propose(actions, np.ones(44), np.zeros(5))
    assert controller.state_dict() == state
