"""Behavioural tests of the new control boundary and causal adaptive implementation."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.execution_adapter import AdapterConfig, ExecutionAdapter  # noqa: E402


def test_native_is_bit_identical_and_does_not_change_rng():
    rng = np.random.default_rng(19)
    random_state = np.random.get_state()
    controller = ExecutionAdapter(AdapterConfig())
    for step in range(32):
        command = rng.standard_normal(8).astype(np.float32)
        untouched = command.copy()
        actual, trace = controller.apply(
            command, rng.standard_normal(7), rng.standard_normal(7), step
        )
        assert actual.dtype == command.dtype
        assert actual.tobytes() == command.tobytes() == untouched.tobytes()
        assert not trace["correction"].any()
    for before, after in zip(random_state, np.random.get_state(), strict=True):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
def test_bounds_and_gripper_preservation(mode):
    controller = ExecutionAdapter(AdapterConfig(mode=mode))
    for step in range(50):
        command = np.r_[np.full(7, (-1.0) ** step * 3), np.float64(-0.17)]
        actual, trace = controller.apply(command, np.zeros(7), np.full(7, 30), step)
        assert np.max(np.abs(actual[:7] - command[:7])) <= 0.1000000001
        assert actual[-1].tobytes() == command[-1].tobytes()
        assert np.max(np.abs(trace["reference_acceleration"])) <= 20
        assert np.all(trace["next_coefficients"] >= controller.config.coefficient_lower)
        assert np.all(trace["next_coefficients"] <= controller.config.coefficient_upper)


def test_reset_restores_first_command_and_initial_coefficients():
    controller = ExecutionAdapter(AdapterConfig(mode="adaptive"))
    command = np.r_[np.ones(7), -1.0]
    first, _ = controller.apply(command, np.zeros(7), np.zeros(7), 0)
    controller.apply(command, np.zeros(7), np.zeros(7), 1)
    assert controller.coefficients.any()
    controller.reset()
    again, trace = controller.apply(command, np.zeros(7), np.zeros(7), 0)
    np.testing.assert_array_equal(first, again)
    assert not trace["coefficients"].any()
    assert not trace["reference_velocity"].any()
    assert not trace["reference_acceleration"].any()


def test_next_coefficients_cannot_change_current_command():
    slow = ExecutionAdapter(AdapterConfig(mode="adaptive", learning_rates=(0, 0, 0)))
    fast = ExecutionAdapter(AdapterConfig(mode="adaptive"))
    command = np.r_[np.ones(7), -1.0]
    a, _ = slow.apply(command, np.zeros(7), np.zeros(7), 0)
    b, _ = fast.apply(command, np.zeros(7), np.zeros(7), 0)
    np.testing.assert_array_equal(a, b)
    assert fast.coefficients.any() and not slow.coefficients.any()


def test_zero_error_stationary_target_remains_unchanged():
    controller = ExecutionAdapter(AdapterConfig(mode="adaptive"))
    command = np.r_[np.ones(7), -1.0]
    for step in range(30):
        actual, _ = controller.apply(command, np.ones(7), np.zeros(7), step)
        np.testing.assert_array_equal(actual, command)
    assert not controller.coefficients.any()


def test_feedback_reduces_lag_in_known_first_order_plant():
    # A sanity check on feedback sign and update timing, not evidence about Jar.
    errors = {}
    for mode in ("native", "fixed", "adaptive"):
        controller = ExecutionAdapter(AdapterConfig(mode=mode))
        q, dq = np.zeros(7), np.zeros(7)
        trajectory = []
        for step in range(120):
            target = np.full(7, min(step * 0.005, 0.4))
            command, _ = controller.apply(np.r_[target, -1.0], q, dq, step)
            previous = q.copy()
            q += 0.15 * (command[:7] - q)
            dq = (q - previous) / 0.05
            trajectory.append(float(np.abs(target - q).mean()))
        errors[mode] = np.mean(trajectory)
    assert errors["fixed"] < errors["native"]
    assert errors["adaptive"] < errors["native"]


def test_invalid_input_and_non_contiguous_steps_rejected():
    for bad in (np.full(8, np.nan), np.zeros(7), np.zeros(8, dtype=int)):
        with pytest.raises(ValueError):
            ExecutionAdapter(AdapterConfig()).apply(bad, np.zeros(7), np.zeros(7), 0)
    with pytest.raises(ValueError):
        ExecutionAdapter(AdapterConfig()).apply(np.zeros(8), np.zeros(7), np.zeros(7), 1)
    with pytest.raises(ValueError):
        replace(AdapterConfig(), dt=0.1)


def test_source_patch_is_exactly_reversible_and_precedes_clipping():
    path = ROOT / "experiments/scripts/patch_adapter05.py"
    spec = importlib.util.spec_from_file_location("patch_adapter05", path)
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    source_path = ROOT / "outputs/adapter05/upstream/maniguard/eval/benchmark.py"
    if not source_path.exists():
        pytest.skip("Pinned upstream source not downloaded on this machine")
    original = source_path.read_text()
    result = patch.patched(original)
    assert result.index("action_adapter.apply(step_idx") < result.index(
        "action_clipped = np.clip(ctrl_action"
    )
    for old, new in reversed(patch.EDITS):
        result = result.replace(new, old)
    assert result == original
