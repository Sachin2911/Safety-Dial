"""Checks for subtle timing and joint-order mistakes in the diagnostic."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.tracking import aligned_arm, requested_sequence  # noqa: E402


def test_alignment_uses_step_ids_and_robot_joint_order():
    data = {
        "step": np.arange(10, 15),
        "exec_step": np.array([10, 12, 13]),
        "exec_clipped": np.arange(24).reshape(3, 8),
        "q": np.arange(45).reshape(5, 9),
    }
    meta = {"arm_control_idx": [6, 5, 4, 3, 2, 1, 0]}
    target, actual, steps = aligned_arm(data, meta, lag=1)
    np.testing.assert_array_equal(steps, [10, 12, 13])
    np.testing.assert_array_equal(target, data["exec_clipped"][:, :7])
    np.testing.assert_array_equal(actual, data["q"][[1, 3, 4]][:, [6, 5, 4, 3, 2, 1, 0]])
    _, actual_later, steps_later = aligned_arm(data, meta, lag=3)
    np.testing.assert_array_equal(steps_later, [10])
    np.testing.assert_array_equal(actual_later, data["q"][[3]][:, [6, 5, 4, 3, 2, 1, 0]])


def test_delayed_tracking_has_expected_lag_and_no_wraparound():
    # Target issued at t is reached at t+2; future samples past the episode are excluded.
    target = np.arange(8, dtype=float)
    measured = np.array([-2, -1, 0, 1, 2, 3, 4, 5, 6], dtype=float)
    data = {"step": np.arange(9), "exec_step": np.arange(8),
            "exec_clipped": np.repeat(target[:, None], 8, axis=1),
            "q": np.repeat(measured[:, None], 9, axis=1)}
    meta = {"arm_control_idx": list(range(7))}
    errors = []
    for lag in range(4):
        desired, actual, steps = aligned_arm(data, meta, lag)
        errors.append(float(np.mean(np.abs(desired - actual))))
        assert len(steps) == min(8, 9 - lag)
    np.testing.assert_allclose(errors, [2, 1, 0, 1])


def test_closure_after_lift_does_not_satisfy_requested_order():
    assert not requested_sequence({"closed": 10, "lift": 9})
    assert requested_sequence({"closed": 10, "lift": 10})
    assert not requested_sequence({"closed": None, "lift": None})
    assert requested_sequence({"closed": 10, "lift": None})
