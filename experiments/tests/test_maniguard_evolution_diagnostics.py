"""Causal diagnostic events and correction magnitudes, using synthetic saved arrays."""

import numpy as np

from maniguard_evolution_diagnostics import phase_diagnostics


def test_phase_times_follow_observed_order_and_goal_timing_comes_from_original_evaluator():
    steps = np.arange(14)
    closed, supported = steps >= 2, steps < 12
    z = {"step": steps, "ap": np.column_stack([closed, supported]),
         "grasping": np.where(steps >= 2, 1, -1),
         "jar_robot_contacts": (steps >= 1).astype(int),
         "monitor_doomed": np.zeros(14, bool)}
    meta = {"ap_names": ["jar_closed", "jar_on_support"]}
    result = {"success": True, "success_first_step": 3, "success_step": 12,
              "first_contact_step": 1, "ever_contacted": True}
    traces = [{"native": [0.] * 7 + [1.], "modified": [.01] * 7 + [-1.],
               "arm_modified": True, "gripper_modified": True} for _ in range(13)]
    d = phase_diagnostics(z, meta, result, traces)
    assert d["first_contact_step"] == 1 and d["first_supported_closure_step"] == 2
    assert d["ten_control_grasp_after_closure_step"] == 11
    assert d["later_closed_grasped_departure_step"] == 12
    assert d["benchmark_ten_control_goal_hold_step"] == 12
    assert d["farthest_valid_phase"] == "carry" and not d["invalid_requested_prefix"]
    np.testing.assert_allclose(d["joint_residual_rms_rad"], [.01] * 7)
    assert d["gripper_overrides"] == 13 and not d["diagnostics_change_fitness"]
    z["ap"][1, 1] = False
    d = phase_diagnostics(z, meta, result, traces)
    assert d["invalid_requested_prefix"] and d["farthest_valid_phase"] == "invalid_requested_prefix"
