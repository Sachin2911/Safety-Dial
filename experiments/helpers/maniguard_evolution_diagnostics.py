"""Passive phase and correction summaries from an already saved episode."""

import numpy as np


def first_step(steps, mask):
    indices = np.flatnonzero(mask)
    return int(steps[indices[0]]) if len(indices) else None


def phase_diagnostics(recording, meta, result, traces):
    z, names = recording, meta["ap_names"]
    steps = z["step"]
    closed = z["ap"][:, names.index("jar_closed")].astype(bool)
    supported = z["ap"][:, names.index("jar_on_support")].astype(bool)
    grasped = z["grasping"] == 1
    closure = first_step(steps, closed)
    departure = first_step(steps, ~supported)
    invalid_prefix = departure is not None and (closure is None or closure > departure)
    stage, streak, grasp_step, carry_step = 0, 0, None, None
    previous_support = None
    for step, c, s, g in zip(steps, closed, supported, grasped, strict=True):
        before = stage
        if c and s:
            stage = max(stage, 1)
        streak = streak + 1 if stage >= 1 and c and g else 0
        if streak >= 10:
            stage = max(stage, 2)
            if grasp_step is None:
                grasp_step = int(step)
        if before >= 2 and previous_support and not s and c and g:
            stage = 3
            if carry_step is None:
                carry_step = int(step)
        previous_support = bool(s)
    contact = first_step(steps, z["jar_robot_contacts"] > 0)
    phases = ("contact" if contact is not None else "reaching", "closure", "grasp", "carry")
    residuals = np.asarray([np.asarray(t["modified"], float)[:7]
                            - np.asarray(t["native"], float)[:7] for t in traces])
    if residuals.shape != (len(traces), 7) or not len(traces):
        raise ValueError("Diagnostics require a nonempty complete action trace")
    return {
        "first_contact_step": contact,
        "original_first_contact_step": result.get("first_contact_step"),
        "first_supported_closure_step": first_step(steps, closed & supported),
        "ten_control_grasp_after_closure_step": grasp_step,
        "later_closed_grasped_departure_step": carry_step,
        "first_departure_step": departure,
        "first_raw_violation_step": first_step(steps, z["monitor_doomed"]),
        "benchmark_first_goal_step": result.get("success_first_step"),
        "benchmark_ten_control_goal_hold_step": result.get("success_step")
        if result["success"] else None,
        "goal_timing_source": "Original evaluator success_first_step and success_step",
        "ordered_stage_reached": stage,
        "invalid_requested_prefix": invalid_prefix,
        "farthest_valid_phase": "invalid_requested_prefix" if invalid_prefix else phases[stage],
        "arm_modified_controls": sum(bool(t["arm_modified"]) for t in traces),
        "arm_modified_fraction": sum(bool(t["arm_modified"]) for t in traces) / len(traces),
        "gripper_overrides": sum(bool(t["gripper_modified"]) for t in traces),
        "joint_residual_rms_rad": np.sqrt(np.mean(residuals ** 2, axis=0)).tolist(),
        "joint_residual_max_abs_rad": np.max(np.abs(residuals), axis=0).tolist(),
        "residual_rms_rad": float(np.sqrt(np.mean(residuals ** 2))),
        "benchmark_goal_held": bool(result["success"]),
        "engaged": bool(result["ever_contacted"]),
        "diagnostics_change_fitness": False,
    }
