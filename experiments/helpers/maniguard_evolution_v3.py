"""Separately versioned improvements, not applied to the running v2 study."""

import numpy as np

from maniguard_adapter import FEATURE_NAMES


def ordered_progress_v3(closed, supported, grasped):
    """A departure must occur after the ten-control grasp has been established."""
    stage, streak, previous_support = 0, 0, None
    for c, s, g in zip(closed, supported, grasped, strict=True):
        before = stage
        if c and s:
            stage = max(stage, 1)
        streak = streak + 1 if stage >= 1 and c and g else 0
        if streak >= 10:
            stage = max(stage, 2)
        if before >= 2 and previous_support and not s and c and g:
            stage = 3
        previous_support = bool(s)
    return stage


def safe_target_contact(contacts, supported, upright, dropped, *, violation, closure, departure):
    prefix_valid = departure is None or (closure is not None and closure <= departure)
    if violation or not prefix_valid:
        return False
    return bool(np.any((np.asarray(contacts) > 0) & np.asarray(supported, bool)
                       & np.asarray(upright, bool) & ~np.asarray(dropped, bool)))


def contact_score_v3(rows):
    if not rows or any(not row.get("valid", False) for row in rows):
        return -1_000_000.0
    n = len(rows)
    primary = sum(row["strict_safe_completion"] for row in rows)
    secondary = sum(row["safe_stage_progress"] + 0.25 * row["safe_target_contact"] for row in rows)
    return float((4 * n + 2) * primary + secondary
                 - sum(row["raw_violation"] for row in rows) / (n + 1))


def mutation_scales(spec, sigma, profile):
    """Keep arm exploration small without suppressing discrete gripper exploration."""
    if profile == "isotropic":
        return np.ones(spec.parameter_count)
    if profile != "conservative_arm_v3" or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Unknown mutation profile or invalid sigma")
    h = spec.hidden_size
    target = np.full(spec.parameter_count, 0.05)
    offset = h * len(FEATURE_NAMES) + h * h + h
    output = target[offset:offset + 9 * h].reshape(9, h)
    output[:7] = 0.025
    output[7] = 0.3
    target[-9:-2] = 0.025
    target[-2] = 0.3
    return target / sigma
