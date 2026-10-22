"""Uniform joint caps and causal refresh for the phase-three recovery study.

These maps constrain joint targets. They are not a physical safety certificate.
Call ``applied`` with the final environment input after downstream clipping.
"""

from dataclasses import dataclass

import numpy as np

TOLERANCE = 1e-6
NORMALIZER = np.sqrt(7.0) * 0.04


@dataclass(frozen=True)
class Config:
    cap: float | None = None
    geometry: str = "direction"
    threshold: float | None = None
    minimum_interval: int = 4

    def __post_init__(self):
        if isinstance(self.cap, bool) or self.cap not in (None, 0.12, 0.08, 0.04):
            raise ValueError("Cap must be disabled or .12/.08/.04 radians")
        if self.geometry not in ("direction", "componentwise"):
            raise ValueError("Unknown cap geometry")
        if self.threshold is not None and (
            isinstance(self.threshold, bool)
            or not np.isfinite(self.threshold)
            or self.threshold < 0
        ):
            raise ValueError("Threshold must be disabled or finite and nonnegative")
        if type(self.minimum_interval) is not int or self.minimum_interval not in (4, 6):
            raise ValueError("Minimum interval must be four or six")


def cap_targets(proposal, joints, config):
    p, q = np.asarray(proposal), np.asarray(joints)
    if p.shape != (8,) or p.dtype.kind != "f" or not np.isfinite(p).all():
        raise ValueError("Eight finite floating point policy targets required")
    if q.shape != (7,) or q.dtype.kind not in "fiu" or not np.isfinite(q).all():
        raise ValueError("Seven finite measured joints required")
    result = p.copy()
    if config.cap is None:
        return result
    # No arithmetic on unchanged proposals, including the gripper coordinate.
    error = p[:7].astype(np.float64) - q.astype(np.float64)
    maximum = float(np.max(np.abs(error)))
    if maximum <= config.cap:
        return result
    if config.geometry == "direction":
        error *= config.cap / maximum
    else:
        error = np.clip(error, -config.cap, config.cap)
    result[:7] = q.astype(np.float64) + error
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite adapted target")
    return result


def final_input(adapted, low, high):
    """Apply recorded action-space bounds, preserving the proposal's dtype."""
    adapted = np.asarray(adapted)
    low, high = np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)
    if (adapted.shape != (8,) or adapted.dtype.kind != "f"
            or low.shape != (8,) or high.shape != (8,)
            or np.isnan(low).any() or np.isnan(high).any() or np.any(low > high)):
        raise ValueError("Valid eight-coordinate bounds required")
    result = np.clip(adapted, low, high).astype(adapted.dtype)
    if not np.isfinite(result).all():
        raise ValueError("Final environment input must be finite")
    return result


def normalized_correction(proposal, executed):
    difference = np.asarray(proposal, dtype=np.float64)[:7] - np.asarray(
        executed, dtype=np.float64
    )[:7]
    difference[np.abs(difference) <= TOLERANCE] = 0.0
    return float(np.linalg.norm(difference) / NORMALIZER)


def direction_angle(proposal, adapted, joints):
    original = np.asarray(proposal, dtype=np.float64)[:7] - joints
    changed = np.asarray(adapted, dtype=np.float64)[:7] - joints
    denominator = np.linalg.norm(original) * np.linalg.norm(changed)
    if denominator == 0:
        return None
    cosine = np.clip(np.dot(original, changed) / denominator, -1, 1)
    return float(np.degrees(np.arccos(cosine)))


class RecoveryController:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.pending = None
        self.since_query = 0
        self.correction = 0.0

    @property
    def horizon(self):
        return self.config.minimum_interval if self.config.threshold == 0 else 8

    def on_query(self):
        if self.pending is not None:
            raise ValueError("Apply pending proposal before a policy query")
        self.since_query = 0
        self.correction = 0.0

    def act(self, proposal, joints):
        if self.pending is not None:
            raise ValueError("Apply pending proposal before the next action")
        adapted = cap_targets(proposal, joints, self.config)
        self.pending = np.asarray(proposal).copy()
        return adapted

    def applied(self, executed):
        executed = np.asarray(executed)
        if self.pending is None or executed.shape != (8,) or not np.isfinite(executed).all():
            raise ValueError("One pending proposal and finite final input required")
        self.correction += normalized_correction(self.pending, executed)
        self.since_query += 1
        self.pending = None

    def replan_reason(self):
        if self.pending is not None:
            raise ValueError("Scheduling must follow execution")
        if self.since_query >= 8:
            return "maximum_interval"
        threshold = self.config.threshold
        if (threshold is not None and self.since_query >= self.config.minimum_interval
                and self.correction >= threshold):
            return "fixed_minimum" if threshold == 0 else "correction_threshold"
        return None
