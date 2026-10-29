"""Evolvable near-jar speed projection for the task 19 Jar adapter.

Task 15's projection arm, with one added gene. A chunk's predicted hand path is checked
at each executed step that starts or ends within `near_radius_m` of the jar base
horizontally. A step that moves the hand toward the jar counts at its full speed; a
step that does not approach counts at `retreat_weight` times its speed. A chunk fails
when the largest counted speed exceeds `speed_limit_mps`, and a failing chunk is scaled
toward the current joints until it passes. With retreat_weight = 1 the check and the
projection are exactly task 15's (`regrasp15_check.features` and `project`).

Shield mode (amendment 3) leaves speed alone. At a query where the jar is tilted at least
`tilt_limit_deg` from vertical and touched the robot within the last `contact_window`
controls (amendment 4; 1 means the query control only), the chunk retraces the arm's own
recorded joint path back to where the hand was `retreat_m` away; consecutive triggers
continue from where the previous retrace ended. The gripper holds the native chunk's first
command.
"""

import math
from dataclasses import asdict, dataclass

import numpy as np

from regrasp15_check import CONTROL_HZ, EXECUTED
from regrasp15_kinematics import predict_path


@dataclass(frozen=True)
class EvoConfig:
    mode: str = "native"  # native | project | shield
    speed_limit_mps: float = float("inf")
    near_radius_m: float = 0.22
    retreat_weight: float = 1.0
    gate: str = "closure"  # closure | always
    stable_controls: int = 10
    candidates: int = 0  # never used; Gate reads only gate and stable_controls
    tilt_limit_deg: float = float("inf")  # shield only
    retreat_m: float = 0.0  # shield only
    contact_window: int = 1  # shield only; controls up to the query that count as contact

    def __post_init__(self):
        if self.mode not in {"native", "project", "shield"}:
            raise ValueError("Unknown adapter mode")
        if self.mode == "shield":
            if (math.isfinite(self.speed_limit_mps) or self.gate != "always"
                    or not 0 < self.tilt_limit_deg < 90
                    or not 0 < self.retreat_m <= 0.3
                    or not isinstance(self.contact_window, int)
                    or not 1 <= self.contact_window <= EXECUTED):
                raise ValueError("Invalid shield settings")
        elif math.isfinite(self.tilt_limit_deg) or self.retreat_m or self.contact_window != 1:
            raise ValueError("Shield genes are set outside shield mode")
        if self.gate not in {"closure", "always"}:
            raise ValueError("Unknown gate")
        if self.mode == "native" and math.isfinite(self.speed_limit_mps):
            raise ValueError("Native mode has no check")
        if self.mode == "project" and not 0 < self.speed_limit_mps < 10:
            raise ValueError("A finite positive speed limit is required")
        if not 0 < self.near_radius_m < 1 or self.stable_controls < 1:
            raise ValueError("Invalid gate geometry")
        if not 0 <= self.retreat_weight <= 1 or self.candidates:
            raise ValueError("Invalid retreat weight or candidate count")

    def settings(self):
        """Settings as journaled; shield genes appear only in shield mode, so native and
        projection journals keep task 19's original format."""
        out = asdict(self)
        if self.mode != "shield":
            del out["tilt_limit_deg"], out["retreat_m"]
        if out["contact_window"] == 1:  # stage 3 journals predate the field
            del out["contact_window"]
        return out


def tilt_deg(quat_xyzw):
    """Angle between the jar's axis and world vertical, from an (x, y, z, w) quaternion."""
    x, y = float(quat_xyzw[0]), float(quat_xyzw[1])
    return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))


def window_contacts(contacts, step, config):
    """Largest robot-jar contact count over the controls in the shield's contact window."""
    return int(max(int(c) for c in contacts[max(0, step - config.contact_window + 1):step + 1]))


def triggered(tilt, contacts, config):
    return config.mode == "shield" and tilt >= config.tilt_limit_deg and int(contacts) > 0


def retrace(native, q_history, eef_history, origin, retreat_m):
    """Chunk retracing the recorded arm joints back from control `origin`.

    The target is the latest earlier control whose hand position lies at least
    `retreat_m` from the hand at `origin` (control 0 if none). Row k targets the joints
    recorded round(min(k + 1, 8) * span / 8) controls before `origin`, so the executed
    steps retrace the recorded path and the rest of the chunk holds the target. Returns
    the chunk and the target control, where a consecutive retrace starts.
    """
    native = np.asarray(native, dtype=np.float64)
    start = np.asarray(eef_history[origin], dtype=np.float64)
    target = 0
    for t in range(origin - 1, -1, -1):
        if np.linalg.norm(np.asarray(eef_history[t], dtype=np.float64) - start) >= retreat_m:
            target = t
            break
    span = origin - target
    result = np.empty_like(native)
    for k in range(native.shape[0]):
        back = round(min(k + 1, EXECUTED) * span / EXECUTED)
        result[k, :7] = np.asarray(q_history[origin - back], dtype=np.float64)
    result[:, 7:] = native[0, 7:]
    return result, target


def features(chunk, q_now, eef_now, jar_pos, yaw, config):
    """Counted hand speeds near the jar over the executed steps."""
    chunk = np.asarray(chunk, dtype=np.float64)
    path = predict_path(q_now, eef_now, chunk[:EXECUTED, :7], yaw)
    points = np.vstack([np.asarray(eef_now, dtype=np.float64)[None], path])
    speed = np.linalg.norm(np.diff(points, axis=0), axis=1) * CONTROL_HZ
    jar = np.asarray(jar_pos, dtype=np.float64)[:2]
    horizontal = np.linalg.norm(path[:, :2] - jar, axis=1)
    start = np.linalg.norm(points[:-1, :2] - jar, axis=1)
    near = np.minimum(horizontal, start) < config.near_radius_m
    approaching = horizontal < start
    counted = np.where(approaching, speed, speed * config.retreat_weight)
    near_speed = float(counted[near].max()) if near.any() else 0.0
    return {"speed_mps": speed, "counted_mps": counted, "near": near,
            "max_near_speed_mps": near_speed, "min_horizontal_m": float(horizontal.min())}


def excess(feature, limit):
    return max(0.0, feature["max_near_speed_mps"] - limit)


def project(native, q_now, eef_now, jar_pos, yaw, config):
    """Slow the native chunk toward the current joints until its counted speed passes."""
    native = np.asarray(native, dtype=np.float64)
    feat = features(native, q_now, eef_now, jar_pos, yaw, config)
    if excess(feat, config.speed_limit_mps) == 0.0:
        return native.copy(), 1.0, feat
    scale = config.speed_limit_mps / feat["max_near_speed_mps"]
    q = np.asarray(q_now, dtype=np.float64)[:7]
    for _ in range(20):
        result = native.copy()
        result[:, :7] = q + scale * (native[:, :7] - q)
        check = features(result, q_now, eef_now, jar_pos, yaw, config)
        if excess(check, config.speed_limit_mps) <= 1e-9:
            return result, float(scale), check
        scale *= 0.9
    return result, float(scale), check
