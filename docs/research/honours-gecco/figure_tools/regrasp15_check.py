"""Phase gate, knock check and candidate selection for the task 15 regrasp adapter.

Pure numpy and deterministic. The gate opens once the benchmark's jar_closed
proposition has held for a fixed number of consecutive controls while the jar is on
its support, upright and not dropped, and closes for the rest of the episode at the
first lift or violation. Inside the gate a candidate chunk is checked by its
predicted hand speed near the jar; the dial is that speed limit. Selection is
minimally invasive: the native chunk runs whenever it passes; otherwise the passing
candidate closest to native in joint space runs; if none passes, the candidate with
the smallest excess runs. Projection, the comparison filter, instead slows the native
chunk until it passes.
"""

from dataclasses import asdict, dataclass

import numpy as np

from regrasp15_kinematics import predict_path

CONTROL_HZ = 20.0
EXECUTED = 8


@dataclass(frozen=True)
class CheckConfig:
    mode: str = "native"  # native | select | project
    speed_limit_mps: float = float("inf")
    near_radius_m: float = 0.22
    candidates: int = 0
    gate: str = "closure"  # closure | always
    stable_controls: int = 10

    def __post_init__(self):
        if self.mode not in {"native", "select", "project"}:
            raise ValueError("Unknown adapter mode")
        if self.gate not in {"closure", "always"}:
            raise ValueError("Unknown gate")
        if self.mode == "native" and (self.candidates or np.isfinite(self.speed_limit_mps)):
            raise ValueError("Native mode has no check or candidates")
        if self.mode == "select" and self.candidates < 1:
            raise ValueError("Selection needs at least one candidate")
        if self.mode == "project" and self.candidates:
            raise ValueError("Projection samples no candidates")
        if self.mode != "native" and not 0 < self.speed_limit_mps < 10:
            raise ValueError("A finite positive speed limit is required")
        if not 0 < self.near_radius_m < 1 or self.stable_controls < 1:
            raise ValueError("Invalid gate geometry")

    def settings(self):
        return asdict(self)


class Gate:
    """Closure latch over per-control monitor propositions.

    The `always` ablation is open from the first control until a violation; a lift
    closes only the closure gate.
    """

    def __init__(self, config):
        self.config = config
        self.run = 0
        self.latched_at = None
        self.closed_at = None
        self.reason = None
        self.seen = 0

    def update(self, step, closed, on_support, upright, dropped):
        """Consume one control's propositions in step order."""
        if step < self.seen:
            raise ValueError("Gate observations must arrive in order")
        self.seen = step + 1
        if self.closed_at is not None:
            return
        if not upright or dropped:
            self.closed_at, self.reason = step, "violation"
            return
        if self.config.gate == "closure" and self.latched_at is not None and not on_support:
            self.closed_at, self.reason = step, "lift"
            return
        self.run = self.run + 1 if closed else 0
        if self.latched_at is None and self.run >= self.config.stable_controls and on_support:
            self.latched_at = step

    @property
    def open(self):
        if self.config.gate == "always":
            return self.closed_at is None
        return self.latched_at is not None and self.closed_at is None


def features(chunk, q_now, eef_now, jar_pos, yaw, near_radius_m):
    """Predicted hand speeds and horizontal jar distances over the executed steps."""
    chunk = np.asarray(chunk, dtype=np.float64)
    path = predict_path(q_now, eef_now, chunk[:EXECUTED, :7], yaw)
    points = np.vstack([np.asarray(eef_now, dtype=np.float64)[None], path])
    speed = np.linalg.norm(np.diff(points, axis=0), axis=1) * CONTROL_HZ
    horizontal = np.linalg.norm(path[:, :2] - np.asarray(jar_pos, dtype=np.float64)[:2], axis=1)
    start = np.linalg.norm(points[:-1, :2] - np.asarray(jar_pos, dtype=np.float64)[:2], axis=1)
    near = np.minimum(horizontal, start) < near_radius_m
    near_speed = float(speed[near].max()) if near.any() else 0.0
    return {
        "speed_mps": speed,
        "horizontal_m": horizontal,
        "near": near,
        "max_near_speed_mps": near_speed,
        "min_horizontal_m": float(horizontal.min()),
        "path": path,
    }


def excess(feature, limit):
    return max(0.0, feature["max_near_speed_mps"] - limit)


def joint_distance(a, b):
    a = np.asarray(a, dtype=np.float64)[:EXECUTED, :7]
    b = np.asarray(b, dtype=np.float64)[:EXECUTED, :7]
    return float(np.sqrt(np.mean((a - b) ** 2)))


def select(native, candidates, q_now, eef_now, jar_pos, yaw, config):
    """Index into [native, *candidates] chosen by the minimally invasive rule."""
    pool = [np.asarray(native, dtype=np.float64)]
    pool += [np.asarray(c, dtype=np.float64) for c in candidates]
    feats = [features(c, q_now, eef_now, jar_pos, yaw, config.near_radius_m) for c in pool]
    over = [excess(f, config.speed_limit_mps) for f in feats]
    distance = [joint_distance(c, pool[0]) for c in pool]
    if over[0] == 0.0:
        choice, rule = 0, "native_passes"
    else:
        passing = [i for i in range(1, len(pool)) if over[i] == 0.0]
        if passing:
            choice, rule = min(passing, key=lambda i: (distance[i], i)), "closest_passing"
        else:
            choice = min(range(len(pool)), key=lambda i: (over[i], distance[i], i))
            rule = "least_excess"
    return choice, rule, feats, over, distance


def project(native, q_now, eef_now, jar_pos, yaw, config):
    """Slow the native chunk toward the current joints until its near speed passes."""
    native = np.asarray(native, dtype=np.float64)
    feat = features(native, q_now, eef_now, jar_pos, yaw, config.near_radius_m)
    over = excess(feat, config.speed_limit_mps)
    if over == 0.0:
        return native.copy(), 1.0, feat
    scale = config.speed_limit_mps / feat["max_near_speed_mps"]
    for _ in range(20):
        result = native.copy()
        q = np.asarray(q_now, dtype=np.float64)[:7]
        result[:, :7] = q + scale * (native[:, :7] - q)
        check = features(result, q_now, eef_now, jar_pos, yaw, config.near_radius_m)
        if excess(check, config.speed_limit_mps) <= 1e-9:
            return result, float(scale), check
        scale *= 0.9
    return result, float(scale), check
