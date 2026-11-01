"""Compact recurrent residual control around a frozen ManiGuard joint policy.

All controller inputs are causal. Object poses, propositions and grasp states are
privileged simulator observations. The passive benchmark monitor never vetoes an
action. Parameter bounds limit commands, not the probability of physical harm.
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.transform import Rotation


FEATURE_NAMES = (
    *(f"eef_in_jar_{axis}" for axis in "xyz"),
    *(f"eef_rotation_in_jar_{axis}" for axis in "xyz"),
    *(f"joint_target_error_{i}" for i in range(7)),
    *(f"joint_velocity_{i}" for i in range(7)),
    *(f"jar_up_{axis}" for axis in "xyz"),
    "native_gripper", "closed", "supported", "upright", "dropped",
    "grasped", "grasp_known", "closed_seen", "grasp_seen",
    "controls_without_progress", "eef_displacement", "contact",
)


@dataclass(frozen=True)
class AdapterSpec:
    hidden_size: int = 4
    correction_limit_rad: float = 0.08
    smoothing: float = 0.5
    gripper_threshold: float = 0.6
    position_scale_m: float = 0.25
    joint_error_scale_rad: float = 0.2
    joint_velocity_scale: float = 2.0
    stall_scale_controls: int = 400

    def __post_init__(self):
        if not isinstance(self.hidden_size, int) or not 1 <= self.hidden_size <= 64:
            raise ValueError("hidden_size must be an integer between 1 and 64")
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Adapter settings must be finite")
        if self.correction_limit_rad < 0 or not 0 <= self.smoothing < 1:
            raise ValueError("Invalid residual bound or smoothing")
        if not 0 < self.gripper_threshold < 1:
            raise ValueError("Invalid gripper deadband")
        if min(self.position_scale_m, self.joint_error_scale_rad,
               self.joint_velocity_scale, self.stall_scale_controls) <= 0:
            raise ValueError("Feature scales must be positive")

    @property
    def parameter_count(self):
        h = self.hidden_size
        return h * len(FEATURE_NAMES) + h * h + h + 9 * h + 9

    def unpack(self, parameters):
        theta = np.asarray(parameters, dtype=np.float64)
        if theta.shape != (self.parameter_count,) or not np.isfinite(theta).all():
            raise ValueError("Wrong parameter count or nonfinite controller parameters")
        h, n = self.hidden_size, len(FEATURE_NAMES)
        sizes = (h * n, h * h, h, 9 * h, 9)
        parts = np.split(theta, np.cumsum(sizes)[:-1])
        return parts[0].reshape(h, n), parts[1].reshape(h, h), parts[2], \
            parts[3].reshape(9, h), parts[4]

    def initial_parameters(self, seed=0):
        theta = np.zeros(self.parameter_count)
        # Diverse hidden features, with exactly zero output corrections at initialization.
        n = self.hidden_size * len(FEATURE_NAMES)
        theta[:n] = np.random.default_rng(seed).normal(0, 1 / np.sqrt(len(FEATURE_NAMES)), n)
        return theta


class ResidualAdapter:
    def __init__(self, spec, parameters, *, enabled=True):
        self.spec, self.enabled = spec, enabled
        self.weights = spec.unpack(parameters)
        self.hidden = np.zeros(spec.hidden_size)
        self.residual = np.zeros(7)
        self.closed_seen = self.grasp_seen = False
        self.previous_stage = self.last_progress_step = self.next_step = 0
        self.previous_eef = None

    def features(self, step, native, observation):
        if not observation.get("valid", False):
            raise ValueError("Invalid causal observation")
        values = {key: np.asarray(observation[key], dtype=np.float64) for key in
                  ("q", "dq", "eef_pos", "eef_quat", "jar_pos", "jar_quat")}
        shapes = {"q": (7,), "dq": (7,), "eef_pos": (3,), "eef_quat": (4,),
                  "jar_pos": (3,), "jar_quat": (4,)}
        if any(v.shape != shapes[k] or not np.isfinite(v).all() for k, v in values.items()):
            raise ValueError("Missing, nonfinite or malformed physical observation")
        ap = observation["ap"]
        keys = ("jar_closed", "jar_on_support", "jar_upright", "jar_dropped")
        if any(k not in ap for k in keys) or observation["grasp"] not in (-1, 0, 1):
            raise ValueError("Unknown proposition or grasp-state convention")
        closed, supported, upright, dropped = (bool(ap[k]) for k in keys)
        grasped = observation["grasp"] == 1
        self.closed_seen |= closed
        self.grasp_seen |= grasped
        stage = int(self.closed_seen) + int(self.closed_seen and grasped)
        if stage > self.previous_stage:
            self.last_progress_step = step
        self.previous_stage = max(stage, self.previous_stage)
        jar_rot = Rotation.from_quat(values["jar_quat"])
        relative_position = jar_rot.inv().apply(values["eef_pos"] - values["jar_pos"])
        relative_rotation = jar_rot.inv() * Rotation.from_quat(values["eef_quat"])
        displacement = 0.0 if self.previous_eef is None else float(
            np.linalg.norm(values["eef_pos"] - self.previous_eef))
        self.previous_eef = values["eef_pos"].copy()
        s = self.spec
        features = np.concatenate((
            relative_position / s.position_scale_m,
            relative_rotation.as_rotvec() / np.pi,
            (native[:7] - values["q"]) / s.joint_error_scale_rad,
            values["dq"] / s.joint_velocity_scale,
            jar_rot.apply([0, 0, 1]),
            [native[7], closed, supported, upright, dropped, grasped,
             observation["grasp"] != 0, self.closed_seen, self.grasp_seen,
             (step - self.last_progress_step) / s.stall_scale_controls,
             displacement / 0.01, bool(observation.get("contact", False))],
        ))
        if features.shape != (len(FEATURE_NAMES),):
            raise AssertionError("Feature contract changed")
        return np.clip(features, -1, 1)

    def apply(self, step, native, observation):
        native = np.asarray(native)
        if step != self.next_step or native.shape != (8,) or native.dtype.kind != "f":
            raise ValueError("Expected contiguous controls and eight floating-point targets")
        if not np.isfinite(native).all() or native[7] not in (-1, 1):
            raise ValueError("Expected finite targets and a binarized native gripper")
        features = self.features(step, native, observation)
        w, recurrent, bias, output, output_bias = self.weights
        before = self.hidden.copy()
        self.hidden = np.tanh(w @ features + recurrent @ self.hidden + bias)
        logits = output @ self.hidden + output_bias
        strength = float(1 / (1 + np.exp(-np.clip(logits[8], -40, 40))))
        requested = self.spec.correction_limit_rad * strength * np.tanh(logits[:7])
        self.residual = (self.spec.smoothing * self.residual
                         + (1 - self.spec.smoothing) * requested)
        corrected = native.copy()
        grip_score = float(np.tanh(logits[7]))
        if self.enabled:
            corrected[:7] += self.residual.astype(native.dtype)
            if strength >= 0.5 and abs(grip_score) >= self.spec.gripper_threshold:
                corrected[7] = -1 if grip_score > 0 else 1
        self.next_step += 1
        return corrected, {
            "step": step, "features": features, "hidden_before": before,
            "native_dtype": str(native.dtype),
            "hidden_after": self.hidden.copy(), "strength": strength,
            "gripper_score": grip_score, "native": native.copy(),
            "modified": corrected.copy(), "enabled": self.enabled,
            "arm_modified": bool(np.any(corrected[:7] != native[:7])),
            "gripper_modified": bool(corrected[7] != native[7]),
        }


def observation_from_row(row, ap_names, arm_indices):
    return {
        "valid": bool(row["ap_valid"]),
        "q": np.asarray(row["q"])[arm_indices],
        "dq": np.asarray(row["dq"])[arm_indices],
        **{key: row[key] for key in ("eef_pos", "eef_quat", "jar_pos", "jar_quat")},
        "grasp": int(row["grasping"]),
        "ap": dict(zip(ap_names, np.asarray(row["ap"]).tolist())),
        "contact": int(row["jar_robot_contacts"]) > 0,
    }
