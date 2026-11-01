"""Small persistent PushT proposal controllers for the new seven-hour campaign.

The 24 coefficients are a linear neural policy over fixed causal features. They
produce residual candidates, never unchecked commands. Numerical execution and
software tests belong on the declared rented hosts only.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import torch

from helpers.lewm18_planner import PlanningCEM

FEATURE_NAMES = ("bias", "hazard_proximity", "latent_stall", "log_goal_latent_cost")
BASIS_NAMES = ("outward_hazard_normal", "clockwise_hazard_tangent", "shifted_plan_direction")
TEMPORAL_NAMES = ("constant", "linear_decay")
FORMAT = "lewm-push-structured-residual-v1"


def _unit(value, fallback=(0.0, 0.0)):
    value = np.asarray(value, dtype=np.float64)
    norm = np.linalg.norm(value)
    return value / norm if norm > 1e-12 else np.asarray(fallback, dtype=np.float64)


def _box_frame(point, centre, half):
    local = point - centre
    q = np.abs(local) - half
    distance = np.linalg.norm(np.maximum(q, 0)) + min(float(q.max()), 0)
    nearest = np.clip(local, -half, half)
    if (q > 0).any():
        normal = _unit(local - nearest)
    else:
        axis = int(np.argmax(q))
        normal = np.zeros(2)
        normal[axis] = 1 if local[axis] >= 0 else -1
    return float(distance), normal


def hazard_frame(point, case, body_radius=15.0):
    """Signed disk clearance and outward normal, with no dynamics prediction."""
    point = np.asarray(point, dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("A finite native pusher position is required")
    kind = case["kind"]
    if kind == "none":
        return 1e6, np.zeros(2)
    origin, basis = np.asarray(case["origin"]), np.asarray(case["basis"])
    if origin.shape != (2,) or basis.shape != (2, 2):
        raise ValueError("Malformed hazard coordinate frame")
    if not np.isfinite(origin).all() or not np.isfinite(basis).all():
        raise ValueError("Nonfinite hazard coordinate frame")
    if not np.allclose(basis @ basis.T, np.eye(2), rtol=0, atol=1e-6):
        raise ValueError("Hazard basis must be orthonormal")
    xy = (point - origin) @ basis.T
    if kind == "island":
        length = np.linalg.norm(xy)
        signed = length - float(case["radius_px"])
        normal = _unit(xy, (1, 0))
    elif kind == "bar":
        nearest = np.array(
            [np.clip(xy[0], -case["segment_half_length_px"], case["segment_half_length_px"]), 0]
        )
        delta = xy - nearest
        signed = np.linalg.norm(delta) - float(case["radius_px"])
        normal = _unit(delta, (0, 1))
    elif kind == "gate":
        extent, offset = case["extent_px"], case["gap_offset_px"]
        half_gap, band = case["gap_width_px"] / 2, case["band_half_width_px"]
        parts = []
        for low, high in ((-extent, offset - half_gap), (offset + half_gap, extent)):
            if high <= low or band <= 0:
                raise ValueError("Invalid gate box")
            parts.append(
                _box_frame(xy, np.array([(low + high) / 2, 0]), np.array([(high - low) / 2, band]))
            )
        signed, normal = min(parts, key=lambda value: value[0])
    else:
        raise ValueError("Unsupported PushT hazard")
    return float(signed - body_radius), normal @ basis


@dataclass(frozen=True)
class PushEvolutionSpec:
    max_horizon: int = 5
    action_block: int = 5
    action_dim: int = 2
    native_residual_bound: float = 0.25
    proximity_length_px: float = 60.0
    body_radius_px: float = 15.0
    margin_px: float = 6.0

    def __post_init__(self):
        if self.max_horizon != 5 or self.action_block != 5 or self.action_dim != 2:
            raise ValueError("The structured contract is five blocks of five 2D actions")
        values = (self.native_residual_bound, self.proximity_length_px, self.body_radius_px)
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError("Declare finite positive proposal scales")
        if not np.isfinite(self.margin_px) or self.margin_px < 0:
            raise ValueError("Invalid fixed margin")

    @property
    def parameter_count(self):
        return len(TEMPORAL_NAMES) * len(BASIS_NAMES) * len(FEATURE_NAMES)

    @property
    def block_action_dim(self):
        return self.action_block * self.action_dim

    def initial_parameters(self, seed=None):
        return np.zeros(self.parameter_count, dtype=np.float64)

    def random_parameters(self, seed, scale=0.1):
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Random-control scale must be positive")
        return np.random.default_rng(seed).normal(0, scale, self.parameter_count)


class PushEvolutionAdapter:
    """Bounded geometry feedback and shifted-plan direction, without true goal XY."""

    def __init__(self, spec, parameters, case, action_mean, action_scale):
        self.spec, self.case = spec, case
        self.parameters = np.asarray(parameters, dtype=np.float64).copy()
        if (
            self.parameters.shape != (spec.parameter_count,)
            or not np.isfinite(self.parameters).all()
        ):
            raise ValueError("Malformed persistent controller weights")
        self.action_mean = np.asarray(action_mean, dtype=np.float64).copy()
        self.action_scale = np.asarray(action_scale, dtype=np.float64).copy()
        if self.action_mean.shape != (2,) or self.action_scale.shape != (2,):
            raise ValueError("Expected the frozen two-dimensional action scaler")
        if (
            not np.isfinite(self.action_mean).all()
            or not np.isfinite(self.action_scale).all()
            or (self.action_scale <= 0).any()
        ):
            raise ValueError("Invalid action scaler")
        if case["kind"] not in ("none", "island", "bar", "gate"):
            raise ValueError("Unsupported hazard")
        self.history, self.plan_context = [], None

    def set_plan_context(self, init_action):
        if init_action is None:
            self.plan_context = None
            return
        value = (
            init_action.detach().cpu().double().numpy()
            if torch.is_tensor(init_action)
            else np.asarray(init_action, dtype=np.float64)
        )
        if (
            value.ndim != 3
            or value.shape[0] != 1
            or value.shape[-1] != 10
            or not np.isfinite(value).all()
        ):
            raise ValueError("Malformed causal shifted native plan")
        self.plan_context = value.copy()

    def features(self, planner_features, checker):
        if getattr(checker, "observed", None) is None:
            raise ValueError("Causal native pusher proprioception is required")
        observed = checker.observed.detach().cpu().double().numpy().reshape(-1)
        if observed.shape != (4,) or not np.isfinite(observed).all():
            raise ValueError("Expected observed pusher x,y,vx,vy")
        clearance, normal = hazard_frame(observed[:2], self.case, self.spec.body_radius_px)
        tangent = np.array([normal[1], -normal[0]])
        plan_direction = np.zeros(2)
        if self.plan_context is not None and self.plan_context.shape[1]:
            native = self.plan_context[0, 0].reshape(5, 2) * self.action_scale + self.action_mean
            plan_direction = _unit(native.mean(0))
        f = planner_features
        required = ("current_cost", "improvement", "replan")
        if not all(np.isfinite(f[key]) for key in required):
            raise ValueError("Nonfinite causal planner feature")
        near = (
            0.0
            if self.case["kind"] == "none"
            else np.exp(-max(0.0, clearance - self.spec.margin_px) / self.spec.proximity_length_px)
        )
        stalled = np.clip((0.02 - f["improvement"]) / 0.02, 0, 1) if f["replan"] else 0.0
        x = np.array([1.0, near, stalled, np.log1p(max(0, f["current_cost"])) / 10])
        return np.clip(x, -1, 1), np.stack([normal, tangent, plan_direction]), clearance

    def forward(self, features, horizon, basis):
        x, basis = np.asarray(features, dtype=np.float64), np.asarray(basis, dtype=np.float64)
        if (
            x.shape != (4,)
            or basis.shape != (3, 2)
            or not np.isfinite(x).all()
            or not np.isfinite(basis).all()
        ):
            raise ValueError("Malformed structured controller inputs")
        if not 1 <= horizon <= self.spec.max_horizon:
            raise ValueError("Proposal horizon exceeds the frozen representation")
        gains = self.parameters.reshape(2, 3, 4) @ x
        directions = gains @ basis
        count = self.spec.max_horizon * self.spec.action_block
        temporal = np.column_stack([np.ones(count), np.linspace(1, -1, count)])
        native = self.spec.native_residual_bound * np.tanh(temporal @ directions)
        residual = native / self.action_scale
        return residual.reshape(self.spec.max_horizon, self.spec.block_action_dim)[:horizon]

    def propose(self, planner_features, current, target, checker, horizon):
        x, basis, clearance = self.features(planner_features, checker)
        values = self.forward(x, horizon, basis)
        self.history.append(
            {
                "replan": int(planner_features["replan"]),
                "features": x.tolist(),
                "basis": basis.tolist(),
                "clearance_px": clearance,
                "proposal": values.tolist(),
                "goal_information": "Frozen current/goal latent task cost and shifted plan only; no exact final-goal XY",
            }
        )
        return torch.as_tensor(values[None], dtype=current.dtype, device=current.device)

    def artifact(self):
        return {
            "format": FORMAT,
            "spec": asdict(self.spec),
            "feature_names": list(FEATURE_NAMES),
            "basis_names": list(BASIS_NAMES),
            "temporal_names": list(TEMPORAL_NAMES),
            "parameters": self.parameters.tolist(),
            "action_mean": self.action_mean.tolist(),
            "action_scale": self.action_scale.tolist(),
        }

    @classmethod
    def load(cls, path, case):
        value = json.loads(Path(path).read_text())
        if (
            value["format"] != FORMAT
            or value["feature_names"] != list(FEATURE_NAMES)
            or value["basis_names"] != list(BASIS_NAMES)
            or value["temporal_names"] != list(TEMPORAL_NAMES)
        ):
            raise ValueError("Controller feature contract differs")
        return cls(
            PushEvolutionSpec(**value["spec"]),
            value["parameters"],
            case,
            value["action_mean"],
            value["action_scale"],
        )


class PushPlanningCEM(PlanningCEM):
    """Supply only the upstream policy's existing warm-start plan to the controller."""

    def solve(self, info_dict, init_action=None):
        if self.proposal_adapter is not None:
            self.proposal_adapter.set_plan_context(init_action)
        return super().solve(info_dict, init_action)
