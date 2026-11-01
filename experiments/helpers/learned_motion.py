"""Supervised dense-motion prediction plus reusable constraint geometry.

No controller equations or simulator previews enter inference. The observed input is
native proprioception (position and velocity), explicitly different from the earlier
image/latent-only prototype. The frozen world model still supplies task progress.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

FORMAT = "safetydial-observed-motion-v1"


@dataclass(frozen=True)
class MotionSpec:
    environment: str = "pusht"
    action_dim: int = 2
    observation_dim: int = 4
    action_block: int = 5
    max_horizon: int = 3
    substeps: int = 10
    length_scale: float = 512.0
    units: str = "px"
    body_radius: float = 15.0
    protected_body: str = "pusher_disk"
    coordinate_frame: str = "arena_xy"
    world_model_sha256: str = ""
    action_scaler_sha256: str = ""

    def __post_init__(self):
        if (self.environment, self.action_dim, self.observation_dim) != ("pusht", 2, 4):
            raise ValueError("Only PushT observed-motion integration is implemented")
        if min(self.action_block, self.max_horizon, self.substeps, self.length_scale) <= 0:
            raise ValueError("Motion dimensions must be positive")

    @property
    def actions(self):
        return self.action_block * self.max_horizon


class DenseMotionModel(nn.Module):
    """Causal affine motion head fitted from simulator-observed trajectories.

    A deliberately simple first supervised model. Causal masks use time indexing,
    without assuming any physical controller gains or motion equations.
    """

    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        inputs = spec.observation_dim + spec.actions * spec.action_dim
        outputs = spec.actions * spec.substeps * 2
        self.linear = nn.Linear(inputs, outputs, dtype=torch.float64)
        self.register_buffer("error_allowance", torch.tensor(float("inf"), dtype=torch.float64))
        self.calibration = None

    def features(self, observed, candidates):
        b, n, h, d = candidates.shape
        if observed.shape != (b, self.spec.observation_dim):
            raise ValueError("Supply native pusher position/velocity, no object or goal state")
        if (
            not 1 <= h <= self.spec.max_horizon
            or d != self.spec.action_dim * self.spec.action_block
        ):
            raise ValueError("Candidate horizon/action dimensions differ from checkpoint")
        observed = observed.to(self.linear.weight.device, torch.float64)
        actions = candidates.to(self.linear.weight.device, torch.float64).flatten(-2)
        padded = actions.new_zeros(b, n, self.spec.actions * self.spec.action_dim)
        padded[..., : actions.shape[-1]] = actions
        value = torch.cat(
            [
                (observed / self.spec.length_scale)[:, None].expand(b, n, -1),
                padded,
            ],
            -1,
        )
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite observed state or candidate actions")
        return value

    def forward(self, observed, candidates):
        features = self.features(observed, candidates)
        b, n, h = candidates.shape[:3]
        path = self.linear(features).reshape(b, n, self.spec.actions * self.spec.substeps, 2)
        return (
            path[..., : h * self.spec.action_block * self.spec.substeps, :] * self.spec.length_scale
        )


def fit_motion(spec, observations, actions, positions):
    """Ordinary least squares, independently masked at every action's time interval."""
    model = DenseMotionModel(spec)
    observed = torch.as_tensor(observations, dtype=torch.float64)
    candidates = torch.as_tensor(actions, dtype=torch.float64).reshape(
        -1, 1, spec.max_horizon, spec.action_block * 2
    )
    x = model.features(observed, candidates)[:, 0]
    y = torch.as_tensor(positions, dtype=torch.float64) / spec.length_scale
    if y.shape != (len(x), spec.actions * spec.substeps, 2):
        raise ValueError("Training targets must contain every future physics position")
    ranks = []
    with torch.no_grad():
        model.linear.weight.zero_()
        model.linear.bias.zero_()
        for action_index in range(spec.actions):
            count = spec.observation_dim + (action_index + 1) * spec.action_dim
            design = torch.cat([x[:, :count], x.new_ones(len(x), 1)], -1)
            lo, hi = action_index * spec.substeps, (action_index + 1) * spec.substeps
            target = y[:, lo:hi].flatten(1)
            solution = torch.linalg.lstsq(design, target, driver="gelsd")
            if int(solution.rank) != design.shape[1]:
                raise ValueError("Training motion excitation is rank-deficient")
            ranks.append(int(solution.rank))
            rows = slice(lo * 2, hi * 2)
            model.linear.weight[rows, :count] = solution.solution[:-1].T
            model.linear.bias[rows] = solution.solution[-1]
    return model.eval().requires_grad_(False), ranks


@torch.inference_mode()
def motion_errors(model, observations, actions, positions):
    spec = model.spec
    observed = torch.as_tensor(observations, dtype=torch.float64)
    candidates = torch.as_tensor(actions, dtype=torch.float64).reshape(
        -1, 1, spec.max_horizon, spec.action_block * 2
    )
    prediction = model(observed, candidates)[:, 0]
    target = torch.as_tensor(positions, dtype=torch.float64)
    errors = (prediction - target).norm(dim=-1)
    return {
        "trajectories": len(errors),
        "physics_points": errors.numel(),
        "mean_position_error_px": float(errors.mean()),
        "max_position_error_px": float(errors.max()),
    }, errors.amax(-1).numpy()


def calibrate_motion(model, trajectory_errors, groups, alpha=0.1):
    groups = np.asarray(groups)
    unique = np.unique(groups)
    errors = np.asarray(trajectory_errors)
    if not 0 < alpha < 1 or not len(unique) or len(errors) != len(groups):
        raise ValueError("Invalid independent motion calibration groups")
    if not np.isfinite(errors).all() or (errors < 0).any():
        raise ValueError("Calibration errors must be finite and nonnegative")
    maxima = np.array([errors[groups == group].max() for group in unique])
    rank = math.ceil((len(unique) + 1) * (1 - alpha))
    allowance = float(np.sort(maxima)[rank - 1]) if rank <= len(unique) else None
    if allowance is not None:
        # Retain a numerical floor; this does not reduce the requested safety margin.
        allowance = max(0.001, allowance)
    model.error_allowance.fill_(float("inf") if allowance is None else allowance)
    model.calibration = {
        "ready": allowance is not None,
        "source_groups": len(unique),
        "alpha": alpha,
        "allowance": allowance,
        "unit": "px",
        "order_statistic": rank,
        "scope": "Source-group worst dense position error; not a universal safety guarantee",
    }
    return model.calibration


def save_motion(path, model, provenance):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    torch.save(
        {
            "format": FORMAT,
            "spec": asdict(model.spec),
            "state_dict": model.state_dict(),
            "calibration": model.calibration,
            "provenance": provenance,
        },
        path,
    )


def load_motion(path, device="cpu"):
    data = torch.load(path, map_location=device, weights_only=True)
    if data["format"] != FORMAT:
        raise ValueError("Unsupported motion checkpoint")
    model = DenseMotionModel(MotionSpec(**data["spec"])).to(device)
    model.load_state_dict(data["state_dict"], strict=True)
    model.calibration = data["calibration"]
    return model.eval().requires_grad_(False), data["provenance"]


class LearnedMotionChecker:
    """Learned motion plus generic geometry. No privileged reference or simulator handle."""

    kind = "pusht"
    mode = "learned"

    def __init__(self, model, constraints, margin, counters):
        if not model.calibration or not model.calibration["ready"]:
            raise ValueError("Independent calibration is required")
        if not torch.isfinite(model.error_allowance):
            raise ValueError("Motion allowance must be finite")
        if not constraints or not math.isfinite(margin) or margin < 0:
            raise ValueError("Explicit constraints and a finite nonnegative margin are required")
        self.model, self.constraints, self.margin, self.counters = (
            model,
            constraints,
            margin,
            counters,
        )
        self.observed = None

    def observe(self, proprio):
        value = torch.as_tensor(np.asarray(proprio).copy(), dtype=torch.float64)
        if value.shape != (4,) or not torch.isfinite(value).all():
            raise ValueError("Expected native observed (pusher x, y, vx, vy)")
        self.observed = value[None]

    @torch.inference_mode()
    def check(self, candidates):
        if self.observed is None:
            raise RuntimeError("Native observation is missing")
        path = self.model(self.observed, candidates)
        b, n = candidates.shape[:2]
        start = self.observed[:, None, None, :2].to(path.device).expand(b, n, 1, 2)
        path = torch.cat([start, path], -2)
        clearance = torch.stack(
            [
                constraint.clearance(path, self.model.spec.body_radius)
                for constraint in self.constraints
            ],
            -1,
        ).amin(-1)
        conservative = clearance - self.model.error_allowance
        violation = (self.margin + 0.001 - conservative).clamp_min(0).sum(-1)
        self.counters["learned_candidate_checks"] += b * n
        self.counters["learned_motion_physics_points"] += b * n * (path.shape[-2] - 1)
        return violation.to(candidates.device), path, candidates.new_zeros(b, n)
