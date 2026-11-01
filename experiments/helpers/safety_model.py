"""Hazard-conditioned chunk-clearance prediction, independent of simulator dynamics.

Inputs contain frozen observation/imagination features, normalized candidate actions and
an explicit geometric constraint. Targets are minimum physical clearance over a chunk,
including its starting point. Ground-truth states are labels, never model inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

FORMAT = "safetydial-clearance-v1"
KINDS = ("circle", "box", "capsule", "workspace")
GEOMETRY_DIM = 12


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scaler_digest(mean, scale):
    value = {"mean": np.asarray(mean).tolist(), "scale": np.asarray(scale).tolist()}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class SafetySpec:
    environment: str
    latent_dim: int
    action_dim: int
    action_block: int
    max_horizon: int
    length_scale: float
    units: str
    protected_body: str
    body_radius: float
    coordinate_frame: str
    world_model_sha256: str
    action_scaler_sha256: str

    def __post_init__(self):
        if self.environment not in ("pusht", "tworoom", "reacher", "cube"):
            raise ValueError("Unknown environment")
        if min(self.latent_dim, self.action_dim, self.action_block, self.max_horizon) < 1:
            raise ValueError("Dimensions and horizons must be positive")
        if not math.isfinite(self.length_scale) or self.length_scale <= 0:
            raise ValueError("A positive physical length scale is required")
        if not math.isfinite(self.body_radius) or self.body_radius < 0:
            raise ValueError("Body radius must be finite and nonnegative")
        if not all((self.units, self.protected_body, self.coordinate_frame)):
            raise ValueError("Units, coordinate frame and protected body must be explicit")
        for digest in (self.world_model_sha256, self.action_scaler_sha256):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Model and action scaler must be identified by SHA-256")

    @property
    def input_dim(self):
        return (
            4 * self.latent_dim
            + self.max_horizon * self.action_block * self.action_dim
            + self.action_dim
            + 2
            + GEOMETRY_DIM
        )


@dataclass(frozen=True)
class Constraint:
    """A planar primitive in an explicit world frame; workspace means stay inside."""

    kind: str
    centre: tuple[float, float]
    half_size: tuple[float, float] = (0.0, 0.0)
    radius: float = 0.0
    axis: tuple[float, float] = (1.0, 0.0)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError("Unsupported constraint primitive")
        if not np.isfinite([*self.centre, *self.half_size, self.radius, *self.axis]).all():
            raise ValueError("Constraint parameters must be finite")
        if min(*self.half_size, self.radius) < 0:
            raise ValueError("Constraint dimensions must be nonnegative")
        if not np.isclose(np.linalg.norm(self.axis), 1, atol=1e-6):
            raise ValueError("Constraint axis must be a unit vector")
        if self.kind in ("box", "workspace") and min(self.half_size) <= 0:
            raise ValueError("Boxes require two positive half-sizes")

    def encode(self, spec):
        return torch.tensor(
            [float(self.kind == kind) for kind in KINDS]
            + [x / spec.length_scale for x in self.centre]
            + list(self.axis)
            + [x / spec.length_scale for x in self.half_size]
            + [self.radius / spec.length_scale, spec.body_radius / spec.length_scale],
            dtype=torch.float32,
        )

    def clearance(self, points, body_radius=0.0):
        """Physical signed clearance. Used only for labels and reference evaluation."""
        p = torch.as_tensor(points)
        centre, axis = p.new_tensor(self.centre), p.new_tensor(self.axis)
        delta = p - centre
        local = torch.stack(
            [(delta * axis).sum(-1), (delta * torch.stack([-axis[1], axis[0]])).sum(-1)],
            dim=-1,
        )
        if self.kind == "circle":
            distance = local.norm(dim=-1) - self.radius
        elif self.kind == "capsule":
            nearest = local[..., 0].clamp(-self.half_size[0], self.half_size[0])
            distance = torch.stack([local[..., 0] - nearest, local[..., 1]], -1).norm(dim=-1)
            distance = distance - self.radius
        else:
            q = local.abs() - p.new_tensor(self.half_size)
            distance = q.clamp_min(0).norm(dim=-1) + q.amax(-1).clamp_max(0)
            if self.kind == "workspace":
                distance = -distance
        return distance - body_radius


def pusht_constraints(case, include_arena=True):
    """Compile the smoke shapes into generic primitives, without predicting motion."""
    origin = np.asarray(case["origin"], dtype=float)
    basis = np.asarray(case["basis"], dtype=float)
    if basis.shape != (2, 2) or not np.allclose(basis @ basis.T, np.eye(2), atol=1e-6):
        raise ValueError("Hazard frame must be orthonormal")

    def box(lo, hi, band):
        centre = origin + np.array([(lo + hi) / 2, 0]) @ basis
        return Constraint("box", tuple(centre), ((hi - lo) / 2, band), axis=tuple(basis[0]))

    kind = case["kind"]
    if kind == "none":
        result = []
    elif kind == "island":
        result = [Constraint("circle", tuple(origin), radius=case["radius_px"])]
    elif kind == "bar":
        result = [
            Constraint(
                "capsule",
                tuple(origin),
                (case["segment_half_length_px"], 0),
                case["radius_px"],
                tuple(basis[0]),
            )
        ]
    elif kind == "gate":
        offset, half = case["gap_offset_px"], case["gap_width_px"] / 2
        extent, band = case["extent_px"], case["band_half_width_px"]
        result = [box(-extent, offset - half, band), box(offset + half, extent, band)]
    else:
        raise ValueError("Unknown PushT hazard")
    if include_arena:
        result.append(Constraint("workspace", (256, 256), (256, 256)))
    return result


def chunk_features(
    spec, observed, previous, predicted, actions, previous_action, history_valid, constraints
):
    """Return (B,S,H,C,F). Future actions after the scored chunk are masked out.

    `predicted` includes the current latent followed by H imagined block endpoints.
    Only two encoded observations and normalized actions enter; no true state or goal.
    """
    b, s, h, d = actions.shape
    if not 1 <= h <= spec.max_horizon or d != spec.action_block * spec.action_dim:
        raise ValueError("Candidate horizon/action dimensions differ from the checkpoint")
    if predicted.shape != (b, s, h + 1, spec.latent_dim):
        raise ValueError("Expected one current and H imagined embeddings")
    if observed.shape != previous.shape or observed.shape != (b, spec.latent_dim):
        raise ValueError("Two observed embeddings are required per batch row")
    if previous_action.shape != (b, spec.action_dim) or history_valid.shape != (b,):
        raise ValueError("Invalid previous-action or observation-history shape")
    if not constraints:
        raise ValueError("At least one explicit constraint is required")
    c = len(constraints)
    device, dtype = actions.device, actions.dtype

    def expand(x):
        return x[:, None, None].expand(b, s, h, -1)

    prefix = actions.new_zeros(b, s, h, spec.max_horizon, d)
    for t in range(h):
        prefix[:, :, t, : t + 1] = actions[:, :, : t + 1]
    depth = torch.arange(1, h + 1, device=device, dtype=dtype) / spec.max_horizon
    depth = depth.view(1, 1, h, 1).expand(b, s, h, 1)
    main = torch.cat(
        [
            expand(observed),
            expand(previous),
            predicted[:, :, :-1],
            predicted[:, :, 1:],
            prefix.flatten(-2),
            expand(previous_action),
            depth,
            expand(history_valid[:, None]),
        ],
        dim=-1,
    )
    geometry = torch.stack([item.encode(spec) for item in constraints]).to(device, dtype)
    geometry = geometry.view(1, 1, 1, c, -1).expand(b, s, h, c, -1)
    value = torch.cat([main.unsqueeze(-2).expand(b, s, h, c, -1), geometry], dim=-1)
    if not torch.isfinite(value).all():
        raise ValueError("Safety-model inputs contain nonfinite values")
    return value.detach()


class ClearanceModel(nn.Module):
    def __init__(self, spec, hidden=(128, 128)):
        super().__init__()
        self.spec, self.hidden = spec, tuple(hidden)
        if not self.hidden or min(self.hidden) < 1:
            raise ValueError("Hidden layer widths must be positive")
        layers, size = [], spec.input_dim
        for width in self.hidden:
            layers.extend([nn.Linear(size, width), nn.SiLU()])
            size = width
        layers.append(nn.Linear(size, 1))
        self.net = nn.Sequential(*layers)
        self.register_buffer("feature_mean", torch.zeros(spec.input_dim))
        self.register_buffer("feature_scale", torch.ones(spec.input_dim))
        self.register_buffer("error_allowance", torch.tensor(float("inf")))
        self.calibration = None

    def fit_normalization(self, features):
        self.feature_mean.copy_(features.mean(0))
        self.feature_scale.copy_(features.std(0, unbiased=False).clamp_min(1e-3))

    def forward(self, features):
        normalized = (features - self.feature_mean) / self.feature_scale
        return self.net(normalized).squeeze(-1)

    def physical_clearance(self, features):
        return self(features) * self.spec.length_scale


def calibrate(model, features, targets, groups, alpha):
    """One-sided held-out allowance from each source group's worst overprediction.

    This accounts for repeated labels within a source group. It does not establish
    coverage under planner-induced distribution shift or unrepresented hazards.
    An insufficient number of groups leaves the model unusable for learned execution.
    """
    if not 0 < alpha < 1:
        raise ValueError("Calibration alpha must lie between zero and one")
    with torch.inference_mode():
        residual = (model.physical_clearance(features) - targets).cpu().numpy()
    groups = np.asarray(groups)
    if len(residual) != len(groups) or not len(groups) or not np.isfinite(residual).all():
        raise ValueError("Invalid calibration examples")
    unique = np.unique(groups)
    maxima = np.array([residual[groups == group].max() for group in unique])
    rank = math.ceil((len(unique) + 1) * (1 - alpha))
    offset = max(0.0, float(np.sort(maxima)[rank - 1])) if rank <= len(unique) else None
    model.error_allowance.fill_(float("inf") if offset is None else offset)
    model.calibration = {
        "alpha": alpha,
        "source_groups": len(unique),
        "examples": len(groups),
        "order_statistic": rank,
        "allowance": offset,
        "unit": model.spec.units,
        "ready": offset is not None,
        "scope": "Empirical source-group calibration; planner distribution shift unverified",
    }
    return model.calibration


def prediction_metrics(model, features, targets, groups, margin):
    with torch.inference_mode():
        predicted = model.physical_clearance(features)
        conservative = predicted - model.error_allowance
    true_safe, accepted = targets >= margin, conservative >= margin
    error = predicted - targets
    return {
        "examples": len(targets),
        "source_groups": len(np.unique(groups)),
        "mae": float(error.abs().mean()),
        "max_overprediction": float(error.max()),
        "unsafe_examples": int((~true_safe).sum()),
        "accepted_examples": int(accepted.sum()),
        "false_safe": int((accepted & ~true_safe).sum()),
        "false_rejected": int((~accepted & true_safe).sum()),
        "margin": margin,
        "unit": model.spec.units,
    }


def save_model(path, model, provenance):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": FORMAT,
            "spec": asdict(model.spec),
            "hidden": list(model.hidden),
            "state_dict": model.state_dict(),
            "calibration": model.calibration,
            "provenance": provenance,
        },
        path,
    )


def load_model(path, expected_spec=None, device="cpu"):
    data = torch.load(path, map_location=device, weights_only=True)
    if data["format"] != FORMAT:
        raise ValueError("Unsupported safety checkpoint format")
    spec = SafetySpec(**data["spec"])
    if expected_spec is not None and spec != expected_spec:
        raise ValueError("Checkpoint model/scaler/body/frame identity does not match")
    model = ClearanceModel(spec, data["hidden"]).to(device)
    model.load_state_dict(data["state_dict"], strict=True)
    model.calibration = data["calibration"]
    return model.eval().requires_grad_(False), data["provenance"]
