"""Compact causal action proposals for a frozen PushT planning stack.

Weights are a persistent genotype. Proposals are candidates, never unchecked actions.
All numerical execution and tests belong on the declared rented host.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


FEATURE_NAMES = (
    "pusher_x",
    "pusher_y",
    "pusher_vx",
    "pusher_vy",
    "task_cost",
    "improvement",
    "prediction_error",
    "previous_feasible",
    "elapsed",
    "hazard_dx",
    "hazard_dy",
    "is_island",
    "is_bar",
    "is_gate",
    "axis_x",
    "axis_y",
    "radius",
    "half_length",
    "gap_width",
    "gap_offset",
    "band_half_width",
    *(f"latent_goal_delta_{i}" for i in range(8)),
)


@dataclass(frozen=True)
class ProposalSpec:
    hidden_size: int = 8
    max_horizon: int = 3
    block_action_dim: int = 10
    normalized_action_bound: float = 1.0
    projection_seed: int = 303032

    def __post_init__(self):
        if not 1 <= self.hidden_size <= 64 or not 1 <= self.max_horizon <= 5:
            raise ValueError("Unsupported proposal dimensions")
        if self.block_action_dim != 10:
            raise ValueError(
                "This proposal representation is for five two-dimensional PushT actions"
            )
        if not np.isfinite(self.normalized_action_bound) or self.normalized_action_bound <= 0:
            raise ValueError("Declare a finite positive normalized proposal bound")

    @property
    def parameter_count(self):
        outputs = self.max_horizon * self.block_action_dim
        return self.hidden_size * (len(FEATURE_NAMES) + 1) + outputs * (self.hidden_size + 1)

    def initial_parameters(self, seed):
        rng = np.random.default_rng(seed)
        h, o = self.hidden_size, self.max_horizon * self.block_action_dim
        return np.concatenate(
            [
                rng.normal(0, 0.2, h * len(FEATURE_NAMES)),
                np.zeros(h),
                rng.normal(0, 0.05, o * h),
                rng.normal(0, 0.03, o),
            ]
        )

    def unpack(self, parameters):
        theta = np.asarray(parameters, dtype=np.float64)
        if theta.shape != (self.parameter_count,) or not np.isfinite(theta).all():
            raise ValueError("Wrong proposal parameter count or nonfinite weights")
        h, o = self.hidden_size, self.max_horizon * self.block_action_dim
        a, b, c = h * len(FEATURE_NAMES), h * (len(FEATURE_NAMES) + 1), h * o
        return theta[:a].reshape(h, -1), theta[a:b], theta[b : b + c].reshape(o, h), theta[b + c :]


class ProposalAdapter:
    def __init__(self, spec, parameters, case):
        self.spec, self.parameters, self.case = spec, np.asarray(parameters).copy(), case
        self.weights = spec.unpack(parameters)
        if case["kind"] not in ("island", "bar", "gate"):
            raise ValueError("Explicit supported virtual hazard required")
        self.history = []

    def features(self, planner_features, current, target, checker):
        if not hasattr(checker, "observed") or checker.observed is None:
            raise ValueError("Native causal proprioception is required")
        proprio = checker.observed.detach().cpu().double().numpy().reshape(-1)
        if proprio.shape != (4,):
            raise ValueError("Expected native pusher position and velocity")
        delta = (target[0, -1] - current[0, -1]).detach().cpu().double().numpy().reshape(-1)
        if not len(delta):
            raise ValueError("Missing frozen goal representation")
        # Fixed random features; no labels, simulator previews or trainable projection.
        projection = np.random.default_rng(self.spec.projection_seed).normal(
            0, 1 / np.sqrt(len(delta)), (8, len(delta))
        )
        shape, f = self.case, planner_features
        relative = (np.asarray(shape["origin"]) - proprio[:2]) / 512
        axis = np.asarray(shape["basis"])[0]
        value = np.concatenate(
            [
                proprio / [512, 512, 512, 512],
                [
                    np.log1p(max(0, f["current_cost"])) / 10,
                    f["improvement"],
                    f["prediction_error"] / (1 + abs(f["prediction_error"])),
                    f["previous_feasible"],
                    f["replan"] / 30,
                ],
                relative,
                [shape["kind"] == k for k in ("island", "bar", "gate")],
                axis,
                np.asarray(
                    [
                        shape.get(k, 0)
                        for k in (
                            "radius_px",
                            "segment_half_length_px",
                            "gap_width_px",
                            "gap_offset_px",
                            "band_half_width_px",
                        )
                    ]
                )
                / 512,
                np.tanh(projection @ delta),
            ]
        )
        if value.shape != (len(FEATURE_NAMES),) or not np.isfinite(value).all():
            raise ValueError("Malformed or nonfinite proposal features")
        return np.clip(value, -1, 1), len(delta)

    def forward(self, features, horizon):
        if not 1 <= horizon <= self.spec.max_horizon:
            raise ValueError("Proposal horizon exceeds the declared representation")
        x = np.asarray(features, dtype=np.float64)
        if x.shape != (len(FEATURE_NAMES),) or not np.isfinite(x).all():
            raise ValueError("Malformed proposal feature vector")
        w, bias, output, output_bias = self.weights
        values = self.spec.normalized_action_bound * np.tanh(
            output @ np.tanh(w @ x + bias) + output_bias
        )
        return values.reshape(self.spec.max_horizon, self.spec.block_action_dim)[:horizon]

    def propose(self, features, current, target, checker, horizon):
        import torch

        x, latent_dim = self.features(features, current, target, checker)
        values = self.forward(x, horizon)
        self.history.append(
            {
                "replan": int(features["replan"]),
                "features": x.tolist(),
                "latent_dim": latent_dim,
                "proposal": values.tolist(),
            }
        )
        return torch.as_tensor(values[None], dtype=current.dtype, device=current.device)

    def artifact(self):
        return {
            "format": "lewm-behavioural-proposal-v1",
            "spec": asdict(self.spec),
            "feature_names": list(FEATURE_NAMES),
            "parameters": self.parameters.tolist(),
        }

    @classmethod
    def load(cls, path, case):
        value = json.loads(Path(path).read_text())
        if value["format"] != "lewm-behavioural-proposal-v1":
            raise ValueError("Unsupported proposal artifact")
        if value["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("Proposal feature contract differs")
        return cls(ProposalSpec(**value["spec"]), value["parameters"], case)
