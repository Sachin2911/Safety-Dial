"""New campaign action contracts and observed-subgoal navigation interfaces.

This module never creates environments or collects observations. Numerical execution
and its tests belong on the newly declared rented hosts. Historical planners remain
unchanged; the cost wrapper projects their mutable population before fitting it.
"""

from dataclasses import dataclass

import numpy as np
import torch

from helpers.lewm18_planner import CostModel


@dataclass(frozen=True)
class ActionContract:
    """One native command domain shared by model prediction, checks and execution."""

    mean: tuple
    scale: tuple
    native_low: tuple
    native_high: tuple

    def __post_init__(self):
        arrays = [
            np.asarray(getattr(self, key), dtype=float)
            for key in ("mean", "scale", "native_low", "native_high")
        ]
        if any(a.ndim != 1 or a.shape != arrays[0].shape for a in arrays):
            raise ValueError("Action contract vectors must have the same one-dimensional shape")
        if not len(arrays[0]) or not all(np.isfinite(a).all() for a in arrays):
            raise ValueError("Action contract must be finite and nonempty")
        if (arrays[1] <= 0).any() or (arrays[2] >= arrays[3]).any():
            raise ValueError("Action scales and native bounds must be strictly ordered")
        for key, a in zip(("mean", "scale", "native_low", "native_high"), arrays, strict=True):
            object.__setattr__(self, key, tuple(float(x) for x in a))

    @property
    def dimension(self):
        return len(self.mean)

    @property
    def normalized_low(self):
        return (np.asarray(self.native_low) - self.mean) / self.scale

    @property
    def normalized_high(self):
        return (np.asarray(self.native_high) - self.mean) / self.scale

    def execute(self, raw):
        """The command passed to the native step; callers retain the proposed raw command."""
        value = np.asarray(raw)
        if value.shape[-1] != self.dimension or not np.isfinite(value).all():
            raise ValueError("Invalid native action")
        return np.clip(value, self.native_low, self.native_high)

    def record(self):
        return {
            "mean": list(self.mean),
            "scale": list(self.scale),
            "native_low": list(self.native_low),
            "native_high": list(self.native_high),
            "normalized_low": self.normalized_low.tolist(),
            "normalized_high": self.normalized_high.tolist(),
        }


class ProjectedCostModel(CostModel):
    """Project in place before model/checker scoring and the caller's CEM elite fit.

    A score-only copy would leave CEM fitting illegal coordinates to clipped scores.
    Mutating the supplied population also fixes its archive and mean selection. The
    original physical command is still clipped at the native boundary for float
    conversion roundoff. Projection does not add candidates or frozen-model queries.
    """

    def __init__(
        self, base, checker, params, counters, action_contract, diagnostic=False, *, project=True
    ):
        super().__init__(base, checker, params, counters, diagnostic=diagnostic)
        self.action_contract = action_contract
        self.project_commands = bool(project)
        self.projection_history = []
        spec = getattr(getattr(checker, "model", None), "spec", None)
        if isinstance(spec, dict) and "normalized_action_low" in spec:
            for key, expected in (
                ("normalized_action_low", action_contract.normalized_low),
                ("normalized_action_high", action_contract.normalized_high),
            ):
                if not np.allclose(spec[key], expected, rtol=0, atol=1e-6):
                    raise ValueError("Motion model and native normalized action bounds differ")

    @torch.no_grad()
    def project_(self, candidates):
        contract = self.action_contract
        if candidates.ndim != 4 or candidates.shape[-1] % contract.dimension:
            raise ValueError("Expected blocked action population (B,S,H,block*action_dim)")
        if not torch.isfinite(candidates).all():
            raise ValueError("Nonfinite candidate action")
        shaped = candidates.reshape(*candidates.shape[:-1], -1, contract.dimension)
        low = candidates.new_tensor(contract.normalized_low)
        high = candidates.new_tensor(contract.normalized_high)
        projected = torch.minimum(torch.maximum(shaped, low), high)
        changed = int((projected != shaped).sum())
        row = {
            "call": len(self.projection_history),
            "candidate_rollouts": int(candidates.shape[0] * candidates.shape[1]),
            "horizon": int(candidates.shape[2]),
            "raw_minimum": float(candidates.min()),
            "raw_maximum": float(candidates.max()),
            "out_of_domain_coordinates": changed,
            "projected_coordinates": changed if self.project_commands else 0,
        }
        if self.project_commands:
            candidates.copy_(projected.reshape_as(candidates))
        row.update(
            projected_minimum=float(candidates.min()), projected_maximum=float(candidates.max())
        )
        self.projection_history.append(row)
        self.counters["candidate_bound_observation_calls"] += 1
        self.counters["action_projection_calls"] += int(self.project_commands)
        self.counters["out_of_domain_candidate_coordinates"] += changed
        self.counters["projected_candidate_coordinates"] += changed if self.project_commands else 0
        return candidates

    def get_cost(self, info, candidates):
        return super().get_cost(info, self.project_(candidates))
