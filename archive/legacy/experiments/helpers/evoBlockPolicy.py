"""Sequential 60-output linear policy, the Gate 0 response to the held-action failure.

Reuses evoPolicy's frozen [z_t, z_t-z_prev] features and persistence. Parameters are
W (60, 384), then b (60,), in time-major motor order: 23,100 in total. Each of the
10 rows is a separate 125 Hz action; the policy observes a new frame every block.
The old held-action policy and all its experiments are unchanged.
"""
from __future__ import annotations

import numpy as np
import torch

from helpers.evoPolicy import ACTION_DIM, FEATURE_DIM, LinearLatentPolicy
from helpers.walkerRules import FRAMESKIP

BLOCK_DIM = FRAMESKIP * ACTION_DIM
N_PARAMS = BLOCK_DIM * (FEATURE_DIM + 1)


class LinearBlockPolicy(LinearLatentPolicy):
    n_params = N_PARAMS

    @staticmethod
    def unpack(theta):
        if theta.shape[-1] != N_PARAMS:
            raise ValueError(f"expected {N_PARAMS} parameters, got {theta.shape[-1]}")
        cut = BLOCK_DIM * FEATURE_DIM
        return (theta[..., :cut].reshape(*theta.shape[:-1], BLOCK_DIM, FEATURE_DIM),
                theta[..., cut:])

    @staticmethod
    def pack(w, b):
        if w.shape[-2:] != (BLOCK_DIM, FEATURE_DIM) or b.shape != (*w.shape[:-2], BLOCK_DIM):
            raise ValueError("expected W (..., 60, 384) and b (..., 60)")
        flat = w.reshape(*w.shape[:-2], BLOCK_DIM * FEATURE_DIM)
        return torch.cat([flat, b], -1) if isinstance(w, torch.Tensor) else np.concatenate([flat, b], -1)

    def act(self, theta, feats: torch.Tensor) -> torch.Tensor:
        """(P, 23100), (P, ..., 384) -> (P, ..., 60), clipped, time-major."""
        theta = torch.as_tensor(theta, dtype=torch.float32, device=feats.device)
        if theta.ndim != 2 or feats.shape[0] != theta.shape[0] or feats.shape[-1] != FEATURE_DIM:
            raise ValueError("theta must be (P, n_params), feats (P, ..., 384)")
        w, b = self.unpack(theta)
        flat = feats.reshape(feats.shape[0], -1, FEATURE_DIM)
        out = torch.einsum("pof,pmf->pmo", w, flat) + b[:, None, :]
        return out.reshape(*feats.shape[:-1], BLOCK_DIM).clamp(-1.0, 1.0)

    def hold(self, actions):
        raise TypeError("LinearBlockPolicy outputs sequential blocks; do not hold an action")

    def raw_block(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.shape[-1] != BLOCK_DIM:
            raise ValueError("expected a 60-dimensional time-major action block")
        return actions.clamp(-1.0, 1.0).reshape(*actions.shape[:-1], FRAMESKIP, ACTION_DIM)

    def model_block(self, actions: torch.Tensor) -> torch.Tensor:
        blocks = self.raw_block(actions.float())
        return ((blocks - self.action_mean) / self.action_std).reshape(*actions.shape[:-1], BLOCK_DIM)

    def state(self) -> dict:
        return {**super().state(), "policy_kind": "linear_sequential_block_v1"}

    @classmethod
    def from_state(cls, state: dict, *, device: str = "cpu"):
        if str(state.get("policy_kind", "")) != "linear_sequential_block_v1":
            raise ValueError("not a sequential-block policy state")
        return super().from_state(state, device=device)
