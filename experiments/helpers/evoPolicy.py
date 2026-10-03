"""Linear latent policy for evolution inside the Walker LeWM (docs/evoPlan/README.md).

Features are `[z_t, z_t - z_(t-1)]` (2 x 192 = 384 numbers, the difference term supplies
velocity), standardised by a fixed affine map fitted once on behaviour-cloning frames and
frozen afterwards. A linear map gives Walker's six motor commands, clipped to [-1, 1] and
held for the 10 environment steps of a model block (12.5 Hz control). That is
6 x 384 + 6 = 2,310 parameters, laid out as W (6, 384) row-major followed by b (6,).

    policy = LinearLatentPolicy(scaler, feature_mean, feature_std, device="cuda")
    a = policy.act(theta, policy.features(z_t, z_prev))      # (P, ..., 6), clipped
    blocks = policy.model_block(a)                            # (P, ..., 60), LeWM action input

`model_block` normalises the held block with the LeWM action scaler in the same operation
order as `WalkerImaginer._flat`, so a held action and the equivalent recorded tape give the
model identical inputs.
"""

from __future__ import annotations

import numpy as np
import torch

from helpers.walkerRules import FRAMESKIP

LATENT_DIM = 192
ACTION_DIM = 6
FEATURE_DIM = 2 * LATENT_DIM
N_PARAMS = ACTION_DIM * FEATURE_DIM + ACTION_DIM


class LinearLatentPolicy:
    def __init__(self, action_scaler, feature_mean=None, feature_std=None, *, device: str = "cpu"):
        self.device = device
        self.action_mean = torch.as_tensor(np.asarray(action_scaler[0]), dtype=torch.float32, device=device)
        self.action_std = torch.as_tensor(np.asarray(action_scaler[1]), dtype=torch.float32, device=device)
        mean = np.zeros(FEATURE_DIM) if feature_mean is None else np.asarray(feature_mean, dtype=np.float64)
        std = np.ones(FEATURE_DIM) if feature_std is None else np.asarray(feature_std, dtype=np.float64)
        if mean.shape != (FEATURE_DIM,) or std.shape != (FEATURE_DIM,) or not (std > 0).all():
            raise ValueError("feature standardisation must be (384,) with positive scales")
        self.feature_mean = torch.as_tensor(mean, dtype=torch.float32, device=device)
        self.feature_std = torch.as_tensor(std, dtype=torch.float32, device=device)

    n_params = N_PARAMS

    # ---- parameters ----------------------------------------------------------------------
    @staticmethod
    def unpack(theta):
        """(..., 2310) -> W (..., 6, 384), b (..., 6). Works for numpy arrays and tensors."""
        if theta.shape[-1] != N_PARAMS:
            raise ValueError(f"expected {N_PARAMS} parameters, got {theta.shape[-1]}")
        w = theta[..., : ACTION_DIM * FEATURE_DIM].reshape(*theta.shape[:-1], ACTION_DIM, FEATURE_DIM)
        return w, theta[..., ACTION_DIM * FEATURE_DIM :]

    @staticmethod
    def pack(w, b):
        """Inverse of `unpack`."""
        lib = torch if isinstance(w, torch.Tensor) else np
        flat = w.reshape(*w.shape[:-2], ACTION_DIM * FEATURE_DIM)
        return lib.concatenate([flat, b], -1) if lib is np else torch.cat([flat, b], -1)

    # ---- features and actions ------------------------------------------------------------
    def features(self, z_t: torch.Tensor, z_prev: torch.Tensor) -> torch.Tensor:
        """(..., 192) latents of the current and previous block ends -> (..., 384) standardised."""
        x = torch.cat([z_t, z_t - z_prev], -1).float()
        return (x - self.feature_mean) / self.feature_std

    def act(self, theta, feats: torch.Tensor) -> torch.Tensor:
        """theta (P, 2310), feats (P, ..., 384) -> clipped actions (P, ..., 6).

        One einsum over the whole population and every root and noise sample.
        """
        theta = torch.as_tensor(theta, dtype=torch.float32, device=feats.device)
        if theta.ndim != 2 or feats.shape[0] != theta.shape[0]:
            raise ValueError("theta must be (P, n_params) and feats (P, ..., 384)")
        w, b = self.unpack(theta)
        lead = feats.shape[1:-1]
        flat = feats.reshape(feats.shape[0], -1, FEATURE_DIM)
        out = torch.einsum("pof,pmf->pmo", w, flat) + b[:, None, :]
        return out.reshape(feats.shape[0], *lead, ACTION_DIM).clamp(-1.0, 1.0)

    def hold(self, actions: torch.Tensor) -> torch.Tensor:
        """Zero-order hold: (..., 6) -> (..., FRAMESKIP, 6), clipped to the action box."""
        a = actions.clamp(-1.0, 1.0)
        return a.unsqueeze(-2).expand(*a.shape[:-1], FRAMESKIP, ACTION_DIM)

    def model_block(self, actions: torch.Tensor) -> torch.Tensor:
        """(..., 6) -> (..., 60): the held block normalised as `WalkerImaginer._flat` does."""
        blocks = self.hold(actions.float())
        blocks = (blocks - self.action_mean) / self.action_std
        return blocks.reshape(*blocks.shape[:-2], FRAMESKIP * ACTION_DIM)

    # ---- persistence -----------------------------------------------------------------------
    def state(self) -> dict:
        return {"action_mean": self.action_mean.cpu().numpy(), "action_std": self.action_std.cpu().numpy(),
                "feature_mean": self.feature_mean.cpu().numpy(), "feature_std": self.feature_std.cpu().numpy()}

    @classmethod
    def from_state(cls, state: dict, *, device: str = "cpu") -> "LinearLatentPolicy":
        return cls((state["action_mean"], state["action_std"]), state["feature_mean"], state["feature_std"], device=device)
