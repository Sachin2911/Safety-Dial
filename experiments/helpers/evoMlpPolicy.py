"""Small MLP over the same frozen latent features, with sequential block outputs.

This is an imitation diagnostic, not an evolutionary search. The stateless parameter
interface lets the existing imagination harness and task-local real executor use the
same controller. ReLU hidden layers and an unclipped linear training head are followed
by action clipping at inference, matching the ridge policy's action contract.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from helpers.evoBlockPolicy import BLOCK_DIM, FEATURE_DIM, LinearBlockPolicy


class MLPBlockPolicy(LinearBlockPolicy):
    def __init__(self, action_scaler, feature_mean=None, feature_std=None, *,
                 hidden=(256, 128), device="cpu"):
        super().__init__(action_scaler, feature_mean, feature_std, device=device)
        self.hidden = tuple(int(n) for n in hidden)
        if not self.hidden or any(n <= 0 for n in self.hidden):
            raise ValueError("hidden widths must be positive")
        widths = (FEATURE_DIM, *self.hidden, BLOCK_DIM)
        self.layer_shapes = tuple(zip(widths[1:], widths[:-1]))
        self.n_params = sum(o * i + o for o, i in self.layer_shapes)

    def unpack(self, theta):
        if theta.shape[-1] != self.n_params:
            raise ValueError(f"expected {self.n_params} parameters, got {theta.shape[-1]}")
        layers, cut = [], 0
        for out_dim, in_dim in self.layer_shapes:
            w = theta[..., cut:cut + out_dim * in_dim].reshape(*theta.shape[:-1], out_dim, in_dim)
            cut += out_dim * in_dim
            b = theta[..., cut:cut + out_dim]
            cut += out_dim
            layers.append((w, b))
        return layers

    def pack(self, layers):
        if len(layers) != len(self.layer_shapes):
            raise ValueError("wrong number of MLP layers")
        pieces = []
        prefix = layers[0][0].shape[:-2]
        for (w, b), (o, i) in zip(layers, self.layer_shapes, strict=True):
            if w.shape != (*prefix, o, i) or b.shape != (*prefix, o):
                raise ValueError("incompatible MLP layer shape")
            pieces.extend([w.reshape(*prefix, o * i), b])
        return torch.cat(pieces, -1) if isinstance(pieces[0], torch.Tensor) else np.concatenate(pieces, -1)

    def act(self, theta, feats):
        theta = torch.as_tensor(theta, dtype=torch.float32, device=feats.device)
        if theta.ndim != 2 or feats.shape[0] != theta.shape[0] or feats.shape[-1] != FEATURE_DIM:
            raise ValueError("theta must be (P, n_params), feats (P, ..., 384)")
        x = feats.reshape(feats.shape[0], -1, FEATURE_DIM)
        for j, (w, b) in enumerate(self.unpack(theta)):
            x = torch.einsum("poi,pmi->pmo", w, x) + b[:, None, :]
            if j < len(self.layer_shapes) - 1:
                x = x.relu()
        return x.reshape(*feats.shape[:-1], BLOCK_DIM).clamp(-1, 1)

    def network(self):
        """Equivalent trainable module, on the policy device, without final clipping."""
        layers = []
        for j, (o, i) in enumerate(self.layer_shapes):
            layers.append(nn.Linear(i, o))
            if j < len(self.layer_shapes) - 1:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers).to(self.device)

    def network_theta(self, net):
        layers = [(layer.weight.detach(), layer.bias.detach()) for layer in net if isinstance(layer, nn.Linear)]
        return self.pack(layers).cpu().numpy()

    def state(self):
        return {**super().state(), "policy_kind": "mlp_sequential_block_v1",
                "hidden": np.asarray(self.hidden, dtype=np.int64)}

    @classmethod
    def from_state(cls, state, *, device="cpu"):
        if str(state.get("policy_kind", "")) != "mlp_sequential_block_v1":
            raise ValueError("not an MLP sequential-block policy state")
        return cls((state["action_mean"], state["action_std"]), state["feature_mean"],
                   state["feature_std"], hidden=state["hidden"], device=device)
