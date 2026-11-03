"""Generic recurrent action modulation for frozen policies and planners.

This module knows nothing about navigation geometry, target retrieval or rewards.
Environment wrappers provide normalized observations and legal normalized base actions.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class AdapterSpec:
    input_dim: int
    action_dim: int
    hidden_dim: int = 32
    residual_bound: float = 1.0
    initialization_seed: int = 8301

    def __post_init__(self):
        if min(self.input_dim, self.action_dim, self.hidden_dim) <= 0:
            raise ValueError("Adapter dimensions must be positive")
        if not 0 < self.residual_bound <= 2:
            raise ValueError("Residual bound must be in (0, 2] normalized action units")


class RecurrentActionAdapter(nn.Module):
    """A learned feature layer, GRU memory, and scale/residual output heads.

    Neutral heads exactly recover the base policy; this is an engineering reference,
    not a handcrafted successful safety policy. Every parameter is available to learning.
    """

    def __init__(self, spec: AdapterSpec):
        super().__init__()
        self.spec = spec
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(spec.initialization_seed)
            self.features = nn.Linear(spec.input_dim, spec.hidden_dim)
            self.memory = nn.GRUCell(spec.hidden_dim, spec.hidden_dim)
            self.head = nn.Linear(spec.hidden_dim, 2 * spec.action_dim)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        self.register_buffer("correction_bound", torch.tensor(float(spec.residual_bound)))

    def initial_state(self, batch_size=1):
        parameter = next(self.parameters())
        return parameter.new_zeros(batch_size, self.spec.hidden_dim)

    def forward(self, observations, state, base_action):
        if observations.shape[-1] != self.spec.input_dim:
            raise ValueError("Observation dimension differs from the frozen adapter spec")
        if base_action.shape[-1] != self.spec.action_dim:
            raise ValueError("Base-action dimension differs from the frozen adapter spec")
        encoded = torch.tanh(self.features(observations))
        state = self.memory(encoded, state)
        scale_logits, residual_logits = self.head(state).chunk(2, dim=-1)
        gain = 1 + torch.tanh(scale_logits)
        residual = self.correction_bound * torch.tanh(residual_logits)
        proposed = gain * base_action + residual
        executed = proposed.clamp(-1, 1)
        return executed, state, dict(gain=gain, residual=residual, proposed=proposed)

    def specification(self):
        return {**asdict(self.spec), "parameter_count": sum(p.numel() for p in self.parameters())}

    def parameters_vector(self):
        return torch.cat([p.detach().reshape(-1).cpu() for p in self.parameters()]).numpy().copy()

    def coordinate_scales(self):
        """Fan-in scaling for standardized evolutionary search coordinates."""
        values = []
        for name, parameter in self.named_parameters():
            if parameter.ndim >= 2:
                fan_in = parameter.shape[1]
            elif name.startswith("features."):
                fan_in = self.spec.input_dim
            else:
                fan_in = self.spec.hidden_dim
            values.extend([1 / math.sqrt(fan_in)] * parameter.numel())
        return np.asarray(values, dtype=np.float32)

    @torch.no_grad()
    def load_vector(self, vector):
        vector = np.asarray(vector, dtype=np.float32)
        expected = sum(p.numel() for p in self.parameters())
        if vector.shape != (expected,) or not np.isfinite(vector).all():
            raise ValueError("Expected one finite value per adapter parameter")
        offset = 0
        for parameter in self.parameters():
            size = parameter.numel()
            parameter.copy_(
                torch.as_tensor(vector[offset : offset + size], device=parameter.device).reshape(
                    parameter.shape
                )
            )
            offset += size


def standardized_parameters(neutral, scales, genotype):
    genotype = np.asarray(genotype, dtype=np.float32)
    if genotype.shape != neutral.shape or scales.shape != neutral.shape:
        raise ValueError("Genotype and coordinate maps must have identical dimensions")
    if not np.isfinite(genotype).all():
        raise ValueError("Nonfinite genotype")
    return neutral + scales * genotype
