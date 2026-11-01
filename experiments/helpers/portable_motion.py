"""Learned native-proprioception transitions and environment-independent geometry.

Native simulator access is confined to collecting labels and executing real actions.
Candidate safety predictions use only this neural network and explicit geometry.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from helpers.safety_model import Constraint


class TransitionModel(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        d, a, p = spec["observation_dim"], spec["action_dim"], spec["substeps"]
        features = d + a + 2 * len(spec.get("angle_indices", []))
        self.net = nn.Sequential(
            nn.Linear(features, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, d + p * 2),
        )
        self.skip = nn.Linear(features, d + p * 2)
        self.register_buffer("xmean", torch.zeros(d + a))
        self.register_buffer("xscale", torch.ones(d + a))
        self.register_buffer("ymean", torch.zeros(d + p * 2))
        self.register_buffer("yscale", torch.ones(d + p * 2))
        self.calibration = None

    def transition(self, obs, action):
        x = (torch.cat([obs, action], -1) - self.xmean) / self.xscale
        angles = obs[..., self.spec.get("angle_indices", [])]
        x = torch.cat([x, angles.sin(), angles.cos()], -1)
        delta = (self.net(x) + self.skip(x)) * self.yscale + self.ymean
        d = self.spec["observation_dim"]
        path = delta[..., d:].reshape(*delta.shape[:-1], self.spec["substeps"], 2)
        path = path + obs[..., None, -2:]
        nxt = obs + delta[..., :d]
        nxt = torch.cat([nxt[..., :-2], path[..., -1, :]], -1)
        return nxt, path

    def forward(self, obs, actions):
        paths = []
        for action in actions.unbind(-2):
            obs, path = self.transition(obs, action)
            paths.append(path)
        return torch.cat(paths, -2)


def clearance(points, constraints, radius):
    if not constraints:
        return points.new_full(points.shape[:-1], 1e6)
    return torch.stack([c.clearance(points, radius) for c in constraints], -1).amin(-1)


class MotionChecker:
    kind = "portable_learned"
    mode = "learned"

    def __init__(self, model, constraints, margin, counters):
        self.model, self.constraints = model, constraints
        self.margin, self.counters = margin, counters
        if not model.calibration["ready"]:
            raise ValueError("Safety model did not pass its independent release gate")

    def observe(self, observed):
        self.observed = torch.as_tensor(observed, device="cuda", dtype=torch.float32)

    @torch.inference_mode()
    def check(self, candidates):
        b, n, h, _ = candidates.shape
        if b != 1 or h > 5:
            raise ValueError("Unsupported candidate dimensions")
        acts = candidates.reshape(n, h * 5, self.model.spec["action_dim"])
        low = acts.new_tensor(self.model.spec["normalized_action_low"])
        high = acts.new_tensor(self.model.spec["normalized_action_high"])
        acts = acts.clamp(low, high)
        initial = self.observed.expand(n, -1)
        path = self.model(initial, acts).reshape(1, n, -1, 2)
        path = torch.cat([initial[None, :, None, -2:], path], -2)
        lower = clearance(path, self.constraints, self.model.spec["body_radius"])
        lower = lower - self.model.calibration["allowance"]
        violation = (self.margin + 1e-6 - lower).clamp_min(0).sum(-1)
        self.counters["learned_candidate_checks"] += n
        self.counters["learned_transition_rows"] += n * h * 5
        if not torch.isfinite(violation).all():
            raise ValueError("Nonfinite learned prediction")
        return violation, path, candidates.new_zeros(b, n)


def load_model(path):
    data = torch.load(Path(path), map_location="cuda", weights_only=False)
    model = TransitionModel(data["spec"]).cuda()
    model.load_state_dict(data["state_dict"])
    model.calibration = data["calibration"]
    return model.eval().requires_grad_(False)


def shape_constraints(shape):
    if shape["kind"] == "none":
        return []
    c, axis = np.asarray(shape["centre"]), np.asarray(shape["axis"])
    if shape["kind"] == "island":
        return [Constraint("circle", tuple(c), radius=shape["radius"])]
    if shape["kind"] == "bar":
        return [
            Constraint("capsule", tuple(c), (shape["half_length"], 0), shape["radius"], tuple(axis))
        ]
    offset, gap, extent = shape["offset"], shape["gap"], shape["extent"]
    result = []
    for lo, hi in ((-extent, offset - gap / 2), (offset + gap / 2, extent)):
        centre = c + axis * ((lo + hi) / 2)
        result.append(
            Constraint("box", tuple(centre), ((hi - lo) / 2, shape["band"]), axis=tuple(axis))
        )
    return result
