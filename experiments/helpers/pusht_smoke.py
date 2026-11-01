"""Simple virtual shapes and full-pusher clearances for the October smoke test."""

import numpy as np
import torch

from helpers.safecem10_planner import pd_response


class Scaler:
    """Public training-set standardization without importing other simulators."""

    def __init__(self, mean, scale):
        self.mean_ = np.asarray(mean, dtype=np.float64)
        self.scale_ = np.asarray(scale, dtype=np.float64)

    def transform(self, value):
        return (np.asarray(value) - self.mean_) / self.scale_

    def inverse_transform(self, value):
        return np.asarray(value) * self.scale_ + self.mean_


def signed_distance(points, shape):
    """Signed Euclidean distance to the physical region, positive outside."""
    if shape["kind"] == "none":
        # Finite sentinel for saved JSON; no physical hazard exists in this control.
        return torch.full_like(points[..., 0], 512.0)
    if shape["kind"] == "island":
        return torch.linalg.vector_norm(points, dim=-1) - shape["radius_px"]
    if shape["kind"] == "bar":
        x = points[..., 0].clamp(-shape["segment_half_length_px"], shape["segment_half_length_px"])
        nearest = torch.stack([x, torch.zeros_like(x)], dim=-1)
        return torch.linalg.vector_norm(points - nearest, dim=-1) - shape["radius_px"]
    if shape["kind"] != "gate":
        raise ValueError("Unknown smoke-test shape")
    offset = shape["gap_offset_px"]
    half = shape["gap_width_px"] / 2
    extent = shape["extent_px"]
    band = shape["band_half_width_px"]
    distances = []
    for lo, hi in [(-extent, offset - half), (offset + half, extent)]:
        centre = points.new_tensor([(lo + hi) / 2, 0])
        radius = points.new_tensor([(hi - lo) / 2, band])
        q = (points - centre).abs() - radius
        distances.append(q.clamp_min(0).norm(dim=-1) + q.max(dim=-1).values.clamp_max(0))
    return torch.minimum(*distances)


def physical_clearance(path, case, pusher_radius=15):
    origin = torch.as_tensor(case["origin"], device=path.device, dtype=path.dtype)
    basis = torch.as_tensor(case["basis"], device=path.device, dtype=path.dtype)
    local = (path - origin) @ basis.T
    return signed_distance(local, case) - pusher_radius


def arena_clearance(path, pusher_radius=15):
    return torch.minimum(path, 512 - path).min(dim=-1).values - pusher_radius


class ShapeChecker:
    """Known PushT pusher dynamics; no MuJoCo or Cube dependency."""

    kind = "pusht"

    def __init__(self, case, margin, scaler, counters, enabled=True):
        self.case, self.margin, self.scaler = case, margin, scaler
        self.counters, self.enabled = counters, enabled
        self.state = None
        self.cache = {}

    def paths(self, candidates):
        raw = candidates.double().reshape(*candidates.shape[:2], -1, 2)
        raw = raw * raw.new_tensor(self.scaler.scale_) + raw.new_tensor(self.scaler.mean_)
        b, n, length, _ = raw.shape
        self.counters["checker_candidate_rollouts"] += b * n
        if length not in self.cache:
            c, g = pd_response(length)
            self.cache[length] = (raw.new_tensor(c), raw.new_tensor(g))
        c, g = self.cache[length]
        start = raw.new_tensor(self.state[:2]).view(1, 1, 1, 2)
        velocity = raw.new_tensor(self.state[-2:]).view(1, 1, 1, 2)
        self.counters["analytic_path_steps"] += b * n * length * 10
        return start + c.view(1, 1, -1, 1) * velocity + torch.einsum("mn,bsnk->bsmk", g, raw)

    def check(self, candidates):
        if not self.enabled:
            b, n = candidates.shape[:2]
            zero = candidates.new_zeros(b, n)
            return zero, candidates.new_zeros(b, n, 1, 2), zero
        path = self.paths(candidates)
        start = path.new_tensor(self.state[:2]).view(1, 1, 1, 2)
        start = start.expand(*path.shape[:2], 1, 2)
        path = torch.cat([start, path], dim=-2)
        clearance = physical_clearance(path, self.case)
        arena = arena_clearance(path)
        requested = torch.minimum(clearance, arena) - self.margin - 0.001
        violation = (-requested).clamp_min(0).sum(dim=-1)
        return violation, path, torch.zeros_like(violation)


def case_geometry(initial, goal, reference_path, protocol):
    start, end = np.asarray(initial)[:2], np.asarray(goal)[:2]
    forward = end - start
    forward /= np.linalg.norm(forward)
    across = np.array([forward[1], -forward[0]])
    path = np.asarray(reference_path)
    lengths = np.r_[0, np.linalg.norm(np.diff(path, axis=0), axis=1).cumsum()]
    order = np.argsort(np.abs(lengths - lengths[-1] / 2), kind="stable")
    candidates = [path[i] for i in order] + [(start + end) / 2]
    margin = protocol["margin_px"]
    for centre in candidates:
        if not ((centre >= 70) & (centre <= 442)).all():
            continue
        if min(np.linalg.norm(centre - start), np.linalg.norm(centre - end)) < 50:
            continue

        def room(direction):
            distances = []
            for axis in range(2):
                if abs(direction[axis]) > 1e-8:
                    edge = 491 if direction[axis] > 0 else 21
                    distances.append((edge - centre[axis]) / direction[axis])
            return min(distances)

        horizontal = across if room(across) >= room(-across) else -across
        basis = np.stack([horizontal, forward])
        cases = {
            kind: {
                "kind": kind,
                "origin": centre.tolist(),
                "basis": basis.tolist(),
                **params,
            }
            for kind, params in protocol["layouts"].items()
        }
        endpoints = torch.tensor(np.stack([start, end]), dtype=torch.float64)
        if all(bool((physical_clearance(endpoints, c) > margin).all()) for c in cases.values()):
            return cases
    raise RuntimeError("No endpoint-clear origin under the frozen layout rule")
