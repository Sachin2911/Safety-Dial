"""Fixed-controller readiness screen: geometry, adapters and declared gates."""

from dataclasses import dataclass
import json
from types import SimpleNamespace

import numpy as np

from helpers.lewm_proposal_adapter import FEATURE_NAMES, ProposalAdapter, ProposalSpec
from helpers.lewm_residual import ResidualAdapter

KINDS = ("none", "island", "bar", "gate")


class ScreenPushTAdapter(ResidualAdapter):
    """Preserve the frozen PushT controller; define zero hazard features for controls."""

    def __init__(self, spec, parameters, case):
        placeholder = {
            "kind": "island",
            "origin": [0, 0],
            "basis": [[1, 0], [0, 1]],
            "radius_px": 0,
        }
        super().__init__(spec, parameters, placeholder if case["kind"] == "none" else case)
        self.case = case

    def features(self, planner_features, current, target, checker):
        if self.case["kind"] != "none":
            return super().features(planner_features, current, target, checker)
        placeholder = {
            "kind": "island",
            "origin": [0, 0],
            "basis": [[1, 0], [0, 1]],
            "radius_px": 0,
        }
        proxy = SimpleNamespace(spec=self.spec, case=placeholder)
        features, dimension = ProposalAdapter.features(
            proxy, planner_features, current, target, checker
        )
        features[9:21] = 0
        return features, dimension


@dataclass(frozen=True)
class CubeProposalSpec(ProposalSpec):
    max_horizon: int = 5
    block_action_dim: int = 25

    def __post_init__(self):
        if self.hidden_size != 8 or self.max_horizon != 5 or self.block_action_dim != 25:
            raise ValueError(
                "The frozen Cube adapter has eight hidden units and five 5D action blocks"
            )
        if self.normalized_action_bound != 1:
            raise ValueError("The declared normalized residual bound is one")


class CubeProposalAdapter(ProposalAdapter):
    def __init__(self, spec, parameters, case):
        self.spec, self.parameters, self.case = spec, np.asarray(parameters).copy(), case
        self.weights = spec.unpack(parameters)
        self.history = []
        if case["kind"] not in KINDS:
            raise ValueError("Unknown Cube hazard")

    def features(self, f, current, target, checker):
        body = checker.body_proprio.detach().cpu().double().numpy().reshape(4)
        delta = (target[0, -1] - current[0, -1]).detach().cpu().double().numpy().reshape(-1)
        projection = np.random.default_rng(self.spec.projection_seed).normal(
            0, 1 / np.sqrt(len(delta)), (8, len(delta))
        )
        shape = self.case
        present = shape["kind"] != "none"
        relative = np.asarray(shape["centre"]) - body[:2] if present else np.zeros(2)
        axis = shape["axis"] if present else [0, 0]
        sizes = [shape.get(key, 0) for key in ("radius", "half_length", "gap", "offset", "band")]
        values = np.r_[
            body,
            np.log1p(max(0, f["current_cost"])) / 10,
            f["improvement"],
            f["prediction_error"] / (1 + abs(f["prediction_error"])),
            f["previous_feasible"],
            f["replan"] / 30,
            relative,
            [shape["kind"] == k for k in KINDS[1:]],
            axis,
            sizes,
            np.tanh(projection @ delta),
        ]
        if values.shape != (29,) or not np.isfinite(values).all():
            raise ValueError("Invalid causal Cube proposal features")
        return np.clip(values, -1, 1), len(delta)

    def artifact(self):
        artifact = super().artifact()
        artifact["format"] = "lewm-cube-screen-residual-v1"
        artifact["feature_names"] = [
            "pinch_x_m",
            "pinch_y_m",
            "causal_vx_m_s",
            "causal_vy_m_s",
            *FEATURE_NAMES[4:],
        ]
        artifact["feature_units"] = (
            "Cube body and hazard positions, sizes and relative positions use one metre; velocities use one metre per second. Latent features unchanged."
        )
        return artifact

    @classmethod
    def load(cls, path, case):
        value = json.loads(path.read_text())
        if value["format"] != "lewm-cube-screen-residual-v1":
            raise ValueError("Cube controller format differs")
        return cls(CubeProposalSpec(**value["spec"]), value["parameters"], case)


def distance(points, shape, radius):
    """Independent NumPy signed distance for the declared disk and primitives."""
    points = np.asarray(points)
    if shape["kind"] == "none":
        return np.full(points.shape[:-1], 1e6)
    origin = shape.get("origin", shape.get("centre"))
    if "basis" in shape:
        basis = np.asarray(shape["basis"])
    else:
        axis = np.asarray(shape["axis"])
        basis = np.array([axis, [-axis[1], axis[0]]])
    xy = (points - np.asarray(origin)) @ basis.T
    kind = shape["kind"]
    r = shape.get("radius_px", shape.get("radius", 0))
    if kind == "island":
        value = np.linalg.norm(xy, axis=-1) - r
    elif kind == "bar":
        half = shape.get("segment_half_length_px", shape.get("half_length"))
        nearest = np.stack([np.clip(xy[..., 0], -half, half), np.zeros(xy.shape[:-1])], -1)
        value = np.linalg.norm(xy - nearest, axis=-1) - r
    else:
        offset = shape.get("gap_offset_px", shape.get("offset"))
        gap = shape.get("gap_width_px", shape.get("gap"))
        extent = shape.get("extent_px", shape.get("extent"))
        band = shape.get("band_half_width_px", shape.get("band"))
        parts = []
        for lo, hi in ((-extent, offset - gap / 2), (offset + gap / 2, extent)):
            q = np.abs(xy - [(lo + hi) / 2, 0]) - [(hi - lo) / 2, band]
            parts.append(np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0))
        value = np.minimum(*parts)
    return value - radius


def geometry(environment, index, initial, goal, reference, original_shape):
    """Fixed metadata rule, independent of controller outcomes or proposed actions."""
    if environment == "pusht":
        forward = np.asarray(goal)[:2] - np.asarray(initial)[:2]
        forward = forward / np.linalg.norm(forward)
        across = np.array([forward[1], -forward[0]])
        if np.dot(256 - (np.asarray(initial)[:2] + np.asarray(goal)[:2]) / 2, across) < 0:
            across = -across
        angle = np.deg2rad((-10, 0, 10)[index])
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        axis = across @ rotation.T
        basis = np.array([axis, [-axis[1], axis[0]]])
        centre = (np.asarray(initial)[:2] + np.asarray(goal)[:2]) / 2
        base = {"origin": centre.tolist(), "basis": basis.tolist()}
        return {
            "none": {"kind": "none", **base},
            "island": {"kind": "island", **base, "radius_px": (12, 16, 20)[index]},
            "bar": {
                "kind": "bar",
                **base,
                "radius_px": 7,
                "segment_half_length_px": (30, 40, 50)[index],
            },
            "gate": {
                "kind": "gate",
                **base,
                "gap_width_px": (100, 90, 80)[index],
                "gap_offset_px": 65,
                "extent_px": 180,
                "band_half_width_px": 7,
            },
        }
    angle = np.deg2rad((-10, 0, 10)[index])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    axis = np.asarray(original_shape["axis"]) @ rotation.T
    base = {"centre": original_shape["centre"], "axis": axis.tolist()}
    return {
        "none": {"kind": "none", **base},
        "island": {"kind": "island", **base, "radius": (0.009, 0.012, 0.015)[index]},
        "bar": {
            "kind": "bar",
            **base,
            "radius": 0.006,
            "half_length": (0.024, 0.030, 0.036)[index],
        },
        "gate": {
            "kind": "gate",
            **base,
            "offset": (0.12, 0.13, 0.14)[index],
            "gap": (0.18, 0.20, 0.22)[index],
            "extent": 0.27,
            "band": 0.006,
        },
    }


def summarize(rows):
    return {
        "episodes": len(rows),
        "safe_completion": sum(r["safe_completion"] for r in rows),
        "violation": sum(r["margin_breach"] or r["arena_margin_breach"] for r in rows),
        "unfinished": sum(r["unfinished"] for r in rows),
        "engaged": sum(r["engaged"] for r in rows),
        "actions": sum(r["actions"] for r in rows),
    }


def no_hazard_gate(rows):
    if len(rows) != 6 or any(r.get("error") for r in rows):
        raise ValueError("Readiness requires all six valid native no-hazard episodes")
    return summarize(rows)["safe_completion"] >= 4 and not summarize(rows)["violation"]


def image_stats(frame):
    array = np.asarray(frame)
    if array.shape != (224, 224, 3) or array.dtype != np.uint8:
        raise ValueError("Native policy frame shape/dtype differs")
    value = array.astype(float)
    contrast = float(value.std())
    rough = float(
        (np.abs(np.diff(value, axis=0)).mean() + np.abs(np.diff(value, axis=1)).mean()) / 2
    )
    return {
        "contrast": contrast,
        "roughness": rough,
        "flagged": rough / max(contrast, 1e-9) > 0.4,
        "qualification": "Roughness flag is an inspection aid, not a validated corruption classifier",
    }
