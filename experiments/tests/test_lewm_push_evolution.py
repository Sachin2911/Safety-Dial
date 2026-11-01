"""Structured PushT controller checks; execute only on the declared rented GPU."""

import collections
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_worldmodel import PlanConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.lewm_navigation_evolution import ActionContract, ProjectedCostModel
from helpers.lewm_push_evolution import (
    PushEvolutionAdapter,
    PushEvolutionSpec,
    PushPlanningCEM,
    hazard_frame,
)
from helpers.pusht_smoke import physical_clearance

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Rented GPU required")
BASE = {"origin": [200, 250], "basis": [[0, 1], [-1, 0]]}
SHAPES = (
    {"kind": "none", **BASE},
    {"kind": "island", **BASE, "radius_px": 18},
    {"kind": "bar", **BASE, "radius_px": 7, "segment_half_length_px": 40},
    {
        "kind": "gate",
        **BASE,
        "gap_width_px": 90,
        "gap_offset_px": 65,
        "extent_px": 180,
        "band_half_width_px": 7,
    },
)
FEATURES = {
    "current_cost": 50.0,
    "improvement": 0.0,
    "prediction_error": 0.0,
    "previous_feasible": 1.0,
    "replan": 0,
}
CONTRACT = ActionContract((0.01, -0.02), (0.2, 0.3), (-1, -1), (1, 1))


def adapter(parameters=None, shape=SHAPES[1]):
    spec = PushEvolutionSpec()
    if parameters is None:
        parameters = spec.initial_parameters()
    return PushEvolutionAdapter(spec, parameters, shape, CONTRACT.mean, CONTRACT.scale)


def test_serialization_zero_anchor_and_native_residual_bounds(tmp_path):
    a = adapter()
    assert a.spec.parameter_count == 24
    checker = SimpleNamespace(observed=torch.tensor([[260.0, 250.0, 0, 0]]))
    x, basis, _ = a.features(FEATURES, checker)
    np.testing.assert_array_equal(a.forward(x, 5, basis), np.zeros((5, 10)))
    weights = a.spec.random_parameters(713, scale=20)
    a = adapter(weights)
    proposal = a.forward(x, 5, basis)
    native = proposal.reshape(-1, 2) * CONTRACT.scale
    assert np.abs(native).max() <= a.spec.native_residual_bound
    np.testing.assert_array_equal(a.forward(x, 3, basis), proposal[:3])
    path = tmp_path / "controller.json"
    path.write_text(json.dumps(a.artifact()))
    b = PushEvolutionAdapter.load(path, SHAPES[1])
    np.testing.assert_array_equal(b.forward(x, 5, basis), proposal)
    with pytest.raises(ValueError):
        adapter(np.zeros(23))


@pytest.mark.parametrize("shape", SHAPES)
def test_shape_frame_matches_independent_monitor(shape):
    points = np.random.default_rng(11).uniform(50, 450, (40, 2))
    expected = physical_clearance(torch.as_tensor(points, dtype=torch.float64), shape).numpy()
    for point, reference in zip(points, expected, strict=True):
        clear, normal = hazard_frame(point, shape)
        if shape["kind"] == "none":
            assert clear > 1e4 and np.array_equal(normal, [0, 0])
        else:
            assert clear == pytest.approx(reference, abs=1e-10)
            assert np.linalg.norm(normal) == pytest.approx(1.0)


class Base(torch.nn.Module):
    def encode(self, info):
        return {"emb": info["pixels"]}

    def get_cost(self, info, candidates):
        info["predicted_emb"] = candidates.new_zeros(
            1, candidates.shape[1], candidates.shape[2] + 1, 1
        )
        return (candidates - 0.3).square().sum((-1, -2))


class Checker:
    kind = "pusht"
    observed = torch.tensor([[260.0, 250.0, 0, 0]], dtype=torch.float64)

    def check(self, candidates):
        z = candidates.new_zeros(candidates.shape[:2])
        return z, candidates[..., :2], z


def solver(proposal=None):
    params = {
        "horizon": 3,
        "iterations": 3,
        "budget_scale": 0.02,
        "variance": 1.0,
        "retain": 0.5,
        "correlate": 0.5,
        "checked": True,
        "objective": "learned",
        "proposal_mode": "residual_mixture",
        "proposal_mixture_fraction": 0.5,
    }
    counters = collections.Counter()
    cost = ProjectedCostModel(Base(), Checker(), params, counters, CONTRACT)
    s = PushPlanningCEM(cost, "pusht", params, 127, "cuda", proposal_adapter=proposal)
    s.configure(
        action_space=gym.spaces.Box(-1, 1, shape=(1, 2)),
        n_envs=1,
        config=PlanConfig(horizon=5, receding_horizon=1, action_block=5),
    )
    return s, counters


def test_zero_controller_preserves_native_actions_and_model_work_over_replans():
    a = adapter()
    native, nc = solver()
    guided, gc = solver(a)
    warm = None
    for _ in range(3):
        info = {key: torch.zeros(1, 1, 1, device="cuda") for key in ("pixels", "goal", "action")}
        reference = native.solve(info, warm)["actions"]
        actual = guided.solve(info, warm)["actions"]
        assert torch.equal(reference, actual)
        warm = reference[:, 1:]
    for key in (
        "predictor_steps",
        "model_batch_calls",
        "model_candidate_rollouts",
        "learned_candidate_checks",
    ):
        assert nc[key] == gc[key]
    assert len(a.history) == 3 and gc["adapter_forward_calls"] == 3
    for row in a.history:
        np.testing.assert_array_equal(
            a.forward(row["features"], len(row["proposal"]), row["basis"]), row["proposal"]
        )


def test_direction_authority_and_causal_context():
    theta = np.zeros(24)
    theta[4] = 2  # Constant temporal mode, clockwise tangent, bias feature.
    a = adapter(theta)
    checker = Checker()
    x, basis, _ = a.features(FEATURES, checker)
    first = a.forward(x, 3, basis).reshape(-1, 2) * CONTRACT.scale
    assert np.linalg.norm(first[0]) > 0.1
    assert np.dot(first[0], basis[1]) > 0
    init = torch.full((1, 2, 10), 0.8)
    a.set_plan_context(init)
    _, with_plan, _ = a.features(FEATURES, checker)
    assert np.linalg.norm(with_plan[2]) == pytest.approx(1.0)
    assert np.array_equal(basis[:2], with_plan[:2])
    with pytest.raises(ValueError):
        a.features(FEATURES, SimpleNamespace(observed=torch.zeros(1, 7)))
    with pytest.raises(ValueError):
        a.set_plan_context(np.ones((1, 2, 2)))
