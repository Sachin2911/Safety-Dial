"""Run on the declared rented GPU: adapter contracts, gates and observation cadence."""

import collections
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_worldmodel import PlanConfig
from stable_worldmodel.policy import WorldModelPolicy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.lewm18_planner import CostModel, PlanningCEM
from helpers.lewm_proposal_adapter import ProposalAdapter, ProposalSpec
from helpers.lewm_screen import (
    CubeProposalAdapter,
    CubeProposalSpec,
    ScreenPushTAdapter,
    distance,
    geometry,
    image_stats,
    no_hazard_gate,
)
from helpers.pusht_smoke import Scaler
from helpers.safety_model import pusht_constraints
from helpers.portable_motion import shape_constraints

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Rented GPU required")
SHAPE = {
    "kind": "bar",
    "origin": [200, 250],
    "basis": [[1, 0], [0, 1]],
    "radius_px": 7,
    "segment_half_length_px": 30,
}
FEATURES = {
    "current_cost": 0.0,
    "improvement": 0.0,
    "prediction_error": 0.0,
    "previous_feasible": 1.0,
    "replan": 0,
}


class Base(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoded = []

    def encode(self, info):
        self.encoded.append(info["pixels"].detach().cpu().tolist())
        return {"emb": info["pixels"]}

    def get_cost(self, info, candidates):
        info["predicted_emb"] = torch.zeros(
            1, candidates.shape[1], candidates.shape[2] + 1, 1, device="cuda"
        )
        return (candidates - 0.3).square().sum((-1, -2))


class Checker:
    kind = "portable_learned"

    def __init__(self):
        self.observed = torch.tensor([[100, 150, 0, 0]], dtype=torch.float64)
        self.body_proprio = torch.tensor([[0.5, 0.1, 0, 0]], dtype=torch.float64)

    def check(self, candidates):
        z = candidates.new_zeros(candidates.shape[:2])
        return z, candidates[..., :2], z


def solver(adapter=None):
    params = dict(
        horizon=5,
        iterations=2,
        budget_scale=0.02,
        variance=1.0,
        retain=0.5,
        correlate=0.5,
        checked=True,
        objective="learned",
        proposal_mode="residual_mixture",
        proposal_mixture_fraction=0.5,
    )
    base = Base()
    counters = collections.defaultdict(int)
    cost = CostModel(base, Checker(), params, counters)
    return PlanningCEM(cost, "cube", params, 71, "cuda", proposal_adapter=adapter), base, counters


def test_push_weights_and_hazard_features_preserved_with_none_extension():
    spec = ProposalSpec()
    weights = spec.initial_parameters(9)
    original = ProposalAdapter(spec, weights, SHAPE)
    screened = ScreenPushTAdapter(spec, weights, SHAPE)
    current = torch.zeros(1, 1, 1, device="cuda")
    target = current + 1
    wanted, _ = original.features(FEATURES, current, target, Checker())
    got, _ = screened.features(FEATURES, current, target, Checker())
    assert np.array_equal(wanted, got)
    none = ScreenPushTAdapter(spec, weights, {"kind": "none"})
    empty, _ = none.features(FEATURES, current, target, Checker())
    assert not empty[9:21].any()
    assert np.array_equal(empty[:9], got[:9]) and np.array_equal(empty[21:], got[21:])


def test_cube_action_contract_and_residual_serialization(tmp_path):
    import json

    spec = CubeProposalSpec()
    assert spec.parameter_count == 1365
    adapter = CubeProposalAdapter(spec, spec.initial_parameters(11), {"kind": "none"})
    current = torch.zeros(1, 1, 1, device="cuda")
    x, _ = adapter.features(FEATURES, current, current + 1, Checker())
    values = adapter.forward(x, 5)
    assert values.shape == (5, 25) and np.abs(values).max() <= 1
    assert not x[9:21].any()
    path = tmp_path / "cube.json"
    path.write_text(json.dumps(adapter.artifact()))
    assert np.array_equal(CubeProposalAdapter.load(path, {"kind": "none"}).forward(x, 5), values)
    with pytest.raises(ValueError):
        CubeProposalSpec(block_action_dim=10)
    scaler = Scaler([0.01] * 5, [0.3, 0.4, 0.6, 0.4, 0.25])
    action = np.array([[9, -7, 5, -5, 2.0]])
    bounded = np.clip(scaler.inverse_transform(action), -1, 1)
    np.testing.assert_allclose(
        scaler.transform(bounded),
        np.clip(action, (-1 - scaler.mean_) / scaler.scale_, (1 - scaler.mean_) / scaler.scale_),
    )


def test_cube_zero_residual_recovers_native_multiple_replans():
    spec = CubeProposalSpec()
    weights = spec.initial_parameters(11)
    weights[240:] = 0
    native, _, nc = solver()
    guided, _, gc = solver(CubeProposalAdapter(spec, weights, {"kind": "none"}))
    for s in (native, guided):
        s.configure(
            action_space=gym.spaces.Box(-1, 1, shape=(1, 5)),
            n_envs=1,
            config=PlanConfig(horizon=5, receding_horizon=5, action_block=5),
        )
    warm = None
    for _ in range(3):
        info = {k: torch.zeros(1, 1, 1, device="cuda") for k in ("pixels", "goal", "action")}
        wanted = native.solve(info, warm)["actions"]
        assert torch.equal(guided.solve(info, warm)["actions"], wanted)
        warm = wanted[:, 1:]
    for key in ("predictor_steps", "model_batch_calls", "model_candidate_rollouts"):
        assert nc[key] == gc[key]


def test_held_images_between_plans_do_not_change_policy_actions():
    policies, bases = [], []
    for _ in range(2):
        s, b, _ = solver()
        policy = WorldModelPolicy(s, PlanConfig(horizon=5, receding_horizon=5, action_block=5))
        policy.set_env(
            type(
                "Env",
                (),
                {
                    "num_envs": 1,
                    "single_action_space": gym.spaces.Box(-1, 1, shape=(1, 5)),
                    "action_space": gym.spaces.Box(-1, 1, shape=(1, 5)),
                },
            )()
        )
        policies.append(policy)
        bases.append(b)
    cached = None
    for index in range(31):
        fresh = np.array([[[index / 100.0]]], dtype=np.float32)
        if not len(policies[1]._action_buffer[0]):
            cached = fresh
        commands = []
        for policy, pixels in zip(policies, (fresh, cached), strict=True):
            commands.append(
                policy.get_action(
                    {
                        "pixels": pixels,
                        "goal": np.ones((1, 1, 1), np.float32),
                        "action": np.zeros((1, 1, 5), np.float32),
                    }
                )
            )
        assert np.array_equal(*commands)
    assert bases[0].encoded == bases[1].encoded
    assert len(bases[0].encoded) == 4


@pytest.mark.parametrize("environment", ["pusht", "cube"])
def test_geometry_agrees_with_independent_constraint_implementation(environment):
    rng = np.random.default_rng(42)
    for index in range(3):
        shapes = geometry(
            environment, index, [120, 120], [380, 380], None, {"centre": [0.5, 0], "axis": [1, 0]}
        )
        for kind in ("island", "bar", "gate"):
            shape = shapes[kind]
            scale, radius = (512, 15) if environment == "pusht" else (1, 0.005)
            points = rng.uniform(0, scale, (50, 2))
            constraints = (
                pusht_constraints(shape, include_arena=False)
                if environment == "pusht"
                else shape_constraints(shape)
            )
            tensor = torch.tensor(points, dtype=torch.float64, device="cuda")
            expected = (
                torch.stack([c.clearance(tensor, radius) for c in constraints], -1)
                .amin(-1)
                .cpu()
                .numpy()
            )
            np.testing.assert_allclose(
                distance(points, shape, radius), expected, rtol=0, atol=1e-10
            )


def test_gates_and_image_inspection_flags():
    rows = [
        {
            "safe_completion": i < 4,
            "margin_breach": False,
            "arena_margin_breach": False,
            "unfinished": i >= 4,
            "engaged": True,
            "actions": 30,
            "error": None,
        }
        for i in range(6)
    ]
    assert no_hazard_gate(rows)
    rows[3]["safe_completion"] = False
    assert not no_hazard_gate(rows)
    with pytest.raises(ValueError):
        no_hazard_gate(rows[:5])
    noise = np.random.default_rng(2).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    assert image_stats(noise)["flagged"]
    with pytest.raises(ValueError):
        image_stats(noise.astype(np.float32))
