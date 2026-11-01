"""Observed-target and learning-interface checks, on rented GPU hosts only."""

import collections
from pathlib import Path
import sys
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
from stable_worldmodel import PlanConfig
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.lewm18_planner import PlanningCEM
from helpers.lewm_navigation_evolution import ActionContract, ProjectedCostModel
from helpers.lewm_navigation_retrieval import (
    RetrievalController,
    RetrievalMemory,
    RetrievalPlanningCEM,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Rented GPU required")


def arrays():
    return {
        "source_embeddings": np.array([[0.0, 0.0], [0.0, 0.0], [20.0, 0.0]], dtype=np.float32),
        "successor_embeddings": np.array([[1.0, 0.0], [2.0, 0.0], [21.0, 0.0]], dtype=np.float32),
        "endpoint_embeddings": np.array([[3.0, 0.0], [4.0, 0.0], [23.0, 0.0]], dtype=np.float32),
        "latent_scale": np.ones(2, dtype=np.float32),
        "source_xy": np.array([[40, 40], [40, 40], [80, 80]], dtype=np.float32),
        "successor_xy": np.array([[45, 40], [50, 40], [85, 80]], dtype=np.float32),
        "endpoint_xy": np.array([[55, 40], [60, 40], [100, 80]], dtype=np.float32),
        "source_episode": np.array([128, 129, 130]),
        "source_step": np.zeros(3, dtype=int),
        "successor_step": np.array([3, 5, 10]),
        "endpoint_step": np.full(3, 25, dtype=int),
    }


def query():
    return torch.tensor([0.0, 0.0], device="cuda"), torch.tensor([4.0, 0.0], device="cuda")


def test_target_is_observed_successor_or_original_goal_and_provenance_replays():
    memory = RetrievalMemory(arrays(), fingerprint="fixed-memory")
    policy = RetrievalController(
        memory, RetrievalController.fixed_ap_parameters(), {"kind": "none"}
    )
    current, final = query()
    chosen = policy.select(current, final, [40.0, 40.0])
    torch.testing.assert_close(chosen, memory.tables["successor_embeddings"][1], rtol=0, atol=0)
    assert policy.history[-1]["target_kind"] == "observed_successor"
    assert policy.history[-1]["source_provenance"]["source_episode"] == 129
    again = RetrievalController(memory, policy.parameters, policy.shape)
    torch.testing.assert_close(again.select(current, final, [40.0, 40.0]), chosen, rtol=0, atol=0)
    # A sufficiently near original goal competes directly rather than interpolating a latent.
    torch.testing.assert_close(policy.select(final, final, [60.0, 40.0]), final, rtol=0, atol=0)
    assert policy.history[-1]["target_kind"] == "original_goal"


def test_hazard_filter_uses_only_observed_successor_body_and_declared_margin():
    memory = RetrievalMemory(arrays())
    shape = {"kind": "island", "centre": [50.0, 40.0], "axis": [1.0, 0.0], "radius": 0.5}
    policy = RetrievalController(memory, policy_parameters(), shape, body_radius=1.0, margin=1.0)
    current, final = query()
    chosen = policy.select(current, final, [40.0, 40.0])
    assert not policy.last_available[1]
    torch.testing.assert_close(chosen, memory.tables["successor_embeddings"][0], rtol=0, atol=0)
    assert policy.last_available[
        -1
    ]  # Original goal admission is the task protocol's responsibility.


def policy_parameters():
    return RetrievalController.fixed_ap_parameters()


def test_ordinary_gradient_learning_can_update_same_genotype_and_final_goal_weight():
    features = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 0],
            [2, 1, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 3],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    available = torch.tensor([True, True, True], device="cuda")
    theta = torch.tensor(
        [1.0, 1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5], device="cuda", requires_grad=True
    )
    loss = -RetrievalController.log_probability(features, available, theta, 1)
    loss.backward()
    assert torch.isfinite(theta.grad).all() and torch.count_nonzero(theta.grad[:7]) >= 5
    assert theta.grad[-1].abs() > 0
    before = theta.detach().clone()
    with torch.no_grad():
        theta -= 0.01 * theta.grad
    assert not torch.equal(theta, before)


def test_invalid_or_time_reversed_observation_memory_rejected(tmp_path):
    values = arrays()
    values["successor_step"][0] = -1
    with pytest.raises(ValueError, match="after its source"):
        RetrievalMemory(values)
    values = arrays()
    path = tmp_path / "memory.npz"
    np.savez_compressed(path, **values)
    loaded = RetrievalMemory.from_npz(path)
    assert loaded.count == 3 and loaded.dimension == 2 and len(loaded.fingerprint) == 64


class Base(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.targets = []

    def encode(self, info):
        return {"emb": info["pixels"]}

    def get_cost(self, info, candidates):
        self.targets.append(info["goal_emb"].detach().clone())
        info["predicted_emb"] = torch.zeros(
            1, candidates.shape[1], candidates.shape[2] + 1, 2, device="cuda"
        )
        return (candidates - 0.2).square().sum((-1, -2))


class Checker:
    kind = "portable_learned"

    def __init__(self):
        self.observed = torch.tensor([40.0, 40.0], device="cuda")
        self.model = SimpleNamespace(
            spec={"normalized_action_low": [-1, -1], "normalized_action_high": [1, 1]}
        )

    def check(self, candidates):
        z = candidates.new_zeros(candidates.shape[:2])
        return z, candidates[..., :2], z


def solver(controller=None, *, native=False):
    params = {
        "horizon": 2,
        "iterations": 2,
        "budget_scale": 0.02,
        "variance": 1.0,
        "retain": 0.25,
        "correlate": 0.0,
        "objective": "learned",
        "checked": True,
        "cem_refinements": True,
        "replan_blocks": 1,
    }
    model, counters = Base(), collections.defaultdict(int)
    cost = ProjectedCostModel(
        model,
        Checker(),
        params,
        counters,
        ActionContract((0.0, 0.0), (1.0, 1.0), (-1.0, -1.0), (1.0, 1.0)),
    )
    policy = (
        PlanningCEM(cost, "pusht", params, 12, "cuda")
        if native
        else RetrievalPlanningCEM(cost, params, 12, "cuda", controller)
    )
    policy.configure(
        action_space=gym.spaces.Box(-1, 1, shape=(1, 2)),
        n_envs=1,
        config=PlanConfig(horizon=5, receding_horizon=1, action_block=5),
    )
    return policy, model, counters


def info():
    current, final = query()
    return {
        "pixels": current[None, None],
        "goal": final[None, None],
        "action": torch.zeros(1, 1, 2, device="cuda"),
    }


def test_planner_substitutes_cached_target_without_extra_live_encoder_queries():
    memory = RetrievalMemory(arrays())
    controller = RetrievalController(memory, policy_parameters(), {"kind": "none"})
    policy, model, counters = solver(controller)
    query_info = info()
    original = query_info["goal"].clone()
    policy.solve(query_info)
    assert counters["encoder_frames"] == 2 and counters["retrieval_queries"] == 1
    for target in model.targets:
        torch.testing.assert_close(
            target[0, -1], memory.tables["successor_embeddings"][1], rtol=0, atol=0
        )
    torch.testing.assert_close(query_info["goal"], original, rtol=0, atol=0)
    assert policy.replans[-1]["target_retrieval"]["memory_index"] == 1


def test_disabled_retrieval_recovers_projected_native_search_exactly():
    native, _, nc = solver(native=True)
    wrapped, _, wc = solver()
    for _ in range(3):
        torch.testing.assert_close(
            native.solve(info())["actions"], wrapped.solve(info())["actions"], rtol=0, atol=0
        )
    for key in (
        "encoder_frames",
        "predictor_steps",
        "model_candidate_rollouts",
        "model_batch_calls",
    ):
        assert nc[key] == wc[key]
