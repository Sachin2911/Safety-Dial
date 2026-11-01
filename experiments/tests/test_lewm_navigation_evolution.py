"""Action contract tests to be executed on declared rented GPU hosts only."""

import collections
from pathlib import Path
import sys
from types import SimpleNamespace

from stable_worldmodel import PlanConfig

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.lewm18_planner import PlanningCEM
from helpers.lewm_navigation_evolution import ActionContract, ProjectedCostModel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Rented GPU required")


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.candidates = []

    def encode(self, info):
        return {"emb": info["pixels"]}

    def get_cost(self, info, candidates):
        self.candidates.append(candidates.detach().clone())
        info["predicted_emb"] = torch.zeros(
            1, candidates.shape[1], candidates.shape[2] + 1, 1, device="cuda"
        )
        return (candidates - 7).square().sum((-1, -2))


class Checker:
    kind = "portable_learned"

    def __init__(self, contract):
        self.contract = contract
        self.candidates = []
        self.model = SimpleNamespace(
            spec={
                "normalized_action_low": contract.normalized_low,
                "normalized_action_high": contract.normalized_high,
            }
        )

    def check(self, candidates):
        self.candidates.append(candidates.detach().clone())
        b, n, h, d = candidates.shape
        shaped = candidates.reshape(b, n, h, -1, self.contract.dimension)
        low, high = [
            candidates.new_tensor(v)
            for v in (self.contract.normalized_low, self.contract.normalized_high)
        ]
        assert bool((shaped >= low).all()) and bool((shaped <= high).all())
        zero = candidates.new_zeros(b, n)
        return zero, candidates[..., :2], zero


def contract():
    return ActionContract((0.2, -0.1), (0.5, 2.0), (-1, -1), (1, 1))


def test_inplace_projection_unifies_model_checker_and_native_command():
    c = contract()
    model, checker, counts = Model(), Checker(c), collections.defaultdict(int)
    cost = ProjectedCostModel(model, checker, {"objective": "learned", "checked": True}, counts, c)
    candidates = torch.tensor([[[[-30.0, 20.0, 0.0, 0.0]]]], device="cuda")
    before = candidates.clone()
    info = {
        "emb": torch.zeros(1, 1, 1, 1, device="cuda"),
        "goal_emb": torch.zeros(1, 1, 1, device="cuda"),
    }
    cost.get_cost(info, candidates)
    assert not torch.equal(before, candidates)
    torch.testing.assert_close(model.candidates[0], checker.candidates[0], rtol=0, atol=0)
    torch.testing.assert_close(model.candidates[0], candidates, rtol=0, atol=0)
    normalized = candidates.reshape(-1, 2).cpu().double().numpy()
    raw = normalized * c.scale + c.mean
    executed = c.execute(raw)
    np.testing.assert_allclose(executed, raw, rtol=0, atol=1e-6)
    assert counts["model_batch_calls"] == 1
    assert counts["model_candidate_rollouts"] == 1
    assert counts["projected_candidate_coordinates"] == 2


def test_mean_archive_and_selected_plan_stay_in_model_action_domain():
    c = contract()
    model, checker, counts = Model(), Checker(c), collections.defaultdict(int)
    params = {
        "horizon": 2,
        "iterations": 3,
        "budget_scale": 0.02,
        "variance": 2.0,
        "retain": 0.5,
        "correlate": 0.0,
        "checked": True,
        "objective": "learned",
        "cem_refinements": True,
        "replan_blocks": 1,
    }
    cost = ProjectedCostModel(model, checker, params, counts, c)
    solver = PlanningCEM(cost, "pusht", params, 401, "cuda")
    import gymnasium as gym

    solver.configure(
        action_space=gym.spaces.Box(-1, 1, shape=(1, 2)),
        n_envs=1,
        config=PlanConfig(horizon=5, receding_horizon=1, action_block=5),
    )
    info = {
        "pixels": torch.zeros(1, 1, 1, device="cuda"),
        "goal": torch.ones(1, 1, 1, device="cuda"),
        "action": torch.zeros(1, 1, 10, device="cuda"),
    }
    for _ in range(3):
        result = solver.solve(info)
        actions = result["actions"] if isinstance(result, dict) else result
        assert torch.isfinite(actions).all()
        for seen in model.candidates:
            checker.check(seen)
    assert counts["projected_candidate_coordinates"] > 0


def test_motion_model_must_match_native_contract():
    c = contract()
    checker = Checker(c)
    checker.model.spec["normalized_action_low"] = [0, 0]
    with pytest.raises(ValueError, match="bounds differ"):
        ProjectedCostModel(Model(), checker, {}, collections.defaultdict(int), c)
    with pytest.raises(ValueError):
        ActionContract((0, 0), (0, 1), (-1, -1), (1, 1))


def test_unprojected_diagnostic_control_preserves_original_cost_semantics():
    from helpers.lewm18_planner import CostModel

    c = contract()
    models = [Model(), Model()]

    class LooseChecker:
        kind = "portable_learned"

        def check(self, candidates):
            zero = candidates.new_zeros(candidates.shape[:2])
            return zero, candidates[..., :2], zero

    counters = [collections.defaultdict(int), collections.defaultdict(int)]
    params = {"objective": "learned", "checked": True}
    old = CostModel(models[0], LooseChecker(), params, counters[0])
    observed = ProjectedCostModel(models[1], LooseChecker(), params, counters[1], c, project=False)
    raw = torch.tensor([[[[-30.0, 20.0, 0.0, 0.0]]]], device="cuda")
    copies = [raw.clone(), raw.clone()]
    infos = [
        {
            "emb": torch.zeros(1, 1, 1, 1, device="cuda"),
            "goal_emb": torch.zeros(1, 1, 1, device="cuda"),
        }
        for _ in range(2)
    ]
    wanted = old.get_cost(infos[0], copies[0])
    actual = observed.get_cost(infos[1], copies[1])
    torch.testing.assert_close(wanted, actual, rtol=0, atol=0)
    torch.testing.assert_close(copies[1], raw, rtol=0, atol=0)
    torch.testing.assert_close(models[0].candidates[0], models[1].candidates[0], rtol=0, atol=0)
    for key in ("model_batch_calls", "model_candidate_rollouts", "predictor_steps"):
        assert counters[0][key] == counters[1][key]
    assert counters[1]["action_projection_calls"] == 0
