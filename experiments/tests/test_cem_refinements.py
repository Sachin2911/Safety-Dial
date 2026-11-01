"""Safety-critical refinement edge cases; run only on the rented GPU host."""

import collections
from pathlib import Path
import sys

import gymnasium as gym
import pytest
import torch
from stable_worldmodel import PlanConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.cem_refinements import choose_checked_mean, fit_distribution
from helpers.lewm18_planner import CostModel, PlanningCEM
from helpers.lewm16_planner import NoSafeAction

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Rented GPU required")


def tensor(value):
    return torch.tensor(value, device="cuda", dtype=torch.float32)


def test_only_safe_elites_and_singleton_keep_spread():
    candidates = tensor([[[[2.0]], [[100.0]], [[200.0]]]])
    mean, spread, log = fit_distribution(
        candidates,
        tensor([[1, 2, 3]]),
        tensor([[1, 0, 0]]).bool(),
        tensor([[[0.0]]]),
        tensor([[[0.7]]]),
        3,
        0.1,
        0.05,
    )
    assert mean.item() == pytest.approx(1.8)
    assert spread.item() == pytest.approx(0.7)
    assert log["fitted_elites"] == 1 and log["feasible_only"]


def test_no_safe_candidates_fit_least_violation_and_preserve_floor():
    candidates = tensor([[[[2.0]], [[2.0]], [[100.0]]]])
    mean, spread, log = fit_distribution(
        candidates,
        tensor([[5, 6, 100]]),
        tensor([[0, 0, 0]]).bool(),
        tensor([[[0.0]]]),
        tensor([[[0.01]]]),
        2,
        0.1,
        0.05,
    )
    assert mean.item() == pytest.approx(1.8)
    assert spread.item() == pytest.approx(0.05)
    assert not log["feasible_only"] and log["fitted_elites"] == 2


def test_better_checked_candidate_beats_a_safe_mean():
    mean, best = tensor([[[5.0]]]), tensor([[[2.0]]])
    chosen, source = choose_checked_mean(mean, 10, True, (3, best))
    assert torch.equal(chosen, best) and source == "best_checked"
    chosen, source = choose_checked_mean(mean, 2, True, (3, best))
    assert torch.equal(chosen, mean) and source == "mean"
    with pytest.raises(ValueError, match="No feasible"):
        choose_checked_mean(mean, 2, False, None)


class Base(torch.nn.Module):
    def encode(self, info):
        return {"emb": info["pixels"]}

    def get_cost(self, info, candidates):
        info["predicted_emb"] = torch.zeros(
            1, candidates.shape[1], candidates.shape[2] + 1, 1, device="cuda"
        )
        return (candidates - 0.3).square().sum((-1, -2))


class Checker:
    kind = "pusht"

    def __init__(self, unsafe=False):
        self.unsafe = unsafe

    def check(self, candidates):
        zero = torch.zeros(candidates.shape[:2], device="cuda")
        return zero + int(self.unsafe), candidates[..., :2], zero


@pytest.mark.parametrize("unsafe", [False, True])
def test_actual_solver_checks_choice_and_accounts_final_queries(unsafe):
    params = dict(
        horizon=3,
        iterations=2,
        budget_scale=0.02,
        variance=1.0,
        retain=0.5,
        correlate=0.5,
        checked=True,
        objective="learned",
    )
    counters = collections.defaultdict(int)
    cost = CostModel(Base(), Checker(unsafe), params, counters)
    solver = PlanningCEM(cost, "pusht", params, 17, "cuda")
    solver.configure(
        action_space=gym.spaces.Box(-1, 1, shape=(1, 2)),
        n_envs=1,
        config=PlanConfig(horizon=5, receding_horizon=1, action_block=5),
    )
    info = {key: tensor([[[0.0]]]) for key in ("pixels", "goal", "action")}
    if unsafe:
        with pytest.raises(NoSafeAction):
            solver.solve(info)
        assert solver.replans[-1]["executed"] == "none"
    else:
        result = solver.solve(info)
        assert torch.isfinite(result["actions"]).all()
        log = solver.replans[-1]
        assert len(log["refinements"]) == 2
        assert all(x["spread_min"] >= 0.05 - 1e-7 for x in log["refinements"])
        assert log["executed"] in ("mean", "best_checked")
    # Search: 2 * 10 * 3; mean scoring: 3; optionally selected-best imagination: 3.
    assert counters["predictor_steps"] in (63, 66)
    assert counters["model_batch_calls"] in (3, 4)
