"""Rented-GPU equivalence, influence, refusal and common-panel recovery checks."""

import collections
import importlib.util
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import pytest
from stable_worldmodel import PlanConfig
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.lewm16_planner import NoSafeAction
from helpers.lewm18_planner import CostModel, PlanningCEM
from helpers.lewm_proposal_adapter import ProposalSpec
from helpers.lewm_residual import ResidualAdapter, ResidualSpec, mix_candidates
from helpers.lewm_residual_study import ResidualStudy

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Rented GPU required")
CASE = {"kind": "island", "origin": [200, 250], "basis": [[1, 0], [0, 1]], "radius_px": 12}


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
        self.observed = torch.tensor([[100, 150, 0, 0]], dtype=torch.float64)

    def check(self, candidates):
        zeros = torch.zeros(candidates.shape[:2], device="cuda")
        return zeros + int(self.unsafe), candidates[..., :2], zeros


def make_solver(adapter=None, *, kind=PlanningCEM, unsafe=False):
    params = dict(
        horizon=3,
        iterations=3,
        budget_scale=0.04,
        variance=1.0,
        retain=0.5,
        correlate=0.5,
        checked=True,
        objective="learned",
        proposal_mode="residual_mixture",
        proposal_mixture_fraction=0.5,
    )
    counters = collections.defaultdict(int)
    cost = CostModel(Base(), Checker(unsafe), params, counters)
    kwargs = {} if adapter is None else {"proposal_adapter": adapter}
    solver = kind(cost, "pusht", params, 17, "cuda", **kwargs)
    solver.configure(
        action_space=gym.spaces.Box(-1, 1, shape=(1, 2)),
        n_envs=1,
        config=PlanConfig(horizon=5, receding_horizon=1, action_block=5),
    )
    return solver, counters


def test_zero_recovers_frozen_native_for_multiple_replans():
    path = Path("reference/lewm18_planner.py")
    assert path.is_file(), "Provide the preserved pre-change planner on the rented host"
    spec = importlib.util.spec_from_file_location("frozen_native_planner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    adapter = ResidualAdapter(ResidualSpec(), ResidualSpec().initial_parameters(5), CASE)
    assert not adapter.forward(np.linspace(-1, 1, 29), 3).any()
    old, old_counts = make_solver(kind=module.PlanningCEM)
    native, native_counts = make_solver()
    residual, residual_counts = make_solver(adapter)
    warm = None
    for _ in range(3):
        info = {k: torch.zeros(1, 1, 1, device="cuda") for k in ("pixels", "goal", "action")}
        wanted = old.solve(info, warm)["actions"]
        assert torch.equal(native.solve(info, warm)["actions"], wanted)
        assert torch.equal(residual.solve(info, warm)["actions"], wanted)
        warm = wanted[:, 1:]
    for key in ("predictor_steps", "model_candidate_rollouts", "model_batch_calls"):
        assert old_counts[key] == native_counts[key] == residual_counts[key]


@pytest.mark.parametrize("unsafe", [False, True])
def test_nonzero_influence_and_safety_refusal(unsafe, monkeypatch):
    theta = ResidualSpec().initial_parameters(5)
    theta[-30:] = 0.5
    adapter = ResidualAdapter(ResidualSpec(), theta, CASE)
    solver, counts = make_solver(adapter, unsafe=unsafe)
    # A controlled candidate population makes guided samples strictly better for
    # this quadratic task. Random noise does not guarantee any guided elite wins.
    monkeypatch.setattr(
        torch,
        "randn",
        lambda *shape, **kwargs: torch.zeros(
            *shape, device=kwargs["device"], dtype=kwargs["dtype"]
        ),
    )
    info = {k: torch.zeros(1, 1, 1, device="cuda") for k in ("pixels", "goal", "action")}
    if unsafe:
        with pytest.raises(NoSafeAction):
            solver.solve(info)
    else:
        solver.solve(info)
        checks = solver.replans[-1]["residual_checks"]
        assert sum(c["guided_elites"] for c in checks) > 0
        assert all(0 < c["new_guided_samples"] < c["samples"] for c in checks)
        assert solver.replans[-1]["executed_violation"] == 0
    assert counts["adapter_guided_candidate_evaluations"] > 0


def test_mixture_preserves_reserved_slots_and_residual_artifact(tmp_path):
    values = torch.zeros(1, 10, 3, 10, device="cuda")
    guided = mix_candidates(values, torch.ones(1, 3, 10, device="cuda") * 0.25, 4, 0.5)
    assert guided.sum() == 3 and not values[:, :7].any() and (values[:, -3:] == 0.25).all()
    with pytest.raises(ValueError):
        mix_candidates(values, torch.zeros(1, 3, 10, device="cuda"), 4, 1.1)
    import json

    adapter = ResidualAdapter(ResidualSpec(), ProposalSpec().initial_parameters(2), CASE)
    path = tmp_path / "residual.json"
    path.write_text(json.dumps(adapter.artifact()))
    assert np.array_equal(ResidualAdapter.load(path, CASE).parameters, adapter.parameters)


def test_common_panel_resume_and_completion_gate(tmp_path, monkeypatch):
    design = {
        "search": {"seed": 22, "population": 4, "sigma": 0.1, "order_seed": 77, "generations": 2},
        "adapter": {"spec": {}, "seed": 5},
        "random_search_seed": 66,
        "diagnostic": [["island", 1]],
        "development": [["island", 2], ["bar", 3]],
        "selection": [["island", 4], ["bar", 5]],
        "maximum_valid_episodes": 51,
    }
    monkeypatch.setattr(ResidualStudy, "diagnostic", lambda self: True)
    calls, stopped = [], False

    def evaluate(key, theta, case, seed):
        nonlocal stopped
        if len(calls) == 13 and not stopped:
            stopped = True
            raise RuntimeError("interrupt")
        calls.append((key, case, seed))
        np.random.seed(seed)
        success = key.startswith("selection_cma") and case == "bar"
        return {
            "safe_completion": success,
            "margin_breach": False,
            "arena_margin_breach": False,
            "unfinished": not success,
            "actions": 20,
            "pusher_path_length_px": 3,
            "safe_progress": 1.0 if success else 0.2,
            "error": None,
        }

    study = ResidualStudy(design, ResidualSpec().initial_parameters(5), tmp_path, evaluate)
    with pytest.raises(RuntimeError, match="interrupt"):
        study.run()
    recovered = ResidualStudy(
        design, ResidualSpec().initial_parameters(5), tmp_path, evaluate, resume=True
    )
    state = recovered.run()
    assert len(calls) == len({k for k, c, s in calls}) == 51
    assert state["phase"] == "development_signal"
    assert state["selection"]["winner"] == "cma" and not state["selection"]["validation_opened"]
    assert all(seed in (2, 3) for key, case, seed in calls if key.startswith(("cma_", "random_")))
    assert len(recovered.es.history) == 2
