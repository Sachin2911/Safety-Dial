"""Proposal integration and optimizer checks; execute only on the rented host."""

import collections
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
from helpers.lewm_proposal_adapter import FEATURE_NAMES, ProposalAdapter, ProposalSpec
from helpers.lewm_proposal_evolution import ProposalEvolution

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Rented GPU required")
CASE = {"kind": "island", "origin": [200, 250], "basis": [[1, 0], [0, 1]], "radius_px": 12}


def test_parameter_contract_and_bounded_replay(tmp_path):
    spec = ProposalSpec()
    assert len(FEATURE_NAMES) == 29 and spec.parameter_count == 510
    theta = spec.initial_parameters(5)
    adapter = ProposalAdapter(spec, theta, CASE)
    features = np.linspace(-1, 1, 29)
    actual = adapter.forward(features, 3)
    assert actual.shape == (3, 10) and np.isfinite(actual).all()
    assert abs(actual).max() <= 1
    assert np.array_equal(actual, adapter.forward(features, 3))
    import json

    path = tmp_path / "adapter.json"
    path.write_text(json.dumps(adapter.artifact()))
    assert np.array_equal(actual, ProposalAdapter.load(path, CASE).forward(features, 3))
    with pytest.raises(ValueError):
        ProposalAdapter(spec, np.full(510, np.nan), CASE)
    with pytest.raises(ValueError):
        adapter.forward(np.zeros(29), 4)


def test_controller_initialization_creates_parent_and_preserves_existing(tmp_path, monkeypatch):
    import json

    from scripts import run_lewm_proposal_smoke as smoke

    monkeypatch.setattr(smoke, "ROOT", tmp_path)
    protocol = {
        "case": CASE,
        "adapter": {"seed": 5, "spec": {}, "checkpoint": "controllers/fresh/initial.json"},
    }
    adapter, path = smoke.controller(protocol)
    original = path.read_bytes()
    assert path.is_file() and json.loads(original)["parameters"] == adapter.parameters.tolist()
    smoke.controller(protocol)
    assert path.read_bytes() == original
    path.write_text("{}")
    with pytest.raises(ValueError):
        smoke.controller(protocol)


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


@pytest.mark.parametrize("unsafe", [False, True])
def test_real_solver_injection_query_budget_and_refusal(unsafe):
    adapter = ProposalAdapter(ProposalSpec(), ProposalSpec().initial_parameters(5), CASE)
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
    solver = PlanningCEM(cost, "pusht", params, 17, "cuda", proposal_adapter=adapter)
    solver.configure(
        action_space=gym.spaces.Box(-1, 1, shape=(1, 2)),
        n_envs=1,
        config=PlanConfig(horizon=5, receding_horizon=1, action_block=5),
    )
    info = {key: torch.zeros(1, 1, 1, device="cuda") for key in ("pixels", "goal", "action")}
    if unsafe:
        with pytest.raises(NoSafeAction):
            solver.solve(info)
    else:
        assert torch.isfinite(solver.solve(info)["actions"]).all()
        assert len(solver.replans[-1]["adapter_checks"]) == 2
        assert solver.replans[-1]["executed_violation"] == 0
    assert counters["adapter_forward_calls"] == 1
    assert counters["adapter_candidate_evaluations"] == 2
    assert counters["predictor_steps"] in (63, 66)
    assert len(adapter.history) == 1
    assert np.array_equal(
        adapter.forward(adapter.history[0]["features"], 3),
        np.asarray(adapter.history[0]["proposal"]),
    )


def test_reject_missing_observation_and_nonfinite_inputs():
    adapter = ProposalAdapter(ProposalSpec(), ProposalSpec().initial_parameters(5), CASE)
    with pytest.raises(ValueError):
        adapter.features({}, torch.zeros(1, 1, 1), torch.zeros(1, 1, 1), object())
    with pytest.raises(ValueError):
        adapter.forward(np.full(29, np.nan), 3)


def test_evolution_checkpoint_resume_and_reject_invalid_fitness(tmp_path):
    initial = ProposalSpec().initial_parameters(5)
    options = dict(seed=7, population=4, sigma=0.05, checkpoint=tmp_path / "search.pkl")
    search = ProposalEvolution(initial, **options)
    candidates = search.candidates()
    restored = ProposalEvolution(initial, **options, resume=True)
    assert np.array_equal(restored.candidates(), candidates)
    with pytest.raises(ValueError):
        restored.submit([float("nan")] * 4)
    restored.submit(-np.square(candidates).sum(1))
    assert len(restored.history) == 1 and restored.pending is None
    again = ProposalEvolution(initial, **options, resume=True)
    assert len(again.history) == 1
    assert again.candidates().shape == (4, 510)
