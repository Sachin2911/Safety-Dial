"""Scientific interface, optimizer gradient, and hard-budget regression checks."""

import collections
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_worldmodel.policy import PlanConfig
from stable_worldmodel.solver import CEMSolver
from test_lewm16 import DummyWorld

from helpers.lewm17_budget import BudgetCounter, ceilings
from helpers.lewm17_planner import CampaignCEM, CostModel, ExactCheck, NoSafeAction
from helpers.lewm17_protocol import load_spec
from helpers.lewm17_selector import (
    anchors,
    es_gradient,
    features_vector,
    mode_params,
    probabilities,
    reinforce_gradient,
)


def setup(params=None, boxes=(), margin=6):
    counters = collections.defaultdict(int)
    scaler = SimpleNamespace(mean_=np.zeros(2), scale_=np.ones(2))
    checker = ExactCheck("tworoom", load_spec("tworoom"), scaler, boxes, margin, counters)
    checker.state, checker.goal = np.array([164.0, 120.0]), np.array([164.0, 170.0])
    p = {
        "horizon": 5,
        "variance": 1.0,
        "retain": 0.0,
        "correlate": 0.0,
        "adaptive": False,
        "checked": True,
        "objective": "latent",
        "budget_scale": 1.0,
    }
    p.update(params or {})
    model = CostModel(DummyWorld(), checker, p, counters)
    solver = CampaignCEM(model, "tworoom", p, 123, "cpu")
    config = PlanConfig(5, 1, action_block=5)
    space = gym.spaces.Box(-1.0, 1.0, (1, 2), dtype=np.float64)
    solver.configure(action_space=space, n_envs=1, config=config)
    info = {
        "pixels": torch.zeros(1, 1, 3, 2, 2),
        "goal": torch.zeros(1, 1, 3, 2, 2),
        "action": torch.zeros(1, 1, 2),
    }
    return model, solver, info, space, config


def test_native_identity():
    _, solver, info, space, config = setup({"checked": False})
    upstream = CEMSolver(
        model=DummyWorld(), num_samples=1000, n_steps=10, topk=10, seed=123, device="cpu"
    )
    upstream.configure(action_space=space, n_envs=1, config=config)
    torch.testing.assert_close(
        solver.solve(dict(info))["actions"], upstream.solve(dict(info))["actions"], rtol=0, atol=0
    )


@pytest.mark.parametrize("mode", range(6))
def test_constant_selector_is_exact_fixed_mode(mode):
    _, fixed, info, _, _ = setup(mode_params("tworoom", mode))
    _, selector, _, _, _ = setup({"weights": anchors("tworoom")[mode], "margin": 6})
    init = None
    for _ in range(2):
        a = fixed.solve(dict(info), init_action=init)["actions"]
        b = selector.solve(dict(info), init_action=init)["actions"]
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        init = a[:, 1:]
    assert selector.decisions[-1]["mode"] == mode


def test_forbidden_features_and_no_safe_fallback():
    f = {
        "current_cost": 3,
        "improvement": 0.2,
        "prediction_error": 0.5,
        "previous_feasible": 0.9,
        "replan": 3,
    }
    x = features_vector(f, "reacher", 0.015)
    assert x.shape == (6,)
    with pytest.raises(ValueError):
        features_vector({**f, "true_goal_distance": 1}, "reacher", 0.015)
    _, solver, info, _, _ = setup(boxes=[(0, 224, 0, 224)])
    with pytest.raises(NoSafeAction):
        solver.solve(info)


def test_reinforce_gradient_matches_exact_expected_reward():
    rng = np.random.default_rng(17)
    w = rng.normal(0, 0.2, (6, 6))
    x = np.array([1.0, -0.2, 0.1, 0.7, 0.5, 1.0])
    reward = np.array([0.0, 1.0, -0.1, 0.3, 0.2, 0.8])
    p = probabilities(w, x)
    # Sum over all actions analytically, no Monte Carlo ambiguity.
    grad = np.zeros((6, 6))
    for a in range(6):
        g, _ = reinforce_gradient([(reward[a], [{"x": x, "probabilities": p, "mode": a}])], 0.2)
        grad += p[a] * g
    epsilon = 1e-5
    numerical = np.empty_like(w)
    for i in range(6):
        for j in range(6):
            delta = np.zeros_like(w)
            delta[i, j] = epsilon
            numerical[i, j] = (
                (probabilities(w + delta, x) - probabilities(w - delta, x)) @ reward / (2 * epsilon)
            )
    np.testing.assert_allclose(grad, numerical, atol=1e-9)
    assert probabilities(np.clip(w + grad, -3, 3), x) @ reward > p @ reward


def test_es_antithetic_sign_and_zero_reward_gradient():
    directions = np.ones((4, 6, 6))
    np.testing.assert_allclose(es_gradient(directions, [1.0, 0.0] * 4), 1 / 0.6)
    np.testing.assert_array_equal(es_gradient(directions, [1.0] * 8), 0)


def test_resource_limits_refuse_before_extra_work_and_normalization_roundtrip():
    for env in ("tworoom", "reacher"):
        counter = BudgetCounter(env)
        for key, maximum in ceilings(env).items():
            counter[key] = maximum
            with pytest.raises(RuntimeError):
                counter[key] += 1
            assert counter[key] == maximum
    from helpers.lewm17_runtime import Scaler

    scaler = Scaler([0.2, -0.1], [0.4, 0.7])
    values = np.array([[10.0, -5.0], [-0.3, 0.8]])
    np.testing.assert_allclose(scaler.inverse_transform(scaler.transform(values)), values)
