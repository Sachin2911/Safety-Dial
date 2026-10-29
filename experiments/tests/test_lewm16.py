"""Consequential contract checks, using synthetic costs and analytic paths only."""

import collections
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_worldmodel.policy import PlanConfig
from stable_worldmodel.solver import CEMSolver

from helpers.lewm16_planner import CampaignCEM, CostModel, ExactCheck, NoSafeAction, adapter_mode
from helpers.lewm16_protocol import load_spec


class DummyWorld(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def encode(self, info):
        return {"emb": torch.zeros(len(info["pixels"]), 1, 2)}

    def get_cost(self, info, actions):
        b, s, h, _ = actions.shape
        info.setdefault("emb", torch.zeros(b, s, 1, 2))
        info.setdefault("goal_emb", torch.zeros(b, 1, 2))
        info["predicted_emb"] = torch.zeros(b, s, h + 1, 2)
        return actions.square().sum((-1, -2))


def setup(boxes=(), checked=False):
    counters = collections.defaultdict(int)
    spec = load_spec("tworoom")
    scaler = SimpleNamespace(mean_=np.zeros(2), scale_=np.ones(2))
    check = ExactCheck("tworoom", spec, scaler, boxes, 0, counters)
    check.state, check.goal = np.array([164.0, 120.0]), np.array([164.0, 170.0])
    params = dict(
        horizon=5,
        variance=1.0,
        retain=0.0,
        adaptive=False,
        checked=checked,
        objective="latent",
        budget_scale=1.0,
    )
    model = CostModel(DummyWorld(), check, params, counters)
    solver = CampaignCEM(model, "tworoom", params, 123, "cpu")
    config = PlanConfig(5, 1, action_block=5)
    space = gym.spaces.Box(-1.0, 1.0, (1, 2), dtype=np.float64)
    solver.configure(action_space=space, n_envs=1, config=config)
    info = {
        "pixels": torch.zeros(1, 1, 3, 2, 2),
        "goal": torch.zeros(1, 1, 3, 2, 2),
        "action": torch.zeros(1, 1, 2),
    }
    return model, solver, info, space, config


def test_native_actions_match_upstream_cem():
    model, solver, info, space, config = setup()
    upstream = CEMSolver(
        model=DummyWorld(), num_samples=1000, n_steps=10, topk=10, seed=123, device="cpu"
    )
    upstream.configure(action_space=space, n_envs=1, config=config)
    expected = upstream.solve(dict(info))["actions"]
    actual = solver.solve(dict(info))["actions"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert model.counters["predictor_steps"] == 50005
    assert model.counters["analytic_path_steps"] > 250000


def test_infeasible_mean_and_candidates_never_execute():
    _, solver, info, _, _ = setup([(0, 224, 0, 224)], checked=True)
    with pytest.raises(NoSafeAction):
        solver.solve(info)


def test_horizon_switch_keeps_policy_buffer_and_counts_work():
    model, solver, info, _, _ = setup()
    solver.params.update(adaptive=True, horizon=3, error_threshold=-1, repair_horizon=2)
    first = solver.solve(dict(info))["actions"]
    second = solver.solve(dict(info), init_action=first[:, 1:])["actions"]
    assert first.shape == second.shape == (1, 5, 10)
    assert [r["horizon"] for r in solver.replans] == [3, 2]
    assert model.counters["predictor_steps"] <= 100010


def test_adapter_rejects_privileged_features():
    features = dict(
        current_cost=1.0,
        improvement=0.0,
        prediction_error=0.1,
        previous_feasible=0.9,
        replan=2,
        true_goal_distance=5.0,
    )
    with pytest.raises(ValueError):
        adapter_mode(features, {"adaptive": True})


def test_joint_repair_retains_exploration_during_model_error():
    features = dict(
        current_cost=10, improvement=-0.1, prediction_error=0.8, previous_feasible=0.8, replan=2
    )
    params = dict(
        adaptive=True, horizon=5, variance=1, retain=0, exploration_multiplier=2, stall_retain=0.5
    )
    assert adapter_mode(features, params) == (2, 1, 0)
    assert adapter_mode(features, {**params, "joint_response": True}) == (2, 2, 0.5)


def test_retention_fraction_changes_actual_candidate_allocation():
    _, solver, info, _, _ = setup()
    solver.params.update(retain=0.25)
    solver.solve(dict(info))
    assert solver.replans[-1]["retained_counts"][1:] == [250] * 9
    solver.params.update(retain=0.5)
    solver.solve(dict(info))
    assert solver.replans[-1]["retained_counts"] == [500] * 10


def test_flipped_render_is_contiguous_without_changing_image():
    from helpers.lewm16_runtime import Runtime

    frame = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)[::-1]
    runtime = Runtime.__new__(Runtime)
    runtime.env = SimpleNamespace(render=lambda: frame)
    runtime.counters = collections.defaultdict(int)
    info = runtime.frame_info({"goal": frame}, np.zeros(2))
    for key in ("pixels", "goal"):
        torch.from_numpy(info[key])
        np.testing.assert_array_equal(info[key][0, 0], frame)


def test_recorder_survives_rebuilt_environment_without_duplicate_hooks():
    from helpers.lewm16_runtime import Runtime

    runtime = Runtime.__new__(Runtime)
    runtime.kind, runtime._hooked_env = "reacher", None
    runtime.counters, runtime.fine_states = collections.defaultdict(int), []
    physics = SimpleNamespace(
        data=SimpleNamespace(qpos=np.zeros(2), qvel=np.zeros(2)),
        named=SimpleNamespace(data=SimpleNamespace(geom_xpos={"finger": np.zeros(3)})),
    )
    runtime.env = SimpleNamespace()
    for _ in range(2):
        runtime.env.env = SimpleNamespace(step=lambda _: "step", physics=physics)
        runtime.install_physics_recorder()
        runtime.install_physics_recorder()
        assert runtime.env.env.step(np.zeros(2)) == "step"
    assert runtime.counters["executed_physics_steps"] == len(runtime.fine_states) == 2
