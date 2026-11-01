"""Failure-accounting checks, executed only on the declared rented GPU."""

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_lewm_navigation_campaign import AccountedModel, ProgressCounter

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Rented GPU required")


class FailingModel(torch.nn.Module):
    def encode(self, info):
        raise RuntimeError("encoder failed after query admission")

    def get_cost(self, info, candidates):
        raise RuntimeError("predictor failed after query admission")


def test_failed_frozen_queries_are_reported_before_exception():
    snapshots = []
    c = ProgressCounter(lambda actions, resources: snapshots.append((actions, resources)))
    c["environment_resets"] += 1
    model = AccountedModel(FailingModel(), c)
    with pytest.raises(RuntimeError, match="encoder failed"):
        model.encode({"pixels": torch.zeros(1, 1, 3, 224, 224, device="cuda")})
    assert snapshots[-1][0] == 0
    assert snapshots[-1][1]["actual_encoder_call_attempts"] == 1
    assert snapshots[-1][1]["actual_encoder_frames_attempted"] == 1
    assert c["actual_encoder_call_successes"] == 0
    with pytest.raises(RuntimeError, match="predictor failed"):
        model.get_cost({}, torch.zeros(1, 3, 2, 10, device="cuda"))
    assert snapshots[-1][1]["actual_cost_call_attempts"] == 1
    assert snapshots[-1][1]["actual_candidate_rollouts_attempted"] == 3
    assert snapshots[-1][1]["actual_candidate_block_steps_attempted"] == 6
    assert c["actual_cost_call_successes"] == 0


def test_simulator_attempt_callback_is_cumulative_and_preserves_reset_costs():
    snapshots = []
    c = ProgressCounter(lambda actions, resources: snapshots.append((actions, resources)))
    c["environment_resets"] += 1
    c["render_frames"] += 1
    c["environment_actions"] += 1
    c["environment_actions"] += 1
    assert [s[0] for s in snapshots] == [0, 1, 2]
    assert snapshots[-1][1]["environment_resets"] == 1
    assert snapshots[-1][1]["render_frames"] == 1
    assert snapshots[-1][1]["environment_actions"] == 2


def test_policy_live_pixels_equal_direct_observed_bank_preprocessing():
    import collections
    from types import SimpleNamespace

    import gymnasium as gym
    import numpy as np
    from stable_worldmodel import PlanConfig
    from stable_worldmodel.policy import WorldModelPolicy

    from helpers.lewm18_planner import PlanningCEM
    from helpers.lewm_navigation_evolution import ActionContract, ProjectedCostModel
    from helpers.pusht_smoke import Scaler
    from helpers.safecem10_planner import image_transform

    class Base(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.images = []

        def encode(self, info):
            self.images.append(info["pixels"].detach().cpu().clone())
            return {"emb": torch.zeros(1, 1, 1, device="cuda")}

        def get_cost(self, info, candidates):
            info["predicted_emb"] = torch.zeros(
                1, candidates.shape[1], candidates.shape[2] + 1, 1, device="cuda"
            )
            return candidates.square().sum((-1, -2))

    class Checker:
        kind = "portable_learned"

        def check(self, candidates):
            zero = candidates.new_zeros(candidates.shape[:2])
            return zero, candidates[..., :2], zero

    params = {
        "horizon": 1,
        "iterations": 1,
        "budget_scale": 0.005,
        "variance": 1.0,
        "checked": True,
        "objective": "learned",
        "replan_blocks": 1,
        "cem_refinements": True,
    }
    base, c = Base(), collections.defaultdict(int)
    normal = Scaler([0.003, -0.052], [0.867, 0.868])
    contract = ActionContract(tuple(normal.mean_), tuple(normal.scale_), (-1.0, -1.0), (1.0, 1.0))
    cost = ProjectedCostModel(base, Checker(), params, c, contract)
    solver = PlanningCEM(cost, "pusht", params, 654, "cuda")
    policy = WorldModelPolicy(
        solver,
        PlanConfig(horizon=5, receding_horizon=1, action_block=5),
        process={"action": normal},
        transform={"pixels": image_transform(), "goal": image_transform()},
    )
    policy.set_env(
        SimpleNamespace(
            num_envs=1,
            single_action_space=gym.spaces.Box(-1, 1, shape=(2,)),
            action_space=gym.spaces.Box(-1, 1, shape=(1, 2)),
        )
    )
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    frame[20:40, 50:70, 0] = 255
    goal = np.roll(frame, 25, axis=0)
    policy.get_action(
        {"pixels": frame[None, None], "goal": goal[None, None], "action": np.zeros((1, 1, 2))}
    )
    assert len(base.images) == 2
    for observed, raw in zip(base.images, (frame, goal), strict=True):
        bank = image_transform()(torch.from_numpy(raw).permute(2, 0, 1)[None]).unsqueeze(1)
        torch.testing.assert_close(observed, bank, rtol=0, atol=0)
        assert observed.dtype == torch.float32
