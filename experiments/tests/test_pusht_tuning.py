"""Selection isolation, constant search work and deterministic random configurations."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from helpers.pusht_smoke import physical_clearance
from helpers.pusht_tuning import configurations, layout, render_goal_image, select_configuration

ROOT = Path(__file__).resolve().parents[2]


def design():
    return json.loads((ROOT / "configs/pusht_tuning_20261029/design.json").read_text())


def row(case, *, safe=False, actions=50, breach=False, split="development"):
    return {
        "study_case_id": case,
        "study_split": split,
        "safe_completion": safe,
        "hazard_violation": breach,
        "margin_breach": breach,
        "arena_margin_breach": False,
        "unfinished": not safe,
        "error": None,
        "actions": actions,
        "pusher_path_length_px": 10 if actions else 0,
        "object_displacement_px": 10,
        "planning_seconds": 1.0,
        "episode_seconds": 2.0,
        "resources": {"environment_actions": actions, "predictor_steps": actions * 100},
    }


def test_frozen_random_search_is_unique_reproducible_and_matches_predictor_work():
    first, second = configurations(design()), configurations(design())
    assert first == second and len(first) == 7 and first[0]["id"] == "current"
    tuples = set()
    for config in first:
        p = config["params"]
        tuples.add(tuple(p[k] for k in design()["search_space"]))
        assert 15 * (750 // p["horizon"]) * p["horizon"] == 11250
        assert p["iterations"] == 15 and p["budget_scale"] == 0.5
        assert not p["adaptive"] and p["checked"] and p["objective"] == "learned"
    assert len(tuples) == 7


def test_selection_rewards_completion_and_never_cheap_inactivity():
    configs = configurations(design())[:2]
    results = {
        "current": [row("a"), row("b")],
        "random-01": [row("a", actions=0), row("b", actions=0)],
    }
    assert select_configuration(configs, results, ["a", "b"])["selected"]["id"] == "current"
    results["random-01"][0] = row("a", safe=True, actions=30)
    assert select_configuration(configs, results, ["a", "b"])["selected"]["id"] == "random-01"
    results["random-01"][1] = row("b", breach=True)
    assert select_configuration(configs, results, ["a", "b"])["selected"]["id"] == "current"


def test_reserved_outcomes_and_incomplete_panels_cannot_select_settings():
    configs = configurations(design())[:1]
    with pytest.raises(ValueError, match="Reserved"):
        select_configuration(configs, {"current": [row("a", split="evaluation")]}, ["a"])
    with pytest.raises(ValueError, match="entire frozen"):
        select_configuration(configs, {"current": [row("a")]}, ["a", "b"])


def test_short_scene_hazard_has_clear_endpoints_and_blocks_the_straight_path():
    initial, goal = np.array([100.0, 200.0]), np.array([158.0, 200.0])
    case = layout(initial, goal, "island", design())
    endpoints = torch.tensor(np.stack([initial, goal]))
    assert (physical_clearance(endpoints, case) >= 6.001).all()
    line = torch.tensor(np.linspace(initial, goal, 100))
    assert physical_clearance(line, case).min() < 6


def test_native_goal_render_updates_cached_shapes_without_extra_physics():
    from stable_worldmodel.envs.pusht.env import PushT

    counters = {"reset_physics_steps": 0}

    class Counted(PushT):
        def _set_state(self, state):
            original = self.space.step

            def step(dt):
                counters["reset_physics_steps"] += 1
                return original(dt)

            self.space.step = step
            try:
                return super()._set_state(state)
            finally:
                self.space.step = original

    env = Counted()
    try:
        env.reset(seed=7, options={"state": np.array([120, 120, 260, 300, 0.4, 0, 0])})
        before = env.render()
        after = render_goal_image(env, np.array([400, 200, 320, 280, 1.2, 0, 0]))
        assert np.abs(before.astype(float) - after.astype(float)).mean() > 1
        assert counters["reset_physics_steps"] == 2
        assert np.allclose(env._get_obs()[:5], [400, 200, 320, 280, 1.2])
    finally:
        env.close()
