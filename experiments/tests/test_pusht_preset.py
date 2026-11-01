"""Check the true preset, budget matching and development-only selection."""

import pytest
import torch

from helpers.pusht_preset import configurations, geometry, select
from helpers.safety_model import pusht_constraints


def test_author_settings_and_equal_episode_search_allowance():
    configs = configurations({"search_seed": 291129, "random_configurations": 6})
    assert len(configs) == 8
    author = configs[0]["params"]
    assert (author["horizon"], author["replan_blocks"], author["iterations"]) == (5, 5, 30)
    assert author["elite"] == 0.1 and author["budget_scale"] == 1
    for config in configs:
        p = config["params"]
        h = p["horizon"]
        r = p["replan_blocks"]
        n = int(300 * 30 * 5 * p["budget_scale"] // (30 * h))
        assert 1 <= r <= h <= 5
        assert n * 30 * h / (r * 5) == 1800
        assert n * 30 * h * ((50 + r * 5 - 1) // (r * 5)) <= 90000
    assert configs == configurations({"search_seed": 291129, "random_configurations": 6})


def row(index, condition, *, safe=False, actions=50, split="development"):
    return {
        "case_index": index,
        "condition": condition,
        "split": split,
        "error": None,
        "safe_completion": safe,
        "margin_breach": False,
        "arena_margin_breach": False,
        "hazard_violation": False,
        "unfinished": not safe,
        "actions": actions,
        "pusher_path_length_px": 20.0,
        "planning_seconds": 1.0,
        "episode_seconds": 2.0,
        "resources": {},
    }


def test_selection_isolated_and_inactivity_never_beats_completion():
    configs = configurations({"search_seed": 1, "random_configurations": 0})
    results = {
        c["id"]: [row(i, k, safe=k == "control") for i in range(6) for k in ("control", "hazard")]
        for c in configs
    }
    assert select(configs, results)["selected"]["id"] == "authors"
    results["working-anchor"][1] = row(0, "hazard", safe=True, actions=25)
    assert select(configs, results)["selected"]["id"] == "working-anchor"
    results["working-anchor"][0] = row(0, "control", safe=False, actions=0)
    assert select(configs, results)["selected"]["id"] == "authors"
    results["working-anchor"][0]["split"] = "evaluation"
    with pytest.raises(ValueError, match="development"):
        select(configs, results)


def test_no_hazard_control_retains_workspace_and_settled_endpoint_admission():
    constraints = pusht_constraints({"kind": "none", "origin": [0, 0], "basis": [[1, 0], [0, 1]]})
    assert len(constraints) == 1 and constraints[0].kind == "workspace"
    assert constraints[0].clearance(torch.tensor([[20.0, 100.0]]), 15).item() == 5
    initial = [20, 100, 200, 200, 0, 0, 0]
    goal = [300, 300, 250, 250, 0, 0, 0]
    assert geometry(initial, goal, [[20, 100], [150, 200], [300, 300]], "island", "mild") is None
