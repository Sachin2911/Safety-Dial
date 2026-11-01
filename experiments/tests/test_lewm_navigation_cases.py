"""Case-admission checks to run only on the declared rented host."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_lewm_navigation_cases import candidate_pair, select_pilot

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Rented GPU required")


def geometry():
    return {
        "source_stride": 5,
        "endpoint_offsets": [100, 75, 50, 25],
        "minimum_displacement": 60.0,
        "maximum_displacement": 160.0,
        "body_radius": 7.0,
        "margin": 3.0,
        "allowance": 12.405431747436523,
        "admission_buffer": 2.0,
        "require_room_contained": True,
        "left_room_max_x": 99.5,
        "right_room_min_x": 124.5,
    }


def test_pair_selection_uses_earliest_source_then_frozen_offset_order():
    route = np.stack([np.full(101, 60.0), np.linspace(30, 180, 101)], -1)
    chosen, rejected = candidate_pair(route, "circle3", geometry())
    assert chosen["source_step"] == 0 and chosen["goal_step"] == 100
    assert chosen["reference_midpoint_step"] == 50
    np.testing.assert_array_equal(chosen["shape"]["centre"], route[50])
    assert not rejected
    shorter, rejected = candidate_pair(route[:51], "none", geometry())
    assert shorter["source_step"] == 0 and shorter["goal_step"] == 50
    assert [v["reason"] for v in rejected[:2]] == ["temporal_endpoint_outside_episode"] * 2


def test_strong_circle_cannot_weaken_fixed_allowance_to_admit_short_pair():
    route = np.stack([np.full(26, 60.0), np.linspace(50, 110, 26)], -1)
    none, _ = candidate_pair(route, "none", geometry())
    mild, _ = candidate_pair(route, "circle3", geometry())
    strong, rejected = candidate_pair(route, "circle7", geometry())
    assert none is not None and mild is not None and strong is None
    failed = [row for row in rejected if row["reason"] == "endpoint_safety_admission"]
    assert failed and failed[0]["threshold"] == pytest.approx(17.405431747436523)
    assert min(failed[0]["clearance"]) < failed[0]["threshold"]


def test_inadmissible_episode_returns_all_reasons_without_changing_source_pool():
    route = np.stack([np.full(31, 60.0), np.linspace(50, 60, 31)], -1)
    chosen, rejected = candidate_pair(route, "none", geometry())
    assert chosen is None
    assert len(rejected) == len(range(0, 31, 5)) * 4
    assert {r["reason"] for r in rejected} <= {
        "temporal_endpoint_outside_episode",
        "endpoint_displacement",
    }


def test_doorway_crossing_reference_is_rejected_before_hazard_outcomes():
    route = np.stack([np.linspace(60, 164, 101), np.linspace(30, 180, 101)], -1)
    chosen, rejected = candidate_pair(route, "circle3", geometry())
    assert chosen is None
    assert any(row["reason"] == "reference_segment_enters_doorway_region" for row in rejected)


def test_pilot_uses_first_two_eligible_per_condition_before_outcomes():
    conditions = ["none"] * 5 + ["circle3", "circle5", "circle7", "circle3", "circle5", "circle7"]
    cases = [
        {"source_episode": 1000 + i, "condition": condition, "split": "development_training"}
        for i, condition in enumerate(conditions)
    ]
    policy = {"per_condition": {"none": 2, "circle3": 2, "circle5": 2, "circle7": 2}}
    chosen = select_pilot(cases, policy)
    assert [c["source_episode"] for c in chosen] == [1000, 1001, 1005, 1006, 1007, 1008, 1009, 1010]
    for condition in policy["per_condition"]:
        assert sum(c["condition"] == condition for c in chosen) == 2
    with pytest.raises(ValueError, match="Insufficient"):
        select_pilot(cases[:-1], policy)


def test_pilot_never_uses_selection_or_final_cases():
    case = {"source_episode": 2000, "condition": "none", "split": "development_selection"}
    with pytest.raises(ValueError, match="cannot enter"):
        select_pilot([case], {"desired_cases": 1, "minimum_cases": 1})
