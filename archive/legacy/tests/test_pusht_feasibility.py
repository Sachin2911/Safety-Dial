"""Witness safety must concern the same hazardous nominal route and actual goal."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from helpers.pushtFeasibility import assess_witness, feasibility_gate
from helpers.pushtGeometry import Disc, T_LOCAL, clearance_trace


def witness(episode=0):
    # Familiar cell (row=1,col=1), positive nominal crossing, safe observed detour.
    hz = Disc(250., 200., 20.)
    start, goal = [100., 150., 0.], [400., 150., 0.]
    nominal = np.array([start, [250., 150., 0.], goal])
    route = np.array([start, [100., 20., 0.], [400., 20., 0.], goal])
    states = np.tile([100., 100., 0., 0., 0., 0., 0.], (4, 1))
    states[:, 2:5] = route
    return {"role": "development", "root": {"root_id": f"dev-{episode}",
        "goal_state": [0., 0., *goal, 0., 0.], "meta": {"episode": episode}},
        "local_polygons": [p.tolist() for p in T_LOCAL], "root_pose": start,
        "states": states.tolist(), "actions": np.zeros((3, 2)).tolist(), "observed_steps": 3,
        "nominal_route_poses": nominal.tolist(), "layout": {"family": "familiar",
            "hazard": hz.to_dict(), "start_clearance": float(clearance_trace(np.array([start]), hz)[0]),
            "goal_clearance": float(clearance_trace(np.array([goal]), hz)[0]),
            "nominal_min_clearance": float(clearance_trace(nominal, hz).min())},
        "replay": {"repeats": 2, "bitwise_equal": True}}


def test_safe_detour_reaches_goal_and_crossing_nominal_is_rejected():
    record = witness()
    assert assess_witness(record)["passes"]
    record["states"][1][2:5] = [250., 150., 0.]
    check = assess_witness(record)
    assert not check["passes"]
    assert check["min_clearance"] <= 0


@pytest.mark.parametrize("change", ["wrong_role", "unrelated_hazard", "goal_not_reached",
    "padded_suffix", "bad_replay", "changed_layout", "arena_exit", "wrong_geometry"])
def test_cannot_substitute_weaker_witness(change):
    record = witness()
    if change == "wrong_role":
        record["role"] = "test"
    elif change == "unrelated_hazard":
        record["nominal_route_poses"] = [record["root_pose"]]
    elif change == "goal_not_reached":
        record["states"][-1][2] = 100.
    elif change == "padded_suffix":
        record["observed_steps"] = 2
    elif change == "bad_replay":
        record["replay"]["bitwise_equal"] = False
    elif change == "changed_layout":
        record["layout"]["start_clearance"] = 999.
    elif change == "arena_exit":
        record["states"][1][0] = -1.
    else:
        record["local_polygons"][0][0][0] += 1.
    assert not assess_witness(record)["passes"]


def test_gate_requires_three_independent_development_sources():
    records = [witness(i) for i in range(3)]
    assert feasibility_gate(records, [0, 1, 2])["passes"]
    assert not feasibility_gate(records, [0, 1])["passes"]
    assert not feasibility_gate([records[0], copy.deepcopy(records[0]), records[1]], [0, 1])["passes"]
