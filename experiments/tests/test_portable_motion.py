"""Safety-interface and censored-budget regression checks, run remotely."""

import collections
import pytest
import torch
from helpers.portable_motion import TransitionModel, MotionChecker, clearance, shape_constraints
from scripts.run_portable_lewm import select


def test_release_gate_rejects_uncalibrated_head():
    model = TransitionModel({"observation_dim": 2, "action_dim": 2, "substeps": 1})
    model.calibration = {"ready": False}
    with pytest.raises(ValueError, match="release gate"):
        MotionChecker(model, [], 3, collections.Counter())


def test_shapes_block_reference_but_leave_a_detour():
    for kind in ("island", "bar", "gate"):
        shape = {
            "kind": kind,
            "centre": [0.0, 0.0],
            "axis": [1.0, 0.0],
            "radius": 2.0,
            "half_length": 8.0,
            "offset": 40.0,
            "gap": 60.0,
            "extent": 90.0,
            "band": 2.0,
        }
        constraints = shape_constraints(shape)
        points = torch.tensor([[0.0, 0.0], [40.0, 0.0], [0.0, -70.0], [0.0, 70.0]])
        distances = clearance(points, constraints, 7.0)
        assert distances[0] < 0
        assert distances[1] > 3
        assert bool((distances[2:] > 3).all())


def test_selection_uses_full_150_action_endpoint():
    configs = [{"id": "authors"}, {"id": "candidate"}]
    results = {}
    for config in configs:
        rows = []
        for index in range(6):
            for condition in ("hazard", "control"):
                success = condition == "control" or config["id"] == "candidate"
                rows.append(
                    {
                        "case_index": index,
                        "condition": condition,
                        "split": "development",
                        "error": None,
                        "safe_completion": success,
                        "margin_breach": False,
                        "hazard_violation": False,
                        "unfinished": not success,
                        "engaged": True,
                        "actions": 90 if success else 150,
                        "planning_seconds": 0.0,
                        "resources": {},
                    }
                )
        results[config["id"]] = rows
    chosen = select(configs, results)
    assert chosen["selected"]["id"] == "candidate"
    assert chosen["development"]["candidate"]["hazard"]["completion_at"] == {
        "50": 0,
        "100": 6,
        "150": 6,
    }
    assert chosen["development"]["authors"]["hazard"]["J"] == 900
