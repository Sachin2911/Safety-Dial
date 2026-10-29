"""Unit tests for the task-14 Cube dial study (geometry and the commanded-path check)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.cube14_geometry import (  # noqa: E402
    band_crossing,
    gap_closed,
    inside_any,
    outcome_metrics,
    signed_distance,
)
from helpers.cube14_planner import (  # noqa: E402
    STEP_M,
    WORKSPACE,
    CubeCostModel,
    commanded_path,
    densify,
    raw_actions,
    tracking_errors,
)

LAYOUT = {"gap_center_x": 0.4, "gap_width": 0.06, "y_band": [0.0, 0.04]}
SPEC = {"hazard": {"left_edge": 0.2, "right_end": 0.65}, "environment": {"eval_budget_steps": 50}}


def boxes():
    from helpers.cube14_geometry import hazard_boxes

    return hazard_boxes(LAYOUT, SPEC)


def test_boxes_gap_and_closure():
    left, right = boxes()
    assert right[0] - left[1] == pytest.approx(0.06)
    assert left[2:] == (0.0, 0.04)
    assert gap_closed(LAYOUT, 0.03) and not gap_closed(LAYOUT, 0.0299)


def test_inside_and_signed_distance():
    xy = np.array([[0.3, 0.02], [0.4, 0.02], [0.3, 0.06]])
    assert inside_any(xy, boxes()).tolist() == [True, False, False]
    d = signed_distance(xy, boxes())
    assert d[0] == pytest.approx(-0.02)
    assert d[1] == pytest.approx(0.03)
    assert d[2] == pytest.approx(0.02)


def test_crossing():
    gap = np.array([[0.4, 0.08], [0.4, 0.02], [0.4, -0.03]])
    band = np.array([[0.3, 0.08], [0.3, 0.02], [0.3, -0.03]])
    assert band_crossing(gap, LAYOUT, SPEC) == "gap"
    assert band_crossing(band, LAYOUT, SPEC) == "band"


def test_commanded_path_clips_to_workspace_and_applies_gain():
    raw = torch.zeros(1, 1, 3, 5, dtype=torch.float64)
    raw[..., 0] = 1.0
    start = np.array([0.58, 0.0, 0.1])
    p = commanded_path(start, raw)[0, 0].numpy()
    assert p[0, 0] == pytest.approx(WORKSPACE[1][0])
    q = commanded_path(np.array([0.3, 0.0, 0.1]), raw, gain=0.5)[0, 0].numpy()
    assert q[0, 0] == pytest.approx(0.3 + 0.5 * STEP_M)
    assert q[1, 0] == pytest.approx(0.3 + STEP_M)


def test_densify_reaches_each_waypoint():
    start = torch.zeros(1, 1, 1, 3, dtype=torch.float64)
    path = torch.tensor([[[[1.0, 0, 0], [1.0, 1.0, 0]]]], dtype=torch.float64)
    d = densify(start, path, 2)[0, 0].numpy()
    assert d.tolist() == [[0.5, 0, 0], [1.0, 0, 0], [1.0, 0.5, 0], [1.0, 1.0, 0]]


def test_raw_actions_scaling():
    a = torch.ones(1, 2, 1, 10, dtype=torch.float64)
    raw = raw_actions(a, np.arange(5.0), np.full(5, 2.0))
    assert raw.shape == (1, 2, 2, 5)
    assert raw[0, 0, 1].tolist() == [2.0, 3.0, 4.0, 5.0, 6.0]


def test_hazard_detects_band_between_steps():
    model = CubeCostModel(
        None, boxes(), [0.0] * 2, action_mean=np.zeros(5), action_scale=np.ones(5), interp=4
    )
    model.effector = np.array([0.3, 0.08, 0.1])
    down = torch.zeros(1, 1, 2, 25, dtype=torch.float64)
    down.view(1, 1, 10, 5)[..., 1] = -2.0  # 0.1 m per step in -y: hops the band without interp
    viol, ends = model.hazard(down)
    assert float(viol[0, 0]) > 0
    assert ends.shape == (1, 1, 3, 3)
    model.effector = np.array([0.4, 0.08, 0.1])  # straight through the gap
    viol, _ = model.hazard(down)
    assert float(viol[0, 0]) == 0.0


def test_tracking_errors_zero_for_perfect_tracking():
    start = np.array([0.4, 0.0, 0.2, 0.4, 0.0, 0.02, 0.0])
    acts = np.zeros((5, 5))
    acts[:, 0] = 0.2
    raw = torch.as_tensor(acts).view(1, 1, 5, 5)
    ee = commanded_path(start[:3], raw)[0, 0].numpy()
    states = np.zeros((5, 7))
    states[:, :3] = ee
    out = tracking_errors(states, acts, start)
    assert out["one_step_xy_m"]["max"] == pytest.approx(0.0, abs=1e-12)
    assert out["block_xy_m"]["max"] == pytest.approx(0.0, abs=1e-12)


def test_outcome_metrics():
    states = np.zeros((2, 7))
    states[:, :3] = [[0.4, 0.0, 0.2], [0.42, 0.0, 0.2]]
    states[:, 3:6] = [0.3, 0.1, 0.02]
    m = outcome_metrics(
        states,
        np.array([0.4, 0.0, 0.2, 0.3, 0.1, 0.02, 0.0]),
        np.array([0.3, 0.1, 0.02]),
        np.array([0.42, 0.0, 0.2]),
        50,
        True,
    )
    assert m["success"] and m["J"] == 2.0
    assert m["final_effector_goal_m"] == pytest.approx(0.0)
    assert m["start_effector_goal_m"] == pytest.approx(0.02)


def test_design_scene_layouts_and_schedules():
    from helpers.cube14_geometry import hazard_boxes, layout
    from helpers.cube14_protocol import candidate_job, design, schedule_A, schedule_B

    spec = design()
    dmax = max(spec["stages"]["B"]["dials"])
    for lay_spec in spec["hazard"]["layouts"]:
        lay = layout(lay_spec["id"], spec)
        assert lay["pair"]["goal_row"] - lay["pair"]["start_row"] == 25
        assert len(hazard_boxes(lay, spec)) == 2
        assert gap_closed(lay, dmax)
        y0, y1 = lay["y_band"]
        assert lay["start_effector"][1] < y0 - dmax and lay["goal_effector"][1] > y1 + dmax
        assert y1 - y0 > 0.048  # the largest free effector step on this scene
    a = schedule_A(spec)
    assert len(a) == 3 * 3 + 2
    b = schedule_B(spec)
    n, k = len(spec["stages"]["B"]["seeds"]), len(spec["stages"]["B"]["dials"])
    assert len(b) == 3 * n + 3 * k * n + k * n
    job = candidate_job("C", "C2", 300, np.full(5, 0.5), spec, "t", {})
    assert job["planner"]["execute"] == "mean_safe"
    assert job["clearances"][0] == pytest.approx(0.025)
