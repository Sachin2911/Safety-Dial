"""Unit tests for the task-13 Reacher dial study (geometry, exact check, schedules)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.reacher13_geometry import (  # noqa: E402
    band_crossing,
    fingertip,
    gap_closed,
    hazard_boxes,
    inside_any,
    layout,
    realized_metrics,
    signed_distance,
)
from helpers.reacher13_protocol import (  # noqa: E402
    candidate_job,
    design,
    schedule_A,
    schedule_B,
)

SPEC = design()


def test_layouts_boxes_and_closure():
    for lay in SPEC["hazard"]["layouts"]:
        boxes = hazard_boxes(lay, SPEC)
        assert len(boxes) == 2
        left, right = boxes
        assert left[1] < right[0]
        assert right[0] - left[1] == pytest.approx(lay["gap_width"])
        assert not gap_closed(lay, lay["gap_width"] / 2 - 1e-6)
        assert gap_closed(lay, lay["gap_width"] / 2)


def test_start_and_target_clear_of_band_and_band_thicker_than_steps():
    env = SPEC["environment"]
    start = fingertip(env["start_qpos"], env["link_lengths"])[0]
    target = fingertip(env["target_qpos"], env["link_lengths"])[0]
    y0, y1 = SPEC["hazard"]["y_band"]
    assert start[1] > y1 and target[1] < y0
    assert y1 - y0 > 0.034  # the largest free fingertip step in the diagnosis


def test_fingertip_matches_links():
    assert fingertip([0.0, 0.0])[0] == pytest.approx([0.24, 0.0])
    assert fingertip([np.pi / 2, 0.0])[0] == pytest.approx([0.0, 0.24], abs=1e-12)


def test_signed_distance_and_inside():
    boxes = [(0.0, 0.1, 0.0, 0.1)]
    xy = np.array([[0.05, 0.05], [0.15, 0.05], [0.05, 0.12]])
    assert inside_any(xy, boxes).tolist() == [True, False, False]
    d = signed_distance(xy, boxes)
    assert d[0] == pytest.approx(-0.05)
    assert d[1] == pytest.approx(0.05)
    assert d[2] == pytest.approx(0.02)


def test_crossing_classification():
    lay = layout("R0", SPEC)
    centre = lay["gap_center_x"]
    through_gap = np.array([[centre, 0.12], [centre, 0.07], [centre, 0.0]])
    assert band_crossing(through_gap, lay, SPEC) == "gap"
    through_band = np.array([[0.0, 0.12], [0.0, 0.07], [0.0, 0.0]])
    assert band_crossing(through_band, lay, SPEC) == "band"
    stays = np.array([[0.1, 0.15], [0.1, 0.14]])
    assert band_crossing(stays, lay, SPEC) == "none"


def test_realized_metrics_counts_true_box_states():
    lay = layout("R2", SPEC)
    states = np.zeros((3, 6))
    states[:, 4:6] = [[0.0, 0.12], [0.0, 0.07], [0.0, 0.0]]
    m = realized_metrics(states, lay, SPEC, terminated=False)
    assert m["hazard_steps"] == 1
    assert m["exposure"] == pytest.approx(1 / 3)
    assert m["J"] == SPEC["environment"]["eval_budget_steps"]
    assert m["crossing"] == "band"


def test_schedules():
    a = schedule_A(SPEC)
    assert len(a) == 3 * 3 + 2
    b = schedule_B(SPEC)
    stage = SPEC["stages"]["B"]
    n = len(stage["seeds"])
    dials = len(stage["dials"])
    assert len(b) == 3 * n + 3 * dials * n + dials * n
    assert len({j["id"] for j in b}) == len(b)


def test_candidate_job_uses_pipeline_and_genes():
    x = np.full(5, 0.5)
    job = candidate_job("C", "R2", 300, x, SPEC, "t", {})
    assert job["planner"]["execute"] == "mean_safe"
    assert len(job["clearances"]) == job["planner"]["horizon"]
    assert job["clearances"][0] == pytest.approx(0.015)


arm_needed = pytest.importorskip


def test_exact_arm_model_replays_env_and_detects_band():
    arm_needed("dm_control")
    from helpers.reacher13_planner import ArmModel, ReacherCostModel, replay_error

    arm = ArmModel(threads=1)
    env = SPEC["environment"]
    state = np.array([*env["start_qpos"], 0.0, 0.0])
    rng = np.random.default_rng(0)
    acts = rng.uniform(-1, 1, size=(1, 10, 2))
    path = arm.paths(state, acts)[0]
    states = np.zeros((10, 6))
    states[:, 4:6] = path
    assert replay_error(states, state, acts[0], arm)["max"] == pytest.approx(0.0, abs=1e-12)
    lay = layout("R2", SPEC)
    model = ReacherCostModel(
        None,
        hazard_boxes(lay, SPEC),
        [0.0] * 2,
        arm=arm,
        action_mean=np.zeros(2),
        action_scale=np.ones(2),
    )
    model.arm_state = state
    cands = torch.as_tensor(acts.reshape(1, 1, 2, 10))
    viol = model.exact_violation(cands)
    xy = path
    hit = inside_any(xy, hazard_boxes(lay, SPEC)).any()
    assert bool(viol[0, 0] > 0) == bool(hit)
