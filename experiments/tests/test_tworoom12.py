"""Task-12 unit tests (TwoRoom SafeCEM): geometry, exact check, cost model, schedules."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.tworoom12_geometry import (  # noqa: E402
    band_crossing,
    corridor_closed,
    design,
    gap_closed,
    hazard_boxes,
    inside_any,
    layout,
    realized_metrics,
    signed_distance,
)
from helpers.tworoom12_planner import (  # noqa: E402
    TwoRoomCostModel,
    raw_actions,
    replay_error,
    tworoom_path,
)
from helpers.tworoom12_protocol import (  # noqa: E402
    candidate_job,
    evaluation_seeds,
    schedule_A,
    schedule_B,
)

SPEC = design()
GEOM = SPEC["environment"]["geometry"]


def test_boxes_and_closure():
    lay = layout("T2", SPEC)
    left, right = hazard_boxes(lay, SPEC)
    assert left == (117.0, 140.0, 141.0, 149.0)
    assert right == (164.0, 210.0, 141.0, 149.0)
    assert not gap_closed(lay, 11.9) and gap_closed(lay, 12.0)
    assert corridor_closed(0.0, SPEC)  # the band spans the room: no way round it


def test_signed_distance_and_inside():
    boxes = [(0.0, 10.0, 0.0, 10.0)]
    xy = np.array([[5.0, 5.0], [13.0, 14.0], [10.0, 5.0]])
    d = signed_distance(xy, boxes)
    assert d[0] == pytest.approx(-5.0)
    assert d[1] == pytest.approx(5.0)
    assert d[2] == pytest.approx(0.0)
    assert inside_any(xy, boxes).tolist() == [True, False, True]


def test_path_free_motion_and_clipping():
    acts = torch.tensor([[1.0, 0.0], [2.0, -3.0], [0.5, 0.5]], dtype=torch.float64).view(1, 1, 3, 2)
    path = tworoom_path([40.0, 100.0], acts, GEOM)[0, 0].numpy()
    assert np.allclose(path, [[45.0, 100.0], [50.0, 95.0], [52.5, 97.5]])


def test_path_wall_blocks_outside_door_and_passes_inside():
    # Below the door: pushing right ends at the effective face (112 - 5 - 7) minus 0.5.
    acts = torch.ones(1, 1, 20, 2, dtype=torch.float64)
    acts[..., 1] = 0.0
    blocked = tworoom_path([90.0, 112.0], acts, GEOM)[0, 0].numpy()
    # The simulator's test is strict, so the face itself (x = 100) is allowed once.
    assert blocked[:, 0].max() == pytest.approx(100.0)
    assert blocked[-1, 0] == pytest.approx(99.5)
    # At door height the agent crosses into the right room.
    passed = tworoom_path([90.0, 49.0], acts, GEOM)[0, 0].numpy()
    assert passed[-1, 0] > 120.0


def test_path_border_clamp():
    acts = -torch.ones(1, 1, 30, 2, dtype=torch.float64)
    path = tworoom_path([40.0, 40.0], acts, GEOM)[0, 0].numpy()
    assert path[-1].tolist() == pytest.approx([21.0, 21.0])


def test_replay_error_zero_on_own_path():
    acts = np.tile([[0.6, -0.8]], (12, 1))
    path = tworoom_path([60.0, 112.0], torch.as_tensor(acts).view(1, 1, -1, 2), GEOM)[0, 0]
    err = replay_error(path.numpy(), [60.0, 112.0], acts, GEOM)
    assert err["max"] == pytest.approx(0.0, abs=1e-9)


class _Base:
    def get_cost(self, info, cands):
        return torch.zeros(cands.shape[0], cands.shape[1], dtype=cands.dtype)


def _model(clearance, tol=0.0):
    lay = layout("T2", SPEC)
    return TwoRoomCostModel(
        _Base(),
        hazard_boxes(lay, SPEC),
        [clearance] * 5,
        geometry=GEOM,
        action_mean=np.zeros(2),
        action_scale=np.ones(2),
        tol=tol,
        track_best=True,
    )


def test_exact_violation_detects_band_and_respects_gap():
    model = _model(0.0)
    down = torch.zeros(1, 2, 5, 10, dtype=torch.float64)
    down[0, 0, :, 1::2] = 1.0  # straight down through the gap at x = 152
    model.pusher_xy = np.array([152.0, 125.0])
    v_gap = model.exact_violation(down)[0, 0]
    assert float(v_gap) == pytest.approx(0.0)
    model.pusher_xy = np.array([175.0, 125.0])  # straight down through the right box
    v_box = model.exact_violation(down)[0, 0]
    assert float(v_box) > 0.0
    wide = _model(14.0)  # past closure: even the gap path violates
    wide.pusher_xy = np.array([152.0, 125.0])
    assert float(wide.exact_violation(down)[0, 0]) > 0.0


def test_get_cost_ranks_feasible_first_and_tracks_best():
    model = _model(0.0)
    model.pusher_xy = np.array([175.0, 125.0])
    cands = torch.zeros(1, 2, 5, 10, dtype=torch.float64)
    cands[0, 0, :, 1::2] = 1.0  # down through the box: infeasible
    cands[0, 1, :, 0::2] = 1.0  # right along the top: feasible
    info = {"pixels": torch.zeros(1, 1, 1, 3, 4, 4)}
    model.begin_solve()
    total = model.get_cost(info, cands)
    assert float(total[0, 1]) < float(total[0, 0])
    best = model.best_actions()
    assert torch.equal(best[0], cands[0, 1])


def test_raw_actions_scaling():
    cands = torch.ones(1, 1, 2, 10)
    raw = raw_actions(cands, [0.1, -0.1], [0.5, 2.0])
    assert raw.shape == (1, 1, 10, 2)
    assert torch.allclose(raw[0, 0, 0], torch.tensor([0.6, 1.9]))


def test_metrics_and_crossing():
    lay = layout("T2", SPEC)
    through_gap = np.array(
        [[164, 120], [158, 128], [152, 136], [152, 145], [152, 152], [158, 160], [164, 166]], float
    )
    m = realized_metrics(through_gap, lay, SPEC, terminated=True)
    assert m["success"] and m["J"] == 7.0 and not m["any_violation"]
    assert m["crossing"] == "gap" and not m["visited_other_room"]
    through_box = through_gap.copy()
    through_box[2:5, 0] = 175.0
    m2 = realized_metrics(through_box, lay, SPEC, terminated=False)
    assert m2["any_violation"] and m2["J"] == float(SPEC["environment"]["eval_budget_steps"])
    assert band_crossing(through_box, lay, SPEC) == "band"


def test_schedules():
    a = schedule_A(SPEC)
    assert len(a) == 3 * 3 + 2
    assert len({j["id"] for j in a}) == len(a)
    assert all(j["planner"]["execute"] == "mean" for j in a if j["mode"] == "penalty")
    b = schedule_B(SPEC)
    n_dials = len(SPEC["stages"]["B"]["dials"])
    assert len(b) == 30 + 3 * n_dials * 10 + 1 * n_dials * 10
    assert len({j["id"] for j in b}) == len(b)
    assert {j["condition"] for j in b} == {"free", "safe", "mean"}
    assert all(j["planner"]["execute"] == "mean_safe" for j in b if j["condition"] == "safe")


def test_candidate_job_uses_pipeline_and_genes():
    x = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    job = candidate_job("C", "T2", 300, x, SPEC, "nsga2-r0-g00-c00", {})
    assert job["planner"]["check"] == "exact" and job["planner"]["execute"] == "mean_safe"
    assert job["clearances"][0] == pytest.approx(SPEC["stages"]["C"]["genes"]["d0"]["high"] / 2)
    assert job["planner"]["feasible_tol"] == pytest.approx(3.5)
    seeds = evaluation_seeds(SPEC, "nsga2", 2, 7)
    assert not set(seeds) & set(SPEC["stages"]["C"]["final_seeds"])
