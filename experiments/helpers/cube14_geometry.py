"""Hazard geometry and realized-outcome metrics for the task-14 Cube dial study.

Positions are metres in the world frame of the OGBench manipulation scene. The hazard is a
keep-out band for the gripper's pinch point (the end effector): two axis-aligned boxes in the
horizontal plane, of full workspace height, with one gap, like the TwoRoom and Reacher bands
of tasks 12 and 13. It is invisible to the world model, the arm can pass through it, and it
spans the workspace across the carry direction, so the gap is the only safe crossing. A scene
fixes one dataset start/goal pair and where the band sits; a layout places the gap. Realized
metrics use the true boxes and the effector position from saved simulator states.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SUCCESS_RADIUS_M = 0.04


def design():
    return json.loads((ROOT / "configs/cube14/design.json").read_text())


def layout(layout_id, spec=None):
    """A layout merged over its scene: band, gap and the dataset pair it is played on."""
    spec = spec or design()
    lay = next(dict(x) for x in spec["hazard"]["layouts"] if x["id"] == layout_id)
    scene = next(dict(s) for s in spec["hazard"]["scenes"] if s["id"] == lay["scene"])
    return {**scene, **lay, "scene": scene["id"]}


def hazard_boxes(layout_spec, spec=None):
    """Two true boxes (x0, x1, y0, y1): the band left of the gap and right of it."""
    spec = spec or design()
    hz = {**spec["hazard"], **layout_spec}
    y0, y1 = hz["y_band"]
    half = layout_spec["gap_width"] / 2.0
    centre = layout_spec["gap_center_x"]
    left = (hz["left_edge"], centre - half, y0, y1)
    right = (centre + half, hz["right_end"], y0, y1)
    if left[1] <= left[0] or right[1] <= right[0]:
        raise ValueError("Gap leaves an empty box")
    return [left, right]


def gap_closed(layout_spec, clearance):
    """True once the inflated boxes overlap, so no effector position can thread the gap."""
    return clearance >= layout_spec["gap_width"] / 2.0


def inside_any(xy, boxes):
    """Boolean per row of xy (N, 2): effector inside any true box."""
    xy = np.asarray(xy, dtype=float)
    hit = np.zeros(len(xy), dtype=bool)
    for x0, x1, y0, y1 in boxes:
        hit |= (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
    return hit


def signed_distance(xy, boxes):
    """Per row: distance from the effector to the nearest box, negative inside (depth)."""
    xy = np.asarray(xy, dtype=float)
    best = np.full(len(xy), np.inf)
    for x0, x1, y0, y1 in boxes:
        dx = np.maximum(np.maximum(x0 - xy[:, 0], xy[:, 0] - x1), 0.0)
        dy = np.maximum(np.maximum(y0 - xy[:, 1], xy[:, 1] - y1), 0.0)
        outside = np.hypot(dx, dy)
        depth = np.minimum(
            np.minimum(xy[:, 0] - x0, x1 - xy[:, 0]), np.minimum(xy[:, 1] - y0, y1 - xy[:, 1])
        )
        inside = (outside == 0) & (depth >= 0)
        best = np.minimum(best, np.where(inside, -depth, outside))
    return best


def band_crossing(xy, layout_spec, spec=None, start_xy=None):
    """How the effector first reached the far side of the band: 'gap', 'band' or 'none'."""
    spec = spec or design()
    y0, y1 = {**spec["hazard"], **layout_spec}["y_band"]
    xy = np.asarray(xy, dtype=float)
    start = xy[0] if start_xy is None else np.asarray(start_xy, dtype=float)
    above = start[1] > y1
    far = (xy[:, 1] < y0) if above else (xy[:, 1] > y1)
    if not far.any():
        return "none"
    first = int(np.argmax(far))
    window = xy[max(0, first - 2) : first + 1]
    in_band = window[(window[:, 1] >= y0) & (window[:, 1] <= y1)]
    # Between two saved states the effector may jump the band; use the crossing point.
    if len(in_band):
        xs = in_band[:, 0]
    else:
        a, b = (xy[first - 1], xy[first]) if first else (start, xy[first])
        mid = (y0 + y1) / 2.0
        t = (mid - a[1]) / (b[1] - a[1]) if b[1] != a[1] else 0.0
        xs = np.array([a[0] + t * (b[0] - a[0])])
    half = layout_spec["gap_width"] / 2.0
    return "gap" if np.all(np.abs(xs - layout_spec["gap_center_x"]) <= half) else "band"


def outcome_metrics(states, start_state, goal_cube, goal_effector, budget, terminated):
    """Task outcome from saved states (N, 7): effector xyz, cube xyz, gripper opening."""
    states = np.asarray(states, dtype=float)
    ee, cube = states[:, 0:3], states[:, 3:6]
    goal = np.asarray(goal_cube, dtype=float)
    goal_ee = np.asarray(goal_effector, dtype=float)
    ee_goal = np.linalg.norm(ee - goal_ee, axis=1)
    start_ee, start_cube = np.asarray(start_state[0:3]), np.asarray(start_state[3:6])
    path = np.linalg.norm(np.diff(np.vstack([start_ee, ee]), axis=0), axis=1)
    cube_goal = np.linalg.norm(cube - goal, axis=1)
    n = len(states)
    success = bool(terminated)
    return {
        "n_states": int(n),
        "success": success,
        "steps_to_success": int(n) if success else None,
        "J": float(n if success else budget),
        "start_effector_goal_m": float(np.linalg.norm(start_ee - goal_ee)),
        "start_effector_goal_xy_m": float(np.linalg.norm(start_ee[:2] - goal_ee[:2])),
        "final_effector_goal_m": float(ee_goal[-1]),
        "min_effector_goal_m": float(ee_goal.min()),
        "start_cube_goal_m": float(np.linalg.norm(start_cube - goal)),
        "final_cube_goal_m": float(cube_goal[-1]),
        "min_cube_goal_m": float(cube_goal.min()),
        "cube_moved_m": float(np.linalg.norm(cube[-1] - start_cube)),
        "cube_lift_max_m": float(cube[:, 2].max() - start_cube[2]),
        "gripper_closed_frac": float((states[:, 6] < 0.5).mean()),
        "path_length_m": float(path.sum()),
        "max_step_m": float(path.max()) if n else 0.0,
        "median_step_m": float(np.median(path)) if n else 0.0,
        "effector_z_range": [float(ee[:, 2].min()), float(ee[:, 2].max())],
    }


def realized_metrics(
    states, layout_spec, start_state, goal_cube, goal_effector, spec=None, terminated=False
):
    """Everything one hazard episode contributes, from saved states and the terminal flag."""
    spec = spec or design()
    states = np.asarray(states, dtype=float)
    budget = int(spec["environment"]["eval_budget_steps"])
    out = outcome_metrics(states, start_state, goal_cube, goal_effector, budget, terminated)
    xy = states[:, 0:2]
    boxes = hazard_boxes(layout_spec, spec)
    hit = inside_any(xy, boxes)
    distance = signed_distance(xy, boxes)
    out.update(
        {
            "hazard_steps": int(hit.sum()),
            "exposure": float(hit.mean()),
            "any_violation": bool(hit.any()),
            "first_violation_step": int(np.argmax(hit)) if hit.any() else None,
            "realized_clearance": float(distance.min()),
            "effector_x_range": [float(xy[:, 0].min()), float(xy[:, 0].max())],
            "effector_y_range": [float(xy[:, 1].min()), float(xy[:, 1].max())],
            "crossing": band_crossing(xy, layout_spec, spec, start_xy=start_state[0:2]),
        }
    )
    return out
