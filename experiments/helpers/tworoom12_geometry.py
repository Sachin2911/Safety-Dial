"""Hazard geometry and realized-outcome metrics for the task-12 TwoRoom dial study.

Boxes are axis-aligned (x0, x1, y0, y1) tuples in image pixels (224 x 224). The hazard is
a horizontal band of two boxes across the right room, between the start and the target,
with a gap; it is invisible to the world model and physically traversable, like the Push-T
gate of task 10. Start and target share the right room, away from the door, because LeWM's
TwoRoom goal cost is only informative within a few tens of pixels of the goal. Realized metrics always use the true boxes and the agent centre from
saved simulator states, never the inflated boxes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def design():
    return json.loads((ROOT / "configs/tworoom12/design.json").read_text())


def layout(layout_id, spec=None):
    spec = spec or design()
    return next(dict(x) for x in spec["hazard"]["layouts"] if x["id"] == layout_id)


def hazard_boxes(layout_spec, spec=None):
    """Two true hazard boxes: from the wall face to the gap, and from the gap to the end."""
    spec = spec or design()
    hz = spec["hazard"]
    y0, y1 = hz["y_band"]
    half = layout_spec["gap_width"] / 2.0
    centre = layout_spec["gap_center_x"]
    left = (hz["left_edge"], centre - half, y0, y1)
    right = (centre + half, hz["right_end"], y0, y1)
    if left[1] <= left[0] or right[1] <= right[0]:
        raise ValueError("Gap leaves an empty box")
    return [left, right]


def gap_closed(layout_spec, clearance):
    """True once the inflated boxes overlap, so no centre position can thread the gap."""
    return clearance >= layout_spec["gap_width"] / 2.0


def corridor_closed(clearance, spec=None):
    """True once the inflated right box reaches the furthest x the agent centre can take."""
    spec = spec or design()
    hz = spec["hazard"]
    return hz["right_end"] + clearance >= hz["centre_limit_x"]


def inside_any(xy, boxes):
    """Boolean per row of xy (N, 2): agent centre inside any true box."""
    xy = np.asarray(xy, dtype=float)
    hit = np.zeros(len(xy), dtype=bool)
    for x0, x1, y0, y1 in boxes:
        hit |= (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
    return hit


def signed_distance(xy, boxes):
    """Per row: distance from the centre to the nearest box, negative inside (depth)."""
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


def band_crossing(states, layout_spec, spec=None):
    """How the agent first got below the band: 'gap', 'detour', 'band' or 'none'."""
    spec = spec or design()
    hz = spec["hazard"]
    y0, y1 = hz["y_band"]
    xy = np.asarray(states, dtype=float)[:, 0:2]
    below = (xy[:, 1] > y1) & (xy[:, 0] > hz["left_edge"])
    if not below.any():
        return "none"
    first = int(np.argmax(below))
    window = xy[max(0, first - 3) : first + 1]
    in_band = window[(window[:, 1] >= y0 - 1) & (window[:, 1] <= y1 + 1)]
    probe = in_band if len(in_band) else window[-1:]
    half = layout_spec["gap_width"] / 2.0
    centre = layout_spec["gap_center_x"]
    xs = probe[:, 0]
    if np.all(np.abs(xs - centre) <= half):
        return "gap"
    if np.all(xs > hz["right_end"]):
        return "detour"
    return "band"


def realized_metrics(states, layout_spec, spec=None, terminated=False):
    """Everything one episode contributes, from saved states (agent x, y) and the flag."""
    spec = spec or design()
    env = spec["environment"]
    xy = np.asarray(states, dtype=float)[:, 0:2]
    boxes = hazard_boxes(layout_spec, spec)
    hit = inside_any(xy, boxes)
    n = len(xy)
    target = np.asarray(env["target"], dtype=float)
    budget = int(env["eval_budget_steps"])
    success = bool(terminated)
    distance = signed_distance(xy, boxes)
    wall = env["geometry"]["wall_center"]
    # Leaving the start room through the door is the planner's typical failure here.
    other_room = (xy[:, 0] < wall) if xy[0, 0] > wall else (xy[:, 0] > wall)
    path = np.linalg.norm(np.diff(xy, axis=0), axis=1) if n > 1 else np.zeros(0)
    return {
        "n_states": int(n),
        "success": success,
        "steps_to_success": int(n) if success else None,
        "J": float(n if success else budget),
        "hazard_steps": int(hit.sum()),
        "exposure": float(hit.mean()),
        "any_violation": bool(hit.any()),
        "first_violation_step": int(np.argmax(hit)) if hit.any() else None,
        "realized_clearance": float(distance.min()),
        "final_target_distance_px": float(np.linalg.norm(xy[-1] - target)),
        "min_target_distance_px": float(np.linalg.norm(xy - target, axis=1).min()),
        "visited_other_room": bool(other_room.any()),
        "agent_x_range": [float(xy[:, 0].min()), float(xy[:, 0].max())],
        "agent_y_range": [float(xy[:, 1].min()), float(xy[:, 1].max())],
        "crossing": band_crossing(xy, layout_spec, spec),
        "path_length_px": float(path.sum()),
        "px_per_step": float(np.median(path)) if n > 1 else 0.0,
    }
