"""Hazard geometry and realized-outcome metrics for the task-13 Reacher dial study.

Positions are metres in the arm plane, with the shoulder at the origin. The fingertip of
the two-link arm is at l1 (cos q1, sin q1) + l2 (cos(q1 + q2), sin(q1 + q2)). The hazard is
a band of two axis-aligned boxes across the fingertip's route with one gap, like the
TwoRoom band of task 12: it is invisible to the world model, the arm can pass through it,
and it spans the arm's whole reach, so the gap is the only safe crossing. Realized metrics
use the true boxes and the fingertip from saved simulator states, never inflated boxes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def design():
    return json.loads((ROOT / "configs/reacher13/design.json").read_text())


def layout(layout_id, spec=None):
    spec = spec or design()
    return next(dict(x) for x in spec["hazard"]["layouts"] if x["id"] == layout_id)


def fingertip(qpos, links=(0.12, 0.12)):
    """Fingertip (N, 2) from joint positions (N, 2), numpy."""
    q = np.asarray(qpos, dtype=np.float64).reshape(-1, 2)
    l1, l2 = links
    a = q[:, 0]
    b = q[:, 0] + q[:, 1]
    return np.stack([l1 * np.cos(a) + l2 * np.cos(b), l1 * np.sin(a) + l2 * np.sin(b)], axis=1)


def hazard_boxes(layout_spec, spec=None):
    """Two true hazard boxes: from the band's left edge to the gap, and from the gap on."""
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
    """True once the inflated boxes overlap, so no fingertip position can thread the gap."""
    return clearance >= layout_spec["gap_width"] / 2.0


def inside_any(xy, boxes):
    """Boolean per row of xy (N, 2): fingertip inside any true box."""
    xy = np.asarray(xy, dtype=float)
    hit = np.zeros(len(xy), dtype=bool)
    for x0, x1, y0, y1 in boxes:
        hit |= (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
    return hit


def signed_distance(xy, boxes):
    """Per row: distance from the fingertip to the nearest box, negative inside (depth)."""
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
    """How the fingertip first reached the far side of the band: 'gap', 'band' or 'none'."""
    spec = spec or design()
    y0, y1 = spec["hazard"]["y_band"]
    xy = np.asarray(xy, dtype=float)
    start = xy[0] if start_xy is None else np.asarray(start_xy, dtype=float)
    above = start[1] > y1
    far = (xy[:, 1] < y0) if above else (xy[:, 1] > y1)
    if not far.any():
        return "none"
    first = int(np.argmax(far))
    window = xy[max(0, first - 2) : first + 1]
    in_band = window[(window[:, 1] >= y0) & (window[:, 1] <= y1)]
    # Between two saved states the fingertip may jump the band; use the crossing point.
    if len(in_band):
        xs = in_band[:, 0]
    else:
        a, b = (xy[first - 1], xy[first]) if first else (start, xy[first])
        mid = (y0 + y1) / 2.0
        t = (mid - a[1]) / (b[1] - a[1]) if b[1] != a[1] else 0.0
        xs = np.array([a[0] + t * (b[0] - a[0])])
    half = layout_spec["gap_width"] / 2.0
    return "gap" if np.all(np.abs(xs - layout_spec["gap_center_x"]) <= half) else "band"


def realized_metrics(states, layout_spec, spec=None, terminated=False):
    """Everything one episode contributes, from saved states and the terminal flag.

    Each state row is (q1, q2, v1, v2, finger_x, finger_y) after one env step."""
    spec = spec or design()
    env = spec["environment"]
    states = np.asarray(states, dtype=float)
    qpos = states[:, 0:2]
    xy = states[:, 4:6]
    boxes = hazard_boxes(layout_spec, spec)
    hit = inside_any(xy, boxes)
    n = len(xy)
    target_q = np.asarray(env["target_qpos"], dtype=float)
    target_xy = fingertip(target_q, env["link_lengths"])[0]
    start_xy = fingertip(env["start_qpos"], env["link_lengths"])[0]
    budget = int(env["eval_budget_steps"])
    success = bool(terminated)
    distance = signed_distance(xy, boxes)
    joint_error = np.abs(qpos - target_q).max(axis=1)
    path = np.linalg.norm(np.diff(np.vstack([start_xy, xy]), axis=0), axis=1)
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
        "final_joint_error_rad": float(joint_error[-1]),
        "min_joint_error_rad": float(joint_error.min()),
        "final_target_distance_m": float(np.linalg.norm(xy[-1] - target_xy)),
        "min_target_distance_m": float(np.linalg.norm(xy - target_xy, axis=1).min()),
        "finger_x_range": [float(xy[:, 0].min()), float(xy[:, 0].max())],
        "finger_y_range": [float(xy[:, 1].min()), float(xy[:, 1].max())],
        "crossing": band_crossing(xy, layout_spec, spec, start_xy=start_xy),
        "path_length_m": float(path.sum()),
        "max_step_m": float(path.max()) if n else 0.0,
    }
