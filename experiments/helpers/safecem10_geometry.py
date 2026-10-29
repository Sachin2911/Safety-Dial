"""Gate geometry and realized-outcome metrics for the task-10 Push-T dial study.

Boxes are axis-aligned (x0, x1, y0, y1) tuples in arena pixels. A gate is two boxes
in one horizontal band with a gap between them. Realized metrics always use the true
boxes and the pusher centre from saved simulator states, never the inflated boxes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARENA = (0.0, 512.0)


def design():
    return json.loads((ROOT / "configs/safecem10/design.json").read_text())


def layout(layout_id, spec=None):
    spec = spec or design()
    return next(dict(x) for x in spec["gate"]["layouts"] if x["id"] == layout_id)


def gate_boxes(layout_spec, spec=None):
    """Two true hazard boxes: left pillar from the arena edge, right pillar to its end."""
    spec = spec or design()
    gate = spec["gate"]
    y0, y1 = gate["y_band"]
    half = layout_spec["gap_width"] / 2.0
    centre = layout_spec["gap_center_x"]
    left = (gate["left_edge"], centre - half, y0, y1)
    right = (centre + half, gate["right_pillar_end"], y0, y1)
    if left[1] <= left[0] or right[1] <= right[0]:
        raise ValueError("Gap leaves an empty pillar")
    return [left, right]


def gap_closed(layout_spec, clearance):
    """True once the inflated pillars overlap, so no centre position can thread the gap."""
    return clearance >= layout_spec["gap_width"] / 2.0


def expand_box(box, margin):
    x0, x1, y0, y1 = box
    return (x0 - margin, x1 + margin, y0 - margin, y1 + margin)


def inside_any(xy, boxes):
    """Boolean per row of xy (N, 2): pusher centre inside any true box."""
    xy = np.asarray(xy, dtype=float)
    hit = np.zeros(len(xy), dtype=bool)
    for x0, x1, y0, y1 in boxes:
        hit |= (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
    return hit


def out_of_arena(xy, arena=ARENA):
    xy = np.asarray(xy, dtype=float)
    return (xy < arena[0]).any(axis=1) | (xy > arena[1]).any(axis=1)


def band_crossing(states, layout_spec, spec=None):
    """How the pusher first crossed the wall band: 'gap', 'detour', 'pillar' or 'none'."""
    spec = spec or design()
    y0, y1 = spec["gate"]["y_band"]
    xy = np.asarray(states)[:, 0:2]
    below = xy[:, 1] > y1
    if not below.any():
        return "none"
    first = int(np.argmax(below))
    # Inspect the states inside or adjacent to the band before the first crossing.
    window = xy[max(0, first - 3) : first + 1]
    in_band = window[(window[:, 1] >= y0 - 1) & (window[:, 1] <= y1 + 1)]
    probe = in_band if len(in_band) else window[-1:]
    half = layout_spec["gap_width"] / 2.0
    centre = layout_spec["gap_center_x"]
    xs = probe[:, 0]
    if np.all(np.abs(xs - centre) <= half):
        return "gap"
    if np.all(xs > spec["gate"]["right_pillar_end"]):
        return "detour"
    return "pillar"


def realized_metrics(states, layout_spec, spec=None, terminated=False):
    """Everything one episode contributes, from saved states and the terminal flag."""
    spec = spec or design()
    env = spec["environment"]
    states = np.asarray(states, dtype=float)
    boxes = gate_boxes(layout_spec, spec)
    pusher = states[:, 0:2]
    block = states[:, 2:4]
    hit = inside_any(pusher, boxes)
    oob = out_of_arena(pusher, tuple(env["arena"]))
    n = len(states)
    goal_block = np.asarray(env["goal_block"])
    start_block = np.asarray(env["start_block"])
    budget = int(env["eval_budget_steps"])
    success = bool(terminated)
    steps = n if success else budget
    return {
        "n_states": int(n),
        "success": success,
        "steps_to_success": int(n) if success else None,
        "J": float(steps),
        "hazard_steps": int(hit.sum()),
        "exposure": float(hit.mean()),
        "any_violation": bool(hit.any()),
        "first_violation_step": int(np.argmax(hit)) if hit.any() else None,
        "oob_steps": int(oob.sum()),
        "oob_frac": float(oob.mean()),
        "final_block_err_px": float(np.linalg.norm(block[-1] - goal_block)),
        "block_moved_px": float(np.linalg.norm(block[-1] - start_block)),
        "min_block_err_px": float(np.linalg.norm(block - goal_block, axis=1).min()),
        "pusher_x_range": [float(pusher[:, 0].min()), float(pusher[:, 0].max())],
        "pusher_y_range": [float(pusher[:, 1].min()), float(pusher[:, 1].max())],
        "crossing": band_crossing(states, layout_spec, spec),
        "px_per_step": float(np.median(np.linalg.norm(np.diff(pusher, axis=0), axis=1)))
        if n > 1
        else 0.0,
    }


def detour_length(start, goal, boxes, margin):
    """Shortest corner-to-corner route around every inflated box, a planning bound."""
    s = np.asarray(start, dtype=float)
    g = np.asarray(goal, dtype=float)
    corners = []
    for box in boxes:
        x0, x1, y0, y1 = expand_box(box, margin)
        corners += [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
    direct = float(np.linalg.norm(g - s))
    best = direct
    for c in corners:
        c = np.asarray(c)
        best = min(best, float(np.linalg.norm(c - s) + np.linalg.norm(g - c)))
    return {"direct": direct, "best_corner_route": best}
