"""Geometric start/goal families for the E4 transfer grid.

Source trajectories are assigned to disjoint development/test roles BEFORE drawing
pairs. This filter then declares familiar versus held-out block-body start AND goal
positions on disjoint checkerboard cells, without consulting a future tape outcome.
A guard around cell boundaries avoids numerically fragile family assignments.

These are held out from additional adaptation and development, not from the released
checkpoint's pretraining or from an independent physical readout's training coverage.
"""
from __future__ import annotations

import numpy as np

SOURCE_FAMILY_PROTOCOL = {
    "version": "block-start-goal-checkerboard-v1",
    "grid": 4,
    "arena_lo": 0.0,
    "arena_hi": 512.0,
    "guard_px": 5.0,
    "familiar_cell_parity": 0,
    "heldout_cell_parity": 1,
    "state_coordinates": "block-body x/y at the branch root and goal",
    "mixed_pairs": "exclude before querying candidate branches",
    "held_out_from": "additional adaptation and development; not base pretraining or physical readout",
}


def position_cell(xy, *, grid=4, guard_px=5.0):
    xy = np.asarray(xy, float)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError("A position must be a finite x/y pair")
    if not isinstance(grid, int) or grid <= 0 or not 0 <= guard_px < 256 / grid:
        raise ValueError("Invalid source-family cell grid or guard")
    if (xy < 0).any() or (xy >= 512).any():
        return None
    width = 512 / grid
    local = xy % width
    if (local < guard_px).any() or (local > width - guard_px).any():
        return None
    col, row = (xy // width).astype(int)
    return int(row), int(col)


def geometric_source_family(start_state, goal_state, *, grid=4, guard_px=5.0):
    """Return a family only if both root and goal occupy its allowed cells."""
    start, goal = np.asarray(start_state, float), np.asarray(goal_state, float)
    if start.shape != (7,) or goal.shape != (7,):
        raise ValueError("Push-T source families require 7-d start and goal states")
    cells = [position_cell(st[2:4], grid=grid, guard_px=guard_px) for st in (start, goal)]
    if any(cell is None for cell in cells):
        return None
    parity = [(row + col) % 2 for row, col in cells]
    if parity[0] != parity[1]:
        return None
    return "familiar" if parity[0] == 0 else "heldout"


def validate_geometric_bank(bank, *, expected_family=None):
    """Check stored family labels against actual root/goal positions, without physics."""
    counts = {"familiar": 0, "heldout": 0}
    for root in bank.roots:
        meta = root.meta
        family = geometric_source_family(meta["state_at_root"], root.goal_state)
        if family is None or meta.get("source_family") != family:
            raise ValueError(f"Root {root.root_id} has an invalid geometric source-family label")
        if meta.get("source_family_protocol") != SOURCE_FAMILY_PROTOCOL["version"]:
            raise ValueError(f"Root {root.root_id} lacks the geometric source protocol")
        if expected_family is not None and family != expected_family:
            raise ValueError(f"Root {root.root_id} violates the {expected_family}-only data role")
        counts[family] += 1
    return counts
