"""Frozen hazard layouts for Push-T: placed across the nominal route, in declared families.

A layout is a virtual hazard (box or disc, pushtGeometry.py) placed across the route the
nominal planner actually takes: its swept T footprint over the root prefix and the
nominal branch. The start and goal footprints stay clear of the hazard by at least the
widest margin tested plus slack, so no test case is unsatisfiable by construction. The
generator is tuned on development cases and then frozen (pushT.md); never drop a test
case because a method fails on it.

Families for E4 transfer:
- familiar: hazard centres in one set of arena cells, one size range;
- heldout: disjoint cells and a different size range.
Starts and goals are split the same way by expert source episode (parity of a seeded
permutation), recorded with the layout file so the split is reproducible.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from helpers.pushtGeometry import (
    ARENA_HI,
    ARENA_LO,
    Box,
    Disc,
    Hazard,
    clearance_trace,
    hazard_from_dict,
    t_polygons_batch,
)

GRID = 4  # arena cells per side
CELL = (ARENA_HI - ARENA_LO) / GRID

# Cells are (row, col) with image-style y down. Familiar = checkerboard "even" cells,
# held-out = "odd" cells, so both families cover the arena but never share a cell.
FAMILIES = {
    "familiar": {"cells": [(r, c) for r in range(GRID) for c in range(GRID) if (r + c) % 2 == 0], "size": (15.0, 25.0)},
    "heldout": {"cells": [(r, c) for r in range(GRID) for c in range(GRID) if (r + c) % 2 == 1], "size": (28.0, 40.0)},
}
# The widest dial tested. Block displacement along nominal routes has median 34 px (E0
# bank), so start/goal clearance requirements above ~25 px reject most roots.
MAX_MARGIN = 20.0
SLACK = 5.0
ROUTE_CROSS_TOL = 8.0  # accept routes whose minimum clearance is at most this (near-crossing)


@dataclass
class Layout:
    hazard: dict
    family: str
    root_id: str
    route_fraction: float
    nominal_min_clearance: float
    start_clearance: float
    goal_clearance: float

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def shape(self) -> Hazard:
        return hazard_from_dict(self.hazard)


def cell_of(xy) -> tuple[int, int]:
    c = int(np.clip((xy[0] - ARENA_LO) // CELL, 0, GRID - 1))
    r = int(np.clip((xy[1] - ARENA_LO) // CELL, 0, GRID - 1))
    return (r, c)


def sample_hazard(rng, centre, size: float, kind: str) -> Hazard:
    cx, cy = float(centre[0]), float(centre[1])
    if kind == "disc":
        return Disc(cx, cy, size)
    aspect = float(rng.uniform(0.7, 1.4))
    hx, hy = size * aspect, size / aspect
    return Box(cx - hx, cx + hx, cy - hy, cy + hy)


def generate_layout(
    rng,
    route_poses: np.ndarray,
    start_pose,
    goal_pose,
    *,
    family: str,
    root_id: str = "",
    max_margin: float = MAX_MARGIN,
    slack: float = SLACK,
    tries: int = 400,
    kinds=("box", "disc"),
    route_fraction_range=(0.4, 1.0),
) -> Layout | None:
    """Place a hazard of `family` across `route_poses` (N, 3), clear of start and goal.

    Accept when: the nominal route's minimum clearance is at most ROUTE_CROSS_TOL (the
    route crosses or grazes the hazard), the hazard centre lies in a family cell, and both
    the start and goal footprints keep clearance >= max_margin + slack.
    """
    fam = FAMILIES[family]
    cells = set(map(tuple, fam["cells"]))
    lo, hi = fam["size"]
    route_poses = np.asarray(route_poses, float)
    n = len(route_poses)
    need = max_margin + slack
    best = None
    for _ in range(tries):
        f = float(rng.uniform(*route_fraction_range))
        pose = route_poses[min(n - 1, int(f * n))]
        verts = t_polygons_batch(pose[None])[0].reshape(-1, 2)
        anchor = verts[int(rng.integers(len(verts)))]
        centre = anchor + rng.normal(0.0, 12.0, size=2)
        if cell_of(centre) not in cells:
            continue
        size = float(rng.uniform(lo, hi))
        hz = sample_hazard(rng, centre, size, kinds[int(rng.integers(len(kinds)))])
        sc = float(clearance_trace(np.asarray(start_pose, float)[None], hz)[0])
        gc = float(clearance_trace(np.asarray(goal_pose, float)[None], hz)[0])
        if sc < need or gc < need:
            continue
        rc = float(clearance_trace(route_poses, hz).min())
        if rc > ROUTE_CROSS_TOL:
            continue
        lay = Layout(hz.to_dict(), family, root_id, f, rc, sc, gc)
        # prefer hazards the route crosses moderately: unsafe nominal, room for a detour
        score = -abs(rc + 8.0)
        if best is None or score > best[0]:
            best = (score, lay)
    return None if best is None else best[1]


def split_source_families(episodes: list[int], seed: int) -> dict[str, list[int]]:
    """Familiar/held-out split of expert source episodes for starts and goals."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(np.asarray(episodes))
    return {"familiar": sorted(int(e) for e in perm[::2]), "heldout": sorted(int(e) for e in perm[1::2])}


def save_layouts(path: Path, layouts: list[Layout], meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta, "layouts": [lay.to_dict() for lay in layouts]}) + "\n")


def load_layouts(path: Path) -> tuple[list[Layout], dict]:
    blob = json.loads(Path(path).read_text())
    return [Layout(**d) for d in blob["layouts"]], blob["meta"]
