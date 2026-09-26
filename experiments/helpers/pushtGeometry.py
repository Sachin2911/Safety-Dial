"""Push-T geometry: the whole-T footprint, virtual hazards, signed clearance, overlays.

Pinned facts (pushT.md): the T is two `pymunk.Poly` shapes on one body. In body
coordinates the bar spans x in [-60, 60], y in [0, 30]; the stem spans x in [-15, 15],
y in [30, 120]. The block xy in the 7-d state is the BODY POSITION, not the centre of
mass (COG is (0, 45) in body coordinates). World vertex = position + R(angle) @ v, which
is what `pymunk.Body.local_to_world` computes; `t_polygons` is checked against it in
`pusht_e0_geometry.py`.

A hazard is a virtual region in arena pixels: an axis-aligned box or a disc. It changes
the rule, not the dynamics. `clearance` is a signed distance in pixels, positive when the
footprint is outside the hazard and negative (penetration depth) when it intersects.
For a disc the value is exact. For a box the negative side is the separating-axis
minimum translation depth, which is the usual convex penetration measure and is
declared as such in the thesis.

Arena: pixels 0..512, image-style y down, rendered at 224 px (scale 224/512).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

T_BAR = np.array([(-60.0, 0.0), (60.0, 0.0), (60.0, 30.0), (-60.0, 30.0)])
T_STEM = np.array([(-15.0, 30.0), (15.0, 30.0), (15.0, 120.0), (-15.0, 120.0)])
T_LOCAL = (T_BAR, T_STEM)
PUSHER_RADIUS = 15.0
ARENA_LO, ARENA_HI = 0.0, 512.0
RENDER_SCALE = 224.0 / 512.0


# --------------------------------------------------------------------------------------
# footprint
# --------------------------------------------------------------------------------------
def polygons_from_env(env) -> tuple[np.ndarray, ...]:
    """Local vertices read from the live pymunk shapes (never hard-coded in production)."""
    body = env.unwrapped.block
    return tuple(np.array([tuple(v) for v in s.get_vertices()], dtype=float) for s in body.shapes)


def rotation(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def t_polygons(pose, local=T_LOCAL) -> list[np.ndarray]:
    """World-frame polygons (each (4, 2)) for pose = (x, y, angle)."""
    x, y, ang = float(pose[0]), float(pose[1]), float(pose[2])
    R = rotation(ang)
    return [np.asarray(v) @ R.T + np.array([x, y]) for v in local]


def t_polygons_batch(poses: np.ndarray, local=T_LOCAL) -> np.ndarray:
    """(N, 3) poses -> (N, n_shapes, 4, 2) world vertices."""
    poses = np.asarray(poses, dtype=float)
    c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
    out = np.empty((len(poses), len(local), 4, 2))
    for k, v in enumerate(local):
        vx, vy = v[:, 0][None, :], v[:, 1][None, :]
        out[:, k, :, 0] = c[:, None] * vx - s[:, None] * vy + poses[:, 0:1]
        out[:, k, :, 1] = s[:, None] * vx + c[:, None] * vy + poses[:, 1:2]
    return out


def pose_from_state(state: np.ndarray) -> np.ndarray:
    """7-d state(s) -> (x, y, angle) of the block."""
    state = np.asarray(state)
    return state[..., 2:5]


# --------------------------------------------------------------------------------------
# hazards
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Box:
    x0: float
    x1: float
    y0: float
    y1: float

    @property
    def vertices(self) -> np.ndarray:
        return np.array([(self.x0, self.y0), (self.x1, self.y0), (self.x1, self.y1), (self.x0, self.y1)], dtype=float)

    @property
    def centre(self) -> tuple[float, float]:
        return (0.5 * (self.x0 + self.x1), 0.5 * (self.y0 + self.y1))

    def to_dict(self) -> dict:
        return {"kind": "box", "x0": self.x0, "x1": self.x1, "y0": self.y0, "y1": self.y1}


@dataclass(frozen=True)
class Disc:
    cx: float
    cy: float
    r: float

    @property
    def centre(self) -> tuple[float, float]:
        return (self.cx, self.cy)

    def to_dict(self) -> dict:
        return {"kind": "disc", "cx": self.cx, "cy": self.cy, "r": self.r}


Hazard = Box | Disc


def hazard_from_dict(d: dict) -> Hazard:
    if d["kind"] == "box":
        return Box(d["x0"], d["x1"], d["y0"], d["y1"])
    if d["kind"] == "disc":
        return Disc(d["cx"], d["cy"], d["r"])
    raise ValueError(f"unknown hazard kind {d['kind']}")


# --------------------------------------------------------------------------------------
# distances
# --------------------------------------------------------------------------------------
def _edges(poly: np.ndarray) -> np.ndarray:
    poly = np.asarray(poly, dtype=float)
    return np.roll(poly, -1, axis=0) - poly


def _point_segment_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = float(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-12))
    t = min(1.0, max(0.0, t))
    return float(np.linalg.norm(p - (a + t * ab)))


def point_in_convex(p: np.ndarray, poly: np.ndarray) -> bool:
    """Convex polygon with consistent winding (either orientation)."""
    e = _edges(poly)
    cross = e[:, 0] * (p[1] - poly[:, 1]) - e[:, 1] * (p[0] - poly[:, 0])
    return bool(np.all(cross >= 0) or np.all(cross <= 0))


def signed_point_polygon(p: np.ndarray, poly: np.ndarray) -> float:
    """Signed distance from a point to a convex polygon: negative inside."""
    d = min(_point_segment_dist(p, poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly)))
    return -d if point_in_convex(p, poly) else d


def _sat_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Minimum overlap over the edge normals of both polygons; <= 0 when separated.

    Returns the positive penetration depth when they intersect, otherwise a negative
    number whose magnitude is the largest separating gap along an axis (NOT the true
    distance; use `_convex_distance` for that).
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    best = np.inf
    for poly in (a, b):
        e = _edges(poly)
        normals = np.stack([-e[:, 1], e[:, 0]], axis=1)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        pa = a @ normals.T  # (4, n)
        pb = b @ normals.T
        overlap = np.minimum(pa.max(0) - pb.min(0), pb.max(0) - pa.min(0))
        m = float(overlap.min())
        if m < best:
            best = m
    return best


def _convex_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distance between two separated convex polygons (vertex-to-edge both ways)."""
    best = np.inf
    for p, q in ((a, b), (b, a)):
        for v in p:
            for i in range(len(q)):
                best = min(best, _point_segment_dist(v, q[i], q[(i + 1) % len(q)]))
    return float(best)


def polygon_clearance(poly: np.ndarray, hazard: Hazard) -> float:
    if isinstance(hazard, Disc):
        return signed_point_polygon(np.array([hazard.cx, hazard.cy]), poly) - hazard.r
    ov = _sat_overlap(poly, hazard.vertices)
    if ov > 0:
        return -ov
    return _convex_distance(poly, hazard.vertices)


def t_clearance(pose, hazard: Hazard, local=T_LOCAL) -> float:
    """Signed clearance of the whole T footprint (min over its shapes)."""
    return min(polygon_clearance(p, hazard) for p in t_polygons(pose, local))


def clearance_trace(poses: np.ndarray, hazard: Hazard, local=T_LOCAL) -> np.ndarray:
    """(N, 3) poses -> (N,) signed clearance."""
    poses = np.asarray(poses, dtype=float)
    return np.array([t_clearance(p, hazard, local) for p in poses])


def unsafe(poses: np.ndarray, hazard: Hazard, local=T_LOCAL) -> bool:
    """U = 1 if the T intersects the hazard at ANY pose in the trace."""
    return bool((clearance_trace(poses, hazard, local) < 0).any())


def pusher_clearance(xy, hazard: Hazard) -> float:
    """Signed clearance of the pusher disc (radius 15) for reference checks."""
    p = np.asarray(xy, dtype=float)
    if isinstance(hazard, Disc):
        return float(np.linalg.norm(p - np.array(hazard.centre))) - hazard.r - PUSHER_RADIUS
    return signed_point_polygon(p, hazard.vertices) - PUSHER_RADIUS


def footprint_bbox(poses: np.ndarray, local=T_LOCAL) -> tuple[float, float, float, float]:
    v = t_polygons_batch(poses, local).reshape(-1, 2)
    return float(v[:, 0].min()), float(v[:, 0].max()), float(v[:, 1].min()), float(v[:, 1].max())


# --------------------------------------------------------------------------------------
# overlays
# --------------------------------------------------------------------------------------
def draw_overlay(
    frame: np.ndarray,
    pose=None,
    hazard: Hazard | None = None,
    *,
    local=T_LOCAL,
    pusher_xy=None,
    colour=(255, 0, 0),
    hazard_colour=(0, 128, 255),
    width: int = 1,
) -> np.ndarray:
    """Draw the T polygons and hazard on a 224 px frame (arena->image scale 224/512)."""
    from PIL import Image, ImageDraw

    img = Image.fromarray(np.asarray(frame).astype(np.uint8)).convert("RGB")
    dr = ImageDraw.Draw(img)
    s = RENDER_SCALE
    if hazard is not None:
        if isinstance(hazard, Box):
            dr.rectangle([hazard.x0 * s, hazard.y0 * s, hazard.x1 * s, hazard.y1 * s], outline=hazard_colour, width=width)
        else:
            dr.ellipse(
                [(hazard.cx - hazard.r) * s, (hazard.cy - hazard.r) * s, (hazard.cx + hazard.r) * s, (hazard.cy + hazard.r) * s],
                outline=hazard_colour,
                width=width,
            )
    if pose is not None:
        for poly in t_polygons(pose, local):
            pts = [(float(x * s), float(y * s)) for x, y in poly]
            dr.polygon(pts, outline=colour)
    if pusher_xy is not None:
        px, py, r = pusher_xy[0] * s, pusher_xy[1] * s, PUSHER_RADIUS * s
        dr.ellipse([px - r, py - r, px + r, py + r], outline=(0, 200, 0), width=width)
    return np.asarray(img)


def tile(frames: list[np.ndarray], ncols: int = 5) -> np.ndarray:
    frames = [np.asarray(f) for f in frames]
    h, w = frames[0].shape[:2]
    nrows = int(np.ceil(len(frames) / ncols))
    canvas = np.zeros((nrows * h, ncols * w, 3), dtype=np.uint8)
    for i, f in enumerate(frames):
        r, c = divmod(i, ncols)
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = f
    return canvas
