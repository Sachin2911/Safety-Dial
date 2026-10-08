"""CPU replay/geometry evidence that never treats terminal padding as observations."""
from __future__ import annotations

import numpy as np

from helpers.pushtGeometry import Box, Disc, T_LOCAL, clearance_trace, t_polygons
from helpers.pushtReplay import ACTION_BLOCK, endpoint_interpolation


def compare_replays(logs):
    if len(logs) < 3:
        raise ValueError("E0 requires at least three independent reset-and-prefix repeats")
    ref = logs[0]
    numeric = ("states", "block_vel", "block_ang_vel")
    flags = ("observed", "observation_valid", "terminated", "truncated", "n_contacts", "actions")
    out = {"repeats": len(logs), "requested_steps": len(ref.actions), "observed_steps": ref.executed_steps,
           "censored": ref.censored, "observation_domain_exit": bool((ref.observed & ~ref.observation_valid).any()),
           "contact_steps": int((ref.n_contacts[ref.observed] > 0).sum()),
           "bitwise": True, "frames_equal": True, "flags_equal": True}
    for key in numeric:
        out["max_abs_" + key] = max(float(np.abs(getattr(log, key) - getattr(ref, key)).max()) for log in logs[1:])
        out["bitwise"] &= all(np.array_equal(getattr(log, key), getattr(ref, key)) for log in logs[1:])
    for key in flags:
        out["flags_equal"] &= all(np.array_equal(getattr(log, key), getattr(ref, key)) for log in logs[1:])
    for log in logs[1:]:
        out["frames_equal"] &= len(log.frames) == len(ref.frames) and all(np.array_equal(a, b) for a, b in zip(log.frames, ref.frames))
    out["bitwise"] = bool(out["bitwise"] and out["flags_equal"])
    return out


def timing_row(log, hazard):
    """Use complete observed action blocks; report excluded terminal remainder explicitly."""
    end = log.executed_steps // ACTION_BLOCK * ACTION_BLOCK
    if end == 0:
        return {"evaluated_steps": 0, "excluded_observed_tail_steps": log.executed_steps,
                "censored": log.censored}
    st = log.states[:end + 1]
    interp = endpoint_interpolation(st)
    dense, sampled = clearance_trace(st[:, 2:5], hazard), clearance_trace(interp[:, 2:5], hazard)
    delta = np.arctan2(np.sin(interp[:, 4] - st[:, 4]), np.cos(interp[:, 4] - st[:, 4]))
    return {"evaluated_steps": end, "excluded_observed_tail_steps": log.executed_steps - end,
            "censored": log.censored, "contact": bool((log.n_contacts[:end + 1] > 0).any()),
            "centre_max_px": float(np.linalg.norm(interp[:, 2:4] - st[:, 2:4], axis=1).max()),
            "angle_max_deg": float(np.degrees(np.abs(delta)).max()),
            "dense_unsafe": bool((dense <= 0).any()), "interpolated_unsafe": bool((sampled <= 0).any()),
            "interp_minus_dense_min_clearance_px": float(sampled.min() - dense.min())}


def live_geometry_check(env, poses):
    """Check production vertices against pymunk, and signs/distances against GEOS."""
    from shapely.geometry import Point, Polygon, box
    from shapely.ops import unary_union

    body = env.unwrapped.block
    shapes = list(body.shapes)
    live_local = [np.asarray([tuple(v) for v in shape.get_vertices()]) for shape in shapes]
    def canonical(poly):
        return sorted(tuple(np.round(vertex, 12)) for vertex in poly)

    local_equal = sorted(canonical(poly) for poly in live_local) == sorted(canonical(poly) for poly in T_LOCAL)
    current = [body.position.x, body.position.y, body.angle]
    actual = [np.asarray([tuple(body.local_to_world(v)) for v in shape.get_vertices()]) for shape in shapes]
    predicted = t_polygons(current, live_local)
    vertex_error = max(float(np.abs(a - b).max()) for a, b in zip(actual, predicted))
    rows = []
    for pose in np.asarray(poses).reshape(-1, 3):
        footprint = unary_union([Polygon(poly) for poly in t_polygons(pose, live_local)])
        x, y = float(pose[0]), float(pose[1])
        for hz in (Box(x - 10, x + 10, y - 10, y + 10), Box(x + 130, x + 145, y + 130, y + 145),
                   Disc(x, y, 15), Disc(x + 150, y + 150, 20)):
            value = float(clearance_trace(np.asarray([pose]), hz)[0])
            if isinstance(hz, Box):
                geometry = box(hz.x0, hz.y0, hz.x1, hz.y1)
                independent_unsafe = footprint.intersects(geometry)
                distance = footprint.distance(geometry)
            else:
                distance = footprint.distance(Point(hz.cx, hz.cy)) - hz.r
                independent_unsafe = distance <= 0
            rows.append({"sign_agrees": bool((value <= 0) == independent_unsafe),
                         "outside_distance_error": abs(value - distance) if value > 0 else 0.0})
    return {"local_vertices_match_production": local_equal, "live_world_vertex_max_abs_error": vertex_error,
            "independent_geometry_cases": len(rows), "all_intersections_agree": all(row["sign_agrees"] for row in rows),
            "outside_distance_max_abs_error": max(row["outside_distance_error"] for row in rows),
            "passes": bool(local_equal and vertex_error < 1e-10 and all(row["sign_agrees"] for row in rows)
                           and max(row["outside_distance_error"] for row in rows) < 1e-9)}
