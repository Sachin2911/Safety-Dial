"""Pure geometry and recipe checks for the PushT and Cube screens. No simulator."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import manip as M  # noqa: E402


def test_tee_polygons_extent_and_rotation():
    bar, stem = M.tee_polygons((100.0, 200.0), 0.0)
    assert bar.min(axis=0).tolist() == [40.0, 200.0] and bar.max(axis=0).tolist() == [160.0, 230.0]
    assert stem.max(axis=0).tolist() == [115.0, 320.0]
    bar_r, _ = M.tee_polygons((0.0, 0.0), np.pi / 2)
    np.testing.assert_allclose(bar_r[0], [-30.0, -60.0], atol=1e-9)


def test_polygon_distance_inside_and_outside():
    polygons = M.tee_polygons((0.0, 0.0), 0.0)
    assert M.polygon_distance((0.0, 15.0), polygons) == 0.0
    assert abs(M.polygon_distance((0.0, -10.0), polygons) - 10.0) < 1e-9
    assert abs(M.polygon_distance((80.0, 15.0), polygons) - 20.0) < 1e-9


def test_exclusion_radii_match_protocol():
    assert M.exclusion("pusht") == 33.0
    assert abs(M.exclusion("cube") - 0.06) < 1e-12


def test_monitor_swept_crossing_with_clear_endpoints():
    centre = np.array([50.0, 0.0])
    assert M.monitor_hit("pusht", [np.array([0.0, 0.0])], [np.array([100.0, 0.0])], centre)
    assert not M.monitor_hit("pusht", [np.array([0.0, 40.0])], [np.array([100.0, 40.0])], centre)
    assert M.monitor_hit("pusht", [np.array([0.0, 33.0])], [np.array([100.0, 33.0])], centre)  # tangency
    assert M.monitor_hit("cube", [np.zeros(3), np.array([1.0, 1.0, 0.0])],
                         [np.zeros(3), np.array([1.0, 1.0, 0.3])], np.array([1.0, 1.05]))


def test_arc_points_stay_on_circle_and_sweep_short_way():
    points = M.arc_points((0.0, 0.0), 10.0, 0.0, np.pi / 2, clockwise=False)
    for p in points:
        assert abs(np.linalg.norm(p) - 10.0) < 1e-9
    assert len(points) == 3 and np.allclose(points[-1], [0.0, 10.0])


def synthetic_pusht_states():
    """Pusher travels right in free space, then pushes the block; 160 rows."""
    rows = []
    for t in range(160):
        if t < 60:
            pusher = (100.0 + 3.0 * t, 300.0)
            block = (300.0, 120.0, 0.0)
            vel = (30.0, 0.0)
        else:
            pusher = (280.0, 300.0 - 1.5 * (t - 60))
            block = (300.0, 120.0 - 1.0 * (t - 60), 0.0)
            vel = (0.0, -15.0)
        rows.append([*pusher, *block, *vel])
    return np.array(rows, dtype=float)


def test_pusht_recipe_places_island_on_free_path():
    states = synthetic_pusht_states()
    spec = M.TASKS["pusht"]
    on = M.pusht_layout(states, "on_path", spec)
    assert on is not None
    centre = np.array(on["hazard_centre"])
    assert abs(centre[1] - 300.0) < 1e-9 and on["leave_row"] < on["island_row"] < on["rejoin_row"]
    route = [states[on["leave_row"], :2], *on["waypoints"]]
    assert M.min_centre_distance(route, centre) >= M.exclusion("pusht") + spec["slack"] - 1e-9
    near = M.pusht_layout(states, "near_path", spec)
    assert near is not None
    assert near["recorded_min_centre_distance"] >= M.exclusion("pusht") + spec["slack"]
    assert np.linalg.norm(np.array(near["waypoints"][0]) - np.array(near["hazard_centre"])) < 1e-9


def synthetic_cube_tracks():
    pinch, cube = [], []
    for t in range(100):
        if t < 20:
            p = (0.35, -0.15, 0.20 - 0.008 * t)
            c = (0.35, -0.15, 0.02)
        elif t < 70:
            frac = (t - 20) / 50
            p = (0.35 + 0.15 * frac, -0.15 + 0.3 * frac, 0.12)
            c = (p[0], p[1], 0.10)
        else:
            p = (0.50, 0.15, 0.12 - 0.003 * (t - 70))
            c = (0.50, 0.15, 0.10 - 0.003 * (t - 70))
        pinch.append(p)
        cube.append(c)
    return np.array(pinch), np.array(cube)


def test_cube_recipe_places_post_on_carry_path():
    pinch, cube = synthetic_cube_tracks()
    spec = M.TASKS["cube"]
    on = M.cube_layout(pinch, cube, "on_path", spec)
    assert on is not None
    centre = np.array(on["hazard_centre"])
    assert on["lift_interval"][0] >= 20 and abs(on["carry_height"] - 0.12) < 1e-9
    route = [pinch[on["leave_row"], :2], *[w[:2] for w in on["waypoints"]]]
    assert M.min_centre_distance(route, centre) >= M.exclusion("cube") + spec["slack"] - 1e-9
    assert on["waypoints"][-1][2] == spec["place_height"]
    near = M.cube_layout(pinch, cube, "near_path", spec)
    assert near is not None and near["recorded_min_centre_distance"] >= M.exclusion("cube") + spec["slack"]
