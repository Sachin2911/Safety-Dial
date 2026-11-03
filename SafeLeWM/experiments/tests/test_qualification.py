"""Pure geometry/accounting checks, run on the rented host before episodes."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.qualification import first_event_costs, segment_distance  # noqa: E402


def test_crossing_detected_between_clear_endpoints():
    assert segment_distance([0, 0], [10, 0], [5, 0]) == 0
    assert np.linalg.norm(np.array([0, 0])-np.array([5, 0])) > 1
    assert np.linalg.norm(np.array([10, 0])-np.array([5, 0])) > 1


def test_stationary_and_tangent():
    assert segment_distance([0, 0], [0, 0], [3, 4]) == 5
    assert segment_distance([-10, 20], [10, 20], [0, 0]) == 20


def test_first_event_not_unsafe_step_count():
    assert first_event_costs([False, True, True, False, True]) == [0, 1, 0, 0, 0]
    assert first_event_costs([False] * 5) == [0] * 5


def test_declared_routes_have_clear_segments():
    for x in [134, 194]:
        route = [[164, 90], [x, 90], [x, 160], [164, 170]]
        assert min(segment_distance(a, b, [164, 130])
                   for a, b in zip(route[:-1], route[1:])) > 20
