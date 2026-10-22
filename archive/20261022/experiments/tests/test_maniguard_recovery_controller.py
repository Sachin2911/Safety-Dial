"""Recovery checks exercise the causal boundary and cap geometry."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.maniguard_recovery_controller import (
    Config,
    RecoveryController,
    cap_targets,
    direction_angle,
    final_input,
)
from scripts.maniguard_recovery_trace_screen import prefixes, timing_rows


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_identity_and_inactive_cap_are_bitwise(dtype):
    p = np.asarray([.01, -.02, 0, .005, 0, 0, -.0, -1], dtype=dtype)
    for cap in (None, .04):
        result = cap_targets(p, np.zeros(7), Config(cap=cap))
        assert result.dtype == p.dtype
        assert result.tobytes() == p.tobytes()


@pytest.mark.parametrize("cap", [.04, .08, .12])
def test_radial_map_preserves_direction_and_gripper(cap):
    rng = np.random.default_rng(20261019)
    for _ in range(100):
        q = rng.uniform(-2, 2, 7)
        e = rng.uniform(-.5, .5, 7)
        p = np.r_[q + e, -1.].astype(np.float32)
        result = cap_targets(p, q, Config(cap=cap))
        assert np.max(np.abs(result[:7] - q)) <= cap + 3e-7
        assert direction_angle(p, result, q) < .002
        assert result[-1] == p[-1]


def test_componentwise_distortion_and_downstream_clipping_are_visible():
    p = np.asarray([.20, .05, 0, 0, 0, 0, 0, -1.])
    q = np.zeros(7)
    radial = cap_targets(p, q, Config(.04))
    component = cap_targets(p, q, Config(.04, "componentwise"))
    assert direction_angle(p, radial, q) < 1e-5
    assert direction_angle(p, component, q) > 20
    low, high = np.full(8, -np.inf), np.full(8, np.inf)
    high[0] = .01
    clipped = final_input(radial, low, high)
    assert direction_angle(p, clipped, q) > 20


def test_trigger_is_causal_and_resets_at_query():
    c = RecoveryController(Config(.04, threshold=.1))
    p = np.asarray([.20, 0, 0, 0, 0, 0, 0, -1.])
    for age in range(1, 5):
        action = c.act(p, np.zeros(7))
        with pytest.raises(ValueError):
            c.replan_reason()
        c.applied(action)
        assert c.replan_reason() == ("correction_threshold" if age == 4 else None)
    c.on_query()
    assert c.correction == 0 and c.since_query == 0
    assert c.replan_reason() is None


def test_actual_clipping_drives_correction_and_roundoff_is_ignored():
    p = np.zeros(8, dtype=np.float64)
    c = RecoveryController(Config(threshold=.1))
    low, high = np.full(8, -np.inf), np.full(8, np.inf)
    low[0] = .02
    for _ in range(4):
        c.applied(final_input(c.act(p, np.zeros(7)), low, high))
    assert c.replan_reason() == "correction_threshold"
    c.on_query()
    action = c.act(p, np.zeros(7))
    action[:7] += 1e-7
    c.applied(action)
    assert c.correction == 0


@pytest.mark.parametrize("threshold,minimum,expected", [(None, 4, 8), (0, 4, 4), (0, 6, 6)])
def test_fixed_schedules(threshold, minimum, expected):
    c = RecoveryController(Config(threshold=threshold, minimum_interval=minimum))
    for age in range(1, expected + 1):
        c.applied(c.act(np.zeros(8), np.zeros(7)))
        assert (c.replan_reason() is not None) == (age == expected)


@pytest.mark.parametrize("kwargs", [
    {"cap": 0}, {"cap": True}, {"threshold": float("nan")},
    {"threshold": -1}, {"threshold": True}, {"minimum_interval": 4.},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_missing_and_nonfinite_actions_are_rejected():
    c = RecoveryController(Config())
    with pytest.raises(ValueError):
        c.applied(np.zeros(8))
    with pytest.raises(ValueError):
        c.act(np.full(8, np.nan), np.zeros(7))
    c.act(np.zeros(8), np.zeros(7))
    with pytest.raises(ValueError):
        c.act(np.zeros(8), np.zeros(7))


def recorded_episode(lengths):
    n = sum(lengths)
    p = np.asarray([.05, 0, 0, 0, 0, 0, 0, -1.])
    return {"job": {"episode_id": "synthetic", "scene": "task_synthetic/base"},
            "queries": np.cumsum([0] + lengths[:-1]).tolist(), "chunk_lengths": lengths,
            "commands": [(p, np.zeros(7), np.full(8, -np.inf), np.full(8, np.inf))] * n}


def test_only_complete_eight_action_prefixes_support_early_refresh():
    e = recorded_episode([8, 4, 3])
    chunks = prefixes(e, .04)
    assert len(chunks) == 1
    assert chunks[0]["query_step"] == 0
    rows = timing_rows([e], [("test", .1)], 4, .04, "synthetic")
    assert rows[0]["complete_eight_action_prefixes"] == 1
    assert rows[0]["prefixes_with_early_opportunity"] == 1
    assert rows[0]["age4"] == 1
    assert rows[0]["discarded_incomplete_prefixes"] == 2


def test_crossing_at_age_eight_is_not_an_early_opportunity():
    e = recorded_episode([8])
    cumulative = prefixes(e, .04)[0]["cumulative"]
    threshold = (cumulative[6] + cumulative[7]) / 2
    rows = timing_rows([e], [("test", threshold)], 4, .04, "synthetic")
    assert rows[0]["prefixes_with_early_opportunity"] == 0


def test_minimum_interval_six_and_first_crossing_per_prefix():
    e = recorded_episode([8, 8])
    rows = timing_rows([e], [("test", .1)], 6, .04, "synthetic")
    assert rows[0]["prefixes_with_early_opportunity"] == 2
    assert rows[0]["age6"] == 2
    assert sum(rows[0][key] for key in ("age4", "age5", "age6", "age7")) == 2
