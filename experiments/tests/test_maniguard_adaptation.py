"""Controller contract tests. These are NOT evidence of manipulation performance."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.maniguard_adaptation import AdapterConfig, ExecutionAdapter, Frame


def frame(q=None, fingers=(0.01, 0.01), prerequisite=None, engaged=None):
    return Frame(np.zeros(7) if q is None else np.asarray(q, dtype=float),
                 np.asarray(fingers), np.zeros(3), np.eye(3, 7), prerequisite, engaged)


def test_identity_exact_and_does_not_mutate_input():
    adapter = ExecutionAdapter(AdapterConfig())
    rng = np.random.default_rng(12)
    for _ in range(20):
        raw = rng.normal(size=8).astype(np.float32)
        saved = raw.copy()
        actual, event = adapter.act(raw, frame())
        assert actual.dtype == raw.dtype
        np.testing.assert_array_equal(actual, saved)
        np.testing.assert_array_equal(raw, saved)
        assert not event["changed"]


def test_motion_is_relative_to_current_joints_and_keeps_gripper():
    adapter = ExecutionAdapter(AdapterConfig(kind="motion", free_step_limit=0.02))
    current = np.full(7, 1.2)
    raw = np.r_[np.full(7, 2.0), -1.0]
    actual, _ = adapter.act(raw, frame(q=current))
    np.testing.assert_allclose(actual[:7], current + 0.02)
    assert actual[-1] == -1


def test_motion_uses_final_clipped_action_as_history():
    adapter = ExecutionAdapter(AdapterConfig(kind="motion", smoothing=0.5, free_step_limit=1))
    adapter.applied(np.r_[np.full(7, 0.2), -1])
    actual, _ = adapter.act(np.r_[np.full(7, 0.4), 1], frame())
    np.testing.assert_allclose(actual[:7], 0.3)


def test_release_delay_expires_and_does_not_retrigger_on_persistent_request():
    adapter = ExecutionAdapter(AdapterConfig(kind="release", release_delay=3, min_closing_steps=2))
    for _ in range(4):
        adapter.act(np.r_[np.zeros(7), -1], frame())
    outputs = [adapter.act(np.r_[np.zeros(7), 1], frame())[0][-1] for _ in range(8)]
    assert outputs == [-1, -1, -1, 1, 1, 1, 1, 1]


def test_release_does_not_hold_without_grasp_evidence():
    adapter = ExecutionAdapter(AdapterConfig(kind="release"))
    for _ in range(10):
        adapter.act(np.r_[np.zeros(7), -1], frame(fingers=(0, 0)))
    actual, event = adapter.act(np.r_[np.zeros(7), 1], frame(fingers=(0, 0)))
    assert actual[-1] == 1 and not event["release_delayed"]


def test_guard_blocks_lift_only_for_engaged_unclosed_target():
    adapter = ExecutionAdapter(AdapterConfig(kind="prerequisite", privileged=True))
    lift = np.array([0, 0, 0.1, 0, 0, 0, 0, -1.0])
    actual, event = adapter.act(lift, frame(prerequisite=False, engaged=True))
    np.testing.assert_array_equal(actual[:7], np.zeros(7))
    assert event["guard_blocked"]
    actual, event = adapter.act(lift, frame(prerequisite=True, engaged=True))
    np.testing.assert_array_equal(actual, lift)
    actual, event = adapter.act(lift, frame(prerequisite=False, engaged=True))
    assert not event["guard_blocked"]  # weak-until prerequisite was discharged


def test_guard_does_not_block_free_arm_or_downward_motion():
    adapter = ExecutionAdapter(AdapterConfig(kind="prerequisite", privileged=True))
    for delta, engaged in ((0.1, False), (-0.1, True)):
        raw = np.array([0, 0, delta, 0, 0, 0, 0, -1.0])
        actual, event = adapter.act(raw, frame(prerequisite=False, engaged=engaged))
        np.testing.assert_array_equal(actual, raw)
        assert not event["guard_blocked"]


def test_guard_requires_explicit_privilege_and_fails_on_missing_state():
    with pytest.raises(ValueError, match="privileged"):
        AdapterConfig(kind="prerequisite")
    adapter = ExecutionAdapter(AdapterConfig(kind="prerequisite", privileged=True))
    with pytest.raises(ValueError, match="object state"):
        adapter.act(np.zeros(8), frame())


def test_handback_settles_for_finite_duration_then_requeries():
    adapter = ExecutionAdapter(AdapterConfig(kind="handback", privileged=True, handback_hold_steps=2))
    lift = np.array([0, 0, 0.1, 0, 0, 0, 0, -1.0])
    adapter.act(lift, frame(prerequisite=False, engaged=True))
    clear = frame(prerequisite=True, engaged=True)
    adapter.on_query(clear)
    for _ in range(2):
        actual, _event = adapter.act(lift, clear)
        np.testing.assert_array_equal(actual[:7], np.zeros(7))
        adapter.applied(actual)
    assert adapter.should_replan(clear)
    adapter.on_query(clear)
    actual, _event = adapter.act(lift, clear)
    np.testing.assert_array_equal(actual, lift)


def test_replanning_respects_minimum_interval_and_uses_execution_error():
    adapter = ExecutionAdapter(AdapterConfig(kind="replan", min_query_interval=2))
    adapter.on_query(frame())
    adapter.applied(np.r_[np.ones(7), -1])
    assert not adapter.should_replan(frame())
    adapter.applied(np.r_[np.ones(7), -1])
    assert adapter.should_replan(frame())
    adapter.on_query(frame())
    assert not adapter.should_replan(frame())


def test_retry_is_bounded_and_physical_motion_opposes_stalled_demand():
    adapter = ExecutionAdapter(AdapterConfig(kind="retry", stall_window=3, retry_steps=2,
                                             retry_limit=1, retry_cooldown=1))
    raw = np.array([0.1, 0, 0, 0, 0, 0, 0, -1.0])
    events = []
    corrections = []
    for _ in range(30):
        actual, event = adapter.act(raw, frame())
        adapter.applied(actual)
        events.append(event)
        if event["retry_active"]:
            corrections.append(actual.copy())
    assert sum(e["retry_started"] for e in events) == 1
    assert len(corrections) == 2
    assert all(a[0] < 0 and a[-1] == -1 for a in corrections)
    assert not events[-1]["retry_active"]


def test_reset_clears_episode_history():
    adapter = ExecutionAdapter(AdapterConfig(kind="prerequisite", privileged=True))
    raw = np.array([0, 0, 0.1, 0, 0, 0, 0, -1.0])
    adapter.act(raw, frame(prerequisite=True, engaged=True))
    adapter.reset()
    assert adapter.act(raw, frame(prerequisite=False, engaged=True))[1]["guard_blocked"]


@pytest.mark.parametrize("bad", [np.full(8, np.nan), np.zeros(7), np.full(8, np.inf)])
def test_invalid_proposals_fail_before_execution(bad):
    with pytest.raises(ValueError):
        ExecutionAdapter(AdapterConfig()).act(bad, frame())


@pytest.mark.parametrize("kwargs", [{"horizon": 0}, {"smoothing": 0}, {"release_delay": -1},
                                   {"kind": "made_up"}, {"lift_margin": float("nan")}])
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        AdapterConfig(**kwargs)
