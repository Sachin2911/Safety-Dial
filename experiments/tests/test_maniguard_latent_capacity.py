"""Protect full native horizons and preserved task partitions before search."""

import pytest

from helpers.maniguard_latent_capacity import (
    comparison_design,
    internal_clock_window_ready,
    planning_bound,
    provider_window_confirmed,
)


def test_capacity_does_not_treat_early_success_as_full_timeout_cost():
    # Startup 80 s, then .5 s per control. A 100-action early success takes
    # 130 s, but a native 2000-action timeout requires 1080 s before margin.
    early = [{"startup_seconds": 80, "elapsed_seconds": 130, "steps": 100}]
    assert planning_bound(early, 2000) == 1350
    full = [{"startup_seconds": 80, "elapsed_seconds": 1080, "steps": 2000}]
    assert planning_bound(full, 2000) == 1350
    assert planning_bound(early, 1100) == 787.5
    with pytest.raises(ValueError):
        planning_bound([{**early[0], "steps": 0}], 2000)


def test_comparison_keeps_four_conditions_independent_and_balanced():
    for family in ["jar", "clutter"]:
        design = comparison_design(family)
        panels, audit = design["training_panels"], design["audit_blocks"]
        assert len(panels) == 3 and all(len(p) == 2 for p in panels)
        assert len(audit) == 4
        train_tasks = {c["scene"] for p in panels for c in p}
        assert train_tasks.isdisjoint(c["scene"] for c in audit)
        for position in range(4):
            assert {case["order"][position] for case in audit} == {"native", "fixed", "ea", "random"}
        assert len({case["seed"] for case in audit}) == 4


def test_earlier_provider_confirmation_cannot_cover_a_later_extension():
    assert provider_window_confirmed("2026-10-20T03:00:00Z")
    for deadline in ["2026-10-20T05:00:00Z", "2026-10-20T14:00:00Z"]:
        assert not provider_window_confirmed(deadline)
    confirmation = {"instance_id": 56777435, "earlier_stop_removed_or_moved": True,
                    "no_provider_stop_before_utc": "2026-10-20T05:00:00Z"}
    assert provider_window_confirmed("2026-10-20T05:00:00Z", confirmation)
    assert not provider_window_confirmed("2026-10-20T14:00:00Z", confirmation)
    confirmation["no_provider_stop_before_utc"] = "2026-10-20T14:00:00Z"
    assert provider_window_confirmed("2026-10-20T14:00:00Z", confirmation)
    confirmation["instance_id"] = 1
    assert not provider_window_confirmed("2026-10-20T14:00:00Z", confirmation)


def test_authorized_internal_clock_needs_live_evidence_after_the_known_provider_stop():
    now = 1792470000
    confirmation = {
        "protection_mode": "authorized_internal_clock_after_elapsed_stop",
        "instance_id": 56777435, "known_provider_job_id": 4986,
        "known_provider_stop_utc": "2026-10-20T03:00:00Z",
        "authorization_messages": ["Go ahead I give you authorisation",
                                   ("You can run the full comparison, you need to get the "
                                    "contribution and result, that is most important")],
        "instance_http_status": 200, "instance_actual_status": "running",
        "instance_checked_epoch": now - 10,
        "watchdog_status": "armed", "watchdog_deadline_utc": "2026-10-20T20:00:00Z",
        "watchdog_checked_epoch": now - 10}
    assert internal_clock_window_ready("2026-10-20T20:00:00Z", confirmation, now=now)
    assert not provider_window_confirmed("2026-10-20T20:00:00Z", confirmation)
    assert not internal_clock_window_ready("2026-10-21T04:00:00Z", confirmation, now=now)
    assert not internal_clock_window_ready("2026-10-20T20:00:00Z", confirmation, now=now + 61)
    confirmation["authorization_messages"] = []
    assert not internal_clock_window_ready("2026-10-20T20:00:00Z", confirmation, now=now)
