"""Protect full native horizons and preserved task partitions before search."""

import pytest

from helpers.maniguard_latent_capacity import comparison_design, planning_bound


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
