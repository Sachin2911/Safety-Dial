from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.pushtSourceFamilies import SOURCE_FAMILY_PROTOCOL, geometric_source_family, validate_geometric_bank
from helpers.studyGates import acquisition_repeatability_gate, require_closedloop_gate


def state(x, y):
    return np.asarray([256, 256, x, y, 0, 0, 0], float)


def test_source_families_are_geometric_not_episode_parity():
    assert geometric_source_family(state(64, 64), state(80, 80)) == "familiar"
    assert geometric_source_family(state(192, 64), state(210, 80)) == "heldout"
    assert geometric_source_family(state(64, 64), state(192, 64)) is None
    assert geometric_source_family(state(128, 64), state(192, 64)) is None


def test_mislabeled_source_roots_cannot_pass_transfer_validation():
    root = SimpleNamespace(root_id="r", goal_state=state(210, 80), meta={
        "state_at_root": state(192, 64), "source_family": "familiar",
        "source_family_protocol": SOURCE_FAMILY_PROTOCOL["version"]})
    with pytest.raises(ValueError, match="invalid geometric"):
        validate_geometric_bank(SimpleNamespace(roots=[root]))
    root.meta["source_family"] = "heldout"
    with pytest.raises(ValueError, match="familiar-only"):
        validate_geometric_bank(SimpleNamespace(roots=[root]), expected_family="familiar")


def passing_report():
    report = {"status": "complete", "gate": {"diagnostic": False, "e2_passed": True}, "split_audit": {"passes": True},
              "source_family_protocol": SOURCE_FAMILY_PROTOCOL, "arms": {"random": {}, "boundary": {}},
              "no_update": {"test": {"at_matched": {"fsa": .2}}},
              "no_update_retention": {"latent_mse_tf": .01}, "boundary_vs_random_test": {}}
    for seed in map(str, range(3)):
        for arm in report["arms"]:
            report["arms"][arm][seed] = {"curve": [{"budget_added": b, "retention": {"latent_mse_tf": .009},
                "eval": {"test": {"at_matched": {"fsa": .1}}}} for b in (64, 128)]}
    for budget in (64, 128):
        report["boundary_vs_random_test"][str(budget)] = {"paired_root_intervals_by_seed": {
            str(seed): {"point": -.03, "lo": -.05, "hi": -.01, "n_boot": 1000} for seed in range(3)}}
    return report


def test_e5_requires_consistent_multiple_seed_multiple_budget_evidence():
    report = passing_report()
    assert require_closedloop_gate(report)["selected_budget"] == 128
    report["boundary_vs_random_test"]["64"]["paired_root_intervals_by_seed"]["2"]["hi"] = .01
    assert not acquisition_repeatability_gate(report)["passes"]


@pytest.mark.parametrize("change", ["diagnostic", "leakage", "one_seed", "retention"])
def test_unqualified_runs_never_unlock_closed_loop(change):
    report = copy.deepcopy(passing_report())
    if change == "diagnostic":
        report["gate"]["diagnostic"] = True
    elif change == "leakage":
        report["split_audit"]["passes"] = False
    elif change == "one_seed":
        report["arms"]["boundary"] = {"0": report["arms"]["boundary"]["0"]}
    else:
        for point in report["arms"]["boundary"]["0"]["curve"]:
            point["retention"]["latent_mse_tf"] = .1
    with pytest.raises(ValueError, match="Closed-loop gate"):
        require_closedloop_gate(report)


def test_goal_retention_binds_twenty_paired_cases_to_exact_checkpoint():
    from helpers.studyGates import require_goal_retention

    from helpers.pushtRetention import RETENTION_PROTOCOL, TERMINATION_POLICY

    rows = [{"root_id": f"r{i}", "episode": i, "case_sha256": f"case-{i}", "valid": True,
             "requested_steps": 250, "executed_steps": 250, "censored": False,
             "censored_future": False, "arena_exit": False, "terminated": False,
             "truncated": False, "completed_on_verified_goal": False,
             "final_coverage": .9, "final_pose_error_px": 10.} for i in range(20)]
    summary = {"n_blocks": 50, "steps_per_block": 5, "protocol": RETENTION_PROTOCOL,
               "termination_policy": TERMINATION_POLICY}
    report = {"repaired_weights_sha256": "abc", "comparison": {"passes": True},
              "base": {"rows": rows, "summary": summary},
              "adapted": {"rows": list(rows), "summary": summary}}
    assert require_goal_retention(report, "abc")["n_paired_cases"] == 20
    with pytest.raises(ValueError, match="does not identify"):
        require_goal_retention(report, "different")
    report["adapted"]["rows"] = list(reversed(rows))
    with pytest.raises(ValueError, match="fixed-horizon"):
        require_goal_retention(report, "abc")
