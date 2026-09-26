"""Coverage and retention regressions independent of GPU/model availability."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from helpers.pushtRetention import RETENTION_PROTOCOL, TERMINATION_POLICY, block_coverage, compare_retention  # noqa: E402


def test_coverage_is_whole_shape_and_periodic():
    assert block_coverage([200, 200, .4], [200, 200, .4]) == pytest.approx(1)
    assert block_coverage([200, 200, .4], [200, 200, .4 + 2 * np.pi]) == pytest.approx(1)
    assert block_coverage([200, 200, .4], [1000, 1000, .4]) == 0
    assert 0 < block_coverage([210, 200, .4], [200, 200, .4]) < 1


def result(coverage, error, n=20):
    return {"rows": [{"root_id": str(i), "valid": True, "final_coverage": coverage,
                      "final_pose_error_px": error, "episode": i, "case_sha256": f"case-{i}",
                      "requested_steps": 250, "executed_steps": 250,
                      "censored": False, "censored_future": False, "arena_exit": False,
                      "terminated": False, "truncated": False, "completed_on_verified_goal": False} for i in range(n)],
            "summary": {"arena_exits": 0, "n_blocks": 50, "steps_per_block": 5,
                        "protocol": RETENTION_PROTOCOL, "termination_policy": TERMINATION_POLICY}}


def test_clear_goal_regression_cannot_pass_retention():
    base = result(.8, 15)
    adapted = result(.5, 50)
    report = compare_retention(base, adapted)
    assert not report["passes"]
    assert report["final_coverage"]["hi"] < 0
    assert report["final_pose_error_px"]["lo"] > 0


def test_incomplete_retention_is_not_a_gate_pass():
    assert not compare_retention(result(.8, 15, 1), result(.8, 15, 1))["passes"]
    assert compare_retention(result(.8, 15), result(.9, 10))["passes"]


@pytest.mark.parametrize("change", ["horizon_mismatch", "short_smoke", "source_duplicate",
    "source_mismatch", "case_mismatch", "censor", "missing_censor", "early_terminal",
    "arena_exit", "nonfinite", "duplicate_ids"])
def test_retention_cannot_pass_invalid_pairing_or_unseen_horizon(change):
    base, adapted = result(.8, 15), result(.9, 10)
    if change == "horizon_mismatch":
        adapted["summary"]["n_blocks"] = 51
    elif change == "short_smoke":
        for report in (base, adapted):
            report["summary"]["n_blocks"] = 1
            for row in report["rows"]:
                row["requested_steps"] = row["executed_steps"] = 5
    elif change == "source_duplicate":
        for report in (base, adapted):
            report["rows"][1]["episode"] = 0
    elif change == "source_mismatch":
        adapted["rows"][0]["episode"] = 100
    elif change == "case_mismatch":
        adapted["rows"][0]["case_sha256"] = "different-goal"
    elif change == "censor":
        adapted["rows"][0]["censored"] = True
    elif change == "missing_censor":
        del adapted["rows"][0]["censored"]
    elif change == "early_terminal":
        adapted["rows"][0].update(executed_steps=5, terminated=True)
    elif change == "arena_exit":
        adapted["rows"][0]["arena_exit"] = True
    elif change == "nonfinite":
        adapted["rows"][0]["final_coverage"] = float("nan")
    else:
        for report in (base, adapted):
            report["rows"][1]["root_id"] = "0"
    check = compare_retention(base, adapted)
    assert not check["passes"]
    assert not check["complete"]
    assert check["reasons"]


def test_reported_pass_cannot_override_retention_integrity():
    from helpers.studyGates import require_goal_retention

    report = {"base": result(.8, 15), "adapted": result(.9, 10),
              "comparison": {"passes": True}, "repaired_weights_sha256": "weights"}
    assert require_goal_retention(report, "weights")["passes"]
    report["adapted"]["rows"][0]["censored"] = True
    with pytest.raises(ValueError, match="fixed-horizon"):
        require_goal_retention(report, "weights")



def test_early_verified_goal_is_complete_but_unseen_future_stays_censored():
    from helpers.pushtGeometry import T_LOCAL
    from helpers.pushtRetention import verified_goal_completion

    base, adapted = result(.8, 15), result(.9, 10)
    row = adapted["rows"][0]
    state = [100., 100., 200., 200., 0., 0., 0.]
    row.update(executed_steps=5, terminated=True, censored=True, censored_future=True,
               final_state=state, goal_state=state, local_polygons=[p.tolist() for p in T_LOCAL],
               final_coverage=1., final_pose_error_px=0., completed_on_verified_goal=True)
    assert verified_goal_completion(row)
    assert compare_retention(base, adapted)["passes"]
    row["truncated"] = True
    assert not compare_retention(base, adapted)["passes"]
    row["truncated"] = False
    row["final_coverage"] = .94
    assert not compare_retention(base, adapted)["passes"]
    row["final_coverage"] = 1.
    row["goal_state"] = [300., 300., 200., 200., 0., 0., 0.]
    assert not compare_retention(base, adapted)["passes"]
