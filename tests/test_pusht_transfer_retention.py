"""Transfer retention must bind actual checkpoints to one frozen independent case set."""
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from helpers.pushtReplay import Root
from helpers.pushtRetention import RETENTION_PROTOCOL, TERMINATION_POLICY, case_identity, load_fixed_cases
from helpers.pushtTransferRetention import checkpoint_plan
from helpers.studyGates import require_goal_retention


def test_case_file_binding_rejects_changed_roots_and_horizons(tmp_path):
    cases = [Root(i, np.zeros(7), np.ones(7), np.zeros((10, 2)), str(i),
                  {"episode": i, "role": "retention"}) for i in range(20)]
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([r.to_dict() for r in cases]))
    _, digest = load_fixed_cases(path, {"roles": {"retention": list(range(20))}})
    rows = [{"root_id": r.root_id, "episode": r.meta["episode"], "case_sha256": case_identity(r),
             "valid": True, "arena_exit": False, "censored": False, "censored_future": False,
             "terminated": False, "truncated": False, "completed_on_verified_goal": False,
             "requested_steps": 250, "executed_steps": 250,
             "final_coverage": .8, "final_pose_error_px": 15.} for r in cases]
    base = {"rows": rows, "summary": {"n_blocks": 50, "steps_per_block": 5,
            "protocol": RETENTION_PROTOCOL, "termination_policy": TERMINATION_POLICY}}
    report = {"base": base, "adapted": copy.deepcopy(base), "case_file_sha256": digest,
              "comparison": {"passes": True}, "repaired_weights_sha256": "weights"}
    assert require_goal_retention(report, "weights", cases_path=path,
        case_sha256=digest, n_blocks=50)["passes"]
    with pytest.raises(ValueError, match="horizon"):
        require_goal_retention(report, "weights", n_blocks=51)
    report["adapted"]["rows"][0]["case_sha256"] = "other-goal"
    with pytest.raises(ValueError, match="rows do not match"):
        require_goal_retention(report, "weights", cases_path=path, case_sha256=digest, n_blocks=50)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="case file changed"):
        require_goal_retention(report, "weights", cases_path=path, case_sha256=digest)


def test_case_loading_rejects_a_training_episode(tmp_path):
    cases = [Root(i, np.zeros(7), np.ones(7), np.zeros((10, 2)), str(i),
                  {"episode": i, "role": "retention"}).to_dict() for i in range(20)]
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases))
    with pytest.raises(ValueError, match="retention source"):
        load_fixed_cases(path, {"roles": {"retention": list(range(1, 21))}})


def acquisition(tmp_path):
    report = {"run_id": "acq", "status": "complete", "gate": {"e2_passed": True},
              "split_audit": {"passes": True}, "arms": {},
              "source_family_protocol": {"version": "block-start-goal-checkerboard-v1"}}
    for arm in ("random", "boundary"):
        report["arms"][arm] = {}
        for seed in ("0", "1", "2"):
            curve = []
            for budget in (128, 512):
                path = tmp_path / f"acq-{arm}-s{seed}-b{budget}" / "weights.pt"
                path.parent.mkdir()
                path.write_bytes(f"{arm}-{seed}-{budget}".encode())
                curve.append({"budget_added": budget,
                    "weights_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "hf_revision": "abc", "hf_repo": "private-model"})
            report["arms"][arm][seed] = {"curve": curve}
    return report


def test_plan_covers_each_arm_and_seed_without_ranking_retention(tmp_path):
    report = acquisition(tmp_path)
    plan = checkpoint_plan(report, tmp_path)
    assert len(plan) == 6
    assert {p["budget_added"] for p in plan} == {512}
    assert {(p["arm"], p["acquisition_seed"]) for p in plan} == {
        (arm, seed) for arm in ("random", "boundary") for seed in ("0", "1", "2")}
    Path(plan[0]["weights"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        checkpoint_plan(report, tmp_path)


def test_plan_rejects_incomplete_acquisition_or_missing_seed(tmp_path):
    report = acquisition(tmp_path)
    report["status"] = "diagnostic_complete"
    with pytest.raises(ValueError, match="completed qualified"):
        checkpoint_plan(report, tmp_path)
    report["status"] = "complete"
    del report["arms"]["boundary"]["2"]
    with pytest.raises(ValueError, match="same three"):
        checkpoint_plan(report, tmp_path)
