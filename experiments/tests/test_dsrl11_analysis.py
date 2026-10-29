"""The task-11 analysis re-verifies episodes and applies the frozen screen and contrasts."""

import importlib.util
import json
from pathlib import Path

import pytest
from test_dsrl11_validation import fixture

from helpers.dsrl11_protocol import CONDITIONS, Dsrl11Ledger, frozen_jobs
from helpers.dsrl11_validation import build_residual_episode

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "analyze_dsrl11", ROOT / "experiments/scripts/analyze_dsrl11.py"
)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def run_dir(tmp_path, jobs):
    run = tmp_path / "run"
    episodes = {}
    for job_id in jobs:
        directory = run / "episodes" / f"{job_id}.attempt1"
        directory.parent.mkdir(parents=True, exist_ok=True)
        fixture(directory, job_id)
        built = build_residual_episode(directory, directory / "learner.npz", job_schedule=None)
        (directory / "validation.json").write_text(json.dumps(built["validation"]))
        episodes[job_id] = {"directory": str(directory), "valid": True, "attempt": 1}
    (run / "queue-state.json").write_text(json.dumps({"episodes": episodes}))
    Dsrl11Ledger(run / "resources.jsonl")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"jobs": frozen_jobs(), "prior_engineering_updates": 21}))
    return run, plan


def test_analysis_reverifies_and_reports_every_arm(tmp_path):
    jobs = ["V0_native", *[f"V0_{c}" for c in CONDITIONS], "B00"]
    run, plan = run_dir(tmp_path, jobs)
    report = analysis.analyze(run, plan)
    assert report["episodes_analyzed"] == 6 and report["invalid_attempts"] == []
    for arm in analysis.ARMS:
        table = report["evaluation"][arm]
        assert table["denominator"] == 1 and table["counts"]["strict_safe_requested"] == 1
    assert report["bank"]["denominator"] == 1
    screen = report["continuation_screen"]
    assert not screen["complete"] and not screen["passed"]
    contrast = report["contrasts"]["r1_minus_r0_with_prior"]
    assert contrast["roots"] == 1 and contrast["strict_safe_requested"]["difference"] == 0
    assert report["resources"]["prior_engineering_updates"] == 21
    row = next(r for r in report["rows"] if r["job_id"] == "V0_r1_prior")
    assert row["residual"]["source"] == "actor" and row["milestones"]["first_grasp_step"] == 1


def test_a_changed_artifact_is_never_counted(tmp_path):
    run, plan = run_dir(tmp_path, ["V0_native"])
    query = next((run / "episodes/V0_native.attempt1").glob("query_*.npz"))
    query.write_bytes(query.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash differs"):
        analysis.analyze(run, plan)


def test_screen_requires_the_frozen_margins_and_reports_ceilings():
    def rows(strict, engaged=8, raw=0):
        return [
            {
                **{key: False for key in analysis.METRICS},
                "strict_safe_requested": i < strict,
                "engaged": i < engaged,
                "raw_violation": i < raw,
                "policy_controls": 1,
                "policy_queries": 1,
                "milestones": {
                    "first_contact_step": None,
                    "first_grasp_step": None,
                    "success_step": None,
                },
                "geometry": {"maximum_lift": 0.0, "minimum_reach_distance_m": 0.1},
            }
            for i in range(8)
        ]

    groups = {arm: rows(1) for arm in analysis.ARMS}
    groups["r1_prior"] = rows(3)
    assert analysis.screen(groups)["passed"]
    groups["r1_prior"] = rows(3, engaged=6)
    assert not analysis.screen(groups)["passed"]
    groups = {arm: rows(7) for arm in analysis.ARMS}
    result = analysis.screen(groups)
    assert result["ceiling_limited"] and not result["passed"]
