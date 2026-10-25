"""The follow-up screen uses fresh seeds and requires evidence beyond numerics."""

import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_dsrl07 as analysis
from helpers.dsrl07_protocol import frozen_jobs


def rows():
    result = []
    for job in frozen_jobs()["evaluation_jobs"]:
        row = {**job, **{key: False for key in analysis.METRICS}}
        success = job["arm"] == "dsrl" and job["block"] < 2
        row.update(
            engaged=True,
            safety_evaluated=True,
            unfinished=not success,
            success=success,
            requested_success=success,
            strict_safe_requested=success,
            safe_requested=success,
            policy_controls=2000,
            policy_queries=250,
            warmup_controls=10,
        )
        result.append(row)
    return result


def test_comparison_requires_complete_fresh_seed_panel():
    data = rows()
    assert analysis.frozen_screen(data, True)["passed"]
    assert not analysis.frozen_screen(data[:-1], True)["assessable"]
    old = copy.deepcopy(data)
    for row in old:
        row["seed"] -= 10000
    assert not analysis.frozen_screen(old, True)["assessable"]
    paired = analysis.paired(data)
    assert len(paired) == 36
    assert {row["seed"] for row in paired} == {72000, 72001}


def test_empty_followup_does_not_inherit_historical_interactions(tmp_path):
    report = analysis.analyze(tmp_path)
    assert report["outcome"] == "engineering_incomplete"
    assert report["resources"]["source_observed"]["learner_updates"] == 0
    assert report["resources"]["actor_forward_requests"] == 0
    assert report["training"]["denominator"] == 0


def test_numerical_failure_is_not_a_negative_learning_result(tmp_path):
    (tmp_path / "numerical-summary.json").write_text(json.dumps({"admit_live_integration": False}))
    report = analysis.analyze(tmp_path)
    assert report["outcome"] == "numerical_integration_failed"
    assert not report["screen"]["assessable"]
    assert not report["integration_pass"]


def test_positive_summary_without_six_live_records_is_not_integration_pass(tmp_path):
    (tmp_path / "numerical-summary.json").write_text(json.dumps({"admit_live_integration": True}))
    (tmp_path / "stage-a-summary.json").write_text(
        json.dumps(
            {"admit": True, "source_parity_valid": True, "cpu_actor_parity": {"valid": True}}
        )
    )
    report = analysis.analyze(tmp_path)
    assert not report["integration_pass"]
    assert not report["screen"]["assessable"]
