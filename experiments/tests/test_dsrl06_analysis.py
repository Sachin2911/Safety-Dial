"""The screen fails closed, with matched denominators and no shaping-only gain."""

import copy
import importlib.util
import json
from pathlib import Path
import sqlite3

import pytest

from helpers.dsrl06_protocol import ProtocolLedger, frozen_jobs


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


DIRECTORY = Path(__file__).resolve().parent
analysis = load("dsrl06_analysis_test", DIRECTORY.parent / "scripts/analyze_dsrl06.py")
validation_fixtures = load("dsrl06_analysis_fixture", DIRECTORY / "test_dsrl06_validation.py")


def scored_rows():
    rows = []
    for job in frozen_jobs()["evaluation_jobs"]:
        row = {**job, **{key: False for key in analysis.METRICS}}
        row.update(
            engaged=True,
            safety_evaluated=True,
            unfinished=True,
            policy_controls=2000,
            policy_queries=250,
            warmup_controls=10,
        )
        if job["arm"] == "dsrl" and job["block"] < 2:
            row.update(strict_safe_requested=True, safe_requested=True, requested_success=True)
        rows.append(row)
    return rows


def test_frozen_screen_requires_every_predeclared_criterion_and_panel():
    rows = scored_rows()
    assert analysis.frozen_screen(rows, True)["passed"]
    assert analysis.frozen_screen(rows[:-1], True)["passed"] is None
    assert analysis.frozen_screen(rows, False)["passed"] is None
    duplicate = copy.deepcopy(rows)
    duplicate[-1]["task"] = duplicate[-5]["task"]
    assert not analysis.frozen_screen(duplicate, True)["assessable"]
    for key, change in (
        ("no_more_raw_violations_than_native", {"raw_violation": True}),
        ("no_more_counted_violations_than_native", {"counted_violation": True}),
    ):
        changed = copy.deepcopy(rows)
        next(row for row in changed if row["arm"] == "dsrl").update(change)
        screen = analysis.frozen_screen(changed, True)
        assert not screen["passed"] and not screen["criteria"][key]
    changed = copy.deepcopy(rows)
    for row in [row for row in changed if row["arm"] == "dsrl"][:2]:
        row["engaged"] = False
    screen = analysis.frozen_screen(changed, True)
    assert screen["criteria"]["engagement_at_least_ten"]
    assert not screen["criteria"]["engagement_no_more_than_one_below_native"]


def test_aggregate_preserves_unscored_safety_and_actual_denominators():
    rows = scored_rows()[:1]
    rows[0].update(engaged=False, safety_evaluated=False)
    result = analysis.totals(rows, 12)
    assert result["denominator"] == 1 and result["expected"] == 12
    assert result["counts"]["safe_failure"] == 1 and result["unscored_safety"] == 1
    assert result["rates"]["raw_violation"] == 0
    assert result["nominal_wilson_95"]["raw_violation"][1] > 0.7
    empty = analysis.totals([], 12)
    assert empty["rates"]["strict_safe_requested"] is None
    assert empty["nominal_wilson_95"]["strict_safe_requested"] is None


def negative_authority_fixture(tmp_path):
    jobs = frozen_jobs()["engineering_jobs"][:4]
    ledger = ProtocolLedger(tmp_path / "resources.jsonl")
    for job in jobs:
        directory = tmp_path / "episodes" / f"{job['job_id']}.attempt1"
        directory.parent.mkdir(exist_ok=True)
        files = validation_fixtures.fixture(directory, job["job_id"])
        journal = analysis.read_events(files["journal"])
        journal[-1:-1] = [{"event": "before_action", "step": step} for step in range(10)]
        files["journal"].write_text("".join(json.dumps(row) + "\n" for row in journal))
        (directory / "job.json").write_text(json.dumps({**job, "attempt": 1}))
        built = validation_fixtures.build_episode(directory)
        (directory / "validation.json").write_text(json.dumps(built["validation"]))
        ledger.reserve_start(job["job_id"])
        ledger.finish_start(job["job_id"], 1, valid=True)
    ledger.charge("policy_controls", 40)
    ledger.charge("warmup_controls", 40)
    ledger.charge("learner_updates", 2, phase="engineering")
    ledger.charge("model_queries", 8)
    with sqlite3.connect(tmp_path / "queries.sqlite") as connection:
        connection.execute("CREATE TABLE queries(purpose TEXT,status TEXT)")
        connection.executemany("INSERT INTO queries VALUES (?,?)", [("policy", "returned")] * 8)
    (tmp_path / "stage-a-summary.json").write_text(
        json.dumps(
            {
                "admit": False,
                "authority": {
                    "valid": True,
                    "admit": False,
                    "reason": "authority_below_minimum",
                    "sensitive_snapshots": 0,
                },
            }
        )
    )


def test_negative_authority_is_a_completed_engineering_result(tmp_path):
    negative_authority_fixture(tmp_path)
    report = analysis.analyze(tmp_path)
    assert report["outcome"] == "engineering_negative_authority"
    assert not report["screen"]["assessable"]
    assert report["engineering"]["denominator"] == 4
    assert report["training"]["denominator"] == 0
    assert report["resources"]["source_observed"]["warmup_controls"] == 40
    assert report["resources"]["source_observed"]["engineering_updates"] == 2
    assert report["resources"]["source_protocol_mismatches"] == {}
    for episode in report["episodes"]:
        assert episode["corrected_discounted_shaped_return"] == pytest.approx(
            episode["discounted_base_return"]
        )
        assert episode["return_identity_error"] == pytest.approx(0)
    output = tmp_path / "report.json"
    analysis.write_report(report, output)
    assert json.loads(output.read_text())["outcome"] == report["outcome"]
    assert output.with_suffix(".md").exists()


def test_bad_source_hash_blocks_the_screen_instead_of_counting_as_failure(tmp_path):
    negative_authority_fixture(tmp_path)
    source = next((tmp_path / "episodes/E0.attempt1").glob("query_*.npz"))
    source.write_bytes(b"corrupted after validation")
    report = analysis.analyze(tmp_path)
    assert report["outcome"] == "invalid_integration"
    assert report["engineering"]["denominator"] == 3
    assert report["screen"]["passed"] is None
    assert "hash differs" in report["attempts_without_valid_receipts"][0]["error"]


def test_mutated_return_identity_is_invalid_integration(tmp_path):
    negative_authority_fixture(tmp_path)
    path = tmp_path / "episodes/E0.attempt1/validation.json"
    value = json.loads(path.read_text())
    value["transitions"][0]["reward"] += 1
    path.write_text(json.dumps(value))
    report = analysis.analyze(tmp_path)
    assert report["outcome"] == "invalid_integration"
    assert report["screen"]["passed"] is None


def test_imported_cpu_attempts_are_counted_once(tmp_path):
    path = tmp_path / "learner/learner-ledger.jsonl"
    path.parent.mkdir()
    rows = [
        {"event": "update_reserved", "phase": "engineering", "origin": "workstation_cpu_preflight"},
        {"event": "update_reserved", "phase": "engineering", "origin": "workstation_cpu_preflight"},
        {"event": "update_reserved", "phase": "engineering"},
        {"event": "update_reserved", "phase": "training"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = analysis.resource_report(tmp_path, 2)
    assert report["source_observed"]["engineering_updates"] == 3
    assert report["source_observed"]["learner_updates"] == 4
    assert report["actor_forward_requests"] == 130


def test_precision_amendment_adds_four_prior_controller_calls(tmp_path):
    stage_a = tmp_path / "stage-a-v2.json"
    stage_a.write_text(json.dumps({"amendment": {"prior_controller_actor_forward_calls": 134}}))
    report = analysis.analyze(tmp_path, stage_a)
    assert report["resources"]["workstation_actor_forward_requests"] == 134
    assert report["resources"]["actor_forward_requests"] == 134
    assert report["resources"]["source_observed"]["engineering_updates"] == 2


def test_early_checkpoint_or_actor_version_mismatch_cannot_be_final(tmp_path):
    checkpoint = tmp_path / "learner/checkpoints/episode12"
    checkpoint.mkdir(parents=True)
    state = {
        "episodes": [{"job_id": f"T{i:02}"} for i in range(24)],
        "actor_version": 6,
        "schema": "dsrl06-learner-checkpoint-v1",
        "phase": "training",
        "pending_transitions": 0,
    }
    path = checkpoint / "state.json"
    path.write_text(json.dumps(state))
    (checkpoint / "learner.msgpack").write_bytes(b"synthetic learner")
    (checkpoint / "replay.npz").write_bytes(b"synthetic replay")
    (checkpoint / "manifest.json").write_text(
        json.dumps({p.name: analysis.digest(p) for p in checkpoint.iterdir()})
    )
    queue = {"last_checkpoint": str(checkpoint)}
    assert not analysis.final_checkpoint_report(tmp_path, queue, [])["verified"]
    checkpoint = checkpoint.rename(checkpoint.with_name("episode24"))
    queue = {"last_checkpoint": str(checkpoint)}
    assert analysis.final_checkpoint_report(tmp_path, queue, [])["verified"]
    assert not analysis.final_checkpoint_report(
        tmp_path, queue, [{"arm": "dsrl", "actor_version": 5}]
    )["verified"]


def test_full_report_uses_fixed_cycles_and_the_final_actor(tmp_path, monkeypatch):
    jobs = frozen_jobs()
    rows = scored_rows()
    for row in rows:
        row["actor_version"] = 6
    for job in jobs["training_jobs"]:
        row = {**scored_rows()[0], **job}
        row.update(
            mode="gaussian" if int(job["job_id"][1:]) < 6 else "actor",
            actor_version=6,
            discounted_shaped_return=-0.05,
            discounted_base_return=0,
            corrected_discounted_shaped_return=0,
            base_reward_sum=0,
            shaping_reward_sum=0,
            undiscounted_reward_sum=0,
            cost=0,
        )
        rows.append(row)
    mapping = {row["job_id"]: row for row in rows}
    for job_id in mapping:
        directory = tmp_path / "episodes" / f"{job_id}.attempt1"
        directory.mkdir(parents=True)
        (directory / "job.json").write_text(json.dumps({"job_id": job_id}))
    monkeypatch.setattr(
        analysis, "episode", lambda directory, *_: mapping[directory.name.split(".")[0]]
    )
    observed = {
        "policy_controls": sum(row["policy_controls"] for row in rows),
        "warmup_controls": 720,
        "model_queries": sum(row["policy_queries"] for row in rows),
    }
    monkeypatch.setattr(
        analysis,
        "resource_report",
        lambda *_: {
            "source_observed": observed,
            "source_protocol_mismatches": {},
            "protocol_ledger": {"usage": observed},
        },
    )
    monkeypatch.setattr(
        analysis, "final_checkpoint_report", lambda *_: {"verified": True, "actor_version": 6}
    )
    report = analysis.analyze(tmp_path)
    assert report["outcome"] == "screen_passed_replication_required"
    assert report["screen"]["passed"]
    curves = report["training_curve_by_fixed_six_task_cycle"]
    assert len(curves) == 4 and all(row["complete_six_task_cycle"] for row in curves)
    assert [row["gaussian_warmup_episodes"] for row in curves] == [6, 0, 0, 0]
    assert all(row["corrected_discounted_shaped_return"] == 0 for row in curves)
    assert len(report["task_seed_block_differences_against_native"]) == 36
    assert report["evaluation"]["dsrl"]["denominator"] == 12
    output = tmp_path / "report.json"
    analysis.write_report(report, output)
    assert output.with_name("report-training.csv").exists()
