"""Prevent benchmark-goal shortcuts and incomplete studies becoming success claims."""

import json

from helpers import maniguard_latent_report as report


def test_partial_report_preserves_requested_sequence_and_failed_start_cost(tmp_path, monkeypatch):
    output = tmp_path / "episode"
    output.mkdir()
    (output / "job.json").write_text(json.dumps({"family": "jar", "scene": "task_0013/base"}))
    (output / "ledger.json").write_text(json.dumps({"policy_control_steps": 666,
                                                    "policy_queries": 84}))
    (tmp_path / "queue.json").write_text(json.dumps({
        "attempts": [{"id": "dev-a0", "job_id": "dev"}],
        "jobs": {"dev": {"definition": {}}}}))
    monkeypatch.setattr(report, "snapshot", lambda run: {
        "attempts": [{"id": "failed-start", "stage": "engineering", "status": "invalid",
                      "output": str(tmp_path / "failed")},
                     {"id": "dev-a0", "stage": "development", "status": "completed",
                      "output": str(output)}], "recorded_totals": {"policy_control_steps": 666}})
    monkeypatch.setattr(report, "read_episode", lambda *a, **k: {
        "score": -2, "outcome": {"C_req": False, "V": True, "benchmark_success": True}})
    result = report.export(tmp_path, tmp_path / "analysis")
    failed, completed = result["attempts"]
    assert failed["physics_steps"] is None
    assert failed["physics_before_callback"] == "unknown"
    assert completed["requested_category"] == "unsafe_unfinished"
    assert completed["benchmark_category"] == "unsafe_completion"
    assert result["outcome_counts"][0]["n"] == 1
    assert result["outcome_counts"][0]["unsafe_unfinished"] == 1
    assert not result["physical_comparison_complete"]
    assert not result["full_goal_completion_assessed"]
    assert len((tmp_path / "analysis/attempts.csv").read_text().splitlines()) == 3


def test_method_cost_curves_include_failed_and_unobserved_attempts(tmp_path):
    search = tmp_path / "search"
    search.mkdir()
    candidate = {"id": "initial-0", "generation": 0, "scores": [-2, 0]}
    (search / "state.json").write_text(json.dumps({
        "optimizers": {method: {"candidates": [candidate]} for method in ("ea", "random")}}))
    failed = tmp_path / "partial"
    failed.mkdir()
    (failed / "ledger.json").write_text(json.dumps({"policy_control_steps": 17,
                                                  "policy_queries": 3}))
    queue = {"attempts": [{"id": "failed", "output": str(failed)},
                          {"id": "unobserved", "output": str(tmp_path / "missing")}],
             "jobs": {"initial-0-case0": {"attempt_ids": ["unobserved", "failed", "valid0"]},
                      "initial-0-case1": {"attempt_ids": ["valid1"]}}}
    verified = {"valid0": {"score": -2, "physical": {"ledger": {
        "policy_control_steps": 100, "policy_queries": 13}}},
        "valid1": {"score": 0, "physical": {"ledger": {
            "policy_control_steps": 2000, "policy_queries": 250}}}}
    curves = report.search_curves(tmp_path, queue, verified)
    assert len(curves) == 2  # The shared acquisition belongs to both information budgets.
    for row in curves:
        assert row["candidate_controls"] == row["cumulative_controls"] == 2117
        assert row["candidate_valid_controls"] == 2100
        assert row["candidate_invalid_controls"] == 17
        assert row["cumulative_policy_queries"] == 266
        assert row["cumulative_unknown_control_attempts"] == 1
        assert row["cumulative_unknown_query_attempts"] == 1
