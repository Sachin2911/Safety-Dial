"""Protect factual command use, milestone order and excluded-output boundaries."""

import copy
import json

import pytest

from helpers.maniguard_review_progress import analyze, episode_progress


def trace_rows(*, stage=4, violated=False):
    rows = []
    for step in range(5):
        closed = step >= 1 and stage >= 1
        supported = step < 3 or stage < 3
        held = step >= 2 and stage >= 2
        if step:
            rows.extend([
                {"type": "proposal", "step": step - 1, "adapted": [0.] * 7 + [-1.]},
                {"type": "executed", "step": step - 1,
                 "action": [0.] * 7 + [-1. if held else 1.],
                 "action_scope": "environment_input_before_controller_output_clipping"},
            ])
        rows.append({"type": "observed", "step": step, "monitor": {
            "valid": True, "observation": {"doomed": violated and step >= 3,
                                            "ap": {"jar_closed": closed,
                                                   "jar_on_support": supported}}}})
        if step:
            rows.append({"type": "goal", "step": step,
                         "detail": {"held": held, "intersects": step == 4 and stage == 4}})
    return rows


def receipt(*, stage=4, violated=False):
    completed = stage == 4
    return {"valid": True, "full_episode": True, "controller_sha256": "synthetic",
            "physical": {"steps": 4}, "score": int(completed and not violated) - 2 * int(violated),
            "outcome": {"C_req": completed, "S_req": completed and not violated,
                        "V": violated, "benchmark_success": completed}}


def write_trace(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_actual_ordered_closure_grasp_lift_transport_uses_observation_steps(tmp_path):
    path = write_trace(tmp_path / "trace.jsonl", trace_rows())
    result = episode_progress(path, receipt())
    assert result["ordered_stage_terminal_secondary"] == 4
    assert result["ordered_stage_first_observations"] == {"1": 1, "2": 2, "3": 3, "4": 4}
    assert result["first_closed_supported_before_violation"] == 1
    assert result["first_held_observation"] == 2
    assert result["executed_close_commands"] == 3
    assert result["executed_close_commands_after_safe_supported_closure"] == 2


def test_closing_proposals_do_not_become_executed_gripper_evidence(tmp_path):
    path = write_trace(tmp_path / "trace.jsonl", trace_rows(stage=1))
    result = episode_progress(path, receipt(stage=1))
    assert result["ordered_stage_terminal_secondary"] == 1
    assert result["executed_close_commands"] == 0
    assert result["executed_close_commands_after_safe_supported_closure"] == 0
    assert result["first_held_observation"] is None


def test_later_violation_removes_secondary_fitness_without_erasing_factual_early_closure(tmp_path):
    path = write_trace(tmp_path / "trace.jsonl", trace_rows(stage=1, violated=True))
    result = episode_progress(path, receipt(stage=1, violated=True))
    assert result["ordered_stage_before_any_violation"] == 1
    assert result["ordered_stage_terminal_secondary"] == 0
    assert result["first_closed_supported_before_violation"] == 1
    assert result["first_doomed_observation"] == 3


def test_lift_before_closure_cannot_be_repaired_into_ordered_transport(tmp_path):
    rows = trace_rows()
    for row in rows:
        if row["type"] == "observed" and row["step"]:
            observation = row["monitor"]["observation"]
            observation["ap"]["jar_on_support"] = False
            observation["ap"]["jar_closed"] = row["step"] >= 2
            observation["doomed"] = True
    result = episode_progress(write_trace(tmp_path / "trace.jsonl", rows), receipt(violated=True))
    assert result["ordered_stage_terminal_secondary"] == 0
    assert result["ordered_stage_before_any_violation"] == 0
    assert result["first_closed_supported_before_violation"] is None


def test_missing_executed_command_and_receipt_disagreement_are_rejected(tmp_path):
    rows = [row for row in trace_rows() if not (row["type"] == "executed" and row["step"] == 2)]
    path = write_trace(tmp_path / "trace.jsonl", rows)
    with pytest.raises(ValueError, match="Incomplete or unordered"):
        episode_progress(path, receipt())
    write_trace(path, trace_rows())
    wrong = copy.deepcopy(receipt())
    wrong["score"] = 0
    with pytest.raises(ValueError, match="disagrees with the receipt"):
        episode_progress(path, wrong)


def test_independent_checking_output_is_excluded_before_opening(tmp_path):
    (tmp_path / "queue.json").write_text(json.dumps({"attempts": [{
        "id": "protected", "stage": "independent_audit", "status": "completed",
        "output": str(tmp_path / "must-not-be-opened"),
    }]}))
    result = analyze(tmp_path)
    assert result["episodes"] == []
    assert result["excluded_attempt_counts_by_stage"] == {"independent_audit": 1}


def test_same_terminal_score_can_hide_progress_without_changing_selection(tmp_path):
    attempts = []
    for case, stage in enumerate([1, 0]):
        name = f"candidate-case{case}-a0"
        output = tmp_path / name
        output.mkdir()
        write_trace(output / "trace.jsonl", trace_rows(stage=stage))
        (output / "verified-receipt.json").write_text(json.dumps(receipt(stage=stage)))
        (output / "job.json").write_text(json.dumps({
            "family": "jar", "scene": f"task_{case}/base", "seed": case,
            "controller_sha256": "synthetic",
        }))
        attempts.append({"id": name, "stage": "shared_initial", "status": "completed",
                         "output": str(output)})
    (tmp_path / "queue.json").write_text(json.dumps({"attempts": attempts}))
    result = analyze(tmp_path)
    panel = result["complete_two_case_training_panels"][0]
    assert panel["original_scores"] == [0, 0] and panel["original_score_sum"] == 0
    assert panel["terminal_ordered_stages"] == [1, 0]
    assert result["comparison_only"]["safe_unfinished_stage_histogram"] == {"0": 1, "1": 1}
    assert result["comparison_only"]["safe_requested_completions"] == 0
