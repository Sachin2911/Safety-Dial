"""Synthetic attribution cases with actual stream replay and no physical execution."""

import copy
import json

import pytest
from test_maniguard_continuation_evidence import response, stream

from helpers.maniguard_continuation_evidence import choice_record, replay_stream
from helpers.maniguard_continuation_handoff import HandoffReference
from helpers.maniguard_continuation_progress import attribute_progress
from helpers.maniguard_latent import initial_weights
from helpers.maniguard_phase2_outcomes import derive_outcome
from helpers.maniguard_runtime import json_value


def fixture(tmp_path, *, mode="clock_handoff", grasp=10, lift=None, goal=None,
            violation=None, clock=8):
    """Synthetic full control trace plus a declared receipt, not a physical certificate."""
    rows = stream(mode)
    selector = HandoffReference(initial_weights()[1], mode=mode, clock_switch_step=clock)
    complete, choice, features = [], None, None
    for row in rows:
        step = row.get("step")
        if row["type"] == "continuation_contract":
            row["contract"] = selector.contract
        elif row["type"] == "observed":
            state = row["monitor"]["observation"]
            state["ap"]["jar_on_support"] = lift is None or step < lift
            state["doomed"] = violation is not None and step >= violation
            if selector.privileged:
                selector.observe(step, closed=state["ap"]["jar_closed"],
                                 supported=state["ap"]["jar_on_support"], violated=state["doomed"])
        elif row["type"] == "latent_features":
            features = row["features"]
            choice = choice_record(selector.select(step, features))
            row["code"] = choice["code"]
            row["selected_controller_sha256"] = choice["controller_sha256"]
        elif row["type"] == "continuation_choice":
            row["choice"] = choice
        elif row["type"] == "continuation_request":
            row["request"]["controller_sha256"] = choice["controller_sha256"]
        elif row["type"] == "noise_receipt":
            result = response(choice, features, step // 8)
            row["receipt"] = result["latent_receipt"]
        complete.append(row)
        if row["type"] == "observed" and step:
            held = grasp is not None and step >= grasp
            intersects = goal is not None and step >= goal
            complete.append({"type": "goal", "step": step, "success": held and intersects,
                             "detail": {"mode": "held_intersection", "held": held,
                                        "intersects": intersects}})
    proof = replay_stream(complete, initial_weights()[1], mode=mode, debounce=3,
                          clock_switch_step=clock, episode="synthetic", seed=123, steps=16)
    switch = proof["state"]["switch_query_step"]
    proof["actual_post_switch_controls"] = 0 if switch is None else 16 - switch
    observations = [{"step": row["step"], **row["monitor"]["observation"]}
                    for row in complete if row["type"] == "observed"]
    goals = [row for row in complete if row["type"] == "goal"]
    result = {"status": "completed", "steps": 16, "ltl_monitored": True,
              "ltl_violated": violation is not None, "success": goals[-1]["success"]}
    outcome = derive_outcome("jar", result, observations, goals)
    assert outcome["valid"]
    receipt = {"valid": True, "full_episode": True, "physical": {
        "family": "jar", "steps": 16, "output": str(tmp_path)}, "outcome": outcome,
        "score": outcome["U"], "stream": proof,
        "continuation_contract_sha256": proof["contract_sha256"],
        "closure_controller_sha256": selector.closure.sha256()}
    path = tmp_path / "trace.jsonl"
    path.write_text("".join(json.dumps(row, default=json_value) + "\n" for row in complete))
    return path, receipt


@pytest.mark.parametrize("grasp,relation,eligible", [(1, "before_switch", False),
                                                   (8, "at_switch", False),
                                                   (10, "after_switch", True)])
def test_first_grasp_is_attributed_strictly_after_the_actual_switch(
    tmp_path, grasp, relation, eligible,
):
    result = attribute_progress(*fixture(tmp_path, grasp=grasp))
    assert result["whole_ordered_stage"] == 2
    assert result["first_ordered_milestones"]["2"] == {
        "first_observation": grasp, "controls_from_switch": grasp - 8, "relation": relation}
    assert result["stage_at_actual_switch"] == (2 if grasp <= 8 else 1)
    assert result["nonprivileged_handoff_progress_eligible"] is eligible
    assert not result["privileged_engineering_progress_eligible"]
    assert result["actual_post_switch_controls"] == 8
    assert not result["dispatch_authorized"] and not result["expansion_authorized"]


def test_a_new_lift_counts_after_a_grasp_already_present_at_switch(tmp_path):
    result = attribute_progress(*fixture(tmp_path, grasp=1, lift=10))
    assert result["stage_at_actual_switch"] == 2 and result["whole_ordered_stage"] == 3
    assert result["new_ordered_milestones_after_switch"] == [3]
    assert result["first_ordered_milestones"]["3"]["controls_from_switch"] == 2
    assert result["nonprivileged_handoff_progress_eligible"]


def test_new_lift_and_transport_preserve_requested_completion_and_relative_times(tmp_path):
    result = attribute_progress(*fixture(tmp_path, grasp=1, lift=10, goal=12))
    assert result["new_ordered_milestones_after_switch"] == [3, 4]
    assert result["first_ordered_milestones"]["4"]["controls_from_switch"] == 4
    assert result["safe_requested_completion"] and result["raw_completion"]
    assert result["nonprivileged_handoff_progress_eligible"]


def test_raw_goal_success_without_lift_stays_incomplete_requested_progress(tmp_path):
    result = attribute_progress(*fixture(tmp_path, grasp=10, goal=12))
    assert result["raw_completion"] and not result["requested_completion"]
    assert result["whole_ordered_stage"] == 2
    assert result["new_ordered_milestones_after_switch"] == [2]


@pytest.mark.parametrize("mode,clock", [("native_only", 8), ("closure_only", 8),
                                      ("clock_handoff", 24), ("clock_handoff", 0)])
def test_no_switch_or_no_closure_prefix_never_qualifies_a_handoff(tmp_path, mode, clock):
    trace, receipt = fixture(tmp_path, mode=mode, clock=clock, grasp=1, lift=10, goal=12)
    result = attribute_progress(trace, receipt)
    assert result["safe_requested_completion"] and result["whole_ordered_stage"] == 4
    assert not result["actual_handoff_observed"]
    prefix = 16 if clock == 24 or mode == "closure_only" else 0
    assert result["closure_prefix_controls"] == prefix
    assert not result["nonprivileged_handoff_progress_eligible"]
    assert not result["privileged_engineering_progress_eligible"]


def test_later_violation_removes_eligibility_without_erasing_earlier_progress(tmp_path):
    result = attribute_progress(*fixture(tmp_path, grasp=10, violation=15))
    assert result["new_ordered_milestone_strictly_after_switch"]
    assert result["whole_ordered_stage"] == 2 and result["whole_terminal_secondary"] == 0
    assert result["physical_violation"] and not result["nonprivileged_handoff_progress_eligible"]


def test_violation_before_switch_cannot_earn_later_progress_credit(tmp_path):
    result = attribute_progress(*fixture(tmp_path, grasp=10, violation=5))
    assert result["actual_handoff_observed"]
    assert not result["new_ordered_milestone_strictly_after_switch"]
    assert not result["nonprivileged_handoff_progress_eligible"]


def test_privileged_progress_is_only_an_engineering_reference(tmp_path):
    result = attribute_progress(*fixture(tmp_path, mode="privileged_lid_handoff"))
    assert result["new_ordered_milestones_after_switch"] == [2]
    assert result["privileged_engineering_progress_eligible"]
    assert not result["nonprivileged_handoff_progress_eligible"]


@pytest.mark.parametrize("change", ["partial", "contract", "switch", "post_controls", "mode"])
def test_receipt_and_actual_full_trace_cannot_be_misbound(tmp_path, change):
    path, original = fixture(tmp_path)
    receipt = copy.deepcopy(original)
    if change == "partial":
        receipt["full_episode"] = False
    elif change == "contract":
        receipt["continuation_contract_sha256"] = "a" * 64
    elif change == "switch":
        receipt["stream"]["state"]["switch_query_step"] = 0
        receipt["stream"]["actual_post_switch_controls"] = 16
    elif change == "post_controls":
        receipt["stream"]["actual_post_switch_controls"] = 1
    else:
        receipt["stream"]["mode"] = "privileged_lid_handoff"
    with pytest.raises(ValueError):
        attribute_progress(path, receipt)
