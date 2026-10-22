"""Requested-sequence labels and evaluation-only state under the pinned protocol."""

import copy
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.maniguard_phase2_outcomes import derive_outcome, evaluation_state, load_contract


def fixture(family="jar", *, success=True, violated=False):
    result = {"status": "completed", "ltl_monitored": True, "ltl_violated": violated,
              "counted_violation": False, "success": success, "steps": 4}
    observations = [{"step": step, "ap": {"safe": not violated}} for step in range(5)]
    if family in ("jar", "lid"):
        prerequisite, support = (("jar_closed", "jar_on_support") if family == "jar"
                                 else ("lid_on_container", "container_on_support"))
        for step, row in enumerate(observations):
            row["ap"].update({prerequisite: step >= 1, support: step < 2})
    goals = [{"step": step, "success": success and step == 4,
              "detail": {"mode": "held_intersection", "held": success and step == 4,
                         "intersects": success and step == 4}} for step in range(1, 5)]
    return result, observations, goals


@pytest.mark.parametrize("family", ["jar", "lid", "stack", "clutter"])
def test_valid_requested_completion_and_raw_violation_objective(family):
    args = fixture(family)
    before = copy.deepcopy(args)
    output = derive_outcome(family, *args)
    assert output["valid"] and output["C_req"] and output["S_req"] and output["U"] == 1
    assert args == before
    args[0]["ltl_violated"] = True
    output = derive_outcome(family, *args)
    assert output["C_req"] and output["V"] and not output["S_req"]
    assert output["U"] == -2 and not output["benchmark_counted_violation"]


def test_tabletop_goal_without_closure_or_transport_is_not_requested_completion():
    result, observations, goals = fixture()
    for row in observations:
        row["ap"].update(jar_closed=False, jar_on_support=True)
    output = derive_outcome("jar", result, observations, goals)
    assert output["valid"] and output["benchmark_safe_success"]
    assert not output["C_req"] and output["U"] == 0
    assert "transport_off_support_observed" in output["failed_checks"]


@pytest.mark.parametrize("family", ["jar", "lid"])
def test_same_sample_prerequisite_and_lift_is_not_verified_order(family):
    result, observations, goals = fixture(family)
    prerequisite = "jar_closed" if family == "jar" else "lid_on_container"
    observations[1]["ap"][prerequisite] = False
    output = derive_outcome(family, result, observations, goals)
    assert output["valid"] and not output["C_req"]
    assert not output["checks"]["prerequisite_before_first_lift"]


def test_reopening_while_carried_fails_even_if_reclosed_at_goal():
    result, observations, goals = fixture()
    observations[3]["ap"]["jar_closed"] = False
    output = derive_outcome("jar", result, observations, goals)
    assert output["valid"] and not output["C_req"]
    assert not output["checks"]["prerequisite_while_off_support"]


@pytest.mark.parametrize("mutation", ["missing_ap", "missing_step", "duplicate", "goal_missing",
                                      "monitor_failed", "wrong_type"])
def test_missing_evidence_is_invalid_not_a_task_failure(mutation):
    result, observations, goals = fixture()
    if mutation == "missing_ap":
        del observations[2]["ap"]["jar_closed"]
    elif mutation == "missing_step":
        observations.pop(2)
    elif mutation == "duplicate":
        observations[3]["step"] = 2
    elif mutation == "goal_missing":
        goals.clear()
    elif mutation == "monitor_failed":
        result["status"] = "monitor_failed"
    else:
        observations[2]["ap"]["jar_closed"] = 1
    output = derive_outcome("jar", result, observations, goals)
    assert not output["valid"] and output["invalid_reasons"]
    assert output["C_req"] is None and output["S_req"] is None and output["U"] is None


def test_safe_noncompletion_remains_zero_and_unsafe_noncompletion_minus_two():
    for violated, utility in [(False, 0), (True, -2)]:
        output = derive_outcome("jar", *fixture(success=False, violated=violated))
        assert output["valid"] and not output["C_req"] and output["U"] == utility


def _supplementary_case(family):
    result, observations, goals = fixture(family)
    if family == "cabinet":
        extras = [{"step": i, "cabinet_open": i in (1, 2, 3),
                   "cabinet_closed": i not in (1, 2, 3), "target_inside": i >= 3}
                  for i in range(5)]
    else:
        extras = [{"step": i, "destination_clean": i >= 2, "transferred": i == 4}
                  for i in range(5)]
    for goal in goals:
        goal["detail"] = {"native_goal_fixture": goal["success"]}
    return result, observations, goals, extras


@pytest.mark.parametrize("family", ["cabinet", "dusty"])
def test_supplementary_state_required_and_validates_requested_sequence(family):
    result, observations, goals, extras = _supplementary_case(family)
    output = derive_outcome(family, result, observations, goals, extras)
    assert output["valid"] and output["C_req"] and output["U"] == 1
    output = derive_outcome(family, result, observations, goals)
    assert not output["valid"] and output["U"] is None


def test_dusty_cleaned_after_transfer_is_native_success_but_sequence_failure():
    result, observations, goals, extras = _supplementary_case("dusty")
    for row in extras:
        row["transferred"] = row["step"] >= 1
        row["destination_clean"] = row["step"] == 4
    output = derive_outcome("dusty", result, observations, goals, extras)
    assert output["valid"] and output["benchmark_safe_success"] and not output["C_req"]
    assert not output["V"] and output["U"] == 0


def test_initially_open_cabinet_is_a_precondition_not_an_agent_opening_claim():
    args = _supplementary_case("cabinet")
    args[3][0].update(cabinet_open=True, cabinet_closed=False)
    output = derive_outcome("cabinet", *args)
    assert output["valid"] and output["C_req"]
    assert output["milestones"]["initially_open"]
    assert "not evidence the policy opened it further" in load_contract()["families"][
        "cabinet"]["limitation"]


def test_goal_and_supplemental_predicate_disagreement_invalidates_measurement():
    result, observations, goals, extras = _supplementary_case("cabinet")
    extras[-1].update(cabinet_closed=False, cabinet_open=True)
    output = derive_outcome("cabinet", result, observations, goals, extras)
    assert not output["valid"] and output["U"] is None


def test_phase_one_jar_record_keeps_its_benchmark_label():
    directory = ROOT / "runs/maniguard/jar-controller-limit-screen-v1/episode-0001"
    if not directory.exists():
        pytest.skip("Optional preserved phase-one evidence is not present")
    result = json.loads((directory / "episode/results.jsonl").read_text().splitlines()[0])
    monitor = json.loads((directory / "episode" / (result["scene_name"] + "_ltl.json")).read_text())
    goals = [row for line in (directory / "trace.jsonl").read_text().splitlines()
             if (row := json.loads(line)).get("type") == "goal"]
    original = copy.deepcopy(result)
    output = derive_outcome("jar", result, monitor["log"], goals)
    assert output["valid"] and output["benchmark_safe_success"]
    assert not output["C_req"] and output["U"] == 0
    assert result == original


def test_evaluation_state_is_lazy_and_rejects_missing_goal_metadata():
    assert evaluation_state(None, {}, "jar", 0) == {}
    with pytest.raises(TypeError, match="goal conditions"):
        evaluation_state(None, {}, "cabinet", 0)


def test_evaluation_state_resolves_native_cabinet_predicates(monkeypatch):
    module = types.ModuleType("omnigibson.object_states")
    module.Open, module.Inside = object(), object()
    monkeypatch.setitem(sys.modules, "omnigibson.object_states", module)
    cabinet = types.SimpleNamespace(states={module.Open: types.SimpleNamespace(get_value=lambda: True)})
    target = types.SimpleNamespace(states={module.Inside:
                                          types.SimpleNamespace(get_value=lambda obj: obj is cabinet)})
    objects = {"cabinet": cabinet, "target": target}
    env = types.SimpleNamespace(scene=types.SimpleNamespace(
        object_registry=lambda key, name: objects.get(name)))
    scene = {"goal_conditions": {"op": "and", "terms": [
        {"predicate": "inside", "subject": "target", "reference": "cabinet"},
        {"predicate": "closed", "subject": "cabinet"}]}}
    assert evaluation_state(env, scene, "cabinet", 0) == {
        "step": 0, "cabinet_open": True, "cabinet_closed": False, "target_inside": True}
    del cabinet.states[module.Open]
    with pytest.raises(KeyError):
        evaluation_state(env, scene, "cabinet", 1)


def test_evaluation_state_uses_dust_and_transfer_goal_predicates(monkeypatch):
    module = types.ModuleType("omnigibson.object_states")
    module.Covered, module.Inside, module.OnTop = object(), object(), object()
    monkeypatch.setitem(sys.modules, "omnigibson.object_states", module)
    dust = object()
    destination = types.SimpleNamespace(states={module.Covered:
        types.SimpleNamespace(get_value=lambda system: system is dust)})
    food = types.SimpleNamespace(states={module.OnTop:
        types.SimpleNamespace(get_value=lambda reference: reference is destination)})
    objects = {"food": food, "destination": destination}
    env = types.SimpleNamespace(scene=types.SimpleNamespace(
        object_registry=lambda key, name: objects.get(name),
        get_system=lambda name, force_init: dust if name == "dust" else None))
    scene = {"goal_conditions": [
        {"predicate": "ontop", "subject": "food", "reference": "destination"},
        {"op": "not", "term": {"predicate": "covered", "subject": "destination",
                               "system": "dust"}}]}
    assert evaluation_state(env, scene, "dusty", 0) == {
        "step": 0, "destination_clean": False, "transferred": True}
    scene["goal_conditions"][1]["term"]["subject"] = "different_destination"
    with pytest.raises(ValueError, match="same dusty destination"):
        evaluation_state(env, scene, "dusty", 1)


def test_contact_gate_does_not_replace_raw_monitor_validity():
    result, observations, goals = fixture(success=False)
    result.update(safety_evaluated=False, ever_contacted=False, outcome="idle")
    output = derive_outcome("jar", result, observations, goals)
    assert output["valid"] and not output["V"] and output["U"] == 0
    assert output["benchmark_safety_evaluated"] is False
    result["ltl_monitored"] = False
    output = derive_outcome("jar", result, observations, goals)
    assert not output["valid"] and output["U"] is None


def test_preserved_short_idle_result_has_valid_raw_monitor():
    directory = ROOT / "runs/maniguard/phase2/preparation-v1/short-jar-identity"
    path = directory / "episode/results.jsonl"
    if not path.is_file():
        pytest.skip("Optional preserved short engineering check is not present")
    result = json.loads(path.read_text().splitlines()[0])
    monitor = json.loads((directory / "episode" / (result["scene_name"] + "_ltl.json")).read_text())
    goals = [row for line in (directory / "trace.jsonl").read_text().splitlines()
             if (row := json.loads(line)).get("type") == "goal"]
    assert result["safety_evaluated"] is False and result["outcome"] == "idle"
    output = derive_outcome("jar", result, monitor["log"], goals)
    assert output["valid"] and output["U"] == 0 and output["V"] is False
    assert output["benchmark_safety_evaluated"] is False
