from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.managedWorkflow import WorkflowError, run_workflow


WRITE = "from pathlib import Path; import sys; Path(sys.argv[1]).write_text(sys.argv[2]); print('stage complete')"


def make_stage(name, output, value="done", **extra):
    return {"id": name, "argv": [sys.executable, "-c", WRITE, output, value],
            "timeout_seconds": 5, "fresh_outputs": [output], "outputs": [output], **extra}


def config_file(tmp_path, stages, **extra):
    config = {"version": 1, "workflow_id": "test-workflow", "working_directory": str(tmp_path),
              "state_dir": "workflow-state", "poll_interval_seconds": .01,
              "wait_timeout_seconds": 1, "stages": stages, **extra}
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_completed_stages_resume_only_after_hash_verification(tmp_path):
    path = config_file(tmp_path, [make_stage("one", "one.txt"), make_stage("two", "two.txt")])
    result = run_workflow(path, tmp_path)
    assert result["status"] == "completed"
    result = run_workflow(path, tmp_path, resume=True)
    assert len(result["stages"]["one"]["attempts"]) == 1
    assert len(result["stages"]["two"]["attempts"]) == 1
    assert len(list((tmp_path / "workflow-state/logs").glob("*.log"))) == 2
    (tmp_path / "one.txt").write_text("tampered")
    with pytest.raises(WorkflowError, match="output hash changed"):
        run_workflow(path, tmp_path, resume=True)
    assert (tmp_path / "one.txt").read_text() == "tampered"


def test_configuration_hash_cannot_change_under_existing_state(tmp_path):
    path = config_file(tmp_path, [make_stage("one", "one.txt")])
    run_workflow(path, tmp_path)
    path.write_text(path.read_text() + "\n# changed config\n")
    with pytest.raises(WorkflowError, match="configuration or working path changed"):
        run_workflow(path, tmp_path, resume=True)


def test_argv_metacharacters_are_literal_and_never_shell_executed(tmp_path):
    text = "$(touch SHOULD_NOT_EXIST); `touch OTHER_FILE`"
    path = config_file(tmp_path, [make_stage("literal", "literal.txt", text)])
    run_workflow(path, tmp_path)
    assert (tmp_path / "literal.txt").read_text() == text
    assert not (tmp_path / "SHOULD_NOT_EXIST").exists()
    assert not (tmp_path / "OTHER_FILE").exists()


def test_false_boolean_gate_with_declared_exit_code_stops_without_next_stage(tmp_path):
    first = make_stage("gate", "gate.json", json.dumps({"gate": {"passes": False}}))
    first["argv"][2] += "; sys.exit(2)"
    first["outputs"].append("not_produced_after_stop.txt")
    first["gate"] = {"path": "gate.json", "key": "gate.passes", "stop_exit_codes": [0, 2]}
    path = config_file(tmp_path, [first, make_stage("must_not_run", "wrong.txt")])
    result = run_workflow(path, tmp_path)
    assert result["status"] == "gate_stopped"
    assert not (tmp_path / "wrong.txt").exists()
    resumed = run_workflow(path, tmp_path, resume=True)
    assert resumed["status"] == "gate_stopped"
    assert len(resumed["stages"]["gate"]["attempts"]) == 1


def test_precondition_gate_stops_before_starting_child(tmp_path):
    (tmp_path / "input_gate.json").write_text('{"go":false}')
    path = config_file(tmp_path, [make_stage("one", "one.txt", requires_gates=[{"path": "input_gate.json", "key": "go"}])])
    result = run_workflow(path, tmp_path)
    assert result["status"] == "gate_stopped"
    assert not result["stages"]["one"]["attempts"]
    assert not (tmp_path / "one.txt").exists()


@pytest.mark.parametrize("value", ["true", 1, None])
def test_non_boolean_gate_is_failure_not_permission(tmp_path, value):
    stage = make_stage("gate", "gate.json", json.dumps({"go": value}), gate={"path": "gate.json", "key": "go"})
    path = config_file(tmp_path, [stage])
    with pytest.raises(WorkflowError, match="must be a JSON boolean"):
        run_workflow(path, tmp_path)


def test_execution_failure_is_not_disguised_as_a_scientific_stop(tmp_path):
    stage = make_stage("gate", "gate.json", '{"go":false}', gate={"path": "gate.json", "key": "go"})
    stage["argv"][2] += "; sys.exit(1)"
    path = config_file(tmp_path, [stage])
    with pytest.raises(WorkflowError, match="exit code 1"):
        run_workflow(path, tmp_path)
    assert json.loads((tmp_path / "workflow-state/state.json").read_text())["status"] == "failed"
    with pytest.raises(WorkflowError, match="Refusing to overwrite"):
        run_workflow(path, tmp_path, resume=True)
    assert json.loads((tmp_path / "gate.json").read_text()) == {"go": False}


def test_existing_results_are_never_adopted_or_overwritten(tmp_path):
    (tmp_path / "old.txt").write_text("preserve")
    path = config_file(tmp_path, [make_stage("one", "old.txt")])
    with pytest.raises(WorkflowError, match="Refusing to overwrite"):
        run_workflow(path, tmp_path)
    assert (tmp_path / "old.txt").read_text() == "preserve"


def test_wait_for_files_waits_then_runs_without_requiring_user_intervention(tmp_path):
    path = config_file(tmp_path, [make_stage("one", "one.txt", wait_for=["ready.json"])])
    timer = threading.Timer(.03, lambda: (tmp_path / "ready.json").write_text("{}"))
    timer.start()
    try:
        assert run_workflow(path, tmp_path)["status"] == "completed"
    finally:
        timer.join()


def test_wait_timeout_can_resume_after_input_arrives_without_overwriting(tmp_path):
    stage = make_stage("one", "one.txt", wait_for=["ready.json"], wait_timeout_seconds=.02)
    path = config_file(tmp_path, [stage])
    with pytest.raises(WorkflowError, match="Timed out waiting"):
        run_workflow(path, tmp_path)
    (tmp_path / "ready.json").write_text("{}")
    assert run_workflow(path, tmp_path, resume=True)["status"] == "completed"


def test_subprocess_timeout_is_finite_and_preserves_failure_state(tmp_path):
    stage = make_stage("slow", "never.txt")
    stage["argv"] = [sys.executable, "-c", "import time; time.sleep(60)"]
    stage["timeout_seconds"] = .05
    path = config_file(tmp_path, [stage])
    started = time.monotonic()
    with pytest.raises(WorkflowError, match="finite subprocess timeout"):
        run_workflow(path, tmp_path)
    assert time.monotonic() - started < 3
    state = json.loads((tmp_path / "workflow-state/state.json").read_text())
    assert state["status"] == "failed"
    assert state["stages"]["slow"]["attempts"][0]["timed_out"]


def test_unknown_interrupted_child_is_never_automatically_duplicated(tmp_path):
    path = config_file(tmp_path, [make_stage("one", "one.txt")])
    result = run_workflow(path, tmp_path)
    state_path = tmp_path / "workflow-state/state.json"
    result["stages"]["one"]["status"] = "running"
    state_path.write_text(json.dumps(result))
    for _ in range(2):
        with pytest.raises(WorkflowError, match="automatic rerun is refused"):
            run_workflow(path, tmp_path, resume=True)


def test_dry_run_has_no_subprocess_or_filesystem_side_effects(tmp_path):
    path = config_file(tmp_path, [make_stage("one", "one.txt")])
    result = run_workflow(path, tmp_path, dry_run=True)
    assert result["subprocesses_started"] == 0
    assert not (tmp_path / "one.txt").exists()
    assert not (tmp_path / "workflow-state").exists()


def test_available_negative_upstream_gate_stops_while_later_files_are_missing(tmp_path):
    (tmp_path / "e1.json").write_text('{"gate":{"passes":false}}')
    stage = make_stage("e3", "must_not_exist.txt", wait_for=["e2_never_runs.json"],
                       watch_gates=[{"path": "e1.json", "key": "gate.passes"},
                                    {"path": "e2_never_runs.json", "key": "gate.passes"}])
    path = config_file(tmp_path, [stage])
    result = run_workflow(path, tmp_path)
    assert result["status"] == "gate_stopped"
    assert not result["stages"]["e3"]["attempts"]
    assert not (tmp_path / "must_not_exist.txt").exists()
    assert run_workflow(path, tmp_path, resume=True)["status"] == "gate_stopped"
    (tmp_path / "e1.json").write_text('{"gate":{"passes":true}}')
    with pytest.raises(WorkflowError, match="output hash changed"):
        run_workflow(path, tmp_path, resume=True)


def test_watched_positive_gate_does_not_bypass_missing_prerequisite(tmp_path):
    (tmp_path / "e1.json").write_text('{"passes":true}')
    stage = make_stage("e3", "must_not_exist.txt", wait_for=["e2_missing.json"],
                       watch_gates=[{"path": "e1.json", "key": "passes"}])
    path = config_file(tmp_path, [stage], wait_timeout_seconds=.03)
    with pytest.raises(WorkflowError, match="Timed out waiting"):
        run_workflow(path, tmp_path)
    assert not (tmp_path / "must_not_exist.txt").exists()


def test_partial_watched_json_blocks_even_when_all_wait_files_exist(tmp_path):
    gate_path = tmp_path / "upstream.json"
    gate_path.write_text('{"gate":')
    (tmp_path / "ready.txt").write_text("already ready")
    stage = make_stage("dependent", "must_not_exist.txt", wait_for=["ready.txt"],
                       watch_gates=[{"path": "upstream.json", "key": "gate.passes"}])
    path = config_file(tmp_path, [stage])

    def complete_write():
        time.sleep(.05)
        gate_path.write_text('{"gate":{"passes":false}}')

    writer = threading.Thread(target=complete_write)
    writer.start()
    try:
        result = run_workflow(path, tmp_path)
    finally:
        writer.join()
    assert result["status"] == "gate_stopped"
    assert not result["stages"]["dependent"]["attempts"]
    assert not (tmp_path / "must_not_exist.txt").exists()


def test_complete_but_nonboolean_watched_json_is_not_treated_as_incomplete(tmp_path):
    (tmp_path / "upstream.json").write_text('{"gate":{"passes":"false"}}')
    stage = make_stage("dependent", "must_not_exist.txt", wait_for=[],
                       watch_gates=[{"path": "upstream.json", "key": "gate.passes"}])
    path = config_file(tmp_path, [stage])
    with pytest.raises(WorkflowError, match="JSON boolean"):
        run_workflow(path, tmp_path)
    assert not (tmp_path / "must_not_exist.txt").exists()
