"""Synthetic phase boundaries and fail-closed analysis, without physical execution."""

import json

import pytest
from test_maniguard_continuation_admission import plan, write
from test_maniguard_continuation_progress import fixture

from helpers import maniguard_review_handoff_gripper as gripper


def measured_fixture(tmp_path, **kwargs):
    path, receipt = fixture(tmp_path, **kwargs)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        if row["type"] == "executed":
            row["action"][7] = -1 if row["step"] >= 8 else 1
        elif row["type"] == "latent_features":
            row["frame"]["fingers"] = [.04, .04] if row["step"] == 0 else [.035, .036]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path, receipt


def test_switch_command_is_native_but_its_measured_positions_are_before_that_action(tmp_path):
    result = gripper.audit_episode(*measured_fixture(tmp_path))
    commands = result["commands"]
    assert commands["closure_prefix"]["controls"] == 8
    assert commands["closure_prefix"]["closing_inputs"] == 0
    assert commands["native_after_switch"]["controls"] == 8
    assert commands["native_after_switch"]["closing_inputs"] == 8
    assert commands["native_after_switch"]["first_closing_control_index"] == 8
    measured = result["query_finger_joint_positions"]
    assert measured["at_switch_before_first_native_action"]["query_steps"] == [8]
    assert measured["strictly_after_switch"]["query_samples"] == 0
    assert measured["strictly_after_switch"]["per_joint"] == []
    assert measured["whole_episode"]["per_joint"][0]["median"] == pytest.approx(.0375)
    assert result["ordered_progress_attribution"]["whole_ordered_stage"] == 2
    assert not result["dispatch_authorized"] and not result["expansion_authorized"]


@pytest.mark.parametrize("mode,prefix,native,post", [
    ("native_only", 0, 16, 0), ("closure_only", 16, 0, 0),
    ("clock_handoff", 8, 0, 8), ("privileged_lid_handoff", 8, 0, 8),
])
def test_all_four_references_partition_commands_without_inventing_a_switch(
    tmp_path, mode, prefix, native, post,
):
    result = gripper.audit_episode(*measured_fixture(tmp_path, mode=mode))
    commands = result["commands"]
    assert commands["closure_prefix"]["controls"] == prefix
    assert commands["native_reference"]["controls"] == native
    assert commands["native_after_switch"]["controls"] == post
    assert commands["whole_episode"]["closing_inputs"] == 8
    if mode in ("native_only", "closure_only"):
        assert result["switch_query_step"] is None
        assert set(result["query_finger_joint_positions"]) == {"whole_episode"}


@pytest.mark.parametrize("clock,prefix,post", [(0, 0, 16), (24, 16, 0)])
def test_immediate_or_missing_clock_trigger_does_not_invent_a_closure_prefix(
    tmp_path, clock, prefix, post,
):
    result = gripper.audit_episode(*measured_fixture(tmp_path, clock=clock))
    assert result["commands"]["closure_prefix"]["controls"] == prefix
    assert result["commands"]["native_after_switch"]["controls"] == post
    if clock == 0:
        assert result["query_finger_joint_positions"]["strictly_after_switch"]["query_steps"] == [8]
        assert not result["ordered_progress_attribution"]["actual_handoff_observed"]


@pytest.mark.parametrize("corruption", ["missing_query", "bad_fingers", "nan_fingers"])
def test_missing_or_invalid_joint_samples_are_refused(tmp_path, corruption):
    path, receipt = measured_fixture(tmp_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    query = next(r for r in rows if r["type"] == "latent_features")
    if corruption == "missing_query":
        rows.remove(query)
    else:
        query["frame"]["fingers"] = [0.] if corruption == "bad_fingers" else [float("nan"), 0.]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        gripper.audit_episode(path, receipt)


def test_whole_episode_violation_is_retained_beside_phase_diagnostics(tmp_path):
    result = gripper.audit_episode(*measured_fixture(tmp_path, violation=15))
    assert result["ordered_progress_attribution"]["physical_violation"]
    assert not result["ordered_progress_attribution"]["nonprivileged_handoff_progress_eligible"]
    assert result["commands"]["native_after_switch"]["closing_inputs"] == 8
    states = result["raw_state_observations"]
    assert states["whole_episode"]["first_held_observation"] == 10
    assert states["whole_episode"]["first_doomed_observation"] == 15
    assert states["strictly_after_switch"]["held_observations"] == 7
    assert states["strictly_after_switch"]["goal_observations"] == 8


def test_incomplete_screen_is_refused_before_any_episode_reader_runs(tmp_path, monkeypatch):
    protocol = plan(tmp_path)
    p = tmp_path / "protocol.json"
    write(p, protocol)
    run = tmp_path / "screen"
    write(run / "screen.json", {"screen_complete": False, "attempts": []})
    monkeypatch.setattr(gripper, "read_episode", lambda path: pytest.fail("No outcome access"))
    with pytest.raises(ValueError, match="All four"):
        gripper.analyze_screen(run, p)


@pytest.mark.parametrize("corruption", [None, "cached_receipt", "cached_progress"])
def test_complete_screen_joins_four_references_and_refuses_changed_cached_results(
    tmp_path, monkeypatch, corruption,
):
    protocol = plan(tmp_path)
    p = tmp_path / "protocol.json"
    write(p, protocol)
    raw = p.read_bytes()
    run = tmp_path / "screen"
    attempts, receipts = [], {}
    for row in protocol["episodes"]:
        output = run / "episodes" / (row["id"] + "-a0")
        output.mkdir(parents=True)
        trace, receipt = measured_fixture(output, mode=row["condition"])
        receipts[output] = receipt
        (output / "protocol.json").write_bytes(raw)
        write(output / "verified-receipt.json", receipt)
        write(output / "progress-attribution.json", gripper.attribute_progress(trace, receipt))
        attempts.append({"id": row["id"] + "-a0", "protocol_episode_id": row["id"],
                         "status": "completed", "condition": row["condition"],
                         "worker": row["worker"], "output": str(output)})
    write(run / "screen.json", {"screen_complete": True, "attempts": attempts,
                               "protocol_sha256": gripper._sha(raw)})
    reader_calls = []

    def synthetic_physical_reader(output):
        reader_calls.append(output)
        return receipts[output]

    monkeypatch.setattr(gripper, "read_episode", synthetic_physical_reader)
    if corruption:
        output = run / "episodes" / (protocol["episodes"][0]["id"] + "-a0")
        target = output / ("verified-receipt.json" if corruption == "cached_receipt"
                           else "progress-attribution.json")
        value = json.loads(target.read_text())
        value["deliberately_changed_synthetic_cache"] = True
        write(target, value)
        preserved = target.read_bytes()
        with pytest.raises(ValueError, match="Cached"):
            gripper.analyze_screen(run, p)
        assert target.read_bytes() == preserved
    else:
        result = gripper.analyze_screen(run, p)
        assert len(result["episodes"]) == len(reader_calls) == 4
        assert result["reused_controls"] == 64
        assert {r["mode"] for r in result["episodes"]} == {
            "native_only", "closure_only", "clock_handoff", "privileged_lid_handoff"}
        assert not result["dispatch_authorized"] and not result["expansion_authorized"]
