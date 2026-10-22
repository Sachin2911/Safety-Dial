"""Synthetic clearance/ledger and source-only job preparation; no dispatch."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from helpers.maniguard_continuation_admission import (
    ROOT,
    ScreenLedger,
    acquire_service_lease,
    require_original_clearance,
)
from helpers.maniguard_continuation_handoff import HandoffReference
from helpers.maniguard_continuation_jobs import prepare_job
from helpers.maniguard_continuation_limits import DEFAULT_LIMITS, epoch
from helpers.maniguard_continuation_source import REQUIRED_SOURCES, prepare_runtime
from helpers.maniguard_latent import initial_weights
from helpers.maniguard_latent_hardware import GPU_UUIDS


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan(tmp_path):
    modes = ["native_only", "closure_only", "clock_handoff", "privileged_lid_handoff"]
    return {"status": "prospectively_frozen", "runtime_limits": dict(DEFAULT_LIMITS),
            "run_directory": str(tmp_path / "screen"),
            "service_lease_directory": str(ROOT / "runs/maniguard-policy-service-locks"),
            "first_screen_budget": {"maximum_full_starts_including_partial_failures": 4,
                                    "automatic_replacements": 0},
            "physical_cutoff_utc": "2026-10-20T19:30:00Z", "instance_id": 56777435,
            "original_run": str(tmp_path / "original"),
            "scene_asset_sha256": "a" * 64,
            "episodes": [{"id": mode, "condition": mode, "worker": [0, 1, 1, 0][i],
                          "wave": i // 2} for i, mode in enumerate(modes)],
            "workers": {str(i): {"port": 18000 + i, "renderer_vulkan_index": i,
                       "service_ledger": str(tmp_path / "original" / f"server-jar-worker{i}-physical.jsonl"),
                       "simulation_physical_device": i, "policy_physical_device": i,
                       "simulation_gpu_uuid": GPU_UUIDS[i], "policy_gpu_uuid": GPU_UUIDS[i]}
                        for i in (0, 1)}}


def test_reservation_is_durable_before_preparation_and_four_starts_include_failures(tmp_path):
    p = tmp_path / "protocol.json"
    write(p, plan(tmp_path))
    ledger = ScreenLedger(tmp_path / "screen", p)
    now = epoch("2026-10-20T16:00:00Z")
    stop = epoch("2026-10-20T20:00:00Z")
    try:
        first = ledger.reserve_wave(0, now=now, actual_stop_epoch=stop)
        assert len(json.loads(ledger.path.read_text())["attempts"]) == 2
        assert all(a["status"] == "reserved" and not Path(a["output"]).exists() for a in first)
        with pytest.raises(ValueError, match="surviving reservations"):
            ledger.reserve_wave(0, now=now, actual_stop_epoch=stop)
        for a in first:
            a["status"] = "completed"
        second = ledger.reserve_wave(1, now=now + 2704, actual_stop_epoch=stop)
        assert len(ledger.state["attempts"]) == 4
        second[0]["status"] = "invalid"
        second[1]["status"] = "completed"
        with pytest.raises(ValueError):
            ledger.reserve_wave(1, now=now + 5408, actual_stop_epoch=stop)
        assert len(ledger.state["attempts"]) == 4
    finally:
        ledger.close()


def test_invalid_first_wave_cannot_be_replaced_or_admit_later_work(tmp_path):
    p = tmp_path / "protocol.json"
    write(p, plan(tmp_path))
    ledger = ScreenLedger(tmp_path / "screen", p)
    try:
        first = ledger.reserve_wave(0, now=epoch("2026-10-20T16:00:00Z"),
                                    actual_stop_epoch=epoch("2026-10-20T20:00:00Z"))
        first[0]["status"], first[1]["status"] = "invalid", "completed"
        with pytest.raises(ValueError, match="no automatic replacement"):
            ledger.reserve_wave(1, now=epoch("2026-10-20T17:00:00Z"),
                                 actual_stop_epoch=epoch("2026-10-20T20:00:00Z"))
        assert len(ledger.state["attempts"]) == 2
    finally:
        ledger.close()


def test_two_independent_owners_cannot_lease_one_service(tmp_path):
    first = acquire_service_lease(tmp_path / "service.lock")
    try:
        with pytest.raises(BlockingIOError):
            acquire_service_lease(tmp_path / "service.lock")
    finally:
        first.close()
    acquire_service_lease(tmp_path / "service.lock").close()


def test_one_protocol_cannot_refresh_its_start_budget_by_changing_output_or_coordinator(tmp_path):
    p = tmp_path / "protocol.json"
    write(p, plan(tmp_path))
    for other in [tmp_path / "other-a", tmp_path / "other-b"]:
        with pytest.raises(ValueError, match="canonical"):
            ScreenLedger(other, p)
        assert not other.exists()
    ledger = ScreenLedger(tmp_path / "screen", p)
    try:
        with pytest.raises(BlockingIOError):
            ScreenLedger(tmp_path / "screen", p)
        rows = ledger.reserve_wave(0, now=epoch("2026-10-20T16:00:00Z"),
                                   actual_stop_epoch=epoch("2026-10-20T20:00:00Z"))
        rows[0]["status"], rows[1]["status"] = "invalid", "completed"
        ledger.save()
    finally:
        ledger.close()
    resumed = ScreenLedger(tmp_path / "screen", p)
    try:
        assert len(resumed.state["attempts"]) == 2
        with pytest.raises(ValueError, match="no automatic replacement"):
            resumed.reserve_wave(1, now=epoch("2026-10-20T17:00:00Z"),
                                  actual_stop_epoch=epoch("2026-10-20T20:00:00Z"))
    finally:
        resumed.close()


@pytest.fixture
def cleared(tmp_path):
    original = tmp_path / "original"
    attempts = [{"id": f"synthetic-{i}", "status": "completed", "engineering": False}
                for i in range(60)]
    write(original / "queue.json", {"attempts": attempts})
    qhash = sha(original / "queue.json")
    design = {"physical_design_complete": True, "verified_full_jobs": 60, "errors": [],
              "pending": [], "protocol_sha256": "f" * 64, "queue_sha256": qhash}
    write(original / "analysis/design-verification.json", design)
    recovered = []
    for a in attempts:
        row = {"attempt": a["id"], "controller_sha256": "c" * 64,
               "policy_controls": 2000, "policy_queries": 250}
        recovered.append(row)
        write(original / "episodes" / a["id"] / "verified-receipt.json", {
            "valid": True, "full_episode": True, "controller_sha256": row["controller_sha256"],
            "physical": {"steps": 2000, "ledger": {"policy_queries": 250}}})
    recovery_path = tmp_path / "synthetic-recovery.json"
    write(recovery_path, {"physical_receipts_reverified": True,
                         "all_archives_redownloaded_and_sha256_checked": True, "episodes": recovered})
    write(original / "original-comparison-clearance.json", {
        "protocol_sha256": "f" * 64, "queue_sha256": qhash,
        "design_verification_sha256": sha(original / "analysis/design-verification.json"),
        "physical_recovery_proof_sha256": sha(recovery_path)})
    write(original / "backups/index.json", {"stages": {"synthetic": {
        "episodes": [a["id"] for a in attempts]}}})
    return {"original_run": str(original), "original_protocol_sha256": "f" * 64,
            "original_physical_recovery_proof": str(recovery_path)}


@pytest.mark.parametrize("broken", [None, "unfinished", "live", "stale_design", "restore", "archive"])
def test_original_clearance_checks_completion_actual_handles_design_and_full_preservation(cleared, broken):
    original = Path(cleared["original_run"])
    identity = None
    if broken in ("unfinished", "live"):
        p = original / "queue.json"
        value = json.loads(p.read_text())
        if broken == "unfinished":
            value["attempts"][-1]["status"] = "running"
        else:
            identity = {"pid": 123, "start_ticks": "synthetic"}
            value["attempts"][0]["process_identity"] = identity
        write(p, value)
    elif broken == "stale_design":
        p = original / "analysis/design-verification.json"
        value = json.loads(p.read_text())
        value["pending"] = ["one checking block"]
        write(p, value)
    elif broken == "restore":
        p = Path(cleared["original_physical_recovery_proof"])
        value = json.loads(p.read_text())
        value["episodes"].pop()
        write(p, value)
    elif broken == "archive":
        write(original / "backups/index.json", {"stages": {}})
    if broken is None:
        assert require_original_clearance(cleared, identity_reader=lambda pid: None)
    else:
        with pytest.raises(ValueError):
            require_original_clearance(cleared, identity_reader=lambda pid: identity)


def test_source_only_preparation_records_components_and_never_starts_a_runtime(tmp_path):
    protocol = plan(tmp_path)
    weights = initial_weights()[1]
    refs = {mode: HandoffReference(weights, mode=mode) for mode in
            ("native_only", "closure_only", "clock_handoff", "privileged_lid_handoff")}
    protocol.update(scene="task_0019/base", seed=6101920,
        closure_controller_sha256=refs["closure_only"].closure.sha256(),
        policy=json.loads((ROOT / "configs/maniguard/phase2/sources.json").read_text())["policies"]["jar"],
        selector_parameters={mode: {"mode": mode, "debounce": 3, "clock_switch_step": 488} for mode in refs},
        selector_contract_sha256={mode: ref.contract_sha256 for mode, ref in refs.items()},
        critical_source_hashes={name: sha(ROOT / name) for name in REQUIRED_SOURCES},
        scene_asset_sha256=sha(ROOT / "data/maniguard/benchmark/jar_transport/task_0019/base/scene_ep1.json"),
        native_evaluator_config_sha256=sha(ROOT / "third_party/ManiGuard/configs/eval/jar_transport_joint.yaml"))
    service = Path(protocol["workers"]["0"]["service_ledger"])
    service.parent.mkdir()
    service.write_bytes(b"")
    preflight = tmp_path / "preflight"
    preflight.mkdir()
    protocol["instrumented_runtime_sha256"] = sha(prepare_runtime(preflight))
    controller = tmp_path / "controller.npz"
    np.savez(controller, weights=weights)
    proto_path = tmp_path / "protocol.json"
    write(proto_path, protocol)
    attempt = {"output": str(tmp_path / "screen/episodes/native_only-a0"), "worker": 0,
               "condition": "native_only", "protocol_episode_id": "native_only",
               "started_epoch": 1, "deadline_epoch": 2701}
    path = prepare_job(proto_path, attempt, controller, {"synthetic_clearance_only": True})
    job = json.loads(path.read_text())
    assert job["code_hashes"] == protocol["critical_source_hashes"]
    assert job["runtime_limits"] == DEFAULT_LIMITS and job["action_horizon"] == 2000
    assert job["continuation"]["mode"] == "native_only"
    assert not (path.parent / "trace.jsonl").exists()
    wrong = copy.deepcopy(protocol)
    wrong["run_directory"] = str(tmp_path / "bad-screen")
    wrong["critical_source_hashes"][REQUIRED_SOURCES[0]] = "0" * 64
    write(proto_path, wrong)
    with pytest.raises(ValueError, match="differs from frozen screen"):
        prepare_job(proto_path, {**attempt, "output": str(tmp_path / "bad-screen/episodes/native_only-a0")},
                    controller, {})
