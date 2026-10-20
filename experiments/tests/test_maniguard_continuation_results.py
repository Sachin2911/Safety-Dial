"""Synthetic full-reader joins; these fixtures are never physical evidence."""

import hashlib
import json

import numpy as np
import pytest
from test_maniguard_continuation_evidence import stream

from helpers.maniguard_continuation_evidence import replay_stream
from helpers.maniguard_continuation_handoff import HandoffReference
from helpers.maniguard_continuation_limits import DEFAULT_LIMITS
from helpers.maniguard_continuation_protocol import validate_job
from helpers.maniguard_continuation_results import read_episode
from helpers.maniguard_continuation_source import REQUIRED_SOURCES, prepare_runtime
from helpers.maniguard_latent import initial_weights
from helpers.maniguard_latent_source import prepare_runtime as prepare_latent
from helpers.maniguard_phase2_outcomes import derive_outcome
from helpers.maniguard_runtime import json_value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, default=json_value) + "\n")


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def full_fixture(tmp_path):
    rows = stream()
    hashes = {}
    for name in REQUIRED_SOURCES:
        path = tmp_path / "study_source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Synthetic source fixture only: " + name)
        hashes[name] = digest(path.read_bytes())
    runtime = tmp_path / "benchmark_instrumented.py"
    runtime.write_text("# Synthetic reader fixture, never executed\n")
    runtime_hash = digest(runtime.read_bytes())
    native_config = tmp_path / "native-config.yaml"
    native_config.write_text("synthetic configuration, never executed")
    config_hash = digest(native_config.read_bytes())
    closure = rows[0]["contract"]["closure_controller_sha256"]
    contract_hash = next(r["choice"]["contract_sha256"] for r in rows
                         if r["type"] == "continuation_choice")
    parameters = {"mode": "privileged_lid_handoff", "debounce": 3, "clock_switch_step": 8}
    protocol = {"status": "prospectively_frozen", "critical_source_hashes": hashes,
                "run_directory": str(tmp_path / "historical-screen"),
                "service_lease_directory": "/synthetic/policy-locks",
                "policy": {"revision": "synthetic"},
                "instrumented_runtime_sha256": runtime_hash, "closure_controller_sha256": closure,
                "scene": "synthetic-scene", "seed": 123,
                "selector_parameters": {"privileged_lid_handoff": parameters},
                "selector_contract_sha256": {"privileged_lid_handoff": contract_hash}}
    protocol["native_evaluator_config_sha256"] = config_hash
    modes = ["native_only", "closure_only", "clock_handoff", "privileged_lid_handoff"]
    protocol.update(
        original_run="/synthetic/original", scene_asset_sha256="a" * 64,
        episodes=[{"id": mode, "condition": mode, "worker": [0, 1, 1, 0][i], "wave": i // 2}
                  for i, mode in enumerate(modes)],
        workers={"0": {"port": 18000, "renderer_vulkan_index": 0,
                       "simulation_gpu_uuid": "synthetic-sim", "policy_gpu_uuid": "synthetic-policy"},
                 "1": {"port": 18001, "renderer_vulkan_index": 1,
                       "simulation_gpu_uuid": "synthetic-sim1", "policy_gpu_uuid": "synthetic-policy1"}},
        first_screen_budget={"maximum_full_starts_including_partial_failures": 4,
                            "automatic_replacements": 0},
        runtime_limits=dict(DEFAULT_LIMITS),
    )
    for worker, spec in protocol["workers"].items():
        spec["service_ledger"] = f"/synthetic/original/server-jar-worker{worker}-physical.jsonl"
    for mode in modes:
        protocol["selector_parameters"][mode] = {**parameters, "mode": mode}
        protocol["selector_contract_sha256"][mode] = HandoffReference(
            initial_weights()[1], **protocol["selector_parameters"][mode]
        ).contract_sha256
    write_json(tmp_path / "protocol.json", protocol)
    job = {"study": "continuation-feasibility-v1", "family": "jar", "random_policy": False,
           "max_steps_override": None, "action_horizon": 2000, "source_profile": "high_fidelity",
           "protocol_sha256": digest((tmp_path / "protocol.json").read_bytes()),
           "code_hashes": hashes, "policy": protocol["policy"], "seed": 123,
           "scene": "synthetic-scene",
           "episode_id": "synthetic", "controller_sha256": closure,
           "simulation_gpu_uuid": "synthetic-sim", "policy_gpu_uuid": "synthetic-policy",
           "adapter": {"kind": "identity"}, "repository_commit": "0" * 40,
           "protocol_episode_id": "privileged_lid_handoff", "condition": "privileged_lid_handoff",
           "output": str(tmp_path / "historical-screen/episodes/privileged_lid_handoff-a0"),
           "service_lock_path": "/synthetic/policy-locks/18000.lock",
           "worker": 0, "policy_port": 18000, "renderer_vulkan_index": 0,
           "runtime_limits": dict(DEFAULT_LIMITS), "admitted_epoch": 0, "deadline_epoch": 2700,
           "native_evaluator_config_sha256": config_hash,
           "scene_asset_sha256": "a" * 64,
           "service_ledger_baseline": {"path": protocol["workers"]["0"]["service_ledger"],
                                       "bytes": 0, "sha256": digest(b"")},
           "continuation": {"mode": "privileged_lid_handoff", "debounce": 3,
                            "clock_switch_step": 8},
           "continuation_contract_sha256": next(r["choice"]["contract_sha256"] for r in rows
                                                if r["type"] == "continuation_choice")}
    write_json(tmp_path / "job.json", job)
    np.savez(tmp_path / "controller.npz", weights=initial_weights()[1])
    write_json(tmp_path / "runtime-provenance.json", {"runtime_sha256": runtime_hash,
                                                    "study": "continuation-feasibility-v1"})
    image = np.zeros((640, 640, 3), np.uint8)
    image_hash = digest(image.tobytes())
    for row in rows:
        if row["type"] == "latent_features":
            row["context"] = {"image_hashes": [image_hash, image_hash]}
            folder = tmp_path / "query_observations"
            folder.mkdir(exist_ok=True)
            np.savez_compressed(folder / f"{row['step']:06d}.npz", features=row["features"],
                                overview=image, wrist=image)
    # The synthetic fixture's scene CRC seed must replace its simple stream seed.
    import zlib

    seed = zlib.crc32(b"123:synthetic-scene") & 0x7FFFFFFF
    for row in rows:
        if row["type"] == "noise_receipt":
            row["receipt"]["seed"] = seed
        if row["type"] == "continuation_request":
            row["request"]["seed"] = seed
    head = [
        {"type": "job", "job": job}, {"type": "initial", "scene": {"name": "synthetic-scene"}},
        {"type": "technical_budget_contract", "limits": dict(DEFAULT_LIMITS)},
        {"type": "cuda_device_verified", "actual_gpu_uuid": job["simulation_gpu_uuid"]},
        {"type": "runtime_contract", "overview_shape": [640, 640, 3],
         "wrist_shape": [640, 640, 3], "reference_asset_paths": ["franka_panda_longfinger"]},
        {"type": "policy_metadata", "metadata": {"audit_service": False, "model_shape": [16, 32],
         "checkpoint_revision": "synthetic", "physical_gpu_uuid": job["policy_gpu_uuid"],
         "source_hashes": hashes}},
    ]
    complete = head
    for row in rows:
        if row["type"] == "latent_features":
            complete.append({"type": "query", "step": row["step"], "image_hashes": {
                "overview_image": image_hash, "wrist_images": image_hash}})
        complete.append(row)
        if row["type"] == "observed" and row["step"]:
            terminal = row["step"] == 16
            complete.append({"type": "goal", "step": row["step"], "success": terminal,
                             "detail": {"mode": "held_intersection", "held": terminal,
                                        "intersects": terminal}})
    result = {"status": "completed", "success": True, "ltl_monitored": True,
              "ltl_violated": False, "steps": 16, "scene_name": "synthetic-scene",
              "ever_contacted": True, "ever_grasped": True}
    complete.append({"type": "result", "result": result})
    trace = tmp_path / "trace.jsonl"
    trace.write_text("".join(json.dumps(row, default=json_value) + "\n" for row in complete))
    service_bytes = "".join(json.dumps(row["receipt"]) + "\n" for row in complete
                            if row["type"] == "noise_receipt").encode()
    (tmp_path / "server-receipts.jsonl").write_bytes(service_bytes)
    write_json(tmp_path / "service-ledger-readback.json", {
        "baseline": job["service_ledger_baseline"], "bytes": len(service_bytes),
        "sha256": digest(service_bytes), "records": 2, "episode": job["episode_id"]})
    write_json(tmp_path / "scene-asset-readback.json", {
        "scene": job["scene"], "sha256": job["scene_asset_sha256"], "checked_epoch": 0})
    write_json(tmp_path / "episode/results.jsonl", result)
    write_json(tmp_path / "episode/synthetic-scene_ltl.json", {
        "total_steps_monitored": 17, "violated": False,
        "log": [row["monitor"]["observation"] for row in complete if row["type"] == "observed"]})
    write_json(tmp_path / "ledger.json", {"physics_accounting_attached": True,
        "policy_control_steps": 16, "policy_queries": 2, "encoder_image_calls": 2,
        "encoder_images": 4, "encoder_text_calls": 1, "native_policy_requests_admitted": 2,
        "native_policy_responses_received": 2, "native_policy_responses_verified": 2,
        "initialization_finished": True, "initialization_completed_epoch": 1,
        "initialization_physics_steps": 0, "budget_violation_reasons": []})
    write_json(tmp_path / "guardian-terminal.json", {"exit_code": 0, "technical_stop_reason": None,
        "started_epoch": 0, "finished_epoch": 2, "deadline_epoch": 2700})
    observations = [{"step": row["step"], **row["monitor"]["observation"]}
                    for row in complete if row["type"] == "observed"]
    goals = [row for row in complete if row["type"] == "goal"]
    outcome = derive_outcome("jar", result, observations, goals)
    proof = replay_stream(complete, initial_weights()[1], **job["continuation"],
                          episode="synthetic", seed=seed, steps=16)
    outcome.update(continuation_contract_sha256=proof["contract_sha256"],
                   closure_controller_sha256=closure, continuation_state=proof["state"],
                   score=outcome["U"])
    write_json(tmp_path / "outcome.json", outcome)
    return tmp_path


def test_full_reader_joins_synthetic_monitor_goal_source_images_and_request_counts(full_fixture):
    result = read_episode(full_fixture)
    assert result["valid"] and result["full_episode"]
    assert result["outcome"]["benchmark_success"] and not result["outcome"]["C_req"]
    assert result["stream"]["actual_post_switch_controls"] == 8
    assert result["stream"]["remaining_native_horizon_at_switch"] == 1992


def test_checked_native_preparer_changes_only_the_study_import(tmp_path):
    base, compound = tmp_path / "base", tmp_path / "compound"
    base.mkdir()
    compound.mkdir()
    before = prepare_latent(base)
    after = prepare_runtime(compound)
    assert after.read_text() == before.read_text().replace(
        "from helpers.maniguard_latent_runtime import STUDY",
        "from helpers.maniguard_continuation_runtime import STUDY", 1,
    )
    provenance = json.loads((compound / "runtime-provenance.json").read_text())
    assert provenance["runtime_sha256"] == digest(after.read_bytes())
    assert provenance["latent_runtime_sha256"] == digest(before.read_bytes())


def test_original_monitor_history_must_match_copied_trace_even_with_same_counts_and_verdict(full_fixture):
    path = full_fixture / "episode/synthetic-scene_ltl.json"
    native = json.loads(path.read_text())
    for row in native["log"]:
        row["ap"]["jar_closed"] = False
    write_json(path, native)
    with pytest.raises(ValueError, match="original native log"):
        read_episode(full_fixture)


@pytest.mark.parametrize("change", ["condition", "worker", "seed", "selector", "port"])
def test_admission_and_reader_share_explicit_frozen_job_binding(full_fixture, change):
    protocol = json.loads((full_fixture / "protocol.json").read_text())
    job = json.loads((full_fixture / "job.json").read_text())
    if change == "selector":
        job["continuation"]["debounce"] = 1
    elif change == "condition":
        job["condition"] = "clock_handoff"
        job["continuation"] = protocol["selector_parameters"]["clock_handoff"]
        job["continuation_contract_sha256"] = protocol["selector_contract_sha256"]["clock_handoff"]
    else:
        key = {"worker": "worker", "seed": "seed", "port": "policy_port"}[change]
        job[key] += 1
    with pytest.raises(ValueError):
        validate_job(protocol, job)


@pytest.mark.parametrize("corruption", ["unfrozen", "source_omission", "source_bytes",
                                      "runtime", "encoder_count", "saved_outcome", "image",
                                      "selector", "case", "native_config", "startup", "guardian",
                                      "scene_asset", "service_tail"])
def test_full_reader_rejects_broken_joins(full_fixture, corruption):
    path = full_fixture
    if corruption in ("unfrozen", "source_omission"):
        p = path / "protocol.json"
        value = json.loads(p.read_text())
        if corruption == "unfrozen":
            value["status"] = "preparation_only"
        else:
            value["critical_source_hashes"].pop(REQUIRED_SOURCES[0])
        write_json(p, value)
        job_path = path / "job.json"
        job = json.loads(job_path.read_text())
        job["protocol_sha256"] = digest(p.read_bytes())
        write_json(job_path, job)
    elif corruption in ("selector", "case"):
        p = path / "job.json"
        value = json.loads(p.read_text())
        if corruption == "selector":
            value["continuation"]["clock_switch_step"] = 16
        else:
            value["seed"] = 124
        write_json(p, value)
    elif corruption == "source_bytes":
        (path / "study_source" / REQUIRED_SOURCES[0]).write_text("changed")
    elif corruption == "native_config":
        (path / "native-config.yaml").write_text("changed")
    elif corruption == "scene_asset":
        p = path / "scene-asset-readback.json"
        value = json.loads(p.read_text())
        value["sha256"] = "0" * 64
        write_json(p, value)
    elif corruption == "service_tail":
        (path / "server-receipts.jsonl").write_bytes(b"")
    elif corruption in ("startup", "guardian"):
        p = path / ("ledger.json" if corruption == "startup" else "guardian-terminal.json")
        value = json.loads(p.read_text())
        value["initialization_completed_epoch" if corruption == "startup" else "finished_epoch"] = 3000
        write_json(p, value)
    elif corruption == "runtime":
        (path / "benchmark_instrumented.py").write_text("changed")
    elif corruption == "encoder_count":
        p = path / "ledger.json"
        value = json.loads(p.read_text())
        value["encoder_images"] = 2
        write_json(p, value)
    elif corruption == "saved_outcome":
        p = path / "outcome.json"
        value = json.loads(p.read_text())
        value["C_req"] = True
        write_json(p, value)
    else:
        p = path / "query_observations/000000.npz"
        with np.load(p, allow_pickle=False) as data:
            values = {key: data[key] for key in data.files}
        values["overview"][0, 0, 0] = 1
        np.savez_compressed(p, **values)
    with pytest.raises(ValueError):
        read_episode(path)
