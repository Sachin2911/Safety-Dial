"""Independent full physical reader for a separately frozen compound experiment."""

import hashlib
import json
import zlib
from pathlib import Path

import numpy as np

from helpers.maniguard_continuation_evidence import replay_stream
from helpers.maniguard_continuation_protocol import validate_job
from helpers.maniguard_continuation_source import REQUIRED_SOURCES
from helpers.maniguard_latent import LatentController, outcome_score
from helpers.maniguard_latent_completion import runtime_errors, source_errors
from helpers.maniguard_phase2_outcomes import derive_outcome
from helpers.maniguard_results import read_episode as read_physical


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def join_monitor_history(rows, native_log):
    observed = [row for row in rows if row["type"] == "observed"]
    if any(not row.get("monitor") or row["monitor"].get("valid") is not True for row in observed):
        raise ValueError("Copied physical monitor observation is unavailable or invalid")
    copied = [row["monitor"]["observation"] for row in observed]
    if copied != native_log:
        raise ValueError("Copied physical monitor history differs from its original native log")
    return len(copied)


def read_episode(output):
    output = Path(output)
    job = json.loads((output / "job.json").read_text())
    protocol_path = output / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if (job["study"] != "continuation-feasibility-v1" or job["family"] != "jar"
            or job["max_steps_override"] is not None or job["random_policy"]
            or job["action_horizon"] != 2000 or job["source_profile"] != "high_fidelity"
            or protocol["status"] != "prospectively_frozen"
            or sha(protocol_path) != job["protocol_sha256"]):
        raise ValueError("Compound fitness requires its separately frozen full-native protocol")
    required = set(REQUIRED_SOURCES)
    if not (required <= set(protocol["critical_source_hashes"])
            and required <= set(job["code_hashes"])):
        raise ValueError("Compound protocol/job omits required execution or validity sources")
    validate_job(protocol, job)
    physical = read_physical(output)
    rows = [json.loads(line) for line in (output / "trace.jsonl").read_text().splitlines()]
    results = [row["result"] for row in rows if row["type"] == "result"]
    raw = [json.loads(line) for line in (output / "episode/results.jsonl").read_text().splitlines()]
    if len(results) != 1 or raw != results:
        raise ValueError("Compound telemetry and original native result disagree")
    result = results[0]
    native_monitor = json.loads((output / "episode" / (result["scene_name"] + "_ltl.json")).read_text())
    join_monitor_history(rows, native_monitor["log"])
    steps = result["steps"]
    if not 1 <= steps <= 2000 or (steps < 2000 and result["success"] is not True):
        raise ValueError("Compound trial shortened the native horizon without native success")
    if [row["job"] for row in rows if row["type"] == "job"] != [job]:
        raise ValueError("Compound telemetry job differs from its immutable job")
    errors = runtime_errors(job, rows, steps)
    errors += source_errors(job, protocol["critical_source_hashes"], protocol["policy"])
    if errors:
        raise ValueError("Compound runtime/source contract differs: " + "; ".join(errors))
    metadata = next(row["metadata"] for row in rows if row["type"] == "policy_metadata")
    for name in ("experiments/helpers/maniguard_latent.py",
                 "experiments/helpers/maniguard_latent_policy.py"):
        if metadata["source_hashes"].get(name) != job["code_hashes"][name]:
            raise ValueError("Actual server actor differs from the frozen compound job")
    for name, expected in job["code_hashes"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid archived source path")
        if sha(output / "study_source" / relative) != expected:
            raise ValueError("Archived compound source checksum mismatch: " + name)
    provenance = json.loads((output / "runtime-provenance.json").read_text())
    if (sha(output / "benchmark_instrumented.py") != provenance["runtime_sha256"]
            or provenance["runtime_sha256"] != protocol["instrumented_runtime_sha256"]
            or provenance["study"] != "continuation-feasibility-v1"):
        raise ValueError("Actual evaluator differs from the prospectively checked compound copy")
    with np.load(output / "controller.npz", allow_pickle=False) as data:
        weights = data["weights"]
    closure = LatentController(weights)
    if (closure.sha256() != job["controller_sha256"]
            or closure.sha256() != protocol["closure_controller_sha256"]):
        raise ValueError("The preselected closure component differs")
    scene = next(row["scene"] for row in rows if row["type"] == "initial")
    seed = zlib.crc32(f"{job['seed']}:{scene['name']}".encode()) & 0x7FFFFFFF
    stream = replay_stream(rows, weights, **job["continuation"],
                           episode=job["episode_id"], seed=seed, steps=steps)
    if stream["contract_sha256"] != job["continuation_contract_sha256"]:
        raise ValueError("Compound job and replayed selector identity disagree")
    features = [row for row in rows if row["type"] == "latent_features"]
    query_images = {row["step"]: row["image_hashes"] for row in rows if row["type"] == "query"}
    for row in features:
        with np.load(output / "query_observations" / f"{row['step']:06d}.npz",
                     allow_pickle=False) as data:
            np.testing.assert_array_equal(data["features"], np.asarray(row["features"], np.float32))
            for index, key in enumerate(("overview", "wrist")):
                image_hash = hashlib.sha256(np.ascontiguousarray(data[key]).tobytes()).hexdigest()
                native_key = "overview_image" if index == 0 else "wrist_images"
                if (image_hash != row["context"]["image_hashes"][index]
                        or image_hash != query_images[row["step"]][native_key]):
                    raise ValueError("Saved compound camera differs from the actual query/encoder input")
    ledger = physical["ledger"]
    count = len(features)
    if any(ledger[key] != count for key in (
        "policy_queries", "encoder_image_calls", "native_policy_requests_admitted",
        "native_policy_responses_received", "native_policy_responses_verified",
    )) or ledger["encoder_images"] != 2 * count or ledger["encoder_text_calls"] != 1:
        raise ValueError("Compound request, return or matched encoder accounting differs")
    observations = [{"step": row["step"], **row["monitor"]["observation"]}
                    for row in rows if row["type"] == "observed"]
    goals = [row for row in rows if row["type"] == "goal"]
    outcome = derive_outcome("jar", result, observations, goals)
    if not outcome["valid"]:
        raise ValueError("Compound requested-completion evidence is invalid")
    saved = json.loads((output / "outcome.json").read_text())
    if (any(saved.get(key) != value for key, value in outcome.items())
            or saved["continuation_contract_sha256"] != stream["contract_sha256"]
            or saved["closure_controller_sha256"] != closure.sha256()
            or saved["continuation_state"] != stream["state"]
            or saved["score"] != outcome_score(outcome["C_req"], outcome["V"])):
        raise ValueError("Saved compound outcome differs from independent physical replay")
    switch = stream["state"]["switch_query_step"]
    qualifying = [row["step"] for row in observations
                  if row["step"] > 0 and row["ap"]["jar_closed"]
                  and row["ap"]["jar_on_support"] and not row["doomed"]]
    stream.update(qualifying_factual_observations=len(qualifying),
                  remaining_native_horizon_at_switch=None if switch is None else 2000 - switch,
                  actual_post_switch_controls=0 if switch is None else max(0, steps - switch),
                  native_continuation_from_trigger_tested=bool(
                      job["continuation"]["mode"] == "privileged_lid_handoff" and switch is not None))
    return {"valid": True, "full_episode": True, "physical": physical, "outcome": outcome,
            "score": outcome_score(outcome["C_req"], outcome["V"]),
            "source_commit": job["repository_commit"], "stream": stream,
            "closure_controller_sha256": closure.sha256(),
            "continuation_contract_sha256": stream["contract_sha256"]}
