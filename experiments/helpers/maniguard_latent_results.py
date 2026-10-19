"""Independent physical receipt checks and controller replay for the latent pilot."""

import hashlib
import json
import zlib
from pathlib import Path

import numpy as np

from helpers.maniguard_latent import LatentController, feature_vector, outcome_score
from helpers.maniguard_latent_policy import array_hash
from helpers.maniguard_phase2_outcomes import derive_outcome
from helpers.maniguard_results import read_episode as read_physical


def read_episode(output, *, require_full=True):
    output = Path(output)
    job = json.loads((output / "job.json").read_text())
    if require_full and (job["max_steps_override"] is not None or job["random_policy"]):
        raise ValueError("Engineering receipts cannot supply comparative fitness")
    physical = read_physical(output)
    rows = [json.loads(line) for line in (output / "trace.jsonl").read_text().splitlines()]
    results = [row["result"] for row in rows if row["type"] == "result"]
    if len(results) != 1:
        raise ValueError("Exactly one native physical result is required")
    result = results[0]
    raw = [json.loads(line) for line in (output / "episode/results.jsonl").read_text().splitlines()]
    if raw != results:
        raise ValueError("Recorded benchmark and telemetry verdicts disagree")
    jobs = [row["job"] for row in rows if row["type"] == "job"]
    if jobs != [job]:
        raise ValueError("Telemetry job differs from the archived job")
    for name, expected in job["code_hashes"].items():
        if hashlib.sha256((output / "study_source" / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Archived source checksum mismatch: " + name)
    provenance = json.loads((output / "runtime-provenance.json").read_text())
    if hashlib.sha256((output / "benchmark_instrumented.py").read_bytes()).hexdigest() != \
            provenance["runtime_sha256"]:
        raise ValueError("Executed evaluator copy differs from its source receipt")
    with np.load(output / "controller.npz", allow_pickle=False) as data:
        controller = LatentController(data["weights"])
    if controller.sha256() != job["controller_sha256"]:
        raise ValueError("Controller checkpoint hash mismatch")
    observed = [row for row in rows if row["type"] == "observed"]
    observations = []
    for row in observed:
        monitor = row["monitor"]
        if not monitor or monitor.get("valid") is not True:
            raise ValueError("Invalid or missing physical monitor observation")
        observations.append({"step": row["step"], **monitor["observation"]})
    goals = [row for row in rows if row["type"] == "goal"]
    outcome = derive_outcome(job["family"], result, observations, goals)
    if not outcome["valid"]:
        raise ValueError("Requested-sequence outcome is invalid: " + str(outcome["invalid_reasons"]))
    steps = result["steps"]
    if [row["step"] for row in observed] != list(range(steps + 1)):
        raise ValueError("Missing, duplicated or unordered physical observations")
    executed = [row for row in rows if row["type"] == "executed"]
    proposals = [row for row in rows if row["type"] == "proposal"]
    if [row["step"] for row in executed] != list(range(steps)) or len(proposals) != steps:
        raise ValueError("Incomplete proposed/executed action history")
    last_executed, previous_eef = None, None
    current_features = None
    current_chunk = None
    chunk_start = None
    previous_rng = None
    query_index = 0
    observed_step, executed_step = -1, -1
    initial_scene = next(row["scene"] for row in rows if row["type"] == "initial")
    expected_seed = zlib.crc32(f"{job['seed']}:{initial_scene['name']}".encode()) & 0x7FFFFFFF
    for row in rows:
        kind = row["type"]
        if kind == "observed":
            if row["step"] != executed_step + 1:
                raise ValueError("Observation is not causally aligned with executed actions")
            observed_step = row["step"]
        elif kind == "latent_features":
            if row["step"] != observed_step:
                raise ValueError("Steering features do not use the current pre-action observation")
            values = np.asarray(row["features"], dtype=np.float32)
            frame = row["frame"]
            replay = feature_vector(frame["joints"], frame["fingers"], frame["eef"],
                                    values[20:28], values[28:36], values[36:44],
                                    previous_executed=last_executed, previous_eef=previous_eef)
            np.testing.assert_array_equal(values, replay)
            np.testing.assert_array_equal(controller.code(values), np.asarray(row["code"], np.float32))
            current_features = values
            previous_eef = frame["eef"]
            saved = output / "query_observations" / f"{row['step']:06d}.npz"
            with np.load(saved, allow_pickle=False) as data:
                np.testing.assert_array_equal(data["features"], values)
                for key, image in [("overview", data["overview"]), ("wrist", data["wrist"])]:
                    expected_hash = row["context"]["image_hashes"][0 if key == "overview" else 1]
                    if hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest() != expected_hash:
                        raise ValueError("Saved actor image differs from encoder input")
        elif kind == "noise_receipt":
            receipt = row["receipt"]
            if (row["step"] != observed_step or receipt["query_index"] != query_index
                    or receipt["seed"] != expected_seed or receipt["episode"] != job["episode_id"]
                    or receipt["weights_sha256"] != controller.sha256()
                    or receipt["features_sha256"] != array_hash(current_features)
                    or receipt["identity_native_path"] != controller.identity
                    or receipt["audit"] is not False or receipt["status"] != "returned"):
                raise ValueError("Server did not execute the causal frozen controller")
            if previous_rng is not None and receipt["rng_before"] != previous_rng:
                raise ValueError("Native policy RNG chain broke between queries")
            previous_rng = receipt["rng_after"]
            raw_actions = np.asarray(row["raw_actions"], dtype=row["raw_action_dtype"])
            if array_hash(raw_actions) != receipt["actions_sha256"]:
                raise ValueError("Server action bytes differ from the received raw chunk")
            current_chunk = raw_actions.astype(np.float32)
            if current_chunk.shape != (16, 8) or not np.isfinite(current_chunk).all():
                raise ValueError("Wrong decoded physical action shape or nonfinite output")
            chunk_start = row["step"]
            query_index += 1
        elif kind == "proposal":
            if row["step"] != observed_step:
                raise ValueError("A proposal uses a stale observation")
            if not job["random_policy"]:
                expected = current_chunk[row["step"] - chunk_start].copy()
                expected[-1] = np.sign(expected[-1]) if abs(expected[-1]) > .01 else -1.
                np.testing.assert_array_equal(np.asarray(row["original"], np.float32), expected)
            np.testing.assert_array_equal(row["original"], row["adapted"])
        elif kind == "executed":
            if row["step"] != observed_step or row["step"] != executed_step + 1:
                raise ValueError("Executed action is causally misaligned")
            last_executed = row["action"]
            executed_step = row["step"]
    if not job["random_policy"] and query_index != physical["ledger"]["policy_queries"]:
        raise ValueError("Policy query accounting differs from actual server receipts")
    if physical["ledger"]["encoder_image_calls"] != len(
            [row for row in rows if row["type"] == "latent_features"]):
        raise ValueError("Semantic encoder accounting differs from query features")
    return {"valid": True, "full_episode": job["max_steps_override"] is None,
            "score": outcome_score(outcome["C_req"], outcome["V"]),
            "outcome": outcome, "physical": physical,
            "source_commit": job["repository_commit"], "controller_sha256": controller.sha256()}
