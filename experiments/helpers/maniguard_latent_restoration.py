"""Bind restored physical episodes to the original queue, without resuming jobs."""

import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

from helpers.maniguard_continuation_results import join_monitor_history
from helpers.maniguard_latent_completion import runtime_errors, source_errors
from helpers.maniguard_latent_results import read_episode

ROOT = Path(__file__).resolve().parents[2]


def require_complete_source_hashes(protocol, hashes):
    """Match the original design verifier's exact thirteen-source authority."""
    expected = {"experiments/helpers/" + name + ".py" for name in (
        "maniguard_latent", "maniguard_latent_policy", "maniguard_latent_runtime",
        "maniguard_latent_source", "maniguard_runtime", "maniguard_source",
        "maniguard_adaptation", "maniguard_phase2_features", "maniguard_phase2_outcomes",
    )}
    expected.update(("experiments/scripts/maniguard_latent_episode.py",
                     "configs/maniguard/phase2/encoder.json",
                     "configs/maniguard/phase2/completion.json", protocol["source_pins"]))
    if (len(expected) != 13 or not isinstance(hashes, dict) or set(hashes) != expected
            or any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                   for value in hashes.values())):
        raise ValueError("Require exactly the 13 frozen source paths and valid SHA256 digests")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_restored_attempt(attempt, location, hashes, policies):
    output = Path(location["output"]).resolve()
    recovery_path = Path(location["recovery_receipt"]).resolve()
    recovery = json.loads(recovery_path.read_text())
    expected = Path(recovery["merged"]).resolve() / "episodes" / attempt["id"]
    archives = recovery["archives"]
    if (output != expected or output == Path(attempt["output"]).resolve()
            or recovery.get("all_archives_redownloaded_and_sha256_checked") is not True
            or not archives or {row["repo_type"] for row in archives} != {"model", "dataset"}
            or any(re.fullmatch(r"[0-9a-f]{40}", row["revision"]) is None
                   or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None for row in archives)):
        raise ValueError("Use a complete separately restored model/dataset stage for this attempt")
    receipt = read_episode(output)
    saved = json.loads((output / "verified-receipt.json").read_text())
    for key in ("valid", "full_episode", "score", "outcome", "source_commit", "controller_sha256"):
        if receipt[key] != saved[key]:
            raise ValueError("Restored physical replay differs from its saved receipt: " + key)
    if not receipt["valid"] or not receipt["full_episode"]:
        raise ValueError("An original full episode has no valid restored physical receipt")
    physical = receipt["physical"]
    if ({k: v for k, v in physical.items() if k != "output"}
            != {k: v for k, v in saved["physical"].items() if k != "output"}):
        raise ValueError("Restored physical counters/evidence differ from the saved receipt")
    original = json.loads((Path(attempt["output"]) / "verified-receipt.json").read_text())
    if saved != original:
        raise ValueError("The restored receipt differs from the original queue's receipt")
    job = json.loads((output / "job.json").read_text())
    if job["episode_id"] != str(Path(attempt["output"]).resolve()):
        raise ValueError("The restored job belongs to a different original episode")
    rows = [json.loads(line) for line in (output / "trace.jsonl").read_text().splitlines()]
    early = attempt["stage"] == "development" and "policy_gpu_uuid" not in job
    errors = runtime_errors(job, rows, physical["steps"], require_device_receipts=not early)
    errors += source_errors(job, {} if attempt["stage"] == "development" else hashes,
                            policies[job["family"]])
    if errors:
        raise ValueError("Restored original runtime/source differs: " + "; ".join(errors))
    result = next(row["result"] for row in rows if row["type"] == "result")
    native = json.loads((output / "episode" / (result["scene_name"] + "_ltl.json")).read_text())
    monitor_count = join_monitor_history(rows, native["log"])
    outcome = receipt["outcome"]
    return {"attempt": attempt["id"], "stage": attempt["stage"],
            "controller_sha256": receipt["controller_sha256"], "policy_controls": physical["steps"],
            "policy_queries": physical["ledger"]["policy_queries"], "C_req": outcome["C_req"],
            "V": outcome["V"], "score": receipt["score"], "scene": job["scene"], "seed": job["seed"],
            "restored_output": str(output), "recovery_receipt": str(recovery_path),
            "recovery_receipt_sha256": digest(recovery_path), "archives": archives,
            "verified_receipt_sha256": digest(output / "verified-receipt.json"),
            "exact_native_monitor_entries_joined": monitor_count,
            "early_development_device_receipt_limitation": early}


def build_inventory(run, protocol_path, paths_path, source_proof_path, *, require_complete=True):
    run, protocol_path, paths_path = Path(run), Path(protocol_path), Path(paths_path)
    qraw = (run / "queue.json").read_bytes()
    queue, protocol = json.loads(qraw), json.loads(protocol_path.read_text())
    paths = json.loads(paths_path.read_text())
    full = [a for a in queue["attempts"] if not a["engineering"]]
    completed = [a for a in full if a["status"] == "completed"]
    names = {a["id"] for a in completed}
    if (len(names) != len(completed) or len(full) > 64
            or (require_complete and (len(completed) != 60
                or any(a["status"] not in ("completed", "invalid") for a in full)))
            or paths["original_queue_sha256"] != hashlib.sha256(qraw).hexdigest()
            or set(paths["episodes"]) != names):
        raise ValueError("Restoration paths must cover the exact current original queue")
    source_proof = json.loads(Path(source_proof_path).read_text())
    if (source_proof["protocol_sha256"] != digest(protocol_path)
            or source_proof["frozen_critical_source_commit"] != protocol["source_commit"]):
        raise ValueError("Original frozen source proof does not match its protocol")
    hashes = source_proof["frozen_critical_source_hashes"]
    require_complete_source_hashes(protocol, hashes)
    for name, expected in hashes.items():
        data = subprocess.check_output(["git", "-C", str(ROOT), "show", protocol["source_commit"] + ":" + name])
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Frozen source proof differs from Git: " + name)
    pins = subprocess.check_output(["git", "-C", str(ROOT), "show",
                                    protocol["source_commit"] + ":" + protocol["source_pins"]])
    policies = json.loads(pins)["policies"]
    rows = []
    for attempt in completed:
        locations = paths["episodes"][attempt["id"]]
        if not locations:
            raise ValueError("An original completed episode has no restored location")
        rows.append(verify_restored_attempt(attempt, locations[0], hashes, policies))
    if (run / "queue.json").read_bytes() != qraw:
        raise ValueError("The original queue changed during restoration validation")
    return {"schema": "maniguard-original-physical-restoration-proof-v1",
            "verified_epoch": time.time(), "protocol_sha256": digest(protocol_path),
            "queue_sha256": hashlib.sha256(qraw).hexdigest(), "episodes": rows,
            "physical_receipts_reverified": True,
            "all_archives_redownloaded_and_sha256_checked": True,
            "all_60_original_valid_episodes_covered": len(rows) == 60,
            "frozen_source_commit": protocol["source_commit"],
            "canonical_frozen_source_paths_verified": sorted(hashes),
            "path_inventory_sha256": digest(paths_path),
            "source_proof_sha256": digest(source_proof_path),
            "verification_source_hashes": {
                "experiments/helpers/" + name + ".py": digest(
                    ROOT / "experiments/helpers" / (name + ".py"))
                for name in ("maniguard_latent_restoration", "maniguard_latent_results",
                             "maniguard_results", "maniguard_latent_completion",
                             "maniguard_continuation_results")},
            "scope": "Full restored-array, action, outcome, source, native cadence, camera, "
                     "asset/device and exact original-monitor replay. Early development GPU "
                     "limitations are retained. This proof does not replace final design "
                     "verification, complete cost accounting or launch permission.",
            "new_physical_controls": 0, "new_neural_policy_or_encoder_calls": 0}
