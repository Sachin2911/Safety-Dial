"""Validate the complete C schedule across its prospectively assigned physical hosts."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_infra import atomic_json, utc
from helpers.unisafe09_protocol import digest
from scripts.validate_unisafe09 import validate_attempt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--parallel-root", type=Path)
    parser.add_argument("--parallel-freeze", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original_path = ROOT / "progress/receipts/09-unisafe-C.json"
    original = json.loads(original_path.read_text())
    allocation_path = ROOT / "progress/receipts/09-unisafe-parallel-allocation.json"
    allocation = json.loads(allocation_path.read_text())
    assignments = {r["job_id"]: r for r in allocation["assignments"]}
    expected = {j["id"]: j for j in original["jobs"]}
    sources = [("original", args.original_root, original_path, 57647787, 150513)]
    if args.parallel_root:
        if not args.parallel_freeze:
            raise ValueError("The added host requires its own source freeze")
        parallel = json.loads(args.parallel_freeze.read_text())
        if parallel["original_C_freeze_sha256"] != digest(original_path):
            raise ValueError("Different original scientific schedule")
        if parallel["allocation_sha256"] != digest(allocation_path):
            raise ValueError("Different host allocation")
        if not parallel["engineering_acceptance"]["passed"]:
            raise ValueError("Added-host engineering was not accepted")
        for name, sha in original["source_sha256"].items():
            if parallel["source_sha256"].get(name) != sha:
                raise ValueError(f"Shared scientific source differs: {name}")
        sources.append(("parallel", args.parallel_root, args.parallel_freeze, 57690181, 36475))
    valid = {}
    invalid = []
    metadata = {}
    for host, root, freeze_path, instance, machine in sources:
        frozen = json.loads(freeze_path.read_text())
        frozen_jobs = {j["id"]: j for j in frozen["jobs"]}
        for intent_file in sorted((root / "attempts").glob("*/intent.json")):
            intent = json.loads(intent_file.read_text())
            job = intent["job"]
            if job["stage"] != "C":
                continue
            folder = intent_file.parent
            if job["id"] not in expected or job != expected[job["id"]]:
                raise ValueError("Unplanned or altered C job")
            if intent["freeze_sha256"] != digest(freeze_path) or job != frozen_jobs[job["id"]]:
                raise ValueError("Attempt used another source freeze")
            assignment = assignments[job["id"]]
            if args.parallel_root and assignment["host"] != host:
                raise ValueError("Job ran on an unassigned physical host")
            if intent["instance_id"] != instance:
                raise ValueError("Attempt belongs to another rental")
            if (folder / "invalidation.json").exists() or not (folder / "summary.json").exists():
                invalid.append({"host": host, "attempt": folder.name, "job_id": job["id"]})
                continue
            process = json.loads((folder / "process-result.json").read_text())
            if process["exit_code"] != 0 or not process["summary_present"]:
                raise ValueError("Invalid process completion")
            summary, _ = validate_attempt(folder)
            if summary["job"] != job or job["id"] in valid:
                raise ValueError("Changed or duplicate completed C job")
            lane = 0
            if host == "parallel":
                lane = assignment["gpu_lane"]
                route = json.loads((folder / "gpu-routing.json").read_text())
                if (frozen["gpu_lanes"][job["id"]] != lane or intent["gpu_lane"] != lane
                        or route["gpu_index"] != lane or route["torch_current_device"] != lane
                        or str(route["renderer_active_cuda_gpus"]).strip(", ") != str(lane)
                        or route["gpu_uuid"].removeprefix("GPU-")
                        != frozen["gpu_uuids"][str(lane)].removeprefix("GPU-")):
                    raise ValueError("Recorded GPU ownership differs from the assigned lane")
            row = dict(summary)
            row.update(physical_host=str(machine), instance_id=instance, gpu_lane=lane,
                       attempt=folder.name, source_freeze_sha256=digest(freeze_path))
            valid[job["id"]] = row
            metadata[job["id"]] = {
                "intent_sha256": digest(intent_file),
                "summary_sha256": digest(folder / "summary.json"),
                "steps_sha256": digest(folder / "steps.jsonl"),
            }
            if len(valid) % 48 == 0:
                print(json.dumps({"validated_C_jobs": len(valid)}), flush=True)
    if set(valid) != set(expected):
        raise ValueError(f"The complete frozen C panel is required: {len(valid)}/384 valid")
    for seed in {r["seed"] for r in valid.values()}:
        block = [r for r in valid.values() if r["seed"] == seed]
        if len(block) != 6 or len({(r["physical_host"], r["gpu_lane"]) for r in block}) != 1:
            raise ValueError("A seed block was split between hosts or GPUs")
    result = {
        "utc": utc(), "passed": True, "freeze_sha256": digest(original_path),
        "execution_protocol": "Authorized host-allocation amendment; original scientific panel",
        "allocation_sha256": digest(allocation_path),
        "source_freezes": {host: digest(f) for host, _, f, _, _ in sources},
        "checks": {"complete_valid_panel": True, "all_action_and_outcome_replays_passed": True,
                   "host_and_gpu_allocation_verified": True, "no_duplicate_completed_jobs": True},
        "invalid_attempts": invalid, "episodes": [valid[j["id"]] for j in original["jobs"]],
        "raw_input_sha256": metadata,
    }
    atomic_json(args.output, result)
    print(json.dumps({"passed": True, "episodes": len(valid), "physical_hosts": len(sources)}))


if __name__ == "__main__":
    main()
