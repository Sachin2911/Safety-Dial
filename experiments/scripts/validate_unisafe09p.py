"""Validate the added host's exact schedules, GPU routing and engineering gates."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_infra import atomic_json, utc
from helpers.unisafe09_protocol import digest
from scripts.validate_unisafe09 import paired_prefix, validate_attempt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text())
    jobs = {j["id"]: j for j in freeze["jobs"]}
    valid = {}
    invalid = []
    routes = {}
    for intent_file in sorted((args.root / "attempts").glob("*/intent.json")):
        intent = json.loads(intent_file.read_text())
        if intent["freeze_sha256"] != digest(args.freeze):
            continue
        folder = intent_file.parent
        if (folder / "invalidation.json").exists() or not (folder / "summary.json").exists():
            invalid.append(folder.name)
            continue
        result = json.loads((folder / "process-result.json").read_text())
        if result["exit_code"] != 0 or not result["summary_present"]:
            raise ValueError("Completed summary has an invalid process exit")
        summary, rows = validate_attempt(folder)
        job_id = summary["job"]["id"]
        if job_id not in jobs or summary["job"] != jobs[job_id] or job_id in valid:
            raise ValueError("Unexpected, altered or duplicated job")
        lane = freeze["gpu_lanes"][job_id]
        route = json.loads((folder / "gpu-routing.json").read_text())
        if (intent["instance_id"] != 57690181 or intent["gpu_lane"] != lane
                or route["gpu_index"] != lane or route["torch_current_device"] != lane
                or str(route["renderer_active_cuda_gpus"]).strip(", ") != str(lane)
                or route["cuda_device"] != f"cuda:{lane}" or route["multi_gpu_requested"]
                or "RTX 4090" not in route["gpu_name"]):
            raise ValueError("Actual GPU routing does not match the assigned lane")
        expected_uuid = freeze["gpu_uuids"][str(lane)]
        if route["gpu_uuid"].removeprefix("GPU-") != expected_uuid.removeprefix("GPU-"):
            raise ValueError("CUDA UUID differs from the measured physical GPU")
        valid[job_id] = (folder, summary, rows)
        routes[job_id] = route
    if set(valid) != set(jobs):
        raise ValueError("The full frozen panel is not complete")
    checks = {"complete_valid_panel": True, "all_action_and_outcome_replays_passed": True,
              "gpu_routing_and_physical_uuids_verified": True,
              "no_optimizer_updates": all(v[1]["training_updates"] == 0 for v in valid.values())}
    prefixes = {}
    if freeze["stage"] == "A_parallel":
        for lane in range(4):
            entries = [v for k, v in valid.items() if freeze["gpu_lanes"][k] == lane]
            checks[f"lane_{lane}_native_identity"] = any(
                v[1]["job"]["condition"] == "native_direct" and v[1]["identity_checks"] > 0
                for v in entries
            )
            checks[f"lane_{lane}_full_continuation"] = any(
                v[1]["continuation_steps"] == 40 for v in entries
            )
        for condition in ("native_direct", "d15_g100", "d40_g100"):
            entries = sorted(
                (v for v in valid.values() if v[1]["job"]["condition"] == condition),
                key=lambda v: freeze["gpu_lanes"][v[1]["job"]["id"]],
            )
            for lane in range(1, 4):
                prefixes[f"{condition}:0-{lane}"] = paired_prefix(entries[0], entries[lane])
    timing = [v for v in valid.values() if v[1]["identity_checks"] == 0]
    projection = {
        "max_steady_startup_seconds": max(v[1]["startup_seconds"] for v in timing),
        "max_measured_seconds_per_control": max(
            (v[1]["elapsed_seconds"]-v[1]["startup_seconds"])/v[1]["controls"] for v in timing
        ),
        "max_process_overhead_seconds": max(max(0.0,
            json.loads((v[0]/"process-result.json").read_text())["elapsed_seconds"]
            - v[1]["elapsed_seconds"]) for v in timing),
        "safety_multiplier": 1.25,
        "concurrency": 4,
    }
    record = {"utc": utc(), "passed": all(checks.values()), "checks": checks,
              "freeze_sha256": digest(args.freeze), "invalid_attempts": invalid,
              "episodes": [v[1] for v in valid.values()], "paired_prefixes": prefixes,
              "gpu_routing": routes, "runtime_projection": projection}
    atomic_json(args.output, record)
    print(json.dumps({k: record[k] for k in ("passed", "checks", "runtime_projection")}))
    if not record["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
