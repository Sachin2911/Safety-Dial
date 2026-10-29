"""Admit a complete scored panel only when its conservative runtime fits the rental."""

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_infra import atomic_json, utc
from helpers.unisafe09_protocol import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--engineering", type=Path, required=True)
    parser.add_argument(
        "--rental", type=Path, default=ROOT / "progress/rental-unisafe09-20261025.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text())
    acceptance = json.loads(args.engineering.read_text())
    rental = json.loads(args.rental.read_text())
    if not acceptance["passed"] or freeze["stage"] not in {"B", "C"}:
        raise ValueError("A complete engineering pass is required for a scored stage")
    timing = acceptance["runtime_projection"]
    bounds = []
    for job in freeze["jobs"]:
        controls = 340 if job["continuation"] else 300
        bound = timing["safety_multiplier"] * (
            timing["max_steady_startup_seconds"]
            + timing["max_process_overhead_seconds"]
            + controls * timing["max_measured_seconds_per_control"]
        )
        bounds.append(bound)
    deadline = dt.datetime.fromisoformat(rental["allocation_deadline_utc"]).timestamp() - 1800
    available = deadline - time.time()
    estimate = sum(bounds)
    # Small launch allowance avoids admitting a stage exactly at its measured boundary.
    required = estimate + 120
    result = {
        "utc": utc(),
        "admitted": required <= available,
        "freeze_sha256": digest(args.freeze),
        "engineering_sha256": digest(args.engineering),
        "job_ids": [j["id"] for j in freeze["jobs"]],
        "per_job_bounds_seconds": bounds,
        "remaining_bound_seconds": estimate,
        "launch_allowance_seconds": 120,
        "available_before_preservation_seconds": available,
        "rate_usd_hour_including_disk": rental["offer"]["dph_total"],
        "stage_compute_disk_bound_usd": estimate / 3600 * rental["offer"]["dph_total"],
        "method": "maximum observed warm startup and per-control time, plus process overhead, times 1.25; full horizons and complete panel",
    }
    atomic_json(args.output, result)
    print(json.dumps(result))
    if not result["admitted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
