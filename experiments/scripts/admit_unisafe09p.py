"""Admit all 192 assigned C jobs using measured concurrent engineering throughput."""

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_infra import atomic_json, utc
from helpers.unisafe09_protocol import digest, verify_freeze
from helpers.unisafe09p_infra import current_finances
from scripts.run_unisafe09p import work_deadline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--engineering", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    freeze = verify_freeze(args.freeze)
    accepted = json.loads(args.engineering.read_text())
    rental = json.loads((ROOT / "progress/rental-unisafe09p-20261025.json").read_text())
    if (freeze["stage"] != "C_parallel" or len(freeze["jobs"]) != 192
            or not accepted["passed"] or freeze["engineering_acceptance_sha256"]
            != digest(args.engineering)):
        raise ValueError("The full assigned panel and accepted concurrent engineering are required")
    timing = accepted["runtime_projection"]
    if timing["concurrency"] != 4:
        raise ValueError("Serial timings cannot admit a four-worker stage")
    per_job = timing["safety_multiplier"] * (
        timing["max_steady_startup_seconds"] + timing["max_process_overhead_seconds"]
        + 340*timing["max_measured_seconds_per_control"]
    )
    lane_counts = [sum(freeze["gpu_lanes"][j["id"]] == lane for j in freeze["jobs"])
                   for lane in range(4)]
    if lane_counts != [48]*4:
        raise ValueError("The complete balanced seed-block assignment is required")
    bound = max(lane_counts)*per_job
    available = work_deadline(rental)-time.time()
    _, _, finance = current_finances()
    # Independent original maximum plus the added-host stop and shared reserve is <$15.
    original_rental = json.loads((ROOT / "progress/rental-unisafe09-20261025.json").read_text())
    independent_upper = (15*original_rental["offer"]["dph_total"] + 6.5
                         + finance["other_reserved_or_actual_usd"])
    checks = {
        "complete_stage_fits_time": bound+120 <= available,
        "combined_guard_not_stopping": not finance["combined_stop"],
        "independent_financial_bounds_fit_total": independent_upper < 15,
    }
    result = {
        "utc": utc(), "admitted": all(checks.values()), "checks": checks,
        "freeze_sha256": digest(args.freeze), "engineering_sha256": digest(args.engineering),
        "job_ids": [j["id"] for j in freeze["jobs"]], "lane_job_counts": lane_counts,
        "per_job_bound_seconds": per_job, "remaining_bound_seconds": bound,
        "launch_allowance_seconds": 120, "available_before_preservation_seconds": available,
        "effective_work_deadline_utc": dt.datetime.fromtimestamp(
            work_deadline(rental), dt.timezone.utc).isoformat(),
        "stage_compute_disk_bound_usd": bound/3600*rental["offer"]["dph_total"],
        "combined_finance_at_admission": finance,
        "method": "Maximum observed concurrent warm startup, control time and overhead, times 1.25; 48 full 340-control jobs per GPU",
        "source_sha256": digest(Path(__file__)),
    }
    atomic_json(args.output, result)
    print(json.dumps({k: result[k] for k in (
        "admitted", "checks", "remaining_bound_seconds", "available_before_preservation_seconds",
        "effective_work_deadline_utc", "stage_compute_disk_bound_usd"
    )}))
    if not result["admitted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
