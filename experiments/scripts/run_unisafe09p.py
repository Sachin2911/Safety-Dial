"""Four fixed GPU lanes with atomic reservations and a shared diagnostic pause."""

import argparse
import datetime as dt
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_infra import atomic_json, utc
from helpers.unisafe09_protocol import digest, verify_freeze
from helpers.unisafe09p_runtime import verify_renderer
from scripts.run_unisafe09 import inventory

OUT = ROOT / "outputs/unisafe09p"


def work_deadline(rental):
    created = dt.datetime.fromisoformat(rental["created_utc"]).timestamp()
    absolute = dt.datetime.fromisoformat(rental["allocation_deadline_utc"]).timestamp()
    money = created + 6.5 / float(rental["offer"]["dph_total"]) * 3600
    return min(absolute, money) - 300 - 1800


def reserve(root, job, freeze, frozen_hash, rental, lane, replacement=False):
    """Count unfinished workers at their full allocation, while holding one lock."""
    with (root / "reservation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / "STOP").exists() or (root / "PAUSE").exists():
            raise RuntimeError("The shared stop or diagnostic pause prevents new work")
        existing = inventory(root)
        previous = [r for r in existing if r["job"]["id"] == job["id"]]
        valid = [r for r in previous if r["valid"]]
        if valid:
            if len(valid) != 1 or valid[0]["freeze_sha256"] != frozen_hash:
                raise ValueError("Duplicate job or incompatible source version")
            return None
        if previous and not replacement:
            raise RuntimeError("An invalid attempt needs a recorded diagnosis before replacement")
        caps = freeze["local_caps"]
        if len(previous) > 1 or (
            previous and sum(r["replacement"] for r in existing) >= caps["replacements"]
        ):
            raise RuntimeError("Infrastructure replacement allowance exhausted")
        maximum = 340 if job["continuation"] else 300
        offline = maximum if job["condition"] == "native_direct" else 0
        if len(existing) >= caps["attempts"]:
            raise RuntimeError("Attempt cap reached")
        if sum(r["charged_controls"] for r in existing) + maximum > caps["controls"]:
            raise RuntimeError("Control-step cap reached")
        if sum(r["offline_queries"] for r in existing) + offline > caps["offline_queries"]:
            raise RuntimeError("Diagnostic query cap reached")
        timeout = 900 if not any(r["gpu_lane"] == lane for r in existing) else 600
        folder = root / "attempts" / f"p{lane}-{len(existing):04d}-{job['id']}"
        folder.mkdir(parents=True, exist_ok=False)
        atomic_json(folder / "intent.json", {
            "utc": utc(), "job": job, "freeze_sha256": frozen_hash,
            "maximum_controls": maximum, "replacement": bool(previous),
            "identity_shadow": bool(offline), "instance_id": rental["instance_id"],
            "gpu_lane": lane, "timeout_seconds": timeout,
        })
        atomic_json(folder / "job.json", job)
        return folder, timeout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--lane", type=int, choices=range(4), required=True)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--only")
    parser.add_argument("--replacement", action="store_true")
    args = parser.parse_args()
    freeze = verify_freeze(args.freeze)
    frozen_hash = digest(args.freeze)
    rental = json.loads((ROOT / "progress/rental-unisafe09p-20261025.json").read_text())
    if os.environ.get("CONTAINER_ID") != str(rental["instance_id"]):
        raise ValueError("This queue belongs only to its recorded rental")
    deadline = work_deadline(rental)
    jobs = [j for j in freeze["jobs"] if freeze["gpu_lanes"][j["id"]] == args.lane]
    if args.only:
        jobs = [j for j in jobs if j["id"] == args.only]
    if not jobs or freeze["stage"] not in {"A_parallel", "C_parallel"}:
        raise ValueError("No assigned frozen jobs")
    if freeze["stage"] == "C_parallel":
        if not args.admission:
            raise ValueError("Confirmation requires complete-stage financial admission")
        admission = json.loads(args.admission.read_text())
        if not admission["admitted"] or admission["freeze_sha256"] != frozen_hash:
            raise ValueError("Invalid stage admission")
        if set(admission["job_ids"]) != {j["id"] for j in freeze["jobs"]}:
            raise ValueError("Admission must cover all 192 assigned jobs")
        if admission["remaining_bound_seconds"] > deadline - time.time():
            raise ValueError("The whole assigned stage no longer fits")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"runner-{args.lane}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(OUT / f"environment/vulkan-lane-{args.lane}.json", verify_renderer(ROOT))
        for job in jobs:
            verify_freeze(args.freeze)
            if deadline - time.time() < 900:
                raise RuntimeError("Insufficient time for the maximum worker timeout")
            reservation = reserve(
                OUT, job, freeze, frozen_hash, rental, args.lane, args.replacement
            )
            if reservation is None:
                continue
            folder, timeout = reservation
            command = [
                "/venv/main/bin/python", "-u", str(ROOT / "experiments/scripts/worker_unisafe09p.py"),
                "--freeze", str(args.freeze.resolve()), "--job", str(folder / "job.json"),
                "--output", str(folder), "--gpu-index", str(args.lane),
            ]
            started = time.monotonic()
            with (folder / "process.log").open("w") as log:
                process = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
                    start_new_session=True,
                )
                atomic_json(folder / "process.json", {"pid": process.pid, "utc": utc()})
                try:
                    while process.poll() is None:
                        if ((OUT / "STOP").exists() or time.time() >= deadline
                                or time.monotonic() - started >= timeout):
                            os.killpg(process.pid, signal.SIGTERM)
                            try:
                                process.wait(20)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL)
                                process.wait()
                            break
                        time.sleep(1)
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(20)
            result = {"exit_code": process.returncode, "elapsed_seconds": time.monotonic()-started,
                      "utc": utc(), "summary_present": (folder / "summary.json").is_file()}
            atomic_json(folder / "process-result.json", result)
            print(json.dumps({"job": job["id"], "lane": args.lane, **result}), flush=True)
            if process.returncode or not result["summary_present"]:
                atomic_json(OUT / "PAUSE", {"utc": utc(), "invalid_attempt": folder.name})
                raise RuntimeError("Invalid worker; all lanes pause after their active episode")
        atomic_json(OUT / f"stage-{freeze['stage']}-lane-{args.lane}-complete.json", {
            "utc": utc(), "freeze_sha256": frozen_hash, "lane": args.lane,
            "jobs": [j["id"] for j in jobs], "partial_request": args.only is not None,
        })


if __name__ == "__main__":
    main()
