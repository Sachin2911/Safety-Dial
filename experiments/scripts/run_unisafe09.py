"""Sequential bounded jobs with a durable reservation and attempt ledger."""

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
from helpers.unisafe09_rendering import verify_renderer

OUT = ROOT / "outputs/unisafe09"


def inventory(root=OUT):
    attempts = []
    for intent in sorted((root / "attempts").glob("*/intent.json")):
        row = json.loads(intent.read_text())
        row["directory"] = str(intent.parent)
        account = intent.parent / "accounting.json"
        summary = intent.parent / "summary.json"
        invalidation = intent.parent / "invalidation.json"
        if summary.exists() and not invalidation.exists():
            result = json.loads(summary.read_text())
            row.update(
                valid=result["status"] == "complete",
                result=result,
                charged_controls=result["controls"],
                controls_uncertain=False,
                offline_queries=result["identity_checks"],
            )
        else:
            data = json.loads(account.read_text()) if account.exists() else {}
            # A dead worker can have advanced physics before its next durable write.
            row.update(
                valid=False,
                accounting=data,
                charged_controls=row["maximum_controls"],
                controls_uncertain=True,
                offline_queries=row["maximum_controls"] if row["identity_shadow"] else 0,
            )
            if invalidation.exists():
                row["invalidation"] = json.loads(invalidation.read_text())
        attempts.append(row)
    return attempts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument(
        "--rental", type=Path, default=ROOT / "progress/rental-unisafe09-20261025.json"
    )
    parser.add_argument("--only", help="Run a single fixed job ID")
    parser.add_argument("--replacement", action="store_true")
    parser.add_argument("--admission", type=Path)
    args = parser.parse_args()
    freeze = verify_freeze(args.freeze)
    if freeze["stage"] == "preparation":
        raise ValueError("Freeze the engineering stage before simulator execution")
    frozen_hash = digest(args.freeze)
    rental = json.loads(args.rental.read_text())
    if str(rental["instance_id"]) != os.environ.get("CONTAINER_ID"):
        raise ValueError("This runner may only execute on its recorded rental")
    deadline = dt.datetime.fromisoformat(rental["allocation_deadline_utc"]).timestamp() - 1800
    if freeze["stage"] in ("B", "C"):
        if not args.admission:
            raise ValueError("Scored stages require a complete-stage financial admission")
        admission = json.loads(args.admission.read_text())
        if not admission["admitted"] or admission["freeze_sha256"] != frozen_hash:
            raise ValueError("Invalid stage admission")
        if len(admission["job_ids"]) != len(freeze["jobs"]):
            raise ValueError("Admission must cover the complete stage")
        if admission["remaining_bound_seconds"] > deadline - time.time():
            raise ValueError("The complete stage no longer fits the time allowance")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(OUT / "environment/vulkan-readiness.json", verify_renderer(ROOT))
        jobs = [j for j in freeze["jobs"] if args.only is None or j["id"] == args.only]
        if not jobs:
            raise ValueError("No such frozen job")
        for job in jobs:
            verify_freeze(args.freeze)
            if (OUT / "STOP").exists() or time.time() >= deadline:
                raise RuntimeError("Preservation margin reached; no further jobs admitted")
            existing = inventory()
            previous = [r for r in existing if r["job"]["id"] == job["id"]]
            valid = [r for r in previous if r["valid"]]
            if valid:
                if len(valid) != 1 or valid[0]["freeze_sha256"] != frozen_hash:
                    raise ValueError("Do not pool or overwrite scored source versions")
                continue
            if previous and not args.replacement:
                raise RuntimeError("Diagnose the invalid attempt before admitting its replacement")
            if len(previous) > 1 or (
                previous
                and sum(r["replacement"] for r in existing)
                >= freeze["design"]["caps"]["infrastructure_replacements"]
            ):
                raise RuntimeError("Frozen replacement allowance exhausted")
            max_controls = 340 if job["continuation"] else 300
            caps = freeze["design"]["caps"]
            if len(existing) >= caps["maximum_attempts"]:
                raise RuntimeError("Attempt cap reached")
            if (
                sum(r["charged_controls"] for r in existing) + max_controls
                > caps["maximum_controls"]
            ):
                raise RuntimeError("Control-step cap reached")
            offline_reserve = max_controls if job["condition"] == "native_direct" else 0
            if (
                sum(r["offline_queries"] for r in existing) + offline_reserve
                > caps["maximum_offline_controller_queries"]
            ):
                raise RuntimeError("Diagnostic query cap would be exceeded")
            timeout = 1800 if not existing else 600
            if deadline - time.time() < timeout:
                raise RuntimeError("This job's timeout would cross the preservation margin")
            directory = OUT / "attempts" / f"{len(existing):04d}-{job['id']}"
            directory.mkdir(parents=True, exist_ok=False)
            intent = {
                "utc": utc(),
                "job": job,
                "freeze_sha256": frozen_hash,
                "maximum_controls": max_controls,
                "replacement": bool(previous),
                "identity_shadow": job["condition"] == "native_direct",
                "instance_id": rental["instance_id"],
                "timeout_seconds": timeout,
            }
            atomic_json(directory / "intent.json", intent)
            atomic_json(directory / "job.json", job)
            command = [
                "/venv/main/bin/python",
                "-u",
                str(ROOT / "experiments/scripts/worker_unisafe09.py"),
                "--freeze",
                str(args.freeze.resolve()),
                "--job",
                str(directory / "job.json"),
                "--output",
                str(directory),
            ]
            started = time.monotonic()
            with (directory / "process.log").open("w") as log:
                process = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, start_new_session=True
                )
                atomic_json(directory / "process.json", {"pid": process.pid, "utc": utc()})
                try:
                    while process.poll() is None:
                        if (
                            (OUT / "STOP").exists()
                            or time.time() >= deadline
                            or time.monotonic() - started >= timeout
                        ):
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
            result = {
                "exit_code": process.returncode,
                "elapsed_seconds": time.monotonic() - started,
                "utc": utc(),
                "summary_present": (directory / "summary.json").is_file(),
            }
            atomic_json(directory / "process-result.json", result)
            atomic_json(OUT / "attempt-ledger.json", {"utc": utc(), "attempts": inventory()})
            print(json.dumps({"job": job["id"], **result}), flush=True)
            if process.returncode or not result["summary_present"]:
                raise RuntimeError("Worker invalid; dependent scoring paused for diagnosis")
        atomic_json(
            OUT / f"stage-{freeze['stage']}-complete.json",
            {
                "utc": utc(),
                "freeze_sha256": frozen_hash,
                "jobs": [j["id"] for j in jobs],
                "partial_request": args.only is not None,
            },
        )


if __name__ == "__main__":
    main()
