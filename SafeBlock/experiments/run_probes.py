"""Run the frozen probe job list sequentially, one simulator process per episode.

    nohup /venv/main/bin/python SafeBlock/experiments/run_probes.py --probe A > ... &

Skips jobs that already have a summary.json, records a ledger, never retries by
itself (a failed job is listed in the ledger and the run continues).
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from probe_common import ROOT, atomic_json, project_path, utc  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, help="key under config['probes']")
    parser.add_argument("--config", type=Path, default=ROOT / "SafeBlock/configs/probes-20261101.json")
    parser.add_argument("--output", type=Path, default=project_path("outputs/probes-20261101"))
    parser.add_argument("--only", help="comma-separated job ids")
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--lock-minutes", type=float, default=20, help="skip jobs another lane started this recently")
    args = parser.parse_args()
    args.output = project_path(args.output)
    args.config = args.config if args.config.is_absolute() else (ROOT / args.config)
    config = json.loads(args.config.read_text())
    probe = config["probes"][args.probe]
    jobs = probe["jobs"]
    if args.only:
        wanted = set(args.only.split(","))
        jobs = [j for j in jobs if j["id"] in wanted]
    root = args.output / args.probe
    root.mkdir(parents=True, exist_ok=True)
    ledger = root / "ledger.jsonl"
    for job in jobs:
        attempt_root = root / job["id"]
        attempts = sorted(attempt_root.glob("attempt-*"))
        if any((a / "summary.json").exists() for a in attempts):
            continue
        # Another lane may be running this job: a fresh worker-start without a result.
        in_progress = [
            a
            for a in attempts
            if not (a / "summary.json").exists()
            and not (a / "worker-error.json").exists()
            and time.time() - max(a.stat().st_mtime, (a / "worker-start.json").stat().st_mtime if (a / "worker-start.json").exists() else 0) < args.lock_minutes * 60
        ]
        if in_progress:
            continue
        attempt = attempt_root / f"attempt-{len(attempts) + 1:02d}"
        command = [
            sys.executable,
            str(ROOT / "SafeBlock/experiments/probe_worker.py"),
            "--task",
            job["task"],
            "--condition",
            job["condition"],
            "--seed",
            str(job["seed"]),
            "--output",
            str(attempt),
            "--config",
            str(args.config),
            "--library",
            str(project_path(config["library"])),
        ] + list(job.get("args", []))
        attempt.mkdir(parents=True)
        started = time.perf_counter()
        entry = {"job": job, "attempt": str(attempt.relative_to(ROOT)), "started_utc": utc()}
        with (attempt / "worker.log").open("w") as log:
            try:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, timeout=args.timeout)
                entry["exit_code"] = completed.returncode
            except subprocess.TimeoutExpired:
                entry["exit_code"] = "timeout"
        entry["seconds"] = time.perf_counter() - started
        entry["finished_utc"] = utc()
        summary = attempt / "summary.json"
        if summary.exists():
            s = json.loads(summary.read_text())
            entry["result"] = {
                k: s.get(k)
                for k in ("outcome", "safe_success", "sustained_safe_success", "violation", "steps", "keepout_episode", "plans", "adapter_blocks_changed", "adapter_gripper_overrides", "episodes")
            }
        with ledger.open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)
    # Aggregate what exists.
    table = {}
    for job in probe["jobs"]:
        for attempt in sorted((root / job["id"]).glob("attempt-*")):
            summary = attempt / "summary.json"
            if summary.exists():
                s = json.loads(summary.read_text())
                if s.get("outcome") == "multi":
                    continue
                row = table.setdefault(job["condition"], {"episodes": 0, "safe_success": 0, "sustained": 0, "violation": 0, "timeout": 0, "keepout": {}})
                row["episodes"] += 1
                row["safe_success"] += int(bool(s["safe_success"]))
                row["sustained"] += int(bool(s.get("sustained_safe_success")))
                row["violation"] += int(bool(s["violation"]))
                row["timeout"] += int(bool(s["timeout"]))
                for k, v in s["keepout_episode"].items():
                    row["keepout"][k] = row["keepout"].get(k, 0) + int(v)
    atomic_json(root / "aggregate.json", {"utc": utc(), "probe": args.probe, "by_condition": table})
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
