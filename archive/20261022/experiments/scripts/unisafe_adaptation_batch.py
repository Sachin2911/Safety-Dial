"""Evaluate declared jobs in fresh simulator processes with bounded concurrency."""

import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.unisafe_source import ROOT, save_implementation, sha256


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    workers = plan.get("workers", 1)
    if workers not in (1, 2):
        raise ValueError("This single-GPU batch runner is limited to one or two processes")
    names = [job["name"] for job in plan["jobs"]]
    if len(set(names)) != len(names) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in names):
        raise ValueError("Job names must be distinct simple path components")
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "plan.json", plan)
    save_implementation(args.output)
    jobs = args.output / "jobs"
    jobs.mkdir()
    started = time.perf_counter()

    def run(job):
        name = job["name"]
        for relative, expected in plan.get("frozen_sources_sha256", {}).items():
            if sha256(ROOT / relative) != expected:
                raise RuntimeError(f"Frozen evaluation source changed: {relative}")
        config = jobs / f"{name}.json"
        destination = args.output / name
        write(config, job["plan"])
        begin = time.perf_counter()
        record = {"name": name, "status": "starting", "directory": str(destination)}
        status_file = jobs / f"{name}.status.json"
        write(status_file, record)
        with (jobs / f"{name}.log").open("w") as log:
            process = subprocess.Popen([
                sys.executable, "-u", str(ROOT / "experiments/scripts/unisafe_adaptation_worker.py"),
                "--plan", str(config.resolve()), "--output", str(destination.resolve()),
            ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            record.update(status="running", pid=process.pid, started_unix=time.time())
            write(status_file, record)
            returncode = process.wait()
        record.update(returncode=returncode, elapsed_seconds=time.perf_counter() - begin)
        summary_path = destination / "summary.json"
        if returncode == 0 and summary_path.exists():
            summary = json.loads(summary_path.read_text())
            if summary["status"] == "complete":
                record.update(
                    status="complete", episodes=len(summary["episodes"]),
                    environment_steps=summary["environment_steps"],
                    physics_steps=summary["physics_steps"], query_counts=summary["query_counts"],
                    outcomes=dict(Counter(e["outcome"] for e in summary["episodes"])),
                )
            else:
                record["status"] = "failed"
        else:
            record["status"] = "failed"
        write(status_file, record)
        return record

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run, job): job["name"] for job in plan["jobs"]}
        for future in as_completed(futures):
            try:
                result = future.result()
            except CancelledError:
                result = {"name": futures[future], "status": "cancelled_before_start"}
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
                result = {"name": futures[future], "status": "failed", "error": repr(error)}
            if result["status"] == "failed":
                for pending in futures:
                    pending.cancel()
            results.append(result)
            write(args.output / "progress.json", results)
            print(json.dumps(result), flush=True)
    complete = [r for r in results if r["status"] == "complete"]
    failed = [r for r in results if r["status"] != "complete"]
    queries = sum((Counter(r["query_counts"]) for r in complete), Counter())
    write(args.output / "summary.json", {
        "status": "failed" if failed else "complete", "jobs": results,
        "completed_episodes": sum(r["episodes"] for r in complete),
        "completed_environment_steps": sum(r["environment_steps"] for r in complete),
        "completed_physics_steps": sum(r["physics_steps"] for r in complete),
        "completed_model_queries": dict(queries),
        "elapsed_seconds": time.perf_counter() - started,
        "sum_process_elapsed_seconds_overlapping": sum(r["elapsed_seconds"] for r in complete),
        "failed_job_count": len(failed), "automatic_retries": 0,
        "limits": "If any job failed, completed totals omit its partial artifacts; audit and include those before reporting study totals. Concurrent wall times overlap.",
    })
    if failed:
        raise SystemExit("Batch contains failed jobs; retained partial artifacts must be inspected")


if __name__ == "__main__":
    main()
