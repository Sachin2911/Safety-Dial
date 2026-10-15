"""Run a bounded, recorded set of continuation and implementation diagnostics."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/unisafe/diagnostics.json")
    args = parser.parse_args()
    baseline = json.loads((args.baseline / "summary.json").read_text())
    assert baseline["status"] == "complete"
    plan = json.loads(args.plan.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    completed = []
    started = time.perf_counter()
    for job in plan["jobs"]:
        destination = args.output / job["name"]
        command = [
            sys.executable,
            "-u",
            str(ROOT / "experiments/scripts/unisafe_baseline.py"),
            "--output",
            str(destination.resolve()),
            *job["args"],
        ]
        print("Starting:", job["name"], flush=True)
        with (args.output / f"{job['name']}.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "experiments/scripts/unisafe_analyze.py"),
                    str(destination.resolve()),
                ],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        summary = json.loads((destination / "summary.json").read_text())
        completed.append({"name": job["name"], "steps": summary["environment_steps"]})
        (args.output / "progress.json").write_text(json.dumps(completed, indent=2) + "\n")
        print(
            "Finished:",
            job["name"],
            [(e["seed"], e["outcome"]) for e in summary["episodes"]],
            flush=True,
        )
    (args.output / "protocol_summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "elapsed_seconds": time.perf_counter() - started,
                "completed_jobs": completed,
                "baseline": str(args.baseline),
            },
            indent=2,
        )
        + "\n"
    )
    print("All follow-up runs finished:", args.output, flush=True)


if __name__ == "__main__":
    main()
