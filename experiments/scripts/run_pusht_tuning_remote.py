"""Run the frozen remote study and preserve wall-clock stage timings and logs."""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    args = parser.parse_args()
    design = json.loads(args.design.read_text())
    if os.environ.get("RUNPOD_POD_ID") != design["execution_pod_id"]:
        raise RuntimeError("This pipeline requires the frozen RunPod host")
    receipt = ROOT / "outputs/runpod-pusht-tuning-20261029"
    ready = json.loads((receipt / "environment/ready.json").read_text())
    transfer = json.loads((receipt / "transfer-verification.json").read_text())
    if not ready["cuda_available"] or transfer["mismatches"]:
        raise RuntimeError("Remote setup and source verification must pass first")
    marker = receipt / "pipeline.json"
    if marker.exists():
        raise FileExistsError("Inspect the recorded pipeline before retrying any stage")
    stages = [
        (
            "tests",
            [
                "-m",
                "pytest",
                "-q",
                "experiments/tests/test_safety_model.py",
                "experiments/tests/test_learned_motion.py",
                "experiments/tests/test_pusht_smoke.py",
                "experiments/tests/test_pusht_tuning.py",
                f"--junitxml={receipt / 'tests.xml'}",
            ],
        ),
        *[
            (
                stage,
                ["experiments/scripts/run_pusht_tuning.py", stage, "--design", str(args.design)],
            )
            for stage in ("prepare", "development", "evaluate")
        ],
        ("report", ["experiments/scripts/report_pusht_tuning.py", "--root", design["output"]]),
    ]
    record = {"pod_id": design["execution_pod_id"], "stages": [], "status": "running"}
    for name, command in stages:
        entry = {"stage": name, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
        record["stages"].append(entry)
        marker.write_text(json.dumps(record, indent=2) + "\n")
        before = time.perf_counter()
        with (receipt / f"{name}.log").open("w") as log:
            result = subprocess.run(
                [sys.executable, "-u", *command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
            )
        entry.update(
            {
                "exit_code": result.returncode,
                "seconds": time.perf_counter() - before,
                "ended_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
        )
        if result.returncode:
            record["status"] = "engineering-stop"
            marker.write_text(json.dumps(record, indent=2) + "\n")
            raise SystemExit(result.returncode)
        marker.write_text(json.dumps(record, indent=2) + "\n")
    record["status"] = "complete"
    marker.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
