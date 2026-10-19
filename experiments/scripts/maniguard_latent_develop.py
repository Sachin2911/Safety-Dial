"""Resume the four full development episodes and bounded saved-input authority audit."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from maniguard_latent_checkpoint import checkpoint

from helpers.maniguard_latent_queue import atomic
from helpers.maniguard_latent_results import read_episode
from helpers.maniguard_latent_services import ensure_policy, service_status, start_runner


def wait_development(run, protocol, jobs, service):
    definitions = json.loads(jobs.read_text())
    resumes = 0
    while True:
        queue = json.loads((run / "queue.json").read_text())
        rows = [queue["jobs"].get(job["id"]) for job in definitions]
        if any(row is not None and row["status"] == "invalid" for row in rows):
            raise RuntimeError("Invalid development receipt; return to diagnosis without replacement")
        if all(row is not None and row["status"] == "completed" for row in rows):
            # A coordinator can still be flushing its final status after the
            # queue marks completion. Wait for its authoritative service state.
            if "RUNNING" in service_status(service):
                time.sleep(5)
                continue
            return
        state = service_status(service)
        if "RUNNING" not in state and "STARTING" not in state:
            if resumes:
                raise RuntimeError("Coordinator returned terminal without a classified receipt; "
                                   "inspect its existing handle and log before another resume")
            # This resumes the saved worker handle if the coordinator alone died.
            start_runner(run, protocol, jobs, service)
            resumes += 1
        time.sleep(5)


def run_audit(run, family):
    queue = json.loads((run / "queue.json").read_text())
    jobs = [queue["jobs"][f"dev-{family}-{condition}"]["output"]
            for condition in ["native", "bias"]]
    for job in jobs:
        read_episode(job, require_full=True)
    output = run / ("authority-" + family)
    if (output / "summary.json").exists():
        summary = json.loads((output / "summary.json").read_text())
        if summary["decoder_calls"] != 160:
            raise ValueError("Incomplete authority audit receipt")
        return
    ensure_policy(run, family, audit=True)
    subprocess.run([sys.executable, str(ROOT / "experiments/scripts/maniguard_latent_audit.py"),
                    "--episodes", *jobs, "--output", str(output)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    args.run, args.protocol = args.run.resolve(), args.protocol.resolve()
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    state_path = args.run / "development-driver.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"phase": "jar-development"}

    def phase(value):
        state.update(phase=value, updated_epoch=time.time())
        atomic(state_path, state)
        status = json.loads((args.run / "status.json").read_text())
        status.update(stage=value, updated_epoch=time.time())
        atomic(args.run / "status.json", status)
        print("Stage", value, flush=True)

    if state["phase"] == "jar-development":
        jobs = args.run / "jar-development-jobs.json"
        wait_development(args.run, args.protocol, jobs, "maniguard_latent_development_jar")
        phase("jar-authority")
    if state["phase"] == "jar-authority":
        run_audit(args.run, "jar")
        checkpoint(args.run, "development-jar")
        phase("clutter-development")
    if state["phase"] == "clutter-development":
        ensure_policy(args.run, "clutter")
        definitions = [{"id": "engineering-clutter-v6", "stage": "engineering",
                        "engineering": True, "max_steps": 32, "family": "clutter",
                        "scene": "task_0039/base", "seed": 6101805,
                        "controller": str(args.run / "initial-controllers/initial-0.npz")}]
        protocol = json.loads(args.protocol.read_text())
        for job in protocol["development_episodes"][2:]:
            definitions.append({k: v for k, v in job.items() if k != "initial_controller"} | {
                "stage": "development", "controller": str(args.run / "initial-controllers"
                                                             / f"initial-{job['initial_controller']}.npz")})
        jobs = args.run / "clutter-development-jobs.json"
        if jobs.exists() and json.loads(jobs.read_text()) != definitions:
            raise ValueError("Clutter development job definitions changed")
        if not jobs.exists():
            atomic(jobs, definitions)
        start_runner(args.run, args.protocol, jobs, "maniguard_latent_development_clutter")
        wait_development(args.run, args.protocol, jobs, "maniguard_latent_development_clutter")
        phase("clutter-authority")
    if state["phase"] == "clutter-authority":
        run_audit(args.run, "clutter")
        checkpoint(args.run, "development-clutter")
        phase("capacity-decision-pending")
    print("Development and bounded authority audit finished; inspect timing before search", flush=True)


if __name__ == "__main__":
    main()
