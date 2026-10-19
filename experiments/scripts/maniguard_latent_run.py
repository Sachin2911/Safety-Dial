"""Resume an immutable serial job list, returning worker failures to recovery."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.maniguard_latent_queue import PhysicalQueue, atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--episode-seconds", type=float, default=180)
    parser.add_argument("--repair-note")
    args = parser.parse_args()
    jobs = json.loads(args.jobs.read_text())
    queue = PhysicalQueue(args.run, args.protocol)
    queue.plan(jobs)
    for job in jobs:
        print("Observe or execute", job["id"], flush=True)
        receipt = queue.evaluate(job["id"], repair_note=args.repair_note,
                                 episode_seconds=args.episode_seconds)
        status_path = args.run / "status.json"
        status = json.loads(status_path.read_text())
        status.update(stage=job["stage"], updated_epoch=time.time(),
                      last_completed_job=job["id"],
                      last_completed_receipt=queue.state["jobs"][job["id"]]["output"],
                      full_episodes_started=sum(not a["engineering"]
                                                for a in queue.state["attempts"]))
        atomic(status_path, status)
        print(json.dumps({"id": job["id"], "score": receipt["score"],
                          "steps": receipt["physical"]["steps"],
                          "elapsed": receipt["physical"]["ledger"]["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
