"""Run the declared task 15 stage D: projection-0.10 on the stage C seeds and tasks.

Stage D was added after the stage B screen and declared in
progress/receipts/15-regrasp-stage-d.json before any stage C outcome was examined. It
reuses the frozen queue, adapter, validation and design of the live freeze unchanged;
only the job list and its ceilings come from the declaration. Its comparator is the stage
C native arm on the same seeds and tasks. Run with the behavior environment's Python
after stage C, on the same rental and policy server.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/scripts"))
sys.path.insert(0, str(ROOT / "experiments/helpers"))
import run_regrasp15  # noqa: E402

DECLARATION = ROOT / "progress/receipts/15-regrasp-stage-d.json"


class StageD(run_regrasp15.Queue):
    def stage_jobs(self):
        self.declaration = json.loads(DECLARATION.read_text())
        validated = {e.get("job") for e in self.history if e["event"] == "validated"}
        if not all(j["job"] in validated for j in self.design["stage_a_jobs"]):
            raise ValueError("Every stage A episode must validate first")
        if not (run_regrasp15.OUT / "stage_c-complete.json").exists():
            raise ValueError("Stage D runs only after the stage C queue has finished")
        if self.declaration["arm"] not in self.design["arms"]:
            raise ValueError("Stage D arm is not a frozen design arm")
        jobs, number = [], 500
        for seed in self.declaration["seeds"]:
            for task in self.declaration["tasks"]:
                jobs.append({"job": number, "stage": "stage_d", "arm": self.declaration["arm"],
                             "task": task, "seed": seed})
                number += 1
        return jobs

    def admit(self, job, attempt):
        frozen = self.design["ceilings"]
        self.design["ceilings"] = {**frozen, **self.declaration["ceilings"]}
        try:
            return super().admit(job, attempt)
        finally:
            self.design["ceilings"] = frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--design", type=Path, default=ROOT / "configs/regrasp15/design.json")
    parser.add_argument("--rental", type=Path,
                        default=ROOT / "progress/rental-regrasp15-20261025.json")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan["source_sha256"].get(str(DECLARATION.relative_to(ROOT))) != hashlib.sha256(
        DECLARATION.read_bytes()
    ).hexdigest():
        raise ValueError("The stage D declaration is not part of this freeze")
    queue = StageD(args.plan.resolve(), args.design.resolve(), args.rental.resolve(), "stage_d")
    try:
        finished = queue.drain()
    except Exception as error:
        (run_regrasp15.OUT / "STOP").write_text(f"stage_d: {type(error).__name__}\n")
        queue.record("queue_error", kind=type(error).__name__, detail=str(error))
        raise
    queue.record("queue_finished", stage="stage_d", starts=queue.starts,
                 replacements=queue.replacements, finished=finished)
    validated = {e.get("job") for e in queue.history if e["event"] == "validated"}
    missing = [j["job"] for j in queue.jobs if j["job"] not in validated]
    (run_regrasp15.OUT / "stage_d-complete.json").write_text(json.dumps(
        {"utc": run_regrasp15.utc(), "missing_jobs": missing, "finished": finished},
        indent=2) + "\n")


if __name__ == "__main__":
    main()
