"""Apply the pre-outcome confirmation rule to the full stage-A panel."""

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.screen08_protocol import select_survivors, sha256, summarize_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/screen08")
    args = parser.parse_args()
    plan_path = ROOT / "progress/receipts/08-screen-plan.json"
    plan = json.loads(plan_path.read_text())
    for name, expected in plan["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    episodes, evidence = [], {}
    for path in sorted((args.root / "episodes").glob("*/validation.json")):
        job = json.loads((path.parent / "job.json").read_text())
        if job["stage"] == "stage_a":
            episodes.append(summarize_episode(path.parent))
            evidence[str(path.relative_to(args.root))] = sha256(path)
    expected_jobs = {j["job"] for j in plan["jobs"]}
    if {e["job"] for e in episodes} != expected_jobs:
        raise ValueError("Stage-A jobs are incomplete")
    survivors, comparisons = select_survivors(episodes)
    records = [json.loads(x) for x in (args.root / "attempts_stage_a.jsonl").read_text().splitlines()]
    elapsed = [e["elapsed_s"] for e in records if e["event"] == "simulator_exit"]
    jobs = [j for j in plan["stage_b_jobs"] if j["arm"] in ("native", *survivors)]
    rental = json.loads((ROOT / "progress/rental-screen08-20261025.json").read_text())
    deadline = dt.datetime.fromisoformat(rental["allocation_deadline_utc"]).timestamp()
    required = 1.2 * max(elapsed) * len(jobs) + 2700 if survivors else 0
    fits = required <= deadline - time.time()
    result = {
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(), "plan_sha256": sha256(plan_path),
        "survivors": survivors, "comparisons": comparisons,
        "admit": bool(survivors and fits), "required_seconds": required,
        "remaining_seconds": deadline - time.time(), "evidence_sha256": evidence,
        "reason": "no_survivors" if not survivors else "admitted" if fits else "insufficient_time",
    }
    with (args.root / "stage-b-decision.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
