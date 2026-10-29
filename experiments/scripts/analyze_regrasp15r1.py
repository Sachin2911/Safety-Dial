"""Task 15 final analysis across the two cohorts, with explicit input roots.

Recovery cohort (replacement host): the frozen stage C primary contrast, r1 select-0.06
against r1 native (one-sided exact Barnard test at alpha 0.05, Wilson intervals,
task-stratified bootstrap), and the declared stage D contrast, r1 project-0.10 against
r1 stage C native (same tests, plus raw-violation episodes one-sided for fewer under
projection). Historical cohort (original host): stage A, the stage B screen and the
partial stage C, descriptive only. Nothing is pooled across cohorts or stages. Also
writes the actual-against-planned schedule and the cumulative interaction counters.

    .venv/bin/python experiments/scripts/analyze_regrasp15r1.py
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/scripts"))
import analyze_regrasp15 as base  # noqa: E402
import analyze_regrasp15_d as stage_d  # noqa: E402
from run_regrasp15r1 import completion, cumulative_usage  # noqa: E402

RECOVERY = ROOT / "configs/regrasp15r1/recovery.json"


def ledger(root, stage):
    path = Path(root) / f"attempts_{stage}.jsonl"
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def validated(events):
    return [e for e in events if e["event"] == "validated"]


def historical_stage_c(root):
    """Locally complete episodes plus the host validation events mirrored in the queue log."""
    complete = {e["job"]: e for e in validated(ledger(root, "stage_c"))}
    log = Path(root) / "logs/stage-c-queue.log"
    events = [json.loads(x) for x in log.read_text().splitlines() if x.startswith("{")]
    mirrored = {e["job"]: e for e in validated(events)}
    launched = [e["job"] for e in events if e["event"] == "simulator_launch"]
    return {
        "episodes_complete_locally": sorted(complete),
        "host_validation_events_only": sorted(set(mirrored) - set(complete)),
        "launched_without_mirrored_outcome": sorted(set(launched) - set(mirrored)),
        "summary_with_mirrored_events": base.summarize(list(mirrored.values())),
        "note": "Original host, interrupted at job 305 or 306; descriptive only.",
    }


def schedule(events, jobs):
    rows = []
    for job in jobs:
        mine = [e for e in events if e.get("job") == job["job"]]
        starts = [e for e in mine if e["event"] == "start_reserved"]
        done = [e for e in mine if e["event"] == "validated"]
        refused = [e for e in mine if e["event"] in ("admission_refused", "admission_closed")]
        status = ("validated" if done else "abandoned" if any(
            e["event"] == "job_abandoned" for e in mine) else "not_admitted" if refused or not
            starts else "unresolved")
        rows.append({**job, "status": status, "attempts": len(starts),
                     "elapsed_s": done[-1]["elapsed_s"] if done else None})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-root", type=Path,
                        default=ROOT / "outputs/regrasp15r1/remote")
    parser.add_argument("--historical-root", type=Path,
                        default=ROOT / "outputs/regrasp15/remote")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/regrasp15r1/analysis")
    args = parser.parse_args()
    recovery = json.loads(RECOVERY.read_text())
    r1 = {stage: ledger(args.recovery_root, stage) for stage in ("stage_a", "stage_c", "stage_d")}
    events = [e for stage in r1.values() for e in stage]
    stage_c = validated(r1["stage_c"])
    stage_d_rows = validated(r1["stage_d"])
    native_c = [r for r in stage_c if r["arm"] == "native"]
    arm_c = recovery["stage_c_decision"]["arm"]
    arm_d = recovery["analysis"]["stage_d"]["treated"][1]
    jobs = {"stage_a": recovery["engineering_jobs"], "stage_c": recovery["stage_c_jobs"],
            "stage_d": recovery["stage_d_jobs"]}
    finished = {stage: any(e["event"] == "queue_finished" and e.get("finished")
                           for e in r1[stage]) for stage in r1}
    server = args.recovery_root / "server/candidates.jsonl"
    decodes = sum(json.loads(x)["k"] for x in server.read_text().splitlines() if x) \
        if server.exists() else 0
    analysis = {
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "inputs": {"recovery_root": str(args.recovery_root),
                   "historical_root": str(args.historical_root)},
        "recovery": {
            "completion": {stage: completion(r1[stage], jobs[stage], finished[stage])
                           for stage in r1},
            "stage_a": base.summarize(validated(r1["stage_a"])),
            "stage_c": base.summarize(stage_c),
            "stage_c_primary": base.contrast(stage_c, arm_c),
            "stage_d": base.summarize(native_c + stage_d_rows),
            "stage_d_primary": base.contrast(native_c + stage_d_rows, arm_d)
            if stage_d_rows else None,
            "stage_d_violations": stage_d.violation_contrast(native_c + stage_d_rows, arm_d)
            if stage_d_rows and native_c else None,
            "schedule": {stage: schedule(r1[stage], jobs[stage]) for stage in r1},
            "usage": cumulative_usage(events, recovery, decodes),
        },
        "historical": {
            "stage_a": base.summarize(validated(ledger(args.historical_root, "stage_a"))),
            "stage_b": base.summarize(validated(ledger(args.historical_root, "stage_b"))),
            "stage_c_partial": historical_stage_c(args.historical_root),
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps({k: analysis["recovery"][k] for k in
                      ("stage_c_primary", "stage_d_primary", "stage_d_violations")}, indent=2))


if __name__ == "__main__":
    main()
