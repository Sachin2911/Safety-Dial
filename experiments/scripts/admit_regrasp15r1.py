"""Financial and runtime admission for the task 15 recovery cohort.

Simulates the frozen recovery schedule (stage A engineering, then stage C in its frozen
interleaved order, then stage D) against the cumulative $10 book and the 20 aggregate
hours, with the 45-minute episode-admission margin and the five-minute early stop. The
episode-time model is fitted on the original host (validated stage B and C episodes):
seconds = startup + per_step * steps + per_request * candidate_requests. Before renting
it is used at that host's speed; at re-admission a speed factor measured from the new
host's warm engineering episodes scales it. Workload per arm (steps, candidate requests)
comes from the original host's stage B and C episodes.

    .venv/bin/python experiments/scripts/admit_regrasp15r1.py --rate 0.43 --out <receipt>
"""

import argparse
import datetime as dt
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.regrasp15_infra import FinancialBook  # noqa: E402

HISTORICAL = ROOT / "outputs/regrasp15/remote"
BOOK = ROOT / "outputs/regrasp15r1/control/finance.json"
MARGIN_S = 45 * 60
EARLY_STOP_S = 5 * 60
OVERHEAD_S = 1.5


def historical_rows():
    events = []
    for name in ("attempts_stage_a.jsonl", "attempts_stage_b.jsonl"):
        events += [json.loads(x) for x in (HISTORICAL / name).read_text().splitlines() if x]
    events += [json.loads(x) for x in (HISTORICAL / "logs/stage-c-queue.log").read_text()
               .splitlines() if x.startswith("{")]
    exits = {(e["job"], e["attempt"]): e for e in events if e["event"] == "simulator_exit"}
    rows = []
    for e in events:
        if e["event"] == "validated":
            rows.append({"stage": e["stage"], "arm": e["arm"], "job": e["job"],
                         "steps": e["outcome"]["steps"],
                         "requests": e["adapter"]["closest_passing"]
                         + e["adapter"]["least_excess"],
                         "elapsed": exits[(e["job"], e["attempt"])]["elapsed_s"]})
    return rows


def fit_model(rows):
    import numpy as np

    sample = [r for r in rows if r["stage"] in ("stage_b", "stage_c")]
    x = np.array([[1.0, r["steps"], r["requests"]] for r in sample])
    y = np.array([r["elapsed"] for r in sample])
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    return {"startup_s": float(coef[0]), "per_step_s": float(coef[1]),
            "per_request_s": float(coef[2]), "episodes": len(sample),
            "residual_sd_s": float(np.std(y - x @ coef))}


def workload(rows):
    by_arm = defaultdict(list)
    for r in rows:
        if r["stage"] in ("stage_b", "stage_c"):
            by_arm[r["arm"]].append(r)
    return {arm: {"steps": sum(r["steps"] for r in v) / len(v),
                  "requests": sum(r["requests"] for r in v) / len(v), "episodes": len(v)}
            for arm, v in by_arm.items()}


def duration(model, load, speed):
    return speed * (model["startup_s"] + model["per_step_s"] * load["steps"]
                    + model["per_request_s"] * load["requests"]) + OVERHEAD_S


def schedule(recovery):
    return [*recovery["stage_c_jobs"], *recovery["stage_d_jobs"]]


def admit(*, rate, setup_s, engineering_s, speed, replacements, book_path, recovery,
          model, loads, elapsed_s=0.0):
    snap = FinancialBook(book_path).snapshot()
    limits = FinancialBook.limits()
    money = min(limits["active_shutdown_usd"] - snap["active_estimate_usd"],
                limits["total_usd"] - snap["other_reserved_or_actual_usd"]
                - snap["active_estimate_usd"] - 0.1)
    hours = limits["aggregate_hours"] - snap["aggregate_hours"]
    # A live allocation is already inside the book; count only what remains of it.
    if snap["active_instances"]:
        nominal_s = snap["remaining_seconds"] + elapsed_s
    else:
        nominal_s = min(money / rate * 3600, hours * 3600)
    last_start = nominal_s - EARLY_STOP_S - MARGIN_S
    mean = sum(duration(model, loads[a], speed) for a in ("native", "select-0.06",
                                                          "project-0.10")) / 3
    t = setup_s + engineering_s + replacements * mean
    admitted = defaultdict(int)
    for job in schedule(recovery):
        if t > last_start:
            break
        admitted[job["stage"]] += 1
        t += duration(model, loads[job["arm"]], speed)
    planned = {"stage_c": len(recovery["stage_c_jobs"]),
               "stage_d": len(recovery["stage_d_jobs"])}
    finish = t
    return {
        "nominal_allocation_hours": nominal_s / 3600,
        "money_remaining_usd": money, "hours_remaining": hours,
        "speed_factor": speed,
        "setup_hours": setup_s / 3600, "engineering_hours": engineering_s / 3600,
        "replacement_allowance": replacements,
        "episode_seconds": {a: duration(model, loads[a], speed)
                            for a in ("native", "select-0.06", "project-0.10")},
        "admitted": dict(admitted), "planned": planned,
        "stage_c_complete": admitted["stage_c"] == planned["stage_c"],
        "stage_d_complete": admitted["stage_d"] == planned["stage_d"],
        "predicted_queue_end_hours": finish / 3600,
        "slack_hours_after_last_start": (last_start - (finish - duration(
            model, loads["project-0.10"], speed))) / 3600,
        "predicted_instance_cost_usd": rate * (finish + 600) / 3600,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--book", type=Path, default=BOOK)
    parser.add_argument("--recovery", type=Path,
                        default=ROOT / "configs/regrasp15r1/recovery.json")
    parser.add_argument("--setup-minutes", type=float, default=56.0)
    parser.add_argument("--engineering-seconds", type=float)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--replacements", type=int, default=2)
    parser.add_argument("--elapsed-seconds", type=float, default=0.0)
    parser.add_argument("--outer-book", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--label", default="pre-rent")
    args = parser.parse_args()
    rows = historical_rows()
    model, loads = fit_model(rows), workload(rows)
    recovery = json.loads(args.recovery.read_text())
    engineering = args.engineering_seconds
    if engineering is None:
        engineering = args.speed * sum(r["elapsed"] for r in rows if r["stage"] == "stage_a")
    common = dict(rate=args.rate, setup_s=args.setup_minutes * 60, engineering_s=engineering,
                  speed=args.speed, replacements=args.replacements, recovery=recovery,
                  model=model, loads=loads, elapsed_s=args.elapsed_seconds)
    result = {"label": args.label, "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "rate_usd_hour": args.rate, "model": model, "workload": loads,
              "book": admit(book_path=args.book, **common)}
    if args.outer_book:
        result["outer_bound_book"] = admit(book_path=args.outer_book, **common)
    result["admit_stage_c"] = result["book"]["stage_c_complete"]
    result["admit_full_cohort"] = (result["book"]["stage_c_complete"]
                                   and result["book"]["stage_d_complete"])
    text = json.dumps(result, indent=2, default=lambda v: math.nan)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
