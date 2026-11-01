"""Close out the fixed one-case probe from preserved, immutable episode evidence."""

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(root):
    run = root / "run"
    status = json.loads((run / "status.json").read_text())
    if status["status"] == "running":
        raise ValueError("Study is still running")
    package = root / "source-v2"
    freeze = json.loads((package / "engineering-freeze.json").read_text())
    for name, expected in freeze["files"].items():
        if digest(package / name) != expected:
            raise ValueError("Preserved source differs from freeze: " + name)
    plan = json.loads((package / "plan.json").read_text())
    sys.path.insert(0, str(package))
    from analyze_jar_grasp_probe import analyze

    scored, engineering, failed, accounting = [], [], [], []
    for job in sorted(run.glob("job*")):
        specification = json.loads((job / "job.json").read_text())
        counters = job / "counters.json"
        seed = job / "preseed-query.json"
        record = {"name": job.name, "job": specification,
                  "counters": json.loads(counters.read_text()) if counters.exists() else {},
                  "preseed": json.loads(seed.read_text()) if seed.exists() else {}}
        accounting.append(record)
        validation_path = job / "validation.json"
        if not validation_path.exists():
            original_rows = list((job / "eval").rglob("results.jsonl"))
            record["original_result"] = (
                json.loads(original_rows[0].read_text().strip()) if original_rows else None
            )
            failed.append(record)
            continue
        original_hash = digest(validation_path)
        result = analyze(job, plan)
        if digest(validation_path) != original_hash:
            raise ValueError("Workstation replay did not reproduce the remote validation exactly")
        result["name"] = job.name
        (scored if specification["stage"] == "scored" else engineering).append(result)
    expected = plan["schedule"]
    actual = [{key: result["job"][key] for key in ("task", "policy_seed", "arm")} for result in scored]
    complete = status["status"] == "completed" and actual == expected and len(scored) == 6
    metrics = ("primary", "benchmark_safe_completion", "requested_completion", "raw_violation",
               "counted_violation", "unfinished", "engaged", "post_trigger_sustained_grasp")
    arms = {}
    for arm in plan["arms"]:
        rows = [result for result in scored if result["job"]["arm"] == arm]
        arms[arm] = {"n": len(rows), **{key: sum(row[key] for row in rows) for key in metrics},
                     "episodes_with_trigger": sum(bool(row["trigger_steps"]) for row in rows),
                     "overridden_controls": sum(row["overridden_controls"] for row in rows),
                     "policy_controls": sum(row["controls"] for row in rows)}
    native, probe = arms["native"], arms["pose_triggered_grasp"]
    active_successes = sum(row["primary"] and row["post_trigger_sustained_grasp"]
                           for row in scored if row["job"]["arm"] == "pose_triggered_grasp")
    signal = bool(complete and probe["primary"] - native["primary"] >= 2
                  and probe["raw_violation"] <= native["raw_violation"]
                  and probe["engaged"] == 3 and active_successes >= 2)
    totals = {}
    for key in ("environment_step_attempts", "environment_step_returns", "reset_attempts",
                "reset_returns", "simulator_step_attempts", "simulator_step_returns",
                "step_physics_attempts", "step_physics_returns", "physics_callbacks",
                "warmup_attempts", "warmup_returns", "query_attempts", "query_returns"):
        totals[key] = sum(record["counters"].get(key, 0) for record in accounting)
    totals["synthetic_preseed_query_attempts"] = sum(record["preseed"].get("attempted", 0)
                                                    for record in accounting)
    totals["synthetic_preseed_query_returns"] = sum(record["preseed"].get("completed", 0)
                                                   for record in accounting)
    totals["all_model_query_attempts"] = totals["query_attempts"] + totals["synthetic_preseed_query_attempts"]
    totals["all_model_query_returns"] = totals["query_returns"] + totals["synthetic_preseed_query_returns"]
    totals["valid_policy_controls"] = sum(row["controls"] for row in scored + engineering)
    totals["reserved_starts"] = status["reserved"]
    totals["valid_scored_episodes"] = len(scored)
    totals["valid_engineering_episodes"] = len(engineering)
    totals["failed_or_unvalidated_starts"] = len(failed)
    totals["physics_before_counter_installation"] = None
    totals["physics_coverage_note"] = "Callback counts cover instrumented work. Initial simulator bootstrap physics is unknown, not zero."
    caps = plan["limits"]
    if not (totals["reserved_starts"] <= caps["max_total_starts"]
            and totals["valid_policy_controls"] <= caps["max_policy_controls_total"]
            and totals["warmup_attempts"] <= caps["max_explicit_warmup_controls_total"]
            and totals["all_model_query_attempts"] <= caps["max_model_query_attempts_total"]):
        raise ValueError("Frozen interaction limit exceeded")
    # Freeze precedes every eligible scored reservation, not merely the first returned outcome.
    ledger = [json.loads(line) for line in (run / "ledger.jsonl").read_text().splitlines()]
    frozen = [row for row in ledger if row["event"] == "scored_stage_frozen"]
    eligible = [row for row in ledger if row["event"] == "start_reserved"
                and row["specification"]["stage"] == "scored"]
    if eligible and not (len(frozen) == 1 and frozen[0]["time"] < min(row["time"] for row in eligible)):
        raise ValueError("Scored freeze did not precede eligible starts")
    result = {
        "schema": "jar-grasp-probe-result-v1", "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "completed_fixed_comparison": complete, "status": status["status"],
        "arms": arms, "predeclared_development_signal": signal,
        "primary_count_difference": probe["primary"] - native["primary"],
        "accounting": totals, "scored_rows": scored, "engineering_rows": engineering,
        "failed_starts": failed, "wall_seconds": status["wall_seconds"],
        "engineering_configuration_correction": status.get("configuration_repair"),
        "source_freeze_sha256": digest(package / "engineering-freeze.json"),
        "scored_freeze_sha256": digest(run / "scored-freeze.json") if frozen else None,
        "independent_workstation_replay": "Exact replay using the same frozen validator, not an independent implementation",
        "limits": ["One selected development task and three seeds per arm",
                   "Rule and grasp reference informed by two historical development trajectories",
                   "Privileged object-pose, proposition and grasp-state inputs",
                   "Matched sampling seeds do not make the simulator bit-reproducible",
                   "No learning, evolutionary or held-out evaluation ran",
                   "One zero-control startup failure retained; timing corrected in the remaining predeclared engineering slot"],
    }
    output = root / "analysis"
    output.mkdir(exist_ok=True)
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    fields = ["name", "arm", "policy_seed", "controls", *metrics, "overridden_controls"]
    with (output / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for row in scored:
            writer.writerow({"name": row["name"], "arm": row["job"]["arm"],
                             "policy_seed": row["job"]["policy_seed"],
                             **{key: row[key] for key in fields if key in row}})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = summarize(args.root)
    print(json.dumps({key: result[key] for key in ("completed_fixed_comparison", "arms",
                                                  "predeclared_development_signal", "accounting")}))
