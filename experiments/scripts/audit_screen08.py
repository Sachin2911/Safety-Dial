"""Audit the complete task-8 snapshot and apply its frozen confirmation rule."""

import argparse
import collections
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.screen08_preservation import sha256, verify_snapshot
from helpers.screen08_protocol import select_survivors


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(snapshot, analysis_path):
    receipt = verify_snapshot(snapshot)
    raw = snapshot / "raw"
    plan_path = ROOT / "progress/receipts/08-screen-plan.json"
    plan = json.loads(plan_path.read_text())
    decision = json.loads((raw / "stage-b-decision.json").read_text())
    analysis = json.loads(analysis_path.read_text())
    episodes = analysis["episodes"]
    require(receipt["completion"]["plan_sha256"] == sha256(plan_path), "Plan hash differs")
    require((raw / "pipeline-exit-code.txt").read_text().strip() == "0", "Pipeline failed")
    for name, expected in plan["source_sha256"].items():
        require(sha256(ROOT / name) == expected, f"Current frozen source differs: {name}")
    confirmation = [job for job in plan["stage_b_jobs"]
                    if decision["admit"] and job["arm"] in ("native", *decision["survivors"])]
    jobs = plan["engineering_jobs"] + plan["jobs"] + confirmation
    expected = {job["job"]: job for job in jobs}
    require(len({e["job"] for e in episodes}) == len(episodes) == len(expected),
            "Episode inventory is incomplete or duplicated")
    for episode in episodes:
        require(episode["job"] in expected, "Unplanned episode")
        require(all(episode.get(k) == v for k, v in expected[episode["job"]].items()),
                "Episode identity differs from frozen schedule")
        require(Path(episode["directory"]).resolve().is_relative_to(raw.resolve()),
                "Analysis did not use the terminal snapshot")
    survivors, comparisons = select_survivors([e for e in episodes if e["stage"] == "stage_a"])
    require(survivors == decision["survivors"] and comparisons == decision["comparisons"],
            "Screen selection cannot be reproduced")
    for name, expected_hash in decision["evidence_sha256"].items():
        require(sha256(raw / name) == expected_hash, "Screen selection evidence changed")

    by_stage = {}
    for stage in ("engineering", "stage_a", "stage_b"):
        planned = [j for j in jobs if j["stage"] == stage]
        path = raw / f"attempts_{stage}.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()] if planned else []
        starts = [e for e in events if e["event"] == "simulator_launch"]
        reserved = [e for e in events if e["event"] == "start_reserved"]
        validated = [e for e in events if e["event"] == "validated"]
        ended = [e for e in events if e["event"] == "simulator_exit"]
        require(len(starts) == len(reserved) == len(validated) == len(ended) == len(planned),
                "Attempt ledger is incomplete or contains replacements")
        for phase in (starts, reserved, validated, ended):
            require({e["job"] for e in phase} == {j["job"] for j in planned},
                    "Attempt phase differs from the planned job inventory")
        require(all(e["attempt"] == 1 for e in starts), "A replacement needs explicit review")
        require(all(e["rc"] == 0 and e["completed_row"] for e in ended), "Failed simulator")
        if planned:
            queue = [e for e in events if e["event"] == "queue_start"]
            require(len(queue) == 1 and queue[0]["instance"] == receipt["instance_id"]
                    and queue[0]["plan_sha256"] == sha256(plan_path),
                    "Queue host or source freeze differs")
            frozen_time = dt.datetime.fromisoformat(plan["frozen_utc"])
            require(all(dt.datetime.fromisoformat(e["utc"]) > frozen_time for e in starts),
                    "Simulator start preceded the protocol freeze")
            if stage == "stage_b":
                decision_time = dt.datetime.fromisoformat(decision["utc"])
                require(all(dt.datetime.fromisoformat(e["utc"]) > decision_time for e in starts),
                        "Confirmation start preceded its admission")
        selected = [e for e in episodes if e["stage"] == stage]
        by_stage[stage] = {
            "starts": len(starts), "validated": len(validated), "replacements": 0,
            "tasks": sorted({e["task"] for e in selected}),
            "seeds": sorted({e["seed"] for e in selected}),
            "policy_controls": sum(e["steps"] for e in selected),
            "warmup_controls": sum(e["warmup_control_steps"] for e in validated),
            "episode_queries": sum(e["queries"] for e in selected),
            "episode_wall_seconds": sum(e["elapsed_s"] for e in ended),
        }

    accounting = analysis["accounting"]
    require(accounting["reserved_starts"] == accounting["simulator_launches"]
            == accounting["validated_episodes"] == len(jobs), "Total starts disagree")
    require(accounting["policy_controls_attempted"] == accounting["policy_controls_returned"]
            == sum(s["policy_controls"] for s in by_stage.values()), "Control totals disagree")
    require(accounting["warmup_controls_attempted"] == accounting["warmup_controls_returned"]
            == 10 * len(jobs), "Warm-up totals disagree")
    require(accounting["query_attempts"] == accounting["query_returns"]
            == sum(e["queries"] for e in episodes) + accounting["model_only_query_attempts"],
            "Model-query totals disagree")
    limits = {"simulator_launches": "max_total_starts",
              "policy_controls_attempted": "policy_control_ceiling",
              "warmup_controls_attempted": "warmup_control_ceiling",
              "query_attempts": "model_query_ceiling", "learner_updates": "learner_update_ceiling"}
    require(all(accounting[k] <= plan[v] for k, v in limits.items()), "Budget exceeded")
    require(accounting["truncated_journal_lines"] == 0, "Truncated interaction journal")
    confirmed = {}
    selected = [e for e in episodes if e["stage"] == "stage_b"]
    native = {(e["task"], e["seed"]): e for e in selected if e["arm"] == "native"}
    rule = plan["survival"]
    for arm in decision["survivors"] if decision["admit"] else []:
        treatment = [e for e in selected if e["arm"] == arm]
        require(len(treatment) == len(native) == 12, "Confirmation is not twelve matched cells")
        delta = {key: sum(e[key] for e in treatment) - sum(e[key] for e in native.values())
                 for key in ("primary", "raw_violation", "counted_violation")}
        active_extra = sum(e["primary"] and e["mechanism_active"]
                           and not native[(e["task"], e["seed"])]["primary"] for e in treatment)
        passed = (delta["primary"] >= rule["extra_primary_successes"]
                  and delta["raw_violation"] <= rule["raw_violation_increase"]
                  and delta["counted_violation"] <= rule["counted_violation_increase"]
                  and active_extra >= rule["active_additional_success_cells"])
        confirmed[arm] = {"difference_counts": delta,
                          "active_additional_success_cells": active_extra,
                          "passes_predeclared_descriptive_rule": passed}
    return {"checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "instance_id": receipt["instance_id"], "snapshot_files": len(receipt["files"]),
            "snapshot_bytes": sum(v["bytes"] for v in receipt["files"].values()),
            "analysis_sha256": sha256(analysis_path), "plan_sha256": sha256(plan_path),
            "source_hashes_checked": len(plan["source_sha256"]),
            "stages": by_stage, "accounting": accounting,
            "screen_selection_reproduced": True, "confirmation": confirmed,
            "stage_episode_counts": dict(collections.Counter(e["stage"] for e in episodes)),
            "limits": "Descriptive confirmation on six development task clusters; no optimizer, "
                      "equivalence, held-out or cross-host claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "outputs/screen08/completed-snapshot")
    parser.add_argument("--analysis", type=Path, default=ROOT / "outputs/screen08/analysis/analysis.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/screen08/analysis/final-audit.json")
    args = parser.parse_args()
    result = audit(args.snapshot, args.analysis)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
