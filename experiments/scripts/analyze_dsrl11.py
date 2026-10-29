"""Offline task-11 outcomes, the frozen continuation screen, contrasts and accounting.

Every admitted episode is re-verified from its hashed artifacts and the native result
before it is counted. Task outcomes, both reward returns and critic diagnostics are
reported separately; a larger shaped return or a clipped critic is never evidence of
better safe completion.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sqlite3
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments/scripts"))
from analyze_dsrl07 import digest, read_events, read_json, wilson  # noqa: E402
from helpers.dsrl11_protocol import (  # noqa: E402
    CONDITIONS,
    EVALUATION_ROOTS,
    PRIMARY_CONDITION,
    Dsrl11Ledger,
    check_return_identity,
)

METRICS = (
    "strict_safe_requested",
    "safe_requested",
    "requested_success",
    "success",
    "raw_violation",
    "counted_violation",
    "unfinished",
    "engaged",
    "safety_evaluated",
    "ever_grasped",
    "closed_once",
    "left_support",
    "lifted_while_grasped",
    "order_failed",
)
ARMS = ("native", *CONDITIONS)


def resolve(path, run_dir):
    path = Path(path)
    marker = "/outputs/dsrl11/"
    if marker in str(path):
        local = run_dir / str(path).split(marker, 1)[1]
        if local.is_file():
            return local
    if not path.is_absolute():
        path = run_dir / path
    if not path.is_file():
        raise ValueError(f"Recorded artifact is missing: {path}")
    return path


def first_index(values):
    hits = np.flatnonzero(values)
    return int(hits[0]) if hits.size else None


def episode(directory, run_dir, job):
    validation = read_json(directory / "validation.json")
    if validation is None or validation.get("valid") is not True:
        raise ValueError("Episode lacks a completed valid recording receipt")
    if validation["job_id"] != job["job_id"]:
        raise ValueError("Validation names a different frozen job")
    artifacts = validation["source_files"]
    for source in [*artifacts.values(), *validation["query_sources"]]:
        if digest(resolve(source["path"], run_dir)) != source["sha256"]:
            raise ValueError("Recorded source hash differs from its validation receipt")
    rows = read_events(resolve(artifacts["result"]["path"], run_dir))
    if len(rows) != 1:
        raise ValueError("One completed native result is required")
    row, labels = rows[0], validation["labels"]
    if (
        row["seed"] != job["seed"]
        or row["episode_seed"] != job["episode_seed"]
        or row["scene_name"] != f"task_{job['task']:04d}/base"
    ):
        raise ValueError("Native result differs from the frozen task and seed")
    expected = {
        "success": bool(row["success"]),
        "raw_violation": bool(row["ltl_violated"]),
        "counted_violation": bool(row["counted_violation"]),
        "unfinished": not row["success"],
        "ever_grasped": bool(row["ever_grasped"]),
    }
    expected["requested_success"] = bool(
        expected["success"] and labels["closed_once"] and not labels["order_failed"]
    )
    expected["strict_safe_requested"] = bool(
        expected["requested_success"] and not expected["raw_violation"]
    )
    expected["safe_requested"] = bool(
        expected["requested_success"] and not expected["counted_violation"]
    )
    if any(labels.get(key) is not value for key, value in expected.items()):
        raise ValueError("Scorer labels disagree with native monitor/result labels")
    returns = {}
    for name, transitions in validation["transitions"].items():
        identity = check_return_identity(transitions)
        if not math.isclose(
            identity["discounted_shaped_return"],
            validation["returns"][name]["discounted_shaped_return"],
            abs_tol=1e-6,
        ):
            raise ValueError("Saved return reconciliation differs from recomputation")
        if bool(transitions[-1]["base_reward"]) != labels["strict_safe_requested"]:
            raise ValueError("Terminal reward differs from the strict primary outcome")
        if sum(item["duration"] for item in transitions) != row["steps"]:
            raise ValueError("Transition durations differ from executed controls")
        returns[name] = identity
    with np.load(resolve(artifacts["recording"]["path"], run_dir), allow_pickle=False) as arrays:
        meta = json.loads(str(arrays["meta"]))
        support = arrays["ap"][:, meta["ap_names"].index("jar_on_support")]
        grasping = arrays["grasping"] == 1
        lifted = bool(np.any(grasping & ~support))
        first_grasp = first_index(grasping)
    residuals = []
    for query in validation["query_sources"]:
        with np.load(resolve(query["path"], run_dir), allow_pickle=False) as arrays:
            residuals.append(arrays["residual_u"].copy())
    residual = np.asarray(residuals)
    return {
        **{k: job.get(k) for k in ("job_id", "stage", "task", "seed", "lane", "condition")},
        "arm": job.get("arm"),
        "phase": job.get("phase"),
        "actor_version": job.get("actor_version", -1),
        **{k: labels[k] for k in labels if k != "valid"},
        "engaged": row["first_contact_step"] is not None,
        "safety_evaluated": bool(row["safety_evaluated"]),
        "left_support": labels["first_departure_step"] is not None,
        "lifted_while_grasped": lifted,
        "milestones": {
            "first_contact_step": row["first_contact_step"],
            "first_grasp_step": first_grasp,
            "first_closed_step": labels["first_closed_step"],
            "first_departure_step": labels["first_departure_step"],
            "success_step": row.get("success_step"),
        },
        "policy_controls": row["steps"],
        "policy_queries": validation["queries"],
        "returns": {
            name: {
                "discounted_shaped": value["discounted_shaped_return"],
                "discounted_base": value["discounted_base_return"],
            }
            for name, value in returns.items()
        },
        "reward_counts": validation["reward_counts"],
        "geometry": validation["geometry"],
        "residual": {
            "source": validation["residual"]["source"],
            "maximum_absolute": float(np.max(np.abs(residual))),
            "mean_absolute": float(np.mean(np.abs(residual))),
            "fraction_abs_ge_0_99": float(np.mean(np.abs(residual) >= 0.99)),
        },
        "validation_sha256": digest(directory / "validation.json"),
    }


def totals(rows):
    count = len(rows)
    counts = {key: sum(bool(row[key]) for row in rows) for key in METRICS}
    return {
        "denominator": count,
        "counts": counts,
        "rates": {key: value / count if count else None for key, value in counts.items()},
        "nominal_wilson_95": {key: wilson(value, count) for key, value in counts.items()},
        "policy_controls": sum(row["policy_controls"] for row in rows),
        "policy_queries": sum(row["policy_queries"] for row in rows),
        "median_first_contact_step": _median(rows, "first_contact_step"),
        "median_first_grasp_step": _median(rows, "first_grasp_step"),
        "median_success_step": _median(rows, "success_step"),
        "mean_maximum_lift": float(np.mean([r["geometry"]["maximum_lift"] for r in rows]))
        if rows
        else None,
        "median_minimum_reach_m": float(
            np.median([r["geometry"]["minimum_reach_distance_m"] for r in rows])
        )
        if rows
        else None,
    }


def _median(rows, key):
    values = [r["milestones"][key] for r in rows if r["milestones"][key] is not None]
    return float(np.median(values)) if values else None


def screen(groups):
    """The frozen practical continuation screen for the primary condition."""
    expected = len(EVALUATION_ROOTS)
    complete = all(len(groups[arm]) == expected for arm in ARMS)
    count = {arm: totals(groups[arm])["counts"] for arm in ARMS}
    primary, native, plain = count[PRIMARY_CONDITION], count["native"], count["r0_none"]
    rules = {
        "two_more_strict_safe_than_native": primary["strict_safe_requested"]
        >= native["strict_safe_requested"] + 2,
        "two_more_strict_safe_than_r0_none": primary["strict_safe_requested"]
        >= plain["strict_safe_requested"] + 2,
        "raw_violations_not_above_native": primary["raw_violation"] <= native["raw_violation"],
        "counted_violations_not_above_native": primary["counted_violation"]
        <= native["counted_violation"],
        "engaged_at_least_7_of_8": primary["engaged"] >= 7,
        "engaged_at_most_one_below_native": primary["engaged"] >= native["engaged"] - 1,
    }
    ceiling = native["strict_safe_requested"] + 2 > expected
    return {
        "primary_condition": PRIMARY_CONDITION,
        "complete": complete,
        "rules": rules,
        "ceiling_limited": ceiling,
        "passed": bool(complete and not ceiling and all(rules.values())),
        "interpretation": "development threshold, not a significance test or risk guarantee",
    }


def contrasts(groups):
    """Paired by evaluation root: shared native draws per root across arms."""
    pairs = {
        "r1_minus_r0_without_prior": ("r1_none", "r0_none"),
        "r1_minus_r0_with_prior": ("r1_prior", "r0_prior"),
        "prior_minus_none_r0": ("r0_prior", "r0_none"),
        "prior_minus_none_r1": ("r1_prior", "r1_none"),
        "primary_minus_native": (PRIMARY_CONDITION, "native"),
    }
    result = {}
    for name, (left, right) in pairs.items():
        a = {row["seed"]: row for row in groups[left]}
        b = {row["seed"]: row for row in groups[right]}
        roots = sorted(set(a) & set(b))
        result[name] = {
            "roots": len(roots),
            **{
                key: {
                    "difference": sum(bool(a[r][key]) for r in roots)
                    - sum(bool(b[r][key]) for r in roots),
                    "left_only": sum(bool(a[r][key]) and not b[r][key] for r in roots),
                    "right_only": sum(bool(b[r][key]) and not a[r][key] for r in roots),
                }
                for key in ("strict_safe_requested", "success", "raw_violation", "ever_grasped")
            },
        }
    return result


def critic_report(run_dir):
    ledger = read_events(run_dir / "learner-learning/learner-ledger.jsonl")
    report = {name: {"calibrations": [], "rounds": []} for name in CONDITIONS}
    paused = None
    for event in ledger:
        name = event.get("condition")
        if event["event"] == "calibration" and name in report:
            report[name]["calibrations"].append(
                {
                    "updates": event["updates"],
                    "scheduled": event["scheduled"],
                    "label": event["label"],
                    "gate_fractions": event["gate_fractions"],
                    "reduced_mean_recorded": event["recorded"]["reduced"].get("mean"),
                    "reduced_mean_sampled": event["sampled"]["reduced"].get("mean"),
                    "entropy": event["policy"].get("entropy_estimate"),
                }
            )
        elif event["event"] == "train_completed" and name in report:
            report[name]["rounds"].append(
                {
                    k: event.get(k)
                    for k in ("round_id", "updates", "condition_updates", "elapsed_seconds")
                }
            )
        elif event["event"] == "comparison_paused":
            paused = {k: event.get(k) for k in ("condition", "reason", "updates", "utc")}
    return {"conditions": report, "paused": paused}


def resources(run_dir, plan):
    ledger = Dsrl11Ledger(run_dir / "resources.jsonl", plan["jobs"])
    summary = ledger.summary()
    queries = {}
    database = run_dir / "queries.sqlite"
    if database.exists():
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as db:
            queries = dict(
                db.execute("SELECT purpose, COUNT(*) FROM queries GROUP BY purpose").fetchall()
            )
    actor_calls = sum(
        sum(e["event"] == "actor_request" for e in read_events(path))
        for path in run_dir.rglob("*-ledger.jsonl")
    )
    return {
        "usage": summary["usage"],
        "caps": dict(ledger.caps),
        "queries_by_purpose": queries,
        "actor_requests": actor_calls,
        "prior_engineering_updates": plan.get("prior_engineering_updates", 0),
    }


def analyze(run_dir, plan_path):
    plan = json.loads(Path(plan_path).read_text())
    jobs = {job["job_id"]: job for group in plan["jobs"].values() for job in group}
    queue = read_json(run_dir / "queue-state.json", {"episodes": {}})
    rows, invalid = [], []
    for job_id, entry in sorted(queue["episodes"].items()):
        directory = run_dir / str(entry["directory"]).split("/outputs/dsrl11/", 1)[-1]
        if entry.get("valid") is not True:
            invalid.append({k: entry.get(k) for k in ("job_id", "attempt", "error_type", "error")})
            continue
        rows.append(episode(directory, run_dir, jobs[job_id]))
    evaluation = [r for r in rows if r["stage"] == "evaluation"]
    groups = {arm: [r for r in evaluation if r["arm"] == arm] for arm in ARMS}
    training = {
        name: {
            phase: totals(
                [
                    r
                    for r in rows
                    if r["stage"] == "online" and r["condition"] == name and r["phase"] == phase
                ]
            )
            for phase in ("warmup", "batch_a", "batch_b")
        }
        for name in CONDITIONS
    }
    return {
        "schema": "dsrl11-analysis-v1",
        "plan_sha256": digest(plan_path),
        "episodes_analyzed": len(rows),
        "invalid_attempts": invalid,
        "evaluation": {arm: totals(groups[arm]) for arm in ARMS},
        "continuation_screen": screen(groups),
        "contrasts": contrasts(groups),
        "bank": totals([r for r in rows if r["stage"] == "bank"]),
        "engineering": totals([r for r in rows if r["stage"] == "engineering"]),
        "training_descriptive": training,
        "returns_descriptive": {
            arm: {
                name: float(np.mean([r["returns"][name]["discounted_shaped"] for r in groups[arm]]))
                if groups[arm]
                else None
                for name in ("r0", "r1")
            }
            for arm in ARMS
        },
        "critic": critic_report(run_dir),
        "numerical": read_json(run_dir / "stage-a/numerical-summary.json"),
        "stage_c": read_json(run_dir / "stage-c-freeze.json"),
        "resources": resources(run_dir, plan),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/dsrl11/remote")
    parser.add_argument(
        "--plan", type=Path, default=ROOT / "progress/receipts/11-dsrl-stage-a.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/dsrl11/analysis.json")
    args = parser.parse_args()
    report = analyze(args.run_dir.resolve(), args.plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "episodes": report["episodes_analyzed"],
                "screen": report["continuation_screen"]["passed"],
                "strict_safe": {
                    arm: report["evaluation"][arm]["counts"]["strict_safe_requested"]
                    for arm in ARMS
                },
            }
        )
    )


if __name__ == "__main__":
    main()
