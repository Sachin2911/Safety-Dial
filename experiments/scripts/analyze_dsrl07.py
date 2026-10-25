"""Offline task-7 labels, budget reconciliation and the frozen exploratory screen."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_protocol import (
    DEVELOPMENT_TASKS,
    EVALUATION_ARMS,
    ProtocolLedger,
    check_return_identity,
)

from helpers.dsrl07_protocol import frozen_jobs

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


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def read_events(path):
    if not path.exists():
        return []
    text = path.read_text()
    if text and not text.endswith("\n"):
        raise ValueError(f"Incomplete append-only journal: {path}")
    return [json.loads(line) for line in text.splitlines()]


def resolve(path, run_dir):
    path = Path(path)
    marker = "/outputs/dsrl07/"
    if marker in str(path):
        local = run_dir / str(path).split(marker, 1)[1]
        if local.is_file():
            return local
    if not path.is_absolute():
        path = run_dir / path
    if not path.is_file():
        raise ValueError(f"Recorded artifact is missing: {path}")
    return path


def wilson(successes, count):
    if not count:
        return None
    z = 1.959963984540054
    probability = successes / count
    denominator = 1 + z * z / count
    center = (probability + z * z / (2 * count)) / denominator
    half = z * math.sqrt(probability * (1 - probability) / count + z * z / (4 * count**2))
    return [max(0, center - half / denominator), min(1, center + half / denominator)]


def totals(episodes, expected=None):
    counts = {key: sum(bool(row[key]) for row in episodes) for key in METRICS}
    classes = Counter(
        ("unsafe_" if row["counted_violation"] else "safe_")
        + ("success" if row["success"] else "failure")
        for row in episodes
    )
    counts.update(
        {
            name: classes[name]
            for name in ("safe_success", "unsafe_success", "safe_failure", "unsafe_failure")
        }
    )
    denominator = len(episodes)
    return {
        "denominator": denominator,
        "expected": expected,
        "counts": counts,
        "rates": {
            key: value / denominator if denominator else None for key, value in counts.items()
        },
        "nominal_wilson_95": {key: wilson(value, denominator) for key, value in counts.items()},
        "unscored_safety": denominator - counts["safety_evaluated"],
        "policy_controls": sum(row["policy_controls"] for row in episodes),
        "policy_queries": sum(row["policy_queries"] for row in episodes),
        "warmup_controls": sum(row["warmup_controls"] for row in episodes),
        "zero_raw_violation_rough_upper_95": min(1.0, 3 / denominator)
        if denominator and counts["raw_violation"] == 0
        else None,
    }


def episode(directory, run_dir, jobs):
    manifest = read_json(directory / "job.json")
    job_id = manifest["job_id"]
    if job_id not in jobs or any(manifest.get(key) != value for key, value in jobs[job_id].items()):
        raise ValueError("Episode differs from the frozen task/seed/arm panel")
    validation = read_json(directory / "validation.json")
    if validation is None or validation.get("valid") is not True:
        raise ValueError("Episode lacks a completed valid recording receipt")
    if validation.get("job_id") != job_id:
        raise ValueError("Validation names a different frozen job")
    artifacts = validation["source_files"]
    for source in [*artifacts.values(), *validation["source_queries"]]:
        if digest(resolve(source["path"], run_dir)) != source["sha256"]:
            raise ValueError("Recorded source hash differs from its validation receipt")
    rows = read_events(resolve(artifacts["result"]["path"], run_dir))
    if len(rows) != 1:
        raise ValueError("One completed native result is required")
    row = rows[0]
    labels = validation["labels"]
    job = jobs[job_id]
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
    transitions = validation["transitions"]
    identity = check_return_identity(transitions)
    if any(
        not math.isclose(identity[key], value, abs_tol=1e-5)
        for key, value in validation["return_identity"].items()
    ):
        raise ValueError("Saved return reconciliation differs from recomputation")
    if bool(transitions[-1]["base_reward"]) != labels["strict_safe_requested"]:
        raise ValueError("Terminal reward differs from the strict primary outcome")
    if sum(item["duration"] for item in transitions) != row["steps"]:
        raise ValueError("Transition durations differ from executed controls")
    with np.load(resolve(artifacts["recording"]["path"], run_dir), allow_pickle=False) as arrays:
        meta = json.loads(str(arrays["meta"]))
        support = arrays["ap"][:, meta["ap_names"].index("jar_on_support")]
        lifted = bool(np.any((arrays["grasping"] == 1) & ~support))
    noise = []
    if validation["mode"] != "native":
        for query in validation["source_queries"]:
            with np.load(resolve(query["path"], run_dir), allow_pickle=False) as arrays:
                noise.append(arrays["noise"][0].copy())
    latent = np.asarray(noise)
    phis = [t["phi_before"] for t in transitions] + [transitions[-1]["phi_after_raw"]]
    return {
        **job,
        **labels,
        "attempt": manifest["attempt"],
        "validation_sha256": digest(directory / "validation.json"),
        "engaged": row["first_contact_step"] is not None,
        "first_contact_step": row["first_contact_step"],
        "safety_evaluated": bool(row["safety_evaluated"]),
        "left_support": labels["first_departure_step"] is not None,
        "lifted_while_grasped": lifted,
        "policy_controls": row["steps"],
        "policy_queries": validation["queries"],
        "warmup_controls": validation["warmup_controls"],
        "mode": validation["mode"],
        "actor_version": validation["actor_version"],
        "cost": labels["cost"],
        "base_reward_sum": sum(t["base_reward"] for t in transitions),
        "shaping_reward_sum": sum(t["shaping_reward"] for t in transitions),
        "undiscounted_reward_sum": sum(t["reward"] for t in transitions),
        "potential_min_mean_max": [min(phis), float(np.mean(phis)), max(phis)],
        "latent": {
            "vectors": len(latent),
            "mean": float(latent.mean()),
            "std": float(latent.std()),
            "maximum_absolute": float(np.max(np.abs(latent))),
            "fraction_abs_ge_0_99": float(np.mean(np.abs(latent) >= 0.99)),
            "per_coordinate_mean": latent.mean(axis=0).tolist(),
            "per_coordinate_std": latent.std(axis=0).tolist(),
        }
        if len(latent)
        else None,
        **identity,
    }


def frozen_screen(episodes, training_complete):
    groups = {arm: [row for row in episodes if row["arm"] == arm] for arm in EVALUATION_ARMS}
    panel = {(task, seed) for task in DEVELOPMENT_TASKS for seed in (72000, 72001)}
    complete = training_complete and all(
        len(rows) == 12 and {(row["task"], row["seed"]) for row in rows} == panel
        for rows in groups.values()
    )
    if not complete:
        return {
            "assessable": False,
            "passed": None,
            "reason": "Final training or evaluation panel incomplete",
        }
    counts = {arm: totals(rows)["counts"] for arm, rows in groups.items()}
    dsrl, native = counts["dsrl"], counts["native"]
    gains = [
        task
        for task in DEVELOPMENT_TASKS
        if sum(row["strict_safe_requested"] for row in groups["dsrl"] if row["task"] == task)
        > sum(row["strict_safe_requested"] for row in groups["native"] if row["task"] == task)
    ]
    criteria = {
        "two_more_primary_than_native": dsrl["strict_safe_requested"]
        >= native["strict_safe_requested"] + 2,
        "more_primary_than_fixed_zero": dsrl["strict_safe_requested"]
        > counts["fixed_zero"]["strict_safe_requested"],
        "more_primary_than_random": dsrl["strict_safe_requested"]
        > counts["random"]["strict_safe_requested"],
        "gains_on_at_least_two_tasks": len(gains) >= 2,
        "no_more_raw_violations_than_native": dsrl["raw_violation"] <= native["raw_violation"],
        "no_more_counted_violations_than_native": dsrl["counted_violation"]
        <= native["counted_violation"],
        "engagement_at_least_ten": dsrl["engaged"] >= 10,
        "engagement_no_more_than_one_below_native": dsrl["engaged"] >= native["engaged"] - 1,
    }
    return {
        "assessable": True,
        "passed": all(criteria.values()),
        "criteria": criteria,
        "gaining_tasks": gains,
    }


def paired(episodes):
    cells = {(row["task"], row["seed"], row["arm"]): row for row in episodes}
    differences = []
    for task in DEVELOPMENT_TASKS:
        for seed in (72000, 72001):
            native = cells.get((task, seed, "native"))
            if native is None:
                continue
            for arm in EVALUATION_ARMS[1:]:
                treated = cells.get((task, seed, arm))
                if treated is not None:
                    differences.append(
                        {
                            "task": task,
                            "seed": seed,
                            "arm": arm,
                            **{key: int(treated[key]) - int(native[key]) for key in METRICS},
                        }
                    )
    return differences


def resource_report(run_dir, prior_updates, prior_actor_forwards=0):
    ledger = (
        ProtocolLedger(run_dir / "resources.jsonl").summary()
        if (run_dir / "resources.jsonl").exists()
        else None
    )
    controls = Counter()
    for path in (run_dir / "episodes").rglob("*.journal.jsonl"):
        controls.update(event["event"] for event in read_events(path))
    events = read_events(run_dir / "learner/learner-ledger.jsonl")
    updates = Counter()
    for event in events:
        if (
            event["event"] == "update_reserved"
            and event.get("origin") != "workstation_cpu_preflight"
        ):
            updates[event["phase"]] += 1
    updates["engineering"] += prior_updates
    query_groups = []
    query_count = None
    database = run_dir / "queries.sqlite"
    if database.exists():
        with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as connection:
            query_count = connection.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
            query_groups = [
                {"purpose": purpose, "status": status, "count": count}
                for purpose, status, count in connection.execute(
                    "SELECT purpose,status,COUNT(*) FROM queries GROUP BY purpose,status"
                )
            ]
    observed = {
        "policy_controls": controls["before_action"],
        "warmup_controls": controls["warmup_attempt"],
        "engineering_updates": updates["engineering"],
        "learning_updates": updates["training"],
        "learner_updates": sum(updates.values()),
    }
    if query_count is not None:
        observed["model_queries"] = query_count
    mismatches = {
        key: {"source_observed": value, "protocol_ledger": ledger["usage"][key]}
        for key, value in observed.items()
        if ledger is not None and value != ledger["usage"][key]
    }
    remote_actor_forwards = sum(
        event["event"] == "actor_request"
        for path in [run_dir / "learner/learner-ledger.jsonl", *run_dir.rglob("actor-ledger.jsonl")]
        for event in read_events(path)
    )
    return {
        "protocol_ledger": ledger,
        "source_observed": observed,
        "source_protocol_mismatches": mismatches,
        "policy_controls_returned": controls["applied"],
        "warmup_controls_returned": controls["warmup_returned"],
        "query_groups": query_groups,
        "remote_actor_forward_requests": remote_actor_forwards,
        "workstation_actor_forward_requests": prior_actor_forwards,
        "actor_forward_requests": prior_actor_forwards + remote_actor_forwards,
        "learner_metrics": [event for event in events if event["event"] == "training_metrics"],
        "learner_rounds": [event for event in events if event["event"] == "train_completed"],
    }


def final_checkpoint_report(run_dir, queue, evaluation):
    checkpoint = queue.get("training_complete", {}).get("checkpoint") or queue.get(
        "last_checkpoint"
    )
    if not checkpoint:
        return {"verified": False, "reason": "No final learner checkpoint recorded"}
    try:
        manifest_path = resolve(Path(checkpoint) / "manifest.json", run_dir)
        if manifest_path.parent.name != "episode24":
            raise ValueError("The scheduled final checkpoint must be episode24")
        manifest = read_json(manifest_path)
        if set(manifest) != {"learner.msgpack", "replay.npz", "state.json"}:
            raise ValueError("Final checkpoint is missing full learner or replay state")
        for name, expected in manifest.items():
            if Path(name).name != name or digest(manifest_path.parent / name) != expected:
                raise ValueError("Final checkpoint manifest hash mismatch")
        state = read_json(manifest_path.parent / "state.json")
        if len(state["episodes"]) != 24 or {row["job_id"] for row in state["episodes"]} != {
            f"T{i:02}" for i in range(24)
        }:
            raise ValueError("Final checkpoint did not incorporate all 24 frozen training jobs")
        if (
            state["schema"] != "dsrl06-learner-checkpoint-v1"
            or state["phase"] != "training"
            or state["pending_transitions"] != 0
            or state["actor_version"] < 1
        ):
            raise ValueError("Final checkpoint is not a completed published training learner")
        version = state["actor_version"]
        if any(row["actor_version"] != version for row in evaluation if row["arm"] == "dsrl"):
            raise ValueError("DSRL evaluation used a different published actor version")
        return {
            "verified": True,
            "actor_version": version,
            "manifest_sha256": digest(manifest_path),
        }
    except (ValueError, KeyError, OSError) as error:
        return {"verified": False, "reason": str(error)}


def analyze(run_dir, stage_a=None):
    run_dir = Path(run_dir).resolve()
    plans = frozen_jobs()
    jobs = {job["job_id"]: job for group in plans.values() for job in group}
    episodes, failures = [], []
    for directory in sorted((run_dir / "episodes").glob("*")):
        if not (directory / "job.json").exists():
            continue
        try:
            episodes.append(episode(directory, run_dir, jobs))
        except (ValueError, KeyError, OSError) as error:
            failures.append(
                {
                    "directory": str(directory),
                    "job_id": read_json(directory / "job.json", {}).get("job_id"),
                    "had_validation_receipt": (directory / "validation.json").exists(),
                    "error": str(error),
                }
            )
    identities = [row["job_id"] for row in episodes]
    if len(identities) != len(set(identities)):
        raise ValueError("Multiple valid attempts for a frozen job cannot enter analysis")
    training = sorted(
        (row for row in episodes if row["stage"] == "training"), key=lambda row: row["job_id"]
    )
    evaluation = [row for row in episodes if row["stage"] == "evaluation"]
    engineering = [row for row in episodes if row["stage"] == "engineering"]
    stage_a = read_json(Path(stage_a), {}) if stage_a is not None else {}
    engineering_summary = read_json(run_dir / "stage-a-summary.json", {})
    numerical = read_json(run_dir / "numerical-summary.json", {})
    integration_pass = bool(
        numerical.get("admit_live_integration")
        and engineering_summary.get("admit")
        and engineering_summary.get("source_parity_valid")
        and engineering_summary.get("cpu_actor_parity", {}).get("valid")
        and {f"E{i}" for i in range(6)} <= set(identities)
    )
    resources = resource_report(
        run_dir,
        stage_a.get("prior_engineering_updates", 0),
        0,
    )
    minimums = {
        "policy_controls": sum(row["policy_controls"] for row in episodes),
        "warmup_controls": sum(row["warmup_controls"] for row in episodes),
        "model_queries": sum(row["policy_queries"] for row in episodes),
    }
    resources["valid_episode_minimum_mismatches"] = {
        key: {
            "valid_episode_minimum": value,
            "source_observed": resources["source_observed"].get(key),
        }
        for key, value in minimums.items()
        if value > 0 and resources["source_observed"].get(key, -1) < value
    }
    queue = read_json(run_dir / "queue-state.json", {})
    unresolved_invalid = [
        job_id for job_id, value in queue.get("episodes", {}).items() if value.get("valid") is False
    ]
    curves = []
    for cycle in range(4):
        rows = [row for row in training if row["cycle"] == cycle]
        if rows:
            curves.append(
                {
                    "cycle": cycle,
                    "jobs": [row["job_id"] for row in rows],
                    "complete_six_task_cycle": len(rows) == 6,
                    "gaussian_warmup_episodes": sum(row["mode"] == "gaussian" for row in rows),
                    "outcomes": totals(rows, 6),
                    **{
                        key: float(np.mean([row[key] for row in rows]))
                        for key in (
                            "discounted_shaped_return",
                            "discounted_base_return",
                            "corrected_discounted_shaped_return",
                            "base_reward_sum",
                            "shaping_reward_sum",
                            "undiscounted_reward_sum",
                            "cost",
                        )
                    },
                }
            )
    final_checkpoint = final_checkpoint_report(run_dir, queue, evaluation)
    screen = frozen_screen(evaluation, len(training) == 24 and final_checkpoint["verified"])
    invalid_receipts = [
        row for row in failures if row["had_validation_receipt"] and row["job_id"] not in identities
    ]
    invalid = bool(
        unresolved_invalid
        or invalid_receipts
        or resources["source_protocol_mismatches"]
        or (episodes and resources["protocol_ledger"] is None)
        or resources["valid_episode_minimum_mismatches"]
        or (evaluation and not final_checkpoint["verified"])
        or ((training or evaluation) and not integration_pass)
    )
    if invalid:
        screen = {
            "assessable": False,
            "passed": None,
            "reason": "Integration or accounting invalid",
        }
    if invalid:
        outcome = "invalid_integration"
    elif numerical and not numerical.get("admit_live_integration"):
        outcome = "numerical_integration_failed"
    elif integration_pass and not training and not evaluation:
        outcome = "engineering_ready_for_learning"
    elif not engineering_summary and not training and not evaluation:
        outcome = "engineering_incomplete"
    elif not screen["assessable"]:
        outcome = "incomplete_budget"
    elif screen["passed"]:
        outcome = "screen_passed_replication_required"
    else:
        outcome = (
            "valid_but_no_observed_gain"
            if all(
                totals([row for row in evaluation if row["arm"] == "dsrl"])["counts"][
                    "strict_safe_requested"
                ]
                <= totals([row for row in evaluation if row["arm"] == arm])["counts"][
                    "strict_safe_requested"
                ]
                for arm in ("native",)
            )
            else "inconclusive_learning"
        )
    return {
        "schema": "dsrl07-analysis-v1",
        "analysis_source_sha256": digest(Path(__file__)),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "outcome": outcome,
        "numerical_integration": numerical,
        "integration_pass": integration_pass,
        "engineering_summary": engineering_summary,
        "screen": screen,
        "final_checkpoint": final_checkpoint,
        "engineering": totals(engineering, 6),
        "training": totals(training, 24),
        "evaluation": {
            arm: totals([row for row in evaluation if row["arm"] == arm], 12)
            for arm in EVALUATION_ARMS
        },
        "per_task": {
            str(task): {
                arm: totals(
                    [row for row in evaluation if row["task"] == task and row["arm"] == arm], 2
                )
                for arm in EVALUATION_ARMS
            }
            for task in DEVELOPMENT_TASKS
        },
        "task_seed_block_differences_against_native": paired(evaluation),
        "training_curve_by_fixed_six_task_cycle": curves,
        "episodes": episodes,
        "attempts_without_valid_receipts": failures,
        "unresolved_invalid_jobs": unresolved_invalid,
        "resources": resources,
        "resource_samples": [
            row
            for row in read_events(run_dir / "queue-events.jsonl")
            if row["event"] == "resource_sample"
        ],
        "episode_runtime": [
            {
                "job_id": key,
                "started_utc": row.get("started_utc"),
                "finished_utc": row.get("finished_utc"),
            }
            for key, row in queue.get("episodes", {}).items()
        ],
        "limitations": [
            "Screen thresholds are practical, not a significance test or a risk guarantee.",
            "Wilson intervals assume independent episodes; seeds do not reset the simulator explicitly.",
            "Benchmark safe/unsafe classes use counted violation; unscored safety is reported separately.",
            "One learner seed on six development tasks cannot establish generalization or optimizer advantage.",
            "Corrected discounted shaped return equals discounted completion return; intermediate shaping is not an efficacy result.",
            "Actor latent saturation is abs(noise)>=0.99, not an arm-joint saturation measurement.",
        ],
    }


def write_report(report, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    curves = report["training_curve_by_fixed_six_task_cycle"]
    if curves:
        path = output.with_name(output.stem + "-training.csv")
        fields = [
            "cycle",
            "complete_six_task_cycle",
            "gaussian_warmup_episodes",
            "discounted_shaped_return",
            "discounted_base_return",
            "corrected_discounted_shaped_return",
            "base_reward_sum",
            "shaping_reward_sum",
            "undiscounted_reward_sum",
            "cost",
        ]
        with path.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(curves)
    lines = [f"Task 7 outcome: {report['outcome']}.", ""]
    numerical = report["numerical_integration"]
    if numerical:
        lines.append(
            f"Numerical integration admitted: {numerical.get('admit_live_integration')}; "
            f"live integration passed: {report['integration_pass']}."
        )
        lines.append("")
    for arm, summary in report["evaluation"].items():
        counts, count = summary["counts"], summary["denominator"]
        lines.append(
            f"- {arm}: strict safe completion {counts['strict_safe_requested']}/{count}; raw violations {counts['raw_violation']}/{count}; counted violations {counts['counted_violation']}/{count}; unfinished {counts['unfinished']}/{count}; engagement {counts['engaged']}/{count}."
        )
    lines.extend(
        [
            "",
            f"Frozen screen assessable: {report['screen']['assessable']}; passed: {report['screen']['passed']}.",
            "",
            *report["limitations"],
        ]
    )
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/dsrl07/remote")
    parser.add_argument(
        "--stage-a", type=Path, default=ROOT / "progress/receipts/07-dsrl-stage-a-v2.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "progress/receipts/07-dsrl-analysis.json"
    )
    args = parser.parse_args()
    report = analyze(args.run_dir, args.stage_a)
    write_report(report, args.output)
    print(json.dumps({"outcome": report["outcome"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
