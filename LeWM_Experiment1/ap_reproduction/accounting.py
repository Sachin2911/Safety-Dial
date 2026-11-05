#!/usr/bin/env python3
"""Reconcile downloaded AP run evidence; never import or run an experiment."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


PHASES = ("preparation", "cache", "engineering", "scientific")
ENTRYPOINTS = {
    "prepare.py": "preparation", "encode_cache.py": "cache",
    "engineering.py": "engineering", "evaluate.py": "scientific",
}
MEASURED_KEYS = {
    "resets": "resets_attempted", "policy_actions": "policy_actions_attempted",
    "predictor_calls": "predictor_calls",
    "predictor_candidate_blocks": "predictor_candidate_blocks",
    "encoder_images": "encoder_images_attempted",
}
COMPLETE_PAIRS = {
    "resets_completed": "resets_attempted",
    "policy_actions_completed": "policy_actions_attempted",
    "setup_controller_calls_completed": "setup_controller_calls",
    "policy_controller_calls_completed": "policy_controller_calls",
    "setup_physics_steps_completed": "setup_physics_steps",
    "policy_physics_steps_completed": "policy_physics_steps",
    "encoder_images_completed": "encoder_images_attempted",
}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Repeated JSON key: {key}")
        result[key] = value
    return result


def read(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=unique_object)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def reconcile(roots: list[Path], config_path: Path, ledger_path: Path | None,
              final: bool, expected_scientific: int) -> dict:
    errors, warnings = [], []

    def require(condition, message):
        if not condition:
            errors.append(message)

    def counts(value, label):
        result = Counter()
        for key, amount in value.items():
            valid = isinstance(amount, int) and not isinstance(amount, bool) and amount >= 0
            require(valid, f"{label}: invalid nonnegative integer counter {key}")
            if valid:
                result[key] = amount
        for completed, attempted in COMPLETE_PAIRS.items():
            require(result[completed] <= result[attempted],
                    f"{label}: {completed} exceeds {attempted}")
        return result

    config = read(config_path)
    caps = config["budgets"]
    snapshots = []
    runs = {}
    for root in roots:
        spec_path = root / "run.json"
        require(spec_path.exists(), f"{root}: missing run.json")
        if not spec_path.exists():
            continue
        spec = read(spec_path)
        run_id = spec["run_id"]
        require(run_id not in runs, f"Duplicate downloaded run identity: {run_id}")
        if run_id in runs:
            continue
        command = spec.get("command", [])
        matches = {ENTRYPOINTS[Path(part).name] for part in command
                   if Path(part).name in ENTRYPOINTS}
        require(len(matches) == 1, f"{run_id}: cannot identify one experiment phase")
        phase = next(iter(matches)) if len(matches) == 1 else "unknown"
        result_path = root / "result.json"
        result = read(result_path) if result_path.exists() else {"status": "in_progress"}
        run_config_path = root / "config.json"
        if run_config_path.exists():
            run_config = read(run_config_path)
            require(run_config.get("budgets") == caps, f"{run_id}: budget configuration differs")
        else:
            require(False, f"{run_id}: missing frozen configuration")
        account_path = root / "metrics/accounting.json"
        failure_path = root / "metrics/failure.json"
        account = read(account_path) if account_path.exists() else None
        failure = read(failure_path) if failure_path.exists() else None
        preparation_path = root / "artifacts/preparation.json"
        preparation = read(preparation_path) if preparation_path.exists() else None
        cache_path = root / "artifacts/cache.json"
        cache = read(cache_path) if cache_path.exists() else None
        runs[run_id] = {
            "root": root, "spec": spec, "phase": phase, "result": result,
            "account": account, "failure": failure, "preparation": preparation, "cache": cache,
            "ledger_counts": Counter(), "attempt_ids": [], "native_outcomes": 0,
        }
        snapshot_path = root / "metrics/study-attempt-ledger.json"
        if snapshot_path.exists():
            snapshots.append((snapshot_path, read(snapshot_path)))
    if ledger_path is not None:
        snapshots.append((ledger_path, read(ledger_path)))
    require(bool(snapshots), "No shared attempt-ledger snapshot was supplied")

    # Repeated snapshots are evidence of the same identities, never extra attempts.
    merged = {}
    for path, snapshot in snapshots:
        for identity, attempt in snapshot.get("attempts", {}).items():
            if identity in merged:
                previous = merged[identity]
                for key in ("phase", "reserved", "run", "started_epoch"):
                    require(previous.get(key) == attempt.get(key),
                            f"{identity}: immutable ledger field {key} changed in {path}")
                if "finished_epoch" in previous and "finished_epoch" in attempt:
                    require(previous == attempt, f"{identity}: finished attempt was overwritten")
            if identity not in merged or "finished_epoch" in attempt:
                merged[identity] = attempt
    if ledger_path is not None:
        selected_path, selected = ledger_path, read(ledger_path)
    elif snapshots:
        selected_path, selected = max(snapshots, key=lambda item: (
            len(item[1].get("attempts", {})),
            sum("finished_epoch" in a for a in item[1].get("attempts", {}).values()),
            max((a.get("finished_epoch", a.get("started_epoch", 0))
                 for a in item[1].get("attempts", {}).values()), default=0),
        ))
    else:
        selected_path, selected = None, {"attempts": {}, "reserved": {}}
    require(selected.get("attempts", {}) == merged,
            "Latest ledger is incomplete or stale relative to supplied snapshots")

    reserved = Counter()
    phase_attempts = {phase: Counter() for phase in PHASES}
    pending, unsuccessful = [], []
    for identity, attempt in merged.items():
        phase, run_id = attempt["phase"], attempt["run"]
        require(phase in PHASES, f"{identity}: unknown ledger phase")
        if phase not in PHASES:
            continue
        phase_attempts[phase]["attempts"] += 1
        allocation = counts(attempt["reserved"], identity + " reservations")
        measured = counts(attempt.get("counts", {}), identity)
        for name, limit in allocation.items():
            reserved[phase + "_" + name] += limit
            actual_key = MEASURED_KEYS.get(name)
            if actual_key:
                require(measured[actual_key] <= limit,
                        f"{identity}: measured {actual_key} exceeds its reservation")
        require(run_id in runs, f"{identity}: missing downloaded run {run_id}")
        if run_id in runs:
            require(runs[run_id]["phase"] == phase, f"{identity}: run/ledger phase mismatch")
            runs[run_id]["ledger_counts"].update(measured)
            runs[run_id]["attempt_ids"].append(identity)
        if "finished_epoch" not in attempt:
            phase_attempts[phase]["inflight_or_interrupted"] += 1
            pending.append({"identity": identity, "run": run_id,
                            "reserved": dict(allocation), "known_counts": dict(measured)})
        elif attempt.get("completed") is True:
            phase_attempts[phase]["completed"] += 1
        else:
            phase_attempts[phase]["failed"] += 1
            unsuccessful.append({"identity": identity, "run": run_id,
                                 "known_counts": dict(measured)})
    require(dict(reserved) == selected.get("reserved", {}),
            "Summed per-attempt reservations differ from the shared ledger totals")
    for key, amount in reserved.items():
        require(key in caps, f"Reservation has no declared cap: {key}")
        if key in caps:
            require(amount <= caps[key], f"Reservation exceeds cap: {key}")

    raw_totals = {phase: Counter() for phase in PHASES}
    outside_totals = {phase: Counter() for phase in PHASES}
    records, failed_runs, unknown_runs = [], [], []
    seen_cases = {}
    native_totals = Counter()
    for run_id, run in runs.items():
        root, phase = run["root"], run["phase"]
        if phase not in PHASES:
            continue
        account, failure = run["account"], run["failure"]
        coverage, count_source = "complete", "metrics/accounting.json"
        if account is not None:
            actual = counts(account.get("counts", {}), run_id)
        elif phase == "preparation":
            actual = Counter()
            count_source = "preparation entrypoint performs transfers and metadata inspection only"
            if run["preparation"] is not None:
                receipt = run["preparation"]
                require(receipt.get("simulator_actions") == receipt.get("model_queries") == 0,
                        f"{run_id}: preparation unexpectedly reports model/simulator work")
        elif failure is not None:
            actual = counts(failure.get("counts", {}), run_id + " failure")
            coverage, count_source = "lower_bound", "metrics/failure.json"
        else:
            actual = run["ledger_counts"].copy()
            count_source, coverage = "finished attempt counts only", "lower_bound"
            if phase == "cache":
                progress = sorted((root / "metrics").glob("cache-*.json"))
                if progress:
                    latest = read(progress[-1])
                    partial = counts(latest.get("counts", {}), run_id + " cache progress")
                    for name, amount in partial.items():
                        actual[name] = max(actual[name], amount)
                    count_source = "cache progress and finished-attempt lower bound"
        if coverage != "complete":
            unknown_runs.append(run_id)
            warnings.append(f"{run_id}: totals remain lower bounds until final accounting arrives")
        outside = Counter()
        for name in set(actual) | set(run["ledger_counts"]):
            difference = actual[name] - run["ledger_counts"][name]
            require(difference >= 0, f"{run_id}: attempt counts exceed run counter {name}")
            if difference > 0:
                outside[name] = difference
        if failure is not None:
            failure_counts = counts(failure.get("counts", {}), run_id + " failure")
            for name, amount in failure_counts.items():
                require(amount <= actual[name], f"{run_id}: failure work is absent from totals")
        if not any(item["run"] == run_id for item in pending):
            non_encoder = {key: value for key, value in outside.items()
                           if key not in ("encoder_images", "encoder_calls")}
            require(not non_encoder,
                    f"{run_id}: non-encoder work has no corresponding attempt: {non_encoder}")
        if run["cache"] is not None:
            receipt = run["cache"]
            require(receipt["rows"] == actual["encoder_images_completed"],
                    f"{run_id}: completed cache frames differ from its row count")
            require(receipt["storage_dtype"] == "float16"
                    and receipt["encode_batch_size"] == 128,
                    f"{run_id}: cache representation or encoding batch differs")
            require(counts(receipt["counts"], run_id + " cache receipt") == actual,
                    f"{run_id}: cache receipt and actual counters differ")
        raw_totals[phase].update(actual)
        outside_totals[phase].update(outside)

        for case_path in sorted((root / "metrics/cases").glob("*.json")):
            compact = read(case_path)["compact"]
            identity = (f"scientific:{compact['task']}:{compact['identity']['record_id']}_"
                        f"{compact['condition']}_{compact['arm']}")
            require(identity not in seen_cases, f"Duplicated scientific outcome: {identity}")
            if identity in seen_cases:
                continue
            seen_cases[identity] = str(case_path)
            require(identity in merged, f"Native outcome has no attempt reservation: {identity}")
            if identity in merged:
                require(merged[identity]["run"] == run_id,
                        f"{identity}: outcome is attributed to a different run")
                if final:
                    require(merged[identity].get("completed") is True,
                            f"{identity}: native outcome exists but its attempt is unfinished")
            run["native_outcomes"] += 1
            native_totals["outcomes"] += 1
            native_totals["policy_actions_completed"] += compact["primitive_steps"]
            native_totals["predictor_calls_completed_outcomes"] += compact["predictor_calls"]
            native_totals["predicted_blocks_completed_outcomes"] += compact[
                "predictor_candidate_blocks"
            ]
        if account is not None and phase == "scientific":
            require(account.get("attempted_outcomes") == len(run["attempt_ids"]),
                    f"{run_id}: scientific attempt count differs from ledger")
            require(account.get("completed_outcomes") == run["native_outcomes"],
                    f"{run_id}: native outcome files are missing or duplicated")
        status = run["result"].get("status", "unknown")
        failed = failure is not None or status in {
            "failed", "timed_out", "stopped", "runner_failed", "interrupted"
        }
        if failed:
            failed_runs.append({
                "run_id": run_id, "phase": phase, "status": status,
                "error": failure.get("error") if failure else run["result"].get("error"),
                "measured_counts": dict(actual), "coverage": coverage,
                "zero_measured_work": coverage == "complete" and not any(actual.values()),
                "failed_before_reservation": not run["attempt_ids"],
            })
        for item in pending:
            if item["run"] == run_id and status != "in_progress":
                warnings.append(f"{item['identity']}: interrupted attempt is explicitly retained")
        records.append({
            "run_id": run_id, "phase": phase, "task": next(
                (part for part in run["spec"].get("command", [])
                 if part in ("cube", "pusht", "reacher", "tworoom")), None),
            "root": str(root), "status": status, "actual_counts": dict(actual),
            "count_source": count_source, "coverage": coverage,
            "attempt_count": len(run["attempt_ids"]), "native_outcomes": run["native_outcomes"],
            "sum_attempt_counts": dict(run["ledger_counts"]),
            "outside_attempt_counts": dict(outside),
            "elapsed_seconds": account.get("elapsed_seconds") if account else (
                run["preparation"].get("seconds") if run["preparation"] else None),
            "spec_sha256": sha(root / "run.json"),
            "accounting_sha256": sha(root / "metrics/accounting.json") if account else None,
            "preparation": None if run["preparation"] is None else {
                "task": run["preparation"]["task"],
                "verified_source_archive_bytes": run["preparation"]["dataset"]["archive_bytes"],
                "dataset_bytes": run["preparation"]["dataset"]["bytes"],
                "dataset_sha256": run["preparation"]["dataset"]["sha256"],
                "note": "Verified asset sizes, not a measurement of actual network traffic",
            },
            "cache": None if run["cache"] is None else {
                key: run["cache"].get(key) for key in (
                    "rows", "storage_dtype", "encode_batch_size", "sha256", "bytes")
            },
        })

    budget_table = {}
    for phase in PHASES:
        for name, actual_key in MEASURED_KEYS.items():
            key = phase + "_" + name
            if key not in caps:
                continue
            amount = raw_totals[phase][actual_key]
            require(amount <= caps[key], f"Measured work exceeds cap: {key}")
            budget_table[key] = {"cap": caps[key], "reserved_upper_bound": reserved[key],
                                 "measured_attempts": amount,
                                 "measured_scope": actual_key,
                                 "measurement_complete": not any(
                                     record["phase"] == phase and record["coverage"] != "complete"
                                     for record in records)}
    scientific = phase_attempts["scientific"]
    full_cohort = (scientific["attempts"] == scientific["completed"] == expected_scientific
                   and len(seen_cases) == expected_scientific)
    require(scientific["attempts"] <= expected_scientific, "Scientific identities exceed cohort")
    if final:
        require(full_cohort, f"Final cohort is incomplete: {len(seen_cases)}/{expected_scientific}")
        require(not unknown_runs, "Final accounting still contains lower-bound-only runs")
        require(not pending, "Final accounting contains unfinished attempt records")
        require(all(run["result"].get("status") != "in_progress" for run in runs.values()),
                "Final report contains a run without a terminal result")
    complete = not unknown_runs and not pending
    return {
        "schema": "lewm-ap-saved-evidence-accounting-v1", "final_requested": final,
        "passed": not errors, "errors": errors, "warnings": warnings,
        "accounting_complete": complete, "full_scientific_cohort": full_cohort,
        "expected_scientific_outcomes": expected_scientific,
        "scientific_completed_native_outcome_totals": dict(native_totals),
        "budgets": budget_table,
        "phases": {phase: {"attempt_status": dict(phase_attempts[phase]),
                           "actual_raw_counts": dict(raw_totals[phase]),
                           "outside_attempt_counts": dict(outside_totals[phase])}
                   for phase in PHASES},
        "encoder_accounting": {
            "cache_images_attempted": raw_totals["cache"]["encoder_images_attempted"],
            "cache_images_completed": raw_totals["cache"]["encoder_images_completed"],
            "cache_calls_attempted": raw_totals["cache"]["encoder_calls"],
            "engineering_online_images_attempted": raw_totals["engineering"]["encoder_images"],
            "scientific_online_images_attempted": raw_totals["scientific"]["encoder_images"],
            "scientific_images_outside_case_counters": outside_totals["scientific"][
                "encoder_images"
            ],
            "explanation": "Global run counters include goal encodings before case reservation",
        },
        "failed_runs": failed_runs, "failed_attempts": unsuccessful,
        "unfinished_attempts": pending, "runs": records,
        "ledger": {"selected": str(selected_path) if selected_path else None,
                   "selected_sha256": sha(selected_path) if selected_path else None,
                   "identities": len(merged), "reserved_upper_bounds": dict(reserved),
                   "snapshots": [{"path": str(path), "sha256": sha(path)}
                                 for path, _ in snapshots]},
        "measurement_scope": [
            "Reservations are conservative upper bounds; they are never measured interactions.",
            "Online encoder/predictor hooks count attempted forwards, including failed calls.",
            "Cache encoding separately records attempted and returned image counts.",
            "Controller transitions and reset-internal controller calls are separate counters.",
            "Physics counters observe MuJoCo mj_step/mj_step2 and Pymunk Space.step only.",
            "Zero integration-hook counts do not mean zero TwoRoom environment transitions.",
            "Per-case counters exclude some goal encodings; use global run totals for model work.",
            "Failed runs before reservation remain visible even when their measured work is zero.",
            "Only supplied downloaded runs and ledger snapshots are covered by this report.",
        ],
        "tworoom_transition_accounting": {
            "mechanism": "Torch position update followed by collision handling in env.step",
            "count": "policy_controller_calls_completed counts returned environment transitions",
            "integration_hooks": "This environment does not call MuJoCo or Pymunk integration",
            "source": "stable-worldmodel 0.0.6, installed tworoom/env.py inspected from rental",
            "source_sha256": "5e1d392de5b02472062dbe872aded67fd465fcc8f7eaa1c02a753b2fc31c61f0",
        },
        "configuration": {"path": str(config_path), "sha256": sha(config_path),
                          "study": config.get("study"), "budgets_fingerprint": fingerprint(caps)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, help="Fresh downloaded shared attempt ledger")
    parser.add_argument("--expected-scientific", type=int, default=1024)
    parser.add_argument("--final", action="store_true", help="Require the complete final cohort")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = reconcile(args.run_root, args.config, args.ledger,
                       args.final, args.expected_scientific)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": report["passed"], "errors": len(report["errors"]),
                      "accounting_complete": report["accounting_complete"],
                      "full_scientific_cohort": report["full_scientific_cohort"]}))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
