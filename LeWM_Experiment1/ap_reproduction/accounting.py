#!/usr/bin/env python3
"""Reconcile downloaded AP run evidence; never import or run an experiment."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


PHASES = ("preparation", "cache", "engineering", "scientific", "maintenance")
ENTRYPOINTS = {
    "prepare.py": "preparation", "encode_cache.py": "cache",
    "engineering.py": "engineering", "evaluate.py": "scientific",
    "retire_inputs.py": "maintenance",
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
BOUNDARY_PAUSE_RUN = "ap-20261105-eval-pusht-s0"


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


def validate_boundary_pause(root: Path, *, ledger=None, backup_proof=None):
    """Validate the one recorded PushT pause, without changing its failed run status.

    This is an exact evidence exception, not permission to discard arbitrary failures.
    Physical trajectory scoring remains the independent scientific auditor's job.
    """
    receipt_path = root / "artifacts/recording-boundary-pause.json"
    if not receipt_path.exists():
        if read(root / "run.json").get("run_id") == BOUNDARY_PAUSE_RUN:
            raise ValueError(f"{BOUNDARY_PAUSE_RUN}: original pause receipt is missing")
        return None
    identity = BOUNDARY_PAUSE_RUN
    revision = "718366d6a3963faae33d297559e6f4233117c675"
    manifest_hash = "361bd8e55b1991407e115abeab7df551f6aac963f4fd705098bf94a7dad11c09"
    repository = "Sachioster/LeWMExp1_AP_Reproduction"

    def require(condition, message):
        if not condition:
            raise ValueError(f"{identity}: boundary pause validation failed: {message}")

    require(sha(root / "manifest.json") == manifest_hash, "unpinned source manifest")
    manifest = read(root / "manifest.json")
    require(manifest.get("run_id") == identity, "unexpected run identity")
    files = manifest["files"]

    def verified_file(relative):
        path = root / relative
        record = files.get(relative)
        require(isinstance(record, dict) and path.is_file(), f"missing {relative}")
        require(path.stat().st_size == record["bytes"] and sha(path) == record["sha256"],
                f"changed {relative}")
        return path

    receipt = read(verified_file("artifacts/recording-boundary-pause.json"))
    spec = read(verified_file("run.json"))
    result = read(verified_file("result.json"))
    account = read(verified_file("metrics/accounting.json"))
    saved_ledger = read(verified_file("metrics/study-attempt-ledger.json"))
    require(spec["run_id"] == receipt["run_id"] == identity, "receipt identity differs")
    require(spec.get("storage", {}).get("repo_id") == repository
            and spec["storage"].get("private") is True, "wrong evidence repository")
    require(result == manifest["result"] and result.get("status") == "failed"
            and result.get("exit_code") == 1, "original failed result was changed")
    require(account.get("completed") is False, "interrupted batch was relabeled completed")
    require(receipt.get("signal") == "SIGINT"
            and receipt.get("all_reserved_cases_completed") is True,
            "not the recorded clean SIGINT boundary")
    require(result["finished_epoch"] >= receipt["epoch"], "result precedes the pause")
    # The shared ledger uses indent=2; its published snapshot uses compact JSON.
    shared_bytes = (json.dumps(saved_ledger, indent=2, allow_nan=False) + "\n").encode()
    require(hashlib.sha256(shared_bytes).hexdigest() == receipt["ledger_sha256"],
            "published ledger does not match the boundary snapshot")
    proof_source = None
    if backup_proof is None:
        proof_path = Path(__file__).parent / "evidence/recording-correction-validation.json"
        attestation = read(proof_path)
        require(attestation["boundary"] == receipt
                and attestation["original_run_result"] == result
                and attestation["original_global_accounting"] == account
                and attestation.get("ledger_repair_required") is False,
                "saved full-backup attestation does not describe these original files")
        backup_proof = attestation["backup"]
        proof_source = {"path": str(proof_path), "sha256": sha(proof_path)}
    require(backup_proof.get("repo_id") == repository
            and backup_proof.get("verified") is True
            and backup_proof.get("revision") == revision
            and backup_proof.get("manifest_sha256") == manifest_hash,
            "missing exact verified full-backup proof")
    expected = [f"scientific:pusht:ap-fresh-pusht-{ordinal:03d}_clean_{arm}"
                for ordinal in range(0, 28, 4)
                for arm in ("gaussian_final", "gaussian_observed")]
    attempts = {key: value for key, value in saved_ledger["attempts"].items()
                if value.get("run") == identity}
    require(receipt.get("completed_identities") == expected and set(attempts) == set(expected),
            "reserved identities differ from the seven completed pairs")
    if ledger is not None:
        current = {key: value for key, value in ledger["attempts"].items()
                   if value.get("run") == identity}
        require(current == attempts, "later ledger changed or added an original run attempt")
    native_names = {f"{key.split(':', 2)[2]}.json" for key in expected}
    require({path.name for path in (root / "metrics/cases").glob("*.json")} == native_names,
            "native outcome inventory differs from reserved identities")
    total, actions = Counter(), 0
    for key in expected:
        attempt = attempts[key]
        finished = attempt.get("finished_epoch")
        require(attempt.get("phase") == "scientific" and attempt.get("completed") is True
                and type(finished) in (int, float) and math.isfinite(finished)
                and attempt["started_epoch"] <= finished <= receipt["epoch"],
                f"unfinished or post-boundary attempt {key}")
        case = key.split(":", 2)[2]
        native = read(verified_file(f"metrics/cases/{case}.json"))
        compact, counts = native["compact"], attempt["counts"]
        require(all(type(value) is int and value >= 0 for value in counts.values()),
                f"invalid counters for {key}")
        ordinal = compact["query_ordinal"]
        require(key == f"scientific:{compact['task']}:{compact['identity']['record_id']}_"
                       f"{compact['condition']}_{compact['arm']}"
                and ordinal in range(0, 28, 4) and compact.get("completed") is True
                and compact["optimizer_seed"] == 26091331 + 1000 * ordinal,
                f"native identity or seed differs for {key}")
        steps, decisions = len(native["actions"]), len(native["decisions"])
        require(steps == compact["primitive_steps"] <= compact["budget"] == 280
                and decisions == compact["decisions"] == math.ceil(steps / 5),
                f"native action/decision counts differ for {key}")
        for name in ("resets_attempted", "resets_completed"):
            require(counts.get(name) == 1, f"reset counts differ for {key}")
        for name in ("policy_actions_attempted", "policy_actions_completed",
                     "policy_controller_calls", "policy_controller_calls_completed"):
            require(counts.get(name, 0) == steps, f"action counts differ for {key}")
        for name, amount in (("predictor_calls", decisions * 31),
                             ("predictor_candidate_blocks", decisions * 9002)):
            require(counts.get(name, 0) == compact[name] == amount,
                    f"model counts differ for {key}")
        for completed, attempted in COMPLETE_PAIRS.items():
            require(counts.get(completed, 0) == counts.get(attempted, 0),
                    f"unfinished instrumented work for {key}: {attempted}")
        verified_file(f"metrics/trajectories/{case}.json.gz")
        for relative in (f"metrics/journals/{case}.jsonl", f"artifacts/videos/{case}.mp4"):
            require(relative in files and files[relative]["bytes"] > 0,
                    f"missing backed-up completed recording {relative}")
        total.update(counts)
        actions += steps
    require(account.get("attempted_outcomes") == account.get("completed_outcomes") == 14,
            "batch outcome counts differ")
    # Seven completed query goals plus ordinal 28's goal before blocked reserve/SIGINT.
    expected_global = total.copy()
    expected_global.update(encoder_calls=8, encoder_images=8)
    require(Counter(account["counts"]) == expected_global,
            "global counts do not reconcile exactly, including the next query goal encode")
    return {"run_id": identity, "original_result": result, "batch_completed": False,
            "all_reserved_cases_completed": True, "scientific_cases": 14,
            "policy_actions": actions, "global_counts": account["counts"],
            "outside_case_goal_encoder_calls": 8, "extra_next_query_goal_encoder_calls": 1,
            "completed_identities": expected, "backup": backup_proof,
            "backup_attestation_source": proof_source,
            "receipt_sha256": sha(receipt_path), "ledger_repaired": False}


def reconcile(roots: list[Path], config_path: Path, ledger_path: Path | None,
              final: bool, expected_scientific: int, partition_plan_path: Path | None = None,
              partition_ledgers: dict | None = None) -> dict:
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
    partition_plan = read(partition_plan_path) if partition_plan_path else None
    partitions = {"global": {"budgets": caps, "tasks": list(config["task_specs"])}}
    if partition_plan is not None:
        require(ledger_path is None, "Use named --partition-ledger with a partition plan")
        require(partition_plan.get("global_budgets") == caps,
                "Partition plan changed the original global budgets")
        entries = partition_plan.get("partitions", [])
        partitions = {entry["id"]: entry for entry in entries}
        require(len(partitions) == len(entries) and bool(entries), "Invalid partition identities")
        assignments, partition_caps = [], Counter()
        for entry in entries:
            assignments.extend(entry["tasks"])
            partition_caps.update(counts(entry["budgets"], entry["id"] + " caps"))
            require(set(entry["budgets"]) == set(caps), "Partition budget keys differ")
        require(sorted(assignments) == sorted(config["task_specs"]),
                "Each task must belong to exactly one host partition")
        require(dict(partition_caps) == caps, "Partition caps do not sum to frozen global caps")
        if final:
            require(partition_plan.get("status") == "authorized", "Host split is not authorized")
            require(all(entry.get("session_id") and entry.get("instance_id") for entry in entries),
                    "Final accounting has an unbound host partition")
    partition_ledgers = partition_ledgers or {}
    require(not partition_ledgers or partition_plan is not None,
            "Named ledgers require a partition plan")
    require(set(partition_ledgers).issubset(partitions), "Unknown named ledger partition")
    snapshots = []
    snapshot_partition = {}
    runs = {}
    readiness_sessions = {}
    readiness_totals = Counter()
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
        partition = "global"
        if partition_plan is not None:
            matching = [key for key, entry in partitions.items()
                        if (entry.get("session_id"), entry.get("instance_id"))
                        == (spec.get("session_id"), spec.get("instance_id"))]
            require(len(matching) == 1, f"{run_id}: session has no unique host partition")
            partition = matching[0] if len(matching) == 1 else "unassigned"
        session_key = (spec.get("session_id"), spec.get("instance_id"))
        readiness = spec.get("readiness")
        require(all(value is not None for value in session_key),
                f"{run_id}: readiness has no session/instance identity")
        require(isinstance(readiness, dict), f"{run_id}: missing startup readiness record")
        if session_key in readiness_sessions:
            session = readiness_sessions[session_key]
            require(session["readiness"] == readiness,
                    f"{run_id}: startup readiness changed within the same session/instance")
            session["source_runs"].append(run_id)
        else:
            readiness_sessions[session_key] = {
                "session_id": session_key[0], "instance_id": session_key[1],
                "readiness": readiness, "source_runs": [run_id],
                "first_source_spec": str(spec_path), "first_source_spec_sha256": sha(spec_path),
            }
            rendering = readiness.get("rendering", {}) if isinstance(readiness, dict) else {}
            for name in ("synthetic_frames", "benchmark_episodes", "physics_steps"):
                value = rendering.get(name)
                valid = isinstance(value, int) and not isinstance(value, bool) and value >= 0
                require(valid, f"{run_id}: missing or invalid startup rendering counter {name}")
                if valid:
                    readiness_totals[name] += value
            require(rendering.get("benchmark_episodes") == rendering.get("physics_steps") == 0,
                    f"{run_id}: startup readiness unexpectedly includes benchmark/physics work")
        command = spec.get("command", [])
        matches = {ENTRYPOINTS[Path(part).name] for part in command
                   if Path(part).name in ENTRYPOINTS}
        require(len(matches) == 1, f"{run_id}: cannot identify one experiment phase")
        phase = next(iter(matches)) if len(matches) == 1 else "unknown"
        tasks = [part for part in command if part in config["task_specs"]]
        if partition in partitions:
            require(len(tasks) == 1 and tasks[0] in partitions[partition]["tasks"],
                    f"{run_id}: command task is outside its assigned host partition")
        result_path = root / "result.json"
        result = read(result_path) if result_path.exists() else {"status": "in_progress"}
        run_config_path = root / "config.json"
        if run_config_path.exists():
            run_config = read(run_config_path)
            entry = partitions.get(partition, {})
            legacy = run_id in entry.get("legacy_global_budget_runs", [])
            expected_caps = caps if legacy else entry.get("budgets")
            require(run_config.get("budgets") == expected_caps,
                    f"{run_id}: budget configuration differs from assigned partition/history")
            if partition_plan is not None and not legacy:
                require(run_config.get("execution_partition") == partition,
                        f"{run_id}: new configuration omits its host partition")
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
        maintenance_receipts = []
        if phase == "maintenance":
            for receipt_path in sorted((root / "artifacts").glob("*retir*.json")):
                maintenance_receipts.append({"path": str(receipt_path),
                                             "sha256": sha(receipt_path),
                                             "receipt": read(receipt_path)})
        runs[run_id] = {
            "root": root, "spec": spec, "phase": phase, "result": result,
            "account": account, "failure": failure, "preparation": preparation, "cache": cache,
            "maintenance_receipts": maintenance_receipts,
            "partition": partition,
            "ledger_counts": Counter(), "attempt_ids": [], "native_outcomes": 0,
        }
        snapshot_path = root / "metrics/study-attempt-ledger.json"
        if snapshot_path.exists():
            snapshots.append((snapshot_path, read(snapshot_path)))
            snapshot_partition[str(snapshot_path)] = partition
    if ledger_path is not None:
        snapshots.append((ledger_path, read(ledger_path)))
        snapshot_partition[str(ledger_path)] = "global"
    for partition, path in partition_ledgers.items():
        snapshots.append((path, read(path)))
        snapshot_partition[str(path)] = partition
    require(bool(snapshots), "No shared attempt-ledger snapshot was supplied")

    # Snapshots are monotonic within a host. Independent host ledgers are disjoint.
    merged, identity_partition, per_partition = {}, {}, {key: {} for key in partitions}
    for path, snapshot in snapshots:
        partition = snapshot_partition[str(path)]
        target = per_partition.setdefault(partition, {})
        for identity, attempt in snapshot.get("attempts", {}).items():
            require(identity not in identity_partition or identity_partition[identity] == partition,
                    f"{identity}: duplicated across independent host ledgers")
            identity_partition.setdefault(identity, partition)
            if identity in target:
                previous = target[identity]
                for key in ("phase", "reserved", "run", "started_epoch"):
                    require(previous.get(key) == attempt.get(key),
                            f"{identity}: immutable ledger field {key} changed in {path}")
                if "finished_epoch" in previous and "finished_epoch" in attempt:
                    require(previous == attempt, f"{identity}: finished attempt was overwritten")
            if identity not in target or "finished_epoch" in attempt:
                target[identity] = attempt
                merged.setdefault(identity, attempt)
                if identity_partition[identity] == partition:
                    merged[identity] = attempt
    selected_by_partition = {}
    selected_reserved = Counter()
    for partition, target in per_partition.items():
        candidates = [(path, value) for path, value in snapshots
                      if snapshot_partition[str(path)] == partition]
        explicit = partition_ledgers.get(partition)
        if partition == "global" and ledger_path is not None:
            explicit = ledger_path
        if explicit is not None:
            path, value = explicit, read(explicit)
        elif candidates:
            path, value = max(candidates, key=lambda item: (
                len(item[1].get("attempts", {})),
                sum("finished_epoch" in a for a in item[1].get("attempts", {}).values()),
                max((a.get("finished_epoch", a.get("started_epoch", 0))
                     for a in item[1].get("attempts", {}).values()), default=0)))
        else:
            path, value = None, {"attempts": {}, "reserved": {}}
            require(not final and not any(r["partition"] == partition for r in runs.values()),
                    f"{partition}: missing independent ledger snapshot")
        require(value.get("attempts", {}) == target,
                f"{partition}: latest ledger is stale or incomplete within its host")
        selected_by_partition[partition] = {"path": path, "value": value}
        selected_reserved.update(value.get("reserved", {}))
    selected_path = selected_by_partition.get("global", {}).get("path")
    selected = {"attempts": merged, "reserved": dict(selected_reserved)}

    reserved = Counter()
    reserved_by_partition = {key: Counter() for key in per_partition}
    phase_attempts = {phase: Counter() for phase in PHASES}
    pending, unsuccessful = [], []
    all_attempts = [(partition, identity, attempt) for partition, target in per_partition.items()
                    for identity, attempt in target.items()]
    for partition, identity, attempt in all_attempts:
        phase, run_id = attempt["phase"], attempt["run"]
        require(phase in PHASES, f"{identity}: unknown ledger phase")
        if phase not in PHASES:
            continue
        phase_attempts[phase]["attempts"] += 1
        allocation = counts(attempt["reserved"], identity + " reservations")
        measured = counts(attempt.get("counts", {}), identity)
        for name, limit in allocation.items():
            reserved[phase + "_" + name] += limit
            reserved_by_partition[partition][phase + "_" + name] += limit
            actual_key = MEASURED_KEYS.get(name)
            if actual_key:
                require(measured[actual_key] <= limit,
                        f"{identity}: measured {actual_key} exceeds its reservation")
        require(run_id in runs, f"{identity}: missing downloaded run {run_id}")
        if run_id in runs:
            require(runs[run_id]["partition"] == partition,
                    f"{identity}: ledger and run belong to different hosts")
            require(runs[run_id]["phase"] == phase, f"{identity}: run/ledger phase mismatch")
            runs[run_id]["ledger_counts"].update(measured)
            runs[run_id]["attempt_ids"].append(identity)
        parts = identity.split(":")
        require(len(parts) > 1 and parts[1] in partitions.get(partition, {}).get("tasks", []),
                f"{identity}: attempt task is outside its assigned host")
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
    for partition, allocation in reserved_by_partition.items():
        require(dict(allocation) == selected_by_partition[partition]["value"].get("reserved", {}),
                f"{partition}: independent ledger reservation sum differs")
        local_caps = partitions.get(partition, {}).get("budgets", {})
        for key, amount in allocation.items():
            require(key in local_caps and amount <= local_caps.get(key, -1),
                    f"{partition}: reservation exceeds partition cap {key}")
    for key, amount in reserved.items():
        require(key in caps, f"Reservation has no declared cap: {key}")
        if key in caps:
            require(amount <= caps[key], f"Reservation exceeds cap: {key}")

    raw_totals = {phase: Counter() for phase in PHASES}
    outside_totals = {phase: Counter() for phase in PHASES}
    records, failed_runs, unknown_runs, boundary_pauses = [], [], [], []
    seen_cases = {}
    native_totals = Counter()
    for run_id, run in runs.items():
        root, phase = run["root"], run["phase"]
        if phase not in PHASES:
            continue
        account, failure = run["account"], run["failure"]
        boundary_pause = None
        if ((root / "artifacts/recording-boundary-pause.json").exists()
                or run_id == BOUNDARY_PAUSE_RUN):
            require(phase == "scientific", f"{run_id}: boundary receipt on a non-scientific run")
            try:
                boundary_pause = validate_boundary_pause(root, ledger=selected)
                boundary_pauses.append(boundary_pause)
            except (OSError, ValueError, KeyError, TypeError) as error:
                require(False, str(error))
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
        if phase == "maintenance":
            require(not any(actual.values()), f"{run_id}: maintenance reports experimental work")
            require(not run["attempt_ids"], f"{run_id}: maintenance reserved experimental work")
            if account is not None:
                require(account.get("simulator_actions") == account.get("model_queries") == 0,
                        f"{run_id}: maintenance action/query declarations differ from zero")
            if run["result"].get("status") == "completed":
                completed_receipts = [item["receipt"] for item in run["maintenance_receipts"]
                                      if Path(item["path"]).name == "retirement.json"]
                require(len(completed_receipts) == 1,
                        f"{run_id}: completed maintenance has no unique final retirement receipt")
                for receipt in completed_receipts:
                    require(receipt.get("deleted") is True,
                            f"{run_id}: final retirement receipt does not confirm deletion")
                    require(receipt.get("simulator_actions") == receipt.get("model_queries") == 0,
                            f"{run_id}: retirement receipt reports experimental work")
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
            if run["native_outcomes"] and account.get("completed") is not True:
                require(boundary_pause is not None,
                        f"{run_id}: incomplete scientific batch has no validated boundary pause")
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
                "maintenance_deleted": account.get("deleted")
                if phase == "maintenance" and account else None,
                "intentional_completed_case_boundary": boundary_pause is not None,
            })
        for item in pending:
            if item["run"] == run_id and status != "in_progress":
                warnings.append(f"{item['identity']}: interrupted attempt is explicitly retained")
        records.append({
            "run_id": run_id, "phase": phase, "partition": run["partition"], "task": next(
                (part for part in run["spec"].get("command", [])
                 if part in ("cube", "pusht", "reacher", "tworoom")), None),
            "root": str(root), "status": status, "actual_counts": dict(actual),
            "batch_completed": account.get("completed") if account else None,
            "intentional_completed_case_boundary": boundary_pause is not None,
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
            "maintenance_receipts": run["maintenance_receipts"],
            "maintenance_status": {key: account.get(key) for key in (
                "task", "completed", "deleted", "simulator_actions", "model_queries")}
            if phase == "maintenance" and account else None,
        })

    budget_table = {}
    missing_run_ids = sorted({attempt["run"] for _, _, attempt in all_attempts
                              if attempt["run"] not in runs})
    partition_reports = {}
    for partition, entry in partitions.items():
        local_records = [record for record in records if record["partition"] == partition]
        local_actual = {phase: Counter() for phase in PHASES}
        for record in local_records:
            local_actual[record["phase"]].update(record["actual_counts"])
        local_budget = {}
        for key, limit in entry["budgets"].items():
            phase, name = key.split("_", 1)
            measured = local_actual[phase][MEASURED_KEYS[name]]
            require(measured <= limit, f"{partition}: measured work exceeds cap {key}")
            local_budget[key] = {"cap": limit, "reserved_upper_bound":
                                 reserved_by_partition.get(partition, {}).get(key, 0),
                                 "measured_attempts": measured}
        chosen = selected_by_partition.get(partition, {})
        path = chosen.get("path")
        partition_reports[partition] = {
            "session_id": entry.get("session_id"), "instance_id": entry.get("instance_id"),
            "tasks": entry["tasks"], "runs": [record["run_id"] for record in local_records],
            "ledger": str(path) if path else None, "ledger_sha256": sha(path) if path else None,
            "identities": len(per_partition.get(partition, {})), "budgets": local_budget,
        }
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
                                     for record in records) and not any(
                                     attempt["phase"] == phase
                                     and attempt["run"] in missing_run_ids
                                     for attempt in merged.values())}
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
    complete = not unknown_runs and not pending and not missing_run_ids
    return {
        "schema": "lewm-ap-saved-evidence-accounting-v1", "final_requested": final,
        "passed": not errors, "errors": errors, "warnings": warnings,
        "accounting_complete": complete, "full_scientific_cohort": full_cohort,
        "missing_downloaded_runs": missing_run_ids,
        "expected_scientific_outcomes": expected_scientific,
        "scientific_completed_native_outcome_totals": dict(native_totals),
        "budgets": budget_table,
        "phases": {phase: {"run_count": sum(record["phase"] == phase for record in records),
                           "attempt_status": dict(phase_attempts[phase]),
                           "actual_raw_counts": dict(raw_totals[phase]),
                           "has_measured_experimental_work": any(raw_totals[phase].values()),
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
        "validated_boundary_pauses": boundary_pauses,
        "unfinished_attempts": pending, "runs": records,
        "infrastructure_readiness": {
            "scope": "Rental startup checks outside benchmark work, before each host's evaluation",
            "deduplication_key": ["session_id", "instance_id"],
            "sessions": list(readiness_sessions.values()),
            "totals": {name: readiness_totals[name] for name in (
                "synthetic_frames", "benchmark_episodes", "physics_steps")},
            "note": "One synthetic OSMesa frame per recorded startup, not one per run; "
                    "startup import/CUDA checks did not evaluate pretrained checkpoints",
        },
        "ledger": {"selected": str(selected_path) if selected_path else None,
                   "selected_sha256": sha(selected_path) if selected_path else None,
                   "identities": len(merged), "reserved_upper_bounds": dict(reserved),
                   "snapshots": [{"path": str(path), "sha256": sha(path)}
                                 for path, _ in snapshots]},
        "host_partitions": partition_reports,
        "partition_plan": {"path": str(partition_plan_path),
                           "sha256": sha(partition_plan_path),
                           "status": partition_plan.get("status")}
                          if partition_plan_path else None,
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
    parser.add_argument("--partition-plan", type=Path, help="Prospective host/cap assignment record")
    parser.add_argument("--partition-ledger", action="append", default=[],
                        help="Fresh independent ledger as PARTITION=PATH; repeat for each host")
    parser.add_argument("--expected-scientific", type=int, default=1024)
    parser.add_argument("--final", action="store_true", help="Require the complete final cohort")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    partition_ledgers = {}
    for value in args.partition_ledger:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in partition_ledgers:
            parser.error("Each --partition-ledger must be a unique PARTITION=PATH")
        partition_ledgers[name] = Path(path)
    report = reconcile(args.run_root, args.config, args.ledger,
                       args.final, args.expected_scientific, args.partition_plan, partition_ledgers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": report["passed"], "errors": len(report["errors"]),
                      "accounting_complete": report["accounting_complete"],
                      "full_scientific_cohort": report["full_scientific_cohort"]}))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
