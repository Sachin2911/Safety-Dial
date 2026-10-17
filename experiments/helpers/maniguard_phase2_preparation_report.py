"""Immutable preparation accounting and offline diagnostics; no process launching."""

import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from helpers.maniguard_phase2_audit import audit_preparation, audit_sources
from helpers.maniguard_phase2_model import decode, forward, genome_hash
from helpers.maniguard_phase2_optim import SearchConfig, WeightSearch, record_hash
from helpers.maniguard_phase2_protocol import compile_schedule
from helpers.maniguard_phase2_results import read_episode
from helpers.maniguard_phase2_services import PRIOR_COUNTS, PRIOR_SCHEMA
from helpers.maniguard_results import read_episode as read_original

ROOT = Path(__file__).resolve().parents[2]


def _json(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _source(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _snapshot(source, target):
    raw = Path(source).read_bytes()
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(raw)
    return json.loads(raw), _source(target)


def _count(value):
    return type(value) is int and value >= 0


def audit_process(receipt):
    """Match prepare.process_succeeded while retaining each failure explanation.

    Older successful receipts did not record a process object. Their absent flags
    remain accepted; an explicitly malformed object is invalid evidence.
    """
    reasons = []
    if receipt.get("returncode") != 0:
        reasons.append("Process returncode is not zero: " + repr(receipt.get("returncode")))
    if receipt.get("process_error"):
        reasons.append("Process launcher error: " + str(receipt["process_error"]))
    process = receipt.get("process", {})
    if not isinstance(process, dict):
        reasons.append("Malformed process evidence")
    else:
        for key in ("error", "cleanup_error", "timed_out", "interrupted", "orphaned_descendants",
                    "process_group_remaining"):
            if process.get(key):
                reasons.append("Process " + key + ": " + str(process[key]))
    return {"valid": not reasons, "invalid_reasons": reasons,
            "returncode": receipt.get("returncode"), "process_error": receipt.get("process_error"),
            "process": process}


def snapshot_setup_lifecycle(setup, output):
    """Capture completed service-management evidence without reading active logs.

    A service can be terminal with status 'started' while its policy server keeps
    running. Only its lifecycle JSON and optional diagnosis are copied; policy.log
    is deliberately excluded because the running service may still append to it.
    Legacy policy-switch receipts terminate at the recorded readiness observation.
    """
    setup, output = Path(setup), Path(output)
    records, sources, issues, pending = [], [], [], []
    paths = sorted(setup.glob("service-*/service.json")) + sorted(setup.glob("policy-switch-*.json"))
    for path in paths:
        relative = path.relative_to(setup)
        try:
            raw = path.read_bytes()
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise TypeError("Lifecycle record must be an object")
            legacy = path.name != "service.json"
            finished = value.get("observed_ready_unix_seconds" if legacy else "finished_at")
            started = value.get("started_unix_seconds" if legacy else "started_at")
            terminal = (finished is not None if legacy else value.get("status") in (
                "started", "reused", "failed"))
            if not terminal:
                pending.append(str(relative))
                continue
            if (type(started) not in (int, float) or type(finished) not in (int, float)
                    or not math.isfinite(started) or not math.isfinite(finished) or finished < started):
                raise ValueError("Terminal lifecycle lacks a valid start/finish interval")
            target = output / "inputs/setup" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(raw)
            source = _source(target)
            sources.append(source)
            record = {"path": str(relative), "family": value.get("family"),
                      "status": "ready_observed" if legacy else value["status"],
                      "started_at": started, "finished_at": finished,
                      "error": value.get("error"), "source": source}
            if not legacy and (path.parent / "diagnosis.json").is_file():
                _, diagnosis_source = _snapshot(path.parent / "diagnosis.json", target.parent / "diagnosis.json")
                sources.append(diagnosis_source)
                record["diagnosis_source"] = diagnosis_source
            records.append(record)
            if path.read_bytes() != raw:
                raise ValueError("Lifecycle record changed during report collection")
        except (OSError, ValueError, TypeError) as error:
            issues.append("Setup lifecycle " + str(relative) + ": " + str(error))
    return records, sources, issues, pending


def select_references(manifest, measured):
    """Require every declared full case before comparing the two fixed candidates."""
    required = manifest["full_jobs"]
    missing = [job["job_id"] for job in required if job["job_id"] not in measured
               or not measured[job["job_id"]].get("valid")]
    output = {"ready": False, "missing_or_invalid_full_jobs": missing,
              "fixed": None, "contextual": manifest["reference_selection"]["contextual"],
              "contextual_threshold_changed": False}
    if len(required) != 24 or missing:
        return output
    families = manifest["families"]
    groups = defaultdict(list)
    for job in required:
        row = measured[job["job_id"]]
        if (row.get("purpose") != "train" or not row.get("full_episode")
                or row.get("short_check") or row.get("family") != job["family"]
                or row.get("scene") != job["scene"] or row.get("seed") != job["seed"]
                or row.get("mode") != job["mode"]
                or row.get("fixed_choices") != job["fixed_choices"]):
            raise ValueError("Reference-selection evidence differs from its assigned training case")
        name = job["job_id"].rsplit("-", 1)[1]
        groups[name].append(row)
    metrics = {}
    for name, rows in groups.items():
        by_family = {family: [row for row in rows if row["family"] == family] for family in families}
        if any(len(values) != 2 for values in by_family.values()):
            raise ValueError("Every preparation reference needs two full training cases per family")
        value = {key: float(np.mean([np.mean([row["outcome"][key] for row in values])
                                    for values in by_family.values()])) for key in ("U", "V", "S_req")}
        queries = [row["costs"].get("policy_queries") for row in rows]
        if not all(_count(value) for value in queries):
            raise ValueError("Reference selection lacks measured policy-query counts")
        value["policy_queries"] = float(np.mean([np.mean([row["costs"]["policy_queries"]
                                                         for row in values])
                                                for values in by_family.values()]))
        metrics[name] = value
    candidates = manifest["reference_selection"]["fixed_candidates"]
    if set(candidates) != {"smooth075", "cap006"}:
        raise ValueError("Fixed reference candidate set changed")
    winner = min(candidates, key=lambda name: (-metrics[name]["U"], metrics[name]["V"],
                                              metrics[name]["policy_queries"], name))
    choices = {tuple(job["fixed_choices"]) for job in required if job["job_id"].endswith("-" + winner)}
    if len(choices) != 1:
        raise ValueError("Selected fixed reference has inconsistent choices")
    output.update(ready=True, metrics=metrics,
                  fixed={"name": winner, "choices": list(choices.pop()), "metrics": metrics[winner]},
                  tie_break_order=["higher family-macro U", "lower family-macro V",
                                   "lower family-macro attempted queries", "name"])
    return output


def activation_diagnostic(pilot, episodes):
    """Argmax activation on recorded engineering inputs; never fit or rank fitness.

    Reset-hidden comparisons isolate input dependence from recurrent-history changes.
    Mutations are diagnostic children of each initial actor, not selected offspring.
    """
    rows, vectors = [], []
    for episode in episodes:
        features = np.asarray(episode["features"], dtype=float)
        if features.ndim != 2 or features.shape[1] != 64 or not np.isfinite(features).all():
            raise ValueError("Activation audit requires finite recorded 64-feature rows")
        if len(features):
            vectors.extend(features)
    if not vectors:
        return {"valid": False, "passed": False, "reason": "No audited engineering features",
                "candidates": [], "physical_work": False}
    inputs = np.asarray(vectors)
    for seed in pilot["optimizer_seeds"]:
        search = WeightSearch("evolution", seed, config=SearchConfig(**pilot["search_config"]))
        initial = search.ask()
        if [candidate["sha256"] for candidate in initial] != pilot["initialization"][
                "shared_first_panel_sha256"][str(seed)]:
            raise ValueError("Initial controller hashes differ from the declared pilot")
        candidates = [(candidate["origin"], candidate["genome"], None) for candidate in initial]
        candidates += [("diagnostic_mutation", search._mutation(candidate["genome"]), candidate["sha256"])
                       for candidate in initial]
        for origin, genome, parent in candidates:
            logits = np.asarray([forward(genome, features)[0] for features in inputs])
            choices = np.asarray([decode(value) for value in logits])
            sequential = []
            for episode in episodes:
                hidden = None
                for features in episode["features"]:
                    value, hidden = forward(genome, features, hidden)
                    sequential.append(decode(value).tolist())
            rows.append({"optimizer_seed": seed, "origin": origin, "actor_sha256": genome_hash(genome),
                         "parent_sha256": parent, "zero_hidden_unique_choices": len(np.unique(choices, axis=0)),
                         "zero_hidden_nonidentity_fraction": float(np.any(choices != 0, axis=1).mean()),
                         "choices_depend_on_inputs": len(np.unique(choices, axis=0)) > 1,
                         "logits_depend_on_inputs": bool(np.max(np.abs(logits - logits[0])) > 1e-8),
                         "sequential_unique_choices": len({tuple(value) for value in sequential})})
    sampled = any(row["choices_depend_on_inputs"] for row in rows if row["origin"] == "random_initialization")
    mutated = any(row["choices_depend_on_inputs"] for row in rows if row["origin"] == "diagnostic_mutation")
    return {"valid": True, "passed": sampled and mutated, "recorded_feature_rows": len(inputs),
            "episodes": len(episodes), "sampled_choice_context_dependence": sampled,
            "mutated_choice_context_dependence": mutated, "candidates": rows,
            "physical_work": False, "controller_changes": False,
            "interpretation": "Input-dependent deterministic choices on the recorded engineering "
                              "feature range; this is not a performance or safety test."}


def throughput_estimate(pilot, terminal, reservation):
    action_envelope = Counter()
    for job in compile_schedule(pilot):
        action_envelope[job["family"]] += job["max_steps"]
    families, total = {}, 0.0
    complete = True
    for family, actions in action_envelope.items():
        eligible = [row for row in terminal if row["family"] == family and row["valid"]
                    and not row["baseline_executor"] and row.get("steps", 0) > 0]
        full = [row for row in eligible if not row["short_check"]]
        evidence = full or eligible
        ratio = max((row["elapsed_seconds"] / row["steps"] for row in evidence), default=None)
        families[family] = {"scheduled_actions_max": actions, "valid_full_samples": len(full),
                            "valid_short_samples": len(eligible) - len(full),
                            "slowest_observed_launcher_seconds_per_action": ratio,
                            "basis": "full" if full else "short_only" if evidence else "missing"}
        if ratio is None:
            complete = False
        else:
            total += ratio * actions
    future = total * reservation["reserved_gpus"] / 3600 if complete else None
    full_basis = all(row["valid_full_samples"] for row in families.values())
    return {"families": families, "future_pilot_gpu_hours_estimate": future,
            "prior_reservation_gpu_hours": reservation["allocated_gpu_seconds"] / 3600,
            "total_gpu_hours_estimate": (future + reservation["allocated_gpu_seconds"] / 3600
                                          if future is not None else None),
            "all_families_have_full_measurements": full_basis,
            "ready_for_freeze_review": complete and full_basis,
            "ceiling_changed": False, "pilot_gpu_hour_ceiling": pilot["resource_envelope"]["gpu_hours_ceiling"],
            "method": "Per-family slowest measured launcher seconds/action times the full scheduled "
                      "action envelope, plus elapsed reservation once. Prefer full episodes; short-only "
                      "startup amortization is provisional. This measured extrapolation is not a hard "
                      "runtime bound and does not estimate future idle or unmeasured service switches."}


def build_report(preparation, setup, output, *, pilot_path=None, timestamp=None):
    """Exclusively create a report directory and optional coordinator-ready prior.

    Only terminal receipts are behaviorally re-audited. Started receipts consume
    slots but their active traces/ledgers are never read. Every source for an emitted
    prior is an immutable copy made inside this report directory.
    """
    preparation, setup, output = Path(preparation).resolve(), Path(setup).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    captured, issues, sources = output / "inputs/preparation", [], []
    manifest, source = _snapshot(preparation / "preparation.json", captured / "preparation.json")
    sources.append(source)
    pilot, source = _snapshot(pilot_path or ROOT / "configs/maniguard/phase2/pilot.json",
                              output / "inputs/pilot.json")
    sources.append(source)
    setup_records = {}
    for name in ("reservation.json", "encoder-smoke.json", "encoder-loader-failure-v1.json"):
        try:
            setup_records[name], source = _snapshot(setup / name, output / "inputs/setup" / name)
            sources.append(source)
        except (OSError, ValueError) as error:
            issues.append(f"Setup {name}: {error}")
    lifecycle, lifecycle_sources, lifecycle_issues, lifecycle_pending = snapshot_setup_lifecycle(setup, output)
    sources.extend(lifecycle_sources)
    issues.extend(lifecycle_issues)
    if lifecycle_pending:
        issues.append("Nonterminal setup lifecycle records: " + ", ".join(lifecycle_pending))
    measured, terminal, features, counts, coverage, unknown, pending = {}, [], [], Counter(), Counter(), [], []
    seen = set()
    for path in sorted((preparation / "attempts").glob("*.json")):
        target = captured / "attempts" / path.name
        try:
            receipt, receipt_source = _snapshot(path, target)
            sources.append(receipt_source)
            job, episode = receipt["job"], Path(receipt["output"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            issues.append(f"Malformed receipt {path.name}: {error}")
            if target.is_file() and not any(value["path"] == str(target) for value in sources):
                sources.append(_source(target))
            unknown.extend({"job_id": path.stem, "count": key} for key in PRIOR_COUNTS)
            continue
        if str(episode.resolve()) in seen:
            issues.append("Duplicate physical output receipt: " + str(episode))
            continue
        seen.add(str(episode.resolve()))
        if receipt.get("status") not in ("valid", "invalid") or "returncode" not in receipt:
            pending.append(job["job_id"])
            unknown.extend({"job_id": job["job_id"], "count": key} for key in PRIOR_COUNTS)
            continue
        recorded_job = {}
        try:
            recorded_job, source = _snapshot(episode / "job.json", output / "inputs/jobs" / path.name)
            sources.append(source)
        except (OSError, ValueError) as error:
            issues.append("Missing terminal job evidence: " + str(error))
        baseline = job["baseline_executor"]
        row = {"valid": False, "costs": receipt.get("ledger", {}), "invalid_reasons": []}
        try:
            if baseline:
                row = read_original(episode)
                source_audit = audit_sources(episode)
                row.update(valid=source_audit["valid"], costs=row["ledger"], source_audit=source_audit,
                           invalid_reasons=source_audit.get("invalid_reasons", []))
            else:
                row = read_episode(episode)
            binding = ("family", "scene", "purpose", "seed", "sampling_seed", "mode",
                       "fixed_choices", "stochastic")
            if any(recorded_job.get(key) != job.get(key) for key in binding):
                raise ValueError("Terminal job does not match its frozen preparation receipt")
            if recorded_job.get("max_steps_override") != job["max_steps"]:
                raise ValueError("Terminal job horizon differs from the frozen preparation receipt")
            if hashlib.sha256(path.read_bytes()).hexdigest() != receipt_source["sha256"]:
                raise ValueError("Receipt changed during report collection")
        except (OSError, ValueError, KeyError, TypeError) as error:
            row["valid"] = False
            row.setdefault("invalid_reasons", []).append(str(error))
        process_audit = audit_process(receipt)
        row["process_audit"] = process_audit
        if not process_audit["valid"]:
            row["valid"] = False
            row.setdefault("invalid_reasons", []).extend(process_audit["invalid_reasons"])
        _write(output / "audited_episodes" / path.name, row)
        sources.append(_source(output / "audited_episodes" / path.name))
        measured[job["job_id"]] = row
        terminal.append({"job_id": job["job_id"], "family": job["family"], "valid": row["valid"],
                         "receipt_status": receipt["status"], "audit_history": receipt.get("audit_history", []),
                         "process_audit": process_audit,
                         "short_check": receipt["short_check"], "baseline_executor": baseline,
                         "steps": row.get("steps"), "elapsed_seconds": receipt.get("elapsed_seconds"),
                         "invalid_reasons": row.get("invalid_reasons", [])})
        ledger_complete = row.get("ledger", receipt.get("ledger", {})).get("phase") == "completed"
        for key in PRIOR_COUNTS:
            value = row.get("costs", {}).get(key)
            if key == "actor_rpc_attempts":
                value = row.get("costs", {}).get(key, 0)  # Local actor; VLA RPCs are policy_queries.
            if baseline and recorded_job.get("phase") == "phase2_baseline_check" and key in (
                    "encoder_image_calls", "encoder_images", "encoder_text_calls", "adapter_decisions"):
                value = 0  # The original executor does not instantiate these phase-two components.
            if _count(value):
                counts[key] += value
                coverage[key] += 1
                baseline_absent_component = baseline and key in (
                    "encoder_image_calls", "encoder_images", "encoder_text_calls", "adapter_decisions")
                if not ledger_complete and key != "actor_rpc_attempts" and not baseline_absent_component:
                    unknown.append({"job_id": job["job_id"], "count": key,
                                    "reason": "incomplete ledger is a lower bound"})
                if row.get("cost_provenance", {}).get(key) == "trace_lower_bound":
                    unknown.append({"job_id": job["job_id"], "count": key, "reason": "trace lower bound"})
            else:
                unknown.append({"job_id": job["job_id"], "count": key})
        if row["valid"] and job["purpose"] == "engineering" and not baseline:
            features.append({"job_id": job["job_id"], "features": [decision["features"]
                                                                     for decision in row["decisions"]]})
    accounting = audit_preparation(captured, receipt_ledgers_only=True)
    issues.extend(accounting["invalid_reasons"])
    for path in preparation.glob("*/job.json"):
        if str(path.parent.resolve()) not in seen:
            issues.append("Unreceipted physical output: " + str(path.parent))
    setup_counts = Counter()
    smoke = setup_records.get("encoder-smoke.json")
    failure = setup_records.get("encoder-loader-failure-v1.json")
    if smoke and failure:
        if smoke.get("finite") is not True or smoke.get("trainable_parameters") != 0:
            issues.append("Encoder smoke lacks finite/frozen validation")
        for key in PRIOR_COUNTS:
            value = smoke.get("costs", {}).get(key, 0)
            failed = failure.get("counts", {}).get(key)
            if not _count(value) or not _count(failed):
                unknown.append({"job_id": "setup", "count": key})
            else:
                setup_counts[key] += value + failed
        if {key: smoke.get("costs", {}).get(key) for key in ("encoder_image_calls", "encoder_images",
                "encoder_text_calls")} != {"encoder_image_calls": 5, "encoder_images": 10, "encoder_text_calls": 1}:
            issues.append("Encoder smoke interactions differ from the recorded five-batch diagnostic")
    else:
        unknown.extend({"job_id": "setup", "count": key} for key in PRIOR_COUNTS)
    all_jobs = manifest["short_jobs"] + manifest["full_jobs"]
    absent = [job["job_id"] for job in all_jobs if job["job_id"] not in measured]
    complete = not absent and not pending and len(terminal) == len(all_jobs)
    activation = activation_diagnostic(pilot, features)
    selection = select_references(manifest, measured)
    stamp = time.time() if timestamp is None else timestamp
    allocation = setup_records.get("reservation.json", {})
    start, gpus = allocation.get("start_unix_seconds"), allocation.get("allocated_gpus")
    if (type(stamp) not in (int, float) or not math.isfinite(stamp) or type(start) not in (int, float)
            or stamp < start or type(gpus) is not int or gpus < 1):
        raise ValueError("Missing or invalid GPU reservation interval")
    reservation = {"timestamp": stamp, "reservation_started_at": start, "reserved_gpus": gpus,
                   "wall_seconds": stamp - start, "allocated_gpu_seconds": gpus * (stamp - start)}
    estimates = throughput_estimate(pilot, terminal, reservation)
    total_counts = {key: counts[key] + setup_counts[key] for key in PRIOR_COUNTS}
    prior_ready = complete and not unknown and not issues
    report = {"schema": "maniguard-phase2-preparation-report-v1", "timestamp": stamp,
              "preparation_complete": complete, "training_ready": False,
              "ready_for_freeze_review": (prior_ready and all(row["valid"] for row in terminal)
                                          and activation["passed"] and selection["ready"]),
              "prior_accounting_ready": prior_ready, "invalid_reasons": issues,
              "absent_or_nonterminal_jobs": absent, "pending_receipts": pending, "terminal_attempts": terminal,
              "accounting": accounting, "reservation": reservation,
              "episode_counts_known": dict(counts), "episode_count_coverage": dict(coverage),
              "setup_counts_known": dict(setup_counts), "total_counts_known": total_counts,
              "setup_lifecycle": lifecycle, "pending_setup_lifecycle": lifecycle_pending,
              "unknown_counts": unknown, "count_totals_are_lower_bounds": bool(unknown or pending),
              "reference_selection": selection,
              "activation": activation, "throughput": estimates,
              "limitations": ["A report does not freeze the pilot or authorize training.",
                              "GPU reservation time includes setup and idle once; episode durations are not added.",
                              "Failed loader wall time is unknown individually but included in the reservation.",
                              ("Service lifecycle records retain setup failures and diagnoses; they do not "
                               "supply missing inference counts or historical process tracebacks."),
                              "Short checks cannot establish full-task success or safety performance."],
              "sources": sources}
    _write(output / "engineering-features.json", features)
    sources.append(_source(output / "engineering-features.json"))
    code_paths = list((ROOT / "experiments/helpers").glob("maniguard_phase2_*.py"))
    code_paths.extend(ROOT / "experiments/scripts" / name for name in (
        "maniguard_phase2_preparation_report.py", "maniguard_phase2_prepare.py"))
    for path in code_paths:
        target = output / "audit_source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(path.read_bytes())
        sources.append(_source(target))
    report["sha256"] = record_hash(report)
    _write(output / "report.json", report)
    if prior_ready:
        prior = {"schema": PRIOR_SCHEMA, **reservation, "counts": total_counts,
                 "sources": sources + [_source(output / "report.json")]}
        prior["sha256"] = record_hash(prior)
        _write(output / "prior-accounting.json", prior)
    return report
