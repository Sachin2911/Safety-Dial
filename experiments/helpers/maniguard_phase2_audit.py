"""Read-only preparation, source-integrity and observed identity checks.

Short checks establish implementation evidence only. Matching nominal seeds do
not establish matching simulator starts; the identity comparison requires exact
recorded initial state and both RGB arrays before comparing trajectories.
"""

import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PREFIX = "configs/maniguard/phase2/"
REQUIRED_CONFIGS = ("encoder.json", "sources.json", "families.json", "completion.json")


def _json(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def audit_sources(output, *, root=ROOT, verify_upstream=True):
    """Check saved bytes against job hashes, without importing archived code.

    Current workspace changes are reported separately and do not invalidate old
    snapshots. Hashes establish internal integrity, not external authentication or
    proof that every imported live module matched its snapshot during execution.
    The generated evaluator's actual bytes are checked against runtime provenance.
    """
    output, root = Path(output), Path(root)
    issues, changed, hashes, configs = [], [], {}, {}
    runtime = None
    try:
        job = _json(output / "job.json")
        declared = job.get("code_hashes")
        if not isinstance(declared, dict) or not declared:
            raise ValueError("Missing archived study-source hashes")
        archive = (output / "study_source").resolve()
        for relative, expected in declared.items():
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Source manifest contains a nonrelative path")
            saved = (archive / path).resolve()
            if not saved.is_relative_to(archive):
                raise ValueError("Archived source resolves outside study_source")
            actual = _sha(saved)
            if actual != expected:
                issues.append("Archived source checksum mismatch: " + relative)
            hashes[relative] = actual
            live = root / path
            if not live.is_file() or _sha(live) != actual:
                changed.append(relative)
        for name in REQUIRED_CONFIGS:
            relative = CONFIG_PREFIX + name
            if relative not in hashes:
                issues.append("Missing required archived configuration: " + relative)
            else:
                configs[name.removesuffix(".json")] = _json(archive / relative)
        runtime = _json(output / "runtime-provenance.json")
        if _sha(output / "benchmark_instrumented.py") != runtime.get("runtime_sha256"):
            issues.append("Generated evaluator checksum mismatch")
        pinned = configs.get("sources", {}).get("maniguard", {}).get("revision")
        if not pinned or runtime.get("revision") != pinned:
            issues.append("Generated evaluator revision differs from archived source pin")
        if verify_upstream and pinned:
            upstream = root / "third_party/ManiGuard"
            source = subprocess.check_output(
                ["git", "-C", str(upstream), "show", pinned + ":maniguard/eval/benchmark.py"])
            if hashlib.sha256(source).hexdigest() != runtime.get("original_sha256"):
                issues.append("Original evaluator hash differs from pinned Git source")
    except (OSError, ValueError, TypeError, KeyError, subprocess.CalledProcessError) as error:
        issues.append(str(error))
    return {"valid": not issues, "invalid_reasons": issues, "snapshot_sha256": hashes,
            "configs": configs, "runtime": runtime, "workspace_differences": changed,
            "loaded_module_attestation": False,
            "limitation": "Snapshots and generated-source bytes are verified; historical live "
                          "imports were not independently attested."}


def _trace(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _array_hash(value):
    value = np.ascontiguousarray(value)
    return {"sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            "shape": list(value.shape), "dtype": str(value.dtype)}


def _initial(output, trace):
    records = [row for row in trace if row.get("type") == "initial"]
    queries = [row for row in trace if row.get("type") == "query" and row.get("step") == 0]
    if len(records) != 1 or len(queries) != 1:
        raise ValueError("Initial observation or first policy query is absent/duplicated")
    with np.load(output / "initial_observation.npz", allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in ("state", "overview", "wrist")}
    for name, array in arrays.items():
        if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
            raise ValueError("Initial observation must contain finite numeric arrays")
        if name != "state" and (array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3):
            raise ValueError("Initial images must retain original uint8 RGB pixels")
    if (not np.array_equal(arrays["state"], records[0].get("state"))
            or not np.array_equal(arrays["state"], queries[0].get("state"))):
        raise ValueError("Initial state archive and recorded query state disagree")
    hashes = {name: _array_hash(value) for name, value in arrays.items()}
    for name, key in (("overview", "overview_image"), ("wrist", "wrist_images")):
        if queries[0].get("image_hashes", {}).get(key) != hashes[name]["sha256"]:
            raise ValueError("Initial image archive and first query hash disagree")
    return hashes


def _monitor_source(root, revision):
    files = ("maniguard/utils/safety_monitor.py", "maniguard/utils/ltl_utils.py")
    hashes = {}
    for relative in files:
        raw = subprocess.check_output(
            ["git", "-C", str(root / "third_party/ManiGuard"), "show", revision + ":" + relative])
        expected = hashlib.sha256(raw).hexdigest()
        if _sha(root / "third_party/ManiGuard" / relative) != expected:
            raise ValueError("Current monitor source differs from pinned Git bytes: " + relative)
        hashes[relative] = expected
    return hashes


def compare_identity(original, identity, *, root=ROOT, verify_upstream=True):
    """Compare completed original/phase-two identity engineering checks.

    ``trajectory_comparison`` stays None when recorded starts differ. Even exact
    recorded starts do not certify unobserved simulator internals; claims are
    confined to the recorded trajectory. No physical work is initiated.
    """
    from helpers.maniguard_phase2_model import genome_hash, identity_genome, load_model
    from helpers.maniguard_phase2_results import read_episode
    from helpers.maniguard_results import read_episode as read_original

    paths = [Path(original).resolve(), Path(identity).resolve()]
    report = {"original": str(paths[0]), "identity": str(paths[1]), "valid": False,
              "invalid_reasons": [], "status": "invalid_evidence", "initial": None,
              "recorded_starts_match": None, "trajectory_comparison": None,
              "performance_claim": False,
              "claim_scope": "Short engineering checks; no full-task performance inference."}
    try:
        jobs = [_json(path / "job.json") for path in paths]
        if jobs[0].get("phase") != "phase2_baseline_check" or jobs[1].get("phase") != "phase2":
            raise ValueError("Expected original-executor then phase-two identity checks")
        if (jobs[0].get("adapter", {}).get("kind") != "identity"
                or jobs[0]["adapter"].get("horizon") != 8):
            raise ValueError("Original executor must use the eight-action identity adapter")
        for job, path in zip(jobs, paths, strict=True):
            if (job.get("purpose") != "engineering" or job.get("stochastic") is not False
                    or job.get("mode") != "learned" or job.get("fixed_choices") is not None
                    or type(job.get("max_steps_override")) is not int
                    or not 1 <= job["max_steps_override"] <= 32):
                raise ValueError("Identity comparison requires deterministic short engineering jobs")
            if (job.get("actor_sha256") != genome_hash(identity_genome())
                    or genome_hash(load_model(path / "actor.npz")[0]) != job["actor_sha256"]):
                raise ValueError("Identity comparison actor is not the declared identity genome")
        keys = ("family", "scene", "seed", "sampling_seed", "policy", "max_steps_override")
        if any(jobs[0].get(key) != jobs[1].get(key) for key in keys):
            raise ValueError("Identity comparison jobs have different nominal conditions")
        sources = [audit_sources(path, root=root, verify_upstream=verify_upstream) for path in paths]
        report["sources"] = sources
        if not all(source["valid"] for source in sources):
            raise ValueError("Identity check has invalid archived source evidence")
        provenance = [source["runtime"] for source in sources]
        if (provenance[0]["original_sha256"] != provenance[1]["original_sha256"]
                or provenance[0]["runtime_sha256"] != provenance[1].get("phase1_runtime_sha256")):
            raise ValueError("Identity executor does not derive from the same original evaluator")
        left_source = (paths[0] / "benchmark_instrumented.py").read_text()
        expected_source = left_source.replace("from helpers.maniguard_runtime import STUDY",
                                             "from helpers.maniguard_phase2_runtime import STUDY")
        expected_source = expected_source.replace("STUDY.start(scene_info, obs)",
                                                 "STUDY.start(scene_info, obs, env, robot, monitor)")
        if expected_source != (paths[1] / "benchmark_instrumented.py").read_text():
            raise ValueError("Generated phase-two evaluator differs beyond the two declared hooks")
        if verify_upstream:
            report["current_monitor_source_sha256"] = _monitor_source(
                Path(root), provenance[0]["revision"])
        original_row, identity_row = read_original(paths[0]), read_episode(paths[1])
        if not identity_row["valid"]:
            raise ValueError("Identity episode audit failed: " + "; ".join(identity_row["invalid_reasons"]))
        report["episode_audits"] = {"original": original_row, "identity": identity_row}
        monitors = [_json(path / "episode" / (job["scene"] + "_ltl.json"))
                    for path, job in zip(paths, jobs, strict=True)]
        if any(monitors[0].get(key) != monitors[1].get(key) for key in ("formula", "constraints")):
            raise ValueError("Monitor definitions differ between identity checks")
        report["monitor_definition_equal"] = True
        traces = [_trace(path / "trace.jsonl") for path in paths]
        report["initial"] = [_initial(path, trace) for path, trace in zip(paths, traces, strict=True)]
        starts_match = report["initial"][0] == report["initial"][1]
        report["recorded_starts_match"] = starts_match
        schedules, transformations = [], []
        for trace, episode in zip(traces, (original_row, identity_row), strict=True):
            count = episode["steps"]
            for kind, field in (("proposal", "original"), ("executed", "action")):
                action_rows = [row for row in trace if row.get("type") == kind]
                if [row.get("step") for row in action_rows] != list(range(count)):
                    raise ValueError("Missing, duplicate or unordered identity " + kind + " records")
                for row in action_rows:
                    value = np.asarray(row.get(field))
                    if value.shape != (8,) or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                        raise ValueError("Invalid identity action vector")
            schedule = [row["step"] for row in trace if row.get("type") == "query"]
            expected = list(range(0, count, 8))
            if episode["ledger"].get("policy_queries") != len(schedule):
                raise ValueError("Identity query trace differs from cost ledger")
            schedules.append({"actual": schedule, "expected": expected, "equal": schedule == expected})
            proposals = [row for row in trace if row.get("type") == "proposal"]
            exact = len(proposals) == count and all(np.array_equal(row["original"], row["adapted"])
                                                   for row in proposals)
            transformations.append(exact)
        report.update(query_schedules=schedules, exact_identity_transformation=transformations)
        if not all(row["equal"] for row in schedules) or not all(transformations):
            raise ValueError("Identity action transformation or eight-action query schedule changed")
        if starts_match:
            comparison = {}
            for kind, field in (("query", "state"), ("proposal", "original"),
                                ("executed", "action"), ("observed", "state")):
                rows = [[row for row in trace if row.get("type") == kind] for trace in traces]
                comparison[kind] = (len(rows[0]) == len(rows[1]) and all(
                    left.get("step") == right.get("step")
                    and np.array_equal(left.get(field), right.get(field))
                    for left, right in zip(*rows, strict=True)))
            # The original hook lacks a step-zero observed event; compare common post-action states.
            observed = [[row for row in trace if row.get("type") == "observed" and row["step"] > 0]
                        for trace in traces]
            comparison["observed"] = len(observed[0]) == len(observed[1]) and all(
                left["step"] == right["step"] and np.array_equal(left["state"], right["state"])
                for left, right in zip(*observed, strict=True))
            report["trajectory_comparison"] = comparison
            report["status"] = ("recorded_trajectory_equal" if all(comparison.values())
                                else "recorded_trajectory_differs")
        else:
            report["status"] = "not_comparable_different_recorded_starts"
        report["valid"] = True
    except (OSError, ValueError, TypeError, KeyError, subprocess.CalledProcessError) as error:
        report["invalid_reasons"].append(str(error))
    return report


def audit_preparation(directory, *, receipt_ledgers_only=False, allocation=None):
    """Account for every receipt, including failures, pending jobs and orphan outputs.

    With receipt_ledgers_only=True, use captured receipt ledgers without opening
    episode ledgers that may still be changing. An optional validated effective
    allocation binds a prospective amendment while retaining the original manifest
    and failed receipts. It changes assignment validation, never raw cost evidence.
    Receipt elapsed time is launcher wall time; pending attempts reserve their full
    timeout. These process totals exclude policy startup, diagnostics and idle GPU
    reservation, which must be reported separately before claiming total GPU hours.
    """
    from helpers.maniguard_phase2_results import COST_FIELDS, OPTIONAL_COUNT_COSTS

    directory = Path(directory).resolve()
    manifest = _json(directory / "preparation.json")
    issues = []
    raw = {key: value for key, value in manifest.items() if key != "sha256"}
    if _digest(raw) != manifest.get("sha256"):
        issues.append("Preparation manifest checksum mismatch")
    if allocation is not None:
        if (allocation.get("base_preparation_sha256") != manifest.get("sha256")
                or allocation.get("sha256") != _digest({key: value for key, value in allocation.items()
                                                        if key != "sha256"})
                or allocation.get("budget") != manifest["budget"]):
            raise ValueError("Effective preparation allocation is unbound or changes the budget")
        from helpers.maniguard_phase2_protocol import validate_preparation_receipt
        for job_id, expected in allocation.get("pre_recovery_receipts", {}).items():
            path = directory / "attempts" / (job_id + ".json")
            try:
                if _digest(_json(path)) != expected:
                    issues.append("Original paid attempt inventory changed: " + job_id)
            except (OSError, ValueError, TypeError):
                issues.append("Original paid attempt inventory is missing or malformed: " + job_id)
    declared = {job["job_id"]: job for job in manifest["short_jobs"] + manifest["full_jobs"]}
    attempts, seen_outputs = [], set()
    for path in sorted((directory / "attempts").glob("*.json")):
        try:
            receipt = _json(path)
            job = receipt["job"]
            output = Path(receipt["output"]).resolve()
            if allocation is not None:
                try:
                    validate_preparation_receipt(receipt, allocation)
                except (ValueError, TypeError, KeyError) as error:
                    issues.append("Receipt differs from effective assignment: " + path.name + ": " + str(error))
            elif receipt.get("protocol_sha256") != manifest["sha256"] or declared.get(job["job_id"]) != job:
                issues.append("Receipt differs from frozen assignment: " + path.name)
            short = job.get("max_steps") is not None
            if (type(receipt.get("short_check")) is not bool or receipt["short_check"] != short
                    or (short and (type(job["max_steps"]) is not int
                        or not 1 <= job["max_steps"] <= manifest["budget"].get("short_actions_per_attempt", 32)))):
                issues.append("Receipt has an invalid short/full horizon: " + path.name)
            duplicate = str(output) in seen_outputs
            if duplicate:
                issues.append("Duplicate output receipts: " + str(output))
            seen_outputs.add(str(output))
            attempts.append({"receipt": str(path), "job_id": job["job_id"], "output": str(output),
                             "short_check": receipt.get("short_check"),
                             "status": receipt.get("status"), "job": job, "duplicate": duplicate,
                             "audit_history": receipt.get("audit_history", []),
                             "initial_status": (receipt.get("audit_history") or [receipt])[0].get("status"),
                             "elapsed_seconds": receipt.get("elapsed_seconds"),
                             "timeout_seconds": receipt.get("timeout_seconds"),
                             "ledger": (_json(output / "ledger.json") if not receipt_ledgers_only
                                        and (output / "ledger.json").is_file()
                                        else receipt.get("ledger", {}))})
        except (OSError, ValueError, TypeError, KeyError) as error:
            issues.append(f"Malformed receipt {path.name}: {error}")
            attempts.append({"receipt": str(path), "status": "malformed", "short_check": None,
                             "duplicate": False, "ledger": {}, "job": {}})
    for path in sorted(directory.glob("*/job.json")):
        if str(path.parent.resolve()) not in seen_outputs:
            job = _json(path)
            issues.append("Episode output has no attempt receipt: " + str(path.parent))
            attempts.append({"receipt": None, "job_id": path.parent.name, "output": str(path.parent),
                             "short_check": job.get("max_steps_override") is not None,
                             "status": "orphan", "job": {"max_steps": job.get("max_steps_override"),
                             "full_horizon": job.get("action_horizon")}, "duplicate": False,
                             "ledger": (_json(path.parent / "ledger.json")
                                        if (path.parent / "ledger.json").is_file() else {})})
    costs, coverage, unknown_cost_attempts, categories = Counter(), Counter(), [], {}
    numeric = lambda value: type(value) in (int, float) and math.isfinite(value) and value >= 0
    budget = manifest["budget"]
    for short, category in ((True, "short"), (False, "full")):
        selected = [row for row in attempts if row["short_check"] is short]
        ceiling = budget[category + "_attempts"]
        elapsed = sum(row["elapsed_seconds"] for row in selected if numeric(row.get("elapsed_seconds")))
        pending = [row for row in selected if not numeric(row.get("elapsed_seconds"))]
        reserved = sum(row["timeout_seconds"] for row in pending if numeric(row.get("timeout_seconds")))
        if any(not numeric(row.get("timeout_seconds")) for row in pending):
            issues.append("Missing timeout exposure in " + category + " attempts")
        time_limit = 3600 * budget["integration_gpu_hour_ceiling" if short
                                  else "full_preparation_gpu_hour_ceiling"]
        categories[category] = {"attempts": len(selected), "ceiling": ceiling,
                                "remaining_slots": ceiling - len(selected),
                                "statuses": dict(Counter(row["status"] for row in selected)),
                                "launcher_seconds_measured": elapsed,
                                "pending_timeout_seconds_reserved": reserved,
                                "process_seconds_ceiling": time_limit,
                                "within_ceiling": len(selected) <= ceiling and elapsed + reserved <= time_limit}
        if not categories[category]["within_ceiling"]:
            issues.append("Preparation " + category + " ceiling exceeded")
    for row in attempts:
        if row["duplicate"]:
            continue
        ledger = row["ledger"]
        for key, value in ledger.items():
            if key not in COST_FIELDS + OPTIONAL_COUNT_COSTS:
                continue
            if numeric(value):
                costs[key] += value
                coverage[key] += 1
            else:
                issues.append("Malformed measured cost " + key + " in " + str(row.get("job_id")))
        if not ledger or ledger.get("phase") != "completed":
            unknown_cost_attempts.append(row.get("job_id", row["receipt"]))
    assigned_actions = sum((row["job"].get("max_steps") or row["job"].get("full_horizon") or 0)
                           for row in attempts)
    action_ceiling = budget["policy_action_ceiling"]
    if assigned_actions > action_ceiling or costs.get("policy_control_steps", 0) > action_ceiling:
        issues.append("Preparation policy-action ceiling exceeded")
    return {"valid": not issues, "invalid_reasons": issues, "attempts": attempts,
            "base_preparation_sha256": manifest["sha256"],
            "amendment_sha256": allocation.get("amendment_sha256") if allocation is not None else None,
            "effective_allocation_sha256": allocation.get("sha256") if allocation is not None else None,
            "attempt_count": len(attempts), "ceilings": categories,
            "unclassified_attempts": sum(row["short_check"] is None for row in attempts),
            "assigned_action_exposure": assigned_actions, "policy_action_ceiling": action_ceiling,
            "measured_cost_totals": dict(costs), "cost_record_counts": dict(coverage),
            "optional_cost_missing_records": {key: sum(not row["duplicate"] for row in attempts)
                                               - coverage[key] for key in OPTIONAL_COUNT_COSTS},
            "incomplete_cost_attempts": unknown_cost_attempts,
            "performance_claim": False,
            "within_all_declared_ceilings": (all(value["within_ceiling"] for value in categories.values())
                and assigned_actions <= action_ceiling
                and costs.get("policy_control_steps", 0) <= action_ceiling),
            "external_gpu_costs": "Policy startup, diagnostics and idle reservation are not in "
                                  "episode receipts; total allocated GPU hours remain unestablished."}
