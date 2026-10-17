"""Audit immutable phase-two episodes, retaining costs of invalid and partial work.

``read_episode(path)`` never turns absent outcomes into task failures. Its ``valid``
flag covers raw monitor, execution, actor replay and recorded-cost consistency.
``full_episode`` distinguishes the configured full horizon (early success allowed)
from short engineering checks. Only valid full episodes may supply optimizer fitness.
``trajectory`` uses the recurrent PPO collection schema; reference likelihoods are
None and ``ppo_eligible`` is false for references, deterministic or short episodes.
"""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from helpers.maniguard_phase2_audit import audit_sources
from helpers.maniguard_phase2_model import decode, forward, genome_hash, load_model, log_prob
from helpers.maniguard_phase2_outcomes import derive_outcome
from helpers.maniguard_results import read_episode as read_legacy_episode

COUNT_COSTS = ("physics_steps", "env_step_attempts", "env_steps_returned", "policy_queries",
               "policy_control_steps", "encoder_image_calls", "encoder_images",
               "encoder_text_calls", "adapter_decisions", "guard_steps", "changed_steps")
TIME_COSTS = ("physics_seconds", "policy_query_seconds", "elapsed_seconds", "encoder_seconds",
              "encoder_load_seconds", "adapter_seconds")
COST_FIELDS = COUNT_COSTS + TIME_COSTS
OPTIONAL_COUNT_COSTS = ("encoder_image_attempts", "encoder_text_attempts")
ROOT = Path(__file__).resolve().parents[2]
METRICS = ("C_req", "S_req", "V", "U", "benchmark_success", "benchmark_safe_success")


def _json(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    return json.loads(path.read_text(), parse_constant=reject)


def _jsonl(path, issues):
    if not path.is_file():
        issues.append(f"Missing evidence file: {path.name}")
        return []
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError("Expected a JSON object")
            rows.append(row)
        except (ValueError, TypeError) as error:
            issues.append(f"Invalid {path.name} line {line_number}: {error}")
    return rows


def _numeric(value, name, *, count=False):
    if (type(value) not in (int, float) or not math.isfinite(value) or value < 0
            or (count and type(value) is not int)):
        raise ValueError(f"Invalid recorded cost: {name}")
    return value


def _array(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"Invalid finite numeric {name}, expected shape {shape}")
    return array


def _audit_costs(ledger, trace):
    costs, provenance, issues = {}, {}, []
    missing = [key for key in COST_FIELDS if key not in ledger]
    for key in COST_FIELDS + OPTIONAL_COUNT_COSTS:
        if key in ledger:
            try:
                costs[key] = _numeric(ledger[key], key, count=key in COUNT_COSTS + OPTIONAL_COUNT_COSTS)
                provenance[key] = "ledger"
            except ValueError as error:
                issues.append(str(error))
    for attempted, completed in (("encoder_image_attempts", "encoder_image_calls"),
                                 ("encoder_text_attempts", "encoder_text_calls")):
        if attempted in costs and completed in costs and costs[attempted] < costs[completed]:
            issues.append("Encoder attempts below completed calls: " + attempted)
    events = Counter(row.get("type") for row in trace)
    observed = {"policy_control_steps": events["executed"], "policy_queries": events["query"],
                "adapter_decisions": events["decision"],
                "encoder_image_calls": events["decision"], "encoder_images": 2 * events["decision"]}
    for key, lower_bound in observed.items():
        if lower_bound and (key not in costs or costs[key] < lower_bound):
            issues.append(f"Ledger {key} is missing or below trace-observed work")
            costs[key] = lower_bound
            provenance[key] = "trace_lower_bound"
    if ledger.get("physics_accounting_attached") is not True:
        issues.append("Physics accounting attachment is unverified; zero cannot establish no work")
    if ledger.get("phase") != "completed":
        issues.append("Incomplete execution: persisted costs are lower bounds on failed work")
    issues.extend("Missing cost field: " + key for key in missing)
    return costs, provenance, issues, missing, observed


def _steps(rows, expected, name):
    if any(type(row.get("step")) is not int for row in rows):
        raise ValueError(f"Invalid {name} step type")
    if [row["step"] for row in rows] != list(expected):
        raise ValueError(f"Missing, duplicate or unordered {name} steps")



def _audit_causality(job, result, trace):
    for name, key, expected in (("job", "job", job), ("result", "result", result)):
        records = [row for row in trace if row.get("type") == name]
        if len(records) != 1 or records[0].get(key) != expected:
            raise ValueError(f"Trace {name} does not match its authoritative file")
    metadata = [row.get("metadata", {}) for row in trace if row.get("type") == "policy_metadata"]
    policy = job["policy"]
    expected = {"serve_config": policy["config"], "checkpoint_revision": policy["revision"],
                "episode_seed_reset": "study_episode_id"}
    if len(metadata) != 1 or any(metadata[0].get(key) != value for key, value in expected.items()):
        raise ValueError("Frozen policy handshake is absent or inconsistent")
    observed_step, executed_step, proposed_step = -1, -1, -1
    for record in trace:
        kind, step = record.get("type"), record.get("step")
        if kind == "observed":
            if step != executed_step + 1:
                raise ValueError("Observation precedes its executed action")
            observed_step = step
        elif kind in ("query", "decision", "proposal", "evaluation_only", "goal"):
            if step != observed_step:
                raise ValueError(f"{kind} did not use the current causal observation")
            if kind == "proposal":
                proposed_step = step
        elif kind == "executed":
            if step != proposed_step or step != observed_step or step != executed_step + 1:
                raise ValueError("Executed action lacks its ordered observation/proposal")
            executed_step = step


def _audit_decisions(job, trace, output, steps, encoder):
    genome, spec, _ = load_model(output / "actor.npz")
    if spec.input_dim != 64 or spec.hidden_dim != 8 or genome_hash(genome) != job["actor_sha256"]:
        raise ValueError("Archived actor schema/hash differs from the episode job")
    decisions = [row for row in trace if row.get("type") == "decision"]
    used_steps = list(range(0, steps, 4))
    expected = used_steps.copy()
    if decisions and decisions[-1].get("step") == steps and steps % 4 == 0:
        expected.append(steps)  # A final unused decision still incurs encoder/adapter costs.
    _steps(decisions, expected, "decision")
    features_records = [row for row in trace if row.get("type") == "phase2_features"]
    if len(features_records) != 1:
        raise ValueError("Expected one episode feature-contract record")
    contract = features_records[0]
    if (contract.get("feature_dim") != 64 or contract.get("information_access") != "observation_only"
            or contract.get("actor_sha256") != job["actor_sha256"]):
        raise ValueError("Feature information-access contract differs from the primary track")
    if (contract.get("encoder") != encoder or contract.get("robot_features") != 40
            or contract.get("vision_features") != 16 or contract.get("text_features") != 8):
        raise ValueError("Encoder provenance or feature partition differs from the frozen contract")
    encoder_sha = encoder["checkpoint_sha256"]
    hidden = np.zeros(spec.hidden_dim)
    rng = np.random.default_rng(job["sampling_seed"]) if job["stochastic"] else None
    for record in decisions:
        step = record["step"]
        if (record.get("observation_step") != step or record.get("mode") != job["mode"]
                or record.get("stochastic") is not job["stochastic"]
                or record.get("actor_sha256") != job["actor_sha256"]):
            raise ValueError("Decision metadata or causal observation timestamp mismatch")
        features = _array(record.get("features"), (64,), "features")
        if np.max(np.abs(features)) > 1 + 1e-7:
            raise ValueError("Features exceed the declared normalized input range")
        choices = _array(record.get("choices"), (4,), "categorical choices")
        if choices.dtype.kind not in "iu" or np.any(choices < 0) or np.any(choices > 2):
            raise ValueError("Decision choices must be integer indices 0 through 2")
        before = _array(record.get("hidden_before"), (8,), "hidden_before")
        if not np.allclose(before, hidden, rtol=0, atol=1e-10):
            raise ValueError("Recurrent history did not reset or continue correctly")
        if job["mode"] == "learned":
            logits, hidden = forward(genome, features, hidden)
            if not np.allclose(record.get("logits"), logits, rtol=0, atol=1e-8):
                raise ValueError("Recorded logits differ from archived actor replay")
            recorded_log_prob = record.get("log_prob")
            if (type(recorded_log_prob) not in (int, float) or not math.isfinite(recorded_log_prob)
                    or abs(recorded_log_prob - log_prob(logits, choices)) > 1e-6):
                raise ValueError("Recorded joint log probability differs from actor replay")
            expected_choices = decode(logits, rng)
        else:
            if record.get("log_prob") is not None or record.get("logits") is not None:
                raise ValueError("Reference controllers must not claim learned-policy likelihoods")
            expected_choices = (np.asarray(job.get("fixed_choices") or [0, 0, 0, 0])
                                if job["mode"] == "fixed" else
                                np.array([int(np.max(np.abs(features[12:19])) > 0.5), 0, 0, 0]))
        if not np.array_equal(choices, expected_choices):
            raise ValueError("Recorded choices differ from declared decoding/reference rule")
        after = _array(record.get("hidden_after"), (8,), "hidden_after")
        if not np.allclose(after, hidden, rtol=0, atol=1e-10):
            raise ValueError("Recorded next hidden state differs from actor replay")
        context = record.get("context", {})
        hashes = context.get("image_hashes")
        if (not isinstance(hashes, list) or len(hashes) != 2
                or any(not isinstance(value, str) or len(value) != 64 for value in hashes)
                or not isinstance(context.get("text_sha256"), str)
                or len(context["text_sha256"]) != 64 or not encoder_sha
                or context.get("checkpoint_sha256") != encoder_sha):
            raise ValueError("Decision image/text/encoder provenance is incomplete")
    footers = [row for row in trace if row.get("type") == "phase2_actor"]
    if (len(footers) != 1 or footers[0].get("decisions") != len(decisions)
            or footers[0].get("actor_sha256") != job["actor_sha256"]
            or footers[0].get("stochastic") is not job["stochastic"]):
        raise ValueError("Episode actor footer is absent or inconsistent")
    return decisions, decisions[:len(used_steps)]


def read_episode(path):
    """Return a JSON-serializable audit row; partial failures retain available costs.

    Top-level identifiers bind coordinator cases: output, episode_id, family, scene,
    seed, sampling_seed, actor_sha256, mode, stochastic, fixed_choices, purpose, action_horizon.
    ``costs`` is a flat map of known numerical costs. Missing fields and trace-based
    lower bounds are explicit in cost_issues/missing_cost_fields/cost_provenance.
    No missing result, reward or elapsed time is filled with a fictitious zero.
    """
    output = Path(path).resolve()
    reasons = []
    job, ledger = {}, {}
    for name, target in (("job.json", job), ("ledger.json", ledger)):
        try:
            value = _json(output / name)
            if not isinstance(value, dict):
                raise TypeError("Expected a JSON object")
            target.update(value)
        except (OSError, ValueError, TypeError) as error:
            reasons.append(f"{name}: {error}")
    trace = _jsonl(output / "trace.jsonl", reasons)
    costs, provenance, cost_issues, missing, observed_work = _audit_costs(ledger, trace)
    row = {key: job.get(key) for key in ("episode_id", "family", "scene", "seed", "sampling_seed",
                                        "actor_sha256", "mode", "stochastic", "fixed_choices", "purpose",
                                        "action_horizon")}
    short = job.get("max_steps_override") is not None
    row.update(output=str(output), episode_id=job.get("episode_id", str(output)), valid=False,
               invalid_reasons=reasons, short_check=short, full_episode=not short,
               benchmark_result=None, outcome=None, trajectory=None, ppo_eligible=False,
               costs=costs, cost_provenance=provenance, cost_issues=cost_issues,
               missing_cost_fields=missing, observed_work=observed_work, ledger=ledger,
               steps=costs.get("policy_control_steps"), decisions=[], source_audit=None,
               audit_reader_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               outcome_reader_sha256=hashlib.sha256((ROOT /
                   "experiments/helpers/maniguard_phase2_outcomes.py").read_bytes()).hexdigest())
    reasons.extend(cost_issues)
    results = _jsonl(output / "episode/results.jsonl", reasons)
    if len(results) != 1:
        reasons.append("Expected exactly one benchmark result, including a failed result if available")
        return row
    result = results[0]
    row["benchmark_result"] = result
    try:
        if job.get("phase") != "phase2" or job.get("mode") not in ("learned", "fixed", "contextual"):
            raise ValueError("Expected a declared phase-two execution job")
        if type(job.get("stochastic")) is not bool:
            raise ValueError("Missing stochastic collection declaration")
        if job["mode"] != "learned" and job["stochastic"]:
            raise ValueError("Reference controller cannot use stochastic learned decoding")
        if any(type(job.get(key)) is not int or job[key] < 0 for key in ("seed", "sampling_seed")):
            raise ValueError("Episode and sampling seeds must be nonnegative integers")
        if job["mode"] != "fixed" and job.get("fixed_choices") is not None:
            raise ValueError("Only fixed reference mode accepts fixed_choices")
        if job.get("random_policy") is not False:
            raise ValueError("Phase two requires the frozen VLA policy")
        source_audit = audit_sources(output)
        row["source_audit"] = source_audit
        if not source_audit["valid"]:
            raise ValueError("Archived source audit failed: " + "; ".join(
                source_audit["invalid_reasons"]))
        archived = source_audit["configs"]
        spec = archived["families"]["families"][job["family"]]
        expected_horizon = job["max_steps_override"] if short else spec["max_steps"]
        if (type(expected_horizon) is not int or job["action_horizon"] != expected_horizon
                or (short and (job["purpose"] != "engineering" or not 1 <= expected_horizon <= 32))):
            raise ValueError("Undeclared or incorrect shortened execution horizon")
        if job.get("policy") != archived["sources"]["policies"][job["family"]]:
            raise ValueError("Frozen VLA policy pin differs from the declared family source")
        if result.get("scene_name") != job["scene"]:
            raise ValueError("Benchmark result belongs to a different scene")
        _audit_causality(job, result, trace)
        legacy = read_legacy_episode(output)
        row["benchmark_categories"] = {key: legacy[key] for key in
                                        ("safe_completion", "unsafe_completion",
                                         "safe_unfinished", "unsafe_unfinished")}
        steps = result["steps"]
        if type(steps) is not int or steps < 1:
            raise ValueError("Completed result needs a positive integer action count")
        if steps > expected_horizon or (not result["success"] and steps != expected_horizon):
            raise ValueError("Completed run ended at an unexplained action horizon")
        monitor = _json(output / "episode" / (result["scene_name"] + "_ltl.json"))
        if (type(monitor.get("violated")) is not bool
                or type(monitor.get("total_steps_monitored")) is not int
                or "error" not in monitor):
            raise ValueError("Raw monitor verdict or monitoring count has an invalid schema")
        observed = [entry for entry in trace if entry.get("type") == "observed"]
        _steps(observed, range(steps + 1), "observed")
        for entry, authoritative in zip(observed, monitor["log"], strict=True):
            copy = entry.get("monitor", {})
            if (copy.get("valid") is not True or copy.get("error") is not None
                    or copy.get("observation") != authoritative):
                raise ValueError("Runtime monitor record is stale or differs from its sidecar")
        executed = [entry for entry in trace if entry.get("type") == "executed"]
        proposed = [entry for entry in trace if entry.get("type") == "proposal"]
        _steps(executed, range(steps), "executed")
        _steps(proposed, range(steps), "proposal")
        for entry in executed:
            _array(entry.get("action"), (8,), "environment-input action")
        for entry in proposed:
            _array(entry.get("original"), (8,), "proposed action")
            _array(entry.get("adapted"), (8,), "adapted action")
            frame = entry.get("frame", {})
            if frame.get("prerequisite") is not None or frame.get("target_engaged") is not None:
                raise ValueError("Privileged state leaked into the primary controller frame")
        goals = [entry for entry in trace if entry.get("type") == "goal"]
        extra = [entry for entry in trace if entry.get("type") == "evaluation_only"
                 and "step" in entry]
        outcome = derive_outcome(job["family"], result, monitor["log"], goals, extra or None)
        row["outcome"] = outcome
        if not outcome["valid"]:
            raise ValueError("Supplemental completion invalid: " + "; ".join(outcome["invalid_reasons"]))
        decisions, used = _audit_decisions(job, trace, output, steps, archived["encoder"])
        row["decisions"] = decisions
        counters = {"policy_queries": sum(entry.get("type") == "query" for entry in trace),
                    "policy_control_steps": steps, "adapter_decisions": len(decisions),
                    "encoder_image_calls": len(decisions), "encoder_images": 2 * len(decisions),
                    "encoder_text_calls": 1}
        for key, expected in counters.items():
            if ledger.get(key) != expected:
                raise ValueError(f"Recorded {key} differs from complete runtime trace")
        if ledger["env_step_attempts"] < ledger["env_steps_returned"] or (
            ledger["env_steps_returned"] < steps
        ):
            raise ValueError("Environment step counters contradict recorded control actions")
        if ledger.get("phase") != "completed":
            raise ValueError("Cost ledger has no completed episode state")
        row["steps"] = steps
        row["trajectory"] = {
            "features": [entry["features"] for entry in used],
            "actions": [entry["choices"] for entry in used],
            "old_log_probs": ([entry["log_prob"] for entry in used]
                              if job["mode"] == "learned" else None),
            "actor_sha256": job["actor_sha256"], "family": job["family"],
            "terminal_utility": outcome["U"], "episode_id": row["episode_id"],
        }
    except (OSError, ValueError, KeyError, TypeError) as error:
        reasons.append(str(error))
    row["valid"] = not reasons
    row["ppo_eligible"] = bool(row["valid"] and not short and job.get("mode") == "learned"
                                and job.get("stochastic") and job.get("purpose") == "train")
    return row


def aggregate(rows):
    """Equal-family full-episode metrics with failed/short work included in costs.

    ``macro`` is unavailable when any full episode is invalid or a represented
    family has no valid full denominator. ``descriptive_macro_valid_only`` exposes
    the available evidence explicitly and must not be used to drop failed jobs
    from optimizer selection. Summed elapsed seconds are not wall-clock duration.
    Duplicate episode IDs/paths are rejected to prevent double-counting work.
    """
    rows = list(rows)
    identifiers = [row["episode_id"] for row in rows]
    paths = [row["output"] for row in rows]
    if len(set(identifiers)) != len(rows) or len(set(paths)) != len(rows):
        raise ValueError("Duplicate episode records cannot be aggregated twice")
    families, totals, missing, coverage = {}, Counter(), Counter(), Counter()
    for row in rows:
        totals.update(row["costs"])
        coverage.update(row["costs"].keys())
        missing.update(row.get("missing_cost_fields", []))
        name = row.get("family") or "unknown"
        families.setdefault(name, []).append(row)
    groups = {}
    for name, records in families.items():
        full = [row for row in records if row["full_episode"]]
        valid = [row for row in full if row["valid"]]
        metrics = {key: (sum(float(row["outcome"][key]) for row in valid) / len(valid)
                         if valid else None) for key in METRICS}
        groups[name] = {"attempts": len(records), "full_episodes": len(full),
                        "short_checks": len(records) - len(full), "valid_full": len(valid),
                        "invalid": sum(not row["valid"] for row in records),
                        "invalid_full": len(full) - len(valid), "metrics": metrics,
                        "counts": {key: sum(row["outcome"][key] for row in valid)
                                   for key in ("C_req", "S_req", "V")}}
    included = [group for group in groups.values() if group["valid_full"]]
    descriptive = ({key: sum(group["metrics"][key] for group in included) / len(included)
                    for key in METRICS} if included else None)
    complete = bool(groups) and all(group["valid_full"] and not group["invalid_full"]
                                    for group in groups.values())
    return {"families": groups, "macro": descriptive if complete else None,
            "descriptive_macro_valid_only": descriptive,
            "families_without_valid_full_episodes": [name for name, group in groups.items()
                                                      if not group["valid_full"]],
            "counts": {"attempts": len(rows), "invalid": sum(not row["valid"] for row in rows),
                       "short_checks": sum(row["short_check"] for row in rows),
                       "valid_full": sum(row["valid"] and row["full_episode"] for row in rows)},
            "cost_totals": dict(totals), "cost_record_counts": dict(coverage),
            "optional_cost_missing_records": {key: len(rows) - coverage[key]
                                               for key in OPTIONAL_COUNT_COSTS},
            "missing_cost_records": dict(missing),
            "costs_complete": not any(row.get("cost_issues") for row in rows),
            "elapsed_interpretation": "Sum of episode elapsed seconds; not study wall time."}
