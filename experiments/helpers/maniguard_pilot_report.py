"""Independent descriptive pilot comparison over a saved coordinator state.

No coordinator is resumed and no episode files, services, or reserved outcomes are
opened. Validation rows reconstruct every run's selection; optimizer seeds are never
ranked. The report verifies saved hashes and binding, not historical physics equality.
"""

import hashlib
import json
import math
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import mean

from helpers.maniguard_main_protocol import SELECTION_SCHEMA
from helpers.maniguard_phase2_model import genome_hash, identity_genome, load_model, save_model
from helpers.maniguard_phase2_optim import record_hash
from helpers.maniguard_phase2_protocol import compile_schedule

SCHEMA = "maniguard-pilot-comparison-v1"
OUTCOMES = ("S_req", "C_req", "V", "U")
REFERENCES = ("identity", "fixed", "contextual")


def _hashed(value, label):
    if value.get("sha256") != record_hash({key: item for key, item in value.items() if key != "sha256"}):
        raise ValueError(label + " content hash mismatch")


def _state(value):
    state = json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else deepcopy(value)
    _hashed(state, "Coordinator state")
    if state.get("version") != "maniguard-phase2-coordinator-v1":
        raise ValueError("Unsupported coordinator state schema")
    protocol = state["protocol"]
    _hashed(protocol, "Protocol")
    if protocol.get("status") != "frozen":
        raise ValueError("Comparison requires a frozen pilot protocol")
    if state.get("source_sha256") != protocol["implementation_files"].get(
            "experiments/helpers/maniguard_phase2_coordinator.py"):
        raise ValueError("Coordinator source differs from its frozen protocol pin")
    schedule = compile_schedule(protocol)
    if record_hash(schedule) != protocol["schedule_sha256"]:
        raise ValueError("Saved protocol schedule hash mismatch")
    if (protocol["families"] != ["jar", "stack", "clutter"] or
            protocol["methods"] != ["evolution", "random", "ppo"] or
            len(protocol["optimizer_seeds"]) != 2 or len(set(protocol["optimizer_seeds"])) != 2):
        raise ValueError("Report requires the declared three-family, three-method, two-seed pilot")
    return state, schedule


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def metrics(rows, families):
    """Observed family means plus raw counts; a missing family yields no macro mean."""
    groups = {family: [row for row in rows if row["family"] == family] for family in families}
    per_family = {}
    for family, group in groups.items():
        values = {key: mean(row["outcome"][key] for row in group) if group else None for key in OUTCOMES}
        values.update(episodes=len(group),
                      counts={key: sum(row["outcome"][key] for row in group) for key in OUTCOMES},
                      unfinished=sum(not row["outcome"]["C_req"] for row in group))
        for name in ("policy_queries", "elapsed_seconds"):
            observed = [row.get("costs", {}).get(name) for row in group]
            values[name] = mean(observed) if group and all(_number(value) for value in observed) else None
        engagement = [(row.get("benchmark_result") or {}).get("ever_contacted") for row in group]
        values["engagement_observed"] = sum(type(value) is bool for value in engagement)
        values["engaged"] = sum(value is True for value in engagement)
        per_family[family] = values
    keys = (*OUTCOMES, "policy_queries", "elapsed_seconds")
    macro = {key: mean(value[key] for value in per_family.values())
             if all(value[key] is not None for value in per_family.values()) else None for key in keys}
    weakest = min(families, key=lambda family: per_family[family]["U"]) if macro["U"] is not None else None
    return {"episodes": len(rows), "per_family": per_family, "macro": macro,
            "weakest_family_by_utility": weakest}


def _selection_metrics(values):
    return {**{key: values["macro"][key] for key in ("U", "V", "S_req", "policy_queries")},
            "episodes": values["episodes"],
            "family_episodes": {family: row["episodes"] for family, row in values["per_family"].items()}}


def _binding(entry, template, protocol):
    reasons, job, row = [], entry.get("job") or {}, entry.get("result") or {}
    if template is None or any(job.get(key) != value for key, value in (template or {}).items()):
        reasons.append("Dispatch differs from the frozen schedule")
    if entry.get("accepted") is not True or row.get("valid") is not True:
        reasons.append("Attempt was not accepted as an audited valid result")
    if row.get("full_episode") is not True or row.get("short_check") is not False:
        reasons.append("Expected a full pilot episode")
    for name in ("family", "scene", "seed", "sampling_seed", "actor_sha256", "mode",
                 "fixed_choices", "stochastic", "purpose"):
        if row.get(name) != job.get(name):
            reasons.append("Result/dispatch mismatch: " + name)
    if row.get("action_horizon") != job.get("max_steps"):
        reasons.append("Result horizon differs from dispatch")
    if not all(isinstance(row.get(name), str) and row[name] for name in ("output", "episode_id", "actor_sha256")):
        reasons.append("Missing physical or actor identity")
    outcome = row.get("outcome") or {}
    if (outcome.get("valid") is not True or any(outcome.get(key) not in (0, 1) for key in ("C_req", "S_req", "V"))
            or outcome.get("S_req") != outcome.get("C_req", 0) * (1 - outcome.get("V", 0))
            or outcome.get("U") != outcome.get("S_req", 0) - 2 * outcome.get("V", 0)):
        reasons.append("Outcome does not establish S_req - 2*V")
    costs = row.get("costs") or {}
    if (row.get("cost_issues") or row.get("missing_cost_fields") or
            not _number(costs.get("policy_queries")) or any(not _number(cost) for cost in costs.values())):
        reasons.append("Missing or inconsistent attempted query costs")
    if template is not None:
        ppo_training = template["stage"] == "training" and template["method"] == "ppo"
        expected_sampling = template["optimizer_seed"] * 1000 + template["ppo_episode"] if ppo_training else 0
        if (job.get("purpose") != template["split"] or job.get("sampling_seed") != expected_sampling
                or job.get("stochastic") is not ppo_training):
            reasons.append("Dispatch sampling or split differs from the frozen execution rule")
        if template["stage"] != "reference_validation" and (
                job.get("mode") != "learned" or job.get("fixed_choices") is not None):
            reasons.append("Learned actor dispatch differs from the frozen execution rule")
    if job.get("stage") != "training" and row.get("stochastic") is not False:
        reasons.append("Validation must use deterministic decoding")
    if job.get("stage") == "reference_validation":
        reference = next((item for item in protocol["references"] if item["name"] == job["method"]), {})
        expected_mode = "contextual" if job["method"] == "contextual" else "fixed"
        choices = None if expected_mode == "contextual" else reference.get("selected_choices", reference.get("choices"))
        if (job.get("mode") != expected_mode or job.get("fixed_choices") != choices
                or row.get("actor_sha256") != genome_hash(identity_genome())):
            reasons.append("Reference execution policy differs from the frozen definition")
    return reasons


def comparison(left, right, *, method, seed, comparator):
    families = left["per_family"]
    deltas, regressions = {}, []
    for family in [*families, "macro"]:
        a = left["macro"] if family == "macro" else left["per_family"][family]
        b = right["macro"] if family == "macro" else right["per_family"][family]
        deltas[family] = {key: a[key] - b[key] if a[key] is not None and b[key] is not None else None
                         for key in (*OUTCOMES, "policy_queries", "elapsed_seconds")}
        for key, delta in deltas[family].items():
            if delta is not None and ((key in ("V", "policy_queries", "elapsed_seconds") and delta > 0)
                                      or (key in ("S_req", "C_req", "U") and delta < 0)):
                regressions.append({"method": method, "optimizer_seed": seed, "comparator": comparator,
                                    "family": family, "metric": key, "delta": delta,
                                    "kind": "cost" if key in ("policy_queries", "elapsed_seconds") else "outcome"})
    return {"method": method, "optimizer_seed": seed, "comparator": comparator,
            "deltas": deltas, "regressions": regressions,
            "physical_trajectory_pairing_established": False}


def _screening(complete, issues, comparisons, protocol, completion_review, mechanism_review):
    by_method = {}
    for method in protocol["methods"]:
        rows = [row for row in comparisons if row["method"] == method and row["comparator"] == "identity"]
        ready = len(rows) == len(protocol["optimizer_seeds"])
        family_gain = {family: mean(row["deltas"][family]["S_req"] for row in rows) if ready else None
                       for family in protocol["families"]}
        utility_gain = mean(row["deltas"]["macro"]["U"] for row in rows) if ready else None
        seed_gain = {str(row["optimizer_seed"]): row["deltas"]["macro"]["S_req"] > 0 for row in rows}
        violation_increases = [{"optimizer_seed": row["optimizer_seed"], "family": family,
                               "delta": row["deltas"][family]["V"]}
                              for row in rows for family in protocol["families"]
                              if row["deltas"][family]["V"] > 0]
        numeric_pass = (ready and sum(value > 0 for value in family_gain.values()) >= 2
                        and utility_gain > 0 and all(seed_gain.values()))
        by_method[method] = {"numerical_screen_passed": numeric_pass,
                             "mean_seed_family_safe_completion_gain": family_gain,
                             "mean_seed_macro_utility_gain": utility_gain,
                             "positive_safe_completion_gain_by_seed": seed_gain,
                             "violation_increases": violation_increases,
                             "completion_review_passed": bool(completion_review and completion_review.get("passed") is True)}
    review_passed = bool(completion_review and completion_review.get("passed") is True)
    passing = [name for name, value in by_method.items() if value["numerical_screen_passed"]]
    if issues:
        decision = "repair_experiment"
    elif not complete:
        decision = "incomplete_pilot"
    elif passing and not review_passed:
        decision = "completion_semantics_review_required"
    elif passing and any(not by_method[name]["violation_increases"] for name in passing):
        decision = "proceed_to_larger_study_screening_only"
    elif passing:
        decision = "explicit_completion_violation_tradeoff_review_required"
    elif any(any(gain is not None and gain > 0 for gain in row["mean_seed_family_safe_completion_gain"].values())
             for row in by_method.values()):
        decision = "insufficient_family_breadth_or_seed_support"
    elif mechanism_review and mechanism_review.get("decisions_activate") is True:
        decision = "reconsider_interface_after_reviewing_mechanisms"
    else:
        decision = "review_mechanisms_and_sparse_completion"
    alternatives = {}
    for method in ("random", "ppo"):
        observed = [row for row in comparisons if row["method"] == method and row["comparator"] == "evolution"]
        alternatives[method] = {
            "per_seed_macro_deltas_against_evolution": {
                str(row["optimizer_seed"]): row["deltas"]["macro"] for row in observed},
            "observed_completion_and_utility_at_least_evolution_in_both_seeds": (
                len(observed) == len(protocol["optimizer_seeds"])
                and all(row["deltas"]["macro"]["S_req"] >= 0 and row["deltas"]["macro"]["U"] >= 0
                        for row in observed)),
            "equivalence_established": False,
        }
    return {"decision": decision, "methods": by_method, "non_evolutionary_comparisons": alternatives,
            "criteria_source": "phase2.md, section 9; descriptive pilot screening only",
            "seed_support_interpretation": "Require positive macro safe-completion gain in both prespecified seeds; never select the best seed.",
            "completion_review": deepcopy(completion_review), "mechanism_review": deepcopy(mechanism_review),
            "similarity": "No similarity margin or statistical superiority test was prespecified. Method comparisons remain descriptive; random/PPO competitiveness warrants adaptation research without an EA claim.",
            "safety_improvement_claim": False, "evolutionary_superiority_claim": False,
            "statistical_significance_claim": False}


def build_report(value, *, completion_review=None, mechanism_review=None):
    """Read a coordinator JSON path or state dict; return a self-contained comparison.

    Reviews are explicit supplied interpretations, retained verbatim, never inferred
    from a short rollout or from aggregate gains. This function writes nothing.
    """
    state, schedule = _state(value)
    protocol, families = state["protocol"], state["protocol"]["families"]
    templates = {job["job_id"]: job for job in schedule}
    issues, invalid, accepted, attempts = list(state.get("blocked_reasons", [])), [], {}, []
    outputs, episodes = Counter(), Counter()
    totals, observed, cost_unknown = defaultdict(float), defaultdict(float), []
    seen_physical = set()
    for entry in state["ledger"]:
        job_id, row = entry["job_id"], entry.get("result") or {}
        outputs[row.get("output")] += 1
        episodes[row.get("episode_id")] += 1
        reasons = _binding(entry, templates.get(job_id), protocol)
        if job_id in accepted:
            reasons.append("Repeated scheduled job")
        physical = (row.get("output"), row.get("episode_id"))
        if physical not in seen_physical:
            seen_physical.add(physical)
            for name, cost in (row.get("costs") or {}).items():
                if _number(cost):
                    totals[name] += cost
            for name, cost in (row.get("observed_work") or {}).items():
                if _number(cost):
                    observed[name] += cost
        malformed_costs = [name for name, cost in (row.get("costs") or {}).items() if not _number(cost)]
        if (row.get("cost_issues") or row.get("missing_cost_fields") or not row.get("costs")
                or malformed_costs):
            cost_unknown.append({"job_id": job_id, "output": row.get("output"),
                                 "issues": row.get("cost_issues"), "missing": row.get("missing_cost_fields"),
                                 "malformed": malformed_costs})
        attempts.append({"job_id": job_id, "stage": (entry.get("job") or {}).get("stage"),
                         "accepted": not reasons, "output": row.get("output"),
                         "episode_id": row.get("episode_id"), "measured_row_sha256": row.get("measured_row_sha256"),
                         "result_summary_sha256": record_hash(row), "invalid_reasons": reasons,
                         "raw_invalid_reasons": row.get("invalid_reasons", [])})
        if reasons:
            invalid.append({"job_id": job_id, "reasons": reasons})
        else:
            accepted[job_id] = row
    duplicated = {value for value, count in outputs.items() if count > 1} | {
        value for value, count in episodes.items() if count > 1}
    ambiguous_costs = (any(len({episode for output, episode in seen_physical if output == value}) > 1
                           for value, count in outputs.items() if count > 1) or
                       any(len({output for output, episode in seen_physical if episode == value}) > 1
                           for value, count in episodes.items() if count > 1))
    if ambiguous_costs:
        cost_unknown.append({"reason": "Conflicting physical identifiers prevent exact cost deduplication"})
    if duplicated:
        issues.append("Duplicated physical output or episode identifiers")
        for attempt in attempts:
            if attempt["output"] in duplicated or attempt["episode_id"] in duplicated:
                attempt["accepted"] = False
                attempt["invalid_reasons"].append("Duplicated physical output or episode identifier")
        accepted = {job_id: row for job_id, row in accepted.items()
                    if row.get("output") not in duplicated and row.get("episode_id") not in duplicated}
    if invalid:
        issues.append("Invalid or mismatched pilot attempts")
    missing = [job_id for job_id in templates if job_id not in accepted]
    runs, selected_rows, references = {}, {}, {}
    for method in protocol["methods"]:
        for seed in protocol["optimizer_seeds"]:
            key = f"{method}/s{seed}"
            training_ids = [job["job_id"] for job in schedule if job["stage"] == "training"
                            and job["method"] == method and job["optimizer_seed"] == seed]
            training = [accepted[job_id] for job_id in training_ids if job_id in accepted]
            nominees = state.get("nominees", {}).get(key, [])
            candidates = []
            for finalist in range(2):
                ids = [job["job_id"] for job in schedule if job["stage"] == "finalist_validation"
                       and job["method"] == method and job["optimizer_seed"] == seed and job["finalist"] == finalist]
                rows = [accepted[job_id] for job_id in ids if job_id in accepted]
                actor = genome_hash(nominees[finalist]["genome"]) if len(nominees) == 2 else None
                valid = len(rows) == len(ids) and actor is not None
                if actor is not None and any(row["actor_sha256"] != actor for row in rows):
                    issues.append("Finalist actor differs from training nomination: " + key)
                    valid = False
                candidates.append({"finalist": finalist, "actor_sha256": actor, "complete": valid,
                                   "metrics": metrics(rows, families), "job_ids": ids})
            selection = None
            if all(candidate["complete"] for candidate in candidates):
                chosen = min(candidates, key=lambda candidate: (-candidate["metrics"]["macro"]["U"],
                             candidate["metrics"]["macro"]["V"], candidate["metrics"]["macro"]["policy_queries"],
                             candidate["finalist"]))
                selection = deepcopy(chosen)
                saved = state.get("selected", {}).get(key)
                if (saved is None or saved.get("finalist") != chosen["finalist"]
                        or saved.get("actor_sha256") != chosen["actor_sha256"]
                        or genome_hash(saved.get("genome", [])) != chosen["actor_sha256"]
                        or saved.get("metrics") != _selection_metrics(chosen["metrics"])):
                    issues.append("Saved selection does not reproduce: " + key)
                selected_rows[key] = [accepted[job_id] for job_id in chosen["job_ids"]]
            runs[key] = {"method": method, "optimizer_seed": seed,
                         "training": {"expected_episodes": len(training_ids), "valid_episodes": len(training),
                                      "job_ids": training_ids, "metrics": metrics(training, families),
                                      "archive_sha256": record_hash(state.get("learners" if method == "ppo" else "searches", {}).get(key))},
                         "nomination_rule": "checkpoints at 48 and 72 episodes" if method == "ppo" else "two finalists from the last shared training panel",
                         "candidates": candidates, "selected": selection}
    for name in REFERENCES:
        ids = [job["job_id"] for job in schedule if job["stage"] == "reference_validation" and job["method"] == name]
        rows = [accepted[job_id] for job_id in ids if job_id in accepted]
        references[name] = {"complete": len(rows) == len(ids), "metrics": metrics(rows, families), "job_ids": ids,
                            "physical_episodes": len(rows), "shared_across_optimizer_seeds": True}
    comparisons = []
    for key, rows in selected_rows.items():
        run = runs[key]
        for name, reference in references.items():
            if reference["complete"]:
                comparisons.append(comparison(metrics(rows, families), reference["metrics"], method=run["method"],
                                              seed=run["optimizer_seed"], comparator=name))
        for other in protocol["methods"]:
            other_key = f"{other}/s{run['optimizer_seed']}"
            if other != run["method"] and other_key in selected_rows:
                comparisons.append(comparison(metrics(rows, families), metrics(selected_rows[other_key], families),
                                              method=run["method"], seed=run["optimizer_seed"], comparator=other))
    method_metrics = {method: metrics([row for key, rows in selected_rows.items()
                                      if runs[key]["method"] == method for row in rows], families)
                      if all(f"{method}/s{seed}" in selected_rows for seed in protocol["optimizer_seeds"]) else None
                      for method in protocol["methods"]}
    complete = not missing and not invalid and not state.get("claims") and len(selected_rows) == 6
    if state.get("claims"):
        cost_unknown.append({"in_flight_claims": deepcopy(state["claims"])})
    report = {"schema": SCHEMA, "state_sha256": state["sha256"], "protocol_sha256": protocol["sha256"],
              "coordinator_source_sha256": state["source_sha256"], "complete": complete, "valid": not issues,
              "validity_scope": "Internal consistency of saved dispatch/result summaries and reconstructed validation selection.",
              "independent_audit": {"episode_artifacts_reaudited": False,
                                    "protocol_freeze_evidence_revalidated": False,
                                    "optimizer_archives_replayed": False,
                                    "screening_authorizes_execution": False},
              "invalid_reasons": issues, "missing_or_invalid_jobs": missing, "invalid_attempts": invalid,
              "attempts": attempts, "in_flight_claims": deepcopy(state.get("claims", {})),
              "all_training_runs": runs, "references": references, "method_metrics_all_seeds": method_metrics,
              "comparisons": comparisons, "regressions": [item for row in comparisons for item in row["regressions"]],
              "costs": {"recorded_totals": dict(totals), "observed_work_lower_bounds": dict(observed),
                        "unknown_records": cost_unknown, "allocated_gpu_hours": None,
                        "recorded_totals_may_double_count": ambiguous_costs,
                        "elapsed_seconds_are_episode_sums_not_reservation_time": True},
              "selection_rule": ["higher family-macro U", "lower family-macro V", "lower attempted query cost", "earlier nominee"],
              "screening": _screening(complete, issues, comparisons, protocol, completion_review, mechanism_review),
              "limitations": ["Validation selected these actors and is not an untouched test set.",
                              "Two optimizer seeds and six validation cases per actor do not establish superiority, equal risk or generalization.",
                              "Reference episodes are reused for seed comparisons, not counted as new independent observations.",
                              "Matching nominal seeds do not establish identical starts or paired physical trajectories.",
                              "This report verifies saved state/source/schedule hashes and result summaries; it does not rerun physics or attest historical live imports.",
                              "Nomination rules are declared provenance; optimizer archives and their link to the training ledger are retained, not independently replayed.",
                              "A supplied completion review is an interpretation, not authorization to launch a main study."]}
    report["sha256"] = record_hash(report)
    return report


def export_selected(value, output):
    """Export every pilot-selected actor with main-planner-compatible provenance.

    This is a model handoff, not main-study authorization or a claim that two pilot
    seeds suffice for main inference. The main freeze requires its own precision,
    selection-provenance, exposure, cost and runtime reviews.
    """
    state, _ = _state(value)
    report = build_report(state)
    if not report["valid"] or not report["complete"]:
        raise ValueError("Export requires a complete consistent pilot; never select only a favorable seed")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)

    def write(name, contents):
        path = output / name
        with path.open("x") as stream:
            json.dump(contents, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    source = write("source-state.json", state)
    comparison_report = write("report.json", report)
    exports, training_runs, selected_actors = [], [], []
    for key, run in report["all_training_runs"].items():
        selected = run["selected"]
        genome = state["nominees"][key][selected["finalist"]]["genome"]
        run_id = key.replace("/", "-")
        path = output / (run_id + ".npz")
        metadata = {"run": key, "finalist": selected["finalist"], "state_sha256": state["sha256"],
                    "protocol_sha256": state["protocol"]["sha256"], "validation_selected": True}
        save_model(path, genome, metadata=metadata)
        loaded, _, saved_metadata = load_model(path)
        if genome_hash(loaded) != selected["actor_sha256"] or saved_metadata != metadata:
            raise ValueError("Exported actor failed round-trip verification")
        actor = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        training = {"run_id": run_id, "method": run["method"], "optimizer_seed": run["optimizer_seed"],
                    "training_protocol_sha256": state["protocol"]["sha256"]}
        selection = {"schema": SELECTION_SCHEMA, **training,
                     "selected_actor_sha256": selected["actor_sha256"],
                     "training_families": list(state["protocol"]["families"]),
                     "selection_families": list(state["protocol"]["families"]),
                     "selection_partition": "validation", "finalist": selected["finalist"],
                     "validation_job_ids": selected["job_ids"], "validation_metrics": selected["metrics"],
                     "source_state": source, "comparison_report": comparison_report,
                     "evidence_scope": deepcopy(report["independent_audit"]),
                     "scope": "Pilot-selected model; main-study authorization remains separate."}
        selection["sha256"] = record_hash(selection)
        selection_source = write(run_id + "-selection.json", selection)
        training_runs.append(training)
        selected_actors.append({"run_id": run_id, "actor": actor,
                                "actor_sha256": selected["actor_sha256"], "selection": selection_source})
        exports.append({"run": key, "run_id": run_id, "path": str(path),
                        "actor_sha256": selected["actor_sha256"], "file_sha256": actor["sha256"],
                        "selection": selection_source})
    manifest = {"schema": "maniguard-selected-pilot-actors-v1", "state_sha256": state["sha256"],
                "exports": exports, "all_training_runs_preserved": True, "source_state": source,
                "comparison_report": comparison_report,
                "training_runs": training_runs, "selected_actors": selected_actors,
                "main_evaluation_authorized": False}
    manifest["sha256"] = record_hash(manifest)
    write("manifest.json", manifest)
    return manifest
