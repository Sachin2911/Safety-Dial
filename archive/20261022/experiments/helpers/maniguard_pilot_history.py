"""Descriptive pilot performance against recorded interaction expenditure.

Training points are evaluated candidate/triplet outcomes, not held-out learning
curves. Rotating panels and PPO's sampled training decisions remain explicit.
No optimizer is resumed, model loaded, or physical record opened here.
"""

import math
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

from helpers.maniguard_phase2_optim import record_hash
from helpers.maniguard_pilot_report import _state, build_report, metrics

SCHEMA = "maniguard-pilot-history-v1"
AXES = ("policy_control_steps", "policy_queries", "physics_steps", "elapsed_seconds")


def _cost(value, axis):
    if axis == "elapsed_seconds":
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    return type(value) is int and value >= 0


def build_history(value):
    """Reconstruct completed three-family evaluations and final selection costs.

    Each training point is located after its last recorded family result. Its x
    coordinate includes every earlier attempted episode in that run, including
    other candidates that were interleaved by family. Final selection points pay
    for all training and both finalists' validation, not only the winning actor.
    Preparation, shared references, startup and idle are reported separately.
    """
    state, schedule = _state(value)
    comparison = build_report(state)
    protocol = state["protocol"]
    occurrences = Counter(entry["job_id"] for entry in state["ledger"])
    accepted = {row["job_id"] for row in comparison["attempts"]
                if row["accepted"] and occurrences[row["job_id"]] == 1}
    templates = {row["job_id"]: row for row in schedule}
    groups, run_jobs = defaultdict(list), defaultdict(list)
    for job in schedule:
        if job["stage"] == "reference_validation":
            continue
        run = f"{job['method']}/s{job['optimizer_seed']}"
        run_jobs[run].append(job["job_id"])
        if job["stage"] == "training":
            slot = job["ppo_batch"] if job["method"] == "ppo" else job["candidate_slot"]
            groups[(run, job["round"], slot)].append(job["job_id"])
    totals = {run: dict.fromkeys(AXES, 0) for run in run_jobs}
    exact = {run: dict.fromkeys(AXES, True) for run in run_jobs}
    costs_by_stage = {run: {stage: dict.fromkeys(AXES, 0)
                           for stage in ("training", "finalist_validation")} for run in run_jobs}
    unknown, rows, snapshots, history_issues = [], {}, {}, []
    for job_id, count in occurrences.items():
        if count > 1:
            history_issues.append({"job_id": job_id, "reason": "Repeated scheduled slot",
                                   "recorded_attempts": count})
    seen_outputs, seen_episodes = set(), set()
    shared = dict.fromkeys(AXES, 0)
    shared_exact = dict.fromkeys(AXES, True)
    for index, entry in enumerate(state["ledger"]):
        job_id, row = entry["job_id"], entry.get("result") or {}
        job = templates.get(job_id)
        if job is None:
            unknown.append({"job_id": job_id, "reason": "Unscheduled attempt"})
            for flags in [*exact.values(), shared_exact]:
                flags.update(dict.fromkeys(AXES, False))
            continue
        output, episode = row.get("output"), row.get("episode_id")
        if output in seen_outputs or episode in seen_episodes:
            unknown.append({"job_id": job_id, "reason": "Repeated physical identity"})
            for flags in [*exact.values(), shared_exact]:
                flags.update(dict.fromkeys(AXES, False))
            continue
        seen_outputs.add(output)
        seen_episodes.add(episode)
        reference = job["stage"] == "reference_validation"
        run = None if reference else f"{job['method']}/s{job['optimizer_seed']}"
        target, valid_cost = (shared, shared_exact) if reference else (totals[run], exact[run])
        if not reference:
            dependencies = [previous for previous in run_jobs[run]
                            if templates[previous]["stage"] == "training" and (
                                job["stage"] == "finalist_validation"
                                or templates[previous]["round"] < job["round"]
                                or (job["method"] == "ppo"
                                    and templates[previous]["round"] == job["round"]
                                    and templates[previous]["ppo_batch"] < job["ppo_batch"]))]
            missing_dependencies = [previous for previous in dependencies if previous not in rows]
            if missing_dependencies:
                valid_cost.update(dict.fromkeys(AXES, False))
                history_issues.append({"job_id": job_id, "run": run,
                                       "reason": "Earlier training dependency missing from ledger prefix",
                                       "missing_job_ids": missing_dependencies})
        cost_issues = bool(row.get("cost_issues") or row.get("missing_cost_fields"))
        for axis in AXES:
            cost = (row.get("costs") or {}).get(axis)
            known = _cost(cost, axis)
            if known:
                target[axis] += cost
                if not reference:
                    costs_by_stage[run][job["stage"]][axis] += cost
            if not known or cost_issues:
                valid_cost[axis] = False
                unknown.append({"job_id": job_id, "axis": axis,
                                "reason": "Missing/invalid counter or unresolved cost audit"})
        if job_id in rows:
            continue  # Retain the first record; duplicated slots are invalid, but both costs were paid.
        rows[job_id] = row
        snapshots[job_id] = {"ledger_index": index, "known_cumulative_cost": deepcopy(target),
                             "cumulative_cost": {axis: target[axis] if valid_cost[axis] else None
                                                 for axis in AXES}}
    if state.get("claims"):
        unknown.append({"reason": "In-flight claims have unresolved expenditure",
                        "claims": deepcopy(state["claims"])})
    points = []
    for (run, panel, slot), ids in groups.items():
        received = [job_id for job_id in ids if job_id in rows]
        complete = len(received) == len(ids)
        valid = complete and all(job_id in accepted for job_id in ids)
        actors = {rows[job_id].get("actor_sha256") for job_id in received}
        if len(actors) != 1:
            valid = False
            if complete:
                history_issues.append({"run": run, "panel": panel, "slot_or_batch": slot,
                                       "reason": "One evaluation group contains different actor hashes"})
        if complete:
            end = max(ids, key=lambda job_id: snapshots[job_id]["ledger_index"])
            spent = deepcopy(snapshots[end])
        else:
            spent = {"ledger_index": None, "known_cumulative_cost": None,
                     "cumulative_cost": dict.fromkeys(AXES)}
        method, seed = run.split("/s")
        points.append({"run": run, "method": method, "optimizer_seed": int(seed),
                       "panel": panel, "slot_or_batch": slot, "job_ids": ids,
                       "received_episodes": len(received), "expected_episodes": len(ids),
                       "complete": complete, "valid": valid,
                       "actor_sha256": next(iter(actors)) if len(actors) == 1 else None,
                       "stochastic_training": method == "ppo", **spent,
                       "metrics": metrics([rows[job_id] for job_id in ids], protocol["families"])
                                  if valid else None})
    selected = []
    for run, record in comparison["all_training_runs"].items():
        ids = run_jobs[run]
        complete = all(job_id in accepted for job_id in ids)
        choice = record["selected"]
        selected.append({"run": run, "method": record["method"],
                         "optimizer_seed": record["optimizer_seed"],
                         "complete_training_and_selection": complete,
                         "expected_episodes": len(ids),
                         "received_episodes": sum(job_id in rows for job_id in ids),
                         "actor_sha256": choice["actor_sha256"] if choice else None,
                         "known_costs_by_stage": costs_by_stage[run],
                         "known_total_cost": totals[run],
                         "total_cost": {axis: totals[run][axis] if exact[run][axis] and complete else None
                                        for axis in AXES},
                         "metrics": choice["metrics"] if choice and complete and comparison["valid"] else None})
    report = {"schema": SCHEMA, "state_sha256": state["sha256"],
              "protocol_sha256": protocol["sha256"], "comparison_report_sha256": comparison["sha256"],
              "valid": comparison["valid"] and not unknown and not history_issues,
              "complete": comparison["complete"], "comparison_invalid_reasons": comparison["invalid_reasons"],
              "axes": list(AXES), "training_evaluations": points, "selected_validation": selected,
              "unknown_costs": unknown, "history_issues": history_issues, "in_flight_claims": deepcopy(state.get("claims", {})),
              "shared_reference_costs": {"known": shared,
                  "exact": {axis: shared[axis] if shared_exact[axis] else None for axis in AXES},
                  "counted_once": True, "included_in_each_run": False},
              "all_recorded_physical_costs": comparison["costs"],
              "preparation_startup_idle_costs_included": False,
              "allocated_gpu_hours": None,
              "limitations": [
                  "Training panels rotate. Do not connect panel changes into a monotone improvement curve.",
                  "Training points are candidate/triplet evaluations, not selected-policy held-out checkpoints.",
                  "Search uses deterministic decisions; PPO samples during training. All finalist validation is deterministic.",
                  "Costs at a point include all earlier recorded work in that run, including interleaved candidates.",
                  "Final selection pays for all 72 training and 12 finalist-validation episodes, including discarded nominees.",
                  "Validation selects these actors; it is not an untouched test set and supports no superiority claim.",
                  "Reference physical records count once. Preparation/startup/idle require separate reservation accounting.",
                  "These are saved-summary consistency checks; optimizer archives and physical artifacts are not independently replayed."]}
    report["sha256"] = record_hash(report)
    return report


def write_figures(report, directory, *, title="Pilot descriptive results"):
    """Write standalone PDF/PNG figures without extrapolation or panel-joining lines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if report.get("schema") != SCHEMA or report.get("sha256") != record_hash(
            {key: value for key, value in report.items() if key != "sha256"}):
        raise ValueError("History report hash/schema mismatch")
    if not report["valid"]:
        raise ValueError("Resolve report bindings and cost audits before rendering comparisons")
    title = title + (" [INCOMPLETE SNAPSHOT]" if not report["complete"] else "")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    methods = ("evolution", "random", "ppo")
    seeds = sorted({row["optimizer_seed"] for row in report["training_evaluations"]})
    colors = {"evolution": "#1768ac", "random": "#d17800", "ppo": "#337a48"}
    markers = {seed: ("o", "s", "^")[index % 3] for index, seed in enumerate(seeds)}
    saved = []
    try:
        for axis, label in (("policy_control_steps", "Recorded policy actions"),
                            ("policy_queries", "Recorded policy RPCs")):
            fig, grid = plt.subplots(2, 3, figsize=(12, 6), sharey="row", constrained_layout=True)
            for panel in range(3):
                for method in methods:
                    for seed in seeds:
                        points = [row for row in report["training_evaluations"]
                                  if row["valid"] and row["panel"] == panel and row["method"] == method
                                  and row["optimizer_seed"] == seed and row["cumulative_cost"][axis] is not None]
                        for line, metric in enumerate(("U", "V")):
                            grid[line, panel].scatter([row["cumulative_cost"][axis] for row in points],
                                [row["metrics"]["macro"][metric] for row in points], s=22,
                                color=colors[method], marker=markers[seed], alpha=.7,
                                label=f"{method}, seed {seed}")
                grid[0, panel].set_title(f"Training panel {panel + 1}")
                grid[1, panel].set_xlabel(label + " within run")
                for line in range(2):
                    grid[line, panel].grid(alpha=.2)
                    grid[line, panel].ticklabel_format(axis="x", style="sci", scilimits=(-3, 4))
            grid[0, 0].set_ylabel("Family-macro utility")
            grid[1, 0].set_ylabel("Family-macro violation rate")
            grid[0, 0].set_ylim(-2.1, 1.1)
            grid[1, 0].set_ylim(-.05, 1.05)
            grid[0, -1].legend(fontsize=7, loc="best")
            fig.suptitle(title + "\nTraining evaluations; rotating panels; PPO samples decisions", fontsize=11)
            for extension in ("png", "pdf"):
                path = directory / f"training-{axis}.{extension}"
                fig.savefig(path, dpi=160)
                saved.append(str(path))
            plt.close(fig)
        fig, grid = plt.subplots(2, 2, figsize=(9, 6), sharey="row", constrained_layout=True)
        for column, (axis, label) in enumerate((("policy_control_steps", "Policy actions"),
                                              ("policy_queries", "Policy RPCs"))):
            for row in report["selected_validation"]:
                if row["metrics"] is None or row["total_cost"][axis] is None:
                    continue
                for line, metric in enumerate(("S_req", "V")):
                    grid[line, column].scatter(row["total_cost"][axis], row["metrics"]["macro"][metric],
                        s=45, color=colors[row["method"]], marker=markers[row["optimizer_seed"]],
                        label=f"{row['method']}, seed {row['optimizer_seed']}")
            grid[1, column].set_xlabel(label + ": training + both nominees' validation")
            for line in range(2):
                grid[line, column].set_ylim(-.05, 1.05)
                grid[line, column].grid(alpha=.2)
        grid[0, 0].set_ylabel("Family-macro safe completion")
        grid[1, 0].set_ylabel("Family-macro violation rate")
        if any(row["metrics"] is not None for row in report["selected_validation"]):
            grid[0, -1].legend(fontsize=7)
        fig.suptitle(title + "\nSelected validation outcomes; six cases/actor; not reserved tests", fontsize=11)
        for extension in ("png", "pdf"):
            path = directory / f"selected-validation.{extension}"
            fig.savefig(path, dpi=160)
            saved.append(str(path))
        plt.close(fig)
    finally:
        plt.close("all")
    return saved
