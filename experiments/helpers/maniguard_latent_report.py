"""Export independently checked outcomes, complete costs and matched search records."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from helpers.maniguard_latent_queue import atomic
from helpers.maniguard_latent_results import read_episode
from helpers.maniguard_latent_status import snapshot


def category(completed, violation):
    return ("unsafe_" if violation else "safe_") + ("completion" if completed else "unfinished")


def write_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build(run):
    run = Path(run).resolve()
    cost = snapshot(run)
    queue = json.loads((run / "queue.json").read_text())
    attempts = {row["id"]: row for row in queue["attempts"]}
    rows, verified = [], {}
    groups = defaultdict(list)
    for partial in cost["attempts"]:
        name = partial["id"]
        attempt = attempts.get(name, {})
        definition = queue["jobs"].get(attempt.get("job_id"), {}).get("definition", {})
        output = Path(partial["output"])
        job_path = output / "job.json"
        job = json.loads(job_path.read_text()) if job_path.exists() else {}
        ledger_path = output / "ledger.json"
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
        row = {"attempt": name, "stage": partial["stage"], "status": partial["status"],
               "family": job.get("family"), "scene": job.get("scene"), "seed": job.get("seed"),
               "worker": definition.get("worker"), "condition": definition.get("condition"),
               "block": definition.get("block"), "controller_sha256": job.get("controller_sha256"),
               "source_commit": job.get("repository_commit"),
               "policy_controls": ledger.get("policy_control_steps"),
               "physics_steps": ledger.get("physics_steps"),
               "policy_queries": ledger.get("policy_queries"),
               "policy_query_seconds": ledger.get("policy_query_seconds"),
               "elapsed_seconds": ledger.get("elapsed_seconds"),
               "physics_before_callback": "unknown",
               "valid_full_episode": False, "requested_category": None,
               "benchmark_category": None, "requested_completion": None,
               "benchmark_completion": None, "violation": None, "score": None}
        row.update(ever_contacted_task_object=None, ever_grasped_task_object=None)
        if partial["status"] == "completed" and job:
            receipt = read_episode(output, require_full=partial["stage"] != "engineering")
            verified[name] = receipt
            if partial["stage"] != "engineering":
                outcome = receipt["outcome"]
                row.update(valid_full_episode=True,
                           ever_contacted_task_object=receipt.get("physical", {}).get(
                               "ever_contacted"),
                           ever_grasped_task_object=receipt.get("physical", {}).get("ever_grasped"),
                           requested_category=category(outcome["C_req"], outcome["V"]),
                           benchmark_category=category(outcome["benchmark_success"], outcome["V"]),
                           requested_completion=outcome["C_req"],
                           benchmark_completion=outcome["benchmark_success"],
                           violation=outcome["V"], score=receipt["score"])
                key = (partial["stage"], row["family"], row["condition"])
                groups[key].append(row)
        rows.append(row)
    summaries = []
    for (stage, family, condition), members in groups.items():
        n = len(members)
        summaries.append({"stage": stage, "family": family, "condition": condition, "n": n,
                          **{key: sum(row["requested_category"] == key for row in members)
                             for key in ["safe_completion", "unsafe_completion", "safe_unfinished",
                                         "unsafe_unfinished"]},
                          "violations": sum(row["violation"] for row in members),
                          "benchmark_completions": sum(row["benchmark_completion"] for row in members),
                          "scope": "Descriptive counts. Development conditions use different tasks and seeds; audit repeats may share tasks. No independent-binomial safety guarantee."})
    authority = {}
    for family in ["jar", "clutter"]:
        directory = run / ("authority-" + family)
        path = directory / "summary.json"
        if path.exists():
            summary = json.loads(path.read_text())
            probes = [json.loads(p.read_text()) for p in directory.glob("o*.json")]
            values = [p["prefix_arm_rms_rad"] for p in probes if p["label"].startswith("d")]
            authority[family] = {**summary,
                                 "summary_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                 "perturbed_prefix_arm_rms_quantiles_rad": np.quantile(
                                     values, [0, .25, .5, .75, 1]).tolist(),
                                 "prefix_gripper_command_changes": sum(
                                     p.get("prefix_gripper_command_changes", 0) for p in probes
                                     if p["label"].startswith("d"))}
    candidates = search_curves(run, queue, verified)
    audit = [row for row in rows if row["stage"] == "independent_audit"
             and row["valid_full_episode"]]
    expected = {(block, condition) for block in range(4)
                for condition in ["native", "fixed", "ea", "random"]}
    observed = [(row["block"], row["condition"]) for row in audit]
    if len(observed) != len(set(observed)):
        raise ValueError("Repeated independent audit block-condition")
    search_path = run / "search/state.json"
    seal_path = run / "search/sealed-finalists.json"
    finished = (set(observed) == expected and search_path.exists() and seal_path.exists()
                and "independent-audit" in json.loads(search_path.read_text())["completed_milestones"])
    return {"schema": "maniguard-latent-analysis-v1", "full_goal_completion_assessed": False,
            "physical_comparison_complete": finished, "accounting": cost,
            "attempts": rows, "outcome_counts": summaries, "authority": authority,
            "candidate_curves": candidates, "audit": audit,
            "initial_observation_diagnostics": initial_diagnostics(audit, cost["attempts"]),
            "interpretation": ("One-seed exploratory physical comparison; ordinary learning, replication and transfer remain untested."
                               if finished else "The selected full physical comparison is incomplete. Development episodes and decoder authority do not establish evolutionary, safety or transfer benefit."),
            "preservation_verification": "Assess immutable remote receipts and restoration independently; this exporter does not certify goal completion.",
            "curve_accounting": "Each method's cumulative curve includes the same eight shared initial episodes once, followed by its own paid descendants. The global cost ledger counts that shared physics once. Best observed scores span different seed panels and are descriptive training observations, not an independent learning curve."}


def initial_diagnostics(audit, attempts):
    """Describe reset-image variation without claiming nominal seeds force equality."""
    outputs = {row["id"]: Path(row["output"]) for row in attempts}
    rows = []
    for block in range(4):
        members = [row for row in audit if row["block"] == block]
        native = next((row for row in members if row["condition"] == "native"), None)
        if native is None:
            continue
        with np.load(outputs[native["attempt"]] / "initial_observation.npz",
                     allow_pickle=False) as saved:
            reference = {key: saved[key].copy() for key in saved.files}
        for member in members:
            with np.load(outputs[member["attempt"]] / "initial_observation.npz",
                         allow_pickle=False) as saved:
                item = {"block": block, "condition": member["condition"],
                        "same_scene_seed": (member["scene"], member["seed"]) == (
                            native["scene"], native["seed"]),
                        "initial_state_rms_from_native": float(np.sqrt(np.mean(
                            (saved["state"].astype(float) - reference["state"]) ** 2)))}
                for key in ["overview", "wrist"]:
                    item[key + "_equal_native"] = bool(np.array_equal(saved[key], reference[key]))
                    item[key + "_changed_pixel_fraction"] = float(np.mean(np.any(
                        saved[key] != reference[key], axis=-1)))
                rows.append(item)
    return rows


def search_curves(run, queue, verified):
    path = run / "search/state.json"
    if not path.exists():
        return []
    state = json.loads(path.read_text())
    rows = []
    for method in ["ea", "random"]:
        cumulative_controls, cumulative_queries, best = 0, 0, None
        candidates = state["optimizers"][method]["candidates"]
        for index, candidate in enumerate(candidates):
            controls, queries, scores = 0, 0, []
            for case in range(2):
                job = queue["jobs"][f"{candidate['id']}-case{case}"]
                receipt = verified[job["attempt_ids"][-1]]
                scores.append(receipt["score"])
                controls += receipt["physical"]["ledger"]["policy_control_steps"]
                queries += receipt["physical"]["ledger"]["policy_queries"]
            if candidate["scores"] != scores:
                raise ValueError("Optimizer fitness differs from independently verified physics")
            cumulative_controls += controls
            cumulative_queries += queries
            score = sum(scores)
            best = score if best is None else max(best, score)
            rows.append({"method": method, "candidate": candidate["id"],
                         "parent": candidate.get("parent"), "candidate_index": index,
                         "generation": candidate["generation"],
                         "score_sum": score, "best_observed_score_sum": best,
                         "candidate_controls": controls, "cumulative_controls": cumulative_controls,
                         "cumulative_policy_queries": cumulative_queries,
                         "shared_initial_data": index < 4})
    return rows


def export(run, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = build(run)
    atomic(output / "summary.json", report)
    rows = report["attempts"]
    write_csv(output / "attempts.csv", rows, list(rows[0]) if rows else ["attempt"])
    curves = report["candidate_curves"]
    write_csv(output / "candidate-curves.csv", curves, list(curves[0]) if curves else ["method", "candidate"])
    audit = report["audit"]
    write_csv(output / "audit.csv", audit, list(rows[0]) if rows else ["attempt"])
    return report
