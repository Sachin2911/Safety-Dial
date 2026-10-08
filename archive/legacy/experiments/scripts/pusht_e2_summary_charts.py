#!/usr/bin/env python3
"""Render saved E2 summary evidence without model, simulator or statistical queries.

The identification intervals reflect unresolved accepted futures, not sampling
confidence. All numbers are exported with source JSON pointers. Original repair
files are hashed before/after; this creates only a fresh sibling output directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("no_update", "adapted", "readout_correction", "fixed_margin")
LABELS = ("No update", "Adapted predictor", "Readout correction", "Fixed margin (80 px fallback)")
BLUE, ORANGE, GREY = "#2878a8", "#d78424", "#5d6670"


def require(value, message):
    if not value:
        raise ValueError(message)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_hashes(path):
    files = sorted(Path(path).rglob("*"))
    require(not any(p.is_symlink() for p in files), "Symlink in source bundle")
    return {str(p.relative_to(path)): file_hash(p) for p in files if p.is_file()}


def normal(value):
    if isinstance(value, dict):
        return {k: normal(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normal(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def metric_numbers(metric):
    n, accepted, bad, unknown = [metric[k] for k in ("n", "n_accepted", "n_false_safe", "n_accepted_censored")]
    require(all(type(x) is int for x in (n, accepted, bad, unknown)) and
            n > 0 and 0 <= bad + unknown <= accepted <= n and min(bad, unknown) >= 0,
            "Invalid decision counts")
    lower, upper = (bad / accepted, (bad + unknown) / accepted) if accepted else (None, None)
    point = bad / accepted if accepted and not unknown else None
    require(normal(metric["fsa"]) == point, "Saved FSA point disagrees with unresolved/zero acceptance")
    require(metric["acceptance_rate"] == accepted / n, "Acceptance count/rate mismatch")
    require(normal(metric["fsa_lower"]) == lower and normal(metric["fsa_upper"]) == upper,
            "Identification endpoints do not match integer counts")
    return {"n": n, "accepted": accepted, "known_false_safe": bad, "accepted_unresolved": unknown,
            "acceptance_rate": accepted / n, "fsa_point": point, "fsa_lower": lower, "fsa_upper": upper,
            "point_status": "undefined_zero_acceptance" if not accepted else
            "undefined_accepted_unresolved" if unknown else "defined"}


def relative_outer_bound(baseline, adapted):
    require(baseline["known_false_safe"] > 0 and adapted["accepted"] > 0,
            "A positive baseline lower bound and nonempty adapted acceptance are required")
    blo = Fraction(baseline["known_false_safe"], baseline["accepted"])
    bhi = Fraction(baseline["known_false_safe"] + baseline["accepted_unresolved"], baseline["accepted"])
    alo = Fraction(adapted["known_false_safe"], adapted["accepted"])
    ahi = Fraction(adapted["known_false_safe"] + adapted["accepted_unresolved"], adapted["accepted"])
    lo, hi = 1 - ahi / blo, 1 - alo / bhi
    return {"lower": float(lo), "upper": float(hi), "lower_exact_fraction": str(lo),
            "upper_exact_fraction": str(hi), "threshold": 0.25,
            "formula": "[1 - adapted_upper/no_update_lower, 1 - adapted_lower/no_update_upper]",
            "interpretation": "Conservative outer identification bounds for the saved fixed accepted sets. Shared labels need not jointly attain the endpoints. Not a confidence interval or population inference."}


def goal_completion(rows):
    identities = [(r["root_id"], r["episode"], r["case_sha256"]) for r in rows]
    require(len(set(identities)) == len(rows) and len({r["episode"] for r in rows}) == len(rows),
            "Duplicate goal-retention case")
    counts = {"full_planned_horizon": 0, "verified_early_goal": 0,
              "early_unverified_terminal": 0, "other_incomplete": 0}
    saved = []
    for r in rows:
        requested, executed = r["requested_steps"], r["executed_steps"]
        require(requested == 250 and 0 <= executed <= requested, "Unexpected planned/executed goal horizon")
        clean = r["valid"] and not r["arena_exit"] and not r["truncated"]
        if executed == requested and clean and not r["censored_future"] and r["outcome_complete"]:
            category = "full_planned_horizon"
        elif executed < requested and r["completed_on_verified_goal"]:
            require(clean and r["terminated"] and r["final_coverage"] >= .95 and r["outcome_complete"],
                    "Unverified early goal cannot be classified as complete")
            category = "verified_early_goal"
        elif executed < requested and r["terminated"] and not r["completed_on_verified_goal"]:
            require(r["censored_future"] and not r["outcome_complete"], "Early terminal must remain incomplete")
            category = "early_unverified_terminal"
        else:
            category = "other_incomplete"
        counts[category] += 1
        saved.append({k: r[k] for k in ("root_id", "episode", "case_sha256", "requested_steps", "executed_steps",
                                        "terminated", "truncated", "censored_future", "outcome_complete",
                                        "completed_on_verified_goal", "final_coverage", "arena_exit")}
                     | {"category": category})
    return {"n_cases": len(rows), "counts": counts, "cases": saved,
            "interpretation": "Execution/completeness counts, not goal-success counts."}


def prepare(paired, repair, audit):
    require(paired["run_id"] == repair["run_id"] and audit["status"] == "evidence_verified", "Evidence identity mismatch")
    require(normal(paired["gate"]) == normal(repair["gate"]) == audit["scientific_gate"], "Saved gates differ")
    require(repair["gate"]["passes"] is False, "Expected the frozen negative E2 outcome")
    arms = {}
    for arm in ARMS:
        saved = paired["banks"]["test"]["arms"][arm]
        require(saved["at_matched"] == normal(repair["arms"][arm]["test"]["at_matched"]), "Paired/report metrics differ")
        arms[arm] = metric_numbers(saved["at_matched"]) | {"margin_px": saved["margin"],
            "source": f"paired_report.json#/banks/test/arms/{arm}/at_matched"}
    require(all(a["fsa_point"] is None for a in arms.values()), "Expected all four test FSA points undefined")
    require(arms["fixed_margin"]["margin_px"] == 80.0 and
            normal(repair["arms"]["adapted"]["dev"]["at_matched"]["fsa"]) is None,
            "Expected recorded 80 px fallback with undefined development target")
    bound = relative_outer_bound(arms["no_update"], arms["adapted"])
    recorded = audit["supplemental_fixed_acceptance_censor_bound"]["relative_reduction_bounds"]
    require(all(bound[k] == recorded[k] for k in ("lower", "upper", "lower_exact_fraction", "upper_exact_fraction")),
            "Exact relative-reduction bound differs from existing audit")
    require(bound["upper"] < bound["threshold"], "Saved-set upper bound does not exclude the gate target")
    clips, goals = {}, {}
    for arm in ("no_update", "adapted"):
        clips[arm] = repair[f"{arm}_retention"]
        require(clips[arm] == paired["retention"][f"{arm}_clips"] == audit["latent_retention_reported"][arm], "Clip summaries differ")
        require(clips[arm]["n"] == 400 and all(math.isfinite(v) and v >= 0 for v in clips[arm].values()), "Invalid clip metric")
        goals[arm] = goal_completion(repair["goal_retention"][arm]["rows"])
        counts = goals[arm]["counts"]
        previous = audit["retention_details"][arm]
        require(goals[arm]["n_cases"] == 20 and counts["full_planned_horizon"] == previous["full_250_step_horizons"]
                and counts["verified_early_goal"] == previous["verified_early_goal_completions"]
                and counts["early_unverified_terminal"] == previous["outcomes_failing_completeness"]
                and counts["other_incomplete"] == 0, "Goal completeness differs from saved audit")
    def ids(arm):
        return {(r["root_id"], r["episode"], r["case_sha256"]) for r in goals[arm]["cases"]}
    require(ids("no_update") == ids("adapted"), "Goal cases are not exactly paired")
    require(repair["goal_retention_cases_sha256"] == audit["case_sha256"] == paired["retention"]["case_file_sha256"], "Case file identities differ")
    require(repair["goal_retention"]["comparison"] == paired["retention"]["goal_comparison"] == audit["retention_comparison"], "Goal comparison differs")
    ledger = repair["simulator_costs"]
    require(ledger == paired["costs"]["all_run_simulator_steps"] == audit["simulator_costs"], "Cost ledgers differ")
    steps = ledger["steps"]
    groups = {"Adaptation root generation": ["collection_root_generation", "collection_proposals_rejected_by_arena", "collection_proposals_rejected_after_float32", "collection_discarded_nonfamiliar_source", "collection_discarded_invalid_root"],
              "Queried adaptation branches": ["collection_branch"],
              "Training / readout context replay": ["context_adapt_training_history", "context_readout_correction_history"],
              "Evaluation context replay": ["evaluation_dev_history", "evaluation_test_history", "evaluation_stress_history"],
              "Goal retention (both models)": ["goal_no_update_goal_retention", "goal_adapted_goal_retention"]}
    require(set(steps) == {k for names in groups.values() for k in names}, "Unmapped simulator cost category")
    costs = {name: {"steps": sum(steps[k] for k in keys), "ledger_keys": keys} for name, keys in groups.items()}
    require(sum(c["steps"] for c in costs.values()) == ledger["total_steps"] == 58475, "Charged cost total differs")
    return {"run_id": repair["run_id"], "saved_gate": normal(repair["gate"]), "test": {"arms": arms,
            "n_roots": paired["banks"]["test"]["n_roots"], "relative_reduction_outer_bound": bound,
            "development_target_acceptance": repair["target_acceptance_rate_dev"],
            "interpretation": "Each arm keeps its recorded development-set threshold. Achieved test acceptance differs. Fixed margin uses the recorded 80 px fallback, not a matched test acceptance rate.",
            "point_fsa": "All four are undefined; plotted intervals have no invented midpoint or point estimate."},
            "clip_retention": {"arms": clips, "source": "repair.json#/{no_update,adapted}_retention",
                               "units": {"latent_mse_tf": "latent squared units", "latent_mse_rollout": "latent squared units", "pose_h5_centre_px_vs_real_readout": "pixels, versus frozen real-image readout (not simulator ground truth)"}},
            "goal_retention": {"arms": goals, "case_file_sha256": audit["case_sha256"],
                               "planned_blocks": 50, "steps_per_block": 5,
                               "comparison": repair["goal_retention"]["comparison"], "source": "repair.json#/goal_retention"},
            "costs": {"groups": costs, "total_steps": 58475, "original_ledger": ledger,
                      "source": "repair.json#/simulator_costs", "scope": audit["accounting_scope"]},
            "recipe_selection": audit["recipe_selection"],
            "limits": ["Saved values and elementary integer arithmetic only. No bootstrap, model, physics, new threshold or example selection.",
                       "Censor identification bounds are not confidence intervals; accepted sets differ across arms.",
                       "The 18.05% upper bound is a conservative saved-set bound, not a population-effect claim. Positive improvement below 25% remains compatible.",
                       "Latent/readout errors are the saved evaluations and were not regenerated. Completeness counts do not measure goal success."]}


def finish(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=170)
    fig.savefig(output / f"{name}.pdf")
    plt.close(fig)


def plot_safety(data, output):
    fig = plt.figure(figsize=(13, 9))
    grid = fig.add_gridspec(2, 2, height_ratios=[3.5, 1.2], hspace=.65, wspace=.15)
    ax, ar = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    for i, arm in enumerate(ARMS):
        d, y = data["test"]["arms"][arm], 3 - i
        lo, hi = d["fsa_lower"] * 100, d["fsa_upper"] * 100
        ax.hlines(y, lo, hi, color=BLUE, linewidth=5)
        ax.vlines((lo, hi), y - .09, y + .09, color=BLUE, linewidth=1.5)
        ax.text((lo + hi) / 2, y + .17, f"{lo:.2f}% to {hi:.2f}%", ha="center", fontsize=10)
        ar.barh(y, 100 * d["acceptance_rate"], color=BLUE, alpha=.8, height=.32)
        ar.text(1, y + .23, f"{d['accepted']:,} / {d['n']:,} accepted ({100*d['acceptance_rate']:.2f}%)", fontsize=10)
        ar.text(1, y - .34, f"Known false-safe: {d['known_false_safe']}; accepted unresolved: {d['accepted_unresolved']}", fontsize=9, color=GREY)
    ax.set_yticks(range(4), LABELS[::-1])
    ax.set_xlim(0, 33)
    ar.set_xlim(0, 100)
    ar.set_yticks([])
    for a in (ax, ar):
        a.set_ylim(-.6, 3.65)
        a.grid(axis="x", alpha=.15)
        a.set_axisbelow(True)
    ax.set_title("Censor identification bounds\nNo FSA point estimates", fontsize=12)
    ax.set_xlabel("False-safe acceptance (%)")
    ar.set_title("Achieved acceptance and counts\nSame 4,096 saved test rows", fontsize=12)
    ar.set_xlabel("Acceptance rate (%)")
    bound = data["test"]["relative_reduction_outer_bound"]
    bx = fig.add_subplot(grid[1, :])
    lo, hi = 100 * bound["lower"], 100 * bound["upper"]
    bx.hlines(0, lo, hi, color=BLUE, linewidth=6)
    bx.vlines((lo, hi), -.1, .1, color=BLUE, linewidth=2)
    bx.axvline(25, color=ORANGE, linestyle="--", linewidth=2)
    bx.text(lo, .18, f"{lo:.2f}%", ha="center")
    bx.text(hi, .18, f"Upper bound {hi:.2f}%", ha="center", weight="bold")
    bx.text(25, .18, "25% gate target", ha="center", color=ORANGE)
    bx.set(xlim=(0, 30), ylim=(-.25, .5), yticks=[], xlabel="Relative reduction: 1 - adapted FSA / no-update FSA (%)")
    bx.set_title("Conservative outer bound for these fixed accepted sets", fontsize=12, pad=12)
    bx.grid(axis="x", alpha=.15)
    fig.suptitle("Push-T E2: uncertainty from unresolved futures remains\nAll four point FSAs are undefined", fontsize=17, y=.98)
    fig.subplots_adjust(left=.22, right=.97, top=.84, bottom=.18)
    fig.text(.03, .045, "Intervals are censor identification bounds, not confidence intervals. No midpoint is an estimate.\n"
             "Test acceptance differs after development calibration; the fixed-margin control uses the recorded 80 px fallback.\n"
             "The 18.05% upper bound excludes 25% only for these saved fixed sets; it is not a population inference.", fontsize=10)
    finish(fig, output, "test_acceptance_and_censor_bounds")


def plot_retention(data, output):
    fig = plt.figure(figsize=(13, 8))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.7, 1.2], hspace=.7, wspace=.35)
    metrics = [("latent_mse_tf", "Teacher-forced latent MSE", "Latent squared units"),
               ("latent_mse_rollout", "Rollout latent MSE", "Latent squared units"),
               ("pose_h5_centre_px_vs_real_readout", "Horizon-5 centre error", "Pixels versus real-image readout")]
    for i, (key, title, unit) in enumerate(metrics):
        ax = fig.add_subplot(grid[0, i])
        values = [data["clip_retention"]["arms"][arm][key] for arm in ("no_update", "adapted")]
        ax.bar([0, 1], values, color=[GREY, BLUE], width=.55)
        for j, value in enumerate(values):
            ax.text(j, value + max(values)*.04, f"{value:.6f}" if i < 2 else f"{value:.3f}", ha="center", fontsize=11)
        ax.set(xticks=[0, 1], xticklabels=["No update", "Adapted"], ylim=(0, max(values)*1.23), ylabel=unit)
        ax.set_title(title, fontsize=12)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    ax = fig.add_subplot(grid[1, :])
    for y, arm in ((1, "no_update"), (0, "adapted")):
        d = data["goal_retention"]["arms"][arm]["counts"]
        full, early = d["full_planned_horizon"], d["early_unverified_terminal"]
        require(d["verified_early_goal"] == 0 and d["other_incomplete"] == 0, "Unexpected category needs its own plotted segment")
        ax.barh(y, full, color=BLUE, height=.42)
        ax.barh(y, early, left=full, color=ORANGE, height=.42)
        ax.text(full/2, y, f"{full} full planned horizons", color="white", ha="center", va="center")
        ax.text(full + early/2, y, str(early), color="white", ha="center", va="center", weight="bold")
    ax.set(yticks=[1, 0], yticklabels=["No update", "Adapted"], xlim=(0, 20), xticks=range(0, 21, 2),
           xlabel="Executed cases (20 identical source cases per model)")
    ax.set_title("Goal-retention execution completeness, not goal success", fontsize=13, pad=14)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="Full 250-action planned horizon"), Patch(color=ORANGE, label="Early unverified terminal: 2 / 4")],
              loc="upper center", bbox_to_anchor=(.5, -.35), ncol=2, frameon=False)
    fig.suptitle("Push-T E2: saved clip errors improve; paired goal retention remains incomplete\n400 held-out clips; values shown without confidence intervals", fontsize=16, y=.98)
    fig.subplots_adjust(left=.08, right=.98, top=.83, bottom=.2)
    fig.text(.025, .025, "Pose error is against the frozen real-image readout, not simulator ground truth. Full-horizon counts do not establish goal attainment.\n"
             "Two no-update and four adapted cases stop early below verified goal coverage; no early goal completion is verified.\n"
             "The retained incomplete cases prevent the paired retention criterion from passing; this is not a completed equivalence test.", fontsize=10)
    finish(fig, output, "retention_summary")


def plot_cost(data, output):
    fig, ax = plt.subplots(figsize=(12, 6))
    groups = data["costs"]["groups"]
    labels, values = list(groups), [d["steps"] for d in groups.values()]
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=BLUE, height=.55)
    for pos, value in zip(y, values, strict=True):
        ax.text(value + 250, pos, f"{value:,}", va="center", fontsize=12)
    ax.set(yticks=y, yticklabels=labels, xlim=(0, max(values)*1.19), xlabel="Charged simulator steps")
    ax.grid(axis="x", alpha=.15)
    ax.set_axisbelow(True)
    ax.set_title("E2 charged simulator cost: 58,475 steps", fontsize=17, pad=20)
    fig.subplots_adjust(left=.32, bottom=.24, right=.97, top=.83)
    fig.text(.025, .055, "Includes failed/discarded adaptation collection, queried branches, repeated context replays and all goal-retention executions.\n"
             "Excludes upstream evaluation-bank generation and diagnosis. Training/model-query compute is separate from simulator steps.\n"
             "This presentation adds zero simulator steps, model queries or optimizer updates.", fontsize=10)
    finish(fig, output, "charged_simulator_cost")


def write_json(path, data):
    with path.open("x") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-dir", type=Path, required=True)
    parser.add_argument("--completion-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    repair_dir, audit_path, output = args.repair_dir.resolve(), args.completion_audit.resolve(), args.output_dir.resolve()
    require(output.parent == repair_dir.parent and output != repair_dir and not output.exists(), "Require a fresh sibling output directory")
    original = tree_hashes(repair_dir)
    source_paths = [repair_dir / "repair.json", repair_dir / "paired-report/paired_report.json", audit_path, Path(__file__)]
    sources = {str(p.relative_to(ROOT)): file_hash(p) for p in source_paths}
    repair, paired, audit = [json.loads(p.read_text()) for p in source_paths[:3]]
    require(file_hash(repair_dir / "repair.json") == paired["inputs"]["repair_report"]["sha256"] == audit["report_sha256"], "Repair report hash mismatch")
    require(original["repair_rows.npz"] == paired["inputs"]["evaluation_rows"]["sha256"] == audit["rows_sha256"], "Saved row-file identity mismatch")
    require(original["goal_retention_cases.json"] == audit["case_sha256"], "Goal case-file hash mismatch")
    tree_digest = hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
    expected = audit["workflow"]["output_hashes"][str(repair_dir)]
    require(expected == {"kind": "directory", "sha256": tree_digest, "n_files": len(original)}, "Original completed result tree differs")
    require(audit["workflow"]["status"] == "gate_stopped" and audit["workflow"]["returncode"] == 0, "Completion audit is not terminal")
    data = prepare(paired, repair, audit)
    data["input_sha256"] = sources
    data["original_repair_tree"] = expected
    data["generated_utc"] = datetime.now(timezone.utc).isoformat()
    output.mkdir()
    plot_safety(data, output)
    plot_retention(data, output)
    plot_cost(data, output)
    write_json(output / "chart_data.json", data)
    lines = ["# Saved E2 summary charts", "", "These figures present immutable saved results. No bootstrap, model, simulator, training, recalibration or example selection was performed.", "",
             "## Test decisions", "", "All four point FSAs are undefined. Lines show censor identification bounds, not confidence intervals or midpoint estimates. Each arm retains its development-set threshold; achieved test acceptance rates differ.", "",
             "| Arm | Margin px | Accepted / total | AR | Known false-safe | Accepted unresolved | FSA bounds |", "|---|---:|---:|---:|---:|---:|---|"]
    for arm in ARMS:
        d = data["test"]["arms"][arm]
        lines.append(f"| {arm} | {d['margin_px']:.6f} | {d['accepted']}/{d['n']} | {100*d['acceptance_rate']:.2f}% | {d['known_false_safe']} | {d['accepted_unresolved']} | {100*d['fsa_lower']:.2f}% to {100*d['fsa_upper']:.2f}% |")
    lines += ["", "The fixed-margin control uses the recorded 80 px fallback because its development FSA target is undefined. The chosen adaptation recipe also used the recorded fallback: zero of 16 candidates met the saved eligibility rule. Neither fallback is reselected here.", "",
              "The conservative fixed-set relative-reduction outer bound is 5.697695% to 18.046952%, computed exactly from the integer counts. Its upper bound is below the 25% gate target. Shared labels need not jointly attain the endpoints. This is not a confidence interval, population-effect estimate, or evidence of no positive gain.", "",
              "## Retention", "", "Saved 400-clip errors are shown in separate units. Pose error is against the real-image readout, not ground-truth geometry. No evaluation was regenerated.", "",
              "| Metric | No update | Adapted | Units |", "|---|---:|---:|---|"]
    for metric, unit in data["clip_retention"]["units"].items():
        lines.append(f"| {metric} | {data['clip_retention']['arms']['no_update'][metric]:.9f} | {data['clip_retention']['arms']['adapted'][metric]:.9f} | {unit} |")
    lines += ["", "Goal-retention execution: 18 baseline and 16 adapted cases reached all 250 planned actions; 2 and 4 ended early without verified goal completion. These are not goal-success counts. All 20 paired cases per model are retained; incomplete outcomes prevent a completed paired-retention conclusion.", "",
              "## Cost", "", "58,475 charged E2 simulator steps. Failed/discarded collection and repeated context queries are included; upstream E1 bank generation and diagnosis are excluded. Model queries and optimizer work are separate costs.", "", "| Category | Steps |", "|---|---:|"]
    for name, d in data["costs"]["groups"].items():
        lines.append(f"| {name} | {d['steps']:,} |")
    lines += ["", "Each chart is provided as PNG and vector PDF. chart_data.json retains all plotted numbers, exact fractions, case identities and source pointers. The original saved scientific gate remains false."]
    (output / "README.md").write_text("\n".join(lines) + "\n")
    require(tree_hashes(repair_dir) == original and all(file_hash(ROOT / p) == h for p, h in sources.items()), "An original changed during plotting")
    write_json(output / "manifest.json", {"kind": "saved_e2_summary_charts", "run_id": repair["run_id"],
        "input_sha256": sources, "original_repair_files_sha256": original, "originals_unchanged": True,
        "upstream_revisions": audit["receipt"], "new_queries": {"simulator": 0, "model": 0, "bootstrap": 0},
        "outputs_sha256": {p.name: file_hash(p) for p in sorted(output.iterdir())},
        "limitations": data["limits"]})
    print(json.dumps({"status": "written", "path": str(output), "manifest_sha256": file_hash(output / "manifest.json")}))


if __name__ == "__main__":
    main()
