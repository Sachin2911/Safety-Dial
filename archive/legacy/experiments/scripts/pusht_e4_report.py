#!/usr/bin/env python3
"""Build the CPU-only E4 transfer report from immutable E3 rows and checkpoint retention.

No model, simulator or external upload is used. Margins stay frozen from E3 development
calibration. Representative/test and stress banks remain separate throughout reporting.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.threads import pin_threads

pin_threads()
import h5py
import numpy as np

from helpers.pushtRetention import case_identity, compare_retention
from helpers.pushtReplay import Root
from helpers.pushtTransferReport import (
    CELLS, check_acquisition, h4_conclusion, load_rows, paired_statistics, source_metadata,
    summarize_variant, validate_layouts, validate_rows,
)
from helpers.runManifest import build_manifest, file_sha256, make_run_id, validate_run_id, write_manifest
from helpers.splitIntegrity import bank_identity, validate_bank_splits


def retention_links(path, acquisition, e2):
    report = json.loads(path.read_text())
    if report.get("status") != "complete" or report.get("acquisition_run") != acquisition["run_id"]:
        raise ValueError("E4 requires the completed retention report for this acquisition study")
    cases_path = Path(e2["goal_retention_cases"])
    digest = file_sha256(cases_path)
    if digest != e2["goal_retention_cases_sha256"] or report.get("case_file_sha256") != digest:
        raise ValueError("E4 retention does not use the frozen E2 case file")
    if report.get("n_blocks") != e2["goal_retention"]["n_blocks"]:
        raise ValueError("E4 retention horizon differs from the frozen E2 protocol")
    expected = [case_identity(Root.from_dict(row)) for row in json.loads(cases_path.read_text())]
    links = {}
    for name, entry in report.get("checkpoints", {}).items():
        detail_path = Path(entry["report"])
        detail = json.loads(detail_path.read_text())
        if (detail.get("repaired_weights_sha256") != entry["weights_sha256"]
                or detail.get("case_file_sha256") != digest
                or detail.get("n_blocks") != report["n_blocks"]):
            raise ValueError("Retention checkpoint or frozen protocol identity changed")
        for variant in ("base", "adapted"):
            if expected != [row.get("case_sha256") for row in detail[variant]["rows"]]:
                raise ValueError("Retention rows differ from the frozen E2 cases")
        comparison = compare_retention(detail["base"], detail["adapted"])
        if comparison["passes"] != detail["comparison"]["passes"]:
            raise ValueError("Retention pass label disagrees with its stored episode outcomes")
        links[name] = {"status": "measured", "weights_sha256": entry["weights_sha256"],
            "report": str(detail_path.resolve()), "sha256": file_sha256(detail_path),
            "case_file_sha256": digest, "comparison": comparison}
    return links


def plain_json(value):
    if isinstance(value, dict):
        return {k: plain_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def number(value):
    return f"{value:.3f}" if value is not None and np.isfinite(value) else "undefined"


def write_markdown(report, output):
    lines = [f"# {report['run_id']}", "", "E4 geometric transfer from the frozen E3 evaluation banks.", "",
        "Margins were calibrated only on development data and then applied unchanged. Acceptance may differ across transfer cells; both acceptance and false-safe counts are reported. Intervals resample source episodes with all sibling roots and tapes together. Zero accepted plans and unresolved accepted futures have undefined point FSA. Only usable bootstrap replicates contribute to intervals. A finite interval from those replicates does not resolve an undefined full-data estimate, so it cannot establish a directional claim or reference advantage.", "",
        "Representative/test and stress results are separate. Held-out states are held out from this study's development and additional adaptation, not necessarily from the released checkpoint's pretraining or the physical probe's coverage.", "",
        "| Bank | Method | Seed | Additional branches | Hazard/source | Roots/sources | Accepted/total | False-safe | AR | FSA [95% interval] | Accepted unresolved | Bootstrap usable/requested (FSA; AR) |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---|---:|---|"]
    variants = [("no_update", "-", 0, report["no_update"])]
    for arm, seeds in report["arms"].items():
        for seed, budgets in seeds.items():
            variants.extend((arm, seed, int(budget), point["evaluation"]) for budget, point in budgets.items())
    for arm, seed, budget, evaluation in variants:
        for bank in ("test", "stress"):
            for cell in CELLS:
                row = evaluation[bank]["cells"][cell]
                ci = row["intervals"]["fsa"]
                ar_ci = row["intervals"]["acceptance_rate"]
                lines.append(f"| {bank} | {arm} | {seed} | {budget} | {cell} | {row['n_roots']}/{row['n_source_episodes']} | {row['n_accepted']}/{row['n']} | {row['n_false_safe']} | {number(row['acceptance_rate'])} | {number(row['fsa'])} [{number(ci['lo'])}, {number(ci['hi'])}] | {row['n_accepted_censored']} | {ci['n_boot_usable']}/{ci['n_boot_requested']}; {ar_ci['n_boot_usable']}/{ar_ci['n_boot_requested']} |")
    lines += ["", "## H4: change in the boundary advantage", "",
        "The contrast is (boundary minus random FSA) in the held-out cell minus that difference in the familiar-hazard/familiar-source cell. Positive values mean the boundary advantage shrinks. These exploratory intervals have no multiple-comparison adjustment; acquisition seeds are reported individually.", ""]
    for bank, cells in report["h4"]["by_bank"].items():
        for cell, result in cells.items():
            lines.append(f"- {bank}, {cell}: {result['conclusion']} ({result['per_seed']}).")
    lines += ["", "| Bank | Seed | Additional branches | Hazard/source | Boundary-random FSA [95% interval] | Bootstrap usable/requested | H4 change [95% interval] | Bootstrap usable/requested | Interpretation |",
        "|---|---|---:|---|---|---:|---|---:|---|"]
    for budget, seeds in report["paired_boundary_random"].items():
        for seed, banks in seeds.items():
            for bank, paired in banks.items():
                for cell in CELLS:
                    delta = paired["boundary_minus_random_fsa"][cell]
                    change = paired["h4_advantage_change"].get(cell)
                    change_text = (f"{number(change['point'])} [{number(change['lo'])}, {number(change['hi'])}]"
                                   if change else "reference")
                    usable = f"{change['n_boot_usable']}/{change['n_boot_requested']}" if change else "-"
                    interpretation = (change["interpretation"] if change else
                                      "reference_cell" if delta["point_defined"] else "undefined_outcome")
                    lines.append(f"| {bank} | {seed} | {budget} | {cell} | {number(delta['point'])} [{number(delta['lo'])}, {number(delta['hi'])}] | {delta['n_boot_usable']}/{delta['n_boot_requested']} | {change_text} | {usable} | {interpretation} |")
    lines += ["", "## Retention and ordinary motion", "",
        "Goal retention is bound to exact checkpoint hashes and E2's fixed cases and maximum horizon. Every reported arm and seed at the final budget has an individual result; tested earlier E5-selected budgets are linked when available. Intermediate checkpoints without a result are explicitly marked unmeasured. Ordinary-motion summaries and held-out expert latent prediction metrics are retained per checkpoint in transfer.json.", ""]
    for arm, seeds in report["arms"].items():
        for seed, budgets in seeds.items():
            for budget, point in budgets.items():
                retention = point["goal_retention"]
                if retention["status"] == "measured":
                    lines.append(f"- {arm}, seed {seed}, budget {budget}: goal-retention passes={retention['comparison']['passes']}; complete outcome evidence={retention['comparison']['complete']}.")
    lines += ["", "No new simulator interaction, predictor update or model-query cost is incurred by this report. Acquisition costs remain those recorded by E3; goal-retention costs are recorded separately in the linked retention report.", ""]
    output.write_text("\n".join(lines))


def write_figure(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    budget = str(report["largest_budget"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, bank in zip(axes, ("test", "stress")):
        for i, seed in enumerate(report["acquisition_seeds"]):
            data = report["paired_boundary_random"][budget][seed][bank]["boundary_minus_random_fsa"]
            values = [data[cell] for cell in CELLS]
            x = np.arange(4) + (i - (len(report["acquisition_seeds"]) - 1) / 2) * .08
            valid = np.array([np.isfinite(v["point"]) and np.isfinite(v["lo"]) and np.isfinite(v["hi"]) for v in values])
            y = np.array([v["point"] for v in values])
            lo, hi = np.array([v["lo"] for v in values]), np.array([v["hi"] for v in values])
            points = ax.plot(x[valid], y[valid], "o", label=f"acquisition seed {seed}")
            color = points[0].get_color()
            # Percentile intervals need not contain their point estimate. Draw their
            # actual endpoints rather than clipping negative error-bar lengths.
            ax.vlines(x[valid], lo[valid], hi[valid], color=color)
            ax.hlines(lo[valid], x[valid] - .02, x[valid] + .02, color=color)
            ax.hlines(hi[valid], x[valid] - .02, x[valid] + .02, color=color)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(range(4), ["familiar / familiar", "familiar / heldout", "heldout / familiar", "heldout / heldout"], rotation=20)
        ax.set_xlabel("hazard layout / start-goal family")
        ax.set_title(f"{'Representative' if bank == 'test' else 'Stress'} bank; +{budget} branches")
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Boundary minus random FSA\nnegative favours boundary")
    axes[1].legend(fontsize=8)
    fig.suptitle("Frozen development margins; source-episode bootstrap 95% intervals")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acquisition-report", type=Path, required=True)
    ap.add_argument("--retention-report", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--bootstrap-seed", type=int, default=20261103)
    ap.add_argument("--run-id", type=validate_run_id)
    args = ap.parse_args()
    if args.bootstrap < 100:
        ap.error("Use at least 100 source-bootstrap resamples; default is 1000")
    if args.output_dir.exists():
        raise FileExistsError(f"Preserving existing E4 report: {args.output_dir}")
    acquisition = json.loads(args.acquisition_report.read_text())
    seeds, budgets = check_acquisition(acquisition)
    e2_path = Path(acquisition["e2_report"])
    if acquisition.get("e2_report_sha256") != file_sha256(e2_path):
        raise ValueError("E2 evidence changed after acquisition")
    e2 = json.loads(e2_path.read_text())
    links = retention_links(args.retention_report, acquisition, e2)
    bank_paths = {name: Path(acquisition["banks_dir"]) / name for name in ("dev", "test", "stress")}
    split_audit = validate_bank_splits(bank_paths)
    sources, expected = {}, {}
    for name, path in bank_paths.items():
        if bank_identity(path) != acquisition["evaluation_bank_identity"][name]:
            raise ValueError("Frozen evaluation bank changed after E3")
        if name == "dev":
            continue
        sources[name] = source_metadata(json.loads((path / "roots.json").read_text())["roots"])
        validate_layouts(json.loads((path / "layouts.json").read_text())["layouts"], sources[name])
        with h5py.File(path / "branches.h5", "r") as bank:
            expected[name] = {(int(root), branch, hazard) for branch, root in enumerate(bank["root_index"][:])
                              for hazard in ("familiar", "heldout")}
    kwargs = {"n_boot": args.bootstrap, "seed": args.bootstrap_seed}
    base_rows = load_rows(args.acquisition_report.parent, acquisition["no_update_rows"])
    report = {"run_id": args.run_id or make_run_id("pusht", "transfer", "e4"), "status": "complete",
        "acquisition_run": acquisition["run_id"], "source_family_protocol": acquisition["source_family_protocol"],
        "split_audit": split_audit, "acquisition_seeds": seeds, "budgets": budgets, "largest_budget": budgets[-1],
        "target_acceptance_rate_dev": acquisition["target_acceptance_rate_dev"], "no_update": {},
        "arms": {}, "paired_boundary_random": {}, "bootstrap": kwargs,
        "retention_report": {"path": str(args.retention_report.resolve()), "sha256": file_sha256(args.retention_report)},
        "additional_simulator_steps": 0, "additional_model_queries": 0,
        "no_update_expert_prediction_retention": acquisition["no_update_retention"]}
    for bank in ("test", "stress"):
        validate_rows(base_rows[bank], sources[bank], expected[bank])
        report["no_update"][bank] = summarize_variant(base_rows[bank], sources[bank], acquisition["no_update"]["margin_matched_dev"], **kwargs)
    all_rows = {}
    for arm, values in acquisition["arms"].items():
        report["arms"][arm] = {}
        for seed, curve in values.items():
            report["arms"][arm][seed] = {}
            for point in curve["curve"]:
                budget = point["budget_added"]
                rows = load_rows(args.acquisition_report.parent, point["evaluation_rows"])
                all_rows[arm, seed, budget] = rows
                name = f"{acquisition['run_id']}-{arm}-s{seed}-b{budget}"
                retention = links.get(name, {"status": "not_measured_at_this_budget"})
                if (budget == budgets[-1] and retention["status"] != "measured"):
                    raise ValueError("Missing final-budget checkpoint goal retention")
                if retention["status"] == "measured" and retention["weights_sha256"] != point["weights_sha256"]:
                    raise ValueError("Retention and E4 rows refer to different checkpoints")
                result = {"weights_sha256": point["weights_sha256"], "goal_retention": retention,
                    "expert_prediction_retention": point["retention"], "charged_steps": point["charged_steps"],
                    "evaluation": {}, "evaluation_rows": point["evaluation_rows"],
                    "acquisition_costs": {key: point[key] for key in ("ledger", "train_time_s",
                        "training_time_cumulative_s", "scoring_time_cumulative_s", "n_queried_branches") if key in point}}
                for bank in ("test", "stress"):
                    validate_rows(rows[bank], sources[bank], expected[bank])
                    result["evaluation"][bank] = summarize_variant(rows[bank], sources[bank], point["eval"]["margin_matched_dev"], **kwargs)
                report["arms"][arm][seed][str(budget)] = result
                print(f"[e4] summarized {arm} seed={seed} budget={budget}", flush=True)
    for budget in budgets:
        report["paired_boundary_random"][str(budget)] = {}
        for seed in seeds:
            left = next(p for p in acquisition["arms"]["random"][seed]["curve"] if p["budget_added"] == budget)
            right = next(p for p in acquisition["arms"]["boundary"][seed]["curve"] if p["budget_added"] == budget)
            if left["charged_steps"] != right["charged_steps"]:
                raise ValueError("Paired E4 acquisition costs differ")
            report["paired_boundary_random"][str(budget)][seed] = {bank: paired_statistics(
                all_rows["random", seed, budget][bank], all_rows["boundary", seed, budget][bank], sources[bank],
                left["eval"]["margin_matched_dev"], right["eval"]["margin_matched_dev"], **kwargs)
                for bank in ("test", "stress")}
    report["h4"] = h4_conclusion(report["paired_boundary_random"], seeds, budgets[-1])
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "transfer.json").write_text(json.dumps(plain_json(report), indent=2, allow_nan=False) + "\n")
    write_markdown(report, args.output_dir / "transfer.md")
    write_figure(report, args.output_dir / "transfer.png")
    write_manifest(args.output_dir, build_manifest(run_id=report["run_id"], kind="transfer-report",
        seeds={"bootstrap": args.bootstrap_seed}, data={"acquisition_report_sha256": file_sha256(args.acquisition_report),
            "e2_report_sha256": file_sha256(e2_path), "retention_report_sha256": file_sha256(args.retention_report),
            "evaluation_bank_identity": acquisition["evaluation_bank_identity"], "no_update_rows": acquisition["no_update_rows"]},
        costs={"additional_simulator_steps": 0, "additional_model_queries": 0},
        metrics={"largest_budget": budgets[-1], "n_acquisition_seeds": len(seeds)}))
    print(f"[e4] complete: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
