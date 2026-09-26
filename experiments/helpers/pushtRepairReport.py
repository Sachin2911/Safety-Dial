"""CPU-only E2 paired tables and deterministic illustrations from saved observations.

This module never simulates, evaluates a model, chooses a margin, or changes a gate.
Examples are the first qualifying paired row in a declared order, not prevalence estimates.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import h5py
import numpy as np

from helpers.dialMetrics import fsa
from helpers.runManifest import file_sha256
from helpers.splitIntegrity import bank_identity

PROTOCOL = "saved-e2-paired-report-v1"
CATEGORIES = {
    "corrected_false_safe": "Known unsafe: no update accepts, adapted rejects",
    "new_false_safe": "Known unsafe: no update rejects, adapted accepts",
    "remaining_false_safe": "Known unsafe: both models accept",
    "restored_safe_acceptance": "Fully observed safe: no update rejects, adapted accepts",
    "new_false_rejection": "Fully observed safe: no update accepts, adapted rejects",
    "unresolved_accepted_future": "No observed violation, censored future, either model accepts",
}
TRUTH_FIELDS = ("cmin_dense", "unsafe_composite", "censored", "observation_domain_exit",
                "hazard_unsafe_observed", "observed_steps", "horizon_steps")


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def row_key(row):
    return (str(row["bank"]), int(row["root"]), int(row["branch"]), str(row["layout"]))


def outcomes(row):
    unsafe = bool(row.get("unsafe_composite", row["cmin_dense"] <= 0))
    censored = bool(row.get("censored", False))
    return unsafe, censored, censored and not unsafe


def paired_rows(before, after):
    """Pair exact bank/root/branch/layout IDs, refusing duplicates or changed truth."""
    maps = []
    for rows in (before, after):
        mapped = {}
        for index, row in enumerate(rows):
            key = row_key(row)
            if key in mapped:
                raise ValueError(f"Duplicate evaluation row identity: {key}")
            mapped[key] = (index, row)
        maps.append(mapped)
    if maps[0].keys() != maps[1].keys():
        raise ValueError("Evaluation arms do not contain identical paired row IDs")
    result = []
    for key in sorted(maps[0]):
        ia, a = maps[0][key]
        ib, b = maps[1][key]
        if outcomes(a) != outcomes(b) or any(a.get(k) != b.get(k) for k in TRUTH_FIELDS):
            raise ValueError(f"Paired evaluation rows disagree on observed truth: {key}")
        result.append({"key": key, "before_index": ia, "after_index": ib, "before": a, "after": b})
    return result


def select_examples(pairs, before_margin, after_margin):
    """All paired rows contribute counts; illustrations use first sorted qualifying IDs."""
    selected = {name: {"definition": definition, "n_qualifying": 0, "example": None}
                for name, definition in CATEGORIES.items()}
    for pair in sorted(pairs, key=lambda p: p["key"]):
        a, b = pair["before"], pair["after"]
        unsafe, censored, unresolved = outcomes(a)
        aa, ab = a["cmin_imagined"] >= before_margin, b["cmin_imagined"] >= after_margin
        flags = {
            "corrected_false_safe": unsafe and aa and not ab,
            "new_false_safe": unsafe and not aa and ab,
            "remaining_false_safe": unsafe and aa and ab,
            "restored_safe_acceptance": not unsafe and not unresolved and not aa and ab,
            "new_false_rejection": not unsafe and not unresolved and aa and not ab,
            "unresolved_accepted_future": unresolved and (aa or ab),
        }
        for category, qualifies in flags.items():
            if not qualifies:
                continue
            selected[category]["n_qualifying"] += 1
            if selected[category]["example"] is None:
                selected[category]["example"] = {
                    "row_id": dict(zip(("bank", "root_index", "branch_index", "layout_family"), pair["key"], strict=True)),
                    "source_row_indices": {"no_update": pair["before_index"], "adapted": pair["after_index"]},
                    "decisions": {"no_update_accepts": bool(aa), "adapted_accepts": bool(ab),
                                  "no_update_margin": before_margin, "adapted_margin": after_margin},
                    "known_unsafe": unsafe, "censored": censored, "unresolved_future": unresolved,
                    "no_update_row": a, "adapted_row": b,
                }
    return selected


def build_paired_report(repair, rows_by_arm):
    if not {"no_update", "adapted"} <= rows_by_arm.keys():
        raise ValueError("Saved E2 rows need both no-update and adapted arms")
    banks = sorted(rows_by_arm["no_update"])
    if any(sorted(values) != banks for values in rows_by_arm.values()):
        raise ValueError("Saved E2 arms contain different bank sets")
    result = {"protocol": PROTOCOL, "run_id": repair["run_id"], "gate": repair["gate"],
        "selection_policy": {"order": "bank name, numeric root index, numeric branch index, layout family",
            "rule": "First qualifying paired row per category and bank; all rows retained in metrics and original NPZ",
            "interpretation": "Illustrations of decision changes, not prevalence estimates or evidence of successful closed-loop control",
            "margins": "Use each arm's recorded development-calibrated margin unchanged",
            "censoring": "An observed violation resolves unsafe even if a later suffix is censored; unobserved futures cannot establish safety"},
        "banks": {}, "costs": {"adaptation_bank": repair.get("adapt_bank", {}),
            "context_replay": repair.get("context_ledger"), "evaluation_replay": repair.get("evaluation_ledger"),
            "all_run_simulator_steps": repair.get("simulator_costs"),
            "accounting": repair.get("simulator_cost_accounting"),
            "chosen_recipe": repair.get("chosen_recipe", {}), "training": repair.get("train_log", {}),
            "recipe_grid": repair.get("recipe_grid", []), "wall_clock_s": repair.get("wall_clock_s"),
            "goal_retention_no_update": repair.get("goal_retention", {}).get("no_update", {}).get("ledger"),
            "goal_retention_adapted": repair.get("goal_retention", {}).get("adapted", {}).get("ledger"),
            "report_additional_simulator_steps": 0, "report_model_queries": 0, "report_optimizer_steps": 0},
        "retention": {"no_update_clips": repair.get("no_update_retention"),
            "adapted_clips": repair.get("adapted_retention"),
            "goal_comparison": repair.get("goal_retention", {}).get("comparison"),
            "goal_no_update_summary": repair.get("goal_retention", {}).get("no_update", {}).get("summary"),
            "goal_adapted_summary": repair.get("goal_retention", {}).get("adapted", {}).get("summary"),
            "case_file_sha256": repair.get("goal_retention_cases_sha256")}}
    margins = {arm: (entry["margin"] if arm == "fixed_margin" else entry["margin_matched_dev"])
               for arm, entry in repair["arms"].items()}
    for bank in banks:
        base = rows_by_arm["no_update"][bank]
        pairs = paired_rows(base, rows_by_arm["adapted"][bank])
        summaries = {}
        for arm, entry in repair["arms"].items():
            rows = rows_by_arm["no_update" if arm == "fixed_margin" else arm][bank]
            paired_rows(base, rows)
            unsafe, censored, _ = zip(*(outcomes(r) for r in rows), strict=True) if rows else ([], [], [])
            metrics = fsa(np.asarray([r["cmin_imagined"] for r in rows]), unsafe, margins[arm], censored=censored)
            saved = entry[bank]["at_matched"]
            for key in ("n", "n_accepted", "n_false_safe", "n_censored", "n_accepted_censored"):
                if metrics[key] != saved[key]:
                    raise ValueError(f"Saved E2 metrics disagree with complete paired rows: {arm}/{bank}/{key}")
            summaries[arm] = {"margin": margins[arm], "at_matched": metrics,
                "auc_dial": entry[bank].get("auc_dial"), "ordinary_motion": entry[bank].get("ordinary_motion"),
                "clearance_error": entry[bank].get("clearance_error")}
        result["banks"][bank] = {"n_rows": len(base), "n_roots": len({r["root"] for r in base}),
            "arms": summaries, "paired_adapted_minus_no_update": repair["arms"]["adapted"][bank].get("paired_fsa_diff_ci"),
            "examples": select_examples(pairs, margins["no_update"], margins["adapted"])}
    return result


def attach_observed_trace(example, bank_dir, roots, layouts, h5):
    """Read a selected branch, retaining observed exits and omitting padded future states."""
    identity = example["row_id"]
    ri, branch = identity["root_index"], identity["branch_index"]
    if int(h5["root_index"][branch]) != ri:
        raise ValueError("Illustrative row root does not match stored branch")
    root = roots[ri]
    layout = layouts[(root["root_id"], identity["layout_family"])]
    states = np.asarray(h5["states"][branch])
    observed = np.asarray(h5["observed"][branch], bool) if "observed" in h5 else np.ones(len(states), bool)
    valid = np.asarray(h5["observation_valid"][branch], bool) if "observation_valid" in h5 else (
        np.isfinite(states[:, :2]).all(axis=1) & (states[:, :2] >= 0).all(axis=1) & (states[:, :2] <= 512).all(axis=1))
    if not observed[0] or observed.shape != (len(states),):
        raise ValueError("Stored branch has an invalid observation mask")
    expected_censor = bool((~observed).any())
    if expected_censor != example["censored"]:
        raise ValueError("Saved row censoring differs from its stored observations")
    example.update({"root_id": root["root_id"], "source_episode": int(root["meta"]["episode"]),
        "bank_path": str(Path(bank_dir).resolve()), "layout": layout,
        "trace": {"observed_state_steps": np.flatnonzero(observed).tolist(),
            "states": states[observed].tolist(), "observation_valid": valid[observed].tolist(),
            "planned_horizon_steps": len(states) - 1,
            "unobserved_steps": np.flatnonzero(~observed).tolist(),
            "frame_steps_used": []}})
    frame_items = []
    if "frames" in h5:
        frames = h5["frames"][branch]
        frame_steps = np.arange(len(frames)) * 5
        usable = [i for i, step in enumerate(frame_steps) if step < len(states) and observed[step] and valid[step]]
        for i in sorted(set(usable[:1] + usable[-1:])):
            frame_items.append((int(frame_steps[i]), frames[i]))
        example["trace"]["frame_steps_used"] = [step for step, _ in frame_items]
    return frame_items


def plot_example(example, frames, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon, Rectangle
    from helpers.pushtGeometry import Box, hazard_from_dict, t_polygons

    states = np.asarray(example["trace"]["states"])
    fig, axes = plt.subplots(1, 1 + len(frames), figsize=(4 * (1 + len(frames)), 4), squeeze=False)
    ax = axes[0, 0]
    hz = hazard_from_dict(example["layout"]["hazard"])
    patch = (Rectangle((hz.x0, hz.y0), hz.x1 - hz.x0, hz.y1 - hz.y0) if isinstance(hz, Box)
             else Circle(hz.centre, hz.r))
    patch.set(facecolor="tab:red", alpha=.25, edgecolor="tab:red")
    ax.add_patch(patch)
    ax.plot(states[:, 0], states[:, 1], ".-", color="tab:orange", label="pusher")
    ax.plot(states[:, 2], states[:, 3], ".-", color="tab:blue", label="block body")
    for i in sorted({0, len(states) // 2, len(states) - 1}):
        for polygon in t_polygons(states[i, 2:5]):
            ax.add_patch(Polygon(polygon, fill=False, edgecolor="tab:blue", alpha=.3 + .6 * i / max(len(states) - 1, 1)))
    ax.set(xlim=(-20, 532), ylim=(532, -20), aspect="equal", title="Observed trajectory and virtual hazard")
    ax.legend(fontsize=7)
    for ax, (step, frame) in zip(axes[0, 1:], frames, strict=True):
        ax.imshow(frame)
        ax.set_title(f"Stored real frame, step {step}")
        ax.axis("off")
    row = example["row_id"]
    fig.suptitle(f"{row['bank']} / {example['root_id']} / branch {row['branch_index']} / {row['layout_family']}", fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=120)
    plt.close(fig)


def _number(value):
    return "undefined" if value is None or (isinstance(value, float) and not math.isfinite(value)) else f"{value:.4g}"


def markdown(report):
    lines = [f"# {report['run_id']}: paired repair report", "",
        f"Recorded repair gate passes: **{report['gate']['passes']}**. This report preserves the original gate and development-selected margins.", "",
        "All saved rows enter the tables. Known unsafe includes hazard contact and arena exits. An accepted unresolved future makes point FSA undefined; the lower/upper bounds retain the missing outcome.", "",
        "| Bank | Arm | Margin | False-safe / accepted / total | AR | FSA | FSA bounds | Accepted unresolved |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for bank, data in report["banks"].items():
        for arm, entry in data["arms"].items():
            metric = entry["at_matched"]
            lines.append(f"| {bank} | {arm} | {_number(entry['margin'])} | {metric['n_false_safe']} / {metric['n_accepted']} / {metric['n']} | {_number(metric['acceptance_rate'])} | {_number(metric['fsa'])} | [{_number(metric['fsa_lower'])}, {_number(metric['fsa_upper'])}] | {metric['n_accepted_censored']} |")
        ci = data["paired_adapted_minus_no_update"]
        if ci:
            point = ci.get("point")
            lines += ["", f"{bank}: paired adapted-minus-no-update FSA {_number(point)}, recorded 95% root-bootstrap interval [{_number(ci.get('lo'))}, {_number(ci.get('hi'))}], {ci.get('n_boot', 'unrecorded')} usable replicates."
                + (" The full-data contrast is undefined; finite conditional bootstrap limits do not establish an improvement." if point is None or not math.isfinite(point) else "")]
    costs = report["costs"]
    ledger = costs["adaptation_bank"].get("ledger", {})
    lines += ["", "## Charged costs and retention", "",
        f"Adaptation collection: {json.dumps(ledger, sort_keys=True)}.", "",
        f"All actual E2 simulator work, including context replays and goal retention: `{json.dumps(costs.get('all_run_simulator_steps'), sort_keys=True)}`. Existing bank-generation costs remain in upstream manifests.", "",
        f"Chosen recipe: `{json.dumps(costs['chosen_recipe'], sort_keys=True)}`. Final training time: {_number(costs['training'].get('wall_clock_s'))} s. Recorded total run time: {_number(costs['wall_clock_s'])} s.", "",
        "Recipe-grid training costs, final training log, separate baseline/adapted goal-retention ledgers and ordinary-motion summaries are preserved in paired_report.json. Generating this report adds zero simulator steps, model queries or optimizer updates.", "",
        "| Retention | No update | Adapted |",
        "|---|---:|---:|"]
    retention = report["retention"]
    for key in ("latent_mse_tf", "latent_mse_rollout"):
        lines.append(f"| {key} | {_number((retention['no_update_clips'] or {}).get(key))} | {_number((retention['adapted_clips'] or {}).get(key))} |")
    for key in ("mean_final_coverage", "mean_final_pose_error_px", "arena_exits", "n_valid"):
        lines.append(f"| goal {key} | {_number((retention['goal_no_update_summary'] or {}).get(key))} | {_number((retention['goal_adapted_summary'] or {}).get(key))} |")
    lines += ["", f"Recorded paired goal-retention comparison: `{json.dumps(json_safe(retention['goal_comparison']), sort_keys=True)}`.", "",
        "## Deterministic illustrations", "",
        "For each category and bank, show the first qualifying row sorted by numeric root index, branch index and layout family. Categories may be absent. These examples illustrate decisions, not prevalence or closed-loop goal-reaching success. No case was selected for a large error magnitude; every unselected outcome remains in the tables and original hashed NPZ. Frames and traces come only from recorded observations; padded future frames are omitted."]
    for bank, data in report["banks"].items():
        for category, selected in data["examples"].items():
            lines += ["", f"### {bank}: {category}", "", selected["definition"] + f". Qualifying rows: {selected['n_qualifying']}."]
            example = selected["example"]
            if example is None:
                lines.append("No qualifying example in this bank.")
                continue
            row = example["row_id"]
            lines += ["", f"Root `{example['root_id']}` (source episode {example['source_episode']}), branch {row['branch_index']}, layout `{row['layout_family']}`; saved row indices `{example['source_row_indices']}`. Predicted minimum clearance: {_number(example['no_update_row']['cmin_imagined'])} to {_number(example['adapted_row']['cmin_imagined'])} px; margins {_number(example['decisions']['no_update_margin'])} and {_number(example['decisions']['adapted_margin'])} px. Censored={example['censored']}, unresolved={example['unresolved_future']}.", "",
                f"![Recorded illustration]({example['figure']})"]
    return "\n".join(lines) + "\n"


def write_repair_report(repair_path, rows_path, banks_dir, output_dir):
    """Create a fresh report from immutable saved inputs; no simulator/model imports."""
    repair_path, rows_path, banks_dir, output_dir = map(Path, (repair_path, rows_path, banks_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError(f"Preserving existing paired report: {output_dir}")
    input_hashes = {"repair.json": file_sha256(repair_path), "repair_rows.npz": file_sha256(rows_path)}
    repair = json.loads(repair_path.read_text())
    if Path(repair["banks_dir"]).resolve() != banks_dir.resolve():
        raise ValueError("Illustration bank directory differs from saved E2 inputs")
    with np.load(rows_path, allow_pickle=False) as saved:
        rows = json.loads(str(saved["rows"]))
    result = build_paired_report(repair, rows)
    identities = {bank: bank_identity(banks_dir / bank) for bank in result["banks"]}
    result["inputs"] = {"repair_report": {"path": str(repair_path.resolve()), "sha256": input_hashes["repair.json"]},
        "evaluation_rows": {"path": str(rows_path.resolve()), "sha256": input_hashes["repair_rows.npz"]},
        "bank_identities": identities, "upstream": repair.get("upstream", {})}
    output_dir.mkdir(parents=True, exist_ok=False)
    for bank, data in result["banks"].items():
        directory = banks_dir / bank
        roots = json.loads((directory / "roots.json").read_text())["roots"]
        layouts = {(r["root_id"], r["family"]): r for r in json.loads((directory / "layouts.json").read_text())["layouts"]}
        with h5py.File(directory / "branches.h5", "r") as h5:
            for category, selected in data["examples"].items():
                example = selected["example"]
                if example is None:
                    continue
                frames = attach_observed_trace(example, directory, roots, layouts, h5)
                filename = f"{bank}-{category}.png"
                plot_example(example, frames, output_dir / filename)
                example["figure"] = filename
    if file_sha256(repair_path) != input_hashes["repair.json"] or file_sha256(rows_path) != input_hashes["repair_rows.npz"]:
        raise ValueError("Saved E2 inputs changed during reporting")
    result = json_safe(result)
    (output_dir / "paired_report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (output_dir / "paired_report.md").write_text(markdown(result))
    manifest = {"kind": "saved-e2-paired-report", "protocol": PROTOCOL, "run_id": repair["run_id"],
        "inputs": result["inputs"], "upstream_manifest": repair.get("manifest"),
        "report_source_sha256": file_sha256(Path(__file__)),
        "outputs_sha256": {p.name: file_sha256(p) for p in sorted(output_dir.iterdir()) if p.is_file()},
        "costs": {"additional_simulator_steps": 0, "model_queries": 0, "optimizer_steps": 0}}
    (output_dir / "manifest.json").write_text(json.dumps(json_safe(manifest), indent=2, allow_nan=False) + "\n")
    return output_dir / "paired_report.json"
