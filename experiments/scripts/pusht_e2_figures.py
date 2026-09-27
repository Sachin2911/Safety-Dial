#!/usr/bin/env python3
"""Create an E2 presentation supplement from immutable saved results and frames.

No model, simulator, optimizer, new example selection or uncertainty computation.
The original report directory and saved scientific gate remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.pushtGeometry import t_polygons  # noqa: E402

ARMS = ("no_update", "adapted", "readout_correction", "fixed_margin")
CATEGORIES = {
    "corrected_false_safe": "Corrected false-safe decision",
    "new_false_safe": "New false-safe decision",
    "remaining_false_safe": "Remaining false-safe decision",
    "restored_safe_acceptance": "Restored safe acceptance",
    "new_false_rejection": "New false rejection",
    "unresolved_accepted_future": "Accepted unresolved future",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(directory):
    return {str(path.resolve()): sha256(path) for path in sorted(directory.rglob("*"))
            if path.is_file()}


def finite(value):
    return value is not None and math.isfinite(float(value))


def number(value):
    return f"{float(value):.6g}" if finite(value) else "undefined"


def selection_context(repair):
    """Describe the saved selection; never choose or substitute a recipe."""
    grid = repair["recipe_grid"]
    if not grid:
        raise ValueError("No saved development recipe grid")
    base_mse = repair["no_update_retention"]["latent_mse_tf"]
    eligible = [row for row in grid if finite(row["dev"]["at_matched"]["fsa"])
                and row["retention"]["latent_mse_tf"] <= 1.15 * base_mse]
    all_undefined = all(not finite(row["dev"]["at_matched"]["fsa"]) for row in grid)
    first_fallback = not eligible and all_undefined
    if first_fallback and repair["chosen_recipe"] != grid[0]["cfg"]:
        raise ValueError("Saved all-undefined selection does not match the first grid recipe")
    fixed = repair["arms"]["fixed_margin"]["margin"]
    undefined_target = not finite(repair["arms"]["adapted"]["dev"]["at_matched"]["fsa"])
    if undefined_target and fixed != 80:
        raise ValueError("Undefined-target fixed-margin fallback is not the saved margin80")
    return {"recipe_count": len(grid), "eligible_count": len(eligible),
            "all_development_fsa_undefined": all_undefined,
            "first_recipe_fallback": first_fallback,
            "chosen_recipe": repair["chosen_recipe"],
            "fixed_margin": fixed, "fixed_margin_undefined_target_fallback": undefined_target}


def metric_cells(metric):
    """Format recorded points and bounds without promoting bounds to estimates."""
    accepted = metric["n_accepted"]
    unresolved = metric["n_accepted_censored"]
    if (not accepted or unresolved) and finite(metric["fsa"]):
        raise ValueError("Zero acceptance or unresolved acceptance cannot have defined FSA")
    if not accepted and any(finite(metric[k]) for k in ("fsa_lower", "fsa_upper")):
        raise ValueError("Zero acceptance cannot have defined FSA bounds")
    return [f"{metric['n_false_safe']} / {accepted} / {metric['n']}",
            number(metric["acceptance_rate"]), number(metric["fsa"]),
            f"[{number(metric['fsa_lower'])}, {number(metric['fsa_upper'])}]", str(unresolved)]


def plot_limits(states, hazard, *, goal_pose=None, padding=20.0):
    """Include the arena, all saved bodies/paths, hazard and any displayed goal."""
    states = np.asarray(states, dtype=float)
    if (states.ndim != 2 or not len(states) or states.shape[1] < 5
            or not np.isfinite(states[:, :5]).all() or not finite(padding) or padding <= 0):
        raise ValueError("Finite recorded poses and positive padding are required")
    points = [np.array([[0.0, 0.0], [512.0, 512.0]]), states[:, :2], states[:, 2:4]]
    for state in states:
        points.extend(t_polygons(state[2:5]))
    if goal_pose is not None:
        goal = np.asarray(goal_pose, dtype=float)
        if goal.shape != (3,) or not np.isfinite(goal).all():
            raise ValueError("Goal pose must be finite x/y/angle")
        points.extend(t_polygons(goal))
    if hazard["kind"] == "box":
        x0, x1, y0, y1 = (float(hazard[k]) for k in ("x0", "x1", "y0", "y1"))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Invalid saved box")
    elif hazard["kind"] == "disc":
        cx, cy, radius = (float(hazard[k]) for k in ("cx", "cy", "r"))
        if radius <= 0:
            raise ValueError("Invalid saved disc")
        x0, x1, y0, y1 = cx - radius, cx + radius, cy - radius, cy + radius
    else:
        raise ValueError("Unknown saved hazard")
    points.append(np.array([[x0, y0], [x1, y1]]))
    points = np.concatenate(points)
    if not np.isfinite(points).all():
        raise ValueError("Nonfinite saved geometry")
    lo, hi = points.min(axis=0) - padding, points.max(axis=0) + padding
    return (float(lo[0]), float(hi[0])), (float(hi[1]), float(lo[1]))


def decision_text(example):
    parts = []
    for arm, label in (("no_update", "No update"), ("adapted", "Adapted")):
        decision = "ACCEPT" if example["decisions"][arm + "_accepts"] else "REJECT"
        parts.append(f"{label}: {decision}; predicted min clearance "
                     f"{number(example[arm + '_row']['cmin_imagined'])} px, "
                     f"margin {number(example['decisions'][arm + '_margin'])} px")
    return "   |   ".join(parts)


def horizon_text(example):
    trace = example["trace"]
    last = max(trace["observed_state_steps"])
    horizon = trace["planned_horizon_steps"]
    if example["unresolved_future"]:
        status = "UNRESOLVED FUTURE: no observed violation does not establish safety"
    elif example["known_unsafe"]:
        status = "Known unsafe from the recorded observations"
    elif example["censored"]:
        raise ValueError("Censored future lacks its unresolved or known-unsafe designation")
    else:
        status = "Fully observed safe trajectory under the saved rule"
    return f"{status}. Observed through step {last} of {horizon}; censored={example['censored']}."


def saved_frames(example, h5):
    """Load only the same recorded frames and verify the saved example trace."""
    branch = example["row_id"]["branch_index"]
    if int(h5["root_index"][branch]) != example["row_id"]["root_index"]:
        raise ValueError("Saved illustration root identity changed")
    observed = np.asarray(h5["observed"][branch], dtype=bool)
    trace = example["trace"]
    if np.flatnonzero(observed).tolist() != trace["observed_state_steps"]:
        raise ValueError("Saved observation mask changed")
    if not np.array_equal(np.asarray(h5["states"][branch])[observed], trace["states"]):
        raise ValueError("Saved observed states changed")
    frames = []
    for step in trace["frame_steps_used"]:
        if step % 5 or step >= len(observed) or not observed[step]:
            raise ValueError("Requested frame is not an observed block endpoint")
        if not bool(h5["observation_valid"][branch, step]):
            raise ValueError("Requested frame is outside the recorded observation domain")
        frames.append((step, np.asarray(h5["frames"][branch, step // 5])))
    return frames


def plot_example(bank, category, example, frames, path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon, Rectangle

    states = np.asarray(example["trace"]["states"])
    hazard = example["layout"]["hazard"]
    limits = plot_limits(states, hazard)
    fig, axes = plt.subplots(1, 1 + len(frames), figsize=(15, 6.8), squeeze=False)
    ax = axes[0, 0]
    ax.add_patch(Rectangle((0, 0), 512, 512, fill=False, edgecolor="#888888",
                           linestyle=":", linewidth=1, label="Arena boundary"))
    if hazard["kind"] == "box":
        patch = Rectangle((hazard["x0"], hazard["y0"]),
                          hazard["x1"] - hazard["x0"], hazard["y1"] - hazard["y0"])
    else:
        patch = Circle((hazard["cx"], hazard["cy"]), hazard["r"])
    patch.set(facecolor="#D55E00", alpha=0.3, edgecolor="#D55E00", label="Virtual hazard")
    ax.add_patch(patch)
    ax.plot(states[:, 0], states[:, 1], ".-", color="#E69F00", label="Pusher path")
    ax.plot(states[:, 2], states[:, 3], ".-", color="#0072B2", label="Block body path")
    for i in sorted({0, len(states) // 2, len(states) - 1}):
        for polygon in t_polygons(states[i, 2:5]):
            ax.add_patch(Polygon(polygon, fill=False, edgecolor="#0072B2",
                                 alpha=0.3 + 0.6 * i / max(len(states) - 1, 1)))
    ax.set(xlim=limits[0], ylim=limits[1], aspect="equal", xlabel="x (arena px)",
           ylabel="y (arena px; increases downward)", title="Recorded geometry")
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.92)
    for ax, (step, frame) in zip(axes[0, 1:], frames, strict=True):
        ax.imshow(frame)
        ax.set_title(f"Stored real frame, step {step}")
        ax.axis("off")
    identity = example["row_id"]
    fig.suptitle(f"{bank.capitalize()}: {CATEGORIES[category]}\n"
                 f"{example['root_id']} | branch {identity['branch_index']} | "
                 f"{identity['layout_family']} hazard", fontsize=13, y=0.98)
    fig.text(0.5, 0.215, decision_text(example), ha="center", fontsize=9)
    fig.text(0.5, 0.157, horizon_text(example), ha="center", fontsize=9,
             color="#7B3294" if example["unresolved_future"] else "#333333")
    fig.text(0.5, 0.10,
             "The geometry shows recorded truth, not predicted trajectories. "
             "Decision changes can reflect both predictions and the saved margins.",
             ha="center", fontsize=8.5)
    fig.text(0.5, 0.05,
             "Stored frames retain their original image bounds. "
             "The virtual hazard is shown only in the geometry panel; the green T is the goal.",
             ha="center", fontsize=8.5)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.81, bottom=0.30, wspace=0.23)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return {"xlim": list(limits[0]), "ylim": list(limits[1])}


def markdown_report(paired, selection):
    lines = [f"# {paired['run_id']}: presentation supplement", "",
             f"Recorded repair gate passes: **{paired['gate']['passes']}**. "
             "All results, margins, cases and the scientific gate are unchanged.", "",
             "This supplement explains the saved results; it adds no model queries, simulator "
             "steps, optimizer updates, example selection or uncertainty estimates.", "",
             "## Development selection and fixed-margin fallback", "",
             f"**{selection['eligible_count']} eligible recipes out of "
             f"{selection['recipe_count']} saved development evaluations.**"]
    if selection["first_recipe_fallback"]:
        lines += ["All development FSA point estimates were undefined. The chosen configuration "
                  "is the existing deterministic first-recipe fallback after inconclusive "
                  "development ranking, not a demonstrated development winner."]
    elif not selection["eligible_count"]:
        lines += ["The chosen recipe came from the existing unfiltered fallback, "
                  "not the eligible development list."]
    lines += ["", "Saved chosen recipe: `" + json.dumps(selection["chosen_recipe"], sort_keys=True)
              + "`."]
    if selection["fixed_margin_undefined_target_fallback"]:
        lines += ["", f"The **{number(selection['fixed_margin'])} px fixed margin** is also an "
                  "existing fallback: the adapted development FSA target is undefined, so the "
                  "matching comparison cannot succeed. This is not an achieved match to a "
                  "defined FSA target."]
    lines += ["", "## Recorded false-safe acceptance", "",
              "AR means acceptance rate. Point FSA is observed unsafe accepted / all accepted "
              "only when accepted outcomes are resolved. An accepted unresolved future or zero "
              "accepted plans keeps point FSA undefined. Known unsafe includes hazard contact "
              "and arena exit.", "",
              "**Censoring bounds are not confidence intervals.** They retain possible outcomes "
              "of accepted unresolved futures. Saved conditional bootstrap limits are reported "
              "separately and cannot resolve an undefined full-data contrast."]
    for bank, data in paired["banks"].items():
        if set(data["arms"]) != set(ARMS):
            raise ValueError("Presentation requires all four recorded E2 arms")
        lines += ["", f"### {bank.capitalize()} bank", "",
                  "| Arm | Margin (px) | False-safe / accepted / total | AR | Point FSA | "
                  "Censoring bounds (not CI) | Accepted unresolved |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for arm in ARMS:
            row = data["arms"][arm]
            cells = [arm, number(row["margin"]), *metric_cells(row["at_matched"])]
            lines.append("| " + " | ".join(cells) + " |")
        interval = data["paired_adapted_minus_no_update"]
        if interval:
            lines += ["", "Paired adapted-minus-no-update full-data FSA contrast: **"
                      + number(interval.get("point")) + "**. Saved root-bootstrap limits: ["
                      + number(interval.get("lo")) + ", " + number(interval.get("hi")) + "]; "
                      + str(interval.get("n_boot")) + " usable replicates."]
            if not finite(interval.get("point")):
                lines += ["These finite-replicate limits are **conditional diagnostics**, "
                          "not evidence of a directional improvement. The original nominal "
                          "95% bootstrap limits cannot establish a defined full-data contrast."]
    retention = paired["retention"]
    lines += ["", "## Retention and charged costs", "",
              "Goal retention remains **incomplete and failing**." if
              not paired["gate"]["retention_complete"] else
              "Goal-retention completion is preserved from the original report.", "",
              "Valid reset cases and observed endpoint averages do not establish completed "
              "goal-retention outcomes. The original paired comparison is preserved:", "",
              "`" + json.dumps(retention["goal_comparison"], sort_keys=True) + "`.", "",
              "| Recorded retention measure | No update | Adapted |", "|---|---:|---:|"]
    for key in ("latent_mse_tf", "latent_mse_rollout"):
        lines.append(f"| {key} | {number(retention['no_update_clips'][key])} | "
                     f"{number(retention['adapted_clips'][key])} |")
    for key in ("mean_final_coverage", "mean_final_pose_error_px", "arena_exits", "n_valid"):
        lines.append(f"| goal {key} | {number(retention['goal_no_update_summary'][key])} | "
                     f"{number(retention['goal_adapted_summary'][key])} |")
    costs = paired["costs"]
    lines += ["", "All-run charged simulator ledger, exactly as saved: `"
              + json.dumps(costs["all_run_simulator_steps"], sort_keys=True) + "`.", "",
              f"Final selected-retraining time: {number(costs['training']['wall_clock_s'])} s. "
              f"Recorded total E2 time: {number(costs['wall_clock_s'])} s.", "",
              "## Same deterministic illustrations", "",
              "These are the exact saved examples, with unchanged identities and categories. "
              "They illustrate decisions, not prevalence or successful closed-loop control. "
              "Predictions and arm-specific margins both affect acceptance. Only recorded "
              "observed states and the previously selected real frames are drawn."]
    for bank, data in paired["banks"].items():
        for category, selected in data["examples"].items():
            example = selected["example"]
            lines += ["", f"### {bank.capitalize()}: {CATEGORIES[category]}", "",
                      selected["definition"] + ". Qualifying rows: "
                      + str(selected["n_qualifying"]) + "."]
            if example is None:
                lines.append("No saved qualifying example.")
                continue
            lines += ["", f"Root `{example['root_id']}`, source episode "
                      f"{example['source_episode']}; saved row indices "
                      f"`{example['source_row_indices']}`.", "", decision_text(example), "",
                      horizon_text(example), "",
                      f"![{bank} {category}, recorded example]({example['figure']})"]
    return "\n".join(lines) + "\n"


def run(source, output):
    import h5py
    import hdf5plugin  # noqa: F401
    import matplotlib

    matplotlib.use("Agg")
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or source in output.parents or output in source.parents:
        raise ValueError("Use a fresh sibling output, never a completed-report subtree or parent")
    original = inventory(source)
    paired_path = source / "paired-report/paired_report.json"
    paired = json.loads(paired_path.read_text())
    repair_path = source / "repair.json"
    repair = json.loads(repair_path.read_text())
    for key, path in (("repair_report", repair_path), ("evaluation_rows", source / "repair_rows.npz")):
        if sha256(path) != paired["inputs"][key]["sha256"]:
            raise ValueError(f"Pinned {key} hash changed")
    context = selection_context(repair)
    markdown = markdown_report(paired, context)
    bank_hashes = {}
    for bank, files in paired["inputs"]["bank_identities"].items():
        directory = Path(repair["banks_dir"]) / bank
        for name, expected in files.items():
            path = directory / name
            if sha256(path) != expected:
                raise ValueError(f"Pinned bank input changed: {path}")
            bank_hashes[str(path.resolve())] = expected
    output.mkdir(parents=True, exist_ok=False)
    examples = []
    for bank, data in paired["banks"].items():
        with h5py.File(Path(repair["banks_dir"]) / bank / "branches.h5", "r") as h5:
            for category, selected in data["examples"].items():
                example = selected["example"]
                if example is None:
                    continue
                name = example["figure"]
                if Path(name).name != name or not name.endswith(".png"):
                    raise ValueError("Unsafe saved example filename")
                frames = saved_frames(example, h5)
                limits = plot_example(bank, category, example, frames, output / name)
                examples.append({"bank": bank, "category": category, "row_id": example["row_id"],
                                 "frame_steps_used": example["trace"]["frame_steps_used"],
                                 "figure": name, "plot_limits": limits})
    (output / "README.md").write_text(markdown)
    data = {"run_id": paired["run_id"], "gate": paired["gate"], "selection_context": context,
            "banks": paired["banks"], "retention": paired["retention"], "costs": paired["costs"],
            "source_inputs": paired["inputs"], "rendered_examples": examples}
    (output / "presentation.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    if inventory(source) != original or any(sha256(p) != h for p, h in bank_hashes.items()):
        raise ValueError("Completed inputs changed while producing the presentation")
    manifest = {"kind": "saved-e2-presentation-supplement", "run_id": paired["run_id"],
                "protocol": "immutable-e2-presentation-v1", "source_directory": str(source),
                "source_sha256": original, "bank_inputs_sha256": bank_hashes,
                "generator_sha256": {str(Path(__file__).resolve()): sha256(__file__),
                    str(Path(__file__).resolve().parents[1] / "helpers/pushtGeometry.py"):
                    sha256(Path(__file__).resolve().parents[1] / "helpers/pushtGeometry.py")},
                "gate_unchanged": paired["gate"], "selection_context": context,
                "original_files_unchanged": len(original), "example_count": len(examples),
                "outputs_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
                "additional_costs": {"simulator_steps": 0, "model_queries": 0,
                                     "optimizer_updates": 0},
                "limitations": ["Saved values and examples only; no new statistical inference.",
                    "Undefined point FSA remains undefined; bounds are not confidence intervals.",
                    "Conditional finite bootstrap limits do not establish a directional effect.",
                    "Raw real-frame pixels and their original image clipping are preserved."]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "examples": len(examples),
                      "original_files_unchanged": len(original),
                      "manifest_sha256": sha256(output / "manifest.json")}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.source_dir, args.output_dir)


if __name__ == "__main__":
    main()
