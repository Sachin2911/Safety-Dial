"""Final locked-comparison tables, case-block uncertainty and exportable figures."""

import collections
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/lewm16-matplotlib")
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.lewm16_analysis import cells, paired_contrasts, read_rows
from helpers.lewm16_protocol import OUT, digest, write_json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def aggregates(rows):
    groups = collections.defaultdict(list)
    for row in rows:
        j = row["job"]
        if row["status"] == "complete":
            groups[(j["stage"], j["env"], j["margin"], j["condition"])].append(row)
    table = []
    for (stage, env, margin, condition), rs in sorted(groups.items()):
        result = dict(stage=stage, env=env, margin=margin, condition=condition, n=len(rs))
        for metric in (
            "requested_safe_completion",
            "benchmark_safe_completion",
            "success",
            "any_violation",
            "margin_breach",
            "unfinished",
            "censored",
        ):
            result[metric] = sum(int(r[metric]) for r in rs)
        for metric in ("J", "seconds", "path_length", "action_norm_mean"):
            result["mean_" + metric] = float(np.mean([r[metric] for r in rs]))
        for metric in ("bounded_control_norm", "clipped_command_fraction"):
            if all(metric in r for r in rs):
                result["mean_" + metric] = float(np.mean([r[metric] for r in rs]))
        result["engaged"] = sum(r["path_length"] > (1 if env == "tworoom" else 0.001) for r in rs)
        result["resources"] = dict(
            sum((collections.Counter(r["counters"]) for r in rs), collections.Counter())
        )
        table.append(result)
    return table


def save(fig, name):
    fig.savefig(OUT / "analysis" / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(OUT / "analysis" / (name + ".png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    manifest = json.loads((ROOT / "progress/receipts/16-final-ab-v1.json").read_text())
    rows = read_rows(OUT / "remote/attempts", ["A", "B"])
    by_id = {r["job"]["id"]: r for r in rows}
    expected = {j["id"] for j in manifest["jobs"]}
    if set(by_id) != expected or any(r["status"] != "complete" for r in rows):
        raise ValueError("Final comparisons are incomplete or invalid; no final report generated")
    destination = OUT / "analysis"
    destination.mkdir(parents=True, exist_ok=True)
    for row in rows:
        path = OUT / "remote/attempts" / row["job"]["id"] / "states.npz"
        commands = np.load(path)["actions"]
        row["bounded_control_norm"] = (
            float(np.linalg.norm(np.clip(commands, -1, 1), axis=1).mean()) if len(commands) else 0.0
        )
        row["clipped_command_fraction"] = (
            float((np.abs(commands) > 1).any(axis=1).mean()) if len(commands) else 0.0
        )
    table, effects = aggregates(rows), paired_contrasts(rows)
    if effects["unpaired"]:
        raise ValueError("Final contrasts contain unpaired episodes")
    modes = {}
    for env in ("tworoom", "reacher"):
        counter, switched, total = collections.Counter(), 0, 0
        for row in rows:
            j = row["job"]
            if j["env"] != env or j["condition"] != "adaptive":
                continue
            plans = json.loads((OUT / "remote/attempts" / j["id"] / "replans.json").read_text())
            baseline = (j["params"]["horizon"], j["params"]["variance"], j["params"]["retain"])
            used = [(p["horizon"], p["variance"], p["retain"]) for p in plans if "variance" in p]
            counter.update(used)
            switched += any(mode != baseline for mode in used)
            total += 1
        modes[env] = {
            "episodes": total,
            "episodes_using_nonbase_mode": switched,
            "mode_replan_counts": [
                {"horizon": k[0], "scale": k[1], "retention": k[2], "n": v}
                for k, v in sorted(counter.items())
            ],
        }
    write_json(
        destination / "final-statistics.json",
        {
            "frozen_manifest_sha256": digest(ROOT / "progress/receipts/16-final-ab-v1.json"),
            "episodes": len(rows),
            "aggregates": table,
            "cells": cells(rows),
            "paired_effects": effects,
            "adaptive_modes": modes,
            "action_provenance": "Replans retain the proposed mean and selected checked block; states.npz actions are the executed environment commands. Bounded simulator inputs are reconstructed offline using the pinned [-1,1] clipping. They are not joint torques; Reacher also applies its motor gear. Original traces remain unchanged.",
            "scope": "Selected controllers, one physical host, two task families. Not an optimizer comparison.",
        },
    )
    colors = {
        "native": "#7d8790",
        "canonical": "#a76828",
        "fixed": "#2361a0",
        "adaptive": "#15936e",
    }
    labels = {
        "native": "Native",
        "canonical": "Canonical SafeCEM",
        "fixed": "Tuned fixed",
        "adaptive": "Adaptive",
    }
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for i, env in enumerate(("tworoom", "reacher")):
        for k, stage in enumerate(("A", "B")):
            ax = axes[i, k]
            for condition in colors if stage == "A" else ["fixed", "adaptive"]:
                rs = [
                    r
                    for r in table
                    if (r["env"], r["stage"], r["condition"]) == (env, stage, condition)
                ]
                x = [r["margin"] * (100 if env == "reacher" else 1) for r in rs]
                y = [100 * r["requested_safe_completion"] / r["n"] for r in rs]
                ax.plot(
                    x, y, "s-" if condition == "fixed" else "o-",
                    label=labels[condition], color=colors[condition],
                    markersize=7 if condition == "fixed" else 4,
                )
            ax.set_xticks(x)
            ax.set(
                title=f"{'TwoRoom' if env == 'tworoom' else 'Reacher'}: "
                f"{'familiar layouts (A)' if stage == 'A' else 'fresh cases (B)'}",
                xlabel="Requested margin (cm)" if env == "reacher" else "Requested margin (px)",
                ylabel="Safe completion (%)",
                ylim=(-3, 103),
            )
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    save(fig, "safe-completion")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True, layout="constrained")
    for ax, env in zip(axes, ("tworoom", "reacher")):
        rs = [
            r
            for r in effects["contrasts"]
            if r["stage"] == "B"
            and r["env"] == env
            and r["metric"] == "requested_safe_completion"
            and r["margin"] != "all"
        ]
        rs.sort(key=lambda r: float(r["margin"]))
        x = np.arange(len(rs))
        y = np.array([r["adaptive_minus_fixed"] * 100 for r in rs])
        lo = np.array([r["interval_95"][0] * 100 for r in rs])
        hi = np.array([r["interval_95"][1] * 100 for r in rs])
        ax.errorbar(
            x,
            y,
            yerr=[np.maximum(0, y - lo), np.maximum(0, hi - y)],
            fmt="o",
            capsize=4,
            color=colors["adaptive"],
        )
        ax.axhline(0, color="grey", linewidth=1)
        ax.set_xticks(x, [f"{float(r['margin']) * (100 if env == 'reacher' else 1):g}" for r in rs])
        ax.set(
            title="TwoRoom" if env == "tworoom" else "Reacher",
            ylabel="Adaptive - fixed (percentage points)",
            xlabel="Requested margin (cm)" if env == "reacher" else "Requested margin (px)",
        )
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Fresh cases: paired 95% case-block bootstrap intervals")
    save(fig, "transfer-difference")
    quick = aggregates(read_rows(OUT / "remote/attempts", ["Q"]))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    names = ["canonical", "short", "memory", "larger", "geometric"]
    names_display = ["Canonical", "Horizon 3", "Memory + correlation", "2x search", "Geometric"]
    palette = ["#a76828", "#2361a0", "#15936e", "#7871a9", "#a4a8ac"]
    for column, (env, high) in enumerate(zip(["tworoom", "reacher"], [6, 0.015])):
        ax = axes[0, column]
        rs = {r["condition"]: r for r in quick if r["env"] == env and r["margin"] == high}
        ax.bar(
            np.arange(5),
            [100 * rs[c]["requested_safe_completion"] / rs[c]["n"] for c in names],
            color=palette,
        )
        ax.set_xticks(
            np.arange(5),
            ["Canonical", "Horizon 3", "Memory +\ncorrelation", "2x search", "Geometric\nreference"],
            fontsize=8,
        )
        ax.set(
            title=("TwoRoom: 6 px" if env == "tworoom" else "Reacher: 1.5 cm")
            + ", 9 episodes per condition",
            ylabel="Safe completion (%)",
            ylim=(0, 108),
        )
        cost_ax = axes[1, column]
        for condition, label, color in zip(names, names_display, palette):
            row = rs[condition]
            work = row["resources"]["predictor_steps"] / row["n"] / 1e6
            completion = 100 * row["requested_safe_completion"] / row["n"]
            cost_ax.scatter(work, completion, color=color, label=label, s=45)
        cost_ax.set(
            xlabel="Mean predictor steps per episode (millions)",
            ylabel="Safe completion (%)",
            ylim=(-5, 108),
        )
        cost_ax.grid(alpha=0.2)
    axes[1, 1].legend(fontsize=7, loc="upper right")
    save(fig, "mechanism-comparison")
    code = destination / "code"
    code.mkdir(exist_ok=True)
    shutil.copy2(__file__, code / Path(__file__).name)
    print(
        json.dumps(
            {
                "episodes": len(rows),
                "statistics": str(destination / "final-statistics.json"),
                "figures": 3,
                "adaptive_modes": modes,
            }
        )
    )


if __name__ == "__main__":
    main()
