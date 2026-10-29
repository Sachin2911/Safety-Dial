"""Analyze the synced task-14 evidence: the stage B dial sweep (response curves, paired
comparisons, best fixed dial) and the stage C fronts and hypervolume (shared with task 10).
Reads outputs/cube14/remote, writes outputs/cube14/analysis.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments/scripts"))
from analyze_safecem10 import AQUA, BLUE, INK, INK2, MUTED, ORANGE, analyze_c, style
from helpers.dsrl06_infra import atomic_json, utc
from helpers.cube14_protocol import design

CONDITION_COLORS = {"safe": BLUE, "mean": ORANGE, "free": AQUA}
CONDITION_LABELS = {
    "safe": "checked plan executed (mean_safe)",
    "mean": "unchecked elite mean executed",
    "free": "unconstrained",
}


def load_results(remote, stage):
    rows = []
    for path in sorted((remote / "attempts").glob("*/result.json")):
        row = json.loads(path.read_text())
        if row["status"] == "complete" and row["job"]["stage"] == stage:
            row["directory"] = str(path.parent)
            rows.append(row)
    return rows


def safe_success(row):
    return bool(row["success"]) and not row["any_violation"]


def group_stats(rows):
    clearances = [r["realized_clearance"] for r in rows]
    execution = [r.get("execution") or {} for r in rows]
    mean_frac = [e["mean_executed_frac"] for e in execution if "mean_executed_frac" in e]
    return {
        "n": len(rows),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "safe_success_rate": float(np.mean([safe_success(r) for r in rows])),
        "exposure_mean": float(np.mean([r["exposure"] for r in rows])),
        "exposure_sd": float(np.std([r["exposure"] for r in rows])),
        "violation_episode_rate": float(np.mean([r["any_violation"] for r in rows])),
        "J_mean": float(np.mean([r["J"] for r in rows])),
        "J_sd": float(np.std([r["J"] for r in rows])),
        "crossings": dict(Counter(r["crossing"] for r in rows)),
        "realized_clearance_median": float(np.median(clearances)),
        "realized_clearance_p10": float(np.percentile(clearances, 10)),
        "min_target_distance_median": float(np.median([r["min_effector_goal_m"] for r in rows])),
        "mean_executed_frac": float(np.mean(mean_frac)) if mean_frac else None,
        "episode_seconds_mean": float(np.mean([r["episode_seconds"] for r in rows])),
    }


def metric_value(row, key):
    if key == "safe_success":
        return float(safe_success(row))
    return float(row[key])


def paired(rows_a, rows_b, key, n_boot=10000):
    """Mean over seeds of b minus a (seed-paired), with a percentile bootstrap interval."""
    by_seed_a = {r["job"]["seed"]: r for r in rows_a}
    diffs = [
        metric_value(r, key) - metric_value(by_seed_a[r["job"]["seed"]], key)
        for r in rows_b
        if r["job"]["seed"] in by_seed_a
    ]
    if not diffs:
        return None
    diffs = np.asarray(diffs)
    boots = np.random.default_rng(0).choice(diffs, size=(n_boot, len(diffs))).mean(axis=1)
    better = (diffs < 0) if key in ("exposure", "J") else (diffs > 0)
    return {
        "units": int(len(diffs)),
        "mean_difference": float(diffs.mean()),
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "units_b_better": int(better.sum()),
        "units_tied": int((diffs == 0).sum()),
    }


def analyze_b(remote, output, spec):
    rows = load_results(remote, "B")
    if not rows:
        raise ValueError("No complete stage B results")
    groups = {}
    for r in rows:
        job = r["job"]
        dial = None if job["condition"] == "free" else float(job["clearances"][0])
        groups.setdefault((job["condition"], job["layout"], dial), []).append(r)
    table = []
    for key in sorted(groups, key=lambda k: (k[1], k[0], -1.0 if k[2] is None else k[2])):
        table.append(
            {"condition": key[0], "layout": key[1], "dial": key[2], **group_stats(groups[key])}
        )
    dials = [float(d) for d in spec["stages"]["B"]["dials"]]
    closure = {
        lay["id"]: lay["gap_width"] / 2.0 for lay in spec["hazard"]["layouts"]
    }  # the gap closes at d = width / 2
    comparisons = {}
    for lay in sorted({k[1] for k in groups}):
        free = groups.get(("free", lay, None), [])
        open_dials = [d for d in dials if d < closure[lay]]
        for cond in ("safe", "mean"):
            if (cond, lay, 0.0) not in groups:
                continue
            for d in dials:
                for metric in ("success", "safe_success", "exposure", "J"):
                    comparisons[f"{lay}/{cond}-vs-free/d{d:g}/{metric}"] = paired(
                        free, groups.get((cond, lay, d), []), metric
                    )
            pooled = [r for d in open_dials for r in groups.get((cond, lay, d), [])]
            for metric in ("success", "safe_success", "exposure", "J"):
                # Pool the open-gap dials: each seed contributes its mean over those dials.
                means = {}
                for r in pooled:
                    means.setdefault(r["job"]["seed"], []).append(metric_value(r, metric))
                free_by_seed = {r["job"]["seed"]: metric_value(r, metric) for r in free}
                diffs = [
                    np.mean(v) - free_by_seed[s] for s, v in means.items() if s in free_by_seed
                ]
                if diffs:
                    boots = np.random.default_rng(0).choice(diffs, size=(10000, len(diffs)))
                    comparisons[f"{lay}/{cond}-vs-free/open/{metric}"] = {
                        "units": len(diffs),
                        "mean_difference": float(np.mean(diffs)),
                        "ci95": [
                            float(np.percentile(boots.mean(axis=1), 2.5)),
                            float(np.percentile(boots.mean(axis=1), 97.5)),
                        ],
                    }
    search = spec["stages"]["C"]["search_layout"]
    if ("mean", search, 0.0) in groups:
        for d in dials:
            for metric in ("success", "safe_success", "exposure", "J"):
                comparisons[f"{search}/safe-vs-mean/d{d:g}/{metric}"] = paired(
                    groups.get(("mean", search, d), []),
                    groups.get(("safe", search, d), []),
                    metric,
                )
    best = {}
    for lay in sorted({k[1] for k in groups}):
        cells = [g for g in table if g["layout"] == lay and g["condition"] == "safe"]
        if cells:
            top = max(cells, key=lambda g: (g["safe_success_rate"], -g["J_mean"]))
            best[lay] = {
                "dial": top["dial"],
                "safe_success_rate": top["safe_success_rate"],
                "J_mean": top["J_mean"],
            }
    summary = {
        "utc": utc(),
        "episodes": len(rows),
        "groups": table,
        "comparisons": comparisons,
        "best_fixed_dial": best,
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "stageB-summary.json", summary)
    lines = [
        "| layout | condition | dial | n | success | safe success | exposure | entries | "
        "gap crossings | realized clearance (median, cm) | elite mean executed |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for g in table:
        dial = "" if g["dial"] is None else f"{100 * g['dial']:g} cm"
        mean_frac = "" if g["mean_executed_frac"] is None else f"{g['mean_executed_frac']:.2f}"
        lines.append(
            f"| {g['layout']} | {g['condition']} | {dial} | {g['n']} | {g['success_rate']:.2f} | "
            f"{g['safe_success_rate']:.2f} | {g['exposure_mean']:.3f} | "
            f"{g['violation_episode_rate']:.2f} | {g['crossings'].get('gap', 0)} | "
            f"{100 * g['realized_clearance_median']:.1f} | {mean_frac} |"
        )
    (output / "stageB-table.md").write_text("\n".join(lines) + "\n")
    plot_b(table, spec, output)
    return summary


def plot_b(table, spec, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layouts = [lay["id"] for lay in spec["hazard"]["layouts"]]
    titles = {
        lay["id"]: f"{lay['id']}: gap {100 * lay['offset_from_route_m']:g} cm off the route"
        for lay in spec["hazard"]["layouts"]
    }
    rows = [
        ("safe_success_rate", "safe success"),
        ("success_rate", "success"),
        ("exposure_mean", "exposure"),
    ]
    fig, axes = plt.subplots(len(rows), len(layouts), figsize=(10.5, 7.6), sharex=True)
    for j, lay in enumerate(layouts):
        free = next((g for g in table if g["layout"] == lay and g["condition"] == "free"), None)
        closes = next(x["gap_width"] for x in spec["hazard"]["layouts"] if x["id"] == lay) / 2
        for i, (key, label) in enumerate(rows):
            ax = axes[i, j]
            style(ax)
            for cond in ("safe", "mean"):
                cells = sorted(
                    (g for g in table if g["layout"] == lay and g["condition"] == cond),
                    key=lambda g: g["dial"],
                )
                if cells:
                    ax.plot(
                        [100 * g["dial"] for g in cells],
                        [g[key] for g in cells],
                        color=CONDITION_COLORS[cond],
                        linewidth=2,
                        marker="o",
                        markersize=4,
                        label=CONDITION_LABELS[cond],
                    )
            if free is not None:
                ax.axhline(
                    free[key],
                    color=CONDITION_COLORS["free"],
                    linewidth=2,
                    linestyle="--",
                    label=CONDITION_LABELS["free"],
                )
            ax.axvline(100 * closes, color=MUTED, linewidth=0.8, linestyle=":")
            if key != "exposure_mean":
                ax.set_ylim(-0.03, 1.03)
            else:
                top = max(0.05, max(g["exposure_mean"] for g in table) * 1.15)
                ax.set_ylim(-0.03 * top, top)
            if j == 0:
                ax.set_ylabel(label, color=INK2, fontsize=9)
            if i == 0:
                ax.set_title(titles[lay], color=INK, fontsize=9)
            if i == len(rows) - 1:
                ax.set_xlabel("clearance dial d (cm)", color=INK2, fontsize=9)
    handles, labels = axes[0, -1].get_legend_handles_labels()
    for ax in axes[0]:
        h, lab = ax.get_legend_handles_labels()
        for hh, ll in zip(h, lab, strict=True):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.text(
        0.5,
        0.955,
        "Dotted line: the gap closes (no safe route beyond it). 10 paired seeds per point.",
        ha="center",
        color=MUTED,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(output / "stageB-response.png", dpi=170)
    fig.savefig(output / "stageB-response.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["B", "C"], required=True)
    parser.add_argument("--remote", type=Path, default=ROOT / "outputs/cube14/remote")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/cube14/analysis")
    args = parser.parse_args()
    spec = design()
    if args.stage == "B":
        summary = analyze_b(args.remote, args.output, spec)
        print(json.dumps({"episodes": summary["episodes"], "best": summary["best_fixed_dial"]}))
    else:
        summary = analyze_c(args.remote, args.output, spec, "C")
        print(json.dumps({"methods": summary["methods"]}, default=str)[:4000])


if __name__ == "__main__":
    main()
