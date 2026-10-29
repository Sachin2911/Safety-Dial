"""Analyze the synced task-10 evidence: stage B response curves and calibration, the stage B2
grounded-check sweep paired with stage B, and stage C or C2 fronts and hypervolume. Reads
outputs/safecem10/remote, writes outputs/safecem10/analysis.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_infra import atomic_json, utc
from helpers.safecem10_evo import hypervolume_2d, pareto_front
from helpers.safecem10_geometry import design, gate_boxes, layout

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, AXIS, GRID, SURFACE = (
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#c3c2b7",
    "#e1e0d9",
    "#fcfcfb",
)
METHOD_COLORS = {"nsga2": BLUE, "grid": ORANGE, "random": AQUA}
METHOD_LABELS = {"nsga2": "NSGA-II", "grid": "grid", "random": "random search"}


def load_results(remote, stage):
    rows = []
    for path in sorted((remote / "attempts").glob("*/result.json")):
        row = json.loads(path.read_text())
        if row["status"] == "complete" and row["job"]["stage"] == stage:
            row["directory"] = str(path.parent)
            rows.append(row)
    return rows


def realized_clearance(states, boxes):
    """Minimum signed distance from the pusher centre to the true boxes over the episode."""
    xy = states[:, 0:2]
    best = np.full(len(xy), np.inf)
    for x0, x1, y0, y1 in boxes:
        dx = np.maximum(np.maximum(x0 - xy[:, 0], 0.0), xy[:, 0] - x1)
        dy = np.maximum(np.maximum(y0 - xy[:, 1], 0.0), xy[:, 1] - y1)
        outside = np.sqrt(dx**2 + dy**2)
        inside = -np.minimum(
            np.minimum(xy[:, 0] - x0, x1 - xy[:, 0]), np.minimum(xy[:, 1] - y0, y1 - xy[:, 1])
        )
        signed = np.where(outside > 0, outside, inside)
        best = np.minimum(best, signed)
    return float(best.min())


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or len(set(y.tolist())) < 2:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx**2).sum() * (ry**2).sum()))


def group_stats(rows, spec):
    lay = layout(rows[0]["job"]["layout"], spec)
    boxes = gate_boxes(lay, spec)
    clearances, path_lengths, above_wall = [], [], []
    start = np.asarray(spec["environment"]["start_pusher"], dtype=float)
    for r in rows:
        states = np.load(Path(r["directory"]) / "states.npz")["states"]
        clearances.append(realized_clearance(states, boxes))
        xy = np.vstack([start[None], states[:, 0:2]])
        path_lengths.append(float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()))
        above_wall.append(float(np.mean(states[:, 1] < spec["gate"]["y_band"][0])))
    nominal = rows[0]["job"]["clearances"][0] if rows[0]["job"]["mode"] == "safe" else None
    return {
        "n": len(rows),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "safe_success_rate": float(
            np.mean([bool(r["success"]) and not r["any_violation"] for r in rows])
        ),
        "exposure_mean": float(np.mean([r["exposure"] for r in rows])),
        "exposure_sd": float(np.std([r["exposure"] for r in rows])),
        "violation_episode_rate": float(np.mean([r["any_violation"] for r in rows])),
        "J_mean": float(np.mean([r["J"] for r in rows])),
        "J_sd": float(np.std([r["J"] for r in rows])),
        "oob_frac_mean": float(np.mean([r["oob_frac"] for r in rows])),
        "block_err_mean": float(np.mean([r["final_block_err_px"] for r in rows])),
        "crossings": dict(Counter(r["crossing"] for r in rows)),
        "final_frac_feasible": float(
            np.mean([r["planner"].get("final_frac_feasible", np.nan) for r in rows])
        ),
        "realized_clearance_median": float(np.median(clearances)),
        "realized_clearance_p10": float(np.percentile(clearances, 10)),
        "realized_clearances": clearances,
        "nominal_honoured_rate": (
            float(np.mean([c >= nominal for c in clearances])) if nominal is not None else None
        ),
        "path_length_median": float(np.median(path_lengths)),
        "above_wall_fraction_mean": float(np.mean(above_wall)),
        "episode_seconds_mean": float(np.mean([r["episode_seconds"] for r in rows])),
        "model_rollouts_mean": float(np.mean([r["model_candidate_rollouts"] for r in rows])),
    }


def analyze_b(remote, output, spec):
    rows = load_results(remote, "B")
    if not rows:
        raise ValueError("No complete stage B results")
    groups = {}
    for r in rows:
        job = r["job"]
        if job["mode"] == "safe":
            key = (job["layout"], "safe", float(job["clearances"][0]), job.get("quantile"))
        else:
            key = (job["layout"], "penalty", float(job["lam"]), None)
        groups.setdefault(key, []).append(r)
    table = []
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2])):
        stats = group_stats(groups[key], spec)
        table.append(
            {"layout": key[0], "mode": key[1], "value": key[2], "quantile": key[3], **stats}
        )
    summary = {"utc": utc(), "episodes": len(rows), "groups": table, "layouts": {}}
    for lay in spec["stages"]["B"]["layouts"]:
        safe = [g for g in table if g["layout"] == lay and g["mode"] == "safe"]
        safe.sort(key=lambda g: g["value"])
        dials = [g["value"] for g in safe]
        summary["layouts"][lay] = {
            "closure_dial": layout(lay, spec)["gap_width"] / 2,
            "spearman_exposure": spearman(dials, [g["exposure_mean"] for g in safe]),
            "spearman_J": spearman(dials, [g["J_mean"] for g in safe]),
            "spearman_success": spearman(dials, [g["success_rate"] for g in safe]),
            "gap_fraction_by_dial": {
                str(g["value"]): g["crossings"].get("gap", 0) / g["n"] for g in safe
            },
            "detour_fraction_by_dial": {
                str(g["value"]): g["crossings"].get("detour", 0) / g["n"] for g in safe
            },
            "penalty": {
                str(g["value"]): {
                    k: g[k] for k in ("success_rate", "exposure_mean", "J_mean", "oob_frac_mean")
                }
                for g in table
                if g["layout"] == lay and g["mode"] == "penalty"
            },
        }
    output.mkdir(parents=True, exist_ok=True)
    slim = json.loads(json.dumps(summary))
    for g in slim["groups"]:
        g.pop("realized_clearances", None)
    atomic_json(output / "stageB-summary.json", slim)
    lines = [
        "| layout | mode | value | n | success | exposure | J | entries | oob | gap | detour | "
        "realized clearance (median) | nominal honoured |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for g in table:
        value = f"d={g['value']:g}" if g["mode"] == "safe" else f"lam={g['value']:g}"
        if g["quantile"]:
            value += f" (p{g['quantile']})"
        honoured = g["nominal_honoured_rate"]
        honoured_text = "" if honoured is None else f"{honoured:.2f}"
        lines.append(
            f"| {g['layout']} | {g['mode']} | {value} | {g['n']} | {g['success_rate']:.2f} | "
            f"{g['exposure_mean']:.3f} | {g['J_mean']:.1f} | {g['violation_episode_rate']:.2f} | "
            f"{g['oob_frac_mean']:.3f} | {g['crossings'].get('gap', 0)} | "
            f"{g['crossings'].get('detour', 0)} | {g['realized_clearance_median']:.1f} | "
            f"{honoured_text} |"
        )
    (output / "stageB-table.md").write_text("\n".join(lines) + "\n")
    plot_b(table, summary, spec, output)
    return summary


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_b(table, summary, spec, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layouts = spec["stages"]["B"]["layouts"]
    metrics = [("success_rate", "success rate"), ("exposure_mean", "exposure"), ("J_mean", "J")]
    fig, axes = plt.subplots(3, len(layouts), figsize=(11, 7.2), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for col, lay in enumerate(layouts):
        safe = sorted(
            [g for g in table if g["layout"] == lay and g["mode"] == "safe" and not g["quantile"]],
            key=lambda g: g["value"],
        )
        quant = [g for g in table if g["layout"] == lay and g["mode"] == "safe" and g["quantile"]]
        pen = [g for g in table if g["layout"] == lay and g["mode"] == "penalty"]
        closure = summary["layouts"][lay]["closure_dial"]
        for row, (key, label) in enumerate(metrics):
            ax = axes[row, col]
            style(ax)
            xs = [g["value"] for g in safe]
            ys = [g[key] for g in safe]
            ax.plot(
                xs, ys, color=BLUE, linewidth=2, marker="o", markersize=5, label="safe CEM dial"
            )
            if quant:
                ax.scatter(
                    [g["value"] for g in quant],
                    [g[key] for g in quant],
                    color=ORANGE,
                    s=42,
                    zorder=3,
                    label="uncertainty-unit dial",
                )
                for g in quant:
                    ax.annotate(
                        f"p{g['quantile']}",
                        (g["value"], g[key]),
                        textcoords="offset points",
                        xytext=(4, 5),
                        fontsize=7,
                        color=INK2,
                    )
            for g in pen:
                ax.axhline(
                    g[key],
                    color=MUTED,
                    linewidth=1,
                    linestyle="--" if g["value"] == 0 else ":",
                    label=f"penalty lam={g['value']:g}",
                )
            ax.axvline(closure, color=AXIS, linewidth=1, linestyle="-.")
            if row == 0:
                ax.set_title(lay, fontsize=10, color=INK, fontweight="bold")
            if col == 0:
                ax.set_ylabel(label, fontsize=9, color=INK2)
            if row == 2:
                ax.set_xlabel("clearance d (px)", fontsize=9, color=INK2)
            if key == "success_rate":
                ax.set_ylim(-0.03, 1.03)
            elif key == "exposure_mean":
                top = max([g["exposure_mean"] for g in table] + [0.1]) * 1.15
                ax.set_ylim(-0.005, top)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    seen = {}
    for h, name in zip(handles, labels):
        seen.setdefault(name, h)
    fig.legend(
        seen.values(),
        seen.keys(),
        loc="lower center",
        ncol=4,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(
        "Gate sweep: response to the clearance dial (dash-dot line = geometric closure)",
        fontsize=11,
        color=INK,
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.14, hspace=0.25, wspace=0.22)
    fig.savefig(output / "stageB-response.png", dpi=170)
    fig.savefig(output / "stageB-response.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(layouts), figsize=(11, 3.2), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for col, lay in enumerate(layouts):
        ax = axes[col]
        style(ax)
        safe = sorted(
            [g for g in table if g["layout"] == lay and g["mode"] == "safe"],
            key=lambda g: g["value"],
        )
        for g in safe:
            ax.scatter(
                [g["value"]] * len(g["realized_clearances"]),
                g["realized_clearances"],
                color=ORANGE if g["quantile"] else BLUE,
                s=14,
                alpha=0.55,
            )
        ax.plot(
            [g["value"] for g in safe],
            [g["realized_clearance_median"] for g in safe],
            color=INK,
            linewidth=1.5,
            label="median realized",
        )
        lim = max([g["value"] for g in safe] + [1.0])
        ax.plot(
            [0, lim], [0, lim], color=AXIS, linewidth=1, linestyle="--", label="nominal = realized"
        )
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_title(lay, fontsize=10, color=INK, fontweight="bold")
        ax.set_xlabel("nominal clearance d (px)", fontsize=9, color=INK2)
        if col == 0:
            ax.set_ylabel("min distance to true box (px)", fontsize=9, color=INK2)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle(
        "Calibration: nominal clearance against realized clearance", fontsize=11, color=INK
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.2, wspace=0.15)
    fig.savefig(output / "stageB-calibration.png", dpi=170)
    fig.savefig(output / "stageB-calibration.pdf")
    plt.close(fig)


B2_SERIES = {
    "frozen": (MUTED, "frozen check (stage B): probe v1, elite mean"),
    "bestonly": (ORANGE, "probe v1, best checked candidate"),
    "fixed": (BLUE, "grounded probe: v2, anchored, best candidate"),
    "exact": (AQUA, "exact controller check, best candidate"),
}


def executed_prediction_errors(row, start):
    """Per replan: error of the check's current-position reading, error of its predicted
    position at the end of the executed block, and whether the executed plan was feasible.
    Stage B executed the elite mean, which was never checked; its final-iteration elite is
    the closest recorded proxy."""
    folder = Path(row["directory"])
    states = np.load(folder / "states.npz")["states"]
    history = json.loads((folder / "history.json").read_text())
    n = int(row["job"]["planner"]["n_steps"])
    block = int(row["job"]["planner"]["action_block"])
    now, nxt, feasible = [], [], []
    for k, rec in enumerate(history[n - 1 :: n]):
        xy = rec.get("executed_xy") or rec.get("elite_xy")
        current = start if k == 0 else states[block * k - 1, :2]
        now.append(float(np.linalg.norm(np.asarray(xy[0]) - current)))
        idx = block * (k + 1) - 1
        if idx < len(states):
            nxt.append(float(np.linalg.norm(np.asarray(xy[1]) - states[idx, :2])))
        if "executed_feasible" in rec:
            feasible.append(bool(rec["executed_feasible"]))
    return now, nxt, feasible


def metric_value(row, key):
    if key == "safe_success":
        return float(bool(row["success"]) and not row["any_violation"])
    return float(row[key])


def paired_difference(groups, cond_a, cond_b, layouts, dials, key, seeds, n_boot=10000):
    """Mean over (layout, seed) units of cond_b minus cond_a, each unit averaged over dials;
    percentile bootstrap interval over units."""
    diffs = []
    for lay in layouts:
        for seed in seeds:
            a, b = [], []
            for d in dials:
                ra = [
                    r
                    for r in groups.get((cond_a, lay, "safe", d, None), [])
                    if r["job"]["seed"] == seed
                ]
                rb = [
                    r
                    for r in groups.get((cond_b, lay, "safe", d, None), [])
                    if r["job"]["seed"] == seed
                ]
                if ra and rb:
                    a.append(metric_value(ra[0], key))
                    b.append(metric_value(rb[0], key))
            if a:
                diffs.append(np.mean(b) - np.mean(a))
    if not diffs:
        return None
    diffs = np.asarray(diffs)
    rng = np.random.default_rng(0)
    boots = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    return {
        "units": int(len(diffs)),
        "mean_difference": float(diffs.mean()),
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "units_b_better": int((diffs < 0).sum() if key in ("exposure", "J") else (diffs > 0).sum()),
        "units_tied": int((diffs == 0).sum()),
    }


def analyze_b2(remote, output, spec):
    b2_rows = load_results(remote, "B2")
    if not b2_rows:
        raise ValueError("No complete stage B2 results")
    layouts = sorted({r["job"]["layout"] for r in b2_rows})
    frozen = [r for r in load_results(remote, "B") if r["job"]["layout"] in layouts]
    groups = {}
    for r in frozen:
        job = r["job"]
        if job["mode"] == "safe":
            key = (
                "frozen",
                job["layout"],
                "safe",
                float(job["clearances"][0]),
                job.get("quantile"),
            )
        else:
            key = (f"frozenlam{job['lam']:g}", job["layout"], "penalty", float(job["lam"]), None)
        groups.setdefault(key, []).append(r)
    for r in b2_rows:
        job = r["job"]
        value = float(job["clearances"][0]) if job["mode"] == "safe" else float(job["lam"])
        key = (job["condition"], job["layout"], job["mode"], value, job.get("quantile"))
        groups.setdefault(key, []).append(r)
    start = np.asarray(spec["environment"]["start_pusher"], dtype=float)
    table = []
    for key in sorted(groups, key=lambda k: (k[1], k[0], k[2], k[3], k[4] or 0)):
        rows = groups[key]
        stats = group_stats(rows, spec)
        now, nxt, feas = [], [], []
        for r in rows:
            a, b, c = executed_prediction_errors(r, start)
            now += a
            nxt += b
            feas += c
        ctrl = [
            r["controller_step_error_px"]["max"] for r in rows if "controller_step_error_px" in r
        ]
        table.append(
            {
                "condition": key[0],
                "layout": key[1],
                "mode": key[2],
                "value": key[3],
                "quantile": key[4],
                **stats,
                "current_reading_error_median": float(np.median(now)) if now else None,
                "executed_block_error_median": float(np.median(nxt)) if nxt else None,
                "executed_block_error_p90": float(np.percentile(nxt, 90)) if nxt else None,
                "executed_feasible_rate": float(np.mean(feas)) if feas else None,
                "controller_step_error_max": float(np.max(ctrl)) if ctrl else None,
            }
        )
    seeds = sorted({r["job"]["seed"] for r in b2_rows})
    grid = [float(d) for d in spec["stages"]["B2"]["dials"]]
    regions = {"open": [d for d in grid if d < 30], "closed": [d for d in grid if d >= 30]}
    paired = {}
    for cond in ("fixed", "exact", "bestonly"):
        lays = [lay for lay in layouts if (cond, lay, "safe", 0.0, None) in groups]
        for region, dials in regions.items():
            for metric in ("success", "safe_success", "exposure", "J"):
                paired[f"{cond}-vs-frozen/{region}/{metric}"] = paired_difference(
                    groups, "frozen", cond, lays, dials, metric, seeds
                )
    probe_report = remote / "probe-v2.json"
    summary = {
        "utc": utc(),
        "episodes_b2": len(b2_rows),
        "episodes_frozen_reused": len(frozen),
        "layouts": layouts,
        "groups": table,
        "paired": paired,
        "probe_v2": json.loads(probe_report.read_text()) if probe_report.exists() else None,
    }
    output.mkdir(parents=True, exist_ok=True)
    slim = json.loads(json.dumps(summary))
    for g in slim["groups"]:
        g.pop("realized_clearances", None)
    atomic_json(output / "stageB2-summary.json", slim)
    lines = [
        "| layout | condition | value | n | success | exposure | J | entries | gap | "
        "realized clearance (median) | nominal honoured | current reading err | "
        "executed block err (median) | executed feasible | path length (median) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def fmt(v, spec_="{:.1f}"):
        return "" if v is None else spec_.format(v)

    for g in table:
        value = f"d={g['value']:g}" if g["mode"] == "safe" else f"lam={g['value']:g}"
        if g["quantile"]:
            value += f" (p{g['quantile']})"
        lines.append(
            f"| {g['layout']} | {g['condition']} | {value} | {g['n']} | {g['success_rate']:.2f} | "
            f"{g['exposure_mean']:.3f} | {g['J_mean']:.1f} | {g['violation_episode_rate']:.2f} | "
            f"{g['crossings'].get('gap', 0)} | {g['realized_clearance_median']:.1f} | "
            f"{fmt(g['nominal_honoured_rate'], '{:.2f}')} | "
            f"{fmt(g['current_reading_error_median'])} | {fmt(g['executed_block_error_median'])} | "
            f"{fmt(g['executed_feasible_rate'], '{:.2f}')} | {g['path_length_median']:.0f} |"
        )
    (output / "stageB2-table.md").write_text("\n".join(lines) + "\n")
    compact = compact_b2(groups, layouts, seeds, start, spec)
    summary["compact"] = compact
    slim["compact"] = compact
    atomic_json(output / "stageB2-summary.json", slim)
    rows_md = [
        "| check | dials | episodes | success | safe success | exposure | episodes entering a box | "
        "nominal honoured | executed-block error (median px) | path length (median px) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in compact:
        rows_md.append(
            f"| {c['condition']} | {c['region']} | {c['n']} | {c['success']:.2f} | "
            f"{c['safe_success']:.2f} | {c['exposure']:.3f} | {c['any_violation']:.2f} | {c['honoured']:.2f} | "
            f"{c['executed_block_error_median']:.1f} | {c['path_length_median']:.0f} |"
        )
    (output / "stageB2-compact.md").write_text("\n".join(rows_md) + "\n")
    plot_b2(table, layouts, spec, output)
    return summary


def compact_b2(groups, layouts, seeds, start, spec):
    """Per check, pooled over layouts and the B2 seeds: open dials (d < closure) and closed
    dials; stage B rows are restricted to the B2 seeds and layouts so the cells are paired."""
    out = []
    for cond in ("frozen", "bestonly", "fixed", "exact"):
        for region in ("open", "closed"):
            rows = []
            for (c, lay, mode, value, quantile), members in groups.items():
                if c != cond or mode != "safe" or quantile or lay not in layouts:
                    continue
                closure = layout(lay, spec)["gap_width"] / 2
                if (value < closure) != (region == "open"):
                    continue
                rows += [r for r in members if r["job"]["seed"] in seeds]
            if not rows:
                continue
            honoured, errors, lengths = [], [], []
            for r in rows:
                lay = layout(r["job"]["layout"], spec)
                states = np.load(Path(r["directory"]) / "states.npz")["states"]
                honoured.append(
                    realized_clearance(states, gate_boxes(lay, spec)) >= r["job"]["clearances"][0]
                )
                xy = np.vstack([start[None], states[:, 0:2]])
                lengths.append(float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()))
                errors += executed_prediction_errors(r, start)[1]
            out.append(
                {
                    "condition": cond,
                    "region": region,
                    "n": len(rows),
                    "success": float(np.mean([r["success"] for r in rows])),
                    "safe_success": float(
                        np.mean([bool(r["success"]) and not r["any_violation"] for r in rows])
                    ),
                    "exposure": float(np.mean([r["exposure"] for r in rows])),
                    "any_violation": float(np.mean([r["any_violation"] for r in rows])),
                    "honoured": float(np.mean(honoured)),
                    "executed_block_error_median": float(np.median(errors)),
                    "path_length_median": float(np.median(lengths)),
                }
            )
    return out


def plot_b2(table, layouts, spec, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [("success_rate", "success rate"), ("exposure_mean", "exposure"), ("J_mean", "J")]
    fig, axes = plt.subplots(3, len(layouts), figsize=(9.5, 7.4), sharex=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    refs = {
        "frozenlam0": (MUTED, "--", "unconstrained, elite mean (stage B)"),
        "bestlam0": (ORANGE, "--", "unconstrained, best candidate"),
        "fixedpen": (BLUE, ":", "grounded probe, penalty lam=1"),
    }
    for col, lay in enumerate(layouts):
        closure = layout(lay, spec)["gap_width"] / 2
        for row, (key, label) in enumerate(metrics):
            ax = axes[row, col]
            style(ax)
            for cond, (color, name) in B2_SERIES.items():
                safe = sorted(
                    [
                        g
                        for g in table
                        if g["layout"] == lay
                        and g["condition"] == cond
                        and g["mode"] == "safe"
                        and not g["quantile"]
                    ],
                    key=lambda g: g["value"],
                )
                if not safe:
                    continue
                ax.plot(
                    [g["value"] for g in safe],
                    [g[key] for g in safe],
                    color=color,
                    linewidth=2,
                    marker="o",
                    markersize=4.5,
                    label=name,
                )
            quant = [
                g
                for g in table
                if g["layout"] == lay and g["condition"] == "fixed" and g["quantile"]
            ]
            for g in quant:
                ax.scatter(
                    [g["value"]],
                    [g[key]],
                    s=46,
                    facecolors=SURFACE,
                    edgecolors=BLUE,
                    linewidths=1.6,
                    zorder=4,
                    label="grounded probe, uncertainty-unit dial",
                )
                ax.annotate(
                    f"p{g['quantile']}",
                    (g["value"], g[key]),
                    textcoords="offset points",
                    xytext=(4, 5),
                    fontsize=7,
                    color=INK2,
                )
            for cond, (color, ls, name) in refs.items():
                for g in table:
                    if g["layout"] == lay and g["condition"] == cond:
                        ax.axhline(g[key], color=color, linewidth=1, linestyle=ls, label=name)
            ax.axvline(closure, color=AXIS, linewidth=1, linestyle="-.")
            if row == 0:
                ax.set_title(lay, fontsize=10, color=INK, fontweight="bold")
            if col == 0:
                ax.set_ylabel(label, fontsize=9, color=INK2)
            if row == 2:
                ax.set_xlabel("clearance d (px)", fontsize=9, color=INK2)
            if key == "success_rate":
                ax.set_ylim(-0.03, 1.03)
            elif key == "exposure_mean":
                top = max([g["exposure_mean"] for g in table] + [0.05]) * 1.15
                ax.set_ylim(-0.003, top)
    handles, labels = [], []
    for ax in axes.flat:
        for h, name in zip(*ax.get_legend_handles_labels()):
            if name not in labels:
                handles.append(h)
                labels.append(name)
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=7.5, frameon=False)
    fig.suptitle(
        "Gate sweep with the safety check grounded, paired with stage B (dash-dot = closure)",
        fontsize=10.5,
        color=INK,
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.2, hspace=0.22, wspace=0.18)
    fig.savefig(output / "stageB2-response.png", dpi=170)
    fig.savefig(output / "stageB2-response.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(layouts), figsize=(9.5, 3.3), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for col, lay in enumerate(layouts):
        ax = axes[0, col]
        style(ax)
        for cond, (color, name) in B2_SERIES.items():
            safe = sorted(
                [
                    g
                    for g in table
                    if g["layout"] == lay
                    and g["condition"] == cond
                    and g["mode"] == "safe"
                    and not g["quantile"]
                ],
                key=lambda g: g["value"],
            )
            if not safe or cond == "bestonly":
                continue
            if cond != "frozen":
                for g in safe:
                    jitter = {"fixed": -1.2, "exact": 1.2}[cond]
                    ax.scatter(
                        [g["value"] + jitter] * len(g["realized_clearances"]),
                        g["realized_clearances"],
                        color=color,
                        s=10,
                        alpha=0.45,
                    )
            ax.plot(
                [g["value"] for g in safe],
                [g["realized_clearance_median"] for g in safe],
                color=color,
                linewidth=1.8,
                label=f"median: {name.split(':')[0]}",
            )
        ax.plot(
            [0, 60], [0, 60], color=AXIS, linewidth=1, linestyle="--", label="nominal = realized"
        )
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_title(lay, fontsize=10, color=INK, fontweight="bold")
        ax.set_xlabel("nominal clearance d (px)", fontsize=9, color=INK2)
        if col == 0:
            ax.set_ylabel("min distance to true box (px)", fontsize=9, color=INK2)
    axes[0, 0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("Calibration with the grounded checks", fontsize=10.5, color=INK)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.84, bottom=0.18, wspace=0.12)
    fig.savefig(output / "stageB2-calibration.png", dpi=170)
    fig.savefig(output / "stageB2-calibration.pdf")
    plt.close(fig)


def analyze_c(remote, output, spec, name="C"):
    stage = spec["stages"]["C"]
    reference = stage["reference_point"]
    folder = remote / f"stage{name}"
    if not folder.exists():
        raise ValueError("No stage C records")
    plan = json.loads((folder / "plan.json").read_text())
    per_run = []
    for run_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        method, rep = run_dir.name.rsplit("-r", 1)
        gens = sorted(run_dir.glob("gen-*.json"))
        all_obj = []
        progress = []
        for path in gens:
            record = json.loads(path.read_text())
            if "objectives" not in record:
                continue
            all_obj += record["objectives"]
            progress.append(
                {
                    "generation": len(progress),
                    "candidates": len(all_obj),
                    "episodes": len(all_obj) * stage["eval_seeds_per_candidate"],
                    "search_hv": hypervolume_2d(all_obj, reference),
                }
            )
        final_path = run_dir / "final.json"
        final = json.loads(final_path.read_text()) if final_path.exists() else {}
        row = {
            "method": method,
            "replicate": int(rep),
            "generations": len(progress),
            "candidates": len(all_obj),
            "search_hv": progress[-1]["search_hv"] if progress else 0.0,
            "progress": progress,
            "final_complete": bool(final.get("complete")),
        }
        for key in ("fresh", "holdout"):
            ids = final.get(f"{key}_job_ids") or []
            per_candidate = {}
            for job_id in ids:
                matches = sorted((remote / "attempts").glob(f"{job_id}*/result.json"))
                rows_c = [json.loads(m.read_text()) for m in matches]
                rows_c = [r for r in rows_c if r.get("status") == "complete"][:1]
                if rows_c:
                    c = int(job_id.rsplit("-c", 1)[1])
                    per_candidate.setdefault(c, []).append(rows_c[0])
            if per_candidate:
                safe = {
                    c: float(np.mean([bool(r["success"]) and not r["any_violation"] for r in rs]))
                    for c, rs in per_candidate.items()
                }
                succ = {
                    c: float(np.mean([r["success"] for r in rs])) for c, rs in per_candidate.items()
                }
                row[f"{key}_safe_success_by_candidate"] = [safe[c] for c in sorted(safe)]
                row[f"{key}_success_by_candidate"] = [succ[c] for c in sorted(succ)]
                row[f"{key}_best_safe_success"] = max(safe.values())
            objs = final.get(f"{key}_objectives")
            if objs:
                front = pareto_front(objs)
                safe_success = [o for o in objs if o[0] == 0.0 and o[1] < reference[1]]
                row[f"{key}_hv"] = hypervolume_2d(objs, reference)
                row[f"{key}_front"] = [objs[i] for i in front]
                row[f"{key}_best_safe_J"] = min((o[1] for o in safe_success), default=None)
                row[f"{key}_min_exposure"] = min(o[0] for o in objs)
                row[f"{key}_min_J"] = min(o[1] for o in objs)
        row["front_genotypes"] = final.get("genotypes")
        row["search_phase_front"] = final.get("search_phase_objectives")
        per_run.append(row)
    methods = {}
    for method in stage["methods"]:
        runs = [r for r in per_run if r["method"] == method]
        if not runs:
            continue
        entry = {"replicates": len(runs)}
        for key in (
            "search_hv",
            "fresh_hv",
            "holdout_hv",
            "fresh_best_safe_success",
            "holdout_best_safe_success",
        ):
            values = [r[key] for r in runs if key in r]
            if values:
                entry[key] = {
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values)),
                    "values": values,
                }
        methods[method] = entry
    pairwise = {}
    for a in stage["methods"]:
        for b in stage["methods"]:
            if a >= b or a not in methods or b not in methods:
                continue
            for key in (
                "fresh_hv",
                "holdout_hv",
                "fresh_best_safe_success",
                "holdout_best_safe_success",
            ):
                va = methods[a].get(key, {}).get("values", [])
                vb = methods[b].get(key, {}).get("values", [])
                if va and vb:
                    pairwise[f"{a}_vs_{b}_{key}"] = {
                        "difference_of_means": float(np.mean(va) - np.mean(vb)),
                        "wins": int(sum(x > y for x in va for y in vb)),
                        "comparisons": len(va) * len(vb),
                    }
    summary = {
        "utc": utc(),
        "plan": plan,
        "methods": methods,
        "pairwise": pairwise,
        "runs": per_run,
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / f"stage{name}-summary.json", summary)
    lines = [
        "| method | replicates | search HV | fresh-seed HV | held-out HV | "
        "best safe success, fresh seeds | best safe success, held-out layout |",
        "|---|---|---|---|---|---|---|",
    ]
    for method, entry in methods.items():
        cells = []
        for key in (
            "search_hv",
            "fresh_hv",
            "holdout_hv",
            "fresh_best_safe_success",
            "holdout_best_safe_success",
        ):
            value = entry.get(key)
            cells.append(f"{value['mean']:.2f} ± {value['sd']:.2f}" if value else "")
        lines.append(
            f"| {METHOD_LABELS[method]} | {entry['replicates']} | " + " | ".join(cells) + " |"
        )
    (output / f"stage{name}-table.md").write_text("\n".join(lines) + "\n")
    plot_c(per_run, summary, spec, output, name)
    return summary


def plot_c(per_run, summary, spec, output, name="C"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference = spec["stages"]["C"]["reference_point"]
    dial_curves = {}
    b2_summary = output / "stageB2-summary.json"
    if name == "C2" and b2_summary.exists():
        for g in json.loads(b2_summary.read_text())["groups"]:
            if g["condition"] == "fixed" and g["mode"] == "safe" and not g["quantile"]:
                dial_curves.setdefault(g["layout"], []).append(
                    (g["value"], g["exposure_mean"], g["J_mean"])
                )
    exposures = [
        p[0]
        for run in per_run
        for key in ("fresh", "holdout")
        for p in (run.get(f"{key}_front") or [])
    ] + [e for curve in dial_curves.values() for _, e, _ in curve]
    x_top = max(exposures + [0.05]) * 1.2
    js = [
        p[1]
        for run in per_run
        for key in ("fresh", "holdout")
        for p in (run.get(f"{key}_front") or [])
    ] + [j for curve in dial_curves.values() for _, _, j in curve]
    y_bottom = max(0.0, min(js + [reference[1]]) - 3.0)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    fig.patch.set_facecolor(SURFACE)
    ax = axes[0]
    style(ax)
    for run in per_run:
        xs = [p["episodes"] for p in run["progress"]]
        ys = [p["search_hv"] for p in run["progress"]]
        ax.plot(
            xs,
            ys,
            color=METHOD_COLORS[run["method"]],
            linewidth=1.6,
            alpha=0.9,
            label=METHOD_LABELS[run["method"]],
        )
    ax.set_xlabel("episodes spent", fontsize=9, color=INK2)
    ax.set_ylabel("search hypervolume", fontsize=9, color=INK2)
    ax.set_title("Search progress (optimistic)", fontsize=10, color=INK, fontweight="bold")
    for idx, key in enumerate(("fresh", "holdout")):
        ax = axes[idx + 1]
        style(ax)
        for run in per_run:
            front = run.get(f"{key}_front")
            if not front:
                continue
            pts = sorted(front)
            ax.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                color=METHOD_COLORS[run["method"]],
                marker="o",
                markersize=5,
                linewidth=1.4,
                alpha=0.9,
                label=METHOD_LABELS[run["method"]],
            )
        lay = spec["stages"]["C"]["search_layout" if key == "fresh" else "holdout_layout"]
        curve = sorted(dial_curves.get(lay, []))
        if curve:
            ax.plot(
                [e for _, e, _ in curve],
                [j for _, _, j in curve],
                color=INK2,
                linestyle="--",
                marker="s",
                markersize=3.5,
                linewidth=1,
                label="single dial d, grounded check (stage B2)",
            )
            for d, e, j in curve:
                ax.annotate(
                    f"{d:g}",
                    (e, j),
                    textcoords="offset points",
                    xytext=(3, -8),
                    fontsize=6,
                    color=INK2,
                )
        ax.set_xlim(-0.02 * x_top, x_top)
        ax.set_ylim(y_bottom, reference[1] + 1.5)
        ax.set_xlabel("exposure (fraction of steps in a true box)", fontsize=9, color=INK2)
        ax.set_ylabel("J (steps to success, 50 = none)", fontsize=9, color=INK2)
        title = "Fresh seeds, search layout" if key == "fresh" else "Fresh seeds, held-out layout"
        ax.set_title(title, fontsize=10, color=INK, fontweight="bold")
    seen = {}
    for ax in axes:
        for h, label in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(label, h)
    fig.legend(
        seen.values(),
        seen.keys(),
        loc="lower center",
        ncol=4,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle("Multi-objective search over the constraint interface", fontsize=11, color=INK)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.25, wspace=0.3)
    fig.savefig(output / f"stage{name}-fronts.png", dpi=170)
    fig.savefig(output / f"stage{name}-fronts.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["B", "B2", "C", "C2"], required=True)
    parser.add_argument("--remote", type=Path, default=ROOT / "outputs/safecem10/remote")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/safecem10/analysis")
    args = parser.parse_args()
    spec = design()
    if args.stage == "B":
        summary = analyze_b(args.remote, args.output, spec)
        print(json.dumps({"episodes": summary["episodes"], "layouts": summary["layouts"]}))
    elif args.stage == "B2":
        summary = analyze_b2(args.remote, args.output, spec)
        print(json.dumps({"episodes": summary["episodes_b2"], "paired": summary["paired"]}))
    else:
        summary = analyze_c(args.remote, args.output, spec, args.stage)
        print(json.dumps({"methods": summary["methods"], "pairwise": summary["pairwise"]}))


if __name__ == "__main__":
    main()
