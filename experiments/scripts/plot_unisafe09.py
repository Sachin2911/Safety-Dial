"""Export scientific figures from complete, validated UNISafe panels only."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/unisafe09-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.ticker import PercentFormatter

LABELS = {
    "d00_g100": "N: native",
    "d15_g100": "E: delay 15",
    "d40_g100": "F: delay 40",
    "reference_random": "R: prior random-search reference",
    "history_only": "H: history only",
    "grasp_latch": "L: grasp latch",
}
SHORT = {key: value.split(":")[0] for key, value in LABELS.items()}


def label(condition):
    if condition in LABELS:
        return LABELS[condition]
    delay, gain = condition.split("_g")
    return f"Delay {int(delay[1:])}, gain {int(gain) / 100:g}"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(folder / f"{name}.{suffix}", dpi=220, facecolor="white")
    plt.close(fig)


def finish(fig, stage, title, note):
    phase = "exploratory selection screen" if stage == "B" else "independent confirmation"
    fig.suptitle(f"UNISafe block plucking | {title}", fontsize=14, weight="bold", y=.99)
    fig.text(.015, .045, f"Stage {stage}: {phase}. {note}", fontsize=9)
    fig.text(.015, .018, "Every rate includes all valid starts; invalid attempts are reported "
             "separately. No pretrained weights were updated.", fontsize=8, color="0.3")
    fig.tight_layout(rect=(0, .08, 1, .95), w_pad=2, h_pad=2)


def response_map(conditions, folder):
    ids = [[f"d{d:02d}_g{g:03d}" for g in (50, 75, 100)] for d in (0, 15, 40)]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.3))
    for ax, metric, title, cmap, upper in zip(
        axes, ("C", "V", "J"),
        ("Safe completion C", "Violations V", "Failure-penalized time J"),
        ("Blues", "Reds", "viridis_r"), (1, 1, 15), strict=True,
    ):
        values = np.array([[conditions[c][metric] for c in row] for row in ids])
        im = ax.imshow(values, vmin=0, vmax=upper, cmap=cmap, aspect="auto")
        for i, row in enumerate(ids):
            for j, condition in enumerate(row):
                data = conditions[condition]
                text = f"{data[metric + '_count']}/{data['valid']}" if metric != "J" else (
                    f"{data['J']:.2f} s"
                )
                rgba = im.cmap(im.norm(values[i, j]))
                brightness = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
                ax.text(j, i, text, ha="center", va="center", fontsize=11,
                        color="black" if brightness > .55 else "white")
        ax.set_xticks(range(3), ["0.50", "0.75", "1.00"])
        ax.set_yticks(range(3), ["0", "15", "40"])
        ax.set_xlabel("Translation gain before the safety filter")
        ax.set_ylabel("Gripper hold delay (control steps)")
        ax.set_title(title, fontsize=11)
        bar = fig.colorbar(im, ax=ax, fraction=.047, pad=.04)
        if metric != "J":
            bar.ax.yaxis.set_major_formatter(PercentFormatter(1))
    finish(fig, "B", "Delay and translation response map",
           "16 paired starts per cell. J assigns 15 s to every failure or unfinished episode.")
    save(fig, folder, "B-response-map")


def speed_completion(stage, conditions, folder):
    # Preserve exact coordinates when several settings share the same C and J.
    # The adjacent table exposes every setting's V, including overlapping points.
    groups = {}
    for condition, row in sorted(conditions.items()):
        key = (round(row["J"], 10), row["C"])
        groups.setdefault(key, []).append((condition, row))
    ordered = sorted(groups.items(), key=lambda item: (-item[0][1], item[0][0]))
    fig, (ax, table_ax) = plt.subplots(
        1, 2, figsize=(12.4, 6.4), gridspec_kw={"width_ratios": [1.7, 1]},
    )
    norm = Normalize(0, 1)
    table_rows = []
    previous_label_y = 1.2
    names = {
        "d00_g100": "Native (N)", "d15_g100": "Delay 15 (E)",
        "d40_g100": "Delay 40 (F)", "reference_random": "Prior search (R)",
        "history_only": "History only (H)", "grasp_latch": "Grasp latch (L)",
    }
    for number, ((j, c), members) in enumerate(ordered, start=1):
        lo = min(row["C_wilson95"][0] for _, row in members)
        hi = max(row["C_wilson95"][1] for _, row in members)
        ax.errorbar(j, c, yerr=[[c - lo], [hi - c]], color="0.72", capsize=3, lw=.8)
        points = ax.scatter(j, c, c=[max(row["V"] for _, row in members)], norm=norm,
                            cmap="plasma", s=70, edgecolor="black", linewidth=.7, zorder=3)
        label_y = min(c + .08, previous_label_y - .08)
        previous_label_y = label_y
        ax.annotate(str(number), (j, c), xytext=(j + (.7 if j < 13 else -.7), label_y),
                    ha="center", va="center", fontsize=8,
                    bbox={"boxstyle": "round,pad=.2", "fc": "white", "ec": "0.6", "lw": .5},
                    arrowprops={"arrowstyle": "-", "color": "0.45", "lw": .6})
        for condition, row in members:
            name = names.get(condition, condition.replace("_g", "/g"))
            table_rows.append([str(number), name, f"{row['C_count']}/{row['valid']}",
                               f"{row['V_count']}/{row['valid']}", f"{row['J']:.2f}"])
    ax.set(xlim=(-.3, 15.4), ylim=(-.04, 1.12),
           xlabel="Failure-penalized completion time J (s)", ylabel="Safe completion C")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
    bar = fig.colorbar(points, ax=ax, pad=.025, fraction=.035)
    bar.set_label("Greatest V at each plotted location", fontsize=9)
    bar.ax.yaxis.set_major_formatter(PercentFormatter(1))
    table_ax.axis("off")
    table_ax.set_title("Every tested setting", fontsize=11, pad=16)
    table = table_ax.table(cellText=table_rows, colLabels=["Point", "Setting", "C", "V", "J (s)"],
                           colWidths=[.12, .38, .18, .16, .16], cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.65)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("0.85")
        cell.set_linewidth(.5)
        if row == 0:
            cell.set_facecolor("#edf2f5")
            cell.set_text_props(weight="bold")
        elif column == 1:
            cell.set_text_props(ha="left")
    finish(fig, stage, "Measured speed and completion",
           "Whiskers: 95% Wilson intervals. Coincident settings share a point; every V appears in the table.")
    save(fig, folder, f"{stage}-speed-completion")


def deadlines(stage, rows, folder):
    ids = sorted({r["condition"] for r in rows})
    ncols = 3
    nrows = (len(ids) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.4, 2.5 * nrows + 1),
                             sharex=True, sharey=True, squeeze=False)
    for ax, condition in zip(axes.flat, ids, strict=False):
        data = [r for r in rows if r["condition"] == condition]
        times = sorted(r["completion_time_s"] for r in data if r["safe_success"])
        x = [0, *times, 15]
        y = [0, *(np.arange(1, len(times) + 1) / len(data)), len(times) / len(data)]
        ax.step(x, y, where="post", lw=2, color="#246b8e")
        ax.set_title(label(condition), fontsize=10, loc="left")
        ax.text(.97, .09, f"{len(times)}/{len(data)} completed", transform=ax.transAxes,
                ha="right", fontsize=8)
        ax.set(xticks=[0, 2, 5, 10, 15], xlim=(0, 15), ylim=(0, 1.03))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[len(ids):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("Deadline (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Safe completion by deadline")
    finish(fig, stage, "Full completion-by-deadline curves",
           "Failures and unfinished starts remain in each denominator.")
    save(fig, folder, f"{stage}-deadlines")


def continuation(conditions, folder):
    ids = sorted(conditions)
    y = np.arange(len(ids))
    fig, ax = plt.subplots(figsize=(10, .65 * len(ids) + 2.5))
    for offset, key, title, color in (
        (-.17, "C_count", "Benchmark safe completion", "#83b6cd"),
        (.17, "sustained_count", "Sustained through 2 s continuation", "#23556d"),
    ):
        values = [conditions[c][key] / conditions[c]["valid"] for c in ids]
        ax.barh(y + offset, values, height=.3, label=title, color=color)
        for i, condition in enumerate(ids):
            row = conditions[condition]
            ax.text(values[i] + .01, i + offset, f"{row[key]}/{row['valid']}", va="center",
                    fontsize=8)
    ax.set_yticks(y, [label(c) for c in ids])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.14)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Fraction of all valid starts")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1), frameon=False, ncol=2, fontsize=9)
    ax.grid(axis="x", alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
    finish(fig, "C", "Stability under continued control",
           "Sustained success requires the full predicate and no violation for all 40 extra controls.")
    save(fig, folder, "C-continuation")


def paired_differences(analysis, folder):
    rows = analysis["primary_simultaneous"]
    if not rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, max(3.8, .65 * len(rows) + 2.1)), sharey=True)
    names = [f"{label(r['candidate'])}\nminus {label(r['reference'])}" for r in rows]
    for ax, metric, title in zip(axes, ("C", "V", "J"),
                                ("Safe completion difference", "Violation difference",
                                 "Penalized time difference (s)"), strict=True):
        for i, row in enumerate(rows):
            point = row[metric]["difference"]
            lo, hi = row[metric]["interval"]
            ax.plot([lo, hi], [i, i], color="#246b8e", lw=2)
            ax.plot(point, i, "o", color="#246b8e", markersize=5)
        ax.axvline(0, color="0.4", lw=.8, linestyle="--")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
        if metric != "J":
            ax.xaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_yticks(range(len(rows)), names, fontsize=8)
    axes[0].invert_yaxis()
    finish(fig, "C", "Predeclared paired comparisons",
           "Bonferroni-adjusted intervals across all primary C, V and J contrasts; familywise alpha 0.05.")
    save(fig, folder, "C-paired-differences")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analysis, validation, freeze = (
        json.loads(p.read_text()) for p in (args.analysis, args.validation, args.freeze)
    )
    if not validation["passed"] or validation["freeze_sha256"] != sha(args.freeze):
        raise ValueError("Figures require a complete validation of this exact freeze")
    stage = freeze["stage"]
    if stage not in {"B", "C"} or analysis["stage"] != stage:
        raise ValueError("Only complete performance panels are plotted")
    rows = validation["episodes"]
    if {r["job"]["id"] for r in rows} != {r["id"] for r in freeze["jobs"]}:
        raise ValueError("The validated panel is incomplete")
    conditions = analysis["conditions"]
    if set(conditions) != {r["condition"] for r in rows}:
        raise ValueError("Analysis and validation conditions differ")
    for condition, summary in conditions.items():
        group = [r for r in rows if r["condition"] == condition]
        if len(group) != (16 if stage == "B" else 64) or summary["valid"] != len(group):
            raise ValueError("A complete condition panel is required")
        if summary["C_count"] != sum(r["safe_success"] for r in group):
            raise ValueError("Completion counts differ")
        if summary["V_count"] != sum(r["violation"] for r in group):
            raise ValueError("Violation counts differ")
        if not np.isclose(summary["J"], np.mean([r["J"] for r in group]), rtol=0, atol=1e-12):
            raise ValueError("Timing aggregates differ")
    if stage == "B":
        response_map(conditions, args.output)
    speed_completion(stage, conditions, args.output)
    deadlines(stage, rows, args.output)
    if stage == "C":
        if not all(r["continuation_enabled"] for r in conditions.values()):
            raise ValueError("Confirmation continuation evidence is missing")
        continuation(conditions, args.output)
        paired_differences(analysis, args.output)
    manifest = {str(p): sha(p) for p in (args.analysis, args.validation, args.freeze)}
    manifest[str(Path(__file__))] = sha(Path(__file__))
    (args.output / f"{stage}-figure-inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"stage": stage, "output": str(args.output), "validated_complete_panel": True}))


if __name__ == "__main__":
    main()
