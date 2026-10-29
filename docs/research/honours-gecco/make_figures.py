"""Build manuscript figures directly from frozen experiment statistics."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "assets"
OUT.mkdir(exist_ok=True)
plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
    }
)

s16 = json.loads((ROOT / "progress/receipts/16-final-statistics.json").read_text())
s17 = json.loads((ROOT / "progress/receipts/17-final-statistics.json").read_text())
rows = []
for env, margin, expected in [("tworoom", 6, (3, 34)), ("reacher", 0.015, (0, 33))]:
    values = []
    for method, count in zip(["canonical", "fixed"], expected, strict=True):
        row = next(
            r
            for r in s16["aggregates"]
            if (r["stage"], r["env"], r["margin"], r["condition"])
            == ("A", env, margin, method)
        )
        assert row["n"] == 36 and row["requested_safe_completion"] == count
        assert row["any_violation"] == row["margin_breach"] == 0
        values.append(count)
    rows.append(values)

fig, ax = plt.subplots(figsize=(3.25, 2.0))
x = np.arange(2)
for i, (label, color, hatch) in enumerate(
    [("Original checked", "#64748b", "//"), ("Better fixed settings", "#137c82", None)]
):
    values = np.array(rows)[:, i]
    bars = ax.bar(x + (i - 0.5) * 0.34, values / 36 * 100, 0.30,
                  label=label, color=color, hatch=hatch, edgecolor="white")
    ax.bar_label(bars, labels=[f"{v}/36" for v in values], padding=3, fontsize=8)
ax.set(xticks=x, xticklabels=["TwoRoom, 6 px", "Reacher, 1.5 cm"],
       ylabel="Safe completion (%)", ylim=(0, 116), yticks=[0, 25, 50, 75, 100])
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), frameon=False, ncol=2,
          handlelength=1, columnspacing=1)
fig.savefig(OUT / "progress-recovery.pdf")
plt.close(fig)

# Full familiar-layout comparison, keeping the clearance and planner axes separate.
methods = [
    ("native", "No safety check", "#b33b39", "o", ":"),
    ("canonical", "Original SafeCEM", "#64748b", "s", "--"),
    ("fixed", "Tuned fixed", "#137c82", "^", "-"),
    ("adaptive", "Feedback", "#8b5ca6", "D", "-."),
]
measures = [
    ("requested_safe_completion", "Finished safely (%)"),
    ("margin_breach", "Breached requested margin (%)"),
    ("unfinished", "Did not finish (%)"),
]
fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.3))
dial_rows = []
for ri, (env, label, unit, factor) in enumerate(
    [("tworoom", "TwoRoom", "pixels", 1), ("reacher", "Reacher", "cm", 100)]
):
    for method, method_label, color, marker, style in methods:
        values = sorted(
            [r for r in s16["aggregates"]
             if (r["stage"], r["env"], r["condition"]) == ("A", env, method)],
            key=lambda r: r["margin"],
        )
        assert len(values) == 3
        for r in values:
            assert r["n"] == r["engaged"] == 36 and r["censored"] == 0
            if method != "native":
                assert r["margin_breach"] == r["any_violation"] == 0
            dial_rows.append({k: r[k] for k in [
                "env", "condition", "margin", "n", "requested_safe_completion",
                "any_violation", "margin_breach", "unfinished", "engaged", "censored",
            ]})
        xs = [r["margin"] * factor for r in values]
        for ci, (metric, title) in enumerate(measures):
            ax = axes[ri, ci]
            ax.plot(xs, [100 * r[metric] / r["n"] for r in values],
                    label=method_label, color=color, marker=marker, linestyle=style,
                    markersize=5, markerfacecolor="none", linewidth=1.3)
            ax.set(xticks=xs, ylim=(-5, 105), yticks=[0, 50, 100],
                   xlabel=f"{label}: margin ({unit})")
            if ri == 0:
                ax.set_title(title, fontsize=8)
            ax.grid(axis="y", color="0.90", linewidth=0.5)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, 1.02), fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.5, w_pad=1.3)
fig.savefig(OUT / "dial-comparison.pdf")
fig.savefig(OUT / "dial-comparison.png", dpi=180)
(OUT / "dial-comparison-data.json").write_text(json.dumps(dial_rows, indent=2) + "\n")
plt.close(fig)

contrasts = s17["evolution_minus"]["reacher"]["0.015"]
fig, ax = plt.subplots(figsize=(3.25, 2.05))
labels = ["Random search", "Policy gradient", "Retuned fixed"]
for y, method in zip([2, 1, 0], ["random", "reinforce", "tuned"], strict=True):
    row = contrasts[method]
    estimate = 100 * row["estimate"]
    lo, hi = np.array(row["interval"]) * 100
    ax.errorbar(estimate, y, xerr=[[estimate - lo], [hi - estimate]],
                fmt="o", color="#137c82", capsize=3, linewidth=1.4, markersize=4)
    ax.text(27, y, f"{estimate:+.1f}", va="center", fontsize=8)
ax.axvline(0, color="0.3", linewidth=0.8)
ax.axvline(10, color="0.5", linestyle="--", linewidth=0.8)
ax.set(yticks=[2, 1, 0], yticklabels=labels, ylim=(-0.6, 2.6), xlim=(-26, 35),
       xlabel="Evolution minus comparator (percentage points)", xticks=[-20, -10, 0, 10, 20, 30])
fig.savefig(OUT / "primary-contrasts.pdf")
plt.close(fig)

# Conceptual paths, deliberately separate from measured experiment trajectories.
fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.85))
paths = [
    ([0.4, 4.6], [0.6, 0.6], "#b33b39", "Reaches the goal\nbut crosses danger"),
    ([0.4, 1.65], [0.6, 0.6], "#64748b", "Avoids danger\nbut does not finish"),
    ([0.4, 1.4, 1.7, 3.3, 3.7, 4.6], [0.6, 0.6, 1.85, 1.85, 0.6, 0.6],
     "#137c82", "Finishes safely\nThis is what we want"),
]
for ax, (xs, ys, color, title) in zip(axes, paths, strict=True):
    ax.add_patch(Rectangle((2.0, 0), 1.0, 1.5, facecolor="#f5cccc", edgecolor="#b33b39"))
    ax.add_patch(Rectangle((2.0, 2.2), 1.0, 0.4, facecolor="#f5cccc", edgecolor="#b33b39"))
    ax.plot(xs, ys, color=color, linewidth=2.0)
    ax.scatter([0.4], [0.6], color="#263d50", s=22, zorder=4)
    ax.scatter([4.6], [0.6], color="#137c82", marker="*", s=60, zorder=4)
    ax.text(0.4, 0.12, "Start", ha="center", fontsize=8)
    ax.text(4.6, 0.12, "Goal", ha="center", fontsize=8)
    ax.text(2.5, 1.05, "Danger", ha="center", fontsize=7, color="#8b2323")
    ax.set(xlim=(-0.1, 5.1), ylim=(-0.1, 2.65), title=title)
    ax.set_axis_off()
fig.tight_layout(w_pad=1.2)
fig.savefig(OUT / "three-outcomes.pdf")
fig.savefig(OUT / "three-outcomes.png", dpi=180)
plt.close(fig)
# The three outcomes of Figure 1, measured at the largest tested clearance in all four
# planning tasks. Each bar is one cell of 36 attempts; tasks are separate studies.
s18 = json.loads((ROOT / "outputs/lewm18-combined/analysis/summary.json").read_text())
panels = [
    ("TwoRoom, 6 px", "familiar layouts", "16", "tworoom", 6),
    ("Reacher, 1.5 cm", "familiar layouts", "16", "reacher", 0.015),
    ("PushT, 12 px", "reserved cases", "18", "pusht", 12),
    ("Cube, 1 cm", "reserved cases", "18", "cube", 0.01),
]
planners = [("native", "No safety check"), ("canonical", "Original SafeCEM"),
            ("fixed", "Tuned fixed"), ("feedback", "Feedback")]
expected_safe = {"tworoom": (1, 3, 34, 36), "reacher": (1, 0, 33, 29),
                 "pusht": (0, 2, 19, 20), "cube": (0, 0, 12, 12)}
outcome_rows = []
fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.75), sharey=True)
for ax, (title, panel, task, env, margin) in zip(axes, panels, strict=True):
    for y, (method, label) in zip([3, 2, 1, 0], planners, strict=True):
        if task == "16":
            r = next(r for r in s16["aggregates"] if (r["stage"], r["env"], r["margin"]) == (
                "A", env, margin) and r["condition"] == {"feedback": "adaptive"}.get(method, method))
            safe, breach = r["requested_safe_completion"], r["margin_breach"]
        else:
            r = next(r for r in s18["tables"] if (r["stage"], r["env"], r["margin"], r["method"])
                     == ("H", env, margin, method))
            safe, breach = r["safe"], r["margin_breach"]
        assert r["n"] == 36 and safe + breach <= 36
        assert (method == "native") or breach == 0
        short = 36 - safe - breach
        outcome_rows.append({"task": env, "margin": margin, "planner": method, "n": 36,
                             "finished_safely": safe, "crossed_boundary": breach,
                             "did_not_finish_safely_without_crossing": short})
        left = 0
        for n, color, text in ((safe, "#137c82", "white"), (breach, "#b33b39", "white"),
                               (short, "#64748b", "white")):
            ax.barh(y, n, left=left, color=color, height=0.68, edgecolor="white", linewidth=0.6)
            if n >= 4:
                ax.text(left + n / 2, y, str(n), ha="center", va="center", color=text,
                        fontsize=7)
            left += n
    assert tuple(row["finished_safely"] for row in outcome_rows[-4:]) == expected_safe[env]
    ax.set(xlim=(0, 36), xticks=[0, 18, 36], ylim=(-0.5, 3.5), xlabel="Attempts (of 36)")
    ax.set_title(f"{title}\n{panel}", fontsize=8)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
axes[0].set(yticks=[3, 2, 1, 0], yticklabels=[label for _, label in planners])
handles = [Rectangle((0, 0), 1, 1, color=c) for c in ("#137c82", "#b33b39", "#64748b")]
fig.legend(handles, ["Finished safely", "Breached the margin", "Stopped short, no breach"],
           loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.55, 1.1),
           handlelength=1, columnspacing=1.2)
fig.tight_layout(w_pad=0.8)
fig.savefig(OUT / "planning-outcomes.pdf")
plt.close(fig)
(OUT / "planning-outcomes-data.json").write_text(json.dumps(outcome_rows, indent=2) + "\n")
# UNISafe confirmation: every setting's 64 attempts split into safe completion, a
# violation (physical failure or boundary exit) and a timeout.
s09 = json.loads((ROOT / "progress/receipts/09-unisafe-C-combined-analysis.json").read_text())
settings = [("d00_g100", "Native"), ("d15_g100", "Sensed release, delay 15"),
            ("history_only", "History-only release"), ("d40_g100", "Sensed release, delay 40"),
            ("reference_random", "Earlier random-search setting"),
            ("d15_g075", "Delay 15, translation 0.75")]
expected = {"d00_g100": (3, 9, 52), "d15_g100": (50, 8, 6), "history_only": (31, 8, 25),
            "d40_g100": (47, 9, 8), "reference_random": (51, 9, 4), "d15_g075": (54, 4, 6)}
fig, ax = plt.subplots(figsize=(3.25, 1.95))
for y, (key, label) in zip(range(len(settings) - 1, -1, -1), settings, strict=True):
    c = s09["conditions"][key]
    o = c["outcomes"]
    counts = (o["success"], o["physical_failure"] + o["boundary_exit"], o["timeout"])
    assert c["valid"] == 64 and sum(counts) == 64 and counts == expected[key]
    assert (c["C_count"], c["V_count"]) == counts[:2]
    left = 0
    for n, color in zip(counts, ("#137c82", "#b33b39", "#64748b"), strict=True):
        ax.barh(y, n, left=left, color=color, height=0.66, edgecolor="white", linewidth=0.6)
        if n >= 4:
            ax.text(left + n / 2, y, str(n), ha="center", va="center", color="white", fontsize=7)
        left += n
ax.set(xlim=(0, 64), xticks=[0, 16, 32, 48, 64], xlabel="Attempts (of 64)",
       yticks=range(len(settings) - 1, -1, -1), yticklabels=[label for _, label in settings],
       ylim=(-0.5, len(settings) - 0.5))
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
handles = [Rectangle((0, 0), 1, 1, color=c) for c in ("#137c82", "#b33b39", "#64748b")]
fig.legend(handles, ["Finished safely", "Violation", "Timed out"], loc="upper center", ncol=3,
           frameon=False, bbox_to_anchor=(0.55, 1.08), handlelength=1, columnspacing=1.0)
fig.savefig(OUT / "unisafe-outcomes.pdf")
plt.close(fig)
print("Figures generated from frozen statistics; plotted recovery counts verified.")
