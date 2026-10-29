"""Build the task 19 (ManiGuard Jar tip shield) figures from frozen analysis files.

Reads outputs/evo19/analysis/summary.json (confirmation and search tallies),
outputs/evo19/analysis/mechanism.json (per-episode failure descriptors, written by
experiments/scripts/mechanism_evo19.py) and outputs/evo19/search/state.json (stage 4
candidates), checks the plotted counts against the receipt values, and writes
assets/evo19-result.pdf and assets/evo19-search.pdf.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
RED, SLATE, TEAL, PALE, DARK = "#b33b39", "#64748b", "#137c82", "#cbd5e1", "#263d50"

ANALYSIS = ROOT / "outputs/evo19/analysis"
s = json.loads((ANALYSIS / "summary.json").read_text())
m = json.loads((ANALYSIS / "mechanism.json").read_text())
state = json.loads((ROOT / "outputs/evo19/search/state.json").read_text())

# Confirmation counts, as reported in the receipt.
mix = s["outcome_mix"]
assert s["primary"]["paired_task_seeds"] == 107
assert mix["native"] == {"never_closed": 12, "stall_after_closure": 26, "strict_safe": 32,
                         "violation": 37}
assert mix["evolved"] == {"never_closed": 14, "stall_after_closure": 40, "strict_safe": 32,
                          "violation": 21}
assert (s["secondary_violations"]["native"], s["secondary_violations"]["evolved"]) == (37, 21)
summary = m["summary"]
assert summary["paired_task_seeds"] == 107
assert summary["shield_rule_vs_logged_retraces_mismatches"] == []
for arm, table in (("native", "rule_native_confirmation"), ("evolved", "rule_shield_confirmation")):
    assert {k: v["episodes"] for k, v in summary[table].items()} == mix[arm]

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.05),
                         gridspec_kw={"width_ratios": [1.3, 1.0, 1.0]})

# (a) Outcome mix on the matched pairs.
ax = axes[0]
segments = [("strict_safe", "Finished safely", TEAL, "white"),
            ("violation", "Violation", RED, "white"),
            ("stall_after_closure", "Stalled after closing", SLATE, "white"),
            ("never_closed", "Never closed", PALE, DARK)]
for y, arm in ((1, "native"), (0, "evolved")):
    left = 0
    for key, label, color, text in segments:
        n = mix[arm][key]
        ax.barh(y, n, left=left, color=color, height=0.62, edgecolor="white", linewidth=0.6,
                label=label if y == 1 else None)
        ax.text(left + n / 2, y, str(n), ha="center", va="center", color=text, fontsize=8)
        left += n
ax.set(yticks=[1, 0], yticklabels=["Native\n$\\pi_{0.5}$", "With tip\nshield"],
       xlim=(0, 107), xticks=[0, 25, 50, 75, 107], xlabel="Matched attempts (of 107)",
       ylim=(-0.55, 1.55))
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.42), ncol=2, frameon=False,
          handlelength=1, columnspacing=0.9, fontsize=7.5)
ax.set_title("(a) Outcomes", fontsize=8, loc="left", pad=30)

# (b) Jar tilt before each native tip, aligned to the violation.
ax = axes[1]
tips = [r for r in m["episodes"] if r["kind"] == "confirm_native"
        and r.get("kind_of_violation") == "tip" and r["category"] == "violation"]
assert len(tips) == summary["native_confirmation"]["by_kind"]["tip"] == 36
grid = np.arange(-120, 11)
stack = np.full((len(tips), grid.size), np.nan)
for i, r in enumerate(tips):
    trace = np.asarray(r["trace_tilt_deg"])
    xs = np.arange(trace.size) + r["trace_offset"]
    ax.plot(xs, np.minimum(trace, 45), color=RED, alpha=0.18, linewidth=0.6)
    stack[i, np.searchsorted(grid, xs[0]):np.searchsorted(grid, xs[0]) + trace.size] = trace
ax.plot(grid, np.minimum(np.nanmedian(stack, axis=0), 45), color=RED, linewidth=1.6,
        label="Median")
limit = s["evolved"]["settings"]["tilt_limit_deg"]
ax.axhline(limit, color=TEAL, linestyle="--", linewidth=1.0)
ax.text(-117, limit + 1.2, f"shield limit {limit:.1f}°", color=TEAL, fontsize=7)
ax.axvline(0, color="0.3", linewidth=0.7)
ax.text(-2, 41.5, "violation", fontsize=7, color="0.3", ha="right")
ax.set(xlim=(-120, 10), ylim=(0, 45), xticks=[-120, -80, -40, 0], yticks=[0, 15, 30, 45],
       xlabel="Controls before violation (20/s)", ylabel="Jar tilt (degrees)")
ax.grid(axis="y", color="0.90", linewidth=0.5)
ax.set_title(f"(b) Tilt before {len(tips)} native tips", fontsize=8, loc="left")

# (c) Hand speed as the tilt last rose through 5 degrees, against task 15's speed limit.
ax = axes[2]
speeds = np.array([r["hand_speed_at_tilt5_mps"] for r in tips])
below = int((speeds < 0.10).sum())
assert below == summary["native_confirmation"]["tips_slower_than_0p10_at_tilt5"] == 23
rng = np.random.default_rng(19)
jitter = rng.uniform(-0.28, 0.28, speeds.size)
ax.scatter(speeds, jitter, s=14, facecolor="none", edgecolor=RED, linewidth=0.9)
ax.axvline(0.10, color=SLATE, linestyle="--", linewidth=1.0)
ax.text(0.112, 0.78, "earlier speed limit\n0.10 m/s", color=SLATE, fontsize=7, va="top")
ax.set(xscale="log", xlim=(0.01, 1.0), ylim=(-0.4, 0.82), yticks=[],
       xlabel="Hand speed at 5° tilt (m/s)")
ax.set_xticks([0.01, 0.03, 0.1, 0.3, 1.0], ["0.01", "0.03", "0.1", "0.3", "1"])
ax.spines["left"].set_visible(False)
ax.set_title(f"(c) {below} of {speeds.size} tips under the limit", fontsize=8, loc="left")
fig.tight_layout(w_pad=3.0)
fig.savefig(OUT / "evo19-result.pdf")
plt.close(fig)

# Search: stage 4 candidates in gene space, and every stage's outcomes.
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55), gridspec_kw={"width_ratios": [1.0, 1.15]})
ax = axes[0]
gens = state["generations4"]
assert len(gens) == 8 and all(len(g["jobs"]) == 6 for g in gens)
style = {"strict_safe": (TEAL, "o", "Finished safely"), "violation": (RED, "x", "Violation"),
         "other": (SLATE, "o", "Did not finish")}
seen = set()
for g in gens:
    for jid in g["jobs"]:
        job = state["jobs"][jid]
        o = job["outcome"]
        key = "strict_safe" if o["strict_safe"] else "violation" if o["raw_violation"] else "other"
        color, marker, label = style[key]
        x, y = job["settings"]["tilt_limit_deg"], 100 * job["settings"]["retreat_m"]
        kwargs = ({"color": color} if marker == "x"
                  else {"facecolor": color if key == "strict_safe" else "none", "edgecolor": color})
        ax.scatter(x, y, marker=marker, s=16, linewidth=0.9, zorder=3,
                   label=None if key in seen else label, **kwargs)
        seen.add(key)
means = [(g["mean_settings"]["tilt_limit_deg"], 100 * g["mean_settings"]["retreat_m"]) for g in gens]
frozen = s["evolved"]["settings"]
means.append((frozen["tilt_limit_deg"], 100 * frozen["retreat_m"]))
ax.plot(*zip(*means), color=DARK, linewidth=1.0, zorder=2, label="Search mean")
ax.scatter(*means[0], marker="s", s=28, color=DARK, zorder=4)
ax.scatter(*means[-1], marker="*", s=80, color="#e0a526", edgecolor=DARK, linewidth=0.6,
           zorder=5, label="Frozen setting")
ax.annotate("start (9°, 7 cm) and frozen\nsetting almost coincide", means[-1],
            xytext=(3.4, 10.4), fontsize=7,
            arrowprops={"arrowstyle": "->", "color": "0.55", "linewidth": 0.7})
ax.set(xlim=(3, 15), ylim=(2, 12.6), yticks=[2, 4, 6, 8, 10, 12], xlabel="Tilt limit (degrees)",
       ylabel="Retreat distance (cm)")
ax.grid(color="0.90", linewidth=0.5)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=3, fontsize=7, frameon=False,
          handletextpad=0.3, columnspacing=0.8)
ax.set_title("(a) Stage 4 search: 8 generations of 6", fontsize=8, loc="left", pad=40)

ax = axes[1]
stages = [(1, "1: speed\nprojection\n(n = 54)"), (2, "2: speed\nprojection,\nall directions\n(n = 12)"),
          (3, "3: tip shield,\ncontact now\n(n = 6)"), (4, "4: tip shield,\ncontact in 8\n(n = 48)")]
expected = {1: (54, 17, 17), 2: (12, 0, 5), 3: (6, 1, 3), 4: (48, 15, 10)}
x = np.arange(len(stages))
for i, (k, _) in enumerate(stages):
    t = s[f"search_stage{k}"]
    assert (t["valid"], t["strict_safe"], t["raw_violation"]) == expected[k]
    for dx, count, color, hatch in ((-0.21, t["strict_safe"], TEAL, None),
                                    (0.21, t["raw_violation"], RED, "//")):
        ax.bar(i + dx, 100 * count / t["valid"], 0.40, color=color, hatch=hatch,
               edgecolor="white", linewidth=0.5)
        ax.text(i + dx, 100 * count / t["valid"] + 2, f"{count}", ha="center",
                fontsize=6.5, zorder=4,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4})
for value, color, label in ((100 * 32 / 107, TEAL, "Native in confirmation: finished safely"),
                            (100 * 37 / 107, RED, "Native in confirmation: violations")):
    ax.axhline(value, color=color, linestyle=":", linewidth=1.0, label=label, zorder=0)
ax.bar(0, 0, color=TEAL, label="Finished safely")
ax.bar(0, 0, color=RED, hatch="//", edgecolor="white", label="Violation")
ax.set(xticks=x, xlim=(-0.6, 3.6), ylim=(0, 60), yticks=[0, 20, 40, 60],
       ylabel="Search attempts (%)")
ax.set_xticklabels([label for _, label in stages], fontsize=7)
ax.grid(axis="y", color="0.90", linewidth=0.5)
handles, labels = ax.get_legend_handles_labels()
order = [labels.index(k) for k in ("Finished safely", "Violation",
                                   "Native in confirmation: finished safely",
                                   "Native in confirmation: violations")]
ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center",
          bbox_to_anchor=(0.5, 1.03), ncol=2, fontsize=7, frameon=False, handletextpad=0.4,
          columnspacing=0.9)
ax.set_title("(b) Every search stage (single attempts)", fontsize=8, loc="left", pad=40)
fig.tight_layout(w_pad=2.0)
fig.savefig(OUT / "evo19-search.pdf")
plt.close(fig)
# History: every completed Jar intervention study against its own native arm. Studies
# are never pooled; each row is intervention minus native within one study, with a
# Newcombe hybrid-score 95% interval for a difference of independent proportions.
# (label, kind, safe_i, safe_n, n_i, n_n, viol_i, viol_n, source)
HISTORY = [
    ("Task 3: retreat and retry when the arm stops", "final", 16, 17, 96, 96, 47, 48,
     "progress/receipts/03-results.md (benchmark safe success; counted violations)"),
    ("Task 5: joint-feedback execution", "final", 1, 1, 12, 12, 3, 5,
     "progress/receipts/05-adapter-results.md (fixed preset)"),
    ("Task 7: DSRL latent steering", "final", 0, 3, 12, 12, 4, 4,
     "progress/receipts/07-dsrl-results.md"),
    ("Task 8: slower joint targets", "final", 2, 2, 12, 12, 3, 6,
     "progress/receipts/08-screen-results.md (stage B confirmation)"),
    ("Task 15: speed projection", "screen", 9, 5, 16, 16, 2, 6,
     "outputs/regrasp15/remote/attempts_stage_b.jsonl (development screen)"),
]
p, v = s["primary"], s["secondary_violations"]
HISTORY.append(("Task 19: tip shield", "final", p["evolved"], p["native"], 107, 107,
                v["evolved"], v["native"], "outputs/evo19/analysis/summary.json"))
assert (p["evolved"], p["native"], v["evolved"], v["native"]) == (32, 32, 21, 37)


def wilson(k, n, z=1.959964):
    centre = (k + z * z / 2) / (n + z * z)
    half = z * np.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return centre - half, centre + half


def newcombe(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    (l1, u1), (l2, u2) = wilson(k1, n1), wilson(k2, n2)
    d = p1 - p2
    return d, d - np.hypot(p1 - l1, u2 - p2), d + np.hypot(u1 - p1, p2 - l2)


fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.25), sharey=True)
ys = np.arange(len(HISTORY))[::-1]
history_rows = []
for ax, (title, ki, kn) in zip(axes, (("Safe completion", 2, 3), ("Violations", 6, 7)),
                               strict=True):
    for y, row in zip(ys, HISTORY, strict=True):
        d, lo, hi = (100 * x for x in newcombe(row[ki], row[4], row[kn], row[5]))
        color = TEAL if row[0].startswith("Task 19") else SLATE
        ax.plot([lo, hi], [y, y], color=color, linewidth=1.4, solid_capstyle="butt")
        ax.scatter(d, y, s=26, zorder=3, color=color, marker="o",
                   facecolor="white" if row[1] == "screen" else color, linewidth=1.2)
        ax.text(66, y, f"{row[ki]}/{row[4]} vs {row[kn]}/{row[5]}", va="center", fontsize=6.5,
                color="0.25")
        history_rows.append({"study": row[0], "status": row[1], "endpoint": title,
                             "intervention": [row[ki], row[4]], "native": [row[kn], row[5]],
                             "difference_points": round(d, 1),
                             "newcombe95_points": [round(lo, 1), round(hi, 1)],
                             "source": row[8]})
    ax.axvline(0, color="0.3", linewidth=0.8)
    ax.set(xlim=(-70, 150), xticks=[-60, -30, 0, 30, 60],
           xlabel="Intervention minus native (points)")
    ax.set_title(f"({'a' if ki == 2 else 'b'}) {title}", fontsize=8, loc="left")
    ax.grid(axis="x", color="0.90", linewidth=0.5)
axes[0].set(yticks=ys, yticklabels=[row[0] for row in HISTORY], ylim=(-0.6, len(HISTORY) - 0.4))
axes[0].tick_params(axis="y", length=0)
fig.tight_layout(w_pad=2.0)
fig.savefig(OUT / "jar-history.pdf")
plt.close(fig)
(OUT / "jar-history-data.json").write_text(json.dumps(history_rows, indent=2) + "\n")
print("Task 19 figures generated; plotted counts verified against summary and mechanism.")
