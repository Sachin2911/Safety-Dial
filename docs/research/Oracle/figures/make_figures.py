"""Draw the paper's charts from figures/data.json. Counts only; no experiment computation.

    python3 Oracle/figures/make_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "data.json").read_text())

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
    "axes.linewidth": 0.6, "xtick.color": "#52514e", "ytick.color": "#52514e",
    "xtick.major.size": 0, "ytick.major.size": 2, "ytick.major.width": 0.5,
})
INK, MUTED, GRID, ACCENT, REST = "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6", "#8a8984"
ORANGE = "#eb6834"


def wilson(k, n, z=1.959964):
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / d
    return max(0, c - h), min(1, c + h)


def headline():
    """Every held-out TwoRoom search, repeat by repeat, against the native reference."""
    panels = DATA["headline"]["panels"]
    fig, axes = plt.subplots(1, 3, figsize=(3.33, 1.72), sharey=True, gridspec_kw=dict(wspace=0.07))
    marks = (("cma", "CMA-ES", 0.0, dict(marker="o", s=26, color=ACCENT, edgecolor="white", linewidth=0.5, zorder=5)),
             ("random", "Random search", -0.24, dict(marker="D", s=11, color=ORANGE, edgecolor="white", linewidth=0.3, zorder=4)),
             ("untrained", "Untrained", 0.24, dict(marker="s", s=10, facecolor="white", edgecolor=MUTED, linewidth=0.7, zorder=3)),
             ("ppo", "PPO", 0.0, dict(marker="x", s=13, color=INK, linewidth=0.8, zorder=4)))
    seen = {}
    for ax, panel in zip(axes, panels):
        n = panel["denominator"]
        native = 100 * panel["native"] / n
        ax.axhline(native, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
        for key, label, dx, style in marks:
            for x, v in enumerate(panel.get(key, [])):
                # PPO sits on the native line, so it is drawn to the side of the CMA-ES dot.
                artist = ax.scatter(x + (0.24 if key == "ppo" else dx), 100 * v / n, **style)
                seen.setdefault(label, artist)
        ax.set_title(panel["title"], fontsize=6.5, color=INK, pad=3)
        ax.set_xlim(-0.55, 2.55)
        ax.set_xticks([0, 1, 2], ["1", "2", "3"])
        ax.set_xlabel("repeat", color=MUTED, fontsize=6.5, labelpad=1)
        ax.yaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.text(-0.5, 0.9, f"native {native:.1f}%", color=MUTED, fontsize=5.5, ha="left")
    axes[0].set_ylabel("safe completion (%)", color=MUTED, fontsize=6.5)
    axes[0].set_ylim(0, 34)
    for ax in axes[1:]:
        ax.tick_params(axis="y", length=0)
    fig.legend(seen.values(), seen.keys(), loc="upper center", ncol=4, frameon=False, fontsize=6.5,
               handletextpad=0.2, columnspacing=1.0, bbox_to_anchor=(0.52, 1.17))
    fig.savefig(HERE / "headline.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def block_frames():
    """Front-camera filmstrips of one start: 128 by 128 frames from recorded runs, at 20 Hz."""
    import numpy as np

    z = np.load(HERE / "block_frames.npz")
    rows = (("evolved", "evolved_steps", "Evolved\nrelease rule"), ("frozen", "frozen_steps", "Released actor,\nfilter live"))
    fig, axes = plt.subplots(2, 5, figsize=(3.33, 1.6), gridspec_kw=dict(wspace=0.04, hspace=0.26))
    for r, (key, steps, label) in enumerate(rows):
        for c in range(5):
            ax = axes[r, c]
            ax.imshow(z[key][c], interpolation="lanczos")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            seconds = z[steps][c] / 20
            ax.set_xlabel(f"{seconds:.1f} s", color=MUTED, fontsize=6, labelpad=1)
        axes[r, 0].set_ylabel(label, color=INK, fontsize=6.5, labelpad=3)
    fig.savefig(HERE / "block_frames.pdf", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)


def strip(ax, panel):
    """One column per controller: a tick at the mean, one dot per repeat, native dashed."""
    n = panel["denominator"]
    native = 100 * panel["native"] / n
    ax.axhline(native, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(len(panel["arms"]) - 0.45, native + 1.3, "native", color=MUTED, fontsize=6.5, ha="right")
    for i, arm in enumerate(panel["arms"]):
        ys = [100 * v / n for v in arm["repeats"]]
        colour = ACCENT if arm.get("highlight") else REST
        seen = {}
        xs = []
        for v in arm["repeats"]:  # equal repeats sit side by side so no dot is hidden
            seen[v] = seen.get(v, 0) + 1
            xs.append(i + 0.16 * (seen[v] - 1) * (-1) ** seen[v])
        ax.scatter(xs, ys, s=20, color=colour, edgecolor="white", linewidth=0.5, zorder=3)
        mean = sum(ys) / len(ys)
        ax.plot([i - 0.28, i + 0.28], [mean, mean], color=colour, lw=1.6, solid_capstyle="round", zorder=2)
    ax.set_xticks(range(len(panel["arms"])), [a["label"] for a in panel["arms"]])
    ax.set_title(panel["title"], fontsize=7.5, color=INK, loc="left", pad=4)
    ax.set_xlim(-0.6, len(panel["arms"]) - 0.4)
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def tworoom():
    panels = DATA["tworoom"]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.33, 1.75), sharey=True,
                             gridspec_kw=dict(width_ratios=[len(p["arms"]) for p in panels], wspace=0.08))
    axes = axes if len(panels) > 1 else [axes]
    for ax, panel in zip(axes, panels):
        strip(ax, panel)
    axes[0].set_ylabel("Safe completion (%)", color=MUTED)
    axes[0].set_ylim(0, max(100 * max(max(a["repeats"]) for a in p["arms"]) / p["denominator"] for p in panels) * 1.15)
    fig.savefig(HERE / "tworoom.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def block():
    rows, n = DATA["block"]["rows"], DATA["block"]["episodes"]
    fig, axes = plt.subplots(1, 2, figsize=(3.33, 1.05), sharey=True, sharex=True, gridspec_kw=dict(wspace=0.08))
    ys = list(range(len(rows)))[::-1]
    for ax, key, title in zip(axes, ("sustained", "violated"), (f"Sustained safe (of {n})", f"Violated (of {n})")):
        for y, row in zip(ys, rows):
            k = row[key]
            lo, hi = wilson(k, n)
            colour = ACCENT if row.get("highlight") else REST
            ax.plot([100 * lo, 100 * hi], [y, y], color=colour, lw=1.1, solid_capstyle="round", zorder=2)
            ax.scatter([100 * k / n], [y], s=18, color=colour, edgecolor="white", linewidth=0.5, zorder=3)
            ax.text(100 * hi + 2, y, str(k), va="center", fontsize=6.5, color=INK)
        ax.set_title(title, fontsize=7.5, color=INK, loc="left", pad=3)
        ax.xaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75])
        ax.set_xlabel("%, Wilson 95% interval", color=MUTED, fontsize=6.5)
    axes[0].set_yticks(ys, [r["label"] for r in rows])
    axes[0].set_ylim(-0.6, len(rows) - 0.4)
    for ax in axes:
        ax.tick_params(axis="y", length=0)
    fig.savefig(HERE / "block.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def layouts():
    """Safe completions per layout and arm: one sequential hue, counts printed in cells."""
    from matplotlib.colors import LinearSegmentedColormap

    d = DATA["layouts"]
    names = dict(native="Native (of 3)", untrained="Untrained", task="Task", blind="Blind", reset="Reset", random="Random", main="Main")
    open_ids = [i for i, c in enumerate(d["order"]) if d["family"][c] == "open"]
    wall_ids = [i for i, c in enumerate(d["order"]) if d["family"][c] != "open"]
    cmap = LinearSegmentedColormap.from_list("blues", ["#f4f3ef", "#b7d3f4", "#2a78d6", "#0f3f7f"])
    fig, axes = plt.subplots(1, 2, figsize=(3.33, 1.28), sharey=True, gridspec_kw=dict(wspace=0.04))
    for ax, ids, title in zip(axes, (open_ids, wall_ids), ("Open layouts", "Wall-constrained layouts")):
        for y, arm in enumerate(d["arms"]):
            top = 3 if arm == "native" else 9
            for x, i in enumerate(ids):
                v = d["safe"][arm][i]
                share = v / top
                ax.add_patch(plt.Rectangle((x, y), 0.94, 0.9, color=cmap(share), lw=0))
                if v:
                    ax.text(x + 0.47, y + 0.45, str(v), ha="center", va="center", fontsize=4.8, color="white" if share > 0.4 else INK)
        ax.set_xlim(0, len(ids))
        ax.set_ylim(len(d["arms"]), 0)
        ax.set_xticks([])
        ax.set_title(title, fontsize=7, color=INK, loc="left", pad=2)
        for side in ax.spines.values():
            side.set_visible(False)
    axes[0].set_yticks([y + 0.45 for y in range(len(d["arms"]))], [names[a] for a in d["arms"]], fontsize=6.5)
    for ax in axes:
        ax.tick_params(axis="y", length=0)
    fig.savefig(HERE / "layouts.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def training():
    """Best safe training completions so far against actual actions, per search."""
    fig, ax = plt.subplots(figsize=(3.33, 1.25))
    styles = dict(main=(ACCENT, "Main (CMA-ES)"), random=("#eb6834", "Random search"))
    for arm, (colour, label) in styles.items():
        for j, rep in enumerate(sorted(DATA["training"][arm])):
            spent, best, xs, ys = 0, None, [0], [0]
            for actions, eligible, safe, score in DATA["training"][arm][rep]:
                spent += actions
                if eligible and (best is None or score > best[0]):
                    best = (score, safe)
                if best:
                    xs.append(spent)
                    ys.append(best[1])
            ax.step([x / 1000 for x in xs], [y + {"main": 0.05, "random": -0.05}[arm] + 0.02 * j for y in ys], where="post", color=colour, lw=1.1, label=label if j == 0 else None, solid_capstyle="round")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 6)
    ax.set_yticks([0, 2, 4, 6])
    ax.set_xlabel("Training actions (thousands)", color=MUTED, fontsize=6.5)
    ax.set_ylabel("Best safe of 8", color=MUTED, fontsize=6.5)
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handlelength=1.4, borderaxespad=0, columnspacing=1.4)
    fig.savefig(HERE / "training.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


tworoom()
block()
layouts()
training()
headline()
block_frames()
