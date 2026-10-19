"""Export the completed recovery diagnostic and vector figures without simulation."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "docs/research/figures/ManiGuard/phase3-recovery"
NOTES = ROOT / "docs/research/research_notes"


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def architecture():
    fig, ax = plt.subplots(figsize=(10.5, 3.15), constrained_layout=True)
    ax.set(xlim=(0, 10.5), ylim=(0, 3.2))
    ax.axis("off")

    def box(x, y, w, text, color="#eef1f4", edge="#65717f"):
        patch = FancyBboxPatch((x, y), w, .74, boxstyle="round,pad=.045,rounding_size=.08",
                               facecolor=color, edgecolor=edge, linewidth=1)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + .37, text, ha="center", va="center", fontsize=9)

    def arrow(start, end, text=None, dashed=False, color="#435365"):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                     linewidth=1.2, color=color, linestyle="--" if dashed else "-"))
        if text:
            ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + .10,
                    text, ha="center", fontsize=8, color=color)

    box(.12, 1.45, 1.65, "Frozen VLA\nRGB + robot input")
    box(2.35, 1.45, 1.8, "Execution controller\ncap + query schedule", "#e3f1f1", "#237e8a")
    box(4.75, 1.45, 1.45, "Action-space\nclipping")
    box(6.8, 1.45, 1.6, "Simulator\n+ actuators")
    box(8.95, 1.45, 1.4, "Native monitor\nlabels only", "#fcf0e9", "#bc6045")
    arrow((1.81, 1.82), (2.30, 1.82), "proposal p")
    arrow((4.20, 1.82), (4.70, 1.82), "adapted")
    arrow((6.25, 1.82), (6.75, 1.82), "input u")
    arrow((8.45, 1.82), (8.90, 1.82))
    ax.plot([7.6, 7.6, 3.25, 3.25], [1.4, .99, .99, 1.4], color="#237e8a", linewidth=1.1)
    ax.text(5.45, 1.08, "Measured joints + final input history; refresh uses past correction",
            ha="center", fontsize=8, color="#237e8a")
    arrow((3.25, 2.23), (.95, 2.23), "Refresh when due; discard stale suffix", color="#237e8a")
    box(4.15, .02, 2.2, "Offline candidate selection\nmeasured episode utility", "#f0edf7", "#77658f")
    ax.plot([9.65, 9.65, 6.4], [1.4, .38, .38], color="#77658f", linestyle="--", linewidth=1.2)
    arrow((6.4, .38), (6.35, .38), dashed=True, color="#77658f")
    ax.text(8.3, .48, "completion + violations", fontsize=8, ha="center", color="#77658f")
    ax.text(5.3, 2.9, "Online execution and offline selection have different information",
            ha="center", fontsize=11, weight="bold", color="#253448")
    fig.savefig(FIGURES / "architecture.pdf")
    fig.savefig(FIGURES / "architecture.png", dpi=180)
    plt.close(fig)


def outcomes():
    part1 = read(NOTES / "ManiGuard phase3 Part1 audit.json")
    original = [r for r in part1["comparison"]["per_scene_and_setting"] if r["scene"] == "all"]
    first = [[r[k] for k in ("safe_complete", "unsafe_complete", "safe_unfinished",
                             "unsafe_unfinished")] for r in original]
    second = [[3, 0, 0, 1], [3, 0, 0, 1], [2, 1, 1, 0], [3, 0, 0, 1]]
    # Assert the compact panel against every published Part2 episode, not prose alone.
    records = read(NOTES / "ManiGuard phase3 Part2 audit.json")["records"]
    verified = {name: [0, 0, 0, 0] for name in "ABCD"}
    for r in records:
        if not r.get("full_episode"):
            continue
        name = r["episode_id"].rsplit("-", 1)[-1]
        if name not in verified:
            continue
        complete = bool(r["C_req"])
        violation = bool(r["V"])
        verified[name][(0 if complete else 2) + int(violation)] += 1
    if [verified[name] for name in "ABCD"] != second:
        raise ValueError("Part2 figure differs from original independently audited counts")
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.3), constrained_layout=True)
    colors = ["#237e8a", "#bd6046", "#dce5ea", "#d8ab9d"]
    names = ["Safe completion", "Unsafe completion", "Safe unfinished", "Unsafe unfinished"]
    for ax, values, labels, title, n in zip(
        axes, (first, second), (["Identity", "Dial 0.5", "Dial 1"],
                              ["A: identity/8", "B: cap/8", "C: cap/4", "D: trigger"]),
        ("Evolutionary dial: fresh repetitions", "Feedback pilot: four conditions"), (8, 4), strict=True
    ):
        values = np.asarray(values)
        bottom = np.zeros(len(values))
        for k, (color, name) in enumerate(zip(colors, names, strict=True)):
            ax.bar(range(len(values)), values[:, k], bottom=bottom, color=color,
                   label=name, width=.62)
            bottom += values[:, k]
        for i, row in enumerate(values):
            ax.text(i, n + .14, f"{row[0]}/{n} safe", ha="center", fontsize=8)
        ax.set(xticks=range(len(values)), xticklabels=labels, ylim=(0, n + .65),
               yticks=range(n + 1), ylabel="Episodes", title=title)
        ax.tick_params(axis="x", labelsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, fontsize=8)
    fig.savefig(FIGURES / "outcomes.pdf")
    fig.savefig(FIGURES / "outcomes.png", dpi=180)
    plt.close(fig)


def schematic():
    fig, ax = plt.subplots(figsize=(4.6, 3.0), constrained_layout=True)
    origin = np.zeros(2)
    vectors = [([.18, .075], "Proposal error", "#5f6b7c"),
               ([.08, .075], "Componentwise", "#bd6046"),
               ([.08, .075 * .08 / .18], "Vector scaling", "#237e8a")]
    for vector, name, color in vectors:
        ax.annotate("", xy=vector, xytext=origin,
                    arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2})
        ax.plot([], [], color=color, label=name, lw=2)
    ax.axvline(.08, color="#8c939c", linestyle=":", lw=1)
    ax.axhline(.08, color="#8c939c", linestyle=":", lw=1)
    ax.set(xlim=(-.006, .2), ylim=(-.004, .1), xlabel="Joint-error coordinate 1 (rad)",
           ylabel="Coordinate 2 (rad)")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.set_title("Cap geometry: schematic in two of seven joints", fontsize=10)
    fig.savefig(FIGURES / "geometry-schematic.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, required=True)
    args = parser.parse_args()
    d = read(args.diagnostic / "diagnostic.json")
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    destination = NOTES / "ManiGuard recovery checkin diagnostic.json"
    shutil.copyfile(args.diagnostic / "diagnostic.json", destination)
    for suffix in ("pdf", "png"):
        shutil.copyfile(args.diagnostic / ("diagnostic." + suffix),
                        FIGURES / ("diagnostic." + suffix))
    architecture()
    outcomes()
    schematic()
    save(FIGURES / "provenance.json", {
        "schema": "maniguard-recovery-checkin-figures-v1",
        "diagnostic_sha256": sha(destination), "new_simulator_controls": 0,
        "new_policy_queries": 0, "new_encoder_calls": 0,
        "historical_controls_reused": d["historical_controls_reused"],
        "generator": str(Path(__file__).relative_to(ROOT)), "generator_sha256": sha(__file__),
        "files": {p.name: sha(p) for p in FIGURES.iterdir() if p.suffix in (".pdf", ".png")}})
    print(json.dumps({"figures": str(FIGURES), "diagnostic": str(destination)}))


if __name__ == "__main__":
    main()
