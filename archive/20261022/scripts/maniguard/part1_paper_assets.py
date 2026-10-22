"""Render GECCO-sized figures from the completed Part 1 audit, with no new queries."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build(audit, output):
    audit, output = Path(audit), Path(output)
    data = json.loads((audit / "tables.json").read_text())
    lineage = json.loads((audit / "lineage.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 9,
                         "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "pdf.fonttype": 42})
    categories = ("safe_complete", "unsafe_complete", "safe_unfinished", "unsafe_unfinished")
    colors = ("#228833", "#cc6677", "#88ccee", "#aa4499")
    figure, axes = plt.subplots(1, 3, figsize=(7, 2.7), layout="constrained")
    for axis, scene in zip(axes, ("task_0029/base", "task_0032/base", "all"), strict=True):
        rows = [r for r in data["per_scene_and_setting"] if r["scene"] == scene]
        base = [0, 0, 0]
        for key, color in zip(categories, colors, strict=True):
            values = [r[key] for r in rows]
            axis.bar(range(3), values, bottom=base, color=color, label=key.replace("_", " "))
            base = [a + b for a, b in zip(base, values, strict=True)]
        axis.set(xticks=range(3), xticklabels=[f"{r['dial']}\nQ={r['queries']}\n"
                                            f"t={r['process_seconds']:.0f}s" for r in rows],
                 title=scene.replace("/base", ""), ylabel="Episodes", xlabel="Dial, queries, time")
        for i, r in enumerate(rows):
            axis.text(i, r["n"] + 0.08, f"n={r['n']}", ha="center", fontsize=7)
        axis.set_ylim(0, max(r["n"] for r in rows) + 0.8)
        axis.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=4, loc="outside lower center", fontsize=7)
    figure.savefig(output / "paper-outcomes.pdf")
    figure.savefig(output / "paper-outcomes.png", dpi=240)
    plt.close(figure)
    totals = [r for r in data["per_scene_and_setting"] if r["scene"] == "all"]
    figure, axis = plt.subplots(figsize=(3.3, 2.5), layout="constrained")
    for key, label, color, offset in (("safe_requested_completion", "Safe completion", colors[0], 7),
                                      ("violations", "Raw violation", colors[1], -11)):
        axis.plot([r["dial"] for r in totals], [r[key] / r["n"] for r in totals],
                  "o", color=color, label=label)
        for r in totals:
            axis.annotate(f"{r[key]}/{r['n']}", (r["dial"], r[key] / r["n"]),
                          xytext=(0, offset), textcoords="offset points", ha="center", fontsize=7)
    axis.set(xlabel="Dial strength", ylabel="Observed fraction", xticks=[0, 0.5, 1],
             ylim=(-0.12, 1), xlim=(-0.13, 1.13))
    axis.legend(loc="upper right", frameon=False)
    figure.savefig(output / "paper-dial-response.pdf")
    figure.savefig(output / "paper-dial-response.png", dpi=240)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(3.3, 2.4), layout="constrained")
    for row in lineage:
        elite = row["kind"] == "elite_reevaluation"
        axis.plot(row["cumulative_recorded_physics_ticks"] / 1000, row["fitness"],
                  "s" if elite else "o", color="#ee7733" if elite else "#4477aa", markersize=5)
    axis.plot([], [], "o", color="#4477aa", label="New candidate")
    axis.plot([], [], "s", color="#ee7733", label="Elite re-evaluated")
    axis.set(xlabel="Recorded physics ticks (thousands)", ylabel="Mean utility", ylim=(-0.65, 1.2))
    axis.legend(loc="lower left", frameon=False)
    figure.savefig(output / "paper-lineage.pdf")
    figure.savefig(output / "paper-lineage.png", dpi=240)
    plt.close(figure)
    manifest = {
        "inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (audit / "tables.json", audit / "lineage.json")},
        "figures": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(output.glob("paper-*")) if p.suffix in (".pdf", ".png")},
        "scope": "Recorded observations only; no new physics, model query or fitted risk curve.",
    }
    (output / "paper-figure-provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.audit, args.output)
