"""Render charts from saved result counts; no experiment execution."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/safelewm-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

results = Path(__file__).resolve().parent
graphs = results / "graphs"
graphs.mkdir(exist_ok=True)
studies = json.loads((results / "record.json").read_text())["studies"]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("studies", nargs="*", help="Optional study IDs; default is all studies")
selected = parser.parse_args().studies
if set(selected) - {study["id"] for study in studies}:
    parser.error("Unknown study ID")
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
    }
)
for s in studies:
    if selected and s["id"] not in selected:
        continue
    if s.get("figure_source"):
        print(f"{s['id']}: retain its rented-host analysis figure ({s['figure_source']})")
        continue
    fig, ax = plt.subplots(
        figsize=(8.4, 2.5 if len(s["labels"]) < 5 else 3.0), layout="constrained"
    )
    y = list(range(len(s["labels"])))
    values = [100 * c / n for c, n in zip(s["complete"], s["denom"])]
    if "safe_complete" in s:
        series = [
            (-0.24, "complete", "#246b8b", "Completed (including violations)"),
            (0, "safe_complete", "#41804d", "Safe completion"),
            (0.24, "violations", "#bd602c", "Any violation"),
        ]
        for offset, field, color, label in series:
            # A null count marks an arm where the outcome is not applicable
            # (for example safety fields of a no-hazard arm); it draws no bar.
            rows = [(i, c, n) for i, (c, n) in enumerate(zip(s[field], s["denom"])) if c is not None]
            ax.barh([i + offset for i, _, _ in rows], [100 * c / n for _, c, n in rows],
                    height=0.2, color=color, label=label)
            for i, c, n in rows:
                ax.text(100 * c / n + 1, i + offset, f"{c}/{n}", va="center", fontsize=9)
            for i, c in enumerate(s[field]):
                if c is None:
                    ax.text(1, i + offset, "not applicable", va="center", fontsize=8, color="#8a9399")
        ax.legend(loc="upper center", bbox_to_anchor=(0.44, -0.3), frameon=False, fontsize=9)
    elif "violations" in s:
        ax.barh(
            [i - 0.17 for i in y],
            values,
            height=0.3,
            color="#246b8b",
            label="Completed (including violations)",
        )
        ax.barh(
            [i + 0.17 for i in y],
            [100 * v / n for v, n in zip(s["violations"], s["denom"])],
            height=0.3,
            color="#bd602c",
            label="Any violation",
        )
        for i, (c, v, n) in enumerate(zip(s["complete"], s["violations"], s["denom"])):
            ax.text(100 * c / n + 1, i - 0.17, f"{c}/{n}", va="center", fontsize=10)
            ax.text(100 * v / n + 1, i + 0.17, f"{v}/{n}", va="center", fontsize=10)
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=1, frameon=False, fontsize=9
        )
    else:
        ax.barh(y, [100] * len(y), height=0.52, color="#e9eef1")
        ax.barh(y, values, height=0.52, color="#246b8b")
        for i, (c, n) in enumerate(zip(s["complete"], s["denom"])):
            ax.text(102, i, f"{c}/{n}", va="center", fontsize=11)
    ax.set_yticks(y, s["labels"])
    ax.invert_yaxis()
    ax.set_xlim(0, 115)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Episodes completed" if "violations" not in s else "Episode rate")
    ax.tick_params(axis="both", length=0)
    fig.savefig(graphs / (s["id"] + ".pdf"), bbox_inches="tight")
    fig.savefig(graphs / (s["id"] + ".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
