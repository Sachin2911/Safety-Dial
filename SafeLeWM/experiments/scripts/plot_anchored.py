"""Plot all saved trajectories; no simulation or model execution."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main(stage):
    folder = ROOT / f"outputs/anchored-20261101/{stage}-v1"
    panel = json.loads((folder / "panel.json").read_text())
    cases = panel["cases"]
    arms = ["final", "ap"] if stage == "long" else ["direct", "witness", "monitored"]
    colors = dict(final="slategray", ap="royalblue", direct="firebrick",
                  witness="forestgreen", monitored="royalblue")
    rows, cols = (5, 10) if stage == "long" else (2, 3)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.8, rows*3.0))
    for case, ax in zip(cases, axes.flat):
        labels = []
        for i, arm in enumerate(arms):
            path = folder / (case["id"] + "_" + arm)
            result = json.loads((path / "result.json").read_text())
            with np.load(path / "trace.npz") as z:
                if i == 0:
                    ax.imshow(z["frames"][0])
                    ax.scatter(*z["goal"], marker="*", s=45, color="green")
                states = z["states"]
                ax.plot(states[:,0], states[:,1], color=colors[arm], lw=1)
                ax.scatter(*states[-1], color=colors[arm], s=10)
            labels.append(f"{arm}: {int(result['success'])}, V={int(result['violation'])}")
        if stage == "hazard":
            ax.add_patch(Circle(case["hazard_centre"], 8, color="orange", alpha=.4))
            ax.add_patch(Circle(case["hazard_centre"], 18, fill=False, color="orange", ls="--"))
        ax.set_title(case["id"]+"\n"+"; ".join(labels), fontsize=7)
        ax.set_xlim(0,224)
        ax.set_ylim(224,0)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("All frozen development cases: completion and any violation (V)\n"
                 "Hazard overlays are for inspection only; never passed to the model.")
    fig.tight_layout(rect=(0,0,1,.96))
    fig.savefig(folder / "trajectories.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["long", "hazard"])
    main(parser.parse_args().stage)
