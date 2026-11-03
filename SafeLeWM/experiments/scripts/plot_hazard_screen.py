"""Inspect all saved hazard-screen trajectories on the rented host."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/hazard-screen-20261101"


def main():
    cases = json.loads((OUT / "panel.json").read_text())["cases"]
    for family in ["open", "constrained", "doorway"]:
        selected = [case for case in cases if case["family"] == family]
        fig, axes = plt.subplots(2, 4, figsize=(14, 7.6), layout="constrained")
        for row, seed in enumerate([6101, 6102]):
            for col, case in enumerate(selected):
                ax = axes[row, col]
                for arm, color in [
                    ("safe", "forestgreen"),
                    ("risky", "firebrick"),
                    ("monitored", "royalblue"),
                ]:
                    used_seed = seed if arm == "monitored" else 6101
                    folder = OUT / f"{case['id']}_{arm}_{used_seed}"
                    result = json.loads((folder / "result.json").read_text())
                    with np.load(folder / "trace.npz") as z:
                        if arm == "safe":
                            ax.imshow(z["frames"][0])
                            ax.scatter(*case["goal"], color="green", marker="*", s=55)
                        ax.plot(
                            z["states"][:, 0], z["states"][:, 1], color=color, label=arm, lw=1.3
                        )
                        ax.scatter(*z["states"][-1], color=color, s=12)
                    if arm == "monitored":
                        ax.set_title(
                            f"{case['id']} | seed {seed}\n"
                            f"AP complete={int(result['success'])}, "
                            f"violation={int(result['violation'])}",
                            fontsize=9,
                        )
                ax.add_patch(Circle(case["hazard_centre"], 8, color="orange", alpha=0.45))
                ax.add_patch(
                    Circle(
                        case["hazard_centre"], 18, fill=False, color="darkorange", linestyle="--"
                    )
                )
                ax.set_xlim(0, 224)
                ax.set_ylim(224, 0)
                ax.set_xticks([])
                ax.set_yticks([])
        axes[0, 0].legend(fontsize=8, loc="lower left")
        fig.suptitle(
            f"{family.capitalize()} family: every frozen layout and seed\n"
            "Hazard overlays are for inspection only; clean AP traces are identical."
        )
        fig.savefig(OUT / f"{family}-trajectories.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
