"""Plot paired selected-setting trajectories against identical hazard geometry."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.portable_motion import clearance, shape_constraints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    out = args.root
    report = json.loads((out / "report.json").read_text())
    panel = json.loads((out / "panel.json").read_text())
    design = json.loads((out / "preparation-freeze.json").read_text())["design"]
    selected = report["selection"]["selected"]["id"]
    name = report["environment"]
    radius = report["motion_report"]["spec"]["body_radius"]
    margin = design["margin"][name]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
    for ax, case in zip(axes.flat, panel["cases"][6:]):
        paths = []
        rows = []
        for condition in ("control", "hazard"):
            folder = out / "evaluation" / selected / case["id"] / condition
            with np.load(folder / "trace.npz") as z:
                paths.append(z["dense"])
            rows.append(json.loads((folder / "result.json").read_text()))
        with np.load(out / "inputs" / (case["id"] + ".npz")) as z:
            reference = z["reference_route"]
        allpoints = np.concatenate([*paths, reference])
        centre = (allpoints.min(0) + allpoints.max(0)) / 2
        span = max(float(np.ptp(allpoints, axis=0).max()), 0.1 if name != "tworoom" else 40) * 1.25
        lo, hi = centre - span / 2, centre + span / 2
        x = np.linspace(lo[0], hi[0], 240)
        y = np.linspace(lo[1], hi[1], 240)
        xx, yy = np.meshgrid(x, y)
        grid = torch.as_tensor(np.stack([xx, yy], -1))
        distance = clearance(grid, shape_constraints(case["shape"]), 0).numpy()
        mask = np.where(distance <= 0, 2, np.where(distance < radius + margin, 1, 0))
        ax.contourf(
            xx, yy, mask, levels=[-0.5, 0.5, 1.5, 2.5], colors=["white", "#fbe3b6", "#d9828b"]
        )
        ax.plot(reference[:, 0], reference[:, 1], ":", color=".65", lw=1, label="Source reference")
        ax.plot(
            paths[0][:, 0], paths[0][:, 1], "--", color="#be7815", lw=1.5, label="Hazard inactive"
        )
        ax.plot(paths[1][:, 0], paths[1][:, 1], color="#087d68", lw=1.8, label="Hazard active")
        ax.scatter(*paths[0][0], c="black", s=20, zorder=5)
        diagnostic = next(
            d
            for d in report["paired_hazard_diagnostics"]
            if d["split"] == "evaluation"
            and d["configuration"] == selected
            and d["case"] == case["id"]
        )
        ax.set_title(
            case["id"]
            + "\n"
            + f"Safe completion: {rows[1]['safe_completion']} | deliberate: {diagnostic['deliberate_safe_adaptation']}",
            fontsize=9,
        )
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), aspect="equal")
        if name == "tworoom":
            ax.invert_yaxis()
        ax.set_xlabel("x (" + design["units"][name] + ")", fontsize=8)
        ax.set_ylabel("y (" + design["units"][name] + ")", fontsize=8)
        ax.tick_params(labelsize=7)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle(
        name
        + ": paired safety adaptation with "
        + selected
        + "\nRed: hazard; amber: protected body + requested margin. Control hazard is inactive and rescored.",
        fontsize=11,
    )
    fig.savefig(out / "safety-adaptation.png", dpi=160)
    plt.close(fig)
    print("Saved paired adaptation figure without new simulation", flush=True)


if __name__ == "__main__":
    main()
