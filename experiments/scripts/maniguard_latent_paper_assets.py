"""Plot the completed decoder audit from its compact, preserved numerical summary.

This consumes no simulator interaction, policy query or encoder call. Intervals
are response distributions over correlated decoder probes, not confidence bounds.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", type=Path,
        default=ROOT / "configs/maniguard/latent-evolution/development-summary-20261019.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "docs/research/figures/ManiGuard/latent-evolution",
    )
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axis = plt.subplots(figsize=(3.35, 2.15), layout="constrained")
    distributions = []
    colors = ["#176A92", "#946239"]
    for family_index, family in enumerate(("jar", "clutter")):
        audit = summary["authority"][family]
        assert audit["decoder_calls"] == 160 and audit["same_noise_identity_pass"]
        assert audit["authority_pass"] and len(audit["directions_passing"]) == 4
        values = [
            np.asarray(audit["perturbed_prefix_arm_rms_quantiles_rad"]),
            np.quantile(audit["native_resampling_prefix_rms"], [0, .25, .5, .75, 1]),
        ]
        for kind, quantiles in enumerate(values):
            assert quantiles.shape == (5,) and np.isfinite(quantiles).all()
            assert np.all(np.diff(quantiles) >= 0)
            position = family_index * 2 + (kind - .5) * .48
            scaled = quantiles * 1000
            axis.hlines(position, scaled[0], scaled[4], color=colors[kind], linewidth=.9)
            axis.hlines(position, scaled[1], scaled[3], color=colors[kind], linewidth=4)
            axis.plot(scaled[2], position, "o", color=colors[kind], markersize=3.7,
                      markeredgecolor="white", markeredgewidth=.55)
            distributions.append({
                "family": family,
                "kind": ("directional_offset", "native_resampling")[kind],
                "quantiles_rad": quantiles.tolist(),
                "comparisons": 128 if kind == 0 else 8,
                "saved_observations": 8,
                "scope": "Correlated decoder responses; not physical safety trials",
            })
    axis.set(yticks=[0, 2], yticklabels=["Jar", "Clutter"], ylim=(-.65, 2.65),
             xlim=(0, 27), xlabel="Executed-prefix arm-target RMS (milliradians)")
    axis.invert_yaxis()
    axis.grid(axis="x", color=".88", linewidth=.5)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.legend(
        [Line2D([0], [0], color=c, marker="o", markersize=3) for c in colors],
        ["Directional offset", "Native resampling"],
        loc="upper right", fontsize=7, frameon=False,
    )
    pdf = args.output / "decoder-authority.pdf"
    png = args.output / "decoder-authority.png"
    fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(png, dpi=220)
    plt.close(fig)
    receipt = {
        "schema": "maniguard-latent-paper-figure-v1",
        "source_summary": str(args.summary.relative_to(ROOT)),
        "source_summary_sha256": sha256(args.summary),
        "script_sha256": sha256(Path(__file__)),
        "versions": {"matplotlib": matplotlib.__version__, "numpy": np.__version__},
        "distributions": distributions,
        "files": {p.name: sha256(p) for p in (pdf, png)},
        "new_physical_controls": 0,
        "new_model_queries": 0,
        "new_encoder_calls": 0,
    }
    (args.output / "provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "files": receipt["files"]}))


if __name__ == "__main__":
    main()
