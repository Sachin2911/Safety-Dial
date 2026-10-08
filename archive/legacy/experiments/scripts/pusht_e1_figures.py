#!/usr/bin/env python3
"""Render two presentation corrections from an immutable E1 JSON report only.

No bank, model, environment, simulator, report rewrite or upload is needed. Missing
FSA values remain undefined. Saved counts support lower bounds; exact censoring
ranges are displayed only where the report retained the necessary counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path

SOURCES = {
    "endpoint": ("True endpoint reference", "#666666", "x", ":"),
    "real_readout": ("Real-image readout", "#0072B2", "o", "-"),
    "imagined": ("Imagined model", "#D55E00", "s", "--"),
    "stationary": ("Stationary reference", "#009E73", "^", "-."),
    "coord_mlp": ("Coordinate MLP", "#CC79A7", "D", (0, (3, 1, 1, 1))),
}
POSE_SOURCES = ("endpoint", "real_readout", "imagined", "coord_mlp")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _count(record, key):
    value = record[key]
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{key} must be a nonnegative integer")
    return int(value)


def _finite(value):
    return value is not None and math.isfinite(float(value))


def dial_value(record):
    """Validate a saved decision row without replacing undefined FSA by a bound.

    A lower bound follows from observed unsafe accepted / all accepted. An upper
    bound requires the saved accepted-unresolved count; global censor counts are
    insufficient. Returned JSON uses null for undefined values.
    """
    accepted, false_safe = _count(record, "n_accepted"), _count(record, "n_false_safe")
    if false_safe > accepted:
        raise ValueError("Observed false-safe count exceeds accepted count")
    ar = float(record["acceptance_rate"])
    if not math.isfinite(ar) or not 0 <= ar <= 1:
        raise ValueError("Acceptance rate must be a finite probability")
    if (accepted == 0) != (ar == 0):
        raise ValueError("Acceptance rate and accepted count disagree")
    point = float(record["fsa"]) if _finite(record.get("fsa")) else None
    unresolved = _count(record, "n_accepted_censored") if "n_accepted_censored" in record else None
    if unresolved is not None and unresolved > accepted - false_safe:
        raise ValueError("Accepted unresolved and known unsafe counts overlap or exceed acceptance")
    out = {
        "margin": float(record["m"]),
        "acceptance_rate": ar,
        "n_accepted": accepted,
        "n_false_safe": false_safe,
        "n_accepted_censored": unresolved,
        "point": point,
        "lower": None,
        "upper": None,
        "status": "zero_acceptance",
    }
    if not accepted:
        if point is not None or any(_finite(record.get(k)) for k in ("fsa_lower", "fsa_upper")):
            raise ValueError("Zero acceptance cannot have a defined false-safe estimate or bound")
        return out
    lower = false_safe / accepted
    upper = (false_safe + unresolved) / accepted if unresolved is not None else None
    if point is not None:
        if not math.isclose(point, lower, abs_tol=1e-12, rel_tol=1e-10) or unresolved:
            raise ValueError("Defined FSA disagrees with saved counts or unresolved acceptance")
        upper = lower
    for key, expected in (("fsa_lower", lower), ("fsa_upper", upper)):
        if key in record and (
            not _finite(record[key])
            or expected is None
            or not math.isclose(float(record[key]), expected, abs_tol=1e-12, rel_tol=1e-10)
        ):
            raise ValueError(f"Saved {key} disagrees with accepted outcome counts")
    out.update(
        lower=lower,
        upper=upper,
        status="defined"
        if point is not None
        else "undefined_censored"
        if unresolved
        else "undefined_lower_bound_only",
    )
    return out


def prepare_dial(report):
    banks = {}
    for name, bank in report["banks"].items():
        sources = {}
        for source in SOURCES:
            rows = [dial_value(row) for row in bank["dial_curves"][source]]
            anchor = dial_value(bank["at_m0"][source])
            if anchor["margin"] != 0:
                raise ValueError("The saved m=0 anchor has a different margin")
            zero_rows = [row for row in rows if row["margin"] == 0]
            for row in zero_rows:
                if any(
                    row[key] != anchor[key]
                    for key in ("acceptance_rate", "n_accepted", "n_false_safe", "point", "lower")
                ):
                    raise ValueError("Saved m=0 anchor differs from its dial row")
            sources[source] = {"curve": rows, "m0": anchor}
        banks[name] = sources
    return banks


def plot_dials(data, path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    fig, axes = plt.subplots(1, len(data), figsize=(15.6, 6.6), squeeze=False, sharey=True)
    for ax, (bank, sources) in zip(axes[0], data.items()):
        defined_count = zero_count = 0
        for source, values in sources.items():
            label, colour, marker, _ = SOURCES[source]
            rows = sorted(values["curve"], key=lambda row: row["acceptance_rate"])
            x = [row["acceptance_rate"] for row in rows]
            defined = [row["point"] if row["point"] is not None else np.nan for row in rows]
            lower = [
                row["lower"] if row["point"] is None and row["lower"] is not None else np.nan
                for row in rows
            ]
            defined_count += sum(row["point"] is not None for row in rows)
            zero_count += sum(row["status"] == "zero_acceptance" for row in rows)
            ax.plot(
                x, defined, color=colour, marker=marker, markersize=3.5, linewidth=1.6, label=label
            )
            ax.plot(x, lower, color=colour, linestyle="--", linewidth=1.35, alpha=0.85)
            anchor = values["m0"]
            if anchor["lower"] is not None and anchor["upper"] is not None:
                ax.errorbar(
                    anchor["acceptance_rate"],
                    anchor["lower"],
                    yerr=[[0.0], [anchor["upper"] - anchor["lower"]]],
                    color=colour,
                    marker=marker,
                    markerfacecolor="white",
                    markersize=7,
                    capsize=5,
                    elinewidth=2,
                    linestyle="none",
                    zorder=5,
                )
        ax.set_title(f"{bank.capitalize()} bank")
        ax.set_xlabel("Acceptance rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.025, 1)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(alpha=0.22)
        if defined_count == 0:
            ax.text(
                0.04,
                0.95,
                "FSA undefined at all saved dial points\nAccepted unresolved futures; lower bounds shown",
                transform=ax.transAxes,
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.95},
            )
        else:
            ax.text(
                0.04,
                0.95,
                "Solid: defined FSA\nDashed: observed lower bound where FSA is undefined",
                transform=ax.transAxes,
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.95},
            )
        if zero_count:
            ax.text(
                0.04,
                0.76,
                f"{zero_count} zero-acceptance entries omitted (FSA undefined)",
                transform=ax.transAxes,
                fontsize=8,
                va="top",
            )
    axes[0][0].set_ylabel("False-safe acceptance / observed lower bound")
    fig.suptitle("Dial results with unresolved outcomes retained", fontsize=15, y=0.98)
    handles = [
        Line2D([0], [0], color=v[1], marker=v[2], label=v[0], linewidth=1.6)
        for v in SOURCES.values()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.145),
        ncol=5,
        frameon=False,
        fontsize=9,
    )
    fig.text(
        0.5,
        0.095,
        "Open markers with vertical caps: saved m=0 lower/upper censoring bounds (not confidence intervals).",
        ha="center",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.052,
        "Dashed values count observed unsafe accepted / all accepted. They are not point FSA estimates; per-margin upper bounds were not saved.",
        ha="center",
        fontsize=9,
    )
    fig.text(
        0.5,
        0.016,
        "Zero accepted plans remain undefined and are never plotted as safe zero. No scientific gate or underlying result was changed.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.055, right=0.99, top=0.87, bottom=0.28, wspace=0.13)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pose(report, path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    banks = list(report["banks"])
    fig, axes = plt.subplots(
        2, len(banks), figsize=(15.6, 8), squeeze=False, sharex=True, sharey="row"
    )
    for column, bank in enumerate(banks):
        for source in POSE_SOURCES:
            label, colour, marker, style = SOURCES[source]
            saved = report["pose_error_by_block"][bank][source]
            for row, key in enumerate(("centre_by_block", "angle_by_block")):
                values = np.asarray(saved[key], float)
                if values.ndim != 1 or not len(values) or np.isinf(values).any():
                    raise ValueError("Invalid saved pose-error series")
                axes[row, column].plot(
                    np.arange(len(values)),
                    values,
                    color=colour,
                    marker=marker,
                    linestyle=style,
                    markersize=5,
                    linewidth=1.8,
                    label=label,
                )
        axes[0, column].set_title(f"{bank.capitalize()} bank")
        for ax in axes[:, column]:
            ax.grid(alpha=0.22)
            ax.set_xticks(np.arange(6))
            ax.set_xlabel("Horizon block (0 = current frame)")
    axes[0, 0].set_ylabel("Mean block centre error (px)")
    axes[1, 0].set_ylabel("Mean block angle error (degrees)")
    handles = [
        Line2D(
            [0],
            [0],
            color=SOURCES[s][1],
            marker=SOURCES[s][2],
            linestyle=SOURCES[s][3],
            label=SOURCES[s][0],
        )
        for s in POSE_SOURCES
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=4,
        frameon=False,
        fontsize=11,
    )
    fig.suptitle("Pose error by horizon, with banks in separate panels", fontsize=16, y=0.985)
    fig.text(
        0.5,
        0.016,
        "Saved report means; one consistent colour, marker and line style per source across all banks.",
        ha="center",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.065, right=0.985, top=0.92, bottom=0.17, hspace=0.3, wspace=0.12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render(report_path, output):
    report_path, output = Path(report_path).resolve(), Path(output).resolve()
    raw = report_path.read_bytes()
    report_hash = hashlib.sha256(raw).hexdigest()
    report = json.loads(raw)
    data = prepare_dial(report)
    if output.exists():
        raise FileExistsError(f"A fresh figure output directory is required: {output}")
    if output == report_path.parent or report_path.parent in output.parents:
        raise ValueError("Corrected figures must be outside the completed decomposition directory")
    import matplotlib

    matplotlib.use("Agg")
    output.mkdir(parents=True)
    plot_dials(data, output / "dial_curves_by_source.png")
    plot_pose(report, output / "pose_error_by_horizon.png")
    if report_path.read_bytes() != raw:
        raise RuntimeError("Source report changed during rendering")
    summary = {
        bank: {
            source: {
                "defined_dial_points": sum(r["point"] is not None for r in values["curve"]),
                "undefined_dial_points": sum(r["point"] is None for r in values["curve"]),
                "zero_acceptance_points": sum(
                    r["status"] == "zero_acceptance" for r in values["curve"]
                ),
                "m0": values["m0"],
            }
            for source, values in sources.items()
        }
        for bank, sources in data.items()
    }
    manifest = {
        "schema_version": 1,
        "kind": "immutable-report-presentation-correction",
        "source_report": {
            "path": str(report_path),
            "sha256": report_hash,
            "run_id": report.get("run_id"),
        },
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__)},
        "figures": {p.name: sha256(p) for p in sorted(output.glob("*.png"))},
        "rationale": {
            "dial": "Original test/stress panels were blank because all saved point FSA values are undefined. Show observed lower bounds with explicit labels and only saved m=0 censoring intervals; do not manufacture full upper curves or safe zero values.",
            "pose": "Separate banks into columns and use one consistent source colour/marker/line style to eliminate repeated-style ambiguity.",
        },
        "numeric_policy": "No re-evaluation. Lower=n_false_safe/n_accepted for positive acceptance; upper requires saved n_accepted_censored. Undefined point FSA stays null. m=0 bounds are identification bounds, not bootstrap confidence intervals.",
        "saved_scientific_gate_unchanged": report.get("gate"),
        "dial_summary": summary,
        "new_model_calls": 0,
        "new_simulator_steps": 0,
        "uploads": 0,
        "source_report_unchanged": True,
    }
    (output / "figures_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    (output / "README.md").write_text(
        "# E1 presentation corrections\n\nThese two figures read only the completed decomposition JSON identified in `figures_manifest.json`. The original report, figures and scientific gate remain unchanged.\n\nThe dial plot distinguishes defined FSA from observed lower bounds when accepted futures remain unresolved. Open markers and capped vertical ranges show the exact saved m=0 censoring bounds; these are not confidence intervals. The report omitted accepted-censored counts from the full dial grid, so upper-bound curves cannot be reconstructed from this JSON. Zero acceptance is undefined, never safe zero.\n\nThe pose plot retains the saved means and separates banks into columns, with a consistent colour, marker and line style per source.\n"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = render(args.report, args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "source_report_sha256": manifest["source_report"]["sha256"],
                "figures": manifest["figures"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
