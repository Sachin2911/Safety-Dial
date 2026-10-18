"""Post hoc matched readout and calibration diagnosis using frozen artifacts only."""

import json
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from followup_evidence import FIGURES, ROOT, RUN, atomic, savefig, sha

sys.path.insert(0, str(ROOT / "experiments"))
from helpers.maniguard_part3_data import load_group
from helpers.maniguard_part3_model import linear, state_for

MODEL = ROOT / "data/maniguard/models/phase3-parts2-3-20261018"


def main():
    began = time.monotonic()
    audit = json.loads((MODEL / "forecast-audit.json").read_text())
    calibration = json.loads((MODEL / "calibration.json").read_text())
    training = json.loads((MODEL / "training-ledger.json").read_text())
    with np.load(MODEL / "artifacts.npz") as f:
        artifacts = {k: f[k].copy() for k in f.files}
    with np.load(MODEL / "forecast-windows.npz") as f:
        forecast = {k: f[k].copy() for k in f.files}
    with np.load(MODEL / "calibration-windows.npz") as f:
        cal = {k: f[k].copy() for k in f.files}
    episodes = {e["episode_id"]: e for e in load_group(RUN / "part3/features", "audit")}
    indices = [
        (str(e), int(t) + 4) for e, t in zip(forecast["episode"], forecast["step"], strict=True)
    ]
    measured = np.asarray([episodes[e]["target"][t] for e, t in indices])
    assert np.array_equal(measured, forecast["actual_target"])
    readouts = {}
    for name in ("pca", "raw", "random", "pca_visual_only"):
        representation = "pca" if name == "pca_visual_only" else name
        states = {e: state_for(value, artifacts, representation) for e, value in episodes.items()}
        actual = np.asarray([states[e][t] for e, t in indices])
        if name == "pca_visual_only":
            actual = actual[:, :32]
            weight = artifacts["readout_visual_only"]
        else:
            weight = artifacts["readout_" + name]
        predicted = linear(actual, weight)
        readouts[name] = {
            "windows": len(measured),
            "margin_mae_deg": float(np.abs(predicted[:, 0] - measured[:, 0]).mean()),
            "progress_mae_m": float(np.abs(predicted[:, 1] - measured[:, 1]).mean()),
        }
    assert np.isclose(readouts["pca"]["margin_mae_deg"], audit["metrics"]["actual_margin_mae_deg"])
    bins = np.asarray(calibration["bins"])
    cal_bins = np.clip(np.digitize(cal["correction"], bins) - 1, 0, len(bins) - 2)
    audit_bins = np.clip(np.digitize(forecast["correction"], bins) - 1, 0, len(bins) - 2)
    shared_quantile = len(set(calibration["error_quantiles"])) == 1
    result = {
        "schema": "maniguard-followup-model-diagnostics-v1",
        "at": time.time(),
        "scope": "Post hoc matched readout and trust diagnosis. No fitting, new data or criterion change; primary gate remains frozen.",
        "matched_actual_future_readouts": readouts,
        "calibration_bin_windows": np.bincount(cal_bins, minlength=6).tolist(),
        "audit_bin_windows": np.bincount(audit_bins, minlength=6).tolist(),
        "all_conditional_quantiles_equal": shared_quantile,
        "trust_masks_exactly_equal": bool(
            np.array_equal(forecast["accepted_m"], forecast["accepted_c"])
        ),
        "accepted_M_windows": int(forecast["accepted_m"].sum()),
        "accepted_C_windows": int(forecast["accepted_c"].sum()),
        "trust_collapse_explanation": "With one common error quantile, C is a positive affine transform of M; common support and transformed median thresholds give the same acceptance.",
        "readout_matrix_calls": 4,
        "readout_examples": 4 * len(measured),
        "new_encoder_calls": 0,
        "new_dynamics_member_calls": 0,
        "VLA_queries": 0,
        "new_physical_actions": 0,
        "source_hashes": {
            name: sha(MODEL / name)
            for name in (
                "artifacts.npz",
                "forecast-audit.json",
                "forecast-windows.npz",
                "calibration.json",
                "calibration-windows.npz",
            )
        },
        "script_sha256": sha(__file__),
    }
    provenance = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for member in range(5):
        values = [r for r in training["member_losses"] if r["member"] == member]
        axes[0].plot(
            [r["epoch"] + 1 for r in values], [r["mse"] for r in values], label=str(member)
        )
    axes[0].set(
        xlabel="Epoch",
        ylabel="Residual MSE (all 41 state coordinates)",
        yscale="log",
        title="Fitting bootstrap loss; not held-out accuracy",
    )
    axes[0].legend(title="Member", ncol=3, fontsize=8)
    names = list(readouts)
    axes[1].bar(
        range(4),
        [readouts[n]["margin_mae_deg"] for n in names],
        color=["#597d90", "#83a5b4", "#bda67b", "#9976a4"],
    )
    axes[1].set(
        xticks=range(4),
        xticklabels=["PCA+q", "Raw+q", "RP+q", "PCA only"],
        ylabel="Actual-state margin MAE (deg)",
        title=f"Matched four-action future observations (n={len(measured)})",
    )
    axes[1].axhline(10, color="black", ls="--", label="Primary PCA+q gate limit")
    axes[1].legend(fontsize=8)
    savefig(fig, "part3-training-readouts", provenance)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    x = np.arange(6)
    axes[0].bar(x - 0.18, result["calibration_bin_windows"], width=0.36, label="Calibration")
    axes[0].bar(x + 0.18, result["audit_bin_windows"], width=0.36, label="Audit")
    axes[0].set(
        xticks=x,
        xticklabels=["0-.5", ".5-1", "1-2", "2-4", "4-8", "8+"],
        xlabel="Factual prefix correction bin",
        ylabel="Windows",
        title="Only one calibration bin has data",
    )
    axes[0].legend(fontsize=8)
    axes[1].scatter(cal["correction"], cal["error"], s=7, alpha=0.3, label="Calibration")
    axes[1].scatter(forecast["correction"], forecast["error4"], s=7, alpha=0.3, label="Audit")
    axes[1].axhline(
        calibration["unconditional_error_quantile"],
        color="black",
        ls="--",
        label="Frozen 0.8 error quantile",
    )
    axes[1].set(xlabel="Factual normalized prefix correction", ylabel="Four-action latent RMSE")
    axes[1].legend(fontsize=7)
    axes[2].bar(
        ["M cal", "C cal", "M audit", "C audit"],
        [
            calibration["M_nominal_actual_fraction"],
            calibration["C_nominal_actual_fraction"],
            audit["metrics"]["M_acceptance"],
            audit["metrics"]["C_acceptance"],
        ],
        color=["#7b9bab", "#ad95b4", "#597d90", "#9976a4"],
    )
    axes[2].axhline(0.5, color="black", ls="--")
    axes[2].set(ylim=(0, 1), ylabel="Accepted windows", title="Identical M/C masks on audit")
    savefig(fig, "part3-calibration", provenance)
    result["seconds"] = time.monotonic() - began
    atomic(RUN / "part3/model-diagnostics.json", result)
    atomic(
        ROOT / "docs/research/research_notes/ManiGuard phase3 Part3 model diagnostics.json", result
    )
    atomic(
        FIGURES / "model-diagnostics-provenance.json",
        {
            "files": provenance,
            "diagnostics_sha256": sha(RUN / "part3/model-diagnostics.json"),
            "script_sha256": sha(__file__),
        },
    )
    print(json.dumps({k: v for k, v in result.items() if k not in ("source_hashes",)}))


if __name__ == "__main__":
    main()
