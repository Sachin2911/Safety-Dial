#!/usr/bin/env python3
"""E1: the four-source decomposition on identical tapes (protocol.md).

Sources of the rule decision for every branch and every hazard layout of its root:
  1 dense truth        true state every env step
  2 endpoint truth     true block-endpoint states, interpolated between them
  3 real readout       frozen probe on encoded REAL endpoint frames, interpolated
  4 imagined readout   the same probe on the model's imagined latents, interpolated
References: stationary block; privileged coordinate-dynamics MLP (coordDynamics.py).

A false-safe decision from source 4 is attributed to the first link in the chain that
already produces it: temporal (2 vs 1), readout (3 vs 2) or imagination (4 vs 3).

Outputs docs/mainPlan/results/e1/: decomposition.json, per-branch table (npz), figures.

    uv run python experiments/scripts/pusht_e1_decompose.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from helpers.threads import apply_torch, pin_threads  # noqa: E402

pin_threads()

import numpy as np  # noqa: E402
import torch  # noqa: E402

apply_torch()

from helpers.branchBank import Bank  # noqa: E402
from helpers.coordDynamics import expert_block_transitions, fit_coord_dynamics  # noqa: E402
from helpers.dialMetrics import auc_dial, clearance_error_stats, fsa, margin_for_acceptance  # noqa: E402
from helpers.imagination import Imaginer  # noqa: E402
from helpers.poseProbes import load_probe  # noqa: E402
from helpers.pushtAssets import H5_PATH, load_model, load_scalers  # noqa: E402
from helpers.pushtLayouts import load_layouts  # noqa: E402
from helpers.runManifest import build_manifest, write_manifest  # noqa: E402

ASSETS_RUN = REPO_ROOT / "runs" / "pusht-assets-20260926-1"
PROBES_RUN = REPO_ROOT / "runs" / "pusht-probes-20260926-1"
STUDY = REPO_ROOT / "data" / "study" / "pusht"
RESULTS = REPO_ROOT / "docs" / "mainPlan" / "results" / "e1"
from helpers.decomposition import SOURCES, analyse_bank, ang_err_deg, by_regime, decision_table  # noqa: E402

MARGINS = np.linspace(-20, 60, 41)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--banks", nargs="+", default=["dev", "test", "stress"])
    ap.add_argument("--probe", default="block_pose_mlp")
    ap.add_argument("--coord-epochs", type=int, default=30)
    args = ap.parse_args()
    t_start = time.time()
    device = "cuda"
    RESULTS.mkdir(parents=True, exist_ok=True)
    model = load_model(device)
    process = load_scalers(ASSETS_RUN / "scalers.npz")
    splits = json.loads((ASSETS_RUN / "splits.json").read_text())
    imaginer = Imaginer(model, process, device)
    probe, probe_stats = load_probe(PROBES_RUN / f"{args.probe}.pt", device)

    # --- coordinate reference (privileged) -------------------------------------------
    X, A, Y = expert_block_transitions(H5_PATH, splits["roles"]["replay"][:600])
    Xv, Av, Yv = expert_block_transitions(H5_PATH, splits["roles"]["retention"][:100])
    print(f"[e1] coord reference: {len(X):,} train / {len(Xv):,} val transitions")
    coord, coord_stats = fit_coord_dynamics(X, A, Y, Xv, Av, Yv, device=device, epochs=args.coord_epochs)
    torch.save(coord.state_dict(), RESULTS.parent.parent.parent.parent / "runs" / "pusht-coord-mlp.pt")

    # --- probe capacity check on development-bank frames (chosen on dev, then frozen) ----
    dev = Bank(STUDY / "dev")
    fr = np.stack([dev.branch(j, frames=True)["frames"][-1] for j in range(0, len(dev), 3)])
    st = np.stack([dev.branch(j)["states"][-1] for j in range(0, len(dev), 3)])
    z = imaginer.encode(fr)
    cap = {}
    for kind in ("linear", "mlp"):
        p, _ = load_probe(PROBES_RUN / f"block_pose_{kind}.pt", device)
        pose = p.predict_pose(z)
        cap[kind] = {"centre_p50_px": float(np.median(np.linalg.norm(pose[:, :2] - st[:, 2:4], axis=1))),
                     "centre_p95_px": float(np.percentile(np.linalg.norm(pose[:, :2] - st[:, 2:4], axis=1), 95)),
                     "angle_p50_deg": float(np.median(ang_err_deg(pose[:, 2], st[:, 4]))), "angle_p95_deg": float(np.percentile(ang_err_deg(pose[:, 2], st[:, 4]), 95)), "n": int(len(z))}
    print(f"[e1] probe capacity on dev-bank frames: {json.dumps(cap)}")

    # --- decomposition per bank ------------------------------------------------------
    all_rows, pose_errs = [], {}
    for name in args.banks:
        bank = Bank(STUDY / name)
        layouts, _ = load_layouts(STUDY / name / "layouts.json")
        res = analyse_bank(name, bank, layouts, imaginer, probe, coord)
        all_rows += res["rows"]
        pose_errs[name] = res["pose_err"]
        print(f"[e1] {name}: {len(res['rows'])} (branch, layout) rows; unsafe {sum(r['cmin_dense'] < 0 for r in res['rows'])}")

    # matched acceptance target: dev AR of the unadapted model (imagined source) at m=0
    dev_rows = [r for r in all_rows if r["bank"] == "dev"]
    target_ar = float(np.mean([r["cmin_imagined"] >= 0 for r in dev_rows])) if dev_rows else 0.5
    m_matched = margin_for_acceptance(np.array([r["cmin_imagined"] for r in dev_rows]), target_ar) if dev_rows else 0.0
    report = {"probe": args.probe, "probe_val_stats": probe_stats["val"], "probe_capacity_dev_frames": cap, "coord_reference": coord_stats,
              "target_acceptance_rate_dev": target_ar, "margin_matched_dev": float(m_matched), "pose_error_by_block": pose_errs, "banks": {}}
    for name in args.banks:
        rows = [r for r in all_rows if r["bank"] == name]
        entry = {"n_rows": len(rows), "n_branches": len({r["branch"] for r in rows}), "n_roots": len({r["root"] for r in rows}),
                 "frac_unsafe": float(np.mean([r["cmin_dense"] < 0 for r in rows])) if rows else None,
                 "at_m0": decision_table(rows, 0.0), "at_matched": decision_table(rows, m_matched),
                 "by_regime_m0": by_regime(rows, 0.0),
                 "dial_curves": {s: [{"m": float(m), **{k: v for k, v in fsa(np.array([r[f"cmin_{s}"] for r in rows]), np.array([r["cmin_dense"] < 0 for r in rows]), m).items() if k in ("fsa", "acceptance_rate", "n_accepted", "n_false_safe")}} for m in MARGINS] for s in SOURCES},
                 "auc_dial": {s: auc_dial(np.array([r[f"cmin_{s}"] for r in rows]), np.array([r["cmin_dense"] < 0 for r in rows])) for s in SOURCES},
                 "clearance_error": {s: clearance_error_stats(np.array([r[f"cmin_{s}"] for r in rows]), np.array([r["cmin_dense"] for r in rows])) for s in SOURCES if s != "dense"},
                 "by_layout": {fam: {"n": sum(r["layout"] == fam for r in rows), "fsa_imagined_m0": decision_table([r for r in rows if r["layout"] == fam], 0.0)["imagined"]["fsa"]} for fam in ("familiar", "heldout")},
                 "mechanistic_contact": {}}
        c_rows = [r for r in rows if r["contact"] and r["layout"] == "familiar"]
        if c_rows:
            d_true = np.array([r["displacement_px"] for r in c_rows])
            d_imag = np.array([r["imag_displacement_px"] for r in c_rows])
            r_true = np.array([r["rotation_deg"] for r in c_rows])
            r_imag = np.array([r["imag_rotation_deg"] for r in c_rows])
            entry["mechanistic_contact"] = {"n": len(c_rows), "true_disp_mean_px": float(d_true.mean()), "imag_disp_mean_px": float(d_imag.mean()),
                                            "disp_ratio_imag_over_true": float(d_imag.sum() / max(d_true.sum(), 1e-9)),
                                            "true_rot_mean_deg": float(r_true.mean()), "imag_rot_mean_deg": float(r_imag.mean()),
                                            "rot_ratio_imag_over_true": float(r_imag.sum() / max(r_true.sum(), 1e-9)),
                                            "frac_under_predicted_disp": float((d_imag < d_true).mean())}
        report["banks"][name] = entry
        t = entry["at_m0"]
        print(f"[e1] {name} @m=0: " + " | ".join(f"{s} FSA {t[s]['fsa']:.3f} AR {t[s]['acceptance_rate']:.2f}" for s in SOURCES) + f" | attribution {t['attribution']}")
    np.savez_compressed(RESULTS / "decomposition_rows.npz", rows=json.dumps(all_rows))
    report["wall_clock_s"] = time.time() - t_start
    report["manifest"] = build_manifest(run_id="pusht-e1-decomposition", kind="analysis",
                                        data={"assets_run": ASSETS_RUN.name, "probes_run": PROBES_RUN.name, "banks": json.loads((RESULTS / "banks.json").read_text()).get("hf_revisions", {})})
    (RESULTS / "decomposition.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    write_manifest(RESULTS, report["manifest"], name="decomposition_manifest.json")
    make_figures(report)
    print(f"[e1] done in {time.time() - t_start:.0f}s")
    return 0


def make_figures(report: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    banks = list(report["banks"])
    fig, axes = plt.subplots(1, len(banks), figsize=(5 * len(banks), 4), squeeze=False)
    for ax, name in zip(axes[0], banks):
        for s in SOURCES:
            if s == "dense":
                continue
            cur = report["banks"][name]["dial_curves"][s]
            ax.plot([c["acceptance_rate"] for c in cur], [c["fsa"] for c in cur], marker=".", label=s)
        ax.set_xlabel("acceptance rate")
        ax.set_ylabel("false-safe acceptance")
        ax.set_title(f"{name} bank: dial curves by source")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "dial_curves_by_source.png", dpi=130)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name in banks:
        pe = report["pose_error_by_block"][name]
        for s in ("real_readout", "imagined", "coord_mlp", "endpoint"):
            axes[0].plot(pe[s]["centre_by_block"], marker="o", label=f"{name}/{s}")
            axes[1].plot(pe[s]["angle_by_block"], marker="o", label=f"{name}/{s}")
    axes[0].set_title("block centre error by horizon block (px, mean)")
    axes[1].set_title("block angle error by horizon block (deg, mean)")
    for ax in axes:
        ax.set_xlabel("block (0 = current frame)")
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(RESULTS / "pose_error_by_horizon.png", dpi=130)
    fig, ax = plt.subplots(figsize=(6, 4))
    labels, temporal, readout, imag = [], [], [], []
    for name in banks:
        a = report["banks"][name]["at_m0"]["attribution"]
        labels.append(name)
        temporal.append(a["temporal"])
        readout.append(a["readout"])
        imag.append(a["imagination"])
    x = np.arange(len(labels))
    ax.bar(x, temporal, label="temporal sampling")
    ax.bar(x, readout, bottom=temporal, label="readout")
    ax.bar(x, imag, bottom=np.array(temporal) + np.array(readout), label="imagination")
    ax.set_xticks(x, labels)
    ax.set_ylabel("false-safe decisions of the imagined source at m=0")
    ax.set_title("attribution to the first failing link")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "attribution_m0.png", dpi=130)


if __name__ == "__main__":
    raise SystemExit(main())
