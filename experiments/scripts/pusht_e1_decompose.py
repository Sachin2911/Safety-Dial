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

    uv run python experiments/scripts/pusht_e1_decompose.py --banks-dir data/study/pusht/e1-clean-1 --results-dir runs/e1-clean-1-results
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
from helpers.hfStore import HFStore  # noqa: E402
from helpers.imagination import Imaginer  # noqa: E402
from helpers.poseProbes import load_probe  # noqa: E402
from helpers.pushtAssets import H5_PATH, load_model, load_scalers  # noqa: E402
from helpers.pushtLayouts import load_layouts  # noqa: E402
from helpers.pushtReplay import StepLedger  # noqa: E402
from helpers.runManifest import validate_run_id, build_manifest, make_run_id, write_manifest, file_sha256  # noqa: E402

ASSETS_RUN = REPO_ROOT / "runs" / "pusht-assets-20260926-1"
PROBES_RUN = REPO_ROOT / "runs" / "pusht-probes-20260926-1"
STUDY = REPO_ROOT / "data" / "study" / "pusht"
RESULTS = REPO_ROOT / "docs" / "mainPlan" / "results" / "e1"
from helpers.splitIntegrity import validate_bank_splits, bank_identity, decomposition_gate  # noqa: E402
from helpers.decomposition import (SOURCES, HORIZON_REPORT_PROTOCOL, analyse_bank, ang_err_deg,
    by_regime, cumulative_horizon_report, decision_table, row_outcomes)  # noqa: E402

MARGINS = np.linspace(-20, 60, 41)


def main() -> int:
    global RESULTS, STUDY, ASSETS_RUN, PROBES_RUN
    ap = argparse.ArgumentParser()
    ap.add_argument("--banks", nargs="+", default=["dev", "test", "stress"])
    ap.add_argument("--probe", default="block_pose_mlp")
    ap.add_argument("--coord-epochs", type=int, default=30)
    ap.add_argument("--banks-dir", type=Path, default=STUDY)
    ap.add_argument("--results-dir", type=Path, required=True, help="Fresh decomposition output directory")
    ap.add_argument("--assets-run", type=Path, default=ASSETS_RUN)
    ap.add_argument("--probes-run", type=Path, default=PROBES_RUN)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--run-id", type=validate_run_id, help="Explicit stable run ID for queued workflows")
    ap.add_argument("--min-false-safe", type=int, default=5)
    ap.add_argument("--min-imagination", type=int, default=3)
    ap.add_argument("--min-imagination-share", type=float, default=0.25)
    args = ap.parse_args()
    RESULTS, STUDY = args.results_dir.resolve(), args.banks_dir.resolve()
    ASSETS_RUN, PROBES_RUN = args.assets_run, args.probes_run
    if "dev" not in args.banks:
        ap.error("The development bank is required for margin and readout choices")
    split_report = validate_bank_splits({name: STUDY / name for name in args.banks})
    run_id = args.run_id or make_run_id("pusht", "decomposition", n=args.n)
    coord_dir = REPO_ROOT / "runs" / f"{run_id}-coord"
    for path in (RESULTS, coord_dir):
        if path.exists():
            ap.error(f"Use a fresh output path/run ID; preserving {path}")
    store = None if args.no_upload else HFStore()
    references = {} if store is None else {
        "assets": store.reference_run("pusht", "assets", ASSETS_RUN),
        "probes": store.reference_run("pusht", "probes", PROBES_RUN),
        "banks": {name: store.reference_run("pusht-banks", "banks", STUDY / name) for name in args.banks},
    }
    t_start = time.time()
    device = "cuda"
    RESULTS.mkdir(parents=True)
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
    coord_dir.mkdir(parents=True)
    torch.save(coord.state_dict(), coord_dir / "weights.pt")
    (coord_dir / "config.yaml").write_text(json.dumps({"epochs": args.coord_epochs, "privileged_state": True}) + "\n")
    write_manifest(coord_dir, build_manifest(run_id=coord_dir.name, kind="coordinate-reference",
        data={"upstream": references, "training_episodes": splits["roles"]["replay"][:600],
              "validation_episodes": splits["roles"]["retention"][:100]}, metrics=coord_stats,
        started_at=t_start))
    (coord_dir / "README.md").write_text("# Coordinate dynamics reference\n\nPrivileged simulator-coordinate baseline for Push-T; not a visual latent model.\n")
    if store is not None:
        store.upload_run("pusht", "references", coord_dir, run_id=coord_dir.name)
        references["coordinate_reference"] = store.reference_run("pusht", "references", coord_dir)

    # --- probe capacity check on development-bank frames (chosen on dev, then frozen) ----
    dev = Bank(STUDY / "dev")
    valid_dev = dev.valid_training_indices()[::3]
    if not len(valid_dev):
        raise ValueError("Development bank has no complete observation-valid probe-check frames")
    fr = np.stack([dev.branch(j, frames=True)["frames"][-1] for j in valid_dev])
    st = np.stack([dev.branch(j)["states"][-1] for j in valid_dev])
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
    all_horizon_rows, all_diagnostic_pose_rows = [], []
    evaluation_ledger = StepLedger()
    for name in args.banks:
        bank = Bank(STUDY / name)
        layouts, _ = load_layouts(STUDY / name / "layouts.json")
        try:
            res = analyse_bank(name, bank, layouts, imaginer, probe, coord, evaluation_ledger=evaluation_ledger)
        finally:
            # Preserve actual attempted replay costs even when evaluation fails.
            (RESULTS / "evaluation_ledger.json").write_text(json.dumps(evaluation_ledger.to_dict(), indent=1) + "\n")
            bank.h5.close()
        all_rows += res["rows"]
        pose_errs[name] = res["pose_err"]
        all_horizon_rows.extend(res["horizon_rows"])
        all_diagnostic_pose_rows.extend(res["diagnostic_pose_rows"])
        print(f"[e1] {name}: {len(res['rows'])} (branch, layout) rows; unsafe {sum(r['cmin_dense'] < 0 for r in res['rows'])}")

    # matched acceptance target: dev AR of the unadapted model (imagined source) at m=0
    dev_rows = [r for r in all_rows if r["bank"] == "dev"]
    target_ar = float(np.mean([r["cmin_imagined"] >= 0 for r in dev_rows])) if dev_rows else 0.5
    m_matched = margin_for_acceptance(np.array([r["cmin_imagined"] for r in dev_rows]), target_ar) if dev_rows else 0.0
    report = {"run_id": run_id, "split_integrity": split_report, "bank_identities": {name: bank_identity(STUDY / name) for name in args.banks}, "probe_sha256": file_sha256(PROBES_RUN / f"{args.probe}.pt"), "probe": args.probe, "probe_val_stats": probe_stats["val"], "probe_capacity_dev_frames": cap, "coord_reference": coord_stats,
              "target_acceptance_rate_dev": target_ar, "margin_matched_dev": float(m_matched), "pose_error_by_block": pose_errs, "banks": {}}
    for name in args.banks:
        rows = [r for r in all_rows if r["bank"] == name]
        observed_rows = [row for row in rows if not row.get("censored", False)]
        entry = {"n_rows": len(rows), "n_branches": len({r["branch"] for r in rows}), "n_roots": len({r["root"] for r in rows}),
                 "frac_observed_composite_unsafe": float(row_outcomes(rows)[0].mean()) if rows else None,
                 "n_censored": int(row_outcomes(rows)[1].sum()),
                 "at_m0": decision_table(rows, 0.0), "at_matched": decision_table(rows, m_matched),
                 "by_regime_m0": by_regime(rows, 0.0),
                 "dial_curves": {s: [{"m": float(m), **{k: v for k, v in fsa(np.array([r[f"cmin_{s}"] for r in rows]), row_outcomes(rows)[0], m, censored=row_outcomes(rows)[1]).items() if k in ("fsa", "acceptance_rate", "n_accepted", "n_false_safe")}} for m in MARGINS] for s in SOURCES},
                 "auc_dial": {s: auc_dial(np.array([r[f"cmin_{s}"] for r in rows]), row_outcomes(rows)[0], censored=row_outcomes(rows)[1]) for s in SOURCES},
                 "clearance_error": {s: (clearance_error_stats(np.array([r[f"cmin_{s}"] for r in observed_rows]), np.array([r["cmin_dense"] for r in observed_rows])) if observed_rows else {"n": 0}) for s in SOURCES if s != "dense"},
                 "by_layout": {fam: {"n": sum(r["layout"] == fam for r in rows), "fsa_imagined_m0": decision_table([r for r in rows if r["layout"] == fam], 0.0)["imagined"]["fsa"]} for fam in sorted({r["layout"] for r in rows})},
                 "mechanistic_contact": {}}
        entry["contact_measurement"] = {
            "typed_branch_rows": sum(r.get("contact_mechanism_identifiable", False) for r in rows),
            "untyped_branch_rows": sum(not r.get("contact_mechanism_identifiable", False) for r in rows),
            "legacy_interpretation": "Untyped n_contacts counts any collision, including walls; it cannot identify pusher-T contact or free motion"}
        c_rows = [r for r in rows if r["contact"] and r["layout"] == "familiar" and not r.get("censored", False) and r.get("observation_valid", True)]
        entry["mechanistic_contact"] = {"available": bool(c_rows)}
        if c_rows:
            d_true = np.array([r["displacement_px"] for r in c_rows])
            d_imag = np.array([r["imag_displacement_px"] for r in c_rows])
            r_true = np.array([r["rotation_deg"] for r in c_rows])
            r_imag = np.array([r["imag_rotation_deg"] for r in c_rows])
            entry["mechanistic_contact"] = {"available": True, "n": len(c_rows), "true_disp_mean_px": float(d_true.mean()), "imag_disp_mean_px": float(d_imag.mean()),
                                            "disp_ratio_imag_over_true": float(d_imag.sum() / max(d_true.sum(), 1e-9)),
                                            "true_rot_mean_deg": float(r_true.mean()), "imag_rot_mean_deg": float(r_imag.mean()),
                                            "rot_ratio_imag_over_true": float(r_imag.sum() / max(r_true.sum(), 1e-9)),
                                            "frac_under_predicted_disp": float((d_imag < d_true).mean())}
        entry["cumulative_horizon"] = cumulative_horizon_report(
            [row for row in all_horizon_rows if row["bank"] == name],
            [row for row in all_diagnostic_pose_rows if row["bank"] == name], matched_margin=m_matched)
        report["banks"][name] = entry
        t = entry["at_m0"]
        print(f"[e1] {name} @m=0: " + " | ".join(f"{s} FSA {t[s]['fsa']:.3f} AR {t[s]['acceptance_rate']:.2f}" for s in SOURCES) + f" | attribution {t['attribution']}")
    report["gate"] = decomposition_gate(report["banks"]["dev"]["at_m0"],
        min_false_safe=args.min_false_safe, min_imagination=args.min_imagination,
        min_imagination_share=args.min_imagination_share)
    np.savez_compressed(RESULTS / "decomposition_rows.npz", rows=json.dumps(all_rows))
    diagnostic_path = RESULTS / "decomposition_horizon_rows.npz"
    np.savez_compressed(diagnostic_path, rows=json.dumps(all_horizon_rows),
        pose_rows=json.dumps(all_diagnostic_pose_rows), protocol=json.dumps(HORIZON_REPORT_PROTOCOL))
    report["supplementary_diagnostics"] = {"file": diagnostic_path.name, "sha256": file_sha256(diagnostic_path),
        "n_horizon_layout_rows": len(all_horizon_rows), "n_physical_branch_horizon_pose_rows": len(all_diagnostic_pose_rows),
        "protocol": HORIZON_REPORT_PROTOCOL}
    report["evaluation_ledger"] = evaluation_ledger.to_dict()
    report["wall_clock_s"] = time.time() - t_start
    report["manifest"] = build_manifest(run_id=run_id, kind="analysis",
                                        data={"upstream": references, "split_integrity": split_report},
                                        costs=evaluation_ledger.to_dict())
    (RESULTS / "decomposition.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    write_manifest(RESULTS, report["manifest"])
    write_manifest(RESULTS, report["manifest"], name="decomposition_manifest.json")
    make_figures(report)
    make_diagnostic_figures(report)
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


def make_diagnostic_figures(report: dict) -> None:
    """Supplementary prefix attribution and regime errors; unavailable values remain gaps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    banks = [name for name, entry in report["banks"].items() if entry.get("cumulative_horizon", {}).get("by_horizon")]
    if not banks:
        return
    labels = {"temporal": "temporal sampling", "readout": "readout", "imagination": "imagination",
              "domain_exit": "domain exit", "censored": "censored prefix", "unavailable_chain": "unavailable source chain"}
    fig, axes = plt.subplots(1, len(banks), figsize=(5 * len(banks), 4), squeeze=False)
    for ax, name in zip(axes[0], banks):
        entries = report["banks"][name]["cumulative_horizon"]["by_horizon"]
        horizons = sorted(map(int, entries))
        bottom = np.zeros(len(horizons))
        for key, label in labels.items():
            counts = np.array([entries[str(k)]["decisions"]["at_m0"].get("attribution", {}).get(key, 0) for k in horizons])
            ax.bar(horizons, counts, bottom=bottom, label=label)
            bottom += counts
        ax.set_title(f"{name}: cumulative attribution at m=0")
        ax.set_xlabel("prefix horizon (action blocks)")
        ax.set_ylabel("observed false-safe imagined decisions")
        ax.set_xticks(horizons)
    axes[0][-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / "attribution_by_cumulative_horizon.png", dpi=130)
    plt.close(fig)

    regimes = {"contact": "pusher contact", "free": "free motion", "rotation>=10deg": "rotation >= 10 degrees",
               "near_boundary(|c|<20)": "near boundary"}
    fig, axes = plt.subplots(len(banks), 3, figsize=(15, 4 * len(banks)), squeeze=False)
    for row_index, name in enumerate(banks):
        entries = report["banks"][name]["cumulative_horizon"]["by_horizon"]
        horizons = sorted(map(int, entries))
        for regime, label in regimes.items():
            values = [entries[str(k)]["by_regime"].get(regime, {}) for k in horizons]
            for column, metric in enumerate(("centre_px", "angle_deg", "clearance")):
                series = []
                for value in values:
                    statistic = (value.get("clearance_error", {}).get("imagined", {}) if metric == "clearance" else
                                 value.get("pose_error", {}).get("imagined", {}).get(metric, {}))
                    mean = statistic.get("mean")
                    series.append(np.nan if mean is None else mean)
                axes[row_index, column].plot(horizons, series, marker="o", label=label)
        for column, title in enumerate(("centre error (px)", "periodic angle error (degrees)", "signed clearance error (px)")):
            ax = axes[row_index, column]
            ax.set_title(f"{name}: imagined {title}")
            ax.set_xlabel("prefix horizon (action blocks)")
            ax.set_xticks(horizons)
            ax.grid(alpha=.3)
        axes[row_index, 2].axhline(0, color="black", linewidth=.7)
    axes[0, -1].legend(fontsize=7)
    fig.suptitle("Regime errors by horizon; unsupported targets omitted", y=1.01)
    fig.tight_layout()
    fig.savefig(RESULTS / "regime_error_by_horizon.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
