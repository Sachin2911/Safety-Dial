#!/usr/bin/env python3
"""Post-hoc diagnostic of the full-block Gate 0, with no new simulator interactions.

Reuses saved outcomes and BC features. Refit the two ridge maps on FIT episodes only
at their previously chosen alphas to compare validation error on the SAME 60-action
target, and split that error into block-mean and within-block components. No model
selection or Gate 0 threshold changes occur here. Write to a new output path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.threads import pin_threads

pin_threads()
import h5py
import hdf5plugin  # noqa: F401
import numpy as np

from evo_block_bc_init import ridge
from helpers.evoStats import clustered_mean_ci
from helpers.runManifest import file_sha256


def errors(pred, truth):
    pred, truth = pred.reshape(-1, 10, 6), truth.reshape(-1, 10, 6)
    pm, ym = pred.mean(1, keepdims=True), truth.mean(1, keepdims=True)
    return {
        "full_block_mse": float(np.mean((pred - truth) ** 2)),
        "block_mean_mse": float(np.mean((pm - ym) ** 2)),
        "within_block_mse": float(np.mean(((pred - pm) - (truth - ym)) ** 2)),
        "predicted_within_block_rms": float(np.sqrt(np.mean((pred - pm) ** 2))),
        "target_within_block_rms": float(np.sqrt(np.mean((truth - ym) ** 2))),
        "mse_per_offset": np.mean((pred - truth) ** 2, axis=(0, 2)).tolist(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--previous", type=Path, required=True)
    ap.add_argument("--roots-h5", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    report = json.loads((a.run / "gate0.json").read_text())
    old = json.loads((a.previous / "gate0.json").read_text())
    rows = json.loads((a.run / "evaluation_rows.json").read_text())
    old_rows = json.loads((a.previous / "evaluation_rows.json").read_text())
    assert report["evaluation_roots_digest"] == old["evaluation_roots_digest"]
    assert [r["root_id"] for r in rows] == [r["root_id"] for r in old_rows]
    assert report["bc"]["episodes"] == old["bc"]["episodes"]
    assert report["bc"]["validation_episodes"] == old["bc"]["validation_episodes"]
    clusters = np.array([r["episode"] for r in rows])
    unsafe = np.array([r["bc_dense_violated"] for r in rows])
    old_unsafe = np.array([r["bc_dense_violated"] for r in old_rows])
    progress = np.array([r["bc_progress"] for r in rows])
    old_progress = np.array([r["bc_progress"] for r in old_rows])
    assert int(unsafe.sum()) == report["gate0"]["violations_bc"]
    with np.load(a.run / "bc_features.npz") as f:
        X, Y, E = f["X"], f["Y"], f["episode"]
    with np.load(a.run / "bc_policy.npz") as f:
        X = (X - f["feature_mean"]) / f["feature_std"]
    val = np.isin(E, report["bc"]["validation_episodes"])
    Ymean = Y.reshape(-1, 10, 6).mean(1)
    # Use the recorded pre-cast float64 standardisation to reproduce the selection MSE.
    with np.load(a.run / "bc_features.npz") as f:
        raw = f["X"]
    X = (raw - raw.mean(0)) / (raw.std(0) + 1e-6)
    W, b = ridge(X[~val], Y[~val], report["bc"]["alpha"])
    pred = np.clip(X[val] @ W.T + b, -1, 1)
    Wh, bh = ridge(X[~val], Ymean[~val], old["bc"]["alpha"])
    held = np.repeat(np.clip(X[val] @ Wh.T + bh, -1, 1)[:, None, :], 10, axis=1)
    validation = {"sequential_block": errors(pred, Y[val]),
                  "held_mean": errors(held, Y[val]), "n_samples": int(val.sum())}
    chosen_mse = next(x["val_mse"] for x in report["bc"]["alpha_sweep"] if x["alpha"] == report["bc"]["alpha"])
    assert np.isclose(validation["sequential_block"]["full_block_mse"], chosen_mse, atol=1e-12)
    with h5py.File(a.roots_h5, "r") as f:
        ids = {v: k for k, v in json.loads(f.attrs["policy_ids"]).items()}
        offset = f["ep_offset"][:]
        names = [ids[int(f["policy_id"][offset[r["episode"]]])] for r in rows]
    by_family = {}
    for name in sorted({n.split("-")[0] for n in names}):
        mask = np.array([n.split("-")[0] == name for n in names])
        by_family[name] = {"n": int(mask.sum()), "violations": int(unsafe[mask].sum()),
                           "old_violations": int(old_unsafe[mask].sum()),
                           "progress_mean": float(progress[mask].mean())}
    # This is descriptive post-hoc analysis on reused evaluation states, not a new gate.
    out = {
        "interpretation": "post-hoc diagnosis; no new policies evaluated, no threshold changes",
        "inputs_sha256": {str(p): file_sha256(p) for p in [a.run / "gate0.json", a.run / "evaluation_rows.json",
                             a.previous / "gate0.json", a.previous / "evaluation_rows.json", a.run / "bc_features.npz",
                             a.roots_h5, Path(__file__)]},
        "costs": {"new_real_steps": 0, "new_imagined_rows": 0, "new_renders": 0},
        "paired": {"new_violation_rate_ci_clustered": clustered_mean_ci(unsafe.astype(float), clusters),
                   "violation_rate_difference_new_minus_old": clustered_mean_ci(unsafe.astype(float)-old_unsafe, clusters),
                   "progress_difference_new_minus_old": clustered_mean_ci(progress-old_progress, clusters),
                   "both_unsafe": int((unsafe & old_unsafe).sum()),
                   "newly_safe": int((~unsafe & old_unsafe).sum()),
                   "newly_unsafe": int((unsafe & ~old_unsafe).sum()),
                   "both_safe": int((~unsafe & ~old_unsafe).sum())},
        "first_unsafe_step_index_median_among_unsafe": float(np.median([r["bc_first_unsafe_step"] for r in rows if r["bc_dense_violated"]])),
        "by_source_policy_family": by_family,
        "validation_same_action_target": validation,
    }
    a.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "inputs_sha256"}, indent=2))


if __name__ == "__main__":
    main()
