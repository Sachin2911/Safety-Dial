#!/usr/bin/env python3
"""E1 step 1: train and freeze the block-pose and pusher probes on expert frames.

Frames come from whole expert episodes in the `probe` split (assets splits.json), read
contiguously; validation is by episode. Linear and small-MLP probes are fitted for
both targets; the choice of capacity is made on validation trajectories here and then
re-checked on development-bank frames in the decomposition script before freezing.

Outputs runs/<run_id>/ with the four probes, stats and a manifest; uploads to
<ns>/safetydial-pusht:probes/<run_id>.

    uv run python experiments/scripts/pusht_e1_probes.py --n-frames 20000
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

apply_torch()

from helpers.hfStore import HFStore  # noqa: E402
from helpers.poseProbes import (  # noqa: E402
    ProbeSpec,
    collect_expert_latents,
    fit_probe,
    pose_to_target,
    pusher_to_target,
    save_probe,
    split_by_trajectory,
)
from helpers.pushtAssets import H5_PATH, load_model  # noqa: E402
from helpers.runManifest import build_manifest, make_run_id, write_manifest  # noqa: E402

ASSETS_RUN = REPO_ROOT / "runs" / "pusht-assets-20260926-1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frames", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--n", type=int, default=1)
    args = ap.parse_args()
    t0 = time.time()
    run_id = make_run_id("pusht", "probes", n=args.n)
    run_dir = REPO_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    model = load_model(device)
    splits = json.loads((ASSETS_RUN / "splits.json").read_text())
    assets_rev = (ASSETS_RUN / "hf_revision.txt").read_text().split()[2]
    rng = np.random.default_rng(args.seed)

    Z, S, E = collect_expert_latents(model, H5_PATH, splits["roles"]["probe"], args.n_frames, rng, device=device)
    print(f"[probes] {len(Z):,} frames from {len(np.unique(E))} episodes in {time.time() - t0:.0f}s")
    tr, va = split_by_trajectory(E, rng, 0.2)
    targets = {"block_pose": pose_to_target(S), "pusher": pusher_to_target(S)}
    results = {}
    for target, Y in targets.items():
        for kind in ("linear", "mlp"):
            spec = ProbeSpec(target=target, kind=kind, epochs=args.epochs, seed=args.seed)
            probe, stats = fit_probe(Z[tr], Y[tr], Z[va], Y[va], spec, device=device, verbose=False)
            stats["n_train_episodes"] = int(len(np.unique(E[tr])))
            stats["n_val_episodes"] = int(len(np.unique(E[va])))
            save_probe(probe, stats, run_dir / f"{target}_{kind}.pt")
            v = stats["val"]
            line = f"[probes] {target:10s} {kind:6s} val r2 {v['r2']:.4f} rmse {v['rmse']:.3f}"
            if "centre_p95_px" in v:
                line += f" centre p50 {v['centre_median_px']:.1f} p95 {v['centre_p95_px']:.1f}px"
            if "angle_p95_deg" in v:
                line += f" angle p50 {v['angle_median_deg']:.1f} p95 {v['angle_p95_deg']:.1f}deg"
            print(line)
            results[f"{target}_{kind}"] = {k: v[k] for k in v if k != "r2_per_dim"} | {"r2_per_dim": v["r2_per_dim"]}
    manifest = build_manifest(
        run_id=run_id, kind="probes", seeds={"rng": args.seed},
        data={"assets_run": ASSETS_RUN.name, "assets_revision": assets_rev, "split": "probe", "n_frames": int(len(Z)),
              "n_episodes": int(len(np.unique(E))), "val_frac_by_episode": 0.2},
        metrics=results, started_at=t0,
    )
    write_manifest(run_dir, manifest)
    (run_dir / "README.md").write_text(f"# {run_id}\n\nPush-T readout probes (block pose x, y, sin, cos; pusher x, y) from frozen 192-d LeWM latents, trained on whole expert episodes of the probe split with an episode-level validation split. Linear and MLP variants. Not safety supervision. See manifest.json.\n")
    if not args.no_upload:
        HFStore().upload_run("pusht", "probes", run_dir, run_id=run_id)
    print(f"[probes] done in {time.time() - t0:.0f}s -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
