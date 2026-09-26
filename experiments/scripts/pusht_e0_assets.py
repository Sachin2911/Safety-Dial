#!/usr/bin/env python3
"""E0 step 1: rebuild the Push-T assets once and push them to Hugging Face.

Produces `runs/<run_id>/` with:
  scalers.npz        fitted StandardScalers (action, proprio, state), as the pilot fitted them
  state_dict.pt      the released weights after key remapping (reproducible load)
  config.json        the released model config
  splits.json        whole-episode role splits of the expert data (seeded)
  manifest.json      revisions (public model + dataset sha, le-wm commit), data sizes, hashes
  README.md          model card
and uploads it to `<ns>/safetydial-pusht:assets/<run_id>` tagged with the run ID.

    uv run python experiments/scripts/pusht_e0_assets.py [--no-upload] [--n 1]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from helpers.threads import apply_torch, pin_threads  # noqa: E402

pin_threads()

import h5py  # noqa: E402
import hdf5plugin  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402

apply_torch()

from helpers.hfStore import HFStore, public_dataset_revision, public_model_revision  # noqa: E402
from helpers.pushtAssets import (  # noqa: E402
    H5_PATH,
    PUBLIC_MODEL_REPO,
    STABLEWM_HOME,
    build_scalers,
    load_model,
    save_scalers,
    scaler_fingerprint,
)
from helpers.runManifest import (  # noqa: E402
    array_sha256,
    build_manifest,
    file_sha256,
    make_run_id,
    write_manifest,
)

# Whole-episode roles. Sizes are provisional (pushT.md E1/E2); the seed fixes them.
SPLIT_SEED = 20260926
SPLIT_SIZES = {
    "probe": 2000,  # probe training frames (E1)
    "replay": 2000,  # original-data replay mixture during adaptation (E2/E3)
    "retention": 500,  # held-out ordinary clips for prediction-error retention
    "roots": 2000,  # expert episodes eligible as root sources, if they replay
}


def make_splits(n_episodes: int) -> dict:
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(n_episodes)
    out, start = {"seed": SPLIT_SEED, "n_episodes": int(n_episodes), "roles": {}}, 0
    for role, size in SPLIT_SIZES.items():
        out["roles"][role] = sorted(int(i) for i in perm[start : start + size])
        start += size
    out["roles"]["reserve"] = sorted(int(i) for i in perm[start:])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--n", type=int, default=1, help="run id suffix")
    args = ap.parse_args()
    t0 = time.time()

    if not H5_PATH.is_file():
        print(f"missing {H5_PATH}; run scripts/download_data.py first")
        return 1

    run_id = make_run_id("pusht", "assets", n=args.n)
    run_dir = REPO_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[assets] run_id={run_id}")

    # -- revisions ------------------------------------------------------------------
    model_sha = public_model_revision(PUBLIC_MODEL_REPO)
    data_sha = public_dataset_revision(PUBLIC_MODEL_REPO)
    print(f"[assets] public model sha={model_sha}\n[assets] public dataset sha={data_sha}")

    # -- dataset facts ----------------------------------------------------------------
    raw = REPO_ROOT / "data" / "raw" / "pusht_expert_train.h5.zst"
    with h5py.File(H5_PATH, "r") as f:
        ep_len = f["ep_len"][:]
        ep_offset = f["ep_offset"][:]
        n_steps = int(f["state"].shape[0])
        pixel_chunks = f["pixels"].chunks
    data_info = {
        "h5_path": str(H5_PATH),
        "h5_bytes": H5_PATH.stat().st_size,
        "zst_bytes": raw.stat().st_size if raw.is_file() else None,
        "n_episodes": int(len(ep_len)),
        "n_steps": n_steps,
        "ep_len_min_mean_max": [int(ep_len.min()), float(ep_len.mean()), int(ep_len.max())],
        "pixels_chunks": list(pixel_chunks) if pixel_chunks else None,
        "ep_len_sha256": array_sha256(ep_len),
        "ep_offset_sha256": array_sha256(ep_offset),
    }
    print(f"[assets] dataset: {data_info['n_episodes']} episodes, {n_steps} steps, "
          f"{data_info['h5_bytes'] / 1e9:.1f} GB decompressed")

    # -- scalers ----------------------------------------------------------------------
    t = time.time()
    process = build_scalers(H5_PATH)
    save_scalers(process, run_dir / "scalers.npz")
    print(f"[assets] scalers fitted in {time.time() - t:.1f}s: {scaler_fingerprint(process)}")

    # -- weights: state dict after key remap, plus config -----------------------------
    model = load_model("cpu")
    torch.save(model.state_dict(), run_dir / "state_dict.pt")
    shutil.copy(STABLEWM_HOME / "hf_pusht" / "config.json", run_dir / "config.json")
    n_params = sum(p.numel() for p in model.parameters())

    # -- splits -----------------------------------------------------------------------
    splits = make_splits(len(ep_len))
    (run_dir / "splits.json").write_text(json.dumps(splits) + "\n")

    # -- manifest and card ------------------------------------------------------------
    manifest = build_manifest(
        run_id=run_id,
        kind="assets",
        seeds={"split": SPLIT_SEED},
        data=data_info,
        upstream_revisions={
            "public_model_repo": PUBLIC_MODEL_REPO,
            "public_model_sha": model_sha,
            "public_dataset_repo": PUBLIC_MODEL_REPO,
            "public_dataset_sha": data_sha,
        },
        metrics={"n_params": n_params},
        extra={
            "scalers": scaler_fingerprint(process),
            "split_sizes": {k: len(v) for k, v in splits["roles"].items()},
            "files": {
                p.name: file_sha256(p)
                for p in run_dir.iterdir()
                if p.is_file() and p.name != "manifest.json"
            },
        },
        started_at=t0,
    )
    write_manifest(run_dir, manifest)
    (run_dir / "README.md").write_text(
        f"# {run_id}\n\n"
        "Push-T study assets for SafetyDial. This is NOT a new model: `state_dict.pt` is the "
        f"released `{PUBLIC_MODEL_REPO}` LeWM at revision `{model_sha}` after the key remap in "
        "`scripts/download_data.py`, stored so the exact tensors are reproducible. "
        "`scalers.npz` holds the StandardScalers fitted on the full expert action/proprio/state "
        "columns, exactly as the historical planner fitted them. `splits.json` assigns whole "
        "expert episodes to roles (probe, replay, retention, roots, reserve). See manifest.json.\n"
    )
    print(f"[assets] wrote {run_dir}")

    if args.no_upload:
        return 0
    store = HFStore()
    sha = store.upload_run("pusht", "assets", run_dir, run_id=run_id)
    (run_dir / "hf_revision.txt").write_text(f"{store.repo_id('pusht')} assets/{run_id} {sha}\n")
    print(f"[assets] done in {time.time() - t0:.0f}s; revision {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
