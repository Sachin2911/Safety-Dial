#!/usr/bin/env python3
"""Validate the action-space arena constraint at the dial setting that broke.

At dial 60 the probe-based arena constraint fails structurally: the probe is trained on
in-arena frames, so once the pusher leaves the 512 px arena it is not visible in the 224 px
frame and the probe's estimate is uninformative. The planner then reports most candidates
feasible while driving off-screen, and the zero hazard violation it achieves is vacuous.

`safeCEM.commanded_positions` computes the same constraint in action space instead, where it
is exact: the pusher is position-controlled, so the commanded trajectory is a closed-form
function of the candidate actions and the current position.

This script runs dial 60 both ways so the difference is reproducible from the repo, and saves
states for the trajectory figures.

    uv run python experiments/scripts/validate_arena_fix.py
    uv run python experiments/scripts/validate_arena_fix.py --seeds 3 --both
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "le-wm"))

import os

os.environ["STABLEWM_HOME"] = str(REPO_ROOT / "data" / "stablewm")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import torch  # noqa: E402

torch.set_num_threads(8)
from sklearn import preprocessing  # noqa: E402

import stable_worldmodel as swm  # noqa: E402
from utils import get_img_preprocessor  # noqa: E402

from helpers.linProbeHelpers import make_pusht_state, wm_transform  # noqa: E402
from helpers.probes import load_or_train_pusher_probe  # noqa: E402
from helpers.safeCEM import run_episode_dial  # noqa: E402

H5_PATH = REPO_ROOT / "data" / "processed" / "pusht_expert_train.h5"
PROBE_CACHE = REPO_ROOT / "data" / "probes" / "pusher_mlp.pt"
OUT_DIR = REPO_ROOT / "docs" / "safeDial" / "results"

# Matches safe_dial_pusht.py, and the calibrated box from that run.
START_PUSHER = (40.0, 40.0)
GOAL_PUSHER = (250.0, 310.0)
START_BLOCK = (300.0, 310.0)
GOAL_BLOCK = (360.0, 310.0)
BLOCK_THETA = np.pi / 4
HAZARD_BOX = (30.104567677974686, 130.10456767797467, 148.66144357681293, 248.66144357681293)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dial", type=float, default=60.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument(
        "--both",
        action="store_true",
        help="also re-run the broken probe-based constraint for a paired comparison",
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("CUDA required")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base_model = swm.policy.AutoCostModel("pusht/lewm").to(device).eval()
    probe, probe_stats = load_or_train_pusher_probe(base_model, H5_PATH, PROBE_CACHE, device=device)

    dataset = swm.data.HDF5Dataset(
        path=str(H5_PATH), keys_to_cache=["action", "proprio", "state"]
    )
    process = {}
    for col in ["action", "proprio", "state"]:
        proc = preprocessing.StandardScaler()
        data = dataset.get_col_data(col)
        data = data[~np.isnan(data).any(axis=1)]
        proc.fit(data)
        process[col] = proc
        if col != "action":
            process[f"goal_{col}"] = proc

    transform = {
        "pixels": wm_transform(get_img_preprocessor("pixels", "pixels", img_size=224), "pixels"),
        "goal": wm_transform(get_img_preprocessor("goal", "goal", img_size=224), "goal"),
    }

    common = dict(
        base_model=base_model,
        probe=probe,
        hazard_box=HAZARD_BOX,
        swm=swm,
        process=process,
        transform=transform,
        start_state=make_pusht_state(START_PUSHER, START_BLOCK, block_theta=BLOCK_THETA),
        goal_state=make_pusht_state(GOAL_PUSHER, GOAL_BLOCK, block_theta=BLOCK_THETA),
        eval_budget=50,
        device=device,
    )

    variants = [("action_space", True)]
    if args.both:
        variants.insert(0, ("probe_based", False))

    payload = {
        "config": {
            "dial": args.dial,
            "hazard_box": list(HAZARD_BOX),
            "start_pusher": list(START_PUSHER),
            "goal_pusher": list(GOAL_PUSHER),
            "start_block": list(START_BLOCK),
            "goal_block": list(GOAL_BLOCK),
            "seeds": list(range(args.seeds)),
            "probe_stats": probe_stats,
        },
        "variants": {},
    }
    states_out = {}

    for name, use_action_space in variants:
        print("\n" + "=" * 72)
        print(f"dial={args.dial:g}  arena constraint: {name}")
        print("=" * 72)
        rows = []
        for s in range(args.seeds):
            r = run_episode_dial(
                mode="safe", dial=args.dial, seed=s, action_space_arena=use_action_space, **common
            )
            st = r["states"]
            moved = float(np.linalg.norm(st[-1, 2:4] - st[0, 2:4]))
            print(
                f"  seed {s}: viol={r['frac_violating']:.3f} block_err={r['final_block_err_px']:6.1f}px "
                f"oob={r['oob_frac']:.3f} block_moved={moved:5.1f}px "
                f"pusher_x[{st[:, 0].min():8.1f},{st[:, 0].max():7.1f}] "
                f"feasible={r.get('final_frac_feasible', float('nan')):.3f}"
            )
            rows.append(
                {
                    "seed": s,
                    "frac_violating": float(r["frac_violating"]),
                    "final_block_err_px": float(r["final_block_err_px"]),
                    "oob_frac": float(r["oob_frac"]),
                    "block_moved_px": moved,
                    "final_frac_feasible": float(r.get("final_frac_feasible", np.nan)),
                }
            )
            states_out[f"{name}_{s}"] = st
        payload["variants"][name] = {
            "episodes": rows,
            "mean": {
                k: float(np.mean([x[k] for x in rows]))
                for k in ("frac_violating", "final_block_err_px", "oob_frac", "block_moved_px")
            },
        }

    (OUT_DIR / "arena_fix.json").write_text(json.dumps(payload, indent=2, default=float))
    np.savez_compressed(OUT_DIR / "arena_fix_states.npz", **states_out)
    print(f"\nwrote {OUT_DIR / 'arena_fix.json'}")
    print(f"wrote {OUT_DIR / 'arena_fix_states.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
