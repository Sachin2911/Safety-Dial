#!/usr/bin/env python3
"""Safety Dial on Push-T: Safe CEM against a hazard placed between start and goal.

Three arms, all on the frozen released `pusht/lewm` checkpoint and the same CEM planner:

  penalty-noarena : C = goal + lam * hazard.   The legacy arm, no arena constraint.
                    Shows the escape exploit: the kinematic pusher leaves the arena.
  penalty         : C = goal + lam * (hazard + arena).  Escape closed, weight still
                    scalarised. Shows that closing the exploit is not sufficient.
  safe            : constraint-priority ranking, dial d = required clearance in px.
                    This is the Safety Dial.

Usage:
    uv run python experiments/scripts/safe_dial_pusht.py --smoke
    uv run python experiments/scripts/safe_dial_pusht.py --seeds 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO_ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "le-wm"))

import os

os.environ["STABLEWM_HOME"] = str(REPO_ROOT / "data" / "stablewm")
# This container has an 11.52-CPU cgroup quota (`/sys/fs/cgroup/cpu.max` = "1152000
# 100000") while `nproc` and `sched_getaffinity` both report 96. Torch sizes its intra-op
# pool from the affinity mask, so it spawns 96 threads into 11.5 cores and thrashes: a
# CEM solve measured 15s against the 1.0s the same planner reaches on a quiet box, with
# the GPU at 0-4% the whole time. Cap the pools before torch is imported.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import torch  # noqa: E402

torch.set_num_threads(8)
from sklearn import preprocessing  # noqa: E402

import stable_worldmodel as swm  # noqa: E402
from utils import get_img_preprocessor  # noqa: E402

from helpers.linProbeHelpers import make_pusht_state, real_violation_stats, wm_transform  # noqa: E402
from helpers.hazardSweep import detour_feasible, estimate_px_per_step  # noqa: E402
from helpers.probes import load_or_train_pusher_probe  # noqa: E402
from helpers.safeCEM import (  # noqa: E402
    calibrate_box_on_path,
    hazard_clearance_report,
    monotonicity,
    run_episode_dial,
    summarize_runs,
)

H5_PATH = REPO_ROOT / "data" / "processed" / "pusht_expert_train.h5"
PROBE_CACHE = REPO_ROOT / "data" / "probes" / "pusher_mlp.pt"
OUT_DIR = REPO_ROOT / "outputs" / "safe_dial"

# Geometry. The hazard constrains the PUSHER, so it must sit on the pusher's route to
# the block, not on the block itself. If the block starts inside the hazard the pusher
# cannot touch it without violating, task and safety become irreconcilable, and every
# setting degenerates to abandoning the task, which is exactly the uninformative result
# the earlier lambda sweep produced. Here the block and its goal are well clear of the
# box, so tightening the dial costs a detour (graceful degradation) rather than the task.
START_PUSHER = (40.0, 40.0)
GOAL_PUSHER = (250.0, 310.0)
START_BLOCK = (300.0, 310.0)
GOAL_BLOCK = (360.0, 310.0)
BLOCK_THETA = np.pi / 4

DIALS = [0.0, 10.0, 20.0, 30.0, 45.0, 60.0]
LAMBDAS = [0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


def build_process(h5_path: Path):
    """Fitted StandardScalers the planner needs, plus goal_ duplicates."""
    dataset = swm.data.HDF5Dataset(path=str(h5_path), keys_to_cache=["action", "proprio", "state"])
    process = {}
    for col in ["action", "proprio", "state"]:
        proc = preprocessing.StandardScaler()
        data = dataset.get_col_data(col)
        data = data[~np.isnan(data).any(axis=1)]
        proc.fit(data)
        process[col] = proc
        if col != "action":
            process[f"goal_{col}"] = proc
    return process


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--smoke", action="store_true", help="one episode per arm, cheap check")
    ap.add_argument(
        "--quick",
        action="store_true",
        help="reduced grids for a contended box: 4 settings per arm instead of 6-7",
    )
    ap.add_argument(
        "--probe-arena",
        action="store_true",
        help=(
            "use the legacy probe-based arena constraint instead of the action-space one. "
            "Required to reproduce docs/safeDial/results/results.json, which was produced "
            "before the action-space constraint existed. It is structurally unable to detect "
            "a genuine arena exit, so it is not the default."
        ),
    )
    ap.add_argument("--eval-budget", type=int, default=50)
    ap.add_argument("--force-probe", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("CUDA required")
        return 1
    if not H5_PATH.is_file():
        print(f"missing dataset {H5_PATH}")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Safety Dial on Push-T: Safe CEM vs penalty CEM")
    print("=" * 78)

    base_model = swm.policy.AutoCostModel("pusht/lewm").to(device).eval()

    probe, probe_stats = load_or_train_pusher_probe(
        base_model, H5_PATH, PROBE_CACHE, device=device, force=args.force_probe
    )

    pixel_prep = get_img_preprocessor("pixels", "pixels", img_size=224)
    goal_prep = get_img_preprocessor("goal", "goal", img_size=224)
    transform = {
        "pixels": wm_transform(pixel_prep, "pixels"),
        "goal": wm_transform(goal_prep, "goal"),
    }
    process = build_process(H5_PATH)

    start_state = make_pusht_state(START_PUSHER, START_BLOCK, block_theta=BLOCK_THETA)
    goal_state = make_pusht_state(GOAL_PUSHER, GOAL_BLOCK, block_theta=BLOCK_THETA)

    if args.smoke:
        dials, lams, noarena_lams = [0.0, 30.0], [0.0, 1.0], [0.0, 1.0]
        seeds = [0]
    elif args.quick:
        # Same planner settings as the full grid, just fewer points on each axis, so
        # every arm stays directly comparable. The noarena arm only needs to show the
        # escape exploit exists, so two points suffice.
        dials, lams, noarena_lams = [0.0, 20.0, 40.0, 60.0], [0.0, 0.1, 1.0, 10.0], [0.0, 1.0]
        seeds = list(range(args.seeds))
    else:
        dials, lams, noarena_lams = DIALS, LAMBDAS, LAMBDAS
        seeds = list(range(args.seeds))

    common = dict(
        base_model=base_model,
        probe=probe,
        swm=swm,
        process=process,
        transform=transform,
        start_state=start_state,
        goal_state=goal_state,
        eval_budget=args.eval_budget,
        device=device,
        action_space_arena=not args.probe_arena,
    )

    print(
        "arena constraint: "
        + ("probe-based (legacy, reproduces the committed table)" if args.probe_arena
           else "action-space (exact)")
    )

    all_rows = {}
    t_start = time.time()

    # --- step 1: unconstrained baseline, to learn the route the planner actually takes ---
    print("\n" + "-" * 78)
    print("step 1: unconstrained baseline (no hazard term), to find the actual route")
    print("-" * 78)
    base_run = run_episode_dial(
        mode="penalty", lam=0.0, dial=0.0, seed=0, hazard_box=(0.0, 1.0, 0.0, 1.0), **common
    )
    pps = estimate_px_per_step(base_run["states"])
    print(
        f"  block err={base_run['final_block_err_px']:.1f}px  "
        f"oob_frac={base_run['oob_frac']:.2f}  px per env step={pps:.1f}"
    )

    # --- step 2: place the hazard ON that route, and clear of start/goal at max dial ---
    print("\n" + "-" * 78)
    print("step 2: calibrate the hazard box on the baseline route, with clearance")
    print("-" * 78)
    hazard_box, box_cands = calibrate_box_on_path(
        base_run["states"],
        START_PUSHER,
        GOAL_PUSHER,
        max_dial=max(dials),
        clearance=5.0,
    )
    v = real_violation_stats(base_run["states"], hazard_box, entity="pusher")
    print(
        f"\n  baseline against the chosen box: violating steps={v['n_violating_steps']} "
        f"(frac {v['frac_violating']:.2f})"
    )
    print("  clearance of start/goal from the inflated box, per dial:")
    for r in hazard_clearance_report(START_PUSHER, GOAL_PUSHER, hazard_box, dials):
        flag = "  <-- START INSIDE" if r["start_inside"] else ("  <-- GOAL INSIDE" if r["goal_inside"] else "")
        print(
            f"    d={r['dial']:>5.1f}  inflated={r['inflated']}  "
            f"start_depth={r['start_depth']:>7.1f}  goal_depth={r['goal_depth']:>7.1f}{flag}"
        )
    feas = detour_feasible(START_PUSHER, GOAL_PUSHER, hazard_box, max(dials), pps, args.eval_budget)
    print(f"  detour feasible at the widest dial: {feas}")
    if v["frac_violating"] == 0:
        print("  WARNING: the unconstrained planner never enters the chosen box, so there")
        print("           is nothing to avoid and the dial sweep measures nothing.")

    common["hazard_box"] = hazard_box

    arms = [
        ("penalty-noarena", dict(mode="penalty", arena_constraint=False), "lam", noarena_lams),
        ("penalty", dict(mode="penalty", arena_constraint=True), "lam", lams),
        ("safe", dict(mode="safe", arena_constraint=True), "dial", dials),
    ]

    for arm_name, arm_kw, key, values in arms:
        print("\n" + "-" * 78)
        print(f"arm: {arm_name}   sweeping {key} over {values}   seeds={seeds}")
        print("-" * 78)
        rows = []
        for val in values:
            for s in seeds:
                kw = dict(common)
                kw.update(arm_kw)
                kw[key] = val
                if key == "lam":
                    kw["dial"] = 0.0
                else:
                    kw["lam"] = 0.0
                r = run_episode_dial(seed=s, **kw)
                rows.append(r)
            g = [r for r in rows if r[key] == val]
            print(
                f"  {key}={val:<7g} viol={np.mean([x['frac_violating'] for x in g]):.3f}  "
                f"block={np.mean([x['final_block_err_px'] for x in g]):6.1f}px  "
                f"oob={np.mean([x['oob_frac'] for x in g]):.3f}  "
                f"feasible={np.nanmean([x.get('final_frac_feasible', np.nan) for x in g]):.3f}"
            )
        all_rows[arm_name] = rows
        table, text = summarize_runs(rows, key=key)
        print()
        print(text)
        mono = monotonicity(table, key=key, metric="viol_mean")
        print(f"  monotonicity of violation vs {key}: {mono}")

    print(f"\ntotal wall clock: {(time.time() - t_start) / 60:.1f} min")

    # --- persist ---
    payload = {
        "config": {
            "start_pusher": START_PUSHER,
            "goal_pusher": GOAL_PUSHER,
            "start_block": START_BLOCK,
            "goal_block": GOAL_BLOCK,
            "block_theta": float(BLOCK_THETA),
            "hazard_box": [float(v) for v in hazard_box],
            "hazard_box_calibration": box_cands,
            "baseline_viol_vs_chosen_box": float(v["frac_violating"]),
            "baseline_block_err_px": float(base_run["final_block_err_px"]),
            "px_per_env_step": float(pps),
            "detour_feasible": {k: float(x) if not isinstance(x, bool) else x for k, x in feas.items()},
            "dials": dials,
            "lambdas": lams,
            "seeds": seeds,
            "eval_budget": args.eval_budget,
            "arena_constraint_mode": "probe_based" if args.probe_arena else "action_space",
            "probe_stats": probe_stats,
        },
        "arms": {},
    }
    for arm_name, rows in all_rows.items():
        key = "dial" if arm_name == "safe" else "lam"
        table, text = summarize_runs(rows, key=key)
        payload["arms"][arm_name] = {
            "key": key,
            "summary": table,
            "monotonicity": monotonicity(table, key=key, metric="viol_mean"),
            "episodes": [
                {k: v for k, v in r.items() if k not in ("states", "cost_history", "frames")}
                for r in rows
            ],
        }

    out_json = OUT_DIR / ("smoke.json" if args.smoke else "results.json")
    out_json.write_text(json.dumps(payload, indent=2, default=float))
    print(f"wrote {out_json}")

    np.savez_compressed(
        OUT_DIR / ("smoke_states.npz" if args.smoke else "results_states.npz"),
        **{
            f"{arm}_{i}": r["states"]
            for arm, rows in all_rows.items()
            for i, r in enumerate(rows)
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
