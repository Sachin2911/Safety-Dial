#!/usr/bin/env python3
"""E5: closed loop, only if E3 holds. Original vs repaired model inside Safe CEM with the
whole-T constraint; a fixed-margin variant; a matched penalty-CEM reference; nominal planning
for context. 20 paired episodes per arm over two hazard layouts, equal candidate budgets, a
declared all-infeasible fallback (execute the least-violating candidate and record it), and an
audit of the action sequence CEM returns (the elite mean is re-scored).

    uv run python experiments/scripts/pusht_e5_closedloop.py --repaired runs/<adapted run>/weights.pt
"""

from __future__ import annotations

import argparse
import copy
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
from helpers.imagination import NominalPlanner, next_warm_start  # noqa: E402
from helpers.poseProbes import load_probe  # noqa: E402
from helpers.pushtAssets import load_model, load_scalers  # noqa: E402
from helpers.pushtGeometry import ARENA_HI, ARENA_LO, clearance_trace  # noqa: E402
from helpers.pushtLayouts import load_layouts  # noqa: E402
from helpers.pushtReplay import make_env, reset_root, run_actions  # noqa: E402
from helpers.safeCemT import SafePlannerT  # noqa: E402

ASSETS_RUN = REPO_ROOT / "runs" / "pusht-assets-20260926-1"
PROBES_RUN = REPO_ROOT / "runs" / "pusht-probes-20260926-1"
STUDY = REPO_ROOT / "data" / "study" / "pusht"
RESULTS = REPO_ROOT / "docs" / "mainPlan" / "results" / "e5"


def run_episode(env, planner, root, hazard, *, n_blocks: int, seed: int) -> dict:
    ctx = reset_root(env, root)
    frames, hist, state, warm = list(ctx.frames), ctx.history_actions.copy(), ctx.state.copy(), None
    goal = np.asarray(root.goal_state)
    states, diags, latency = [state.copy()], [], []
    for b in range(n_blocks):
        pr = planner.plan(frames, hist, ctx.goal_frame, seed=seed * 1000 + b, pusher_xy=state[:2], init_future=warm)
        latency.append(pr.solve_time)
        diags.append(pr.diag or {"frac_feasible": pr.frac_feasible})
        log = run_actions(env, pr.blocks[0].reshape(-1, 2), record_frames=True)
        frames = (frames + [log.frames[-1]])[-3:]
        hist = np.concatenate([hist[1:], pr.blocks[:1]], 0)
        warm = next_warm_start(pr.blocks)
        state = log.states[-1].copy()
        states.extend(log.states[1:])
    st = np.asarray(states)
    cl = clearance_trace(st[:, 2:5], hazard)
    px = st[:, :2]
    return {"violation_steps": int((cl < 0).sum()), "any_violation": bool((cl < 0).any()), "min_clearance": float(cl.min()),
            "final_block_err_px": float(np.linalg.norm(st[-1, 2:4] - goal[2:4])), "final_angle_err_deg": float(np.degrees(abs(np.arctan2(np.sin(st[-1, 4] - goal[4]), np.cos(st[-1, 4] - goal[4]))))),
            "arena_exit_steps": int(((px < ARENA_LO) | (px > ARENA_HI)).any(1).sum()), "all_infeasible_solves": int(sum(d.get("frac_feasible", 1.0) == 0.0 for d in diags)),
            "elite_infeasible_solves": int(sum(not d.get("elite_feasible", True) for d in diags)), "latency_s_mean": float(np.mean(latency)), "n_env_steps": int(len(st) - 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repaired", required=True, help="predictor-side weights.pt of the repaired model")
    ap.add_argument("--dial", type=float, default=None, help="Safe CEM margin; default: matched margin from E2 (adapted)")
    ap.add_argument("--fixed-margin", type=float, default=None, help="default: E2 fixed-margin control")
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--episodes", type=int, default=10, help="per layout")
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--samples", type=int, default=300)
    args = ap.parse_args()
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    base = load_model(device)
    process = load_scalers(ASSETS_RUN / "scalers.npz")
    probe, _ = load_probe(PROBES_RUN / "block_pose_mlp.pt", device)
    repaired = copy.deepcopy(base)
    sd = torch.load(args.repaired, map_location="cpu")
    missing, unexpected = repaired.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    repaired.eval()
    e2 = json.loads((REPO_ROOT / "docs" / "mainPlan" / "results" / "e2" / "repair.json").read_text())
    dial = args.dial if args.dial is not None else float(e2["arms"]["adapted"]["margin_matched_dev"])
    fixed = args.fixed_margin if args.fixed_margin is not None else float(e2["arms"]["fixed_margin"]["margin"])
    bank = Bank(STUDY / "test")
    layouts, _ = load_layouts(STUDY / "test" / "layouts.json")
    fam = [lay for lay in layouts if lay.family == "familiar" and lay.nominal_min_clearance < -5]
    rng = np.random.default_rng(0)
    rng.shuffle(fam)
    env = make_env()
    # Layouts the NOMINAL closed-loop controller actually violates (replanning can dodge an
    # open-loop crossing); the first two found are used for every arm.
    picks, screened = [], []
    nominal = NominalPlanner(base, process, device, num_samples=args.samples)
    for lay in fam:
        root = next(r for r in bank.roots if r.root_id == lay.root_id)
        r = run_episode(env, nominal, root, lay.shape, n_blocks=args.blocks, seed=0)
        screened.append({"root_id": lay.root_id, "nominal_violation_steps": r["violation_steps"]})
        if r["any_violation"]:
            picks.append(lay)
        if len(picks) == 2 or len(screened) >= 24:
            break
    if len(picks) < 2:
        print(f"[e5] only {len(picks)} layouts with nominal violations among {len(screened)} screened; using what exists")
    arms = {
        "nominal": lambda m, hz: NominalPlanner(m, process, device, num_samples=args.samples),
        "safe_original": lambda m, hz: SafePlannerT(base, process, device, probe=probe, hazard=hz, dial=dial, num_samples=args.samples),
        "safe_repaired": lambda m, hz: SafePlannerT(repaired, process, device, probe=probe, hazard=hz, dial=dial, num_samples=args.samples),
        "safe_original_fixed_margin": lambda m, hz: SafePlannerT(base, process, device, probe=probe, hazard=hz, dial=fixed, num_samples=args.samples),
        "penalty_original": lambda m, hz: SafePlannerT(base, process, device, probe=probe, hazard=hz, dial=dial, mode="penalty", lam=args.lam, num_samples=args.samples),
    }
    report = {"dial": dial, "fixed_margin": fixed, "lam": args.lam, "layouts": [lay.to_dict() for lay in picks], "screened": screened, "episodes_per_layout": args.episodes, "arms": {}}
    for arm, make in arms.items():
        rows = []
        for lay in picks:
            root = next(r for r in bank.roots if r.root_id == lay.root_id)
            planner = make(base, lay.shape)
            for e in range(args.episodes):
                r = run_episode(env, planner, root, lay.shape, n_blocks=args.blocks, seed=e)
                r["layout"] = lay.root_id
                rows.append(r)
        keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float, bool))]
        report["arms"][arm] = {"episodes": rows, "mean": {k: float(np.mean([r[k] for r in rows])) for k in keys}}
        m = report["arms"][arm]["mean"]
        print(f"[e5] {arm:28s} violation_steps {m['violation_steps']:.2f} any {m['any_violation']:.2f} block_err {m['final_block_err_px']:.1f}px arena_exit {m['arena_exit_steps']:.2f} infeasible {m['all_infeasible_solves']:.2f} latency {m['latency_s_mean']:.2f}s")
    report["wall_clock_s"] = time.time() - t0
    (RESULTS / "closedloop.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(f"[e5] done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
