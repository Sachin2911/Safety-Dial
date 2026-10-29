"""Planner diagnosis for task 14 (Cube) before any hazard is designed.

Task 12 showed that a LeWM planner can be unable to reach goals the design assumes. This
script measures, with the frozen model and LeWM's planner settings and no hazard:
  1. success on random dataset start/goal pairs 25 steps apart (LeWM's own protocol), and
     how many of those are already solved at the start (the cube never moved);
  2. success on pairs in which the cube travels at least 8 cm in the plane (a real carry);
  3. that resetting to a dataset state and replaying the dataset's actions reproduces the
     recorded effector and cube paths, and that our frames match the dataset's;
  4. how well the commanded-path model predicts the effector the planner actually moves,
     with the per-step data to fit its gain offline;
  5. with --kinds reach, success on reach pairs (the cube stays put and the gripper travels
     at least --min-reach in the plane) under the gripper rule: the pinch point within
     --radius of the goal frame's. The free planner completed no carry in the first run.
Writes outputs/cube14/diagnosis/. Remote only (needs the GPU, checkpoint and dataset).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
os.environ.setdefault("STABLEWM_HOME", str(ROOT / "data/cube14/stablewm"))
from helpers.cube14_protocol import design
from helpers.dsrl06_infra import atomic_json, utc

OUT = ROOT / "outputs/cube14/diagnosis"


def choose_pairs(f, n, offset, rng, min_carry=None):
    """n (start_row, goal_row) pairs; with min_carry, the cube moves that far in the plane."""
    import numpy as np

    lengths = np.asarray(f["ep_len"][:])
    offsets = np.asarray(f["ep_offset"][:])
    pairs, tries = [], 0
    while len(pairs) < n:
        tries += 1
        if tries > 200 * n:
            raise RuntimeError("Too few pairs meet the carry threshold")
        ep = int(rng.integers(0, len(lengths)))
        t = int(rng.integers(1, lengths[ep] - offset - 1))
        start = int(offsets[ep]) + t
        goal = start + offset
        a = np.asarray(f["privileged_block_0_pos"][start])
        b = np.asarray(f["privileged_block_0_pos"][goal])
        carry = float(np.linalg.norm(b[:2] - a[:2]))
        if min_carry is not None and carry < min_carry:
            continue
        pairs.append(
            {"episode": ep, "t": t, "start_row": start, "goal_row": goal, "carry_m": carry}
        )
    return pairs


def choose_reach_pairs(f, n, offset, rng, min_reach, max_cube=0.01):
    """n pairs in which the cube moves less than max_cube and the gripper at least min_reach."""
    import numpy as np

    lengths = np.asarray(f["ep_len"][:])
    offsets = np.asarray(f["ep_offset"][:])
    pairs, tries = [], 0
    while len(pairs) < n:
        tries += 1
        if tries > 400 * n:
            raise RuntimeError("Too few pairs meet the reach threshold")
        ep = int(rng.integers(0, len(lengths)))
        t = int(rng.integers(1, lengths[ep] - offset - 1))
        start = int(offsets[ep]) + t
        goal = start + offset
        cube = np.asarray(f["privileged_block_0_pos"][[start, goal]])
        ee = np.asarray(f["proprio_effector_pos"][[start, goal]])
        moved = float(np.linalg.norm(cube[1] - cube[0]))
        reach = float(np.linalg.norm(ee[1, :2] - ee[0, :2]))
        if moved > max_cube or reach < min_reach:
            continue
        pairs.append(
            {
                "episode": ep,
                "t": t,
                "start_row": start,
                "goal_row": goal,
                "carry_m": moved,
                "reach_xy_m": reach,
            }
        )
    return pairs


def replay_check(rows, pair_rows):
    """Reset to the dataset state, replay the dataset actions, compare recorded paths."""
    import numpy as np

    from helpers.cube14_env import GoalCubeEnv

    env = GoalCubeEnv()
    out = []
    for p in pair_rows:
        pair = rows.pair(p["start_row"], p["goal_row"])
        env.reset(
            seed=0,
            options={
                "state": pair["state"],
                "goal_cube_pos": pair["goal_cube_pos"],
                "marker_pos": pair["marker_pos"],
            },
        )
        ee, cube = [], []
        for a in pair["actions"]:
            _, _, _, _, info = env.step(a)
            ee.append(info["state"][:3])
            cube.append(info["state"][3:6])
        ee_err = np.linalg.norm(np.asarray(ee) - pair["effector_path"], axis=1)
        cube_err = np.linalg.norm(np.asarray(cube) - pair["cube_path"], axis=1)
        out.append(
            {
                "start_row": p["start_row"],
                "effector_max_m": float(ee_err.max()),
                "effector_first_m": float(ee_err[0]),
                "cube_max_m": float(cube_err.max()),
                "final_success": bool(env._success),
            }
        )
    return out


def render_check(rows, pair_rows):
    """Our frame at a dataset state against the dataset's own frame for that row."""
    import numpy as np
    from PIL import Image

    from helpers.cube14_env import GoalCubeEnv

    env = GoalCubeEnv()
    out = {}
    for k, p in enumerate(pair_rows):
        pair = rows.pair(p["start_row"], p["goal_row"])
        variants = {"marker_dataset_target": pair["marker_pos"], "marker_goal": None}
        for name, marker in variants.items():
            options = {"state": pair["state"], "goal_cube_pos": pair["goal_cube_pos"]}
            if marker is not None:
                options["marker_pos"] = marker
            obs, _ = env.reset(seed=0, options=options)
            ours = np.asarray(obs).astype(np.uint8)
            theirs = pair["start_image"]
            diff = np.abs(ours.astype(float) - theirs.astype(float))
            out[f"{k}-{name}"] = {
                "mean_abs_diff": float(diff.mean()),
                "frac_pixels_over_30": float((diff.max(axis=-1) > 30).mean()),
            }
            if name == "marker_dataset_target":
                Image.fromarray(ours).save(OUT / f"render-ours-{k}.png")
                Image.fromarray(theirs).save(OUT / f"render-dataset-{k}.png")
    return out


def free_job(spec, seed, pair, tag, **planner):
    params = dict(spec["planner"])
    params.update(planner)
    return {
        "id": f"D-{tag}-{seed}",
        "stage": "D",
        "layout": None,
        "seed": int(seed),
        "mode": "penalty",
        "lam": 0.0,
        "clearances": [0.0] * int(params["horizon"]),
        "planner": params,
        "pair": {"start_row": pair["start_row"], "goal_row": pair["goal_row"]},
    }


def fit_gain(rows_out):
    """Least-squares per-axis gain (and carry) of realized on commanded effector steps."""
    import numpy as np

    from helpers.cube14_planner import STEP_M, WORKSPACE

    lo, hi = np.asarray(WORKSPACE[0]), np.asarray(WORKSPACE[1])
    cmd, real, prev = [], [], []
    for r in rows_out:
        ee = np.vstack([r["start"][:3], np.asarray(r["states"])[:, :3]])
        acts = np.asarray(r["actions"])
        p0 = np.asarray(r["prev_step"])
        for k in range(min(len(acts), len(ee) - 1)):
            target = np.clip(ee[k] + STEP_M * acts[k, :3], lo, hi)
            cmd.append(target - ee[k])
            real.append(ee[k + 1] - ee[k])
            prev.append(ee[k] - ee[k - 1] if k else p0)
    cmd, real, prev = map(np.asarray, (cmd, real, prev))
    gain = [float((cmd[:, i] @ real[:, i]) / max(cmd[:, i] @ cmd[:, i], 1e-12)) for i in range(3)]
    both = []
    for i in range(3):
        X = np.stack([cmd[:, i], prev[:, i]], axis=1)
        coef, *_ = np.linalg.lstsq(X, real[:, i], rcond=None)
        both.append([float(c) for c in coef])
    resid_unit = np.linalg.norm((real - cmd)[:, :2], axis=1)
    resid_gain = np.linalg.norm((real - cmd * np.asarray(gain))[:, :2], axis=1)
    return {
        "steps": int(len(cmd)),
        "gain_only": gain,
        "gain_and_carry": both,
        "unit_gain_xy_residual_m": {
            "median": float(np.median(resid_unit)),
            "p95": float(np.percentile(resid_unit, 95)),
            "max": float(resid_unit.max()),
        },
        "fitted_gain_xy_residual_m": {
            "median": float(np.median(resid_gain)),
            "p95": float(np.percentile(resid_gain, 95)),
            "max": float(resid_gain.max()),
        },
        "commanded_step_xy_m": {
            "median": float(np.median(np.linalg.norm(cmd[:, :2], axis=1))),
            "p95": float(np.percentile(np.linalg.norm(cmd[:, :2], axis=1), 95)),
            "max": float(np.linalg.norm(cmd[:, :2], axis=1).max()),
        },
    }


def main():
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kinds", nargs="+", default=["random", "carry"])
    parser.add_argument(
        "--rows", nargs="*", default=[], help="explicit reach pairs START:GOAL (kind 'rows')"
    )
    parser.add_argument("--random-pairs", type=int, default=24)
    parser.add_argument("--carry-pairs", type=int, default=24)
    parser.add_argument("--reach-pairs", type=int, default=24)
    parser.add_argument("--min-carry", type=float, default=0.08)
    parser.add_argument("--min-reach", type=float, default=0.12)
    parser.add_argument("--radius", type=float, default=0.03, help="gripper rule, metres")
    parser.add_argument("--seeds", type=int, default=1, help="episodes per pair")
    parser.add_argument("--samples", type=int, default=None, help="override CEM samples")
    parser.add_argument("--topk", type=int, default=None, help="override CEM elites")
    parser.add_argument("--tag", default="lewm")
    args = parser.parse_args()
    spec = design()
    spec["environment"]["success"] = {"rule": "cube", "effector_radius_m": args.radius}
    OUT.mkdir(parents=True, exist_ok=True)
    offset = int(spec["environment"]["goal_offset_steps"])
    from helpers.cube14_runtime import DatasetRows, Runtime

    rows = DatasetRows()
    rng = np.random.default_rng(2026102514)
    chosen = {
        "random": lambda: choose_pairs(rows.f, args.random_pairs, offset, rng),
        "carry": lambda: choose_pairs(
            rows.f, args.carry_pairs, offset, rng, min_carry=args.min_carry
        ),
        "reach": lambda: choose_reach_pairs(
            rows.f, args.reach_pairs, offset, rng, min_reach=args.min_reach
        ),
        "rows": lambda: [
            {"start_row": int(a), "goal_row": int(b), "carry_m": None}
            for a, b in (item.split(":") for item in args.rows)
        ],
    }
    kinds = {kind: chosen[kind]() for kind in args.kinds}
    probe = next(iter(kinds.values()))
    checks = {
        "replay": replay_check(rows, probe[:6]),
        "render": render_check(rows, probe[:2]),
    }
    atomic_json(OUT / f"checks-{args.tag}.json", {"utc": utc(), **checks})
    print(json.dumps({"checks": checks}), flush=True)
    runtime = Runtime(device="cuda", spec=spec, rows=rows)
    overrides = {}
    if args.samples:
        overrides["num_samples"] = args.samples
    if args.topk:
        overrides["topk"] = args.topk
    rows_out = []
    stream = (OUT / f"results-{args.tag}.jsonl").open("a")
    for kind, pairs in kinds.items():
        for i, pair in enumerate(pairs):
            for s in range(args.seeds):
                seed = 100 + i + 1000 * s
                job = free_job(spec, seed, pair, f"{args.tag}-{kind}{i:02d}", **overrides)
                if kind in ("reach", "rows"):
                    job["success_rule"] = "effector"
                tick = time.perf_counter()
                outcome = runtime.run(job)
                m = outcome["metrics"]
                info = runtime.rows.pair(pair["start_row"], pair["goal_row"])
                row = {
                    "id": job["id"],
                    "kind": kind,
                    **pair,
                    "seconds": time.perf_counter() - tick,
                    **{k: m[k] for k in m if k not in ("planner", "execution")},
                    "start": outcome["start"].round(5).tolist(),
                    "states": outcome["states"].round(5).tolist(),
                    "actions": outcome["actions"].round(5).tolist(),
                    "prev_step": info["prev_step"].round(6).tolist(),
                    "dataset_effector_path": info["effector_path"][:, :3].round(4).tolist(),
                }
                rows_out.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                keys = ("id", "carry_m", "success", "J", "start_effector_goal_m")
                keys += ("final_effector_goal_m", "min_effector_goal_m", "seconds")
                print(json.dumps({k: row.get(k) for k in keys}), flush=True)

    def rate(kind, trivial=None):
        sel = [r for r in rows_out if r["kind"] == kind]
        if trivial is not None:
            sel = [r for r in sel if (r["start_cube_goal_m"] <= 0.04) == trivial]
        return {
            "n": len(sel),
            "success": float(np.mean([r["success"] for r in sel])) if sel else None,
        }

    blocks = [r["tracking"]["block_xy_m"]["max"] for r in rows_out]
    blocks = [b for b in blocks if b is not None]
    summary = {
        "utc": utc(),
        "tag": args.tag,
        "planner": {**spec["planner"], **overrides},
        "checks": checks,
        **{kind: rate(kind) for kind in kinds},
        "random_trivial": rate("random", trivial=True) if "random" in kinds else None,
        "random_nontrivial": rate("random", trivial=False) if "random" in kinds else None,
        "thresholds": {"min_carry_m": args.min_carry, "min_reach_m": args.min_reach},
        "gripper_radius_m": args.radius,
        "tracking_unit_gain": {
            "one_step_xy_max_m": float(
                max(r["tracking"]["one_step_xy_m"]["max"] for r in rows_out)
            ),
            "block_xy_p95_m": float(np.percentile(blocks, 95)) if blocks else None,
            "block_xy_max_m": float(max(blocks)) if blocks else None,
        },
        "gain_fit": fit_gain(rows_out),
        "max_step_m": float(max(r["max_step_m"] for r in rows_out)),
        "median_step_m": float(np.median([r["median_step_m"] for r in rows_out])),
        "seconds_per_episode": float(np.median([r["seconds"] for r in rows_out])),
        "episodes": len(rows_out),
    }
    atomic_json(OUT / f"summary-{args.tag}.json", summary)
    print(json.dumps({"summary": {k: v for k, v in summary.items() if k != "checks"}}))


if __name__ == "__main__":
    main()
