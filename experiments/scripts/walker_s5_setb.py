#!/usr/bin/env python3
"""S5: build set B for LeWM-B (walker2d.md). Set B is set A with N ordinary steps replaced by
branches drawn the way S4's random arm draws them (same proposal generator, same root
distribution). Each branch becomes a 120-step episode segment: the root's 20-step recorded
history (its actions replayed from the earliest history state) followed by the 100-step tape,
re-simulated with exact restore so the stored (qpos, qvel) are consistent. N is the largest
S4 budget in simulator steps (branches * 100).

    uv run python experiments/scripts/walker_s5_setb.py --branches 512 [--near-failure-frac 0.25 --name setB_plus]
Then: uv run python experiments/scripts/walker_s2_train_lewm.py --data data/study/walker2d/setB.h5 --name lewm-b
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from helpers.threads import pin_threads  # noqa: E402

pin_threads()

import h5py  # noqa: E402
import hdf5plugin  # noqa: E402,F401
import numpy as np  # noqa: E402

from helpers.hfStore import HFStore  # noqa: E402
from helpers.locoCollect import SCHEMA, StateHDF5Writer  # noqa: E402
from helpers.locoEnv import make_loco_env  # noqa: E402
from helpers.walkerBank import iter_episodes, sample_roots  # noqa: E402
from helpers.walkerRules import FRAMESKIP, execute_branch, propose_tapes  # noqa: E402

DATA = REPO_ROOT / "data" / "study" / "walker2d"


def segment_episode(env, root, tape, ep_idx: int, policy_id: int = 200) -> dict:
    """Re-simulate history actions + tape from the earliest history state; state schema rows."""
    actions = np.concatenate([root.history_actions.reshape(-1, 6), tape], 0)
    log = execute_branch(env, root.history_qpos[0], root.history_qvel[0], actions)
    n = len(actions)
    u = env.unwrapped
    obs = []
    for t in range(n):
        u.set_state(log.qpos[t], log.qvel[t])
        obs.append(u._get_obs())
    healthy = ((log.qpos[:-1, 1] > 0.8) & (log.qpos[:-1, 1] < 2.0) & (np.abs(log.qpos[:-1, 2]) < 1.0)).astype("u1")
    cols = {"qpos": log.qpos[:-1], "qvel": log.qvel[:-1], "action": actions.astype("f4"), "observation": np.asarray(obs, "f4"), "x_velocity": log.x_velocity,
            "reward": np.zeros(n, "f4"), "cost": (log.x_velocity > 2.3415).astype("u1"), "healthy": healthy, "terminated": np.zeros(n, "u1"), "truncated": np.zeros(n, "u1"),
            "episode_idx": np.full(n, ep_idx, "i4"), "step_idx": np.arange(n, dtype="i4"), "policy_id": np.full(n, policy_id, "u1")}
    return {k: np.asarray(v, dtype=SCHEMA[k][0]) for k, v in cols.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA))
    ap.add_argument("--branches", type=int, default=512)
    ap.add_argument("--tapes-per-root", type=int, default=8)
    ap.add_argument("--near-failure-frac", type=float, default=0.0, help="LeWM-B+: fraction of branch steps drawn from stress proposals")
    ap.add_argument("--name", default="setB")
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    data_dir = Path(args.data_dir)
    rng = np.random.default_rng(args.seed)
    env = make_loco_env("Walker2d", "v1", render=False, terminate_when_unhealthy=False, six_tuple=True)
    env.reset(seed=0)
    with h5py.File(data_dir / "setA.h5", "r") as f:
        attrs = dict(f.attrs)
        dims = {k: f[k].shape[1] for k in ("qpos", "qvel", "action", "observation")}
    seg_len = (2 + 10) * FRAMESKIP
    n_branch_steps = args.branches * seg_len
    # roots from the acquisition role (episode index mod 4 in {2, 3}), as S4's random arm
    n_roots = int(np.ceil(args.branches / args.tapes_per_root))
    n_stress = int(round(args.near_failure_frac * n_roots))
    roots = sample_roots(data_dir / "roots.h5", rng, n_roots - n_stress, kind="representative", episode_filter=lambda e: e % 4 in (2, 3), prefix="b")
    if n_stress:
        roots += sample_roots(data_dir / "roots.h5", rng, n_stress, kind="stress", episode_filter=lambda e: e % 4 in (2, 3), prefix="bs")
    w = StateHDF5Writer(data_dir / f"{args.name}.h5", dims, {**attrs, "set": args.name, "seed": args.seed, "n_branch_steps": n_branch_steps})
    # copy set A episodes minus a random subset whose total length ~ n_branch_steps
    eps = list(iter_episodes(data_dir / "setA.h5"))
    order = rng.permutation(len(eps))
    dropped, dropped_steps = set(), 0
    for i in order:
        if dropped_steps >= n_branch_steps:
            break
        dropped.add(int(i))
        dropped_steps += len(eps[i][1]["qpos"])
    ep_out = 0
    with h5py.File(data_dir / "setA.h5", "r") as f:
        ln, off = f["ep_len"][:], f["ep_offset"][:]
        for i in range(len(ln)):
            if i in dropped:
                continue
            a, b = int(off[i]), int(off[i] + ln[i])
            cols = {k: f[k][a:b] for k in SCHEMA}
            cols["episode_idx"] = np.full(b - a, ep_out, "i4")
            w.write_episode(cols)
            ep_out += 1
    n_added = 0
    for root in roots:
        stress = root.root_id.startswith("bs")
        items = propose_tapes(rng, root, n_random=args.tapes_per_root, sigmas=(0.3, 0.5) if stress else (0.1, 0.2, 0.4), n_bursts=args.tapes_per_root // 2 if stress else 0)
        for tape, kind, _ in items[: args.tapes_per_root]:
            w.write_episode(segment_episode(env, root, tape, ep_out))
            ep_out += 1
            n_added += 1
            if n_added >= args.branches:
                break
        if n_added >= args.branches:
            break
    w.close()
    stats = {"name": args.name, "n_branches": n_added, "branch_steps": n_added * seg_len, "dropped_setA_episodes": len(dropped), "dropped_steps": dropped_steps, "n_episodes": ep_out, "wall_clock_s": time.time() - t0}
    out = REPO_ROOT / "docs" / "mainPlan" / "results" / "s5"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps(stats, indent=1) + "\n")
    if not args.no_upload:
        HFStore().upload_file("walker2d-data", data_dir / f"{args.name}.h5", f"data/{args.name}/{args.name}.h5", message=f"add {args.name}")
    print(f"[s5] {json.dumps(stats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
