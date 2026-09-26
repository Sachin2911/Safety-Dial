#!/usr/bin/env python3
"""S1: collect Walker2d data sets A (pretraining), probe and roots with the exported policies.

State-only HDF5 with lazy rendering (helpers.locoCollect, imported never edited). Set A is
expert-like: mid and late PPO / PPO-Lag checkpoints with action noise sigma in {0, 0.05, 0.1},
benchmark termination on, early checkpoints capped at 10% of steps. The probe set adds
continuations past health violations (termination off, stop_after_unhealthy) and random /
early-policy episodes for pose and speed coverage. Whole episodes are split into roles.

    uv run python experiments/scripts/walker_s1_collect.py --steps-a 3000000
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


from helpers.hfStore import HFStore  # noqa: E402
from helpers.walkerProtocol import file_sha256, root_roles
from helpers.locoCollect import StateHDF5Writer, collect, rollout_episode  # noqa: E402
from helpers.locoData import RenderContext, render_fingerprint  # noqa: E402
from helpers.locoEnv import env_manifest, make_loco_env  # noqa: E402
from helpers.locoPolicies import make_policy  # noqa: E402
from helpers.runManifest import build_manifest, make_run_id, write_manifest  # noqa: E402
from helpers.walkerAssets import evaluate_policy, export_ids, list_exports, load_exported_actor  # noqa: E402

OUT = REPO_ROOT / "data" / "study" / "walker2d"
RESULTS = REPO_ROOT / "docs" / "mainPlan" / "results" / "s1"


def dims_for(env) -> dict:
    u = env.unwrapped
    return {"qpos": u.model.nq, "qvel": u.model.nv, "action": env.action_space.shape[0], "observation": env.observation_space.shape[0]}


def ladder(env, n_eval=3, seed=0, policy_dir=None) -> list[dict]:
    """Measure every exported checkpoint in the vendored env; return sorted by speed."""
    rows = []
    for p in (list_exports(Path(policy_dir)) if policy_dir else list_exports()):
        pol = load_exported_actor(p, env.action_space, seed=seed)
        m = evaluate_policy(env, pol, n_episodes=n_eval, seed=seed)
        rows.append({"path": str(p), "name": pol.name, **m})
        print(f"[s1] {pol.name:14s} return {m['return_mean']:7.1f} speed {m['speed_mean']:.2f} cost_rate {m['cost_rate']:.3f} fall_rate {m['fall_rate']:.2f} len {m['len_mean']:.0f}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, help="fresh data directory; default includes run ID")
    ap.add_argument("--policy-dir", type=Path, help="exported actor folder")
    ap.add_argument("--steps-a", type=int, default=3_000_000)
    ap.add_argument("--steps-probe", type=int, default=300_000)
    ap.add_argument("--steps-roots", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--n", type=int, default=1)
    args = ap.parse_args()
    t0 = time.time()
    if min(args.steps_a, args.steps_probe, args.steps_roots) < 1:
        ap.error("all step budgets must be positive")
    run_id = make_run_id("walker2d", "data", n=args.n)
    out = args.data_dir or OUT / run_id
    out.mkdir(parents=True, exist_ok=False)
    results = RESULTS / run_id
    results.mkdir(parents=True, exist_ok=False)
    env = make_loco_env("Walker2d", "v1", render=False)
    ctx = RenderContext()
    fp = render_fingerprint(ctx)
    attrs = {"render_fingerprint": fp, "env_manifest": json.dumps(env_manifest(ctx.env)), "run_id": run_id, "policy_ids": json.dumps(export_ids(args.policy_dir) if args.policy_dir else export_ids()), "protocol_version": 2}
    rows = ladder(env, policy_dir=args.policy_dir)
    (results / "policy_ladder.json").write_text(json.dumps(rows, indent=1) + "\n")
    # competent = long episodes and positive return; early = the rest (capped at 10% of set A)
    competent = [r for r in rows if r["len_mean"] >= 400 and r["return_mean"] > 500]
    early = [r for r in rows if r not in competent]
    if not competent:
        print("[s1] no competent checkpoint yet; train longer before collecting set A")
        return 1
    print(f"[s1] competent checkpoints: {[r['name'] for r in competent]}; early: {[r['name'] for r in early]}")

    def mixed_policy(rows_pool, sigmas, seed):
        pols = []
        for r in rows_pool:
            for s in sigmas:
                pols.append(load_exported_actor(Path(r["path"]), env.action_space, seed=seed, action_noise=s))
        return make_policy_mixture(pols, seed)

    stats = {}
    # ---- set A --------------------------------------------------------------------------
    wA = StateHDF5Writer(out / "setA.h5", dims_for(env), attrs)
    role_writer = UniqueEpisodeWriter(wA)
    stats["A_competent"] = collect(env, mixed_policy(competent, (0.0, 0.05, 0.1), args.seed), role_writer, n_steps=max(1, int(args.steps_a * (0.9 if early else 1))), seed=args.seed)
    # collect() finishes whole episodes of at most 1,000 steps. Leave one episode
    # of headroom so the early component cannot exceed ten percent of set A.
    early_budget = max(0, stats["A_competent"]["n_steps"] // 9 - 1000)
    if early and early_budget:
        stats["A_early"] = collect(env, mixed_policy(early, (0.0, 0.05), args.seed + 1), role_writer, n_steps=early_budget, seed=args.seed + 10_000_000)
    pretraining_episodes = len(wA.ep_lens)
    wA.close()
    # ---- probe set: coverage incl. continuations past health violations -----------------
    env_nt = make_loco_env("Walker2d", "v1", render=False, terminate_when_unhealthy=False)
    wP = StateHDF5Writer(out / "probe.h5", dims_for(env), attrs)
    n = 0
    ep = 0
    pols = [make_policy("random", env.action_space, seed=5), make_policy("scripted_forward", env.action_space, seed=6)] + [load_exported_actor(Path(r["path"]), env.action_space, seed=7, action_noise=0.1) for r in rows]
    while n < args.steps_probe:
        pol = pols[ep % len(pols)]
        cols = rollout_episode(env_nt, pol, seed=args.seed + 20_000_000 + ep, stop_after_unhealthy=60)
        cols["episode_idx"][:] = ep
        wP.write_episode(cols)
        n += len(cols["reward"])
        ep += 1
    wP.close()
    stats["probe"] = {"n_steps": n, "n_episodes": ep}
    # ---- roots: held-out episodes of the set-A policies ---------------------------------
    wR = StateHDF5Writer(out / "roots.h5", dims_for(env), attrs)
    stats["roots"] = collect(env, mixed_policy(competent, (0.0, 0.05, 0.1), args.seed + 2), wR, n_steps=args.steps_roots, seed=args.seed + 30_000_000)
    wR.close()
    split_manifest = {"protocol_version": 2, "pretraining": list(range(pretraining_episodes)), "probe": list(range(ep)), "roots": root_roles(stats["roots"]["n_episodes"]), "seed_bases": {"A_competent": args.seed, "A_early": args.seed + 10_000_000, "probe": args.seed + 20_000_000, "roots": args.seed + 30_000_000}}
    (out / "splits.json").write_text(json.dumps(split_manifest, indent=2) + "\n")
    from omegaconf import OmegaConf

    OmegaConf.save(OmegaConf.create({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}), out / "config.yaml")
    fingerprints = {p.name: file_sha256(p) for p in out.glob("*.h5")}
    manifest = build_manifest(run_id=run_id, kind="data", seeds={"seed": args.seed}, data={"render_fingerprint": fp, "policies": rows, "stats": stats, "sha256": fingerprints, "splits": split_manifest}, started_at=t0)
    write_manifest(out, manifest)
    (out / "README.md").write_text(f"# {run_id}\n\nWalker2d state-only data (set A, probe, roots) collected in the vendored SafetyWalker2dVelocity-v1 with exported OmniSafe actors. See manifest.json.\n")
    if not args.no_upload:
        store = HFStore()
        rev = store.upload_run("walker2d-data", "data", out, run_id=run_id)
        manifest["hf_revision"] = rev
        (out / "upload_receipt.json").write_text(json.dumps({"repo_id": store.repo_id("walker2d-data"), "revision": rev, "path": f"data/{run_id}"}, indent=2) + "\n")
    (results / "collection.json").write_text(json.dumps({"run_id": run_id, "stats": stats, "hf_revision": manifest.get("hf_revision"), "wall_clock_s": time.time() - t0}, indent=1) + "\n")
    env.close()
    env_nt.close()
    ctx.close()
    print(f"[s1] done in {time.time() - t0:.0f}s")
    return 0


class UniqueEpisodeWriter:
    """Keep episode IDs unique when multiple policy components share a file."""

    def __init__(self, writer):
        self.writer = writer

    def write_episode(self, columns):
        columns["episode_idx"][:] = len(self.writer.ep_lens)
        return self.writer.write_episode(columns)


def make_policy_mixture(policies, seed):
    from helpers.locoPolicies import MixedPolicy

    return MixedPolicy(policies, seed=seed)


if __name__ == "__main__":
    raise SystemExit(main())
