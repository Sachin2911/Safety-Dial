#!/usr/bin/env python3
"""Evaluate CMA-ES snapshots of stages 5b and 6 on the 256 evaluation roots.

Runs alongside the CMA-ES shards (evo_cmaes.py): it polls the stage run folder, batches
snapshots that have appeared, and evaluates each snapshot's best-so-far and distribution-mean
policies in imagination (k = 0, and for noisy runs also at the run's own noise level with its
n samples) and in the real simulator (dense truth, real readout), on a persistent pool of CPU
workers, so real evaluation overlaps imagination on the GPU. Results land next to the
snapshots (`eval/gNNNN.npz`); it exits when every declared snapshot is evaluated.

    uv run python experiments/scripts/evo_eval_snapshots.py --stage 5b --run-id RUN_ID
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

from helpers.threads import apply_torch, pin_threads  # noqa: E402

pin_threads(2)
import numpy as np  # noqa: E402
import torch  # noqa: E402

apply_torch()

STAGES = {"5b": ("stage5", "s5b"), "6": ("stage6", "s6")}


def expected_snapshots(runs, eval_generations) -> list[tuple[str, int, dict]]:
    return [(r["name"], int(g), r) for r in runs for g in eval_generations if int(g) <= r["generations"]]


def main() -> int:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=sorted(STAGES), required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--poll", type=float, default=20.0)
    ap.add_argument("--batch", type=int, default=16, help="snapshots per evaluation batch")
    args = ap.parse_args()
    from helpers.evoImagine import ClosedLoopImaginer
    from helpers.evoInputs import EvoRun, fetch_inputs, load_stage_config
    from helpers.evoPolicy import LinearLatentPolicy
    from helpers.evoRanking import segment_metrics
    from helpers.evoReal import RealPool
    from helpers.evoRoots import encode_histories, evaluation_roots, rootset_digest
    from helpers.locoData import RenderContext
    from helpers.poseProbes import load_probe
    from helpers.walkerLewm import load_walker_model

    cfg_name, tag = STAGES[args.stage]
    cfg = load_stage_config(cfg_name)
    run = EvoRun.create(cfg, tag, args.run_id, resume=True)
    runs = json.loads((run.run_dir / "runs.json").read_text())
    todo = expected_snapshots(runs, cfg.eval_generations)
    paths = fetch_inputs(cfg.inputs, ["model", "probes", "data", "banks"])
    device = "cuda"
    model, scaler = load_walker_model(paths["model"], device)
    probe_path = paths["probes"] / "walker_mlp.pt"
    probe, _ = load_probe(probe_path, device)
    sigma = np.load(REPO_ROOT / cfg.noise_result.sigma_file)
    im = ClosedLoopImaginer(model, scaler, probe, device=device, sigma=torch.as_tensor(sigma, dtype=torch.float32, device=device))
    bc = np.load(REPO_ROOT / cfg.bc_policy.file)
    policy = LinearLatentPolicy.from_state(bc, device=device)
    ctx = RenderContext()
    ev = encode_histories(evaluation_roots(paths["banks"], paths["data"] / "roots.h5", seed=cfg.roots.evaluation_seed,
                                           extra_per_episode=cfg.roots.extra_per_episode), im.encode, ctx,
                          cache_path=run.run_dir / "evaluation_z_hist.npz", identity={"model": cfg.inputs.model.revision})
    ctx.close()
    R = len(ev.roots)
    (run.run_dir / "evaluation_clusters.json").write_text(json.dumps({
        "source_episode": ev.source_episode.tolist(), "root_ids": ev.root_ids, "digest": rootset_digest(ev)}) + "\n")
    zh = torch.as_tensor(ev.z_hist, device=device)
    pool = RealPool(model_dir=paths["model"], probe_path=probe_path, policy_state=policy.state(),
                    n_workers=int(cfg.run_sizes.real_workers), device=device, max_envs=int(cfg.evaluation.max_envs),
                    chunk_tasks=int(cfg.evaluation.chunk_tasks), encode_batch=int(cfg.evaluation.encode_batch))
    t0, n_done = time.time(), 0
    try:
        while True:
            pending = [(name, g, r) for name, g, r in todo
                       if (run.run_dir / "cma" / name / "snapshots" / f"g{g:04d}.npz").is_file()
                       and not (run.run_dir / "cma" / name / "eval" / f"g{g:04d}.npz").is_file()]
            if not pending:
                if all((run.run_dir / "cma" / n / "eval" / f"g{g:04d}.npz").is_file() for n, g, _ in todo):
                    break
                time.sleep(args.poll)
                continue
            batch = pending[: args.batch]
            snaps = [np.load(run.run_dir / "cma" / name / "snapshots" / f"g{g:04d}.npz") for name, g, _ in batch]
            thetas = np.stack([t for s in snaps for t in (s["best_theta"], s["mean_theta"])])
            imag0 = segment_metrics(im.rollout_policies(policy, thetas, zh, ev.hist_blocks)["readout"][:, :, 0])
            real = pool.evaluate(thetas, ev.roots, chunk_tasks=int(cfg.evaluation.chunk_tasks))
            T = len(thetas)
            real_viol = real["dense_violated"].reshape(T, R)
            real_ret = real["progress"].reshape(T, R)
            readout = segment_metrics(real["readout"].reshape(T, R, -1, 3))
            for i, (name, g, r) in enumerate(batch):
                sl = slice(2 * i, 2 * i + 2)
                out = {"imag0_violated": imag0["violated"][sl], "imag0_ret": imag0["ret"][sl],
                       "real_violated": real_viol[sl], "real_ret": real_ret[sl], "real_first_step": real["dense_first_step"].reshape(T, R)[sl],
                       "readout_violated": readout["violated"][sl], "readout_ret": readout["ret"][sl],
                       "endpoint_violated": (real["endpoint_clearance"].reshape(T, R, -1)[sl] <= 0).any(-1)}
                if r["noise_k"] > 0:
                    own = im.rollout_policies(policy, thetas[sl], zh, ev.hist_blocks, n_samples=r["n_samples"],
                                              k=r["noise_k"], seed=int(cfg.eval_noise_seed))
                    mo = segment_metrics(own["readout"])  # (2, R, n)
                    out.update(imag_own_violated=mo["violated"], imag_own_ret=mo["ret"])
                dest = run.run_dir / "cma" / name / "eval"
                dest.mkdir(parents=True, exist_ok=True)
                tmp = dest / f"g{g:04d}.tmp.npz"
                np.savez_compressed(tmp, **out)
                os.replace(tmp, dest / f"g{g:04d}.npz")
            n_done += len(batch)
            print(f"[evo-eval {args.stage}] {n_done} snapshots evaluated ({len(pending) - len(batch)} pending, "
                  f"{time.time() - t0:.0f}s)", flush=True)
    finally:
        pool.close()
    (run.run_dir / "eval_complete.json").write_text(json.dumps({"n_snapshots": len(todo), "wall_clock_s": time.time() - t0,
                                                                "imagined_rows": int(im.counters.imagined_rows)}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
