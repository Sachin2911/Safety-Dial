#!/usr/bin/env python3
"""Stages 5b and 6 of the evolution study (docs/evoPlan/README.md): CMA-ES in imagination.

One stage run holds many CMA-ES runs (a grid of population size, ranking rule, noise level and
seed, declared in configs/evo/stage5.yaml `cmaes` or configs/evo/stage6.yaml `grid`). Every
run starts at theta_BC, scores candidates on K fitness roots resampled each generation (common
to all candidates of a generation), checkpoints so it resumes after a restart, and saves
snapshots (best so far and distribution mean) at the declared generations. Snapshots are
evaluated on the evaluation roots by evo_eval_snapshots.py, which runs alongside on CPU
workers. A seed index fixes CMA-ES sampling, root resampling and noise draws, so runs that
differ only in rule or noise level are paired.

    uv run python experiments/scripts/evo_cmaes.py --stage 5b --init            # prints RUN_ID
    uv run python experiments/scripts/evo_cmaes.py --stage 5b --run-id RUN_ID --shard 0 --n-shards 3
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


def run_grid(cfg, stage: str) -> list[dict]:
    """Deterministic list of CMA-ES runs for the stage (order fixes shard assignment)."""
    sz = cfg.run_sizes
    runs = []
    if stage == "5b":
        c = cfg.cmaes
        for lam in c.popsizes:
            for s in c.seeds:
                runs.append({"name": f"deb-k0-lam{lam}-s{s}", "rule": "deb", "lam_penalty": None, "noise_k": 0.0,
                             "n_samples": 1, "popsize": int(lam), "seed": int(s), "generations": int(c.generations)})
    else:
        c = cfg.grid
        for k in c.noise_levels:
            for rule in c.rules:
                for s in c.seeds:
                    name, base = str(rule.name), str(rule.kind)
                    runs.append({"name": f"{name}-k{k}-lam{c.popsize}-s{s}", "rule": base,
                                 "lam_penalty": float(rule.lam_multiple) * float(sz.penalty_scale) if base == "fixed" else None,
                                 "noise_k": float(k), "n_samples": 1 if float(k) == 0 else int(sz.n_samples_noisy),
                                 "popsize": int(c.popsize), "seed": int(s), "generations": int(c.generations)})
    if len({r["name"] for r in runs}) != len(runs):
        raise ValueError("run names must be unique")
    return runs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=sorted(STAGES), required=True)
    ap.add_argument("--init", action="store_true", help="create the stage run folder and exit")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()
    from helpers.evoInputs import EvoRun, fetch_inputs, load_stage_config

    cfg_name, tag = STAGES[args.stage]
    cfg = load_stage_config(cfg_name)
    if args.init:
        run = EvoRun.create(cfg, tag, args.run_id)
        runs = run_grid(cfg, args.stage)
        (run.run_dir / "runs.json").write_text(json.dumps(runs, indent=1) + "\n")
        print(run.run_id)
        return 0
    if not args.run_id or not 0 <= args.shard < args.n_shards:
        ap.error("--run-id and a valid --shard/--n-shards are required")
    run = EvoRun.create(cfg, tag, args.run_id, resume=True)
    runs = run_grid(cfg, args.stage)
    if json.loads((run.run_dir / "runs.json").read_text()) != runs:
        raise ValueError("run grid differs from the one recorded at --init")

    from helpers.evoImagine import ClosedLoopImaginer
    from helpers.evoPolicy import LinearLatentPolicy
    from helpers.evoRoots import encode_histories, fitness_roots
    from helpers.evoRun import CMAConfig, imagined_score_fn, run_cmaes
    from helpers.locoData import RenderContext
    from helpers.poseProbes import load_probe
    from helpers.walkerLewm import load_walker_model

    paths = fetch_inputs(cfg.inputs, ["model", "probes", "banks"])
    device = "cuda"
    model, scaler = load_walker_model(paths["model"], device)
    probe, _ = load_probe(paths["probes"] / "walker_mlp.pt", device)
    sigma = np.load(REPO_ROOT / cfg.noise_result.sigma_file)
    im = ClosedLoopImaginer(model, scaler, probe, device=device, sigma=torch.as_tensor(sigma, dtype=torch.float32, device=device))
    bc = np.load(REPO_ROOT / cfg.bc_policy.file)
    policy = LinearLatentPolicy.from_state(bc, device=device)
    ctx = RenderContext()
    fit = encode_histories(fitness_roots(paths["banks"]), im.encode, ctx,
                           cache_path=run.run_dir / "fitness_z_hist.npz", identity={"model": cfg.inputs.model.revision})
    ctx.close()
    sz = cfg.run_sizes
    mine = runs[args.shard :: args.n_shards]
    for r in mine:
        out_dir = run.run_dir / "cma" / r["name"]
        if (out_dir / "final.json").is_file():
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        c = CMAConfig(popsize=r["popsize"], generations=r["generations"], sigma0=float(sz.sigma0), seed=r["seed"],
                      k_roots=int(sz.k_roots), n_samples=r["n_samples"], noise_k=r["noise_k"], rule=r["rule"],
                      lam=r["lam_penalty"], eval_generations=tuple(int(g) for g in cfg.eval_generations),
                      checkpoint_every=int(cfg.checkpoint_every), cma_options={"CMA_diagonal": bool(sz.cma_diagonal)})
        score_fn = imagined_score_fn(im, policy, fit, n_samples=r["n_samples"], noise_k=r["noise_k"], seed=r["seed"])
        res = run_cmaes(c, bc["theta"], score_fn, out_dir, n_fitness_roots=len(fit.roots), resume=True,
                        log=lambda m: None)
        np.savez(out_dir / "final.npz", best_theta=res["best_theta"], mean_theta=res["mean_theta"])
        final = {**r, "best_stats": res["best_stats"], "counters": res["counters"], "wall_clock_s": time.time() - t0}
        (out_dir / "final.json").write_text(json.dumps(final, indent=1, default=float) + "\n")
        print(f"[evo-cma {args.stage} shard {args.shard}] {r['name']} done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
