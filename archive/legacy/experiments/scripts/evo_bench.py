#!/usr/bin/env python3
"""Stage 3a of the evolution study (docs/evoPlan/README.md): throughput and run sizes.

Measures, on this machine: the cgroup CPU quota; imagined predictor rows per second for
population x roots x noise batches, alone and with several imagination processes sharing the
GPU; pycma ask/tell time at 2,310 dimensions; real segments per second (render, encode, act,
simulate) against the number of worker processes, checking that worker count does not change
the real results. Then applies the rules declared in configs/evo/stage3.yaml, before any
stage 4 to 6 outcome exists, to fix: sigma0 (CMA-ES initial step and the perturbation unit),
the fixed-penalty scale, and the number of fitness roots K per generation.

    uv run python experiments/scripts/evo_bench.py [--run-id ID] [--no-upload]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from helpers.threads import apply_torch, cpu_quota_cores, pin_threads  # noqa: E402

pin_threads()
import numpy as np  # noqa: E402
import torch  # noqa: E402

apply_torch()

from helpers.evoImagine import ClosedLoopImaginer  # noqa: E402
from helpers.evoInputs import EvoRun, fetch_inputs, input_revisions, load_stage_config  # noqa: E402
from helpers.evoPolicy import LinearLatentPolicy  # noqa: E402
from helpers.evoRanking import segment_metrics  # noqa: E402
from helpers.evoReal import evaluate_parallel  # noqa: E402
from helpers.evoRoots import encode_histories, evaluation_roots, fitness_roots, tuning_roots  # noqa: E402
from helpers.locoData import RenderContext  # noqa: E402
from helpers.poseProbes import load_probe  # noqa: E402
from helpers.runManifest import build_manifest, file_sha256  # noqa: E402
from helpers.walkerLewm import load_walker_model  # noqa: E402
from helpers.walkerRules import HORIZON_BLOCKS  # noqa: E402


def time_rollouts(im, policy, theta, z_hist, hist_blocks, *, n, k, reps: int, max_batch: int) -> dict:
    zh = torch.as_tensor(z_hist, device=im.device)
    im.rollout_policies(policy, theta, zh, hist_blocks, n_samples=n, k=k, seed=0, max_batch=max_batch)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for r in range(reps):
        im.rollout_policies(policy, theta, zh, hist_blocks, n_samples=n, k=k, seed=r + 1, max_batch=max_batch)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / reps
    rows = theta.shape[0] * len(z_hist) * n * HORIZON_BLOCKS
    return {"P": int(theta.shape[0]), "K": int(len(z_hist)), "n": n, "rows": rows, "s_per_call": dt, "rows_per_s": rows / dt}


def _imagine_worker(job):
    """Spawned process: time repeated rollouts for `seconds` (aggregate-throughput probe)."""
    sys.path.insert(0, str(REPO_ROOT / "experiments"))
    torch.set_num_threads(1)
    model, scaler = load_walker_model(Path(job["model_dir"]), "cuda")
    probe, _ = load_probe(Path(job["probe_path"]), "cuda")
    im = ClosedLoopImaginer(model, scaler, probe, device="cuda", sigma=job["sigma"])
    policy = LinearLatentPolicy.from_state(job["policy_state"], device="cuda")
    zh = torch.as_tensor(job["z_hist"], device="cuda")
    kw = dict(n_samples=job["n"], k=job["k"], max_batch=job["max_batch"])
    im.rollout_policies(policy, job["theta"], zh, job["hist_blocks"], seed=0, **kw)
    torch.cuda.synchronize()
    rows, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < job["seconds"]:
        im.rollout_policies(policy, job["theta"], zh, job["hist_blocks"], seed=rows + 1, **kw)
        rows += job["theta"].shape[0] * len(job["z_hist"]) * job["n"] * HORIZON_BLOCKS
    torch.cuda.synchronize()
    return rows, time.perf_counter() - t0


def time_pycma(n_params: int, popsize: int, diagonal: bool, gens: int) -> float:
    import cma

    rs = np.random.RandomState(0)
    es = cma.CMAEvolutionStrategy(np.zeros(n_params), 0.01, {
        "popsize": popsize, "seed": 1, "randn": rs.randn, "verbose": -9, "CMA_diagonal": diagonal,
        "tolfun": 0, "tolfunhist": 0, "tolx": 0, "tolstagnation": 10**9, "tolflatfitness": 10**9, "maxiter": 10**9})
    rng = np.random.default_rng(0)
    t = time.perf_counter()
    for _ in range(gens):
        X = es.ask()
        es.tell(X, rng.permutation(popsize).astype(float))
    return (time.perf_counter() - t) / gens


def main() -> int:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--noise-run", required=True, help="stage 3 noise-calibration run id (sigma.npy)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    cfg = load_stage_config("stage3")
    sigma_path = REPO_ROOT / "docs" / "evoPlan" / "results" / "stage3-noise" / args.noise_run / "sigma.npy"
    b, sz = cfg.bench, cfg.sizing
    paths = fetch_inputs(cfg.inputs)
    run = EvoRun.create(cfg, "s3-bench", args.run_id)
    print(f"[evo-s3-bench] run {run.run_id}", flush=True)
    device = "cuda"
    quota = cpu_quota_cores()
    model, scaler = load_walker_model(paths["model"], device)
    probe_path = paths["probes"] / "walker_mlp.pt"
    probe, _ = load_probe(probe_path, device)
    sigma = np.load(sigma_path)
    im = ClosedLoopImaginer(model, scaler, probe, device=device, sigma=torch.as_tensor(sigma, dtype=torch.float32, device=device))
    bc = np.load(REPO_ROOT / cfg.bc_policy.file)
    theta_bc = bc["theta"]
    policy = LinearLatentPolicy.from_state(bc, device=device)
    ctx = RenderContext()
    fit = encode_histories(fitness_roots(paths["banks"]), im.encode, ctx)
    tune = encode_histories(tuning_roots(paths["banks"]), im.encode, ctx)
    ev = evaluation_roots(paths["banks"], paths["data"] / "roots.h5", seed=cfg.roots.evaluation_seed,
                          extra_per_episode=cfg.roots.extra_per_episode)
    ctx.close()

    # ---- sigma0 and the fixed-penalty scale, from tuning roots only -----------------------
    zt = torch.as_tensor(tune.z_hist, device=device)
    feats = policy.features(zt[:, 2], zt[:, 1])
    sq = float((feats ** 2).sum(-1).mean())
    sigma0 = float(cfg.sigma0.target_action_rms / np.sqrt(sq + 1.0))
    base = im.rollout_policies(policy, theta_bc[None], zt, tune.hist_blocks)
    m0 = segment_metrics(base["readout"][0, :, 0])
    safe = m0["ret"][~m0["violated"]]
    penalty_scale = float(np.median(safe)) if len(safe) else float("nan")
    rng = np.random.default_rng(cfg.bench.seed)
    sens = {}
    for mult in cfg.sigma0.reference_multiples:
        th = theta_bc[None] + mult * sigma0 * rng.standard_normal((64, theta_bc.size))
        mm = segment_metrics(im.rollout_policies(policy, th, zt, tune.hist_blocks)["readout"][:, :, 0])
        sens[str(mult)] = {"violation_rate": float(mm["violated"].mean()), "return_median": float(np.median(mm["ret"]))}

    # ---- imagination throughput ------------------------------------------------------------
    grid = []
    for P in b.popsizes:
        for K in b.k_roots:
            for n in (1, sz.n_samples_noisy):
                th = theta_bc[None] + sigma0 * rng.standard_normal((P, theta_bc.size))
                sel = rng.choice(len(fit.roots), K, replace=False)
                r = time_rollouts(im, policy, th, fit.z_hist[sel], fit.hist_blocks[sel], n=n, k=0.0 if n == 1 else 1.0,
                                  reps=b.reps, max_batch=b.max_batch)
                grid.append(r)
                print(f"[evo-s3-bench] P {P:4d} K {K:3d} n {n}: {r['rows_per_s']:,.0f} rows/s ({r['s_per_call'] * 1e3:.0f} ms/call)", flush=True)
    single = max(r["rows_per_s"] for r in grid if r["P"] == 64 and r["n"] == sz.n_samples_noisy)
    conc = {}
    sel = rng.choice(len(fit.roots), max(b.k_roots), replace=False)
    th = theta_bc[None] + sigma0 * rng.standard_normal((64, theta_bc.size))
    job = {"model_dir": str(paths["model"]), "probe_path": str(probe_path), "sigma": sigma.astype(np.float32),
           "policy_state": policy.state(), "z_hist": fit.z_hist[sel], "hist_blocks": fit.hist_blocks[sel],
           "theta": th, "n": sz.n_samples_noisy, "k": 1.0, "max_batch": b.max_batch, "seconds": b.concurrency_seconds}
    for c in b.concurrency:
        with ProcessPoolExecutor(c, mp_context=mp.get_context("spawn")) as pool:
            res = list(pool.map(_imagine_worker, [job] * c))
        conc[str(c)] = sum(r for r, _ in res) / max(dt for _, dt in res)
        print(f"[evo-s3-bench] {c} imagination processes: {conc[str(c)]:,.0f} rows/s aggregate", flush=True)
    best_c = max(conc, key=conc.get)
    rate = max(conc[best_c], single)

    # ---- pycma -----------------------------------------------------------------------------
    cma_t = {f"{'diag' if d else 'full'}_{p}": time_pycma(theta_bc.size, p, d, b.cma_generations)
             for d in (True, False) for p in b.popsizes}

    # ---- real throughput and worker-count determinism --------------------------------------
    th = np.stack([theta_bc, theta_bc + sigma0 * rng.standard_normal(theta_bc.size)])
    pairs = [(p, r) for p in range(2) for r in range(len(ev.roots))]
    real_rates, reference = {}, None
    for w in b.real_workers:
        t = time.perf_counter()
        out = evaluate_parallel(th, ev.roots, pairs, model_dir=paths["model"], probe_path=probe_path,
                                policy_state=policy.state(), n_workers=w, device=device, max_envs=b.max_envs,
                                chunk_tasks=int(b.chunk_tasks), encode_batch=int(b.encode_batch))
        dt = time.perf_counter() - t
        real_rates[str(w)] = {"segments_per_s": len(pairs) / dt, "seconds": dt}
        same = None if reference is None else bool(np.array_equal(out["x_velocity"], reference["x_velocity"])
                                                     and np.array_equal(out["readout"], reference["readout"]))
        real_rates[str(w)]["identical_to_first"] = same
        reference = reference or out
        print(f"[evo-s3-bench] real, {w} workers: {len(pairs) / dt:.1f} segments/s (identical: {same})", flush=True)
    real_rate = max(v["segments_per_s"] for v in real_rates.values())
    best_w = max(real_rates, key=lambda k: real_rates[k]["segments_per_s"])

    # ---- run sizes (rules declared in stage3.yaml) -----------------------------------------
    G, seeds = sz.generations, sz.seeds
    samples_6 = 1 + (len(sz.noise_levels) - 1) * sz.n_samples_noisy
    choice, projections = None, {}
    for K in sz.k_roots_candidates:
        rows_5b = sum(sz.popsizes_5b) * K * HORIZON_BLOCKS * G * seeds
        rows_6 = sz.popsize_6 * K * HORIZON_BLOCKS * G * seeds * sz.n_rule_configs * samples_6
        hours = (rows_5b + rows_6) / rate / 3600
        projections[str(K)] = {"rows_5b": rows_5b, "rows_6": rows_6, "imagination_hours": hours}
        if hours <= sz.imagination_budget_hours:
            choice = K
    real_segments = sum(int(v) for v in sz.real_segments.values())
    sizes = {"k_roots": choice, "generations": G, "n_samples_noisy": sz.n_samples_noisy, "sigma0": sigma0,
             "penalty_scale": penalty_scale, "fixed_lambdas": [m * penalty_scale for m in sz.fixed_lambda_multiples],
             "imagination_processes": int(best_c), "real_workers": int(best_w), "cma_diagonal": bool(sz.cma_diagonal),
             "projected_imagination_hours": projections[str(choice)]["imagination_hours"] if choice else None,
             "projected_real_hours": real_segments / real_rate / 3600, "fits_budget": choice is not None}
    report = {"run_id": run.run_id, "stage": "3-bench", "cpu_quota_cores": quota, "visible_cores": os.cpu_count(),
              "sigma0": {"value": sigma0, "mean_feature_sq_norm": sq, "reference_by_multiple": sens},
              "penalty_scale": {"value": penalty_scale, "n_safe_tuning_segments": int(len(safe)),
                                "theta_bc_tuning_violation_rate": float(m0["violated"].mean())},
              "imagination_grid": grid, "imagination_single_rows_per_s": single, "imagination_concurrency": conc,
              "pycma_s_per_generation": cma_t, "real": real_rates, "projections": projections, "run_sizes": sizes,
              "imagined_rows": int(im.counters.imagined_rows), "wall_clock_s": time.time() - t0}
    run.write_json("bench.json", report)
    run.write_json("run_sizes.json", sizes)
    manifest = build_manifest(run_id=run.run_id, kind="evo-s3-bench", seeds={"bench": cfg.bench.seed},
                              data={"noise_run": args.noise_run, "sigma_sha256": file_sha256(sigma_path),
                                    "bc_policy_sha256": file_sha256(REPO_ROOT / cfg.bc_policy.file)},
                              upstream_revisions=input_revisions(paths), metrics=sizes,
                              costs={"real_steps": len(pairs) * 100 * len(b.real_workers), "imagined_rows": report["imagined_rows"]},
                              started_at=t0)
    run.finish(manifest, upload=not args.no_upload,
               readme=f"# {run.run_id}\n\nStage 3 throughput benchmark and run sizes of docs/evoPlan.\n")
    print(f"[evo-s3-bench] sizes: {sizes}; {time.time() - t0:.0f}s", flush=True)
    return 0 if choice is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
