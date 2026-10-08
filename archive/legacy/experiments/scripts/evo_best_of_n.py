#!/usr/bin/env python3
"""Stage 5a of the evolution study (docs/evoPlan/README.md): pure selection and Gate 2.

Per seed, 4,096 policies are drawn around theta_BC and scored once in imagination (k = 0) on
K fitness roots common to all of them. For each N, the pick is the best of the first N under
Deb's rule (lower imagined violation rate, then higher imagined return, then lower index), so
N = 1 is an unselected draw. Every distinct pick is evaluated on the 256 evaluation roots in
imagination (k = 0) and in the real simulator. This isolates selection pressure from the
optimiser's dynamics. Gate 2 (provisional, configs/evo/stage5.yaml): the real-minus-imagined
violation gap at the largest N exceeds the gap at N = 1, with a paired source-episode cluster
bootstrap interval of the difference (mean over seeds) excluding zero.

    uv run python experiments/scripts/evo_best_of_n.py [--run-id ID] [--no-upload]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from helpers.threads import apply_torch, pin_threads  # noqa: E402

pin_threads()
import numpy as np  # noqa: E402
import torch  # noqa: E402

apply_torch()

from helpers.evoImagine import ClosedLoopImaginer  # noqa: E402
from helpers.evoInputs import EvoRun, fetch_inputs, input_revisions, load_stage_config  # noqa: E402
from helpers.evoPolicy import LinearLatentPolicy  # noqa: E402
from helpers.evoRanking import candidate_stats, deb_ranks, segment_metrics  # noqa: E402
from helpers.evoReal import evaluate_parallel  # noqa: E402
from helpers.evoRoots import encode_histories, evaluation_roots, fitness_roots, rootset_digest  # noqa: E402
from helpers.evoStats import paired_gap_difference, wilson  # noqa: E402
from helpers.locoData import RenderContext  # noqa: E402
from helpers.poseProbes import load_probe  # noqa: E402
from helpers.runManifest import build_manifest, file_sha256  # noqa: E402
from helpers.walkerLewm import load_walker_model  # noqa: E402


def best_of_first(p: np.ndarray, mean_ret: np.ndarray, n: int) -> int:
    """Index of Deb's best among candidates 0..n-1 (ties to the lower index)."""
    return int(np.argmin(deb_ranks(p[:n], mean_ret[:n])))


def main() -> int:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    cfg = load_stage_config("stage5")
    c = cfg.best_of_n
    paths = fetch_inputs(cfg.inputs)
    run = EvoRun.create(cfg, "s5a", args.run_id)
    print(f"[evo-s5a] run {run.run_id}", flush=True)
    device = "cuda"
    model, scaler = load_walker_model(paths["model"], device)
    probe_path = paths["probes"] / "walker_mlp.pt"
    probe, _ = load_probe(probe_path, device)
    im = ClosedLoopImaginer(model, scaler, probe, device=device)
    bc = np.load(REPO_ROOT / cfg.bc_policy.file)
    theta_bc = bc["theta"]
    policy = LinearLatentPolicy.from_state(bc, device=device)
    ctx = RenderContext()
    fit = encode_histories(fitness_roots(paths["banks"]), im.encode, ctx)
    ev = encode_histories(evaluation_roots(paths["banks"], paths["data"] / "roots.h5", seed=cfg.roots.evaluation_seed,
                                           extra_per_episode=cfg.roots.extra_per_episode), im.encode, ctx)
    ctx.close()
    clusters, R = ev.source_episode, len(ev.roots)
    sigma = float(c.sigma_multiple) * float(cfg.run_sizes.sigma0)
    k_roots = int(cfg.run_sizes.k_roots)
    Ns = [int(n) for n in c.n_values]

    picks, records = [], []
    for s in c.seeds:
        rng = np.random.default_rng([int(c.seed_base), int(s)])
        thetas = theta_bc[None] + sigma * rng.standard_normal((int(c.n_policies), theta_bc.size))
        sel = np.sort(rng.choice(len(fit.roots), k_roots, replace=False))
        out = im.rollout_policies(policy, thetas, torch.as_tensor(fit.z_hist[sel], device=device), fit.hist_blocks[sel],
                                  max_batch=int(c.max_batch))
        m = segment_metrics(out["readout"][:, :, 0])
        st = candidate_stats(m["violated"], m["ret"])
        for n in Ns:
            i = best_of_first(st["p"], st["mean_ret"], n)
            records.append({"seed": int(s), "N": n, "index": i, "fitness_violation_rate": float(st["p"][i]),
                            "fitness_return": float(st["mean_ret"][i]), "fitness_roots": sel.tolist()})
            picks.append(thetas[i])
        print(f"[evo-s5a] seed {s}: picks {[r['index'] for r in records[-len(Ns):]]}", flush=True)
    picks = np.stack(picks)
    uniq, inverse = np.unique(picks, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    np.savez(run.run_dir / "picks.npz", theta=uniq, inverse=inverse)

    # ---- evaluation roots: imagination (k = 0) and the real simulator ---------------------
    imag = segment_metrics(im.rollout_policies(policy, uniq, torch.as_tensor(ev.z_hist, device=device), ev.hist_blocks)["readout"][:, :, 0])
    pairs = [(p, r) for p in range(len(uniq)) for r in range(R)]
    real = evaluate_parallel(uniq, ev.roots, pairs, model_dir=paths["model"], probe_path=probe_path,
                             policy_state=policy.state(), n_workers=int(cfg.run_sizes.real_workers), device=device,
                             max_envs=cfg.evaluation.max_envs,
                             chunk_tasks=int(cfg.evaluation.chunk_tasks),
                             encode_batch=int(cfg.evaluation.encode_batch))
    U = len(uniq)
    real_viol = real["dense_violated"].reshape(U, R).astype(float)
    real_ret = real["progress"].reshape(U, R)
    readout = segment_metrics(real["readout"].reshape(U, R, -1, 3))
    for rec, u in zip(records, inverse, strict=True):
        k_real = int(real_viol[u].sum())
        rec.update({"eval_imagined_violation_rate": float(imag["violated"][u].mean()), "eval_imagined_return": float(imag["ret"][u].mean()),
                    "eval_real_violation_rate": float(real_viol[u].mean()), "eval_real_violation_wilson": wilson(k_real, R),
                    "eval_real_return": float(real_ret[u].mean()), "eval_readout_violation_rate": float(readout["violated"][u].mean()),
                    "gap": float(real_viol[u].mean() - imag["violated"][u].mean())})

    # ---- Gate 2 ----------------------------------------------------------------------------
    g = cfg.gate2
    hi, lo = max(Ns), min(Ns)
    idx = {(r["seed"], r["N"]): u for r, u in zip(records, inverse, strict=True)}
    seeds = [int(s) for s in c.seeds]
    arr = {n: (np.stack([real_viol[idx[(s, n)]] for s in seeds]), np.stack([imag["violated"][idx[(s, n)]].astype(float) for s in seeds]))
           for n in (hi, lo)}
    diff = paired_gap_difference(arr[hi][0], arr[hi][1], arr[lo][0], arr[lo][1], clusters, n_boot=g.n_boot, seed=g.bootstrap_seed)
    gate = {"pass": bool(diff["point"] > 0 and diff["lo"] > 0), "provisional": True, "N_high": hi, "N_low": lo, **diff}
    curve = {}
    for n in Ns:
        rows = [r for r in records if r["N"] == n]
        curve[str(n)] = {k: float(np.mean([r[k] for r in rows])) for k in
                         ("fitness_violation_rate", "fitness_return", "eval_imagined_violation_rate", "eval_imagined_return",
                          "eval_real_violation_rate", "eval_real_return", "eval_readout_violation_rate", "gap")}
    report = {"run_id": run.run_id, "stage": "5a", "gate2": gate, "curve": curve, "records": records,
              "sigma": sigma, "k_roots": k_roots, "n_unique_picks": int(U), "evaluation_roots_digest": rootset_digest(ev),
              "counters": {"imagined_rows": int(im.counters.imagined_rows), "real": real["counters"].as_dict()},
              "wall_clock_s": time.time() - t0}
    run.write_json("best_of_n.json", report)
    manifest = build_manifest(run_id=run.run_id, kind="evo-s5a", seeds={"seed_base": int(c.seed_base), "seeds": seeds, "bootstrap": g.bootstrap_seed},
                              data={"bc_policy_sha256": file_sha256(REPO_ROOT / cfg.bc_policy.file)},
                              upstream_revisions=input_revisions(paths), metrics={"gate2": gate, "curve": curve},
                              costs={"real_steps": U * R * 100, "imagined_rows": int(im.counters.imagined_rows)}, started_at=t0)
    run.finish(manifest, upload=not args.no_upload,
               readme=f"# {run.run_id}\n\nStage 5a (best of N) of docs/evoPlan: picks.npz holds the distinct picks; best_of_n.json the curve and Gate 2.\n")
    print(f"[evo-s5a] Gate 2 {'PASS' if gate['pass'] else 'FAIL'}: gap difference {diff['point']:.4f} [{diff['lo']:.4f}, {diff['hi']:.4f}] "
          f"(gap N={hi} {diff['gap_hi']:.4f}, N={lo} {diff['gap_lo']:.4f}); {time.time() - t0:.0f}s", flush=True)
    return 0 if gate["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
