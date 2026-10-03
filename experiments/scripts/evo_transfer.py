#!/usr/bin/env python3
"""Stage 4 of the evolution study (docs/evoPlan/README.md): transfer check and Gate 1.

About 100 policies spanning good to bad (theta_BC, Gaussian perturbations of it at four
scales, and interpolations of it toward zero) are each scored on the 256 evaluation start
states in imagination (k = 0) and in the real simulator (closed loop: render, encode, act,
10 steps, repeat; dense truth at every step). Gate 1 (provisional, configs/evo/stage4.yaml)
asks whether imagined and real rankings agree: Spearman correlation across policies of the
imagined and real mean return, and of the imagined and real violation rate, each at least the
threshold with a 95% source-episode cluster-bootstrap interval above zero.

    uv run python experiments/scripts/evo_transfer.py [--run-id ID] [--no-upload]
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
from helpers.evoRanking import segment_metrics  # noqa: E402
from helpers.evoReal import evaluate_parallel  # noqa: E402
from helpers.evoRoots import encode_histories, evaluation_roots, rootset_digest  # noqa: E402
from helpers.evoStats import clustered_spearman, wilson  # noqa: E402
from helpers.locoData import RenderContext  # noqa: E402
from helpers.poseProbes import load_probe  # noqa: E402
from helpers.runManifest import build_manifest, file_sha256  # noqa: E402
from helpers.walkerLewm import load_walker_model  # noqa: E402


def transfer_policies(theta_bc: np.ndarray, sigma0: float, cfg) -> tuple[np.ndarray, list[dict]]:
    """theta_BC, per-scale Gaussian perturbations and interpolations toward zero."""
    rng = np.random.default_rng(cfg.policies.seed)
    thetas, meta = [theta_bc], [{"kind": "bc"}]
    for mult in cfg.policies.perturbation_multiples:
        for i in range(cfg.policies.per_scale):
            thetas.append(theta_bc + mult * sigma0 * rng.standard_normal(theta_bc.size))
            meta.append({"kind": "perturbation", "multiple": float(mult), "index": i})
    for a in np.linspace(*cfg.policies.interpolation_range, cfg.policies.n_interpolations):
        thetas.append(float(a) * theta_bc)
        meta.append({"kind": "interpolation", "alpha": float(a)})
    return np.stack(thetas), meta


def main() -> int:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    cfg = load_stage_config("stage4")
    paths = fetch_inputs(cfg.inputs)
    run = EvoRun.create(cfg, "s4", args.run_id)
    print(f"[evo-s4] run {run.run_id}", flush=True)
    device = "cuda"
    model, scaler = load_walker_model(paths["model"], device)
    probe_path = paths["probes"] / "walker_mlp.pt"
    probe, _ = load_probe(probe_path, device)
    im = ClosedLoopImaginer(model, scaler, probe, device=device)
    bc = np.load(REPO_ROOT / cfg.bc_policy.file)
    policy = LinearLatentPolicy.from_state(bc, device=device)
    thetas, meta = transfer_policies(bc["theta"], float(cfg.run_sizes.sigma0), cfg)
    np.savez(run.run_dir / "policies.npz", theta=thetas)
    ctx = RenderContext()
    ev = encode_histories(evaluation_roots(paths["banks"], paths["data"] / "roots.h5", seed=cfg.roots.evaluation_seed,
                                           extra_per_episode=cfg.roots.extra_per_episode), im.encode, ctx)
    ctx.close()
    clusters = ev.source_episode

    # ---- imagination (k = 0) on the evaluation roots ---------------------------------------
    imag = im.rollout_policies(policy, thetas, torch.as_tensor(ev.z_hist, device=device), ev.hist_blocks)
    mi = segment_metrics(imag["readout"][:, :, 0])  # (P, R)
    # ---- real simulator on the same roots ----------------------------------------------------
    pairs = [(p, r) for p in range(len(thetas)) for r in range(len(ev.roots))]
    real = evaluate_parallel(thetas, ev.roots, pairs, model_dir=paths["model"], probe_path=probe_path,
                             policy_state=policy.state(), n_workers=int(cfg.run_sizes.real_workers), device=device,
                             max_envs=cfg.evaluation.max_envs,
                             chunk_tasks=int(cfg.evaluation.chunk_tasks),
                             encode_batch=int(cfg.evaluation.encode_batch))
    P, R = len(thetas), len(ev.roots)
    real_viol = real["dense_violated"].reshape(P, R).astype(float)
    real_ret = real["progress"].reshape(P, R)
    readout = segment_metrics(real["readout"].reshape(P, R, -1, 3))
    imag_viol, imag_ret = mi["violated"].astype(float), mi["ret"]

    # ---- Gate 1 ---------------------------------------------------------------------------
    g = cfg.gate1
    rho_ret = clustered_spearman(imag_ret, real_ret, clusters, n_boot=g.n_boot, seed=g.bootstrap_seed)
    rho_viol = clustered_spearman(imag_viol, real_viol, clusters, n_boot=g.n_boot, seed=g.bootstrap_seed)
    ok_ret = bool(rho_ret["point"] >= g.min_spearman and rho_ret["lo"] > 0)
    ok_viol = bool(rho_viol["point"] >= g.min_spearman and rho_viol["lo"] > 0)
    gate = {"pass": ok_ret and ok_viol, "provisional": True, "return": {**rho_ret, "pass": ok_ret},
            "violation_rate": {**rho_viol, "pass": ok_viol},
            "thresholds": {"min_spearman": g.min_spearman, "interval_lower_bound_above": 0.0}}
    per_policy = []
    for p in range(P):
        k_real = int(real_viol[p].sum())
        per_policy.append({**meta[p], "imagined_violation_rate": float(imag_viol[p].mean()), "imagined_return": float(imag_ret[p].mean()),
                           "real_violation_rate": float(real_viol[p].mean()), "real_violation_wilson": wilson(k_real, R),
                           "real_return": float(real_ret[p].mean()), "readout_violation_rate": float(readout["violated"][p].mean()),
                           "readout_return": float(readout["ret"][p].mean())})
    report = {"run_id": run.run_id, "stage": 4, "gate1": gate, "n_policies": P, "n_roots": R,
              "evaluation_roots_digest": rootset_digest(ev), "policies": per_policy,
              "counters": {"imagined_rows": int(im.counters.imagined_rows), "real": real["counters"].as_dict()},
              "wall_clock_s": time.time() - t0}
    run.write_json("gate1.json", report)
    np.savez_compressed(run.run_dir / "scores.npz", imag_viol=imag_viol, imag_ret=imag_ret, real_viol=real_viol,
                        real_ret=real_ret, readout_viol=readout["violated"], readout_ret=readout["ret"], clusters=clusters)
    np.savez_compressed(run.results_dir / "scores.npz", imag_viol=imag_viol, imag_ret=imag_ret, real_viol=real_viol,
                        real_ret=real_ret, readout_viol=readout["violated"], readout_ret=readout["ret"], clusters=clusters)
    manifest = build_manifest(run_id=run.run_id, kind="evo-s4", seeds={"policies": cfg.policies.seed, "bootstrap": g.bootstrap_seed},
                              data={"bc_policy_sha256": file_sha256(REPO_ROOT / cfg.bc_policy.file)},
                              upstream_revisions=input_revisions(paths), metrics={"gate1": gate},
                              costs={"real_steps": P * R * 100, "imagined_rows": int(im.counters.imagined_rows)}, started_at=t0)
    run.finish(manifest, upload=not args.no_upload,
               readme=f"# {run.run_id}\n\nStage 4 transfer check of docs/evoPlan: {P} policies scored in imagination and in the real simulator on the 256 evaluation roots.\n")
    print(f"[evo-s4] Gate 1 {'PASS' if gate['pass'] else 'FAIL'}: return rho {rho_ret['point']:.3f} [{rho_ret['lo']:.3f}, {rho_ret['hi']:.3f}], "
          f"violation rho {rho_viol['point']:.3f} [{rho_viol['lo']:.3f}, {rho_viol['hi']:.3f}]; {time.time() - t0:.0f}s", flush=True)
    return 0 if gate["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
