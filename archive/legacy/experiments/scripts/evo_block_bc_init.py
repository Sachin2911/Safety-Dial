#!/usr/bin/env python3
"""Stage 2 of the evolution study (docs/evoPlan/README.md): theta_BC and Gate 0.

1. Render and encode set-A frames of the declared policies every `frame_stride` steps. The
   feature at step t is [z_t, z_t - z_(t-10)] and the target is all 10 sequential actions of the
   following block (steps t..t+9). Whole episodes are split into fit and validation sets; the
   ridge strength with the lowest validation error of the clipped prediction is chosen and the
   policy is refit on all selected episodes. The feature standardisation is fitted on the same
   frames and frozen into the policy. This is theta_BC.
2. Run theta_BC closed loop on the 256 evaluation start states in the real simulator (render,
   encode, act, 10 steps, repeat), replay the recorded policies' own tapes from the same
   snapshots, and score theta_BC in imagination (k = 0) for reference.
3. Gate 0 (provisional thresholds in configs/evo/stage2_block.yaml).

    uv run python experiments/scripts/evo_block_bc_init.py [--run-id ID] [--no-upload]
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

pin_threads()
import h5py  # noqa: E402
import hdf5plugin  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402

apply_torch()

from helpers.evoImagine import ClosedLoopImaginer  # noqa: E402
from helpers.evoInputs import EvoRun, fetch_inputs, input_revisions, load_stage_config  # noqa: E402
from helpers.evoBlockPolicy import FEATURE_DIM, LinearBlockPolicy  # noqa: E402
from helpers.evoRanking import segment_metrics  # noqa: E402
from helpers.evoBlockReal import evaluate_blocks_parallel  # noqa: E402
from helpers.evoRoots import encode_histories, evaluation_roots, rootset_digest  # noqa: E402
from helpers.evoStats import clustered_mean_ci, wilson  # noqa: E402
from helpers.locoData import RenderContext  # noqa: E402
from helpers.locoEnv import make_loco_env  # noqa: E402
from helpers.poseProbes import load_probe  # noqa: E402
from helpers.runManifest import build_manifest, file_sha256  # noqa: E402
from helpers.walkerLewm import FRAMESKIP, load_walker_model  # noqa: E402
from helpers.walkerRules import execute_branch  # noqa: E402
from helpers.walkerValidation import verify_render_fingerprint  # noqa: E402


def select_episodes(h5_path: Path, names, n: int, seed: int, val_fraction: float):
    """Episodes of the named policies, a seeded subset, split by whole episode."""
    with h5py.File(h5_path, "r") as f:
        ids = json.loads(f.attrs["policy_ids"])
        off, ln = f["ep_offset"][:].astype(np.int64), f["ep_len"][:].astype(np.int64)
        first = f["policy_id"][:][off]
    wanted = np.array([ids[name] for name in names])
    eligible = np.where(np.isin(first, wanted) & (ln > 2 * FRAMESKIP))[0]
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(eligible, size=min(n, len(eligible)), replace=False))
    n_val = int(round(val_fraction * len(chosen)))
    val = np.sort(chosen[rng.permutation(len(chosen))[:n_val]])
    return chosen, val, {"n_eligible": int(len(eligible)), "policy_ids": {k: ids[k] for k in names}}


def bc_frames(h5_path: Path, episodes, ctx: RenderContext, encode, stride: int):
    """Features (N, 384) raw, targets (N, 60) time-major next-block actions, episode (N,)."""
    X, Y, E = [], [], []
    frames_rendered = 0
    with h5py.File(h5_path, "r") as f:
        off, ln = f["ep_offset"][:].astype(np.int64), f["ep_len"][:].astype(np.int64)
        for episode_index, e in enumerate(episodes):
            if episode_index % 25 == 0:
                print(f"[evo-s2-block] encoding episode {episode_index + 1}/{len(episodes)}", flush=True)
            a, b = int(off[e]), int(off[e] + ln[e])
            qpos, qvel, act = f["qpos"][a:b], f["qvel"][a:b], f["action"][a:b].astype(np.float64)
            rows = np.arange(0, b - a, stride)
            z = encode(ctx.render_many(qpos[rows], qvel[rows])).cpu().numpy().astype(np.float64)
            frames_rendered += len(rows)
            pos = {int(t): i for i, t in enumerate(rows)}
            for t in rows:
                t = int(t)
                if t - FRAMESKIP < 0 or t + FRAMESKIP > b - a:
                    continue
                block = act[t : t + FRAMESKIP]
                if not np.isfinite(block).all():
                    continue
                zt, zp = z[pos[t]], z[pos[t - FRAMESKIP]]
                X.append(np.concatenate([zt, zt - zp]))
                Y.append(block.reshape(-1))
                E.append(e)
    return np.asarray(X), np.asarray(Y), np.asarray(E), frames_rendered


def ridge(X, Y, alpha: float):
    """Ridge with an unpenalised intercept: returns W (60, F), b (60,)."""
    xm, ym = X.mean(0), Y.mean(0)
    Xc, Yc = X - xm, Y - ym
    W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ Yc).T
    return W, ym - W @ xm


def clipped_mse(X, Y, W, b) -> float:
    return float(np.mean((np.clip(X @ W.T + b, -1.0, 1.0) - Y) ** 2))


def main() -> int:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    cfg = load_stage_config("stage2_block")
    np.random.seed(int(cfg.bc.episode_seed))
    torch.manual_seed(int(cfg.bc.episode_seed))
    paths = fetch_inputs(cfg.inputs)
    run = EvoRun.create(cfg, "s2-block", args.run_id)
    print(f"[evo-s2-block] run {run.run_id}", flush=True)
    device = "cuda"
    model, scaler = load_walker_model(paths["model"], device)
    probe_path = paths["probes"] / "walker_mlp.pt"
    probe, _ = load_probe(probe_path, device)
    im = ClosedLoopImaginer(model, scaler, probe, device=device)
    ctx = RenderContext()
    setA = paths["data"] / cfg.bc.source
    verify_render_fingerprint(ctx, setA)

    # ---- 1. behaviour cloning ------------------------------------------------------------
    chosen, val_eps, sel_info = select_episodes(setA, list(cfg.bc.policies), cfg.bc.n_episodes,
                                                cfg.bc.episode_seed, cfg.bc.val_fraction)
    X, Y, E, n_frames = bc_frames(setA, chosen, ctx, im.encode, cfg.bc.frame_stride)
    np.savez(run.run_dir / "bc_features.npz", X=X, Y=Y, episode=E)
    is_val = np.isin(E, val_eps)
    f_mean, f_std = X.mean(0), X.std(0) + 1e-6
    Xs = (X - f_mean) / f_std
    sweep = []
    for alpha in cfg.bc.ridge_alphas:
        W, b = ridge(Xs[~is_val], Y[~is_val], float(alpha))
        sweep.append({"alpha": float(alpha), "val_mse": clipped_mse(Xs[is_val], Y[is_val], W, b),
                      "fit_mse": clipped_mse(Xs[~is_val], Y[~is_val], W, b)})
    best = min(sweep, key=lambda r: (r["val_mse"], -r["alpha"]))
    W, b = ridge(Xs, Y, best["alpha"]) if cfg.bc.refit_on_all else ridge(Xs[~is_val], Y[~is_val], best["alpha"])
    theta_bc = LinearBlockPolicy.pack(W, b)
    policy = LinearBlockPolicy(scaler, f_mean, f_std, device=device)
    pred = np.clip(Xs @ W.T + b, -1, 1)
    r2 = 1 - ((pred - Y) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
    bc_report = {"episodes": chosen.tolist(), "validation_episodes": val_eps.tolist(), "selection": sel_info,
                 "n_frames_rendered": int(n_frames), "n_samples": int(len(X)), "n_val_samples": int(is_val.sum()),
                 "alpha_sweep": sweep, "alpha": best["alpha"], "r2_per_action_all": r2.tolist(),
                 "feature_dim": FEATURE_DIM, "output_dim": 60, "n_params": policy.n_params,
                 "theta_norm": float(np.linalg.norm(theta_bc))}
    print(f"[evo-s2-block] BC: {len(X)} samples from {len(chosen)} episodes, alpha {best['alpha']}, val mse {best['val_mse']:.4f}", flush=True)
    # Learned weights stay in the private HF bundle, never in git.
    np.savez(run.run_dir / "bc_policy.npz", theta=theta_bc, **policy.state(), alpha=best["alpha"])
    run.write_json("bc_fit.json", bc_report)
    print("[evo-s2-block] starting real evaluation", flush=True)

    # ---- 2. evaluation roots: theta_BC real, recorded tapes, theta_BC imagined -----------
    ev = evaluation_roots(paths["banks"], paths["data"] / "roots.h5", seed=cfg.roots.evaluation_seed,
                          extra_per_episode=cfg.roots.extra_per_episode)
    if len(ev.roots) != cfg.roots.n_evaluation:
        raise ValueError(f"expected {cfg.roots.n_evaluation} evaluation roots, got {len(ev.roots)}")
    clusters = ev.source_episode
    pairs = [(0, r) for r in range(len(ev.roots))]
    real = evaluate_blocks_parallel(theta_bc[None], ev.roots, pairs, model_dir=paths["model"], probe_path=probe_path,
                             policy_state=policy.state(), n_workers=cfg.evaluation.n_workers, device=device,
                             max_envs=cfg.evaluation.max_envs,
                             chunk_tasks=int(cfg.evaluation.chunk_tasks),
                             encode_batch=int(cfg.evaluation.encode_batch))
    env = make_loco_env("Walker2d", "v1", render=False, terminate_when_unhealthy=False, six_tuple=True)
    env.reset(seed=0)
    rec_viol, rec_prog = [], []
    for root in ev.roots:
        log = execute_branch(env, root.qpos, root.qvel, root.policy_tape)
        rec_viol.append(log.unsafe()["health"])
        rec_prog.append(float(log.qpos[-1, 0] - log.qpos[0, 0]))
    env.close()
    rec_viol, rec_prog = np.asarray(rec_viol), np.asarray(rec_prog)
    print("[evo-s2-block] real evaluation complete; scoring imagination", flush=True)
    ev = encode_histories(ev, im.encode, ctx)
    imag = im.rollout_policies(policy, theta_bc[None], torch.as_tensor(ev.z_hist, device=device), ev.hist_blocks)
    imag_m = segment_metrics(imag["readout"][0, :, 0])
    readout_m = segment_metrics(real["readout"])
    ctx.close()

    # ---- 3. Gate 0 -----------------------------------------------------------------------
    g = cfg.gate0
    n = len(ev.roots)
    k_bc, k_rec = int(real["dense_violated"].sum()), int(rec_viol.sum())
    allowed = max(g.max_violation_ratio * k_rec, g.min_resolvable_count)
    prog_bc, prog_rec = float(real["progress"].mean()), float(rec_prog.mean())
    violation_ok = bool(k_bc <= allowed)
    progress_ok = bool(prog_bc >= g.min_progress_ratio * prog_rec)
    gate = {"pass": violation_ok and progress_ok, "provisional": True, "violation_ok": violation_ok,
            "progress_ok": progress_ok, "n": n, "violations_bc": k_bc, "violations_recorded": k_rec,
            "violations_allowed": allowed, "rate_bc": k_bc / n, "rate_bc_wilson": wilson(k_bc, n),
            "rate_recorded": k_rec / n, "rate_recorded_wilson": wilson(k_rec, n),
            "progress_bc_mean": prog_bc, "progress_recorded_mean": prog_rec,
            "progress_ratio": prog_bc / prog_rec if prog_rec else float("nan"),
            "progress_bc_ci": clustered_mean_ci(real["progress"], clusters),
            "progress_recorded_ci": clustered_mean_ci(rec_prog, clusters),
            "thresholds": {"max_violation_ratio": g.max_violation_ratio, "min_resolvable_count": g.min_resolvable_count,
                           "min_progress_ratio": g.min_progress_ratio}}
    reference = {"imagined_k0": {"violation_rate": float(imag_m["violated"].mean()), "return_mean": float(imag_m["ret"].mean())},
                 "real_readout": {"violation_rate": float(readout_m["violated"].mean()), "return_mean": float(readout_m["ret"].mean())},
                 "endpoint_truth_violation_rate": float((real["endpoint_clearance"] <= 0).any(1).mean())}
    rows = [{"root_id": r.root_id, "episode": int(r.episode), "step": int(r.step),
             "bc_dense_violated": bool(real["dense_violated"][i]), "bc_progress": float(real["progress"][i]),
             "bc_first_unsafe_step": int(real["dense_first_step"][i]),
             "recorded_violated": bool(rec_viol[i]), "recorded_progress": float(rec_prog[i]),
             "bc_imagined_violated": bool(imag_m["violated"][i]), "bc_imagined_return": float(imag_m["ret"][i]),
             "bc_readout_violated": bool(readout_m["violated"][i])} for i, r in enumerate(ev.roots)]
    report = {"run_id": run.run_id, "stage": "2-block", "interpretation": dict(cfg.interpretation), "gate0": gate, "bc": bc_report, "reference": reference,
              "evaluation_roots_digest": rootset_digest(ev), "wall_clock_s": time.time() - t0}
    run.write_json("gate0.json", report)
    run.write_json("evaluation_rows.json", rows)
    counters = real["counters"].as_dict()
    data_stats = json.loads((paths["data"] / "manifest.json").read_text())["data"]["stats"]
    costs = {"setA_collection_steps_reused": int(data_stats["A_competent"]["n_steps"] + data_stats["A_early"]["n_steps"]),
             "probe_collection_steps_reused": int(data_stats["probe"]["n_steps"]),
             "root_collection_steps_reused": int(data_stats["roots"]["n_steps"]),
             "bc_data_source": "subset of existing setA; no new collection","real_steps_bc": int(n * 100), "real_steps_recorded_replay": int(n * 100),
             "bc_frames_rendered": int(n_frames), "imagined_history_frames_rendered": int(n * 3),
             "main_process_encodes": int(im.counters.encodes), "imagined_rows": int(im.counters.imagined_rows),
             "real_executor_counters": counters}
    manifest = build_manifest(run_id=run.run_id, kind="evo-s2-block",
                              seeds={"episode_seed": cfg.bc.episode_seed, "evaluation_seed": cfg.roots.evaluation_seed},
                              data={"setA_sha256": file_sha256(setA), "bc_policy_sha256": file_sha256(run.run_dir / "bc_policy.npz")},
                              upstream_revisions=input_revisions(paths), metrics={"gate0": gate}, costs=costs, started_at=t0,
                              extra={"implementation_sha256": {str(p.relative_to(REPO_ROOT)): file_sha256(p)
                                     for p in [Path(__file__), REPO_ROOT / "experiments/helpers/evoBlockPolicy.py",
                                               REPO_ROOT / "experiments/helpers/evoBlockReal.py",
                                               REPO_ROOT / "configs/evo/stage2_block.yaml"]}})
    run.finish(manifest, upload=not args.no_upload,
               readme=f"# {run.run_id}\n\nStage 2 of docs/evoPlan: behaviour-cloned sequential-block linear latent policy theta_BC (bc_policy.npz) and Gate 0.\n")
    verdict = "PASS" if gate["pass"] else "FAIL"
    print(f"[evo-s2-block] Gate 0 {verdict}: violations {k_bc}/{n} (allowed {allowed:g}, recorded {k_rec}); "
          f"progress {prog_bc:.3f} m vs recorded {prog_rec:.3f} m (ratio {gate['progress_ratio']:.2f}); {time.time() - t0:.0f}s", flush=True)
    return 0 if gate["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
