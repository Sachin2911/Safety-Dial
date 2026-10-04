#!/usr/bin/env python3
"""Fit a full-block MLP and compare it with ridge on separate development states.

    uv run python experiments/scripts/evo_mlp_diagnostic.py --run-id ID --no-upload

The configuration is fixed in configs/evo/stage2_mlp.yaml. Validation MSE selects
among checkpoints from three fixed-budget seeds. No refit on validation episodes,
no Gate 0 test roots, and no simulator feedback in model selection.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.threads import apply_torch, pin_threads

pin_threads()
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

apply_torch()
from evo_block_bc_init import ridge
from evo_block_gate0_report import errors
from helpers.evoBlockPolicy import LinearBlockPolicy
from helpers.evoBlockReal import BlockRealExecutor
from helpers.evoImagine import ClosedLoopImaginer, Counters
from helpers.evoInputs import EvoRun, fetch_inputs, input_revisions, load_stage_config
from helpers.evoMlpFit import fit_seed, select_fit
from helpers.evoMlpPolicy import MLPBlockPolicy
from helpers.evoMlpReal import MLPRealExecutor
from helpers.evoRanking import segment_metrics
from helpers.evoRoots import encode_histories, rootset_digest, tuning_roots
from helpers.evoStats import clustered_mean_ci, wilson
from helpers.locoData import RenderContext
from helpers.locoEnv import make_loco_env
from helpers.poseProbes import load_probe
from helpers.runManifest import build_manifest, file_sha256
from helpers.walkerLewm import load_walker_model
from helpers.walkerRules import execute_branch
from helpers.walkerValidation import verify_render_fingerprint


def load_features(cfg):
    source = ROOT / cfg.cache.run
    if not (source / "bc_features.npz").is_file():
        from helpers.hfStore import HFStore
        source = HFStore().download_run("walker2d", cfg.cache.archive_path, revision=cfg.cache.archive_revision)
    for name, expected in [("bc_features.npz", cfg.cache.features_sha256),
                           ("bc_fit.json", cfg.cache.split_sha256),
                           ("bc_policy.npz", cfg.cache.policy_sha256)]:
        if file_sha256(source / name) != expected:
            raise ValueError(f"pinned cache changed: {name}")
    split = json.loads((source / "bc_fit.json").read_text())
    with np.load(source / "bc_features.npz") as f:
        X, Y, E = f["X"], f["Y"], f["episode"]
    with np.load(source / "bc_policy.npz") as f:
        state = {k: f[k] for k in ("action_mean", "action_std", "feature_mean", "feature_std", "policy_kind")}
    val = np.isin(E, split["validation_episodes"])
    if not val.any() or val.all() or set(np.unique(E)) != set(split["episodes"]):
        raise ValueError("cloning source split does not match cached episodes")
    # Match the stored policy's float32 feature transform at deployment.
    X = ((X.astype(np.float32) - state["feature_mean"]) / state["feature_std"]).astype(np.float32)
    return source, X, Y.astype(np.float32), E, val, state, split


def brief(out):
    n = len(out["progress"])
    k = int(out["dense_violated"].sum())
    return {"n": n, "violations": k, "violation_rate": k/n, "violation_wilson": wilson(k, n),
            "progress_mean": float(out["progress"].mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--no-upload", action="store_true")
    a = ap.parse_args()
    start = time.time()
    cfg = load_stage_config("stage2_mlp")
    source, X, Y, E, is_val, state, split = load_features(cfg)
    run = EvoRun.create(cfg, "s2-mlp", a.run_id)
    device = "cuda"
    policy = MLPBlockPolicy((state["action_mean"], state["action_std"]), state["feature_mean"],
                            state["feature_std"], hidden=cfg.policy.hidden, device=device)
    linear = LinearBlockPolicy.from_state(state, device=device)
    Xfit, Yfit, Xval, Yval = [torch.as_tensor(v, device=device) for v in
                             (X[~is_val], Y[~is_val], X[is_val], Y[is_val])]
    print(f"[mlp] {policy.n_params} parameters, {len(Xfit)} fit and {len(Xval)} validation samples", flush=True)
    trials, history, total_updates = [], [], 0
    for seed in cfg.training.seeds:
        best, hist, updates = fit_seed(policy, Xfit, Yfit, Xval, Yval, seed=int(seed),
            epochs=int(cfg.training.epochs), batch_size=int(cfg.training.batch_size),
            lr=float(cfg.training.learning_rate), weight_decay=float(cfg.training.weight_decay),
            progress=lambda r: print(f"[mlp] seed {r['seed']} epoch {r['epoch']}: validation MSE {r['val_mse']:.6f}", flush=True))
        trials.append(best)
        history += hist
        total_updates += updates
        np.savez(run.run_dir / f"mlp_seed_{seed}.npz", theta=best["theta"], **policy.state(),
                 seed=best["seed"], epoch=best["epoch"])
    selected = select_fit(trials)
    theta = selected["theta"]
    np.savez(run.run_dir / "mlp_policy.npz", theta=theta, **policy.state(), seed=selected["seed"], epoch=selected["epoch"])
    # Refit the ridge comparator on precisely the same FIT episodes, without validation data.
    W, b = ridge(X[~is_val].astype(np.float64), Y[~is_val].astype(np.float64), cfg.training.linear_baseline_alpha)
    th_linear = linear.pack(W, b)
    np.savez(run.run_dir / "linear_fit_policy.npz", theta=th_linear, **linear.state())
    with torch.inference_mode():
        pred_mlp = policy.act(theta[None], Xval[None]).cpu().numpy()[0]
        pred_linear = linear.act(th_linear[None], Xval[None]).cpu().numpy()[0]
    offline = {"selected": {k:v for k,v in selected.items() if k != "theta"},
               "trials": [{k:v for k,v in trial.items() if k != "theta"} for trial in trials],
               "linear": errors(pred_linear, Y[is_val]), "mlp": errors(pred_mlp, Y[is_val]),
               "n_fit_samples": int((~is_val).sum()), "n_val_samples": int(is_val.sum()),
               "n_fit_episodes": len(np.unique(E[~is_val])), "n_val_episodes": len(np.unique(E[is_val])),
               "selected_policy_n_params": policy.n_params, "optimizer_updates": total_updates,
               "refit_on_validation": False}
    run.write_json("fit.json", offline)
    run.write_json("training_history.json", history)
    print(f"[mlp] selected seed {selected['seed']} epoch {selected['epoch']}; evaluating development only", flush=True)

    paths = fetch_inputs(cfg.inputs)
    roots = tuning_roots(paths["banks"])
    if len(roots) != cfg.real.expected_roots or any(e % 4 != 0 for e in roots.source_episode):
        raise ValueError("expected exactly the 24 reserved development roots")
    model, scaler = load_walker_model(paths["model"], device)
    if not all(np.array_equal(np.asarray(s), np.asarray(state[k])) for s,k in
               zip(scaler, ["action_mean", "action_std"])):
        # The persisted policy scaler is float32, while fitted upstream scalers are float64.
        if not all(np.array_equal(np.asarray(s,dtype=np.float32), state[k]) for s,k in
                   zip(scaler, ["action_mean", "action_std"])):
            raise ValueError("cache action scaler differs from pinned LeWM")
    probe, _ = load_probe(paths["probes"] / "walker_mlp.pt", device)
    im = ClosedLoopImaginer(model, scaler, probe, device=device)
    ctx = RenderContext()
    verify_render_fingerprint(ctx, paths["data"] / "setA.h5")
    roots = encode_histories(roots, im.encode, ctx)
    real, imagined, counters = {}, {}, Counters()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(int(cfg.real.cpu_threads))
    try:
        for label, cls, pol, th in [("linear", BlockRealExecutor, linear, th_linear),
                                    ("mlp", MLPRealExecutor, policy, theta)]:
            ex = cls(model, scaler, probe, pol, device=device, ctx=ctx,
                     max_envs=int(cfg.real.max_envs), encode_batch=int(cfg.real.encode_batch))
            try:
                real[label] = ex.run(th[None], roots.roots, record_qpos=True, z_hist=roots.z_hist)
                counters.add(ex.counters)
            finally:
                ex.close()
            np.savez_compressed(run.run_dir / f"real_{label}.npz", **real[label])
            imagined[label] = segment_metrics(im.rollout_policies(pol, th[None], roots.z_hist,
                                                                 roots.hist_blocks)["readout"][0,:,0])
    finally:
        torch.set_num_threads(previous_threads)
        ctx.close()
    env = make_loco_env("Walker2d", "v1", render=False, terminate_when_unhealthy=False, six_tuple=True)
    env.reset(seed=0)
    recorded = []
    try:
        for root in roots.roots:
            log = execute_branch(env, root.qpos, root.qvel, root.policy_tape)
            recorded.append({"violated": bool(log.unsafe()["health"]),
                             "progress": float(log.qpos[-1,0]-log.qpos[0,0])})
    finally:
        env.close()
    with h5py.File(paths["data"] / "roots.h5", "r") as f:
        ids = {v:k for k,v in json.loads(f.attrs["policy_ids"]).items()}
        off = f["ep_offset"][:]
        families = [ids[int(f["policy_id"][off[r.episode]])].split("-")[0] for r in roots.roots]
    summary = {label:brief(out) for label,out in real.items()}
    rec_progress = float(np.mean([r["progress"] for r in recorded]))
    for label in real:
        summary[label]["progress_ratio_recorded"] = summary[label]["progress_mean"] / rec_progress
        summary[label]["imagined_violation_rate"] = float(imagined[label]["violated"].mean())
        summary[label]["real_readout_violation_rate"] = float(segment_metrics(real[label]["readout"])["violated"].mean())
    by_family = {}
    for family in sorted(set(families)):
        mask = np.array([v==family for v in families])
        by_family[family] = {label:brief({k:v[mask] for k,v in out.items()}) for label,out in real.items()}
    paired = {"violation_rate_mlp_minus_linear": clustered_mean_ci(
                  real["mlp"]["dense_violated"].astype(float)-real["linear"]["dense_violated"], roots.source_episode),
              "progress_mlp_minus_linear": clustered_mean_ci(real["mlp"]["progress"]-real["linear"]["progress"], roots.source_episode)}
    rows = [{"root_id": r.root_id,"episode":int(r.episode),"step":int(r.step),"family":families[i],
             "recorded":recorded[i], **{label:{"violated":bool(real[label]["dense_violated"][i]),
                 "progress":float(real[label]["progress"][i]),
                 "first_unsafe_step":int(real[label]["dense_first_step"][i]),
                 "imagined_violated":bool(imagined[label]["violated"][i])} for label in real}}
             for i,r in enumerate(roots.roots)]
    report = {"run_id":run.run_id,"role":"development diagnostic only; Gate 0 not run",
              "offline":offline,"development":summary,"by_family":by_family,"paired":paired,
              "recorded":{"n":len(recorded),"violations":sum(r["violated"] for r in recorded),"progress_mean":rec_progress},
              "rootset_digest":rootset_digest(roots),"wall_clock_s":time.time()-start}
    run.write_json("diagnostic.json",report)
    run.write_json("development_rows.json",rows)
    files=[Path(__file__),ROOT/'experiments/helpers/evoMlpPolicy.py',ROOT/'experiments/helpers/evoMlpFit.py',
           ROOT/'experiments/helpers/evoMlpReal.py',ROOT/'configs/evo/stage2_mlp.yaml']
    for path in files:
        dest=run.run_dir/'source'/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest)
    upstream=input_revisions(paths)
    upstream['cloning_cache']={'repo_id':'Sachioster/safetydial-walker2d',
                               'path':cfg.cache.archive_path,'revision':cfg.cache.archive_revision}
    manifest=build_manifest(run_id=run.run_id,kind='evo-s2-mlp-diagnostic',
        seeds={"training":list(cfg.training.seeds)},upstream_revisions=upstream,started_at=start,
        data={"cache_sha256":cfg.cache.features_sha256,"split_sha256":cfg.cache.split_sha256,
              "mlp_policy_sha256":file_sha256(run.run_dir/'mlp_policy.npz')},
        metrics={"development":summary,"offline":offline},
        costs={"new_real_steps":int(counters.real_steps+len(recorded)*100),
               "real_executor":counters.as_dict(),"imagined":im.counters.as_dict(),
               "history_renders":len(roots)*3,"optimizer_updates":total_updates,
               "cached_bc_frames_reused":int(split['n_frames_rendered']),
               "upstream_costs_from_cache_manifest":json.loads((source/'manifest.json').read_text())['costs']},
        extra={"implementation_sha256":{str(p.relative_to(ROOT)):file_sha256(p) for p in files},
               "gate0_run":False})
    run.finish(manifest,upload=not a.no_upload,
               readme=f"# {run.run_id}\n\nFull-block MLP imitation diagnostic on development states only. No Gate 0 rerun.\n")
    print(json.dumps({"offline_mse_linear":offline['linear']['full_block_mse'],
                      "offline_mse_mlp":offline['mlp']['full_block_mse'],"development":summary,
                      "by_family":by_family,"wall_clock_s":report['wall_clock_s']},indent=2),flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
