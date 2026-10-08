#!/usr/bin/env python3
"""Compare fixed-architecture MLPs with matched budgets and broader demonstrations.

    uv run python experiments/scripts/evo_coverage_diagnostic.py --run-id ID --no-upload

Both arms use the same family-balanced validation pool for checkpoint selection.
Development evaluation follows evo_mlp_diagnostic at e8f650b, with the same physics
and feature protocol. Test-role Gate 0 states never enter this diagnostic.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault('MUJOCO_GL','egl')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
from helpers.threads import apply_torch,pin_threads

pin_threads()
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

apply_torch()
from evo_block_bc_init import bc_frames,select_episodes
from evo_block_gate0_report import errors
from evo_mlp_diagnostic import brief,load_features
from helpers.evoCoverage import coverage_indices,fit_coverage_seed
from helpers.evoImagine import ClosedLoopImaginer,Counters
from helpers.evoInputs import EvoRun,fetch_inputs,input_revisions,load_stage_config
from helpers.evoMlpFit import select_fit
from helpers.evoMlpPolicy import MLPBlockPolicy
from helpers.evoMlpReal import MLPRealExecutor
from helpers.evoRanking import segment_metrics
from helpers.evoRoots import encode_histories,rootset_digest,tuning_roots
from helpers.evoStats import clustered_mean_ci
from helpers.locoData import RenderContext
from helpers.locoEnv import make_loco_env
from helpers.poseProbes import load_probe
from helpers.runManifest import build_manifest,file_sha256
from helpers.walkerLewm import load_walker_model
from helpers.walkerRules import execute_branch
from helpers.walkerValidation import verify_render_fingerprint


def main():
    ap=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--run-id',default=None)
    ap.add_argument('--no-upload',action='store_true')
    a=ap.parse_args()
    start=time.time()
    cfg=load_stage_config('stage2_coverage')
    source,Xlag,Ylag,Elag,_,state,split=load_features(cfg)
    run=EvoRun.create(cfg,'s2-coverage',a.run_id)
    device='cuda'
    paths=fetch_inputs(cfg.inputs)
    model,scaler=load_walker_model(paths['model'],device)
    probe,_=load_probe(paths['probes']/'walker_mlp.pt',device)
    im=ClosedLoopImaginer(model,scaler,probe,device=device)
    ctx=RenderContext()
    verify_render_fingerprint(ctx,paths['data']/'setA.h5')
    chosen,ppo_val,sel=select_episodes(paths['data']/'setA.h5',list(cfg.ppo.policies),
        int(cfg.ppo.n_episodes),int(cfg.ppo.episode_seed),float(cfg.ppo.val_fraction))
    if len(chosen)!=cfg.ppo.n_episodes:
        raise ValueError('not enough PPO episodes for the declared comparison')
    Xraw,Yraw,Eppo,n_frames=bc_frames(paths['data']/'setA.h5',chosen,ctx,im.encode,int(cfg.ppo.frame_stride))
    ctx.close()
    bc_encodes=im.counters.encodes
    np.savez(run.run_dir/'ppo_features.npz',X=Xraw,Y=Yraw,episode=Eppo)
    Xppo=((Xraw.astype(np.float32)-state['feature_mean'])/state['feature_std']).astype(np.float32)
    Yppo=Yraw.astype(np.float32)
    idx,weights,plan=coverage_indices(Elag,split['validation_episodes'],Eppo,ppo_val,
        seed=int(cfg.coverage.seed),fit_episodes_per_family=int(cfg.coverage.fit_episodes_per_family))
    np.savez(run.run_dir/'sample_indices.npz',**idx,validation_weights=weights)
    plan.update(ppo_episodes=chosen.tolist(),ppo_validation_episodes=ppo_val.tolist(),
                lag_validation_episodes=split['validation_episodes'],ppo_selection=sel,
                ppo_frames_rendered=n_frames,ppo_samples=len(Xppo))
    run.write_json('data_plan.json',plan)
    Xv=np.concatenate([Xlag[idx['validation_lag']],Xppo[idx['validation_ppo']]])
    Yv=np.concatenate([Ylag[idx['validation_lag']],Yppo[idx['validation_ppo']]])
    lag_n=len(idx['validation_lag'])
    val_slices={'PPOLag':slice(0,lag_n),'PPO':slice(lag_n,len(Xv))}
    datasets={
        'baseline':(Xlag[idx['baseline_lag']],Ylag[idx['baseline_lag']]),
        'mixed':(np.concatenate([Xlag[idx['mixed_lag']],Xppo[idx['mixed_ppo']]]),
                 np.concatenate([Ylag[idx['mixed_lag']],Yppo[idx['mixed_ppo']]])),
    }
    assert len(datasets['baseline'][0])==len(datasets['mixed'][0])
    policy=MLPBlockPolicy((state['action_mean'],state['action_std']),state['feature_mean'],state['feature_std'],
                           hidden=cfg.policy.hidden,device=device)
    Xval,Yval=[torch.as_tensor(v,device=device) for v in (Xv,Yv)]
    offline,thetas,history,update_counts={},{},[],{}
    print(f"[coverage] {plan['training_samples_per_arm']} fit samples per arm; {len(Xv)} common validation samples",flush=True)
    for label,(Xf,Yf) in datasets.items():
        Xfit,Yfit=[torch.as_tensor(v,device=device) for v in (Xf,Yf)]
        trials,total= [],0
        for seed in cfg.training.seeds:
            best,hist,updates=fit_coverage_seed(policy,Xfit,Yfit,Xval,Yval,seed=int(seed),
                epochs=int(cfg.training.epochs),batch_size=int(cfg.training.batch_size),
                lr=float(cfg.training.learning_rate),weight_decay=float(cfg.training.weight_decay),val_weights=weights,
                progress=lambda r,label=label:print(f"[coverage] {label} seed {r['seed']} epoch {r['epoch']} balanced MSE {r['val_mse']:.6f}",flush=True))
            trials.append(best)
            total+=updates
            history += [{'arm':label,**row} for row in hist]
            np.savez(run.run_dir/f'{label}_seed_{seed}.npz',theta=best['theta'],**policy.state(),
                     seed=best['seed'],epoch=best['epoch'])
        selected=select_fit(trials)
        thetas[label]=selected['theta']
        np.savez(run.run_dir/f'{label}_policy.npz',theta=selected['theta'],**policy.state(),
                 seed=selected['seed'],epoch=selected['epoch'])
        with torch.inference_mode():
            pred=policy.act(selected['theta'][None],Xval[None]).cpu().numpy()[0]
        offline[label]={'selected':{k:v for k,v in selected.items() if k!='theta'},
            'trials':[{k:v for k,v in trial.items() if k!='theta'} for trial in trials],
            'validation_by_family':{family:errors(pred[part],Yv[part]) for family,part in val_slices.items()},
            'training_samples':len(Xf),'optimizer_updates':total}
        update_counts[label]=total
    if len(set(update_counts.values()))!=1:
        raise AssertionError('training update budgets differ')
    total_updates=sum(update_counts.values())
    run.write_json('fit.json',offline)
    run.write_json('training_history.json',history)
    print('[coverage] checkpoint selection complete; evaluating development states',flush=True)
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
        for label, cls, pol, th in [("baseline", MLPRealExecutor, policy, thetas["baseline"]),
                                    ("mixed", MLPRealExecutor, policy, thetas["mixed"])]:
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
    paired = {"violation_rate_mixed_minus_baseline": clustered_mean_ci(
                  real["mixed"]["dense_violated"].astype(float)-real["baseline"]["dense_violated"], roots.source_episode),
              "progress_mixed_minus_baseline": clustered_mean_ci(real["mixed"]["progress"]-real["baseline"]["progress"], roots.source_episode)}
    rows = [{"root_id": r.root_id,"episode":int(r.episode),"step":int(r.step),"family":families[i],
             "recorded":recorded[i], **{label:{"violated":bool(real[label]["dense_violated"][i]),
                 "progress":float(real[label]["progress"][i]),
                 "first_unsafe_step":int(real[label]["dense_first_step"][i]),
                 "imagined_violated":bool(imagined[label]["violated"][i])} for label in real}}
             for i,r in enumerate(roots.roots)]
    report={'run_id':run.run_id,'role':'development coverage diagnostic; Gate 0 not run',
            'data_plan':plan,'offline':offline,'development':summary,'by_family':by_family,'paired':paired,
            'recorded':{'n':len(recorded),'violations':sum(r['violated'] for r in recorded),'progress_mean':rec_progress},
            'rootset_digest':rootset_digest(roots),'wall_clock_s':time.time()-start}
    run.write_json('diagnostic.json',report)
    run.write_json('development_rows.json',rows)
    files=[Path(__file__),ROOT/'experiments/helpers/evoCoverage.py',ROOT/'configs/evo/stage2_coverage.yaml',
           ROOT/'experiments/helpers/evoMlpPolicy.py',ROOT/'experiments/helpers/evoMlpReal.py',
           ROOT/'experiments/scripts/evo_mlp_diagnostic.py',ROOT/'experiments/scripts/evo_block_bc_init.py']
    for p in files:
        dest=run.run_dir/'source'/p.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,dest)
    upstream=input_revisions(paths)
    upstream['cloning_cache']={'repo_id':'Sachioster/safetydial-walker2d',
                              'path':cfg.cache.archive_path,'revision':cfg.cache.archive_revision}
    manifest=build_manifest(run_id=run.run_id,kind='evo-s2-coverage-diagnostic',started_at=start,
        seeds={'training':list(cfg.training.seeds),'ppo_episodes':cfg.ppo.episode_seed,'coverage':cfg.coverage.seed},
        upstream_revisions=upstream,
        data={'lag_cache_sha256':cfg.cache.features_sha256,'lag_split_sha256':cfg.cache.split_sha256,
              'ppo_cache_sha256':file_sha256(run.run_dir/'ppo_features.npz'),
              'sample_indices_sha256':file_sha256(run.run_dir/'sample_indices.npz'),
              'policy_sha256':{k:file_sha256(run.run_dir/f'{k}_policy.npz') for k in thetas}},
        metrics={'development':summary,'offline':offline},
        costs={'new_real_steps':int(counters.real_steps+len(recorded)*100),'real_executor':counters.as_dict(),
               'imagined':im.counters.as_dict(),'new_ppo_bc_renders':n_frames,'new_ppo_bc_encodes':bc_encodes,
               'history_renders':len(roots)*3,'optimizer_updates':total_updates,'updates_per_arm':update_counts,
               'cached_lag_frames_reused':int(split['n_frames_rendered']),
               'upstream_costs_from_cache_manifest':json.loads((source/'manifest.json').read_text())['costs']},
        extra={'implementation_sha256':{str(p.relative_to(ROOT)):file_sha256(p) for p in files},'gate0_run':False})
    run.finish(manifest,upload=not a.no_upload,
               readme=f'# {run.run_id}\n\nMatched-budget demonstration-coverage diagnostic on development states only.\n')
    print(json.dumps({'development':summary,'by_family':by_family,'paired':paired,
                      'wall_clock_s':report['wall_clock_s']},indent=2),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
