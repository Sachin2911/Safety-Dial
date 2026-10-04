#!/usr/bin/env python3
"""Controlled previous-action ablation using the original mixed demonstration splits."""
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
from helpers.threads import pin_threads
pin_threads()
import numpy as np
import torch
from evo_mlp_diagnostic import load_features,brief
from helpers.evoCoverage import fit_coverage_seed
from helpers.evoHistoryData import cached_previous_actions
from helpers.evoHistoryPolicy import HistoryBlockPolicy
from helpers.evoHistoryReal import HistoryRealExecutor
from helpers.evoHistoryImagine import HistoryImaginer
from helpers.evoImagine import Counters
from helpers.evoInputs import EvoRun,fetch_inputs,input_revisions,load_stage_config
from helpers.evoMlpFit import select_fit
from helpers.evoRanking import segment_metrics
from helpers.evoRoots import tuning_roots,encode_histories,rootset_digest
from helpers.evoStats import clustered_mean_ci
from helpers.locoData import RenderContext
from helpers.poseProbes import load_probe
from helpers.runManifest import build_manifest,file_sha256
from helpers.walkerLewm import load_walker_model
from helpers.walkerValidation import verify_render_fingerprint


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id',required=True)
    args=ap.parse_args()
    started=time.time()
    torch.set_num_threads(8)
    cfg=load_stage_config('stage2_action_history')
    source=ROOT/'runs'/cfg.source.run_id
    receipt=json.loads((source/'hf_upload.json').read_text())
    assert receipt['revision']==cfg.source.revision
    request=json.loads((ROOT/'docs/evoPlan/results/stage2-coverage'/cfg.source.run_id/'archive_request.json').read_text())
    hashes={i['path']:i['sha256'] for i in request['files']}
    used=['ppo_features.npz','sample_indices.npz','data_plan.json','mixed_policy.npz',
          'development_rows.json','manifest.json']
    assert all(file_sha256(source/n)==hashes[n] for n in used)
    _,Xlag,Ylag,Elag,_,state,_=load_features(cfg)
    with np.load(source/'ppo_features.npz') as f:
        Xppo=((f['X'].astype(np.float32)-state['feature_mean'])/state['feature_std']).astype(np.float32)
        Yppo,Eppo=f['Y'].astype(np.float32),f['episode']
    with np.load(source/'sample_indices.npz') as f:
        idx={k:f[k] for k in f.files}
    paths=fetch_inputs(cfg.inputs)
    past_lag=cached_previous_actions(paths['data']/'setA.h5',Elag,Ylag)
    past_ppo=cached_previous_actions(paths['data']/'setA.h5',Eppo,Yppo)
    run=EvoRun.create(cfg,'s2-action-history',args.run_id)
    np.savez_compressed(run.run_dir/'previous_actions.npz',lag=past_lag,ppo=past_ppo)
    past_lag=((past_lag.reshape(-1,10,6)-state['action_mean'])/state['action_std']).reshape(-1,60)
    past_ppo=((past_ppo.reshape(-1,10,6)-state['action_mean'])/state['action_std']).reshape(-1,60)
    def joined(key):
        return (np.concatenate([Xlag[idx[f'{key}_lag']],Xppo[idx[f'{key}_ppo']]]),
                np.concatenate([Ylag[idx[f'{key}_lag']],Yppo[idx[f'{key}_ppo']]]),
                np.concatenate([past_lag[idx[f'{key}_lag']],past_ppo[idx[f'{key}_ppo']]]))
    xf,yf,af=joined('mixed')
    xv,yv,av=joined('validation')
    assert len(xf)==45986 and len(xv)==23640
    offline,policies,thetas,history={},{},{},[]
    for label,use_history in [('zero_history',False),('action_history',True)]:
        policy=HistoryBlockPolicy((state['action_mean'],state['action_std']),state['feature_mean'],
            state['feature_std'],hidden=cfg.policy.hidden,use_history=use_history,device='cuda')
        policies[label]=policy
        Xfit=torch.tensor(np.concatenate([xf,af if use_history else np.zeros_like(af)],axis=1),device='cuda')
        Xval=torch.tensor(np.concatenate([xv,av if use_history else np.zeros_like(av)],axis=1),device='cuda')
        Yfit,Yval=[torch.tensor(y,device='cuda') for y in [yf,yv]]
        trials,updates=[],0
        for seed in cfg.training.seeds:
            best,hist,n=fit_coverage_seed(policy,Xfit,Yfit,Xval,Yval,seed=int(seed),
                epochs=int(cfg.training.epochs),batch_size=int(cfg.training.batch_size),
                lr=float(cfg.training.learning_rate),weight_decay=float(cfg.training.weight_decay),
                val_weights=idx['validation_weights'],
                progress=lambda r,label=label:print(f"[history] {label} seed {r['seed']} epoch {r['epoch']} validation {r['val_mse']:.6f}",flush=True))
            trials.append(best)
            history.extend([dict(arm=label,**row) for row in hist])
            updates+=n
            np.savez(run.run_dir/f'{label}_seed_{seed}.npz',theta=best['theta'],**policy.state(),
                     seed=best['seed'],epoch=best['epoch'])
        selected=select_fit(trials)
        thetas[label]=selected['theta']
        np.savez(run.run_dir/f'{label}_policy.npz',theta=selected['theta'],**policy.state(),
                 seed=selected['seed'],epoch=selected['epoch'])
        offline[label]=dict(selected={k:v for k,v in selected.items() if k!='theta'},
            trials=[{k:v for k,v in x.items() if k!='theta'} for x in trials],
            optimizer_updates=updates,n_params=policy.n_params,fit_samples=len(xf),
            validation_samples=len(xv))
        assert updates==13500
    run.write_json('fit.json',offline)
    run.write_json('training_history.json',history)
    print('[history] checkpoint selection frozen; evaluating development',flush=True)
    model,scaler=load_walker_model(paths['model'],'cuda')
    probe,_=load_probe(paths['probes']/'walker_mlp.pt','cuda')
    im=HistoryImaginer(model,scaler,probe,device='cuda')
    ctx=RenderContext()
    verify_render_fingerprint(ctx,paths['data']/'setA.h5')
    roots=encode_histories(tuning_roots(paths['banks']),im.encode,ctx)
    old=json.loads((source/'development_rows.json').read_text())
    assert roots.root_ids==[r['root_id'] for r in old] and len(roots)==24
    real,imagined,counters={},{},Counters()
    torch.set_num_threads(1)
    try:
        for label,policy in policies.items():
            ex=HistoryRealExecutor(model,scaler,probe,policy,device='cuda',ctx=ctx,
                max_envs=int(cfg.real.max_envs),encode_batch=int(cfg.real.encode_batch))
            try:
                real[label]=ex.run(thetas[label][None],roots.roots,z_hist=roots.z_hist,record_qpos=True)
                counters.add(ex.counters)
            finally:
                ex.close()
            np.savez_compressed(run.run_dir/f'real_{label}.npz',**real[label])
            imagined[label]=segment_metrics(im.rollout_policies(policy,thetas[label][None],
                roots.z_hist,roots.hist_blocks)['readout'][0,:,0])
    finally:
        ctx.close()
    families=np.array([r['family'] for r in old])
    summary={label:{**brief(out),'imagined_violations':int(imagined[label]['violated'].sum()),
                    'real_readout_violations':int(segment_metrics(out['readout'])['violated'].sum())}
             for label,out in real.items()}
    by_family={f:{label:brief({k:v[families==f] for k,v in out.items()})
                  for label,out in real.items()} for f in ['PPO','PPOLag']}
    paired={name:clustered_mean_ci(real['action_history'][key].astype(float)-real['zero_history'][key].astype(float),
                                  roots.source_episode)
            for name,key in [('violation_difference','dense_violated'),('progress_difference','progress')]}
    rows=[{**{k:old[i][k] for k in ['root_id','episode','step','family','recorded']},
           'source_mixed':old[i]['mixed'],**{label:dict(violated=bool(out['dense_violated'][i]),
             progress=float(out['progress'][i]),first_unsafe_step=int(out['dense_first_step'][i]),
             imagined_violated=bool(imagined[label]['violated'][i])) for label,out in real.items()}}
          for i in range(24)]
    costs=dict(real_steps=counters.real_steps,imagined_rows=im.counters.imagined_rows,
               renders=counters.renders+72+64,encodes=counters.encodes+72,
               optimizer_updates=27000,wall_s=time.time()-started,
               source_features_and_outcomes='reused pinned coverage cache; no new collection')
    report=dict(development=summary,by_family=by_family,paired=paired,offline=offline,costs=costs,
                gate0_run=False,rootset_digest=rootset_digest(roots))
    run.write_json('diagnostic.json',report)
    run.write_json('development_rows.json',rows)
    files=['configs/evo/stage2_action_history.yaml','experiments/scripts/evo_history_diagnostic.py',
           'experiments/tests/test_evo_history.py']+[
           f'experiments/helpers/evoHistory{n}.py' for n in ['Policy','Data','Imagine','Real']]
    for name in files:
        target=run.run_dir/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    manifest=build_manifest(run_id=run.run_id,kind='evo-action-history',costs=costs,started_at=started,
        upstream_revisions=input_revisions(paths),seeds=dict(fitting=list(cfg.training.seeds)),
        data=dict(source=receipt,source_files={n:hashes[n] for n in used},original_cache=dict(cfg.cache),
                  rootset_digest=rootset_digest(roots)),
        extra=dict(code_sha256={n:file_sha256(ROOT/n) for n in files},gate0_run=False))
    run.finish(manifest,upload=False,readme=f'# {run.run_id}\n\nPrevious-action ablation. See REPORT.md.\n')
    print(json.dumps(dict(development=summary,by_family=by_family,paired=paired,costs=costs)),flush=True)


if __name__=='__main__':
    main()
