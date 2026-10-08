#!/usr/bin/env python3
"""Three-frame/two-action-block context versus matched masked context."""
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
from helpers.evoContextPolicy import ContextBlockPolicy,context_training_features
from helpers.evoContextReal import ContextRealExecutor
from helpers.evoContextImagine import ContextImaginer
from helpers.evoInformation import fit_candidates,candidate_key
from helpers.evoImagine import Counters
from helpers.evoInputs import EvoRun,fetch_inputs,input_revisions,load_stage_config
from helpers.evoRoots import RootSet,load_roots_json,tuning_roots,encode_histories,rootset_digest
from helpers.evoRanking import segment_metrics
from helpers.evoStats import clustered_mean_ci
from helpers.locoData import RenderContext
from helpers.poseProbes import load_probe
from helpers.runManifest import build_manifest,file_sha256
from helpers.walkerLewm import load_walker_model
from helpers.walkerValidation import verify_render_fingerprint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    started=time.time()
    torch.set_num_threads(8)
    cfg=load_stage_config('stage2_context')
    source=ROOT/'runs'/cfg.source.run_id
    receipt=json.loads((source/'hf_upload.json').read_text())
    assert receipt['revision']==cfg.source.revision
    request=json.loads((ROOT/'docs/evoPlan/results/stage2-coverage'/cfg.source.run_id/'archive_request.json').read_text())
    hashes={x['path']:x['sha256'] for x in request['files']}
    consumed=['ppo_features.npz','sample_indices.npz','development_rows.json','data_plan.json','manifest.json']
    assert all(file_sha256(source/n)==hashes[n] for n in consumed)
    history=ROOT/'runs'/cfg.history_cache.run_id/'previous_actions.npz'
    assert file_sha256(history)==cfg.history_cache.previous_actions_sha256
    validation_source=ROOT/'runs'/cfg.validation_control.source_run
    assert file_sha256(validation_source/'validation_roots.json')==cfg.validation_control.roots_sha256
    assert file_sha256(validation_source/'selection.json')==cfg.validation_control.selection_sha256
    _,xl,yl,el,_,state,_=load_features(cfg)
    with np.load(source/'ppo_features.npz') as f:
        xp=((f['X'].astype(np.float32)-state['feature_mean'])/state['feature_std']).astype(np.float32)
        yp,ep=f['Y'].astype(np.float32),f['episode']
    with np.load(source/'sample_indices.npz') as f:
        idx={k:f[k] for k in f.files}
    with np.load(history) as f:
        al,ap=[((f[k].reshape(-1,10,6)-state['action_mean'])/state['action_std']).reshape(-1,60)
               for k in ['lag','ppo']]
    full_lag,valid_lag=context_training_features(xl,al,el,use_context=True)
    full_ppo,valid_ppo=context_training_features(xp,ap,ep,use_context=True)
    fit_lag=idx['mixed_lag'][valid_lag[idx['mixed_lag']]]
    fit_ppo=idx['mixed_ppo'][valid_ppo[idx['mixed_ppo']]]
    n=min(len(fit_lag),len(fit_ppo))
    rng=np.random.default_rng(int(cfg.sampling.seed))
    fit_lag=np.sort(rng.choice(fit_lag,n,replace=False))
    fit_ppo=np.sort(rng.choice(fit_ppo,n,replace=False))
    val_lag=idx['validation_lag'][valid_lag[idx['validation_lag']]]
    val_ppo=idx['validation_ppo'][valid_ppo[idx['validation_ppo']]]
    Xfit=np.concatenate([full_lag[fit_lag],full_ppo[fit_ppo]])
    Xval=np.concatenate([full_lag[val_lag],full_ppo[val_ppo]])
    Yfit=np.concatenate([yl[fit_lag],yp[fit_ppo]])
    Yval=np.concatenate([yl[val_lag],yp[val_ppo]])
    weights=np.r_[np.full(len(val_lag),.5/len(val_lag)),np.full(len(val_ppo),.5/len(val_ppo))]
    run=EvoRun.create(cfg,'s2-context',args.run_id)
    np.savez(run.run_dir/'indices.npz',fit_lag=fit_lag,fit_ppo=fit_ppo,val_lag=val_lag,val_ppo=val_ppo,weights=weights)
    run.write_json('data_plan.json',dict(fit_per_family=n,fit_samples=2*n,
        validation_lag=len(val_lag),validation_ppo=len(val_ppo),
        removed_early_context=True,source_file='setA.h5',seed=int(cfg.sampling.seed)))
    policies,pool,training_history,updates={},{},[],{}
    for arm,use_context in [('short_context',False),('full_context',True)]:
        p=ContextBlockPolicy((state['action_mean'],state['action_std']),state['feature_mean'],
            state['feature_std'],hidden=cfg.policy.hidden,use_context=use_context,device='cuda')
        policies[arm]=p
        xf,xv=Xfit.copy(),Xval.copy()
        if not use_context:
            xf[:,384:636]=0
            xv[:,384:636]=0
        tensors=[torch.tensor(v,device='cuda') for v in [xf,Yfit,xv,Yval]]
        candidates,total=[],0
        for seed in cfg.training.seeds:
            c,h,count=fit_candidates(p,*tensors,weights,seed=int(seed),epochs=int(cfg.training.epochs),
                checkpoint_epochs=list(cfg.training.checkpoint_epochs),batch_size=int(cfg.training.batch_size),
                lr=float(cfg.training.learning_rate),weight_decay=float(cfg.training.weight_decay),
                progress=lambda r,arm=arm:print(f"[context] {arm} {r['seed']} epoch {r['epoch']} MSE {r['val_mse']:.5f}",flush=True))
            candidates.extend(c)
            total+=count
            training_history.extend([dict(arm=arm,**r) for r in h])
        updates[arm]=total
        pool[arm]=candidates
        for c in candidates:
            np.savez(run.run_dir/f"{arm}_seed_{c['seed']}_epoch_{c['epoch']}.npz",
                     theta=c['theta'],**p.state(),seed=c['seed'],epoch=c['epoch'])
    assert len(set(updates.values()))==1
    run.write_json('training_history.json',training_history)
    paths=fetch_inputs(cfg.inputs)
    model,scaler=load_walker_model(paths['model'],'cuda')
    probe,_=load_probe(paths['probes']/'walker_mlp.pt','cuda')
    im=ContextImaginer(model,scaler,probe,device='cuda')
    ctx=RenderContext()
    verify_render_fingerprint(ctx,paths['data']/'setA.h5')
    val=RootSet.from_roots('setA_validation_control',load_roots_json(validation_source/'validation_roots.json'))
    val=encode_histories(val,im.encode,ctx)
    recorded=json.loads((validation_source/'selection.json').read_text())['recorded']
    minimum_progress=float(cfg.validation_control.progress_floor)*recorded['progress_mean']
    counters=Counters()
    selected,selection_rows={},[]
    torch.set_num_threads(1)
    for arm,candidates in pool.items():
        ex=ContextRealExecutor(model,scaler,probe,policies[arm],device='cuda',ctx=ctx,max_envs=64,encode_batch=64)
        try:
            for c in candidates:
                out=ex.run(c['theta'][None],val.roots,z_hist=val.z_hist,record_qpos=True)
                row=dict(arm=arm,seed=c['seed'],epoch=c['epoch'],val_mse=c['val_mse'],**brief(out))
                selection_rows.append(row)
                np.savez_compressed(run.run_dir/f"validation_{arm}_{c['seed']}_{c['epoch']}.npz",
                    **{k:out[k] for k in ['qpos','qvel','actions','dense_violated','dense_first_step','progress']})
                print(f"[context] validation {arm} {c['seed']}/{c['epoch']}: {row['violations']}/64",flush=True)
            counters.add(ex.counters)
        finally:
            ex.close()
        chosen=min([r for r in selection_rows if r['arm']==arm],key=lambda r:candidate_key(r,minimum_progress))
        candidate=next(c for c in candidates if (c['seed'],c['epoch'])==(chosen['seed'],chosen['epoch']))
        selected[arm]=dict(row=chosen,theta=candidate['theta'])
        np.savez(run.run_dir/f'{arm}_policy.npz',theta=candidate['theta'],**policies[arm].state(),
                 seed=candidate['seed'],epoch=candidate['epoch'])
    run.write_json('selection.json',dict(recorded=recorded,minimum_progress=minimum_progress,
        candidates=selection_rows,selected={k:v['row'] for k,v in selected.items()}))
    print('[context] selection frozen; evaluating development',flush=True)
    torch.set_num_threads(8)
    dev=encode_histories(tuning_roots(paths['banks']),im.encode,ctx)
    torch.set_num_threads(1)
    real,imagined={},{}
    try:
        for arm in pool:
            ex=ContextRealExecutor(model,scaler,probe,policies[arm],device='cuda',ctx=ctx,max_envs=24,encode_batch=64)
            try:
                real[arm]=ex.run(selected[arm]['theta'][None],dev.roots,z_hist=dev.z_hist,record_qpos=True)
                counters.add(ex.counters)
            finally:
                ex.close()
            np.savez_compressed(run.run_dir/f'real_{arm}.npz',**real[arm])
            imagined[arm]=segment_metrics(im.rollout_policies(policies[arm],selected[arm]['theta'][None],
                dev.z_hist,dev.hist_blocks)['readout'][0,:,0])
    finally:
        ctx.close()
    old=json.loads((source/'development_rows.json').read_text())
    assert dev.root_ids==[r['root_id'] for r in old]
    family=np.array([r['family'] for r in old])
    summary={k:{**brief(v),'imagined_violations':int(imagined[k]['violated'].sum())} for k,v in real.items()}
    by_family={f:{k:brief({name:a[family==f] for name,a in v.items()}) for k,v in real.items()} for f in ['PPO','PPOLag']}
    rows=[dict(root_id=r['root_id'],episode=r['episode'],family=r['family'],recorded=r['recorded'],
        **{arm:dict(violated=bool(out['dense_violated'][i]),first_unsafe_step=int(out['dense_first_step'][i]),
                   progress=float(out['progress'][i])) for arm,out in real.items()}) for i,r in enumerate(old)]
    paired={name:clustered_mean_ci(real['full_context'][key].astype(float)-real['short_context'][key].astype(float),
        dev.source_episode) for name,key in [('violation_difference','dense_violated'),('progress_difference','progress')]}
    costs=dict(real_steps=counters.real_steps,imagined_rows=im.counters.imagined_rows,
        renders=counters.renders+264+64,encodes=counters.encodes+264,optimizer_updates=sum(updates.values()),
        updates_per_arm=updates,wall_s=time.time()-started,recorded_validation_reference='reused pinned outcome, zero new steps')
    run.write_json('diagnostic.json',dict(development=summary,by_family=by_family,paired=paired,costs=costs,
        gate0_run=False,n_params=policies['full_context'].n_params))
    run.write_json('development_rows.json',rows)
    files=['configs/evo/stage2_context.yaml','experiments/scripts/evo_context_diagnostic.py',
           'experiments/tests/test_evo_context.py']+[f'experiments/helpers/evoContext{k}.py' for k in ['Policy','Real','Imagine']]
    for name in files:
        p=run.run_dir/'source'/name
        p.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,p)
    run.finish(build_manifest(run_id=run.run_id,kind='evo-context-diagnostic',costs=costs,
        started_at=started,upstream_revisions=input_revisions(paths),
        data=dict(source=receipt,source_files={k:hashes[k] for k in consumed},
                  history_cache=dict(cfg.history_cache),validation_source=dict(cfg.validation_control),
                  validation_digest=rootset_digest(val),development_digest=rootset_digest(dev)),
        extra=dict(code_sha256={k:file_sha256(ROOT/k) for k in files},gate0_run=False)),
        upload=False,readme=f'# {run.run_id}\n\nFull temporal context comparison.\n')
    print(json.dumps(dict(development=summary,by_family=by_family,costs=costs)),flush=True)


if __name__=='__main__':
    main()
