#!/usr/bin/env python3
"""Visual versus state reference with independent closed-loop validation selection."""
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
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from evo_mlp_diagnostic import load_features,brief
from helpers.evoHistoryPolicy import HistoryBlockPolicy
from helpers.evoHistoryReal import HistoryRealExecutor
from helpers.evoHistoryImagine import HistoryImaginer
from helpers.evoInformation import cached_state_features,StateReferencePolicy,evaluate_state,fit_candidates,candidate_key
from helpers.evoImagine import Counters
from helpers.evoInputs import EvoRun,fetch_inputs,input_revisions,load_stage_config
from helpers.evoReal import _physics_env
from helpers.evoRoots import RootSet,tuning_roots,encode_histories,rootset_digest
from helpers.evoRanking import segment_metrics
from helpers.evoStats import clustered_mean_ci
from helpers.locoData import RenderContext
from helpers.poseProbes import load_probe
from helpers.runManifest import build_manifest,file_sha256
from helpers.walkerLewm import load_walker_model
from helpers.walkerRules import roots_from_episode,execute_branch
from helpers.walkerValidation import verify_render_fingerprint


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id',required=True)
    args=ap.parse_args()
    start=time.time()
    cfg=load_stage_config('stage2_information')
    source=ROOT/'runs'/cfg.source.run_id
    receipt=json.loads((source/'hf_upload.json').read_text())
    assert receipt['revision']==cfg.source.revision
    requested=json.loads((ROOT/'docs/evoPlan/results/stage2-coverage'/cfg.source.run_id/'archive_request.json').read_text())
    hashes={v['path']:v['sha256'] for v in requested['files']}
    used=['ppo_features.npz','sample_indices.npz','data_plan.json','development_rows.json','manifest.json']
    assert all(file_sha256(source/k)==hashes[k] for k in used)
    past_path=ROOT/'runs'/cfg.history_cache.run_id/'previous_actions.npz'
    assert file_sha256(past_path)==cfg.history_cache.previous_actions_sha256
    _,xl,yl,el,_,state,_=load_features(cfg)
    with np.load(source/'ppo_features.npz') as f:
        xp=((f['X'].astype(np.float32)-state['feature_mean'])/state['feature_std']).astype(np.float32)
        yp,ep=f['Y'].astype(np.float32),f['episode']
    with np.load(source/'sample_indices.npz') as f:
        idx={k:f[k] for k in f.files}
    with np.load(past_path) as f:
        al,apast=[((f[k].reshape(-1,10,6)-state['action_mean'])/state['action_std']).reshape(-1,60)
                  for k in ['lag','ppo']]
    paths=fetch_inputs(cfg.inputs)
    sl=cached_state_features(paths['data']/'setA.h5',el)
    sp=cached_state_features(paths['data']/'setA.h5',ep)
    def join(a,b,role):
        return np.concatenate([a[idx[f'{role}_lag']],b[idx[f'{role}_ppo']]])
    sf,sv=join(sl,sp,'mixed'),join(sl,sp,'validation')
    sm,ss=sf.mean(0),sf.std(0)+1e-6
    af,av=join(al,apast,'mixed'),join(al,apast,'validation')
    datasets={
        'visual_history':(np.concatenate([join(xl,xp,'mixed'),af],1),
                          np.concatenate([join(xl,xp,'validation'),av],1)),
        'state_reference':(np.concatenate([(sf-sm)/ss,np.zeros((len(sf),350),np.float32),af],1),
                           np.concatenate([(sv-sm)/ss,np.zeros((len(sv),350),np.float32),av],1))}
    yf,yv=join(yl,yp,'mixed'),join(yl,yp,'validation')
    run=EvoRun.create(cfg,'s2-information',args.run_id)
    np.savez_compressed(run.run_dir/'state_features.npz',lag=sl,ppo=sp,state_mean=sm,state_std=ss)
    plan=json.loads((source/'data_plan.json').read_text())
    val_roots=[]
    rng=np.random.default_rng(int(cfg.validation_control.seed))
    with h5py.File(paths['data']/'setA.h5') as f:
        off,ln=f['ep_offset'][:],f['ep_len'][:]
        for family,key in [('PPOLag','lag'),('PPO','ppo')]:
            episodes=np.array(plan[f'{key}_validation_episodes'])
            eligible=episodes[ln[episodes]>=120]
            chosen=np.sort(rng.choice(eligible,int(cfg.validation_control.roots_per_family),replace=False))
            for e in chosen:
                t=int(rng.integers(20,ln[e]-100+1))
                lo,hi=int(off[e]),int(off[e]+ln[e])
                data={k:f[k][lo:hi] for k in ['qpos','qvel','action','x_velocity']}
                r=roots_from_episode(data,[t],episode=int(e),prefix='val-setA')[0]
                r.meta.update(source_file='setA.h5',role='validation',family=family)
                val_roots.append(r)
    validation=RootSet.from_roots('setA_validation_control',val_roots)
    assert len(validation)==64
    run.write_json('validation_roots.json',dict(roots=[r.to_dict() for r in val_roots]))
    theta_pool,policies,history={},{},[]
    torch.set_num_threads(8)
    for arm,(xf,xv) in datasets.items():
        extra=dict(state_mean=sm,state_std=ss) if arm=='state_reference' else {}
        cls=StateReferencePolicy if arm=='state_reference' else HistoryBlockPolicy
        p=cls((state['action_mean'],state['action_std']),state['feature_mean'],state['feature_std'],
              hidden=cfg.policy.hidden,device='cuda',**extra)
        policies[arm]=p
        Xfit,Xval,Yfit,Yval=[torch.tensor(x,device='cuda') for x in [xf,xv,yf,yv]]
        candidates=[]
        for seed in cfg.training.seeds:
            c,h,n=fit_candidates(p,Xfit,Yfit,Xval,Yval,idx['validation_weights'],
                seed=int(seed),epochs=int(cfg.training.epochs),checkpoint_epochs=list(cfg.training.checkpoint_epochs),
                batch_size=int(cfg.training.batch_size),lr=float(cfg.training.learning_rate),
                weight_decay=float(cfg.training.weight_decay),
                progress=lambda r,arm=arm:print(f"[information] {arm} seed {r['seed']} epoch {r['epoch']} MSE {r['val_mse']:.5f}",flush=True))
            assert n==4500
            candidates.extend(c)
            history.extend([dict(arm=arm,**r) for r in h])
        theta_pool[arm]=candidates
        assert len(candidates)<=cfg.validation_control.max_candidates_per_arm
        for c in candidates:
            np.savez(run.run_dir/f"{arm}_seed_{c['seed']}_epoch_{c['epoch']}.npz",
                     theta=c['theta'],**p.state(),seed=c['seed'],epoch=c['epoch'])
    run.write_json('training_history.json',history)
    model,scaler=load_walker_model(paths['model'],'cuda')
    probe,_=load_probe(paths['probes']/'walker_mlp.pt','cuda')
    im=HistoryImaginer(model,scaler,probe,device='cuda')
    ctx=RenderContext()
    verify_render_fingerprint(ctx,paths['data']/'setA.h5')
    validation=encode_histories(validation,im.encode,ctx)
    torch.set_num_threads(1)
    env=_physics_env()
    try:
        refs=[execute_branch(env,r.qpos,r.qvel,r.policy_tape) for r in val_roots]
    finally:
        env.close()
    recorded=dict(violations=sum(log.unsafe()['health'] for log in refs),
                  progress_mean=float(np.mean([log.qpos[-1,0]-log.qpos[0,0] for log in refs])))
    minimum_progress=float(cfg.validation_control.progress_floor)*recorded['progress_mean']
    counters=Counters()
    state_steps=0
    selected,validation_rows={},[]
    ex=HistoryRealExecutor(model,scaler,probe,policies['visual_history'],device='cuda',ctx=ctx,
                           max_envs=64,encode_batch=64)
    try:
        for arm,candidates in theta_pool.items():
            eval_policy=StateReferencePolicy.from_state(policies[arm].state()) if arm=='state_reference' else policies[arm]
            for c in candidates:
                if arm=='state_reference':
                    out=evaluate_state(eval_policy,c['theta'],val_roots)
                    state_steps+=6400
                else:
                    out=ex.run(c['theta'][None],val_roots,z_hist=validation.z_hist,record_qpos=True)
                row=dict(arm=arm,seed=c['seed'],epoch=c['epoch'],val_mse=c['val_mse'],**brief(out))
                validation_rows.append(row)
                np.savez_compressed(run.run_dir/f"validation_{arm}_{c['seed']}_{c['epoch']}.npz",
                    **{k:out[k] for k in ['qpos','qvel','actions','dense_violated','dense_first_step','progress']})
                print(f"[information] validation {arm} {c['seed']}/{c['epoch']}: {row['violations']}/64",flush=True)
            chosen=min([r for r in validation_rows if r['arm']==arm],key=lambda r:candidate_key(r,minimum_progress))
            best=next(c for c in candidates if c['seed']==chosen['seed'] and c['epoch']==chosen['epoch'])
            selected[arm]=dict(row=chosen,theta=best['theta'])
            np.savez(run.run_dir/f'{arm}_policy.npz',theta=best['theta'],**policies[arm].state(),
                     seed=best['seed'],epoch=best['epoch'])
        counters.add(ex.counters)
    finally:
        ex.close()
    run.write_json('selection.json',dict(recorded=recorded,minimum_progress=minimum_progress,
                   candidates=validation_rows,selected={k:v['row'] for k,v in selected.items()}))
    print('[information] real validation selection frozen; evaluating development',flush=True)
    torch.set_num_threads(8)
    dev=encode_histories(tuning_roots(paths['banks']),im.encode,ctx)
    torch.set_num_threads(1)
    real={}
    try:
        p=policies['visual_history']
        ex=HistoryRealExecutor(model,scaler,probe,p,device='cuda',ctx=ctx,max_envs=24,encode_batch=64)
        try:
            real['visual_history']=ex.run(selected['visual_history']['theta'][None],dev.roots,
                                          z_hist=dev.z_hist,record_qpos=True)
            counters.add(ex.counters)
        finally:
            ex.close()
        p=StateReferencePolicy.from_state(policies['state_reference'].state())
        real['state_reference']=evaluate_state(p,selected['state_reference']['theta'],dev.roots)
        state_steps+=2400
        imagined=segment_metrics(im.rollout_policies(policies['visual_history'],
            selected['visual_history']['theta'][None],dev.z_hist,dev.hist_blocks)['readout'][0,:,0])
    finally:
        ctx.close()
    old=json.loads((source/'development_rows.json').read_text())
    assert dev.root_ids==[r['root_id'] for r in old]
    family=np.array([r['family'] for r in old])
    summary={k:brief(v) for k,v in real.items()}
    summary['visual_history']['imagined_violations']=int(imagined['violated'].sum())
    by_family={f:{k:brief({name:x[family==f] for name,x in v.items()}) for k,v in real.items()} for f in ['PPO','PPOLag']}
    rows=[dict(root_id=r['root_id'],episode=r['episode'],family=r['family'],recorded=r['recorded'],
               **{k:dict(violated=bool(v['dense_violated'][i]),progress=float(v['progress'][i]),
                          first_unsafe_step=int(v['dense_first_step'][i])) for k,v in real.items()})
          for i,r in enumerate(old)]
    paired={name:clustered_mean_ci(real['visual_history'][key].astype(float)-real['state_reference'][key].astype(float),
        dev.source_episode) for name,key in [('visual_minus_state_violations','dense_violated'),
                                            ('visual_minus_state_progress','progress')]}
    for arm,out in real.items():
        np.savez_compressed(run.run_dir/f'real_{arm}.npz',**out)
    costs=dict(real_steps=counters.real_steps+state_steps+6400,state_reference_steps=state_steps,
        visual_steps=counters.real_steps,recorded_validation_steps=6400,optimizer_updates=27000,
        imagined_rows=im.counters.imagined_rows,renders=counters.renders+264+64,
        encodes=counters.encodes+264,wall_s=time.time()-start)
    run.write_json('diagnostic.json',dict(development=summary,by_family=by_family,paired=paired,costs=costs,
        gate0_run=False,selection_uses='64 setA validation roots, no development outcomes',
        n_params=policies['visual_history'].n_params))
    run.write_json('development_rows.json',rows)
    files=['configs/evo/stage2_information.yaml','experiments/helpers/evoInformation.py',
           'experiments/scripts/evo_information_diagnostic.py','experiments/tests/test_evo_information.py']
    for name in files:
        path=run.run_dir/'source'/name
        path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,path)
    manifest=build_manifest(run_id=run.run_id,kind='evo-information-diagnostic',costs=costs,started_at=start,
        upstream_revisions=input_revisions(paths),seeds=dict(fitting=list(cfg.training.seeds),
                                                            validation=int(cfg.validation_control.seed)),
        data=dict(source=receipt,source_files={k:hashes[k] for k in used},original_cache=dict(cfg.cache),
            history_cache=dict(cfg.history_cache),validation_digest=rootset_digest(validation),
            development_digest=rootset_digest(dev)),
        extra=dict(code_sha256={k:file_sha256(ROOT/k) for k in files},gate0_run=False))
    run.finish(manifest,upload=False,readme=f'# {run.run_id}\n\nInformation and control-validation diagnostic.\n')
    print(json.dumps(dict(development=summary,by_family=by_family,costs=costs)),flush=True)


if __name__=='__main__':
    main()
