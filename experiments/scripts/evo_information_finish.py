#!/usr/bin/env python3
"""Resume the declared information experiment after its configuration parsing failure.

No training is rerun. Every candidate and the original roots are hash-pinned.
The sole correction is converting the intended half-reference progress rule to 0.5.
"""
from evo_information_diagnostic import *
from helpers.evoRoots import load_roots_json
from helpers.evoReal import _physics_env


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    start=time.time()
    cfg=load_stage_config('stage2_information_resume')
    assert cfg.resume.rerun_training is False
    assert float(cfg.validation_control.progress_floor)==.5
    training=ROOT/'runs'/cfg.resume.training_run
    training_hashes=json.loads((ROOT/cfg.resume.artifacts_manifest).read_text())
    assert all(file_sha256(training/name)==digest for name,digest in training_hashes.items())
    source=ROOT/'runs'/cfg.source.run_id
    receipt=json.loads((source/'hf_upload.json').read_text())
    assert receipt['revision']==cfg.source.revision
    requested=json.loads((ROOT/'docs/evoPlan/results/stage2-coverage'/cfg.source.run_id/'archive_request.json').read_text())
    hashes={v['path']:v['sha256'] for v in requested['files']}
    used=['ppo_features.npz','sample_indices.npz','data_plan.json','development_rows.json','manifest.json']
    assert all(file_sha256(source/name)==hashes[name] for name in used)
    paths=fetch_inputs(cfg.inputs)
    run=EvoRun.create(cfg,'s2-information-resume',args.run_id)
    history=json.loads((training/'training_history.json').read_text())
    run.write_json('training_history.json',history)
    val_roots=load_roots_json(training/'validation_roots.json')
    validation=RootSet.from_roots('setA_validation_control',val_roots)
    assert len(validation)==64
    shutil.copy2(training/'validation_roots.json',run.run_dir/'validation_roots.json')
    shutil.copy2(training/'config.yaml',run.run_dir/'original_training_config.yaml')
    shutil.copy2(training/'failure.json',run.run_dir/'source_failure.json')
    theta_pool,policies={},{}
    for arm in ['visual_history','state_reference']:
        candidates=[]
        for seed in cfg.training.seeds:
            rows=[r for r in history if r['arm']==arm and r['seed']==seed]
            best=min(rows,key=lambda r:r['val_mse'])
            epochs=sorted(set(list(cfg.training.checkpoint_epochs)+[best['epoch']]))
            for epoch in epochs:
                row=next(r for r in rows if r['epoch']==epoch)
                path=training/f'{arm}_seed_{seed}_epoch_{epoch}.npz'
                with np.load(path) as f:
                    state={k:f[k] for k in f.files}
                cls=StateReferencePolicy if arm=='state_reference' else HistoryBlockPolicy
                policies[arm]=cls.from_state(state,device='cuda')
                candidates.append(dict(**row,theta=state['theta']))
                shutil.copy2(path,run.run_dir/path.name)
        theta_pool[arm]=candidates
    torch.set_num_threads(8)
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
        visual_steps=counters.real_steps,recorded_validation_steps=6400,optimizer_updates=0,reused_training_updates=27000,
        imagined_rows=im.counters.imagined_rows,renders=counters.renders+264+64,
        encodes=counters.encodes+264,wall_s=time.time()-start)
    costs.update(failed_setup_real_steps=6400,failed_setup_renders=256,failed_setup_encodes=192,
                 cumulative_real_steps_including_failed_setup=costs['real_steps']+6400)
    run.write_json('diagnostic.json',dict(development=summary,by_family=by_family,paired=paired,costs=costs,
        gate0_run=False,selection_uses='64 setA validation roots, no development outcomes',
        n_params=policies['visual_history'].n_params))
    run.write_json('development_rows.json',rows)
    files=['configs/evo/stage2_information.yaml','experiments/helpers/evoInformation.py',
           'experiments/scripts/evo_information_diagnostic.py','experiments/tests/test_evo_information.py']
    files += ['configs/evo/stage2_information_resume.yaml',
              'configs/evo/information_training_artifacts.json',
              'experiments/scripts/evo_information_finish.py',
              'experiments/tests/test_evo_information_resume.py']
    for name in files:
        path=run.run_dir/'source'/name
        path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,path)
    manifest=build_manifest(run_id=run.run_id,kind='evo-information-diagnostic',costs=costs,started_at=start,
        upstream_revisions=input_revisions(paths),seeds=dict(fitting=list(cfg.training.seeds),
                                                            validation=int(cfg.validation_control.seed)),
        data=dict(training_artifacts=training_hashes,source_training_run=cfg.resume.training_run,source=receipt,source_files={k:hashes[k] for k in used},original_cache=dict(cfg.cache),
            history_cache=dict(cfg.history_cache),validation_digest=rootset_digest(validation),
            development_digest=rootset_digest(dev)),
        extra=dict(code_sha256={k:file_sha256(ROOT/k) for k in files},gate0_run=False))
    run.finish(manifest,upload=False,readme=f'# {run.run_id}\n\nInformation and control-validation diagnostic.\n')
    print(json.dumps(dict(development=summary,by_family=by_family,costs=costs)),flush=True)


if __name__=='__main__':
    main()
