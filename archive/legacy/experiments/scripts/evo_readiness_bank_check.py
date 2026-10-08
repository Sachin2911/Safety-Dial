#!/usr/bin/env python3
"""Six-episode development validation of charged collection and initial-feature caching."""
from evo_clean_targets_diagnostic import *
from helpers.evoReal import _physics_env
from helpers.evoReadinessBanks import (
    CollectionRole,collect_source_banks,encode_collected_banks,
    generate_source_episode,load_encoded_banks,
)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    started=time.time()
    cfg=load_stage_config('stage5_bank_check')
    p=cfg.validation
    source=load_stage_config(str(p.source_actor_pool_config)).fresh_bank
    paths=fetch_inputs(cfg.inputs,names=['model','data','policies'])
    manifest=paths['data']/'manifest.json'
    assert file_sha256(manifest)==source.actor_pool_manifest_sha256
    historical=json.loads(manifest.read_text())['data']['policies']
    names=[row['name'] for row in historical if row['len_mean']>=400 and row['return_mean']>500]
    assert names==list(source.actor_names)
    actor_hashes={name:file_sha256(paths['policies']/f'{name}.pt') for name in names}
    run=EvoRun.create(cfg,'s5-bank-check',args.run_id)
    files=['configs/evo/stage5_bank_check.yaml','configs/evo/stage2_readiness_baseline.yaml',
        'experiments/helpers/evoReadinessBanks.py','experiments/helpers/evoReadinessQueries.py',
        'experiments/scripts/evo_readiness_bank_check.py','experiments/tests/test_evo_readiness_banks.py']
    hashes={name:file_sha256(ROOT/name) for name in files}
    for name in files:
        dest=run.run_dir/'source'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,dest)
    provenance=dict(purpose=str(p.data_role_after_collection),code_sha256=hashes,
        actor_sha256=actor_hashes,source_actor_manifest_sha256=file_sha256(manifest),
        upstream=input_revisions(paths),main_stage5=False)
    run.write_json('declaration.json',dict(provenance=provenance,
        maximum_real_steps=int(p.maximum_real_steps),expected_predictor_rows=0,
        expected_history_renders_and_encodes=int(p.expected_history_renders_and_encodes),controller_queries=0))
    counter=dict(real_steps=0,teacher_queries=0,predictor_rows=0,renders=0,encodes=0)
    specs=[CollectionRole(name,int(p.roots_per_role),int(p.environment_seed_bases[name]),
        int(p.actor_noise_seed_bases[name]),int(p.maximum_attempts_per_required_root),
        int(p.maximum_steps_per_episode)) for name in ['fitness','selection','evaluation']]
    actors={}
    env=ctx=model=None
    original_predict=None
    torch.set_num_threads(1)
    bank_dir=run.run_dir/'banks'
    try:
        env=_physics_env()
        def collect(request,counts):
            key=(request['actor'],request['action_noise'])
            if key not in actors:
                actors[key]=load_exported_actor(paths['policies']/f'{key[0]}.pt',env.action_space,
                                                action_noise=key[1],device='cpu')
            actor=actors[key]
            actor.rng=np.random.default_rng(request['actor_noise_seed'])
            return generate_source_episode(env,actor,seed=request['environment_seed'],
                max_steps=request['max_steps'],counter=counts)
        kw=dict(actor_names=names,action_noise=list(source.action_noise),provenance=provenance,
                episode_callback=collect,counter=counter)
        paused=collect_source_banks(bank_dir,specs,**kw,max_new_attempts=int(p.pause_after_new_episode_attempts))
        assert not paused['complete']
        collection=collect_source_banks(bank_dir,specs,**kw)
        assert collection['complete']
        before=counter.copy()
        collect_source_banks(bank_dir,specs,**kw)
        assert counter==before
        assert counter['real_steps']<=p.maximum_real_steps
        assert collection['accounting']['completed_costs']['real_steps']==counter['real_steps']
        env.close()
        env=None
        print(f'[bank-check] six development roots collected; {counter["real_steps"]} charged steps',flush=True)
        model,scaler=load_walker_model(paths['model'],'cuda')
        original_predict=model.predict
        def counted_predict(z,a,*args,**kwargs):
            counter['predictor_rows']+=len(z)
            return original_predict(z,a,*args,**kwargs)
        model.predict=counted_predict
        im=HistoryImaginer(model,scaler,None,device='cuda')
        ctx=RenderContext()
        counter['renders']+=int(p.fingerprint_renders)
        verify_render_fingerprint(ctx,paths['data']/'setA.h5')
        def encode(root):
            counter['renders']+=3
            frames=ctx.render_many(root.history_qpos,root.history_qvel)
            counter['encodes']+=3
            return im.encode(frames).detach().cpu().numpy()
        encoder_provenance=dict(model=input_revisions(paths)['model'],code_sha256=hashes,
            rendering_fingerprint_source=source.actor_pool_manifest_sha256,
            encoding_batch='three history frames for one root')
        encoding_args=dict(encoder_provenance=encoder_provenance,encode_callback=encode,
                           read_costs=lambda:counter)
        initial=encode_collected_banks(bank_dir,**encoding_args,max_new_roots=int(p.pause_after_new_encoded_roots))
        assert not initial['complete']
        encoded=encode_collected_banks(bank_dir,**encoding_args)
        assert encoded['complete']
        before=counter.copy()
        encode_collected_banks(bank_dir,**encoding_args)
        assert counter==before
        metadata,roots,banks=load_encoded_banks(bank_dir)
        assert sum(len(b) for b in banks.values())==p.expected_encoded_roots
        assert counter['encodes']==p.expected_history_renders_and_encodes
        assert counter['renders']==p.expected_history_renders_and_encodes+p.fingerprint_renders
        assert counter['predictor_rows']==0
        assert len({r.episode for rows in roots.values() for r in rows})==p.expected_encoded_roots
        costs=dict(counter,gradient_updates=0,wall_s=time.time()-started)
        run.write_json('check.json',dict(passed=True,costs=costs,accepted_episodes=sum(len(b) for b in banks.values()),
            attempted_episodes=len(collection['attempts']),rejected_short_episodes=sum(not r['eligible'] for r in collection['attempts']),
            collection_pause_resume=True,encoding_pause_resume=True,completed_reuse_cost_zero=True,
            independent_source_episode_keys={role:list(bank.episode_keys) for role,bank in banks.items()},
            encoded_banks_sha256=metadata['encoded_banks_sha256'],controller_queries=0,
            data_role='development_only_not_main_final',main_stage5=False,rosarl_compared=False))
    except BaseException as exc:
        run.write_json('failure.json',dict(error=type(exc).__name__,message=str(exc),costs=counter))
        raise
    finally:
        if original_predict is not None:
            model.predict=original_predict
        if ctx is not None:
            ctx.close()
        if env is not None:
            env.close()
    run.finish(build_manifest(run_id=run.run_id,kind='evo-readiness-bank-check-v1',costs=costs,
        started_at=started,upstream_revisions=input_revisions(paths),data=provenance,
        seeds=dict(environment=dict(p.environment_seed_bases),actor_noise=dict(p.actor_noise_seed_bases)),
        extra=dict(code_sha256=hashes,main_stage5=False)),upload=False,
        readme=f'# {run.run_id}\n\nSix-episode development bank workflow validation.\n')
    print(json.dumps(dict(passed=True,costs=costs)),flush=True)


if __name__=='__main__':
    main()
