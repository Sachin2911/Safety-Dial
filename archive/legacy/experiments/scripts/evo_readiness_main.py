#!/usr/bin/env python3
"""Guarded readiness main workflow. Defaults to query-free review, not execution."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault('MUJOCO_GL','egl')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
import numpy as np
import yaml
from omegaconf import OmegaConf

from helpers.evoHistoryImagine import HistoryImaginer
from helpers.evoHistoryReal import HistoryRealExecutor
from helpers.evoInputs import EvoRun,input_revisions
from helpers.evoReadiness import reference_arrays
from helpers.evoReadinessAnalysis import analyze_readiness,CrossedBootstrap
from helpers.evoReadinessBanks import collect_source_banks,encode_collected_banks,generate_source_episode,load_encoded_banks
from helpers.evoReadinessProtocol import launch_blockers,protocol_plan
from helpers.evoReadinessQueries import ImaginedSearchScorer,QueryStore,atomic_json
from helpers.evoReadinessResults import load_study_outcomes,save_paired_outcomes
from helpers.evoReadinessRuntime import initial_residual_diagnostics,load_frozen_runtime
from helpers.evoReadinessStudy import evaluate_frozen,run_searches
from helpers.evoReal import _physics_env
from helpers.locoData import RenderContext
from helpers.runManifest import build_manifest,file_sha256
from helpers.walkerAssets import load_exported_actor
from helpers.walkerValidation import verify_render_fingerprint
import helpers.evoHistoryReal as history_real_module
import helpers.evoReal as real_module
import helpers.evoReadinessBanks as bank_module


def check_launch(config,repository=ROOT):
    plan=protocol_plan(config)
    blockers=launch_blockers(config,repository)
    if 'implementation' not in config:
        blockers.append('executable asset/runtime settings are missing')
    if config.get('implementation',{}).get('collection_purpose')!='fresh_main_final':
        blockers.append('main collection purpose must be fresh_main_final')
    return plan,blockers


def source_inventory(protocol_path):
    files=[str(Path(__file__).resolve().relative_to(ROOT)),str(protocol_path.resolve().relative_to(ROOT)),
        'pyproject.toml','uv.lock','configs/evo/inputs.yaml','configs/evo/power_pilot_sources.json','configs/evo/stage2_readiness_baseline.yaml']
    files+=sorted(str(p.relative_to(ROOT)) for p in (ROOT/'experiments/helpers').glob('*.py'))
    files+=sorted(str(p.relative_to(ROOT)) for p in (ROOT/'experiments/tests').glob('test_evo_readiness*.py'))
    files=sorted(set(files))
    subprocess.run(['git','ls-files','--error-unmatch','--',*files],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','diff','--quiet','HEAD','--',*files],cwd=ROOT,check=True)
    return {name:file_sha256(ROOT/name) for name in files}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'configs/evo/stage5_main_candidate.yaml')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--run-id')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stop-after',choices=['collection','encoding','search','evaluation'])
    args=parser.parse_args()
    config=yaml.safe_load(args.protocol.read_text())
    plan,blockers=check_launch(config)
    if not args.execute or blockers:
        print(json.dumps(dict(execution_ready=not blockers,launch_blockers=blockers,
            counts=plan['counts'],candidate_batch_shapes=plan['batch_shapes'],experiment_run=False),indent=2))
        if args.execute:
            raise SystemExit('Main execution refused before loading assets or issuing queries.')
        return
    if not args.run_id:
        parser.error('--run-id is required with --execute')
    code=source_inventory(args.protocol)
    run=EvoRun.create(OmegaConf.create(config),'s5-main',args.run_id,resume=args.resume)
    identity=dict(protocol_sha256=file_sha256(args.protocol),source_sha256=code)
    launch_path=run.run_dir/'launch.json'
    if launch_path.exists():
        launch=json.loads(launch_path.read_text())
        if launch['identity']!=identity:
            raise ValueError('protocol or executable source changed since launch')
    else:
        started=time.time()
        launch=dict(identity=identity,started_at=started,
            deadline=started+3600*float(config['budget']['maximum_GPU_hours_proposed']))
        run.write_json('launch.json',launch)
        for name in code:
            target=run.run_dir/'source'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,target)
    def deadline():
        if time.time()>launch['deadline']:
            raise TimeoutError('declared wall-time cap reached; no automatic extension')
    deadline()
    runtime=load_frozen_runtime(config,ROOT,require_archived=True)
    provenance=dict(**runtime['provenance'],source_sha256=code,purpose='fresh_main_final')
    counter=dict(real_steps=0,teacher_queries=0,predictor_rows=0,renders=0,encodes=0,diagnostic_policy_calls=0)
    model=runtime['model']
    original_predict=model.predict
    originals={history_real_module:history_real_module._step,real_module:real_module._step,bank_module:bank_module._step}
    env=ctx=ex=None
    startup_receipts_path=run.run_dir/'startup_receipts.json'
    startups=json.loads(startup_receipts_path.read_text()) if startup_receipts_path.exists() else []
    def predict(z,a,*args,**kwargs):
        deadline()
        counter['predictor_rows']+=len(z)
        return original_predict(z,a,*args,**kwargs)
    def step_wrapper(original,charge):
        def step(*args,**kwargs):
            deadline()
            if charge:
                counter['real_steps']+=1
            return original(*args,**kwargs)
        return step
    model.predict=predict
    for module,original in originals.items():
        module._step=step_wrapper(original,module is not bank_module)
    def counts():
        out=counter.copy()
        if ex is not None:
            out['renders']+=ex.counters.renders
            out['encodes']+=ex.counters.encodes
        return out
    banks_path=run.run_dir/'banks'
    work=run.run_dir/'study'
    try:
        env=_physics_env()
        actors={}
        def episode(request,charged):
            deadline()
            key=(request['actor'],request['action_noise'])
            if key not in actors:
                actors[key]=load_exported_actor(runtime['paths']['policies']/f'{key[0]}.pt',
                    env.action_space,action_noise=key[1],device='cpu')
            actor=actors[key]
            actor.rng=np.random.default_rng(request['actor_noise_seed'])
            out=generate_source_episode(env,actor,seed=request['environment_seed'],max_steps=request['max_steps'],counter=charged)
            if request['attempt']%32==0:
                print(f'[main] source {request["role"]} attempt {request["attempt"]}, current-process real steps {counter["real_steps"]}',flush=True)
            return out
        collection=collect_source_banks(banks_path,plan['collection_roles'],actor_names=runtime['actor_names'],
            action_noise=list(runtime['actor_config'].action_noise),provenance=provenance,
            episode_callback=episode,counter=counter)
        env.close()
        env=None
        if args.stop_after=='collection':
            return
        ctx=RenderContext()
        deadline()
        counter['renders']+=64
        verify_render_fingerprint(ctx,runtime['paths']['data']/'setA.h5')
        startups.append(dict(time=time.time(),fingerprint_renders=64))
        atomic_json(startup_receipts_path,startups)
        im=HistoryImaginer(model,runtime['scaler'],runtime['probe'],device='cuda',sigma=runtime['sigma'])
        def encode(root):
            deadline()
            counter['renders']+=3
            frames=ctx.render_many(root.history_qpos,root.history_qvel)
            counter['encodes']+=3
            return im.encode(frames).detach().cpu().numpy()
        encoding=encode_collected_banks(banks_path,encoder_provenance=provenance,
            encode_callback=encode,read_costs=counts)
        encoded,physical_roots,banks=load_encoded_banks(banks_path)
        if args.stop_after=='encoding':
            return
        study_provenance=dict(**provenance,encoded_banks_sha256=encoded['encoded_banks_sha256'])
        def factory(spec,folder,identity):
            print(f'[main] search {spec.name}',flush=True)
            return ImaginedSearchScorer(im,runtime['policy'],banks['fitness'],banks['selection'],
                config=spec.config,noise_k=spec.noise_k,
                store=QueryStore(folder/'queries',study_identity=identity),read_costs=counts,
                max_batch=config['implementation']['max_imagined_batch'])
        done=run_searches(work,plan['search_specs'],banks,runtime['zero'],provenance=study_provenance,
            evaluation_plan=plan['evaluation_plan'],scorer_factory=factory)
        assert done['complete']
        diagnostic_store=QueryStore(run.run_dir/'initial_action_diagnostics',study_identity=done['identity'])
        def policy_call():
            deadline()
            counter['diagnostic_policy_calls']+=1
        for spec in plan['search_specs']:
            path=work/'searches'/spec.name/'queries/fitness-g0000.npz'
            with np.load(path,allow_pickle=False) as saved:
                candidates=saved['theta'].copy()
            diagnostic_store.execute(spec.name,request=dict(first_fitness_query_sha256=file_sha256(path),
                calibration_features_sha256=provenance['calibration_features_sha256']),
                expected_costs=dict(predictor_rows=0,real_steps=0,diagnostic_policy_calls=spec.config.population+1),
                callback=lambda:initial_residual_diagnostics(runtime['policy'],runtime['calibration_features'],candidates,on_policy_call=policy_call),
                read_costs=counts)
        if args.stop_after=='search':
            return
        print('[main] all selections frozen; paired final evaluation begins',flush=True)
        ex=HistoryRealExecutor(model,runtime['scaler'],runtime['probe'],runtime['policy'],device='cuda',ctx=ctx,
            max_envs=64,encode_batch=64)
        final_roots=physical_roots['evaluation']
        def real(theta):
            deadline()
            return ex.run(theta[None],final_roots,z_hist=banks['evaluation'].z_hist,record_qpos=True)
        def reference():
            deadline()
            return reference_arrays(ex.run_tapes(final_roots,np.stack([r.policy_tape for r in final_roots])))
        def audit(theta,k,n,seed):
            deadline()
            result=im.rollout_policies(runtime['policy'],theta[None],banks['evaluation'].z_hist,
                banks['evaluation'].hist_blocks,n_samples=n,k=k,seed=seed,
                max_batch=config['implementation']['max_imagined_batch'])
            return dict(readout=result['readout'][0],root_readout=result['root_readout'])
        evaluated=evaluate_frozen(work,identity=done['identity'],evaluation_bank=banks['evaluation'],
            real_callback=real,reference_callback=reference,audit_callback=audit,read_costs=counts,
            **plan['evaluation_plan'])
        assert evaluated['complete']
        if args.stop_after=='evaluation':
            return
        outcomes=load_study_outcomes(work,expected_roots=final_roots)
        save_paired_outcomes(run.run_dir/'analysis',outcomes)
        a=config['analysis']
        analysis=analyze_readiness(outcomes['cells'],outcomes['baseline'],search_seeds=outcomes['search_seeds'],
            episode_keys=outcomes['episode_keys'],population_sizes=config['search']['population_sizes'],
            checkpoints=[g for g in config['search']['checkpoints'] if g],
            low_pressure=(a['low_pressure']['population'],a['low_pressure']['generation']),
            high_pressure=(a['high_pressure']['population'],a['high_pressure']['generation']),
            replicates=a['bootstrap_replicates'],bootstrap_seed=a['bootstrap_seed'],
            minimum_progress_ratio=1-a['proposed_allowable_relative_progress_loss'])
        boot=CrossedBootstrap(len(outcomes['search_seeds']),len(outcomes['episode_keys']),
            replicates=a['bootstrap_replicates'],seed=a['bootstrap_seed'])
        decomposition=[]
        for key,diag in sorted(outcomes['diagnostics'].items()):
            cell=outcomes['cells'][key]
            decomposition.append(dict(condition=list(key),
                dense_minus_endpoint=boot.interval(cell['real'].astype(float)-diag['endpoint_truth']),
                endpoint_minus_real_readout=boot.interval(diag['endpoint_truth'].astype(float)-diag['real_readout']),
                real_readout_minus_imagined=boot.interval(diag['real_readout'].astype(float)-cell['imagined'][key[0]])))
        run.write_json('analysis.json',analysis)
        run.write_json('decomposition.json',decomposition)
        run.write_json('analysis_provenance.json',outcomes['provenance'])
        accounting=[collection['accounting'],encoding['accounting'],evaluated['accounting'],diagnostic_store.accounting()]
        for spec in plan['search_specs']:
            accounting.append(QueryStore(work/'searches'/spec.name/'queries',study_identity=done['identity']).accounting())
        assert not any(x['pending_queries'] for x in accounting)
        costs={}
        for record in accounting:
            for key,value in record['completed_costs'].items():
                costs[key]=costs.get(key,0)+value
        costs['renders']+=sum(s['fingerprint_renders'] for s in startups)
        assert costs['predictor_rows']==plan['counts']['total_predictor_rows']
        assert costs['real_steps']==collection['accounting']['completed_costs']['real_steps']+plan['counts']['final_real_steps']
        assert costs['real_steps']<=plan['counts']['maximum_new_real_steps']
        costs.update(gradient_updates=0,wall_s=time.time()-launch['started_at'])
        run.write_json('costs.json',costs)
        report=f'''# Readiness main study: {run.run_id}\n\nThe locked four-arm study completed on fresh source episodes. Historical Gate 0 and the transfer screen remain failures.\n\nCo-primary estimates and intervals are in `analysis.json`; the real-risk/progress criteria are reported separately. A smaller prediction gap alone is not a safety benefit.\n\nEvery selection was frozen before final outcomes. Source, query and analysis identities and complete incremental costs are retained. These short branches do not establish safe full-episode control.\n'''
        (run.results_dir/'README.md').write_text(report)
        run.finish(build_manifest(run_id=run.run_id,kind='evo-readiness-main-v1',costs=costs,
            started_at=launch['started_at'],upstream_revisions=input_revisions(runtime['paths']),
            data=study_provenance,seeds=dict(search=config['search']['search_seeds'],bootstrap=a['bootstrap_seed']),
            extra=dict(source_sha256=code,protocol_sha256=identity['protocol_sha256'])),upload=False,readme=report)
        print(json.dumps(dict(complete=True,primary=analysis['primary'],costs=costs)),flush=True)
    except BaseException as exc:
        run.write_json('failure.json',dict(error=type(exc).__name__,message=str(exc),current_process_costs=counts(),
            note='Completed query receipts and unresolved journals retain prior/partial costs. No implicit retry or budget extension.'))
        raise
    finally:
        model.predict=original_predict
        for module,original in originals.items():
            module._step=original
        if ex is not None:
            ex.close()
        if ctx is not None:
            ctx.close()
        if env is not None:
            env.close()


if __name__=='__main__':
    main()
