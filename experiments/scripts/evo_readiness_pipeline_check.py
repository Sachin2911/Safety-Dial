#!/usr/bin/env python3
"""Bounded full-model/simulator validation of study scheduling and query reuse."""
from evo_clean_targets_diagnostic import *
from helpers.evoReadiness import reference_arrays
from helpers.evoReadinessCMA import ReadinessCMAConfig
from helpers.evoReadinessQueries import ImaginedSearchScorer, LatentBank, QueryStore
from helpers.evoReadinessSearch import FixedGainResidualPolicy
from helpers.evoReadinessStudy import SearchSpec, evaluate_frozen, run_searches
from helpers.evoResidualPolicy import ResidualHistoryPolicy
import helpers.evoHistoryReal as history_real_module
import helpers.evoReal as real_module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    started=time.time()
    cfg=load_stage_config('stage5_pipeline_check')
    p=cfg.validation
    sources=json.loads((ROOT/p.source_hashes).read_text())
    sources[str(p.baseline_run)]['real_recorded.npz']=str(p.recorded_file_sha256)
    for name,hashes in sources.items():
        assert all(file_sha256(ROOT/'runs'/name/f)==h for f,h in hashes.items())
    base,residual_dir,noise_dir=[ROOT/'runs'/p[k] for k in ['baseline_run','residual_run','noise_run']]
    roots=load_roots_json(base/'roots.json')
    with np.load(base/'history_latents.npz') as f:
        zh,hist=f['z_hist'],f['hist_blocks']
    with np.load(noise_dir/'calibration.npz') as f:
        sigma,fit_ids,check_ids=f['sigma'],f['fit_indices'],f['check_indices']
    roles=dict(fitness=fit_ids[:8],selection=check_ids[:4],evaluation=check_ids[4:8])
    banks={name:LatentBank(tuple(f'{p.baseline_run}:episode:{roots[i].episode}' for i in ids),
        zh[ids],hist[ids]) for name,ids in roles.items()}
    torch.set_num_threads(1)
    with np.load(residual_dir/'residual_policies.npz') as f:
        residual=ResidualHistoryPolicy.from_state({k:f[k] for k in f.files},device='cuda')
    with torch.inference_mode():
        features=residual.features(torch.as_tensor(zh[fit_ids,-1],device='cuda'),
            torch.as_tensor(zh[fit_ids,-2],device='cuda'),hist[fit_ids,-1].reshape(-1,60))
        rms=residual.residual_features(features).double().square().mean(0).sqrt().clamp_min(float(p.feature_RMS_floor)).float()
    policy=FixedGainResidualPolicy(residual,rms)
    zero=np.zeros(policy.n_params,np.float64)
    paths=fetch_inputs(cfg.inputs,names=['model','probes','data'])
    model,scaler=load_walker_model(paths['model'],'cuda')
    probe,_=load_probe(paths['probes']/'walker_mlp.pt','cuda')
    im=HistoryImaginer(model,scaler,probe,device='cuda',sigma=sigma)
    run=EvoRun.create(cfg,'s5-pipeline-check',args.run_id)
    files=['configs/evo/stage5_pipeline_check.yaml','configs/evo/power_pilot_sources.json',
        'experiments/helpers/evoReadinessCMA.py','experiments/helpers/evoReadinessQueries.py',
        'experiments/helpers/evoReadinessStudy.py','experiments/helpers/evoReadinessSearch.py',
        'experiments/scripts/evo_readiness_pipeline_check.py','experiments/tests/test_evo_readiness_study.py']
    hashes={name:file_sha256(ROOT/name) for name in files}
    for name in files:
        dest=run.run_dir/'source'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,dest)
    provenance=dict(sources=sources,upstream=input_revisions(paths),code_sha256=hashes,
        feature_rms_sha256=array_sha256(rms.cpu().numpy()),role='inspected_development_validation')
    run.write_json('declaration.json',dict(provenance=provenance,episode_indices=roles,
        expected_predictor_rows=int(p.expected_total_predictor_rows),expected_real_steps=int(p.expected_real_steps),
        main_stage5=False,rosarl_compared=False))
    charged=dict(predictor_rows=0,real_steps=0)
    original_predict=model.predict
    original_steps={history_real_module:history_real_module._step,real_module:real_module._step}
    def predict(z,a,*args,**kwargs):
        charged['predictor_rows']+=len(z)
        return original_predict(z,a,*args,**kwargs)
    def wrap_step(original):
        def step(*args,**kwargs):
            charged['real_steps']+=1
            return original(*args,**kwargs)
        return step
    model.predict=predict
    for module,original in original_steps.items():
        module._step=wrap_step(original)
    ex=ctx=None
    def counts():
        return dict(charged,renders=0 if ex is None else ex.counters.renders,
                    encodes=0 if ex is None else ex.counters.encodes)
    try:
        specs=[]
        for seed in map(int,p.search_seeds):
            for k in map(float,p.noise_levels):
                n=1 if k==0 else int(p.positive_noise_samples)
                specs.append(SearchSpec(k,ReadinessCMAConfig(population=int(p.population),
                    generations=int(p.generations),sigma0=float(p.sigma0),seed=seed,n_fitness_roots=8,
                    fitness_roots_per_generation=4,fitness_samples=n,selection_roots=4,
                    selection_samples=n,penalty_mode='zero',checkpoints=tuple(p.checkpoints))))
        evaluation=dict(audit_samples=dict(zero_noise=1,positive_noise=int(p.positive_noise_samples)),
                        noise_levels=[0.,1.],horizon_blocks=10)
        def factory(spec,folder,identity):
            return ImaginedSearchScorer(im,policy,banks['fitness'],banks['selection'],
                config=spec.config,noise_k=spec.noise_k,
                store=QueryStore(folder/'queries',study_identity=identity),read_costs=counts)
        work=run.run_dir/'study'
        kwargs=dict(provenance=provenance,evaluation_plan=evaluation,scorer_factory=factory)
        paused=run_searches(work,specs,banks,zero,**kwargs,
                           max_generations_this_call=int(p.search_pause_after_completed_generations))
        assert not paused['complete'] and not (work/'frozen_selections.json').exists()
        done=run_searches(work,specs,banks,zero,**kwargs)
        assert done['complete'] and len(done['frozen']['picks'])==p.expected_selected_slots
        assert charged==dict(predictor_rows=int(p.expected_search_predictor_rows),real_steps=0)
        before=counts()
        run_searches(work,specs,banks,zero,**kwargs)
        assert counts()==before
        print('[pipeline-check] all picks frozen; completed-search reuse costs zero',flush=True)
        ctx=RenderContext()
        verify_render_fingerprint(ctx,paths['data']/'setA.h5')
        ex=HistoryRealExecutor(model,scaler,probe,policy,device='cuda',ctx=ctx,max_envs=64,encode_batch=64)
        ids=roles['evaluation']
        selected_roots=[roots[i] for i in ids]
        def real(theta):
            return ex.run(theta[None],selected_roots,z_hist=zh[ids],record_qpos=True)
        def reference():
            return reference_arrays(ex.run_tapes(selected_roots,np.stack([r.policy_tape for r in selected_roots])))
        def audit(theta,k,n,seed):
            out=im.rollout_policies(policy,theta[None],zh[ids],hist[ids],n_samples=n,k=k,seed=seed,max_batch=16384)
            return dict(readout=out['readout'][0],actions=out['actions'][0],root_readout=out['root_readout'])
        kw=dict(identity=done['identity'],evaluation_bank=banks['evaluation'],real_callback=real,
            reference_callback=reference,audit_callback=audit,read_costs=counts,**evaluation)
        first=evaluate_frozen(work,**kw,max_new_queries=int(p.evaluation_pause_after_completed_queries))
        assert not first['complete']
        final=evaluate_frozen(work,**kw)
        assert final['complete'] and final['completed_queries']==p.expected_evaluation_queries
        assert not final['accounting']['pending_queries']
        before=counts()
        evaluate_frozen(work,**kw)
        assert counts()==before
        checks={}
        for current,old_name in [('baseline-real','real_controller'),('reference-real','real_recorded')]:
            with np.load(work/'evaluation'/f'{current}.npz') as new,np.load(base/f'{old_name}.npz') as old:
                fields=['actions','qpos','qvel','progress','dense_violated']+(['readout'] if current=='baseline-real' else [])
                checks[current]={key:bool(np.array_equal(new[key],old[key][ids])) for key in fields}
        assert all(all(v.values()) for v in checks.values())
        assert charged==dict(predictor_rows=int(p.expected_total_predictor_rows),real_steps=int(p.expected_real_steps))
        assert ex.counters.real_steps==charged['real_steps']
        costs=dict(**counts(),fingerprint_renders=64,total_renders=ex.counters.renders+64,
            gradient_updates=0,collection_steps=0,wall_s=time.time()-started)
        run.write_json('check.json',dict(passed=True,controls=checks,costs=costs,
            search_resume_completed=True,evaluation_resume_completed=True,
            completed_query_reuse_cost_zero=True,all_picks_frozen_before_real=True,
            main_study_executed=False,rosarl_compared=False))
    except BaseException as exc:
        run.write_json('failure.json',dict(error=type(exc).__name__,message=str(exc),costs=counts()))
        raise
    finally:
        model.predict=original_predict
        for module,original in original_steps.items():
            module._step=original
        if ex is not None:
            ex.close()
        if ctx is not None:
            ctx.close()
    run.finish(build_manifest(run_id=run.run_id,kind='evo-readiness-pipeline-check-v1',costs=costs,
        started_at=started,upstream_revisions=input_revisions(paths),data=provenance,
        seeds=dict(search=list(p.search_seeds)),extra=dict(code_sha256=hashes,main_stage5=False)),
        upload=False,readme=f'# {run.run_id}\n\nFrozen-model and simulator pipeline validation.\n')
    print(json.dumps(dict(passed=True,costs=costs)),flush=True)


if __name__=='__main__':
    main()
