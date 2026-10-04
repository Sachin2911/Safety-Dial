#!/usr/bin/env python3
"""Small frozen-model check of complete versus generation-boundary resumed searches."""
from evo_clean_targets_diagnostic import *
from helpers.evoReadinessCMA import ReadinessCMAConfig, run_readiness_cma
from helpers.evoReadinessSearch import FixedGainResidualPolicy
from helpers.evoResidualPolicy import ResidualHistoryPolicy
from helpers.evoRun import noise_seed


def exact_equal(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(exact_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    return a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    started = time.time()
    cfg = load_stage_config('stage5_resume_check')
    p = cfg.validation
    sources = json.loads((ROOT/p.source_hashes).read_text())
    for name, hashes in sources.items():
        assert all(file_sha256(ROOT/'runs'/name/f) == h for f, h in hashes.items())
    base, residual_dir, noise_dir = [ROOT/'runs'/p[k] for k in ['baseline_run','residual_run','noise_run']]
    with np.load(base/'history_latents.npz') as f:
        zh, hist = f['z_hist'], f['hist_blocks']
    with np.load(noise_dir/'calibration.npz') as f:
        sigma, fit_ids, check_ids = f['sigma'], f['fit_indices'], f['check_indices']
    roles = dict(fitness=fit_ids[:8], selection=check_ids[:4])
    assert not set(roles['fitness']) & set(roles['selection'])
    torch.set_num_threads(1)
    with np.load(residual_dir/'residual_policies.npz') as f:
        residual = ResidualHistoryPolicy.from_state({k:f[k] for k in f.files}, device='cuda')
    with torch.inference_mode():
        features = residual.features(torch.as_tensor(zh[fit_ids,-1],device='cuda'),
            torch.as_tensor(zh[fit_ids,-2],device='cuda'),hist[fit_ids,-1].reshape(-1,60))
        rms = residual.residual_features(features).double().square().mean(0).sqrt().clamp_min(float(p.feature_RMS_floor)).float()
    policy = FixedGainResidualPolicy(residual,rms)
    zero = np.zeros(policy.n_params,np.float64)
    paths = fetch_inputs(cfg.inputs,names=['model','probes'])
    model, scaler = load_walker_model(paths['model'],'cuda')
    probe, _ = load_probe(paths['probes']/'walker_mlp.pt','cuda')
    imaginer = HistoryImaginer(model,scaler,probe,device='cuda',sigma=sigma)
    run = EvoRun.create(cfg,'s5-resume-check',args.run_id)
    files = ['configs/evo/stage5_resume_check.yaml','configs/evo/power_pilot_sources.json',
        'experiments/scripts/evo_readiness_resume_check.py','experiments/helpers/evoReadinessCMA.py',
        'experiments/helpers/evoReadinessSearch.py','experiments/helpers/evoHistoryImagine.py',
        'experiments/helpers/evoResidualPolicy.py','experiments/helpers/evoRanking.py',
        'experiments/helpers/evoRun.py']
    hashes = {name:file_sha256(ROOT/name) for name in files}
    for name in files:
        dest = run.run_dir/'source'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,dest)
    provenance = dict(sources=sources,upstream=input_revisions(paths),code_sha256=hashes,
        feature_rms_sha256=array_sha256(rms.cpu().numpy()),roles={k:v.tolist() for k,v in roles.items()})
    run.write_json('declaration.json',dict(provenance=provenance,started_at=started,
        expected_predictor_rows=int(p.expected_predictor_rows),real_steps=0,
        duplicate_queries_charged=True,main_stage5=False,rosarl_compared=False))
    original_predict = model.predict
    charged = dict(predictor_rows=0,real_steps=0,gradient_updates=0)
    ledger, comparisons = [], {}
    def counted_predict(z,a,*args,**kwargs):
        charged['predictor_rows'] += len(z)
        return original_predict(z,a,*args,**kwargs)
    model.predict = counted_predict
    try:
        for k in map(float,p.noise_levels):
            n = 1 if k==0 else int(p.samples_k1)
            c = ReadinessCMAConfig(population=int(p.population),generations=int(p.generations),
                sigma0=float(p.sigma0),seed=int(p.seed),n_fitness_roots=8,
                fitness_roots_per_generation=4,fitness_samples=n,selection_roots=4,
                selection_samples=n,penalty_mode='zero',checkpoints=(0,1,2))
            states = {}
            for mode in ['full','resumed']:
                folder = run.run_dir/f'k{k:g}-{mode}'
                def query(X,indices,g,phase):
                    before = charged['predictor_rows']
                    entry = dict(k=k,mode=mode,phase=phase,generation=g,complete=False)
                    arrays = {name:[] for name in ['violated','ret','ret_pre']}
                    try:
                        seed = noise_seed(c.seed,g if phase=='fitness' else 100000)
                        for theta in X:
                            out = imaginer.rollout_policies(policy,theta[None],zh[indices],hist[indices],
                                n_samples=n,k=k,seed=seed,max_batch=16384)
                            metrics = segment_metrics(out['readout'][0])
                            for name in arrays:
                                arrays[name].append(metrics[name].ravel())
                        result = {name:np.stack(values) for name,values in arrays.items()}
                        result['rows'] = charged['predictor_rows']-before
                        entry['complete'] = True
                        return result
                    finally:
                        entry['predictor_rows'] = charged['predictor_rows']-before
                        ledger.append(entry)
                        run.write_json('query_ledger.json',ledger)
                def fitness(X,indices,g):
                    return query(X,roles['fitness'][indices],g,'fitness')
                def selection(theta,g):
                    return query(theta[None],roles['selection'],g,'selection')
                before = charged['predictor_rows']
                if mode=='resumed':
                    paused = run_readiness_cma(c,zero,fitness,selection,folder,
                        provenance=provenance,max_generations_this_call=int(p.pause_after_generation))
                    assert paused['generation']==1
                state = run_readiness_cma(c,zero,fitness,selection,folder,provenance=provenance)
                assert state['generation']==c.generations
                assert charged['predictor_rows']-before == state['costs']['fitness_rows']+state['costs']['selection_rows']
                states[mode] = state
            fields = ['identity','generation','penalty','nominees','history','snapshots','costs']
            checks = {field:bool(exact_equal(states['full'][field],states['resumed'][field])) for field in fields}
            comparisons[f'k{k:g}'] = dict(checks=checks,costs_per_search=states['full']['costs'],
                candidates_sha256=[h['asked_sha256'] for h in states['full']['history']])
            assert all(checks.values())
        assert charged['predictor_rows']==int(p.expected_predictor_rows)
        assert sum(x['predictor_rows'] for x in ledger)==charged['predictor_rows']
        charged['wall_s'] = time.time()-started
        run.write_json('check.json',dict(passed=True,comparisons=comparisons,costs=charged,
            full_model=True,real_control_evaluated=False,scientific_effect_claim=False))
    except BaseException as exc:
        run.write_json('failure.json',dict(error=type(exc).__name__,message=str(exc),costs=charged))
        raise
    finally:
        model.predict = original_predict
        run.write_json('validation_window.json',dict(started_at=started,ended_at=time.time(),
            note='May overlap the development pilot; exclude overlapping timings from throughput estimates.'))
    run.finish(build_manifest(run_id=run.run_id,kind='evo-readiness-resume-check-v1',costs=charged,
        started_at=started,upstream_revisions=input_revisions(paths),data=provenance,
        seeds=dict(search=int(p.seed)),extra=dict(code_sha256=hashes,main_stage5=False)),upload=False,
        readme=f'# {run.run_id}\n\nFull-model generation-boundary resume validation.\n')
    print(json.dumps(dict(passed=True,costs=charged)),flush=True)


if __name__=='__main__':
    main()
