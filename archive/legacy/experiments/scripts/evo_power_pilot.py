#!/usr/bin/env python3
"""Bounded readiness-route search/variance pilot on inspected development episodes."""
from evo_clean_targets_diagnostic import *
from helpers.evoResidualPolicy import ResidualHistoryPolicy
from helpers.evoReadinessSearch import FixedGainResidualPolicy, NomineeSelector, pilot_uncertainty, task_returns
from helpers.evoRun import CMAConfig, noise_seed, run_cmaes
from helpers.evoRanking import rank_candidates
import helpers.evoHistoryReal as real_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    started = time.time()
    cfg = load_stage_config('stage5_power_pilot')
    p = cfg.power_pilot
    sources = json.loads((ROOT/p.source_hashes).read_text())
    for name, hashes in sources.items():
        assert all(file_sha256(ROOT/'runs'/name/f) == h for f, h in hashes.items())
    base_dir, res_dir, noise_dir = [ROOT/'runs'/p[k] for k in ['baseline_run', 'residual_run', 'noise_run']]
    roots = load_roots_json(base_dir/'roots.json')
    assert len(roots) == len({r.episode for r in roots}) == 256
    with np.load(base_dir/'history_latents.npz') as f:
        zh, hist = f['z_hist'], f['hist_blocks']
    with np.load(noise_dir/'calibration.npz') as f:
        sigma, calibration_idx = f['sigma'], f['fit_indices']
    rest = np.random.default_rng(int(p.role_seed)).permutation(np.setdiff1d(np.arange(256), calibration_idx))
    role = dict(fitness=np.concatenate([calibration_idx, rest[:32]]), selection=rest[32:96], assessment=rest[96:])
    assert [len(role[k]) for k in role] == [96, 64, 96]
    assert len(set(np.concatenate(list(role.values())))) == 256
    torch.set_num_threads(int(p.torch_threads))
    with np.load(res_dir/'residual_policies.npz') as f:
        residual = ResidualHistoryPolicy.from_state({k:f[k] for k in f.files}, device='cuda')
    with torch.inference_mode():
        feats = residual.features(torch.as_tensor(zh[calibration_idx, -1], device='cuda'),
            torch.as_tensor(zh[calibration_idx, -2], device='cuda'), hist[calibration_idx, -1].reshape(-1, 60))
        rms = residual.residual_features(feats).double().square().mean(0).sqrt().clamp_min(float(p.feature_RMS_floor)).float()
    policy = FixedGainResidualPolicy(residual, rms)
    zero = np.zeros(policy.n_params, np.float64)
    assert policy.n_params == p.evolved_parameters
    paths = fetch_inputs(cfg.inputs)
    model, scaler = load_walker_model(paths['model'], 'cuda')
    probe, _ = load_probe(paths['probes']/'walker_mlp.pt', 'cuda')
    im = HistoryImaginer(model, scaler, probe, device='cuda', sigma=sigma)
    run = EvoRun.create(cfg, 's5-power-pilot', args.run_id)
    run.write_json('episode_roles.json', dict(indices=role, calibration_indices=calibration_idx,
        source_episodes={k:[roots[i].episode for i in ids] for k, ids in role.items()},
        all_roles='previously_inspected_development', new_episode_collection=False))
    np.savez(run.run_dir/'preconditioner.npz', feature_rms=rms.cpu().numpy(), calibration_indices=calibration_idx)
    files = ['configs/evo/stage5_power_pilot.yaml', 'configs/evo/power_pilot_sources.json',
        'experiments/helpers/evoReadinessSearch.py', 'experiments/scripts/evo_power_pilot.py',
        'experiments/tests/test_evo_readiness_search.py']
    for name in files:
        dest = run.run_dir/'source'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, dest)
    charged = dict(predictor_rows=0, real_steps=0, fitness_rows=0, selection_rows=0, assessment_rows=0, zero_control_rows=0)
    original_predict, original_step = model.predict, real_module._step
    def counted_predict(z, a, *args, **kwargs):
        charged['predictor_rows'] += len(z)
        return original_predict(z, a, *args, **kwargs)
    def counted_step(*args, **kwargs):
        charged['real_steps'] += 1
        return original_step(*args, **kwargs)
    model.predict, real_module._step = counted_predict, counted_step
    ctx = ex = None
    fit_times, all_picks, selected_theta = [], [], []
    def predict(theta, indices, k, n, seed, phase):
        if time.time()-started > float(p.resource_cap_hours)*3600:
            raise TimeoutError('declared pilot wall-time cap reached')
        before = charged['predictor_rows']
        try:
            return im.rollout_policies(policy, np.asarray(theta)[None], zh[indices], hist[indices],
                n_samples=n, k=float(k), seed=int(seed), max_batch=int(p.max_imagined_batch))
        finally:
            charged[phase] += charged['predictor_rows']-before
    try:
        original_idx = np.asarray(json.loads((res_dir/'root_split.json').read_text())['source_indices'])
        control = predict(zero, original_idx, 0., 1, 0, 'zero_control_rows')
        with np.load(res_dir/'imagined_000.npz') as old:
            match = {k:bool(np.array_equal(old[k], control[k])) for k in ['actions','readout','root_readout']}
        run.write_json('zero_imagined_control.json', match)
        assert all(match.values())
        for search_seed in p.search_seeds:
            for pop in p.population_sizes:
                for noise_k in p.noise_levels:
                    seed, pop, noise_k = int(search_seed), int(pop), float(noise_k)
                    name = f'k{noise_k:g}-pop{pop}-seed{seed}'
                    folder = run.run_dir/name
                    folder.mkdir(exist_ok=False)
                    selector = NomineeSelector()
                    ns = 1 if noise_k == 0 else int(p.samples.selection_k1)
                    select_seed = noise_seed(seed, 100000)
                    def add_nominee(theta, generation):
                        out = predict(theta, role['selection'], noise_k, ns, select_seed, 'selection_rows')
                        metrics = segment_metrics(out['readout'][0])
                        selector.add(theta, metrics)
                        np.savez_compressed(folder/f'selection_g{generation:04d}.npz', theta=theta,
                            readout=out['readout'][0], score=selector.scores[-1])
                    add_nominee(zero, 0)
                    def score_fn(X, indices, generation):
                        tick = time.perf_counter()
                        rows, pre, flags = [], [], []
                        indices = role['fitness'][indices]
                        n = 1 if noise_k == 0 else int(p.samples.fitness_k1)
                        for candidate in X:
                            out = predict(candidate, indices, noise_k, n, noise_seed(seed, generation), 'fitness_rows')
                            m = segment_metrics(out['readout'][0])
                            rows.append(m['ret'].ravel())
                            pre.append(m['ret_pre'].ravel())
                            flags.append(m['violated'].ravel())
                        full, pre, flags = map(np.stack, [rows, pre, flags])
                        u = task_returns(dict(violated=flags, ret=full, ret_pre=pre))
                        ranks, _ = rank_candidates('fixed', violated=flags, ret=u, ret_pre=u, lam=0.)
                        winner = int(np.argmin(ranks))
                        np.savez_compressed(folder/f'fitness_g{generation+1:04d}.npz', theta=X,
                            source_indices=indices, violated=flags, full_progress=full, task_return=u, nominee=winner)
                        torch.cuda.synchronize()
                        fit_times.append(dict(search=name, generation=generation+1, rows=len(X)*len(indices)*n*10,
                                              seconds=time.perf_counter()-tick))
                        add_nominee(X[winner], generation+1)
                        return dict(violated=flags, ret=u, ret_pre=u, rows=len(X)*len(indices)*n*10)
                    cma_cfg = CMAConfig(popsize=pop, generations=int(p.generations), sigma0=float(p.sigma0),
                        seed=seed, k_roots=int(p.fitness_roots_per_generation), n_samples=1 if noise_k==0 else int(p.samples.fitness_k1),
                        noise_k=noise_k, rule='fixed', lam=0., eval_generations=(), checkpoint_every=1,
                        cma_options={'CMA_diagonal':True})
                    print(f'[power] starting {name}', flush=True)
                    result = run_cmaes(cma_cfg, zero, score_fn, folder/'cma', n_fitness_roots=96,
                                       resume=False, log=lambda msg:print(f'[{name}] {msg}', flush=True))
                    assert result['counters']['generations'] == p.generations and len(selector.theta)==p.generations+1
                    for generation in p.checkpoints:
                        nominee, theta = selector.best(int(generation))
                        all_picks.append(dict(search=name, seed=seed, population=pop, noise_k=noise_k,
                            generation=int(generation), nominee_generation=nominee, selection_score=selector.scores[nominee],
                            slot=len(selected_theta)))
                        selected_theta.append(theta)
                    run.write_json('execution_progress.json', dict(completed_searches=len(all_picks)//len(p.checkpoints),
                        frozen_picks=all_picks, charged=charged), results=False)
        np.savez(run.run_dir/'selected_policies.npz', theta=np.stack(selected_theta), feature_rms=rms.cpu().numpy())
        run.write_json('frozen_selections.json', dict(picks=all_picks,
            parameters_sha256=file_sha256(run.run_dir/'selected_policies.npz'),
            before_new_real_evaluation=True, main_stage5=False, rosarl_compared=False))
        print('[power] all selections frozen; beginning paired real assessment', flush=True)
        idx = role['assessment']
        assessment_roots = [roots[j] for j in idx]
        ctx = RenderContext()
        verify_render_fingerprint(ctx, paths['data']/'setA.h5')
        ex = HistoryRealExecutor(model, scaler, probe, policy, device='cuda', ctx=ctx,
                                max_envs=int(p.real_max_envs), encode_batch=int(p.encode_batch))
        baseline_real = ex.run(zero[None], assessment_roots, z_hist=zh[idx], record_qpos=True)
        np.savez_compressed(run.run_dir/'real_baseline.npz', **baseline_real)
        with np.load(base_dir/'real_controller.npz') as old:
            checks = {k:bool(np.array_equal(old[k][idx],baseline_real[k])) for k in ['actions','qpos','qvel','progress','dense_violated','readout']}
        run.write_json('zero_real_control.json', checks)
        assert all(checks.values())
        audit = {}
        for seed in map(int,p.search_seeds):
            audit[seed] = {}
            for k in [0.,1.]:
                out = predict(zero, idx, k, 1 if k==0 else int(p.samples.audit_k1), noise_seed(seed,200000), 'assessment_rows')
                m = segment_metrics(out['readout'][0])
                audit[seed][k] = m['violated'].mean(-1)
                np.savez_compressed(run.run_dir/f'baseline_audit_seed{seed}_k{k:g}.npz', readout=out['readout'][0])
        truth, progress, imagination, summaries = [], [], [], []
        for pick, theta in zip(all_picks, selected_theta):
            j = pick['slot']
            if time.time()-started > float(p.resource_cap_hours)*3600:
                raise TimeoutError('declared pilot wall-time cap reached')
            real = ex.run(theta[None], assessment_roots, z_hist=zh[idx], record_qpos=True)
            np.savez_compressed(run.run_dir/f'real_{j:03d}.npz', **real)
            truth.append(real['dense_violated'])
            progress.append(real['progress'])
            imagined = {}
            for k in [0.,1.]:
                out = predict(theta, idx, k, 1 if k==0 else int(p.samples.audit_k1), noise_seed(pick['seed'],200000), 'assessment_rows')
                m = segment_metrics(out['readout'][0])
                imagined[k] = m['violated'].mean(-1)
                np.savez_compressed(run.run_dir/f'audit_{j:03d}_k{k:g}.npz', readout=out['readout'][0])
            imagination.append(imagined)
            summaries.append(dict(**pick, real_violations=int(real['dense_violated'].sum()),
                n_roots=len(idx), real_progress=float(real['progress'].mean()),
                imagined_k0=float(imagined[0.].mean()), imagined_k1=float(imagined[1.].mean())))
            print(f'[power] real slot {j+1}/{len(all_picks)}: {summaries[-1]["real_violations"]}/{len(idx)}', flush=True)
        contrasts = {}
        matrices = {}
        for k in [0.,1.]:
            gap, real_delta = [], []
            for seed in map(int,p.search_seeds):
                lo = next(x['slot'] for x in all_picks if x['seed']==seed and x['noise_k']==k and x['population']==16 and x['generation']==1)
                hi = next(x['slot'] for x in all_picks if x['seed']==seed and x['noise_k']==k and x['population']==256 and x['generation']==16)
                gap.append((truth[hi].astype(float)-imagination[hi][k])-(truth[lo].astype(float)-imagination[lo][k]))
                real_delta.append(truth[hi].astype(float)-truth[lo])
            matrices[f'gap_k{k:g}'], matrices[f'real_k{k:g}'] = np.stack(gap), np.stack(real_delta)
        matrices['noise_difference_in_gap_amplification'] = matrices['gap_k1']-matrices['gap_k0']
        for name, values in matrices.items():
            contrasts[name] = pilot_uncertainty(values, seed=int(p.analysis.bootstrap_seed), n_boot=int(p.analysis.bootstrap_replicates))
        np.savez_compressed(run.run_dir/'paired_assessment.npz', **matrices,
            real_violated=np.stack(truth), real_progress=np.stack(progress),
            imagined_k0=np.stack([m[0.] for m in imagination]), imagined_k1=np.stack([m[1.] for m in imagination]),
            baseline_violated=baseline_real['dense_violated'], baseline_progress=baseline_real['progress'], assessment_indices=idx)
        for label in ['fitness','selection','assessment','zero_control']:
            assert charged[label+'_rows'] == p['expected_'+label+'_rows'], (label, charged)
        assert charged['predictor_rows'] == p.expected_total_predictor_rows
        assert charged['real_steps'] == p.expected_real_steps == ex.counters.real_steps
        run.write_json('assessments.json', summaries)
        run.write_json('timing.json', fit_times)
        costs = dict(**charged, imagined_rows=charged['predictor_rows'], renders=ex.counters.renders+64,
            encodes=ex.counters.encodes, gradient_updates=0, CMA_generations=int(p.expected_CMA_generations),
            candidate_evaluations=int(p.expected_candidate_evaluations), wall_s=time.time()-started,
            collection_steps=0, history_frames_reused=256*3)
        run.write_json('pilot.json', dict(contrasts=contrasts, costs=costs,
            real_baseline=dict(violations=int(baseline_real['dense_violated'].sum()), n_roots=len(idx), progress=float(baseline_real['progress'].mean())),
            main_stage5_run=False, rosarl_compared=False, supervisor_approval_claim=False,
            next='Use pilot variance and throughput for review; no automatic final sample-size choice.'))
    except BaseException as exc:
        run.write_json('failure.json', dict(error=type(exc).__name__, message=str(exc), charged=charged,
            frozen_picks=all_picks, note='No silent retry; any partial costs are retained.'))
        raise
    finally:
        model.predict, real_module._step = original_predict, original_step
        if ex is not None:
            ex.close()
        if ctx is not None:
            ctx.close()
    run.finish(build_manifest(run_id=run.run_id, kind='evo-readiness-power-pilot-v1', costs=costs,
        started_at=started, upstream_revisions=input_revisions(paths), data=dict(sources=sources, roles=role),
        seeds=dict(roles=p.role_seed, search=list(p.search_seeds), bootstrap=p.analysis.bootstrap_seed),
        extra=dict(code_sha256={name:file_sha256(ROOT/name) for name in files}, main_stage5=False,
            no_rosarl_comparison=True, all_roots_previously_inspected=True)), upload=False,
        readme=f'# {run.run_id}\n\nBounded development search/variance pilot.\n')
    print(json.dumps(dict(contrasts=contrasts,costs=costs)), flush=True)


if __name__ == '__main__':
    main()
