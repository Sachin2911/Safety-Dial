#!/usr/bin/env python3
"""Bounded noise audit on archived traces; no new real interaction or model fitting."""
from evo_clean_targets_diagnostic import *
from helpers.evoImagine import one_step_residuals, open_loop_errors, sigma_from_residuals
from helpers.evoNoiseDiagnostic import calibration_split, residual_summary, trace_windows
from helpers.evoResidualPolicy import ResidualHistoryPolicy
from helpers.evoTransferPilot import rank_audit, selected_effect
from helpers.evoRanking import rank_candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    started = time.time()
    cfg = load_stage_config('stage3_noise_diagnostic')
    settings = cfg.noise_diagnostic
    baseline = ROOT/'runs'/settings.baseline_run
    transfer = ROOT/'runs'/settings.transfer_run
    refs = json.loads((ROOT/settings.sources).read_text())
    for name, ref in refs.items():
        source = ROOT/'runs'/name
        assert json.loads((source/'hf_upload.json').read_text())['revision'] == ref['revision']
        assert all(file_sha256(source/file) == digest for file, digest in ref['files'].items())
    roots = load_roots_json(baseline/'roots.json')
    split = json.loads((transfer/'root_split.json').read_text())
    evaluation_idx = np.asarray(split['source_indices'])
    fit_idx, check_idx = calibration_split(len(roots), evaluation_idx,
        n_fit=int(settings.calibration_fit_episodes), n_check=int(settings.calibration_check_episodes),
        seed=int(settings.role_seed))
    assert len({r.episode for r in roots}) == len(roots)
    with np.load(baseline/'history_latents.npz') as f:
        zh, history_actions = f['z_hist'], f['hist_blocks']
    with np.load(baseline/'real_controller.npz') as f:
        real_z, actions = f['z'], f['actions']
    with np.load(transfer/'residual_policies.npz') as f:
        state = {k: f[k] for k in f.files}
    torch.set_num_threads(1)
    policy = ResidualHistoryPolicy.from_state(state, device='cuda')
    theta = state['theta']
    assert theta.shape == (101, 1021)
    meta = json.loads((transfer/'candidate_plan.json').read_text())
    paths = fetch_inputs(cfg.inputs)
    model, scaler = load_walker_model(paths['model'], 'cuda')
    probe, _ = load_probe(paths['probes']/'walker_mlp.pt', 'cuda')
    im = HistoryImaginer(model, scaler, probe, device='cuda')
    run = EvoRun.create(cfg, 's3-noise-diagnostic', args.run_id)
    run.write_json('episode_roles.json', dict(fit_indices=fit_idx, check_indices=check_idx,
        transfer_indices=evaluation_idx, all_roles='development', source_run=baseline.name,
        disjoint_whole_episodes=True, includes_unsafe_real_frames=True))
    costs = dict(real_steps=0, optimizer_updates=0, renders=0, encodes=0,
                 calibration_predictor_rows=0, open_loop_predictor_rows=0)
    calibration = {}
    residual_arrays = {}
    try:
        for label, idx in [('fit', fit_idx), ('check', check_idx)]:
            zw, aw, zs, acts = trace_windows(zh[idx], history_actions[idx], real_z[idx], actions[idx])
            normalized = im.flat(aw.reshape(len(aw), 3, 10, 6))
            residual = one_step_residuals(model, zw, normalized, batch=int(settings.predictor_batch))
            costs['calibration_predictor_rows'] += len(zw)
            residual_arrays[label] = residual.cpu().numpy()
            calibration[label] = residual_summary(residual_arrays[label])
            if label == 'fit':
                sigma = sigma_from_residuals(residual)
            else:
                norm_seq = im.flat(acts.reshape(len(idx), 12, 10, 6))
                calibration['check_open_loop_rms_by_block'] = open_loop_errors(
                    model, zs, norm_seq, 10, batch=int(settings.predictor_batch))
                costs['open_loop_predictor_rows'] += len(idx)*10
        assert np.isfinite(sigma).all() and (sigma >= 0).all()
        im.sigma = sigma
        calibration.update(sigma_rms=float(np.sqrt(np.mean(sigma*sigma))),
            sigma_min=float(sigma.min()), sigma_max=float(sigma.max()),
            fit_rows_are_correlated_within_64_episodes=True,
            bias_subtraction=False, mean_noise_zero=True)
        np.savez_compressed(run.run_dir/'calibration.npz', sigma=sigma,
            fit_residuals=residual_arrays['fit'], check_residuals=residual_arrays['check'],
            fit_indices=fit_idx, check_indices=check_idx)
        run.write_json('calibration.json', calibration)
        z_eval, h_eval = zh[evaluation_idx], history_actions[evaluation_idx]
        zero = im.rollout_policies(policy, theta[:1], z_eval, h_eval,
                                   max_batch=int(settings.max_batch))
        with np.load(transfer/'imagined_000.npz') as old:
            same = {k:bool(np.array_equal(zero[k], old[k])) for k in ['actions', 'readout', 'root_readout']}
        run.write_json('zero_control.json', same)
        assert all(same.values()), 'zero-noise archived baseline changed'
        print('[noise] calibration and zero-control complete', flush=True)
        level_arrays = {}
        selections = {}
        for level in settings.levels:
            k = float(level)
            outputs, metrics, baseline_actions = [], [], None
            for j in range(len(theta)):
                out = im.rollout_policies(policy, theta[j:j+1], z_eval, h_eval,
                    n_samples=int(settings.samples_per_root), k=k, seed=int(settings.noise_seed),
                    max_batch=int(settings.max_batch))
                outputs.append(out['readout'][0])
                metrics.append(segment_metrics(out['readout'][0]))
                # No need to rerun a selected policy: retain actions until its selection freezes.
                if baseline_actions is None:
                    all_actions = np.empty((len(theta), *out['actions'][0].shape), np.float32)
                    baseline_actions = out['actions'][0].copy()
                all_actions[j] = out['actions'][0]
            flags = np.stack([m['violated'] for m in metrics])
            progress = np.stack([m['ret'] for m in metrics])
            ret_pre = np.stack([m['ret_pre'] for m in metrics])
            ranks, info = rank_candidates('deb', violated=flags[:, :32].reshape(len(theta), -1),
                ret=progress[:, :32].reshape(len(theta), -1), ret_pre=ret_pre[:, :32].reshape(len(theta), -1))
            winner = int(np.argmin(ranks))
            selections[str(k)] = dict(winner=winner, metadata=meta[winner],
                selection_roots=32, assessment_roots=32, noise_samples=int(settings.samples_per_root),
                predicted_failure=info['p'], predicted_progress=info['mean_ret'], ranking=ranks,
                frozen_before_loading_real_scores=True,
                caveat='Real outcomes were observed in earlier pilot; this remains development analysis.')
            np.savez_compressed(run.run_dir/f'noise_k{k:g}.npz', readout=np.stack(outputs),
                violated=flags, progress=progress, ret_pre=ret_pre,
                baseline_actions=baseline_actions, selected_actions=all_actions[winner],
                selected_index=winner)
            level_arrays[str(k)] = (flags.mean(-1), progress.mean(-1))
            run.write_json('selection.json', selections)
            run.write_json('execution_progress.json', dict(completed_levels=list(level_arrays),
                predictor_rows=im.counters.imagined_rows, **costs), results=False)
            print(f'[noise] k={k:g}: mean predicted risk {flags.mean():.4f}; winner {winner}', flush=True)
        # Selection for every noise level is frozen before this read. This is still reused development evidence.
        with np.load(transfer/'scores.npz') as f:
            truth = {k:f[k] for k in ['real_viol', 'real_ret', 'episodes', 'imag_viol', 'imag_ret']}
        idx_residual = np.array([j for j, m in enumerate(meta) if m['kind'] != 'attenuation'])
        result = {}
        for label, (risk, progress) in level_arrays.items():
            ranks = {}
            for view, idx in [('all_candidates', np.arange(len(theta))), ('residual_candidates_only', idx_residual)]:
                ranks[view] = dict(violation=rank_audit(risk[idx], truth['real_viol'][idx], truth['episodes'],
                    n_boot=int(settings.bootstrap_replicates), seed=int(settings.bootstrap_seed)),
                    progress=rank_audit(progress[idx], truth['real_ret'][idx], truth['episodes'],
                    n_boot=int(settings.bootstrap_replicates), seed=int(settings.bootstrap_seed)))
            winner = selections[label]['winner']
            result[label] = dict(ranking=ranks,
                transfer_screen_pass=all(v['transfer_screen_pass'] for v in ranks['residual_candidates_only'].values()),
                mean_predicted_failure=float(risk[idx_residual].mean()),
                residual_candidate_risk_range=[float(risk[idx_residual].mean(-1).min()), float(risk[idx_residual].mean(-1).max())],
                selected_candidate=winner,
                selected_all64_real_violations=int(truth['real_viol'][winner].sum()),
                selected_all64_real_progress=float(truth['real_ret'][winner].mean()),
                assessment_effect=selected_effect(truth['real_viol'], risk, truth['real_ret'],
                    selected=winner, assessment_indices=np.arange(32, 64), episodes=truth['episodes'],
                    n_boot=int(settings.bootstrap_replicates), seed=int(settings.bootstrap_seed)))
        costs.update(noisy_and_zero_predictor_rows=im.counters.imagined_rows,
            imagined_rows=im.counters.imagined_rows+costs['calibration_predictor_rows']+costs['open_loop_predictor_rows'],
            wall_s=time.time()-started,
            reused_real_steps=0, source_costs_already_charged_to_original_runs=True)
        assert costs['imagined_rows'] == settings.total_predictor_rows
        run.write_json('diagnostic.json', dict(calibration=calibration, levels=result, costs=costs,
            original_gate0_passed=False, official_gate1_run=False, full_fix_grid_authorized=False,
            promising_levels=[k for k, v in result.items() if v['transfer_screen_pass']],
            all_results_development_only=True))
    except BaseException as exc:
        run.write_json('failure.json', dict(error=type(exc).__name__, message=str(exc), costs=costs,
            imagined_rollout_rows=im.counters.imagined_rows))
        raise
    files = ['configs/evo/stage3_noise_diagnostic.yaml', 'configs/evo/noise_diagnostic_sources.json',
        'experiments/helpers/evoNoiseDiagnostic.py', 'experiments/scripts/evo_noise_diagnostic.py',
        'experiments/tests/test_evo_noise_diagnostic.py']
    for name in files:
        dest = run.run_dir/'source'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, dest)
    run.finish(build_manifest(run_id=run.run_id, kind='evo-noise-diagnostic-v1', costs=costs,
        started_at=started, upstream_revisions=input_revisions(paths), data=refs,
        seeds=dict(roles=settings.role_seed, noise=settings.noise_seed, bootstrap=settings.bootstrap_seed),
        extra=dict(code_sha256={name:file_sha256(ROOT/name) for name in files},
                   all_results_development_only=True, no_real_interaction=True)), upload=False,
        readme=f'# {run.run_id}\n\nBounded cached-data noise diagnostic.\n')
    print(json.dumps(dict(levels=result, costs=costs)), flush=True)


if __name__ == '__main__':
    main()
