#!/usr/bin/env python3
"""Fixed residual-controller transfer pilot, with a frozen imagined-only winner."""
from evo_clean_targets_diagnostic import *
from helpers.evoEnsemblePolicy import EnsembleHistoryPolicy
from helpers.evoResidualPolicy import ResidualHistoryPolicy, residual_candidates
from helpers.evoTransferPilot import rank_audit, selected_effect
from helpers.evoRanking import rank_candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    started = time.time()
    cfg = load_stage_config('stage4_transfer_pilot')
    source = ROOT/'runs'/cfg.pilot.source_run
    hashes = json.loads((ROOT/cfg.pilot.source_hashes).read_text())
    assert all(file_sha256(source/name) == digest for name, digest in hashes.items())
    assert json.loads((source/'audit.json').read_text())['readiness']['research_pilot_ready']
    all_roots = load_roots_json(source/'roots.json')
    take = np.random.default_rng(int(cfg.pilot.root_seed)).permutation(len(all_roots))[:int(cfg.pilot.n_roots)]
    with np.load(source/'history_latents.npz') as f:
        zh = f['z_hist'][take]
    bank = RootSet.from_roots('readiness_development_transfer_pilot', [all_roots[i] for i in take], z_hist=zh)
    n_select = int(cfg.pilot.selection_roots)
    selection_idx, assessment_idx = np.arange(n_select), np.arange(n_select, len(bank))
    assert len(bank) == 64 and len(np.unique(bank.source_episode)) == 64
    assert not set(bank.source_episode[selection_idx]).intersection(bank.source_episode[assessment_idx])
    with np.load(source/'frozen_policy.npz') as f:
        frozen = {k: f[k] for k in f.files}
    torch.set_num_threads(1)
    base = EnsembleHistoryPolicy.from_state(frozen, device='cuda')
    projection = np.random.default_rng(int(cfg.residual.projection_seed)).normal(
        size=(444, int(cfg.residual.rank))).astype(np.float32)/np.sqrt(444)
    policy = ResidualHistoryPolicy(base, frozen['theta'], projection)
    zcal = torch.as_tensor(bank.z_hist[selection_idx], device='cuda')
    calibration = policy.features(zcal[:, -1], zcal[:, -2], bank.hist_blocks[selection_idx, -1].reshape(n_select, 60))
    theta, meta = residual_candidates(policy, calibration, directions=int(cfg.residual.directions),
        rms_scales=list(cfg.residual.target_raw_action_rms), gains=list(cfg.residual.base_gains),
        seed=int(cfg.residual.candidate_seed))
    assert theta.shape == (cfg.residual.candidate_count, cfg.residual.parameters)
    run = EvoRun.create(cfg, 's4-transfer-pilot', args.run_id)
    np.savez(run.run_dir/'residual_policies.npz', theta=theta, **policy.state())
    with torch.inference_mode():
        base_actions = policy.act(theta[:1], calibration[None])[0]
        for j, m in enumerate(meta):
            actual = policy.act(theta[j:j+1], calibration[None])[0]
            m.update(candidate=j, calibration_clipped_action_rms=float((actual-base_actions).square().mean().sqrt()))
    run.write_json('candidate_plan.json', meta)
    run.write_json('root_split.json', dict(source_run=source.name, source_indices=take,
        roots=[dict(root_id=r.root_id, source_episode=r.episode,
                    role='imagined_selection' if j < n_select else 'paired_assessment') for j, r in enumerate(bank.roots)],
        selection_count=n_select, assessment_count=len(assessment_idx), role_after_run='development'))
    paths = fetch_inputs(cfg.inputs)
    model, scaler = load_walker_model(paths['model'], 'cuda')
    probe, _ = load_probe(paths['probes']/'walker_mlp.pt', 'cuda')
    im = HistoryImaginer(model, scaler, probe, device='cuda')
    imagined, im_times = [], []
    # Fixed shape and candidate-local calls, even for the baseline.
    for j in range(len(theta)):
        tick = time.perf_counter()
        out = im.rollout_policies(policy, theta[j:j+1], bank.z_hist, bank.hist_blocks)
        torch.cuda.synchronize()
        im_times.append(time.perf_counter()-tick)
        imagined.append(segment_metrics(out['readout'][0, :, 0]))
        np.savez_compressed(run.run_dir/f'imagined_{j:03d}.npz', **out)
    imag_viol = np.stack([v['violated'] for v in imagined])
    imag_ret = np.stack([v['ret'] for v in imagined])
    imag_pre = np.stack([v['ret_pre'] for v in imagined])
    ranks, info = rank_candidates('deb', violated=imag_viol[:, selection_idx], ret=imag_ret[:, selection_idx],
                                 ret_pre=imag_pre[:, selection_idx])
    winner = int(np.argmin(ranks))
    run.write_json('imagined_selection.json', dict(winner=winner, winner_metadata=meta[winner],
        ranking_rule='Deb', selected_on='32 imagined-selection roots', ranking=ranks,
        predicted_failure=info['p'], predicted_progress=info['mean_ret'], frozen_before_real_evaluation=True))
    print(f'[transfer] imagined winner frozen: {winner} {meta[winner]}', flush=True)
    ctx = RenderContext()
    verify_render_fingerprint(ctx, paths['data']/'setA.h5')
    ex = HistoryRealExecutor(model, scaler, probe, policy, device='cuda', ctx=ctx,
                             max_envs=int(cfg.execution.max_envs), encode_batch=int(cfg.execution.encode_batch))
    real_viol, real_ret, readout_viol, readout_ret, real_times = [], [], [], [], []
    baseline_check = None
    try:
        for j in range(len(theta)):
            tick = time.perf_counter()
            out = ex.run(theta[j:j+1], bank.roots, z_hist=bank.z_hist, record_qpos=True)
            torch.cuda.synchronize()
            real_times.append(time.perf_counter()-tick)
            np.savez_compressed(run.run_dir/f'real_{j:03d}.npz', **out)
            if j == 0:
                with np.load(source/'real_controller.npz') as old:
                    baseline_check = {k: bool(np.array_equal(out[k], old[k][take]))
                                      for k in ['qpos', 'qvel', 'actions', 'dense_violated', 'progress', 'readout']}
                run.write_json('baseline_equivalence.json', baseline_check)
                assert all(baseline_check.values()), 'zero residual changed the archived baseline'
            real_viol.append(out['dense_violated'])
            real_ret.append(out['progress'])
            ro = segment_metrics(out['readout'])
            readout_viol.append(ro['violated'])
            readout_ret.append(ro['ret'])
            run.write_json('execution_progress.json', dict(completed_candidates=j+1,
                costs=ex.counters.as_dict(), imagined_rows=im.counters.imagined_rows), results=False)
            if j % 10 == 0 or j+1 == len(theta):
                print(f'[transfer] candidate {j+1}/{len(theta)}: {int(out["dense_violated"].sum())}/64 real, {int(imag_viol[j].sum())}/64 imagined', flush=True)
        counter = ex.counters.as_dict()
    except BaseException as exc:
        run.write_json('failure.json', dict(phase='real_evaluation', error=type(exc).__name__, message=str(exc),
            completed_candidates=len(real_viol), completed_costs=ex.counters.as_dict(),
            imagined_rows=im.counters.imagined_rows))
        raise
    finally:
        ex.close()
        ctx.close()
    real_viol, real_ret, readout_viol, readout_ret = map(np.stack, [real_viol, real_ret, readout_viol, readout_ret])
    np.savez_compressed(run.run_dir/'scores.npz', imag_viol=imag_viol, imag_ret=imag_ret,
        real_viol=real_viol, real_ret=real_ret, readout_viol=readout_viol, readout_ret=readout_ret,
        episodes=bank.source_episode, source_indices=take, theta=theta)
    views = dict(all_candidates=np.arange(len(theta)),
        residual_candidates_only=np.array([j for j, m in enumerate(meta) if m['kind'] != 'attenuation']),
        local_residual_rms_le_0_10=np.array([j for j, m in enumerate(meta)
            if m['kind'] == 'baseline' or (m['kind'] == 'residual' and m['target_raw_rms'] <= .1)]))
    n_boot, seed = int(cfg.analysis.bootstrap_replicates), int(cfg.analysis.bootstrap_seed)
    ranking = {name: dict(progress=rank_audit(imag_ret[idx], real_ret[idx], bank.source_episode, n_boot=n_boot, seed=seed),
        violation=rank_audit(imag_viol[idx], real_viol[idx], bank.source_episode, n_boot=n_boot, seed=seed))
        for name, idx in views.items()}
    effect = selected_effect(real_viol, imag_viol, real_ret, selected=winner,
        assessment_indices=assessment_idx, episodes=bank.source_episode, n_boot=n_boot, seed=seed,
        target_effect=float(cfg.analysis.root_only_effect_for_power))
    summaries = []
    for j, m in enumerate(meta):
        summaries.append(dict(**m, real_violations=int(real_viol[j].sum()), imagined_violations=int(imag_viol[j].sum()),
            real_progress=float(real_ret[j].mean()), imagined_progress=float(imag_ret[j].mean()),
            real_readout_violations=int(readout_viol[j].sum())))
    costs = dict(real_steps=counter['real_steps'], imagined_rows=im.counters.imagined_rows,
        renders=counter['renders']+64, encodes=counter['encodes'], optimizer_updates=0,
        collection_steps=0, reused_collection_steps=247351, reused_history_encodes=192,
        wall_s=time.time()-started)
    assert costs['real_steps'] == cfg.execution.simulator_steps and costs['imagined_rows'] == cfg.execution.imagined_rows
    timing = dict(imagined_seconds=float(sum(im_times)), real_seconds=float(sum(real_times)),
        imagined_rows_per_second=float(costs['imagined_rows']/sum(im_times)),
        real_segments_per_second=float(len(theta)*len(bank)/sum(real_times)),
        includes_first_call=True, excludes_setup_and_file_writes=True)
    run.write_json('pilot.json', dict(ranking=ranking, imagined_winner=winner, assessment_effect=effect,
        baseline=summaries[0], selected=summaries[winner], costs=costs, timing=timing,
        original_gate0_passed=False, official_gate1_run=False,
        transfer_screen_pass=all(v['transfer_screen_pass'] for v in ranking['residual_candidates_only'].values())))
    run.write_json('per_policy.json', summaries)
    files = ['configs/evo/stage4_transfer_pilot.yaml', 'configs/evo/transfer_pilot_sources.json',
        'experiments/helpers/evoResidualPolicy.py', 'experiments/helpers/evoTransferPilot.py',
        'experiments/scripts/evo_transfer_pilot.py', 'experiments/tests/test_evo_transfer_pilot.py']
    for name in files:
        path = run.run_dir/'source'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, path)
    run.finish(build_manifest(run_id=run.run_id, kind='evo-transfer-pilot-v1', costs=costs,
        started_at=started, upstream_revisions=input_revisions(paths), data=dict(source_run=source.name,
            source_hashes=hashes, rootset_digest=rootset_digest(bank)), seeds=dict(root=cfg.pilot.root_seed,
                projection=cfg.residual.projection_seed, candidates=cfg.residual.candidate_seed,
                bootstrap=cfg.analysis.bootstrap_seed), extra=dict(code_sha256={k:file_sha256(ROOT/k) for k in files},
                    official_gate1_run=False, original_gate0_changed=False)),
        upload=False, readme=f'# {run.run_id}\n\nFixed residual-controller transfer pilot.\n')
    print(json.dumps(dict(ranking=ranking, assessment_effect=effect, timing=timing, costs=costs)), flush=True)


if __name__ == '__main__':
    main()
