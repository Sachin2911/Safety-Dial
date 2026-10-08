#!/usr/bin/env python3
"""Validation-selected averages of existing independently trained visual controllers."""
from evo_clean_targets_diagnostic import *
from helpers.evoEnsemblePolicy import EnsembleHistoryPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    started = time.time()
    cfg = load_stage_config('stage2_ensemble')
    hashes = json.loads((ROOT/cfg.ensemble.source_hashes).read_text())
    for name, files in hashes.items():
        assert all(file_sha256(ROOT/'runs'/name/file) == digest for file, digest in files.items())
    control = ROOT/'runs'/cfg.ensemble.source_runs.recorded
    control_selection = json.loads((control/'selection.json').read_text())
    minimum_progress = control_selection['minimum_progress']
    members, arrays, shared = {}, {}, None
    for key, source in cfg.ensemble.source_runs.items():
        run_source = ROOT/'runs'/source
        rows = json.loads((run_source/'selection.json').read_text())['candidates']
        if key == 'recorded':
            rows = [r for r in rows if r['arm'] == 'visual_history']
        members[key], arrays[key] = [], []
        for seed in cfg.ensemble.seeds:
            row = min([r for r in rows if r['seed'] == seed], key=lambda r: candidate_key(r, minimum_progress))
            prefix = 'visual_history' if key == 'recorded' else 'clean'
            name = f'{prefix}_seed_{seed}_epoch_{row["epoch"]}.npz'
            path = run_source/name
            with np.load(path) as f:
                state = {k: f[k] for k in f.files}
            if shared is None:
                shared = state
            for field in ['feature_mean', 'feature_std', 'action_mean', 'action_std', 'hidden', 'use_history']:
                assert np.array_equal(state[field], shared[field])
            members[key].append(dict(source_run=source, file=name, sha256=hashes[source][name], validation=row))
            arrays[key].append(state['theta'])
    run = EvoRun.create(cfg, 's2-ensemble', args.run_id)
    run.write_json('member_selection.json', dict(members=members, minimum_progress=minimum_progress,
                                                selection_uses_development=False))
    path_members = dict(recorded_three=arrays['recorded'], clean_three=arrays['clean'],
                        combined_six=arrays['recorded']+arrays['clean'])
    policies, thetas = {}, {}
    for arm, values in path_members.items():
        policies[arm] = EnsembleHistoryPolicy(HistoryBlockPolicy.from_state(shared, device='cuda'), len(values))
        thetas[arm] = np.concatenate(values)
        np.savez(run.run_dir/f'{arm}_policy.npz', theta=thetas[arm], **policies[arm].state())
    paths = fetch_inputs(cfg.inputs)
    torch.set_num_threads(8)
    model, scaler = load_walker_model(paths['model'], 'cuda')
    probe, _ = load_probe(paths['probes']/'walker_mlp.pt', 'cuda')
    imaginer = HistoryImaginer(model, scaler, probe, device='cuda')
    ctx = RenderContext()
    verify_render_fingerprint(ctx, paths['data']/'setA.h5')
    val = RootSet.from_roots('setA_validation_control', load_roots_json(control/'validation_roots.json'))
    val = encode_histories(val, imaginer.encode, ctx)
    torch.set_num_threads(1)
    counters, rows = Counters(), []
    for arm in cfg.ensemble.arms:
        executor = HistoryRealExecutor(model, scaler, probe, policies[arm], device='cuda', ctx=ctx,
                                       max_envs=64, encode_batch=64)
        try:
            out = executor.run(thetas[arm][None], val.roots, z_hist=val.z_hist, record_qpos=True)
            counters.add(executor.counters)
        finally:
            executor.close()
        rows.append(dict(arm=arm, n_params=policies[arm].n_params, **brief(out)))
        np.savez_compressed(run.run_dir/f'validation_{arm}.npz',
            **{k: out[k] for k in ['qpos', 'qvel', 'actions', 'dense_violated', 'dense_first_step', 'progress']})
        print(f'[ensemble] {arm}: {rows[-1]["violations"]}/64 validation', flush=True)
    selected = min(rows, key=lambda r: (r['progress_mean'] < minimum_progress, r['violations'],
                                       -r['progress_mean'], r['arm']))
    winner = selected['arm']
    run.write_json('selection.json', dict(candidates=rows, selected=selected,
        minimum_progress=minimum_progress, recorded=control_selection['recorded']))
    shutil.copy2(run.run_dir/f'{winner}_policy.npz', run.run_dir/'selected_policy.npz')
    print(f'[ensemble] winner frozen: {winner}; evaluating development', flush=True)
    torch.set_num_threads(8)
    dev = encode_histories(tuning_roots(paths['banks']), imaginer.encode, ctx)
    torch.set_num_threads(1)
    executor = HistoryRealExecutor(model, scaler, probe, policies[winner], device='cuda', ctx=ctx,
                                   max_envs=24, encode_batch=64)
    try:
        out = executor.run(thetas[winner][None], dev.roots, z_hist=dev.z_hist, record_qpos=True)
        counters.add(executor.counters)
        imagined = segment_metrics(imaginer.rollout_policies(policies[winner], thetas[winner][None],
            dev.z_hist, dev.hist_blocks)['readout'][0, :, 0])
    finally:
        executor.close()
        ctx.close()
    np.savez_compressed(run.run_dir/'real_selected.npz', **out)
    old = json.loads((ROOT/'runs'/cfg.source.run_id/'development_rows.json').read_text())
    assert dev.root_ids == [r['root_id'] for r in old]
    family = np.array([r['family'] for r in old])
    summary = dict(**brief(out), imagined_violations=int(imagined['violated'].sum()),
                   real_readout_violations=int(segment_metrics(out['readout'])['violated'].sum()))
    by_family = {f: brief({k: v[family == f] for k, v in out.items()}) for f in ['PPO', 'PPOLag']}
    costs = dict(real_steps=counters.real_steps, renders=counters.renders+264+64,
        encodes=counters.encodes+264, imagined_rows=imaginer.counters.imagined_rows,
        optimizer_updates=0, reused_sources=list(hashes), wall_s=time.time()-started)
    assert costs['real_steps'] == 21600
    run.write_json('diagnostic.json', dict(selected=winner, development=summary, by_family=by_family,
                                          costs=costs, gate0_run=False))
    run.write_json('development_rows.json', [dict(root_id=r['root_id'], episode=r['episode'],
        family=r['family'], recorded=r['recorded'], violated=bool(out['dense_violated'][i]),
        first_unsafe_step=int(out['dense_first_step'][i]), progress=float(out['progress'][i]))
        for i, r in enumerate(old)])
    files = ['configs/evo/stage2_ensemble.yaml', 'configs/evo/ensemble_sources.json',
        'experiments/helpers/evoEnsemblePolicy.py', 'experiments/scripts/evo_ensemble_diagnostic.py',
        'experiments/tests/test_evo_ensemble.py']
    for name in files:
        p = run.run_dir/'source'/name
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, p)
    run.finish(build_manifest(run_id=run.run_id, kind='evo-ensemble-diagnostic', costs=costs,
        started_at=started, upstream_revisions=input_revisions(paths), data=dict(source_hashes=hashes,
            validation_digest=rootset_digest(val), development_digest=rootset_digest(dev)),
        extra=dict(code_sha256={k: file_sha256(ROOT/k) for k in files}, gate0_run=False)),
        upload=False, readme=f'# {run.run_id}\n\nUniform functional ensemble comparison.\n')
    print(json.dumps(dict(selected=winner, development=summary, by_family=by_family, costs=costs)), flush=True)


if __name__ == '__main__':
    main()
