#!/usr/bin/env python3
"""Frozen Walker controller baseline on fresh whole episodes, with a separate research audit."""
from evo_ensemble_diagnostic import *
from helpers.evoReadiness import baseline_audit, generate_episode, representative_root, reference_arrays
from helpers.evoReal import _physics_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    started = time.time()
    cfg = load_stage_config('stage2_readiness_baseline')
    assert cfg.readiness.freeze_controller and not cfg.readiness.fit_or_select_on_fresh_bank
    frozen = ROOT/'runs'/cfg.frozen_controller.run_id/cfg.frozen_controller.file
    assert file_sha256(frozen) == cfg.frozen_controller.sha256
    with np.load(frozen) as f:
        state = {k: f[k] for k in f.files}
    policy = EnsembleHistoryPolicy.from_state(state, device='cuda')
    theta = state['theta']
    paths = fetch_inputs(cfg.inputs)
    manifest_path = paths['data']/'manifest.json'
    assert file_sha256(manifest_path) == cfg.fresh_bank.actor_pool_manifest_sha256
    historical = json.loads(manifest_path.read_text())['data']['policies']
    names = [r['name'] for r in historical if r['len_mean'] >= 400 and r['return_mean'] > 500]
    assert names == list(cfg.fresh_bank.actor_names)
    run = EvoRun.create(cfg, 's2-readiness-baseline', args.run_id)
    shutil.copy2(frozen, run.run_dir/'frozen_policy.npz')
    episodes_dir = run.run_dir/'episodes'
    episodes_dir.mkdir()
    counter = dict(real_steps=0, teacher_queries=0)
    attempts, roots, hashes, actors = [], [], {}, {}
    rng = np.random.default_rng(int(cfg.fresh_bank.sampling_seed))
    root_rng = np.random.default_rng(int(cfg.fresh_bank.root_sampling_seed))
    torch.set_num_threads(1)
    env = _physics_env()
    try:
        for i in range(int(cfg.fresh_bank.max_attempted_episodes)):
            name = names[int(rng.integers(len(names)))]
            sigma = float(cfg.fresh_bank.action_noise[int(rng.integers(len(cfg.fresh_bank.action_noise)))])
            key = (name, sigma)
            if key not in actors:
                path = paths['policies']/f'{name}.pt'
                actors[key] = load_exported_actor(path, env.action_space, action_noise=sigma, device='cpu')
                hashes[name] = file_sha256(path)
            actor = actors[key]
            noise_seed = int(cfg.fresh_bank.actor_noise_seed_base)+i
            episode_seed = int(cfg.fresh_bank.episode_seed_base)+i
            actor.rng = np.random.default_rng(noise_seed)
            episode = generate_episode(env, actor, seed=episode_seed,
                max_steps=int(cfg.fresh_bank.max_steps_per_episode), counter=counter)
            np.savez_compressed(episodes_dir/f'episode_{i:04d}.npz', **episode,
                                actor=name, action_noise=sigma, episode_seed=episode_seed, noise_seed=noise_seed)
            root = representative_root(episode, episode_id=i, rng=root_rng)
            row = dict(episode=i, actor=name, family=name.split('-')[0], sigma=sigma,
                       episode_seed=episode_seed, noise_seed=noise_seed, steps=len(episode['action']),
                       eligible=root is not None)
            if root is not None:
                root.meta.update(source_file=f'{run.run_id}/episodes/episode_{i:04d}.npz',
                                 role='prospective_baseline_audit', actor=name, sigma=sigma,
                                 family=name.split('-')[0], seed=episode_seed)
                roots.append(root)
                row.update(root_id=root.root_id, root_step=root.step)
            attempts.append(row)
            run.write_json('collection_progress.json', dict(costs=counter, accepted=len(roots),
                                                            attempts=attempts), results=False)
            if len(attempts) % 32 == 0:
                print(f'[readiness] {len(roots)} roots / {len(attempts)} episodes, {counter["real_steps"]} collection steps', flush=True)
            if len(roots) == int(cfg.fresh_bank.n_roots):
                break
        if len(roots) != int(cfg.fresh_bank.n_roots):
            raise RuntimeError('declared episode cap exhausted without enough roots')
    except BaseException as exc:
        run.write_json('failure.json', dict(phase='fresh_collection', error=type(exc).__name__,
                                           message=str(exc), costs=counter, attempts=len(attempts)))
        raise
    finally:
        env.close()
    bank = RootSet.from_roots('fresh_baseline_v1', roots)
    assert len(np.unique(bank.source_episode)) == len(bank)
    run.write_json('roots.json', dict(roots=[r.to_dict() for r in roots]), results=False)
    run.write_json('collection.json', dict(attempts=attempts, costs=counter, roots=len(roots),
        rejected_episodes=len(attempts)-len(roots), source_actor_sha256=hashes,
        rootset_digest=rootset_digest(bank), original_role='prospective_baseline_audit',
        role_after_outcomes='development_only_not_final_search_test'))
    print('[readiness] fresh bank frozen; evaluating fixed controller and recorded tapes', flush=True)
    torch.set_num_threads(8)
    model, scaler = load_walker_model(paths['model'], 'cuda')
    probe, _ = load_probe(paths['probes']/'walker_mlp.pt', 'cuda')
    im = HistoryImaginer(model, scaler, probe, device='cuda')
    ctx = RenderContext()
    verify_render_fingerprint(ctx, paths['data']/'setA.h5')
    bank = encode_histories(bank, im.encode, ctx)
    np.savez_compressed(run.run_dir/'history_latents.npz', z_hist=bank.z_hist, hist_blocks=bank.hist_blocks)
    torch.set_num_threads(1)
    ex = HistoryRealExecutor(model, scaler, probe, policy, device='cuda', ctx=ctx,
                             max_envs=64, encode_batch=64)
    try:
        real = ex.run(theta[None], bank.roots, z_hist=bank.z_hist, record_qpos=True)
        recorded = reference_arrays(ex.run_tapes(bank.roots, np.stack([r.policy_tape for r in bank.roots])))
        counters = ex.counters.as_dict()
        imagined = segment_metrics(im.rollout_policies(policy, theta[None], bank.z_hist,
                                                       bank.hist_blocks)['readout'][0, :, 0])
    finally:
        ex.close()
        ctx.close()
    np.savez_compressed(run.run_dir/'real_controller.npz', **real)
    np.savez_compressed(run.run_dir/'real_recorded.npz', **recorded)
    np.savez_compressed(run.run_dir/'imagined_metrics.npz', **imagined)
    audit = baseline_audit(real['dense_violated'], imagined['violated'], real['progress'],
        recorded['progress'], n_boot=int(cfg.readiness.bootstrap_replicates),
        seed=int(cfg.readiness.bootstrap_seed), min_progress_ratio=float(cfg.readiness.progress_ratio_lower_ci_min),
        max_failure_upper=float(cfg.readiness.baseline_failure_upper_ci_max),
        max_gap_width=float(cfg.readiness.baseline_gap_ci_width_max))
    family = np.array([r.meta['family'] for r in roots])
    by_family = {name: brief({k: v[family == name] for k, v in real.items()}) for name in ['PPO', 'PPOLag']}
    old_gate_limit = max(2*int(recorded['dense_violated'].sum()), 3)
    original_threshold = dict(applied_to='new_whole_episode_bank_not_original_256_roots',
        violation_limit=old_gate_limit, actual_violations=int(real['dense_violated'].sum()),
        counts_pass=bool(real['dense_violated'].sum() <= old_gate_limit),
        progress_pass=bool(real['progress'].mean() >= .5*recorded['progress'].mean()),
        historical_gate0_changed=False)
    costs = dict(real_steps=counter['real_steps']+counters['real_steps'],
        collection=counter, evaluation=counters, renders=counters['renders']+3*len(roots)+64,
        encodes=counters['encodes']+3*len(roots), imagined_rows=im.counters.imagined_rows,
        optimizer_updates=0, wall_s=time.time()-started)
    assert counters['real_steps'] == 2*100*len(roots)
    run.write_json('audit.json', dict(readiness=audit, controller=brief(real), recorded=brief(recorded),
        imagined_violations=int(imagined['violated'].sum()),
        real_readout_violations=int(segment_metrics(real['readout'])['violated'].sum()),
        by_family=by_family, original_threshold=original_threshold, costs=costs))
    run.write_json('rows.json', [dict(root_id=r.root_id, episode=r.episode, actor=r.meta['actor'],
        family=r.meta['family'], real_violated=bool(real['dense_violated'][i]),
        imagined_violated=bool(imagined['violated'][i]), progress=float(real['progress'][i]),
        recorded_violated=bool(recorded['dense_violated'][i]), recorded_progress=float(recorded['progress'][i]))
        for i, r in enumerate(roots)])
    files = ['configs/evo/stage2_readiness_baseline.yaml', 'experiments/helpers/evoReadiness.py',
             'experiments/scripts/evo_readiness_baseline.py', 'experiments/tests/test_evo_readiness.py']
    for name in files:
        path = run.run_dir/'source'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, path)
    run.finish(build_manifest(run_id=run.run_id, kind='evo-readiness-baseline-v1', costs=costs,
        started_at=started, upstream_revisions=input_revisions(paths), data=dict(rootset_digest=rootset_digest(bank),
            frozen_controller=dict(cfg.frozen_controller), original_actor_pool_manifest=file_sha256(manifest_path),
            actors_sha256=hashes), seeds=dict(episode_base=cfg.fresh_bank.episode_seed_base,
                actor_noise_base=cfg.fresh_bank.actor_noise_seed_base, sampling=cfg.fresh_bank.sampling_seed,
                root_sampling=cfg.fresh_bank.root_sampling_seed, bootstrap=cfg.readiness.bootstrap_seed),
        extra=dict(code_sha256={k: file_sha256(ROOT/k) for k in files}, original_gate0_changed=False)),
        upload=False, readme=f'# {run.run_id}\n\nProspective fresh-episode baseline measurement audit.\n')
    print(json.dumps(dict(readiness=audit, controller=brief(real), recorded=brief(recorded), costs=costs)), flush=True)


if __name__ == '__main__':
    main()
