#!/usr/bin/env python3
"""One paired teacher-label aggregation round, fit episodes only and development evaluation."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))
from helpers.threads import pin_threads
pin_threads()
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

from evo_mlp_diagnostic import brief, load_features
from helpers.evoAggregation import augmentation_indices, select_training_starts, teacher_label
from helpers.evoCoverage import fit_coverage_seed
from helpers.evoImagine import ClosedLoopImaginer, Counters
from helpers.evoInputs import EvoRun, fetch_inputs, input_revisions, load_stage_config
from helpers.evoMlpFit import select_fit
from helpers.evoMlpPolicy import MLPBlockPolicy
from helpers.evoMlpReal import MLPRealExecutor
from helpers.evoRanking import segment_metrics
from helpers.evoReal import _physics_env
from helpers.evoRoots import RootSet, encode_histories, rootset_digest, tuning_roots
from helpers.evoStats import clustered_mean_ci
from helpers.evoTeacherDiagnostic import teacher_branch
from helpers.locoData import RenderContext
from helpers.poseProbes import load_probe
from helpers.runManifest import build_manifest, file_sha256
from helpers.walkerAssets import load_exported_actor
from helpers.walkerLewm import load_walker_model
from helpers.walkerRules import roots_from_episode
from helpers.walkerValidation import verify_render_fingerprint


def features_from_rollout(policy, history, ends):
    z = np.concatenate([history[:, -2:], ends], axis=1)
    with torch.inference_mode():
        return np.stack([policy.features(torch.as_tensor(z[:, b+1], device=policy.device),
                                         torch.as_tensor(z[:, b], device=policy.device)).cpu().numpy()
                         for b in range(10)], axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    started = time.time()
    cfg = load_stage_config('stage2_aggregation')
    source = ROOT / 'runs' / cfg.source.run_id
    receipt = json.loads((source / 'hf_upload.json').read_text())
    assert receipt['revision'] == cfg.source.revision
    old_results = ROOT / 'docs/evoPlan/results/stage2-coverage' / cfg.source.run_id
    request = json.loads((old_results / 'archive_request.json').read_text())
    hashes = {f['path']: f['sha256'] for f in request['files']}
    consumed = ['ppo_features.npz', 'mixed_policy.npz', 'sample_indices.npz',
                'data_plan.json', 'development_rows.json', 'manifest.json', 'real_mixed.npz']
    assert all(file_sha256(source / name) == hashes[name] for name in consumed)
    _, Xlag, Ylag, Elag, _, state, _ = load_features(cfg)
    with np.load(source / 'ppo_features.npz') as f:
        Xppo = ((f['X'].astype(np.float32) - state['feature_mean']) / state['feature_std']).astype(np.float32)
        Yppo, Eppo = f['Y'].astype(np.float32), f['episode']
    with np.load(source / 'sample_indices.npz') as f:
        indices = {k: f[k] for k in f.files}
    with np.load(source / 'mixed_policy.npz') as f:
        frozen_state = {k: f[k] for k in f.files}
    assert all(np.array_equal(state[k], frozen_state[k]) for k in
               ['feature_mean', 'feature_std', 'action_mean', 'action_std'])
    plan = json.loads((source / 'data_plan.json').read_text())
    old_rows = json.loads((source / 'development_rows.json').read_text())
    policy = MLPBlockPolicy.from_state(frozen_state, device='cuda')
    paths = fetch_inputs(cfg.inputs)
    run = EvoRun.create(cfg, 's2-aggregation', args.run_id)

    # Select roots by fixed seed and episode length alone, before querying outcomes.
    roots, teachers, family = [], [], []
    with h5py.File(paths['data'] / 'setA.h5', 'r') as f:
        offsets, lengths = f['ep_offset'][:], f['ep_len'][:]
        ids = {int(v): k for k, v in json.loads(f.attrs['policy_ids']).items()}
        for j, (name, key) in enumerate([('PPOLag', 'lag'), ('PPO', 'ppo')]):
            starts = select_training_starts(plan[f'fit_episodes_selected_{key}'],
                plan[f'{key}_validation_episodes'], lengths,
                count=int(cfg.collection.roots_per_family), seed=int(cfg.collection.seed)+j)
            for e, t in starts:
                lo, hi = int(offsets[e]), int(offsets[e] + lengths[e])
                ep = {k: f[k][lo:hi] for k in ['qpos', 'qvel', 'action', 'x_velocity']}
                root = roots_from_episode(ep, [t], episode=e, prefix='fit-setA')[0]
                assert root.meta['policy_tape_padding_steps'] == 0
                actor = ids[int(f['policy_id'][lo])]
                assert actor.split('-')[0] == name
                root.meta.update(source_file='setA.h5', role='fit', teacher=actor)
                roots.append(root)
                teachers.append(actor)
                family.append(name)
    fit_roots = RootSet.from_roots('setA_fit_aggregation', roots)
    assert len(fit_roots) == 128 and len(set(fit_roots.source_episode)) == 128
    # IDs are local to their HDF5 file: setA fit and roots.h5 development are separate collections.
    run.write_json('collection_plan.json', dict(source_file='setA.h5', roots=[
        dict(root_id=r.root_id, episode=r.episode, step=r.step, teacher=t, family=f)
        for r, t, f in zip(roots, teachers, family)], source_coverage_plan=plan))
    model, scaler = load_walker_model(paths['model'], 'cuda')
    probe, _ = load_probe(paths['probes'] / 'walker_mlp.pt', 'cuda')
    im = ClosedLoopImaginer(model, scaler, probe, device='cuda')
    ctx = RenderContext()
    verify_render_fingerprint(ctx, paths['data'] / 'setA.h5')
    torch.set_num_threads(1)
    fit_roots = encode_histories(fit_roots, im.encode, ctx)
    ex = MLPRealExecutor(model, scaler, probe, policy, device='cuda', ctx=ctx,
                         max_envs=64, encode_batch=int(cfg.real.encode_batch))
    env = _physics_env()
    actors, actor_hashes = {}, {}
    try:
        print('[aggregation] collecting frozen student trajectories', flush=True)
        student = ex.run(frozen_state['theta'][None], roots, z_hist=fit_roots.z_hist, record_qpos=True)
        teacher_logs = []
        for i, (r, name) in enumerate(zip(roots, teachers)):
            if name not in actors:
                path = paths['policies'] / f'{name}.pt'
                actors[name] = load_exported_actor(path, env.action_space, action_noise=0., device='cpu')
                actor_hashes[name] = file_sha256(path)
            teacher_logs.append(teacher_branch(env, r, r.policy_tape, actors[name], takeover_step=0))
        teacher = {k: np.stack([v[k] for v in teacher_logs]) for k in ['qpos', 'qvel', 'actions']}
        teacher['z'] = np.stack([
            ex._encode_ends(ex._render(teacher['qpos'][:, b], teacher['qvel'][:, b])).cpu().numpy()
            for b in range(10, 101, 10)], axis=1)
        collected, collection_steps, label_arrays = {}, {}, {}
        for label, trajectory in [('teacher_states', teacher), ('student_states', student)]:
            Xnew = features_from_rollout(policy, fit_roots.z_hist, trajectory['z'])
            targets, branches, charged = [], [], 12800
            for i, r in enumerate(roots):
                tape = trajectory['actions'][i].reshape(100, 6)
                target_blocks, branch_ends = [], []
                for b in cfg.collection.boundaries:
                    y, qp, qv, nsteps = teacher_label(env, r, tape, actors[teachers[i]], boundary=int(b))
                    charged += nsteps
                    assert np.array_equal(qp[:b+1], trajectory['qpos'][i, :b+1])
                    assert np.array_equal(qv[:b+1], trajectory['qvel'][i, :b+1])
                    if label == 'teacher_states':
                        assert np.array_equal(y.reshape(10, 6), tape[b:b+10])
                    target_blocks.append(y)
                    branch_ends.append(np.stack([qp[-1], qv[-1]]))
                targets.append(target_blocks)
                branches.append(branch_ends)
            targets = np.asarray(targets, dtype=np.float32)
            collected[label] = (Xnew.reshape(-1, 384), targets.reshape(-1, 60))
            collection_steps[label] = charged
            assert charged == cfg.collection.simulator_steps_per_arm
            label_arrays.update({f'{label}_X': Xnew, f'{label}_Y': targets,
                                 f'{label}_branch_ends': np.asarray(branches)})
            np.savez_compressed(run.run_dir / f'collection_{label}.npz',
                                **{k: trajectory[k] for k in ['qpos', 'qvel', 'actions', 'z']})
            print(f'[aggregation] {label}: {len(Xnew)*10} labels, {charged} simulator steps', flush=True)
        np.savez_compressed(run.run_dir / 'labels.npz', **label_arrays,
                            episode=fit_roots.source_episode, history_z=fit_roots.z_hist)
        collection_counter = ex.counters.as_dict()
    finally:
        env.close()
        ex.close()

    # Match every original and augmentation index between arms, by family.
    new_indices, old_indices = {}, {}
    for j, key in enumerate(['lag', 'ppo']):
        n = len(indices[f'mixed_{key}'])
        old_indices[key], new_indices[key] = augmentation_indices(n, 640,
            fraction=float(cfg.augmentation.fraction), seed=int(cfg.augmentation.seed)+j)
    np.savez(run.run_dir / 'mixture_indices.npz',
             **{f'old_{k}': v for k, v in old_indices.items()},
             **{f'new_{k}': v for k, v in new_indices.items()})
    datasets = {}
    for label, (Xnew, Ynew) in collected.items():
        xs, ys = [], []
        for j, (key, Xold, Yold, Eold) in enumerate([('lag', Xlag, Ylag, Elag), ('ppo', Xppo, Yppo, Eppo)]):
            old = indices[f'mixed_{key}'][old_indices[key]]
            new = new_indices[key] + j * 640
            assert not set(Eold[old]).intersection(plan[f'{key}_validation_episodes'])
            xs.extend([Xold[old], Xnew[new]])
            ys.extend([Yold[old], Ynew[new]])
        datasets[label] = (np.concatenate(xs), np.concatenate(ys))
    assert all(len(x) == 45986 for x, _ in datasets.values())
    Xval = torch.as_tensor(np.concatenate([Xlag[indices['validation_lag']],
                                          Xppo[indices['validation_ppo']]]), device='cuda')
    Yval = torch.as_tensor(np.concatenate([Ylag[indices['validation_lag']],
                                          Yppo[indices['validation_ppo']]]), device='cuda')
    torch.set_num_threads(8)
    offline, thetas, history = {}, {}, []
    for label, (xf, yf) in datasets.items():
        Xfit, Yfit = [torch.as_tensor(x, device='cuda') for x in [xf, yf]]
        trials, updates = [], 0
        for seed in cfg.training.seeds:
            best, hist, n = fit_coverage_seed(policy, Xfit, Yfit, Xval, Yval,
                seed=int(seed), epochs=int(cfg.training.epochs), batch_size=int(cfg.training.batch_size),
                lr=float(cfg.training.learning_rate), weight_decay=float(cfg.training.weight_decay),
                val_weights=indices['validation_weights'],
                progress=lambda r, label=label: print(
                    f"[aggregation] {label} seed {r['seed']} epoch {r['epoch']} validation {r['val_mse']:.6f}", flush=True))
            trials.append(best)
            updates += n
            history.extend([dict(arm=label, **r) for r in hist])
            np.savez(run.run_dir / f'{label}_seed_{seed}.npz', theta=best['theta'], **policy.state(),
                     seed=best['seed'], epoch=best['epoch'])
        best = select_fit(trials)
        thetas[label] = best['theta']
        np.savez(run.run_dir / f'{label}_policy.npz', theta=best['theta'], **policy.state(),
                 seed=best['seed'], epoch=best['epoch'])
        offline[label] = dict(selected={k: v for k, v in best.items() if k != 'theta'},
            trials=[{k: v for k, v in t.items() if k != 'theta'} for t in trials],
            samples=45986, original_samples=34490, augmented_draws=11496,
            unique_new_labels=1280, optimizer_updates=updates)
        assert updates == 13500
    run.write_json('fit.json', offline)
    run.write_json('training_history.json', history)
    print('[aggregation] selection frozen; evaluating development roots', flush=True)
    torch.set_num_threads(1)
    dev = tuning_roots(paths['banks'])
    assert dev.root_ids == [r['root_id'] for r in old_rows] and len(dev) == 24
    torch.set_num_threads(8)  # Match coverage history encoding before its single-thread executor.
    dev = encode_histories(dev, im.encode, ctx)
    torch.set_num_threads(1)
    real, imagined, counters = {}, {}, Counters()
    try:
        for label, theta in thetas.items():
            executor = MLPRealExecutor(model, scaler, probe, policy, device='cuda', ctx=ctx,
                max_envs=int(cfg.real.max_envs), encode_batch=int(cfg.real.encode_batch))
            try:
                real[label] = executor.run(theta[None], dev.roots, record_qpos=True, z_hist=dev.z_hist)
                counters.add(executor.counters)
            finally:
                executor.close()
            np.savez_compressed(run.run_dir / f'real_{label}.npz', **real[label])
            imagined[label] = segment_metrics(im.rollout_policies(policy, theta[None],
                dev.z_hist, dev.hist_blocks)['readout'][0, :, 0])
    finally:
        ctx.close()
    summary = {label: {**brief(out), 'imagined_violations': int(imagined[label]['violated'].sum()),
               'real_readout_violations': int(segment_metrics(out['readout'])['violated'].sum())}
               for label, out in real.items()}
    families = np.array([r['family'] for r in old_rows])
    by_family = {family: {label: brief({k: v[families == family] for k, v in out.items()})
                          for label, out in real.items()} for family in ['PPO', 'PPOLag']}
    paired = {key: clustered_mean_ci(
        real['student_states'][field].astype(float) - real['teacher_states'][field].astype(float),
        dev.source_episode) for key, field in [('violation_difference', 'dense_violated'),
                                               ('progress_difference', 'progress')]}
    rows = [{**{k: old[k] for k in ['root_id', 'episode', 'step', 'family', 'recorded']},
             'source_mixed': old['mixed'], **{label: dict(
                 violated=bool(out['dense_violated'][i]), progress=float(out['progress'][i]),
                 first_unsafe_step=int(out['dense_first_step'][i]),
                 imagined_violated=bool(imagined[label]['violated'][i]))
                 for label, out in real.items()}} for i, old in enumerate(old_rows)]
    costs = dict(real_steps=sum(collection_steps.values()) + counters.real_steps,
        collection_steps_per_arm=collection_steps, development_real_steps=counters.real_steps,
        collection_executor=collection_counter, development_executor=counters.as_dict(),
        history_renders=3*(128+24), fingerprint_renders=64,
        total_renders=collection_counter['renders']+counters.renders+3*(128+24)+64,
        total_encodes=collection_counter['encodes']+counters.encodes+3*(128+24),
        imagined_rows=im.counters.imagined_rows, optimizer_updates=27000,
        original_outcomes='archived source mixed and recorded results reused; no new steps',
        wall_s=time.time()-started)
    assert costs['real_steps'] == 171200
    report = dict(run_id=run.run_id, development=summary, by_family=by_family, paired=paired,
        offline=offline, costs=costs, gate0_run=False, rootset_digest=rootset_digest(dev),
        source_mixed=dict(violations=sum(r['mixed']['violated'] for r in old_rows),
                         progress_mean=float(np.mean([r['mixed']['progress'] for r in old_rows]))),
        recorded=dict(violations=sum(r['recorded']['violated'] for r in old_rows),
                      progress_mean=float(np.mean([r['recorded']['progress'] for r in old_rows]))))
    run.write_json('diagnostic.json', report)
    run.write_json('development_rows.json', rows)
    files = ['configs/evo/stage2_aggregation.yaml', 'experiments/helpers/evoAggregation.py',
             'experiments/scripts/evo_aggregation_diagnostic.py', 'experiments/tests/test_evo_aggregation.py']
    for name in files:
        out = run.run_dir / 'source' / name
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, out)
    manifest = build_manifest(run_id=run.run_id, kind='evo-aggregation-diagnostic',
        started_at=started, seeds=dict(collection=int(cfg.collection.seed), augmentation=int(cfg.augmentation.seed),
                                       fitting=list(cfg.training.seeds)), costs=costs,
        upstream_revisions=input_revisions(paths),
        data=dict(source=receipt, source_files={name: hashes[name] for name in consumed},
                  original_cache=dict(cfg.cache), fit_roots_digest=rootset_digest(fit_roots),
                  development_roots_digest=rootset_digest(dev), actor_sha256=actor_hashes),
        extra=dict(code_sha256={name: file_sha256(ROOT / name) for name in files}, gate0_run=False,
                   upstream_costs=json.loads((source / 'manifest.json').read_text())['costs']))
    run.finish(manifest, upload=False, readme=f'# {run.run_id}\n\nPaired teacher-labelled augmentation. See REPORT.md.\n')
    print(json.dumps(dict(development=summary, by_family=by_family, paired=paired, costs=costs)), flush=True)


if __name__ == '__main__':
    main()
