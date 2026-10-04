#!/usr/bin/env python3
"""Broad clean-teacher targets versus an immutable matched recorded-target control."""
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
from omegaconf import OmegaConf

from evo_mlp_diagnostic import brief, load_features
from helpers.evoCleanTargets import cached_steps, check_fit_indices, clean_teacher_block
from helpers.evoHistoryImagine import HistoryImaginer
from helpers.evoHistoryPolicy import HistoryBlockPolicy
from helpers.evoHistoryReal import HistoryRealExecutor
from helpers.evoImagine import Counters
from helpers.evoInformation import candidate_key, fit_candidates
from helpers.evoInputs import EvoRun, fetch_inputs, input_revisions, load_stage_config
from helpers.evoRanking import segment_metrics
from helpers.evoReal import _physics_env
from helpers.evoRoots import RootSet, encode_histories, load_roots_json, rootset_digest, tuning_roots
from helpers.evoStats import clustered_mean_ci
from helpers.locoData import RenderContext
from helpers.poseProbes import load_probe
from helpers.runManifest import array_sha256, build_manifest, file_sha256
from helpers.walkerAssets import load_exported_actor
from helpers.walkerLewm import load_walker_model
from helpers.walkerValidation import verify_render_fingerprint


def verified_source(stage, run_id, revision, consumed):
    source = ROOT / 'runs' / run_id
    receipt = json.loads((source / 'hf_upload.json').read_text())
    assert receipt['revision'] == revision
    request = json.loads((ROOT / 'docs/evoPlan/results' / stage / run_id / 'archive_request.json').read_text())
    hashes = {v['path']: v['sha256'] for v in request['files']}
    assert all(file_sha256(source / name) == hashes[name] for name in consumed)
    return source, dict(receipt=receipt, files_sha256={k: hashes[k] for k in consumed})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    started = time.time()
    cfg = load_stage_config('stage2_clean_targets')
    torch.set_num_threads(8)
    source, source_ref = verified_source('stage2-coverage', cfg.source.run_id, cfg.source.revision,
        ['ppo_features.npz', 'sample_indices.npz', 'data_plan.json', 'development_rows.json'])
    control, control_ref = verified_source('stage2-information-resume', cfg.control.source_run,
        cfg.control.revision, ['original_training_config.yaml', 'selection.json', 'validation_roots.json',
                              'visual_history_policy.npz', 'real_visual_history.npz', 'diagnostic.json',
                              'training_history.json'])
    control_cfg = OmegaConf.load(control / 'original_training_config.yaml')
    assert list(cfg.policy.hidden) == list(control_cfg.policy.hidden)
    for k in ['seeds', 'epochs', 'batch_size', 'learning_rate', 'weight_decay', 'checkpoint_epochs']:
        assert cfg.training[k] == control_cfg.training[k]
    assert file_sha256(control/'validation_roots.json') == cfg.validation_control.roots_sha256
    assert file_sha256(control/'selection.json') == cfg.validation_control.selection_sha256
    past = ROOT/'runs'/cfg.history_cache.run_id/'previous_actions.npz'
    assert file_sha256(past) == cfg.history_cache.previous_actions_sha256
    _, xl, yl, el, _, state, _ = load_features(cfg)
    with np.load(source/'ppo_features.npz') as f:
        xp = ((f['X'].astype(np.float32)-state['feature_mean'])/state['feature_std']).astype(np.float32)
        yp, ep = f['Y'].astype(np.float32), f['episode']
    with np.load(source/'sample_indices.npz') as f:
        idx = {k: f[k] for k in f.files}
    with np.load(past) as f:
        al, ap = [((f[k].reshape(-1, 10, 6)-state['action_mean'])/state['action_std']).reshape(-1, 60)
                  for k in ['lag', 'ppo']]
    def join(a, b, role):
        return np.concatenate([a[idx[f'{role}_lag']], b[idx[f'{role}_ppo']]])
    xf = np.concatenate([join(xl, xp, 'mixed'), join(al, ap, 'mixed')], 1)
    xv = np.concatenate([join(xl, xp, 'validation'), join(al, ap, 'validation')], 1)
    recorded_y, yv = join(yl, yp, 'mixed'), join(yl, yp, 'validation')
    assert len(xf) == cfg.collection.examples and len(xv) == 23640
    with np.load(control/'visual_history_policy.npz') as f:
        control_state = {k: f[k] for k in f.files}
    for k in ['feature_mean', 'feature_std', 'action_mean', 'action_std']:
        assert np.array_equal(state[k], control_state[k])
    policy = HistoryBlockPolicy.from_state(control_state, device='cuda')
    assert policy.use_history and policy.n_params == 374588
    plan = json.loads((source/'data_plan.json').read_text())
    paths = fetch_inputs(cfg.inputs)
    data = paths['data']/'setA.h5'
    run = EvoRun.create(cfg, 's2-clean-targets', args.run_id)
    run.write_json('data_plan.json', dict(fit_examples=len(xf), validation_examples=len(xv),
        fit_feature_sha256=array_sha256(xf), validation_feature_sha256=array_sha256(xv),
        recorded_fit_target_sha256=array_sha256(recorded_y), validation_target_sha256=array_sha256(yv),
        indices_source=source_ref, history_cache=dict(cfg.history_cache), control=control_ref,
        fit_episodes_lag=plan['fit_episodes_selected_lag'], fit_episodes_ppo=plan['fit_episodes_selected_ppo'],
        excluded_validation_lag=plan['lag_validation_episodes'], excluded_validation_ppo=plan['ppo_validation_episodes']))
    collection = dict(real_steps=0, teacher_queries=0)
    teachers, teacher_hashes, clean_arrays, descriptors = {}, {}, [], []
    label_summaries = {}
    torch.set_num_threads(1)
    env = _physics_env()
    try:
        for family, key, episodes, targets in [('PPOLag', 'lag', el, yl), ('PPO', 'ppo', ep, yp)]:
            selected = idx[f'mixed_{key}']
            check_fit_indices(episodes, selected, plan[f'fit_episodes_selected_{key}'],
                              plan[f'{key}_validation_episodes'])
            times = cached_steps(data, episodes, targets)
            selected_e, selected_t = episodes[selected], times[selected]
            n = len(selected)
            clean = np.empty((n, 60), np.float32)
            ends = np.empty((n, 2, 9))
            clearances = np.empty((n, 10))
            progress = np.empty(n)
            teacher_ids = np.empty(n, np.int64)
            with h5py.File(data, 'r') as f:
                offsets, lengths = f['ep_offset'][:], f['ep_len'][:]
                names = {int(v): k for k, v in json.loads(f.attrs['policy_ids']).items()}
                for j, e in enumerate(np.unique(selected_e)):
                    lo, hi = int(offsets[e]), int(offsets[e]+lengths[e])
                    qp, qv = f['qpos'][lo:hi], f['qvel'][lo:hi]
                    ids = f['policy_id'][lo:hi]
                    assert np.all(ids == ids[0])
                    name = names[int(ids[0])]
                    assert name.split('-')[0] == family
                    if name not in teachers:
                        path = paths['policies']/f'{name}.pt'
                        teachers[name] = load_exported_actor(path, env.action_space, action_noise=0., device='cpu')
                        teacher_hashes[name] = file_sha256(path)
                    positions = np.flatnonzero(selected_e == e)
                    for pos in positions:
                        t = int(selected_t[pos])
                        out = clean_teacher_block(env, qp[t], qv[t], teachers[name], collection)
                        clean[pos] = out['actions'].reshape(60)
                        ends[pos] = np.stack([out['qpos'][-1], out['qvel'][-1]])
                        clearances[pos], progress[pos] = out['clearance'], out['progress']
                        teacher_ids[pos] = ids[0]
                    run.write_json('collection_progress.json', dict(**collection, family=family,
                        last_episode=int(e)), results=False)
                    if (j+1) % 20 == 0:
                        print(f'[clean] {family}: {j+1}/120 episodes, {collection["real_steps"]} steps', flush=True)
            np.savez_compressed(run.run_dir/f'labels_{key}.npz', clean=clean, recorded=targets[selected],
                episode=selected_e, step=selected_t, cache_index=selected, teacher_id=teacher_ids,
                branch_ends=ends, clearance=clearances, progress=progress)
            delta = (clean-targets[selected]).reshape(n, 10, 6)
            label_summaries[family] = dict(examples=n, full_block_mse=float(np.square(delta).mean()),
                first_action_mse=float(np.square(delta[:, 0]).mean()),
                unsafe_blocks=int((clearances <= 0).any(axis=1).sum()), progress_mean=float(progress.mean()))
            clean_arrays.append(clean)
            descriptors.extend([dict(episode=int(e), family=family) for e in np.unique(selected_e)])
    except BaseException as exc:
        run.write_json('failure.json', dict(phase='teacher_collection', error=type(exc).__name__,
                                           message=str(exc), costs=collection))
        raise
    finally:
        env.close()
    assert collection['real_steps'] == cfg.collection.simulator_steps
    run.write_json('collection.json', dict(costs=collection, by_family=label_summaries,
                                          teacher_sha256=teacher_hashes, fit_episodes=descriptors))
    yf = np.concatenate(clean_arrays)
    torch.set_num_threads(8)
    tensors = [torch.tensor(x, device='cuda') for x in [xf, yf, xv, yv]]
    candidates, history, updates = [], [], 0
    for seed in cfg.training.seeds:
        c, h, n = fit_candidates(policy, *tensors, idx['validation_weights'], seed=int(seed),
            epochs=int(cfg.training.epochs), checkpoint_epochs=list(cfg.training.checkpoint_epochs),
            batch_size=int(cfg.training.batch_size), lr=float(cfg.training.learning_rate),
            weight_decay=float(cfg.training.weight_decay), progress=lambda r: print(
                f'[clean] seed {r["seed"]} epoch {r["epoch"]} MSE {r["val_mse"]:.5f}', flush=True))
        updates += n
        candidates.extend(c)
        history.extend(h)
        for candidate in c:
            np.savez(run.run_dir/f'clean_seed_{seed}_epoch_{candidate["epoch"]}.npz',
                     theta=candidate['theta'], **policy.state(), seed=seed, epoch=candidate['epoch'])
        run.write_json('training_history.json', history)
    assert updates == cfg.control.original_training_updates
    model, scaler = load_walker_model(paths['model'], 'cuda')
    probe, _ = load_probe(paths['probes']/'walker_mlp.pt', 'cuda')
    imaginer = HistoryImaginer(model, scaler, probe, device='cuda')
    ctx = RenderContext()
    verify_render_fingerprint(ctx, data)
    val = RootSet.from_roots('setA_validation_control', load_roots_json(control/'validation_roots.json'))
    val = encode_histories(val, imaginer.encode, ctx)
    old_selection = json.loads((control/'selection.json').read_text())
    recorded = old_selection['recorded']
    minimum_progress = float(cfg.validation_control.progress_floor)*recorded['progress_mean']
    assert minimum_progress == old_selection['minimum_progress']
    torch.set_num_threads(1)
    counters, validation_rows = Counters(), []
    executor = HistoryRealExecutor(model, scaler, probe, policy, device='cuda', ctx=ctx,
                                   max_envs=64, encode_batch=64)
    try:
        for c in candidates:
            out = executor.run(c['theta'][None], val.roots, z_hist=val.z_hist, record_qpos=True)
            row = dict(seed=c['seed'], epoch=c['epoch'], val_mse=c['val_mse'], **brief(out))
            validation_rows.append(row)
            np.savez_compressed(run.run_dir/f'validation_clean_{c["seed"]}_{c["epoch"]}.npz',
                **{k: out[k] for k in ['qpos', 'qvel', 'actions', 'dense_violated', 'dense_first_step', 'progress']})
            print(f'[clean] validation {c["seed"]}/{c["epoch"]}: {row["violations"]}/64', flush=True)
        counters.add(executor.counters)
    finally:
        executor.close()
    chosen = min(validation_rows, key=lambda r: candidate_key(r, minimum_progress))
    best = next(c for c in candidates if (c['seed'], c['epoch']) == (chosen['seed'], chosen['epoch']))
    np.savez(run.run_dir/'clean_policy.npz', theta=best['theta'], **policy.state(),
             seed=best['seed'], epoch=best['epoch'])
    run.write_json('selection.json', dict(recorded=recorded, minimum_progress=minimum_progress,
        candidates=validation_rows, selected=chosen, reused_control=old_selection['selected']['visual_history']))
    print('[clean] selection frozen; evaluating development', flush=True)
    torch.set_num_threads(8)
    dev = encode_histories(tuning_roots(paths['banks']), imaginer.encode, ctx)
    torch.set_num_threads(1)
    executor = HistoryRealExecutor(model, scaler, probe, policy, device='cuda', ctx=ctx,
                                   max_envs=24, encode_batch=64)
    try:
        new = executor.run(best['theta'][None], dev.roots, z_hist=dev.z_hist, record_qpos=True)
        counters.add(executor.counters)
        imagined = segment_metrics(imaginer.rollout_policies(policy, best['theta'][None],
            dev.z_hist, dev.hist_blocks)['readout'][0, :, 0])
    finally:
        executor.close()
        ctx.close()
    np.savez_compressed(run.run_dir/'real_clean.npz', **new)
    with np.load(control/'real_visual_history.npz') as f:
        old = {k: f[k] for k in f.files}
    old_rows = json.loads((source/'development_rows.json').read_text())
    assert dev.root_ids == [r['root_id'] for r in old_rows]
    family = np.array([r['family'] for r in old_rows])
    real = dict(recorded_targets_reused=old, clean_teacher_targets=new)
    summary = {k: brief(v) for k, v in real.items()}
    summary['clean_teacher_targets'].update(imagined_violations=int(imagined['violated'].sum()),
        real_readout_violations=int(segment_metrics(new['readout'])['violated'].sum()))
    summary['recorded_targets_reused']['imagined_violations'] = json.loads((control/'diagnostic.json').read_text())['development']['visual_history']['imagined_violations']
    by_family = {f: {k: brief({name: v[family == f] for name, v in out.items()})
                    for k, out in real.items()} for f in ['PPO', 'PPOLag']}
    paired = {name: clustered_mean_ci(new[field].astype(float)-old[field].astype(float), dev.source_episode)
              for name, field in [('violation_difference', 'dense_violated'), ('progress_difference', 'progress')]}
    costs = dict(real_steps=collection['real_steps']+counters.real_steps,
        teacher_collection=collection, evaluation_real_steps=counters.real_steps,
        optimizer_updates=updates, reused_control_training_updates=cfg.control.original_training_updates,
        imagined_rows=imaginer.counters.imagined_rows, renders=counters.renders+264+64,
        encodes=counters.encodes+264, reused_control_outcomes=True, wall_s=time.time()-started)
    run.write_json('diagnostic.json', dict(development=summary, by_family=by_family, paired=paired,
        costs=costs, gate0_run=False, n_params=policy.n_params))
    rows = [dict(root_id=r['root_id'], episode=r['episode'], family=r['family'], recorded=r['recorded'],
        **{arm: dict(violated=bool(out['dense_violated'][i]), progress=float(out['progress'][i]),
                     first_unsafe_step=int(out['dense_first_step'][i])) for arm, out in real.items()})
        for i, r in enumerate(old_rows)]
    run.write_json('development_rows.json', rows)
    files = ['configs/evo/stage2_clean_targets.yaml', 'experiments/helpers/evoCleanTargets.py',
        'experiments/scripts/evo_clean_targets_diagnostic.py', 'experiments/tests/test_evo_clean_targets.py',
        'experiments/helpers/evoInformation.py', 'experiments/helpers/evoHistoryPolicy.py',
        'experiments/helpers/evoHistoryReal.py', 'experiments/helpers/evoHistoryImagine.py']
    for name in files:
        p = run.run_dir/'source'/name
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, p)
    run.finish(build_manifest(run_id=run.run_id, kind='evo-clean-targets-diagnostic', costs=costs,
        started_at=started, seeds=dict(fitting=list(cfg.training.seeds)), upstream_revisions=input_revisions(paths),
        data=dict(source=source_ref, control=control_ref, original_cache=dict(cfg.cache),
                  history_cache=dict(cfg.history_cache), teacher_sha256=teacher_hashes,
                  validation_digest=rootset_digest(val), development_digest=rootset_digest(dev)),
        extra=dict(code_sha256={k: file_sha256(ROOT/k) for k in files}, gate0_run=False)),
        upload=False, readme=f'# {run.run_id}\n\nBroad clean source-teacher targets.\n')
    print(json.dumps(dict(development=summary, by_family=by_family, costs=costs)), flush=True)


if __name__ == '__main__':
    main()
