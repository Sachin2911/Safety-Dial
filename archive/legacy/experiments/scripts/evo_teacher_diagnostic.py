#!/usr/bin/env python3
"""Audit teacher disagreement and fixed-boundary takeovers on 24 development roots."""
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

from helpers.evoInputs import EvoRun, fetch_inputs, input_revisions, load_stage_config
from helpers.evoReal import _physics_env
from helpers.evoRoots import rootset_digest, tuning_roots
from helpers.evoTeacherDiagnostic import action_error_summary, teacher_branch
from helpers.runManifest import build_manifest, file_sha256
from helpers.walkerAssets import load_exported_actor


def brief(out):
    return {k: out[k] for k in ('violated', 'first_unsafe_step', 'progress')}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id', required=True)
    args = ap.parse_args()
    started = time.time()
    torch.set_num_threads(1)
    cfg = load_stage_config('stage2_teacher')
    source = ROOT / 'runs' / cfg.source.run_id
    receipt = json.loads((source / 'hf_upload.json').read_text())
    assert receipt['revision'] == cfg.source.revision
    assert receipt['repo_id'] == 'Sachioster/safetydial-walker2d'
    original_results = ROOT / 'docs/evoPlan/results/stage2-coverage' / cfg.source.run_id
    requested = json.loads((original_results / 'archive_request.json').read_text())
    hashes = {item['path']: item['sha256'] for item in requested['files']}
    consumed = ['real_mixed.npz', 'development_rows.json', 'manifest.json']
    assert all(file_sha256(source / name) == hashes[name] for name in consumed)
    paths = fetch_inputs(cfg.inputs, names=['banks', 'data', 'policies'])
    roots = tuning_roots(paths['banks'])
    assert len(roots) == cfg.real.expected_roots
    with np.load(source / 'real_mixed.npz') as data:
        saved = {k: data[k] for k in data.files}
    previous = json.loads((source / 'development_rows.json').read_text())
    assert roots.root_ids == [r['root_id'] for r in previous]
    assert np.array_equal(saved['pairs'], np.column_stack([np.zeros(24, dtype=int), np.arange(24)]))
    with h5py.File(paths['data'] / 'roots.h5', 'r') as f:
        ids = {int(v): k for k, v in json.loads(f.attrs['policy_ids']).items()}
        off = f['ep_offset'][:]
        teachers = []
        for root in roots.roots:
            start = int(off[root.episode]) + root.step
            assert np.array_equal(f['qpos'][start], root.qpos)
            assert np.array_equal(f['qvel'][start], root.qvel)
            assert np.array_equal(f['action'][start:start+100], root.policy_tape)
            teachers.append(ids[int(f['policy_id'][start])])
    run = EvoRun.create(cfg, 's2-teacher', args.run_id)
    env = _physics_env()
    rows, mixed_logs, recorded_logs, held_logs, takeover_logs = [], [], [], [], []
    actors, actor_hashes = {}, {}
    try:
        for i, (root, name) in enumerate(zip(roots.roots, teachers)):
            if name not in actors:
                path = paths['policies'] / f'{name}.pt'
                actors[name] = load_exported_actor(path, env.action_space, action_noise=0., device='cpu')
                actor_hashes[name] = file_sha256(path)
            teacher = actors[name]
            tape = saved['actions'][i].reshape(100, 6)
            replay = teacher_branch(env, root, tape, teacher)
            for k in ('qpos', 'qvel'):
                if not np.array_equal(replay[k], saved[k][i]):
                    raise AssertionError(f'{root.root_id}: saved mixed {k} replay mismatch')
            assert replay['violated'] == saved['dense_violated'][i]
            assert replay['first_unsafe_step'] == saved['dense_first_step'][i]
            mixed_logs.append(replay)
            recorded = teacher_branch(env, root, root.policy_tape, teacher)
            assert recorded['violated'] == previous[i]['recorded']['violated']
            assert recorded['progress'] == previous[i]['recorded']['progress']
            recorded_logs.append(recorded)
            held = teacher_branch(env, root, tape, teacher, takeover_step=0,
                                  feedback_interval=int(cfg.real.held_teacher_interval))
            held_logs.append(held)
            takeovers = []
            for boundary in cfg.real.takeover_steps:
                out = teacher_branch(env, root, tape, teacher, takeover_step=int(boundary))
                for k in ('qpos', 'qvel'):
                    assert np.array_equal(out[k][:boundary+1], saved[k][i, :boundary+1])
                takeovers.append(out)
            takeover_logs.append(takeovers)
            rows.append(dict(root_id=root.root_id, episode=root.episode, teacher=name,
                             family=name.split('-')[0], mixed=brief(replay), recorded=brief(recorded),
                             held_teacher=brief(held), takeovers=[
                                 {'takeover_step': int(b), **brief(out)}
                                 for b, out in zip(cfg.real.takeover_steps, takeovers)]))
            print(f"[teacher] {i+1}/24 {root.root_id} mixed={replay['violated']} "
                  f"teacher={takeovers[0]['violated']}", flush=True)
    finally:
        env.close()
    arrays = {}
    keys = ('qpos', 'qvel', 'actions', 'teacher_actions', 'clearance')
    for label, logs in [('mixed', mixed_logs), ('recorded', recorded_logs), ('held', held_logs)]:
        arrays.update({f'{label}_{k}': np.stack([out[k] for out in logs]) for k in keys})
    arrays.update({f'takeover_{k}': np.array([[out[k] for out in row] for row in takeover_logs])
                   for k in keys})
    np.savez_compressed(run.run_dir / 'trajectories.npz', **arrays)
    errors = {}
    for label, logs in [('mixed', mixed_logs), ('recorded', recorded_logs)]:
        errors[label] = {}
        for family in ['all', 'PPO', 'PPOLag']:
            idx = [i for i, r in enumerate(rows) if family == 'all' or r['family'] == family]
            errors[label][family] = action_error_summary(
                arrays[f'{label}_actions'][idx], arrays[f'{label}_teacher_actions'][idx],
                [logs[i]['first_unsafe_step'] for i in idx])
    summary = {}
    for label, logs in [('mixed', mixed_logs), ('recorded', recorded_logs),
                        ('teacher', [row[0] for row in takeover_logs]), ('held_teacher', held_logs)]:
        summary[label] = {family: {
            'n': len(idx := [i for i, r in enumerate(rows) if family == 'all' or r['family'] == family]),
            'violations': sum(logs[i]['violated'] for i in idx),
            'progress_mean': float(np.mean([logs[i]['progress'] for i in idx]))}
            for family in ['all', 'PPO', 'PPOLag']}
    costs = dict(real_steps=len(rows) * (3 + len(cfg.real.takeover_steps)) * 100,
                 teacher_queries=len(rows) * (3 + len(cfg.real.takeover_steps)) * 100,
                 imagined_rows=0, renders=0, encodes=0, optimizer_updates=0,
                 scientific_wall_s=time.time() - started)
    assert costs['real_steps'] == cfg.real.expected_simulator_steps
    run.write_json('rows.json', rows)
    run.write_json('diagnostic.json', dict(summary=summary, action_error=errors, costs=costs,
                                          gate0_run=False, bitwise_replay_verified=True))
    code = ['configs/evo/stage2_teacher.yaml', 'experiments/helpers/evoTeacherDiagnostic.py',
            'experiments/scripts/evo_teacher_diagnostic.py', 'experiments/tests/test_evo_teacher.py']
    for name in code:
        dest = run.run_dir / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    manifest = build_manifest(run_id=run.run_id, kind='evo-teacher-diagnostic', costs=costs,
        upstream_revisions=input_revisions(paths), started_at=started,
        data=dict(source=receipt, source_files={name: hashes[name] for name in consumed},
                  rootset_digest=rootset_digest(roots), teacher_sha256=actor_hashes),
        extra=dict(gate0_run=False, code_sha256={name: file_sha256(ROOT / name) for name in code},
                   original_source_costs=json.loads((source / 'manifest.json').read_text()).get('costs'),
                   interpretation='privileged teacher diagnostic, no new controller training'))
    run.finish(manifest, upload=False, readme=f'# {run.run_id}\n\nTeacher diagnosis on development roots. See REPORT.md.\n')
    print(json.dumps({'run_id': run.run_id, 'summary': summary, 'costs': costs}), flush=True)


if __name__ == '__main__':
    main()
