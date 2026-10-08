#!/usr/bin/env python3
"""Check two fixed final teachers across all existing development roots."""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
from helpers.threads import pin_threads
pin_threads()
import numpy as np
import torch
from helpers.evoInputs import EvoRun,fetch_inputs,input_revisions,load_stage_config
from helpers.evoReal import _physics_env
from helpers.evoRoots import tuning_roots,rootset_digest
from helpers.evoTeacherDiagnostic import teacher_branch
from helpers.runManifest import build_manifest,file_sha256
from helpers.walkerAssets import load_exported_actor


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id',required=True)
    args=ap.parse_args()
    start=time.time()
    cfg=load_stage_config('stage2_single_teacher_probe')
    paths=fetch_inputs(cfg.inputs,names=['policies','banks'])
    roots=tuning_roots(paths['banks'])
    assert len(roots)==24
    torch.set_num_threads(1)
    run=EvoRun.create(cfg,'s2-single-teacher-probe',args.run_id)
    env=_physics_env()
    rows,summary,hashes=[],{},{}
    try:
        for name in cfg.teachers:
            path=paths['policies']/f'{name}.pt'
            hashes[name]=file_sha256(path)
            teacher=load_exported_actor(path,env.action_space,action_noise=0.)
            logs=[teacher_branch(env,r,r.policy_tape,teacher,takeover_step=0) for r in roots.roots]
            np.savez_compressed(run.run_dir/f'{name}.npz',
                **{k:np.stack([r[k] for r in logs]) for k in ['qpos','qvel','actions','clearance']})
            summary[name]=dict(violations=sum(r['violated'] for r in logs),
                              progress=float(np.mean([r['progress'] for r in logs])))
            rows.extend([dict(teacher=name,root_id=r.root_id,episode=r.episode,
                         **{k:log[k] for k in ['violated','first_unsafe_step','progress']})
                         for r,log in zip(roots.roots,logs)])
    finally:
        env.close()
    costs=dict(real_steps=4800,teacher_queries=4800,optimizer_updates=0,
               imagined_rows=0,renders=0,encodes=0,wall_s=time.time()-start)
    run.write_json('diagnostic.json',dict(summary=summary,rows=rows,costs=costs,gate0_run=False))
    files=['configs/evo/stage2_single_teacher_probe.yaml','experiments/scripts/evo_single_teacher_probe.py']
    for name in files:
        target=run.run_dir/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    run.finish(build_manifest(run_id=run.run_id,kind='evo-single-teacher-probe',
        costs=costs,upstream_revisions=input_revisions(paths),
        data=dict(rootset_digest=rootset_digest(roots),teacher_sha256=hashes),
        extra=dict(code_sha256={n:file_sha256(ROOT/n) for n in files},gate0_run=False)),
        upload=False,readme=f'# {run.run_id}\n\nFixed teacher diagnostic on development starts.\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
