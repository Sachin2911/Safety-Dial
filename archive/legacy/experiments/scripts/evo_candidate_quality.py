#!/usr/bin/env python3
"""Bounded development diagnosis of candidate generation versus imagined ranking."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault('MUJOCO_GL','egl')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
import numpy as np
import yaml
from omegaconf import OmegaConf

from helpers.evoCandidateQuality import ActionLimitedPolicy, build_pool, diagnostic_budget
from helpers.evoCandidateQuality import execute_diagnostic
from helpers.evoFollowup import split_banks
from helpers.evoHistoryImagine import HistoryImaginer
from helpers.evoHistoryReal import HistoryRealExecutor
from helpers.evoInputs import EvoRun, input_revisions
from helpers.evoReadinessBanks import load_encoded_banks
from helpers.evoReadinessQueries import atomic_json
from helpers.evoReadinessRuntime import load_frozen_runtime
from helpers.locoData import RenderContext
from helpers.runManifest import build_manifest, file_sha256
from helpers.walkerValidation import verify_render_fingerprint
import helpers.evoHistoryReal as real_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,
                        default=ROOT/'configs/evo/candidate_quality_20261005.yaml')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--run-id')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stop-after',choices=['selection'])
    args = parser.parse_args()
    config = yaml.safe_load(args.protocol.read_text())
    c = config['candidate_quality']
    rows,theta,parent_files = build_pool(config,ROOT)
    plan = diagnostic_budget(config,len(rows))
    print(json.dumps(plan),flush=True)
    if not args.execute:
        return
    if not args.run_id or not c['execution_enabled']:
        parser.error('execution requires an enabled protocol and unique run ID')
    files = [Path(__file__),ROOT/'experiments/scripts/evo_candidate_quality_report.py',
        args.protocol,ROOT/'pyproject.toml',ROOT/'uv.lock',
        ROOT/'docs/evoPlan/protocols/candidate-quality-20261005.md',
        ROOT/'scripts/managed/evo_candidate_quality.sh']
    files += list((ROOT/'experiments/helpers').glob('*.py'))
    files += list((ROOT/'experiments/tests').glob('test_evo_candidate_quality*.py'))
    files += [ROOT/'configs/evo'/name for name in
              ['inputs.yaml','power_pilot_sources.json','stage2_readiness_baseline.yaml']]
    names = sorted({str(p.resolve().relative_to(ROOT)) for p in files})
    subprocess.run(['git','ls-files','--error-unmatch','--',*names],cwd=ROOT,
                   check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','diff','--quiet','HEAD','--',*names],cwd=ROOT,check=True)
    code = {name:file_sha256(ROOT/name) for name in names}
    run = EvoRun.create(OmegaConf.create(config),'6-candidate-quality',args.run_id,resume=args.resume)
    identity = dict(source_sha256=code,protocol_sha256=file_sha256(args.protocol),
                    parent_files_sha256=parent_files)
    launch_path = run.run_dir/'launch.json'
    if launch_path.exists():
        launch = json.loads(launch_path.read_text())
        if launch['identity'] != identity:
            raise ValueError('source/protocol/parent changed since launch')
    else:
        started = time.time()
        launch = dict(identity=identity,started_at=started,deadline=started+c['max_hours']*3600)
        run.write_json('launch.json',launch)
        for name in code:
            target = run.run_dir/'source'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,target)
    def deadline():
        if time.time()>launch['deadline']:
            raise TimeoutError('absolute wall cap reached; no automatic extension')
    deadline()
    encoded,physical,parents = load_encoded_banks(ROOT/c['reused_bank_path'])
    banks,roots = split_banks(parents,physical,c['banks'])
    del banks['fitness'], roots['fitness']
    runtime = load_frozen_runtime(config,ROOT,require_archived=True)
    provenance = dict(**runtime['provenance'],**identity,
        parent_bank_sha256=encoded['encoded_banks_sha256'],exploratory=True,
        no_new_source_episodes=True,phase='candidate_quality')
    counter = dict(real_steps=0,predictor_rows=0,renders=0,encodes=0)
    model = runtime['model']
    original_predict,original_step = model.predict,real_module._step
    ctx = ex = None
    def predict(z,a,*args,**kwargs):
        deadline()
        counter['predictor_rows'] += len(z)
        return original_predict(z,a,*args,**kwargs)
    def step(*args,**kwargs):
        deadline()
        counter['real_steps'] += 1
        return original_step(*args,**kwargs)
    model.predict,real_module._step = predict,step
    def counts():
        out = counter.copy()
        if ex is not None:
            out['renders'] += ex.counters.renders
            out['encodes'] += ex.counters.encodes
        return out
    startup_path = run.run_dir/'startup_receipts.json'
    startups = json.loads(startup_path.read_text()) if startup_path.exists() else []
    try:
        ctx = RenderContext()
        counter['renders'] += 64
        verify_render_fingerprint(ctx,runtime['paths']['data']/'setA.h5')
        startups.append(dict(time=time.time(),fingerprint_renders=64))
        atomic_json(startup_path,startups)
        im = HistoryImaginer(model,runtime['scaler'],runtime['probe'],device='cuda',sigma=runtime['sigma'])
        ex = HistoryRealExecutor(model,runtime['scaler'],runtime['probe'],runtime['policy'],
                                device='cuda',ctx=ctx,max_envs=64,encode_batch=64)
        def real(role,row,th):
            deadline()
            ex.policy = ActionLimitedPolicy(runtime['policy'],row['cap'])
            out = ex.run(th[None],roots[role],z_hist=banks[role].z_hist,record_qpos=True)
            if not np.allclose(out['qpos'][:,0],np.stack([r.qpos for r in roots[role]]),rtol=0,atol=1e-12):
                raise ValueError('root identity/order changed')
            out.update(ex.policy.statistics())
            if row['cap'] is not None and float(out['action_delta_max'])>row['cap']+2e-7:
                raise ValueError('executed action correction exceeded cap')
            return out
        def imagined(role,row,th,k,n):
            deadline()
            policy = ActionLimitedPolicy(runtime['policy'],row['cap'])
            out = im.rollout_policies(policy,th[None],banks[role].z_hist,
                banks[role].hist_blocks,n_samples=n,k=k,seed=c['audit_seed'],
                max_batch=config['implementation']['max_imagined_batch'])
            return dict(readout=out['readout'][0],root_readout=out['root_readout'])
        result = execute_diagnostic(run.run_dir/'study',config,rows,theta,banks,
            provenance=provenance,real_callback=real,imagined_callback=imagined,
            read_costs=counts,stop_after=args.stop_after)
        if not result['complete']:
            print(json.dumps(result),flush=True)
            return
        costs = dict(result['costs'])
        costs['renders'] += sum(s['fingerprint_renders'] for s in startups)
        costs.update(gradient_updates=0,new_source_collection_steps=0,
                     wall_s=time.time()-launch['started_at'])
        run.write_json('costs.json',costs)
        run.write_json('analysis.json',result['analysis'])
        run.finish(build_manifest(run_id=run.run_id,kind='evo-candidate-quality-v1',costs=costs,
            started_at=launch['started_at'],upstream_revisions=input_revisions(runtime['paths']),
            data=provenance,seeds=dict(sampling=c['sampling_seed'],audit=c['audit_seed'],
                                     bootstrap=c['bootstrap_seed']),extra=identity),upload=False)
        print(json.dumps(dict(complete=True,costs=costs,
            eligible_on_both=result['analysis']['eligible_on_both'])),flush=True)
    except BaseException as exc:
        run.write_json('failure.json',dict(error=type(exc).__name__,message=str(exc),
            current_process_costs=counts(),partial_queries_are_not_automatically_replayed=True))
        raise
    finally:
        model.predict,real_module._step = original_predict,original_step
        if ex is not None:
            ex.close()
        if ctx is not None:
            ctx.close()


if __name__=='__main__':
    main()
