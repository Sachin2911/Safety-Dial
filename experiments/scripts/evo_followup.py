#!/usr/bin/env python3
"""Execute the bounded Stage 6 development study; no fresh confirmation is implied."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault('MUJOCO_GL', 'egl')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments'))
import numpy as np
import yaml
from omegaconf import OmegaConf

from helpers.evoFollowup import split_banks, study_plan
from helpers.evoFollowupStudy import DualAuditScorer, execute_study
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


def source_inventory(protocol):
    files = [str(Path(__file__).resolve().relative_to(ROOT)),
             str(protocol.resolve().relative_to(ROOT)), 'pyproject.toml', 'uv.lock',
             'configs/evo/inputs.yaml', 'configs/evo/power_pilot_sources.json',
             'configs/evo/stage2_readiness_baseline.yaml',
             'scripts/managed/evo_followup_development.sh',
             'docs/evoPlan/protocols/stage6-followup-20261005.md']
    files += [str(p.relative_to(ROOT)) for p in (ROOT/'experiments/helpers').glob('*.py')]
    files += [str(p.relative_to(ROOT)) for p in (ROOT/'experiments/tests').glob('test_evo_followup*.py')]
    files = sorted(set(files))
    subprocess.run(['git','ls-files','--error-unmatch','--',*files],cwd=ROOT,
                   check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','diff','--quiet','HEAD','--',*files],cwd=ROOT,check=True)
    return {name:file_sha256(ROOT/name) for name in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'configs/evo/stage6_development_20261005.yaml')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--run-id')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stop-after',choices=['search','selection'])
    args = parser.parse_args()
    config = yaml.safe_load(args.protocol.read_text())
    c = config['followup']
    plan = study_plan(config)
    # A separate fresh-bank runner and locked sizing review are the next phase.
    if c['phase']=='confirmation' or c['bank_source']!='reused_stage5':
        plan['blockers'].append('this executable supports engineering/development only')
    if not args.execute or plan['blockers']:
        print(json.dumps(dict(execution_ready=not plan['blockers'],**plan),indent=2))
        if args.execute:
            raise SystemExit('Execution refused before loading assets.')
        return
    if not args.run_id:
        parser.error('--run-id required for execution')
    code = source_inventory(args.protocol)
    run = EvoRun.create(OmegaConf.create(config),'s6-'+c['phase'],args.run_id,resume=args.resume)
    identity = dict(protocol_sha256=file_sha256(args.protocol),source_sha256=code)
    launch_path = run.run_dir/'launch.json'
    if launch_path.exists():
        launch = json.loads(launch_path.read_text())
        if launch['identity'] != identity:
            raise ValueError('executable source or protocol changed since launch')
    else:
        started = time.time()
        launch = dict(identity=identity,started_at=started,deadline=started+3600*c['max_hours'])
        run.write_json('launch.json',launch)
        for name in code:
            target = run.run_dir/'source'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,target)
    def deadline():
        if time.time()>launch['deadline']:
            raise TimeoutError('declared wall cap reached; no implicit extension')
    deadline()
    encoded,physical,parent_banks = load_encoded_banks(ROOT/c['reused_bank_path'])
    banks,roots = split_banks(parent_banks,physical,c['banks'])
    runtime = load_frozen_runtime(config,ROOT,require_archived=True)
    provenance = dict(**runtime['provenance'],source_sha256=code,
        parent_encoded_banks_sha256=encoded['encoded_banks_sha256'],
        parent_bank_path=c['reused_bank_path'],purpose=c['phase'],
        reused_historical_outcomes=True)
    counter = dict(real_steps=0,predictor_rows=0,renders=0,encodes=0)
    model = runtime['model']
    original_predict,original_step = model.predict,real_module._step
    ctx = ex = None
    def predict(z,a,*a2,**kw):
        deadline()
        counter['predictor_rows'] += len(z)
        return original_predict(z,a,*a2,**kw)
    def step(*a,**kw):
        deadline()
        counter['real_steps'] += 1
        return original_step(*a,**kw)
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
        def factory(cfg,arm,folder,identity):
            return DualAuditScorer(im,runtime['policy'],banks,cfg,arm,folder,identity,counts,
                c['search']['noisy_selection_samples'],config['implementation']['max_imagined_batch'])
        def real(role,theta):
            deadline()
            result = ex.run(theta[None],roots[role],z_hist=banks[role].z_hist,record_qpos=True)
            expected = np.stack([r.qpos for r in roots[role]])
            if not np.allclose(result['qpos'][:,0],expected,rtol=0,atol=1e-12):
                raise ValueError('simulator root identity/order changed')
            return result
        def audit(theta,k,n,seed):
            deadline()
            out = im.rollout_policies(runtime['policy'],theta[None],banks['evaluation'].z_hist,
                banks['evaluation'].hist_blocks,n_samples=n,k=k,seed=seed,
                max_batch=config['implementation']['max_imagined_batch'])
            return dict(readout=out['readout'][0],root_readout=out['root_readout'])
        result = execute_study(run.run_dir/'study',config,banks,runtime['zero'],
            provenance=provenance,scorer_factory=factory,real_callback=real,audit_callback=audit,
            read_costs=counts,stop_after=args.stop_after)
        if not result['complete']:
            print(json.dumps(result),flush=True)
            return
        costs = dict(result['actual_costs'])
        costs['renders'] = costs.get('renders',0)+sum(x['fingerprint_renders'] for x in startups)
        costs.update(gradient_updates=0,new_source_collection_steps=0,
                     wall_s=time.time()-launch['started_at'])
        run.write_json('costs.json',costs)
        run.write_json('analysis.json',result['analysis'])
        report = (f'# Stage 6 {c["phase"]}: {run.run_id}\n\n'
            'Matched model-query search and independent simulator shortlist selection completed. '
            'Source episodes are reused from Stage 5: all estimates are exploratory. '
            'The starting controller was always available as a fallback. '
            'All shortlists and final selections were frozen before their respective next phase.\n\n'
            'This validates the pipeline and informs subsequent fresh-bank study sizing; '
            'it does not establish generalization or safe full-episode control. '
            'See analysis.json and costs.json for outcomes and complete incremental costs.\n')
        (run.results_dir/'README.md').write_text(report)
        run.finish(build_manifest(run_id=run.run_id,kind='evo-followup-development-v1',costs=costs,
            started_at=launch['started_at'],upstream_revisions=input_revisions(runtime['paths']),
            data=provenance,seeds=dict(search=c['seeds'],bootstrap=c['bootstrap_seed']),
            extra=identity),upload=False,readme=report)
        print(json.dumps(dict(complete=True,phase=c['phase'],costs=costs,
                             primary=result['analysis']['primary'])),flush=True)
    except BaseException as exc:
        run.write_json('failure.json',dict(error=type(exc).__name__,message=str(exc),
            current_process_costs=counts(),note='Pending query journals forbid implicit replay.'))
        raise
    finally:
        model.predict,real_module._step = original_predict,original_step
        if ex is not None:
            ex.close()
        if ctx is not None:
            ctx.close()


if __name__=='__main__':
    main()
