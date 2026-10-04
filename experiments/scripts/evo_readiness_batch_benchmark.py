#!/usr/bin/env python3
"""Prospective main-shape timing on repeated existing development features, no physics."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault('MUJOCO_GL','egl')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from helpers.evoHistoryImagine import HistoryImaginer
from helpers.evoInputs import EvoRun
from helpers.evoReadinessProtocol import protocol_plan
from helpers.evoReadinessQueries import QueryStore,digest_array,digest_json
from helpers.evoReadinessRuntime import initial_residual_diagnostics,load_frozen_runtime
from helpers.runManifest import build_manifest,file_sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    config_path=ROOT/'configs/evo/stage5_batch_benchmark.yaml'
    cfg=yaml.safe_load(config_path.read_text())
    candidate_path=ROOT/cfg['main_protocol']
    if file_sha256(candidate_path)!=cfg['main_protocol_sha256']:
        raise ValueError('main planning candidate changed since benchmark declaration')
    candidate=yaml.safe_load(candidate_path.read_text())
    plan=protocol_plan(candidate)
    repeats=cfg['warmup_repeats']+cfg['measured_repeats']
    expected=sum(s['roots']*s['samples']*cfg['horizon_blocks'] for s in cfg['shapes'])*repeats
    if expected!=cfg['expected_predictor_rows']:
        raise ValueError('benchmark declaration accounting mismatch')
    files=['configs/evo/stage5_batch_benchmark.yaml',cfg['main_protocol'],'configs/evo/inputs.yaml',
        'configs/evo/power_pilot_sources.json','configs/evo/stage2_readiness_baseline.yaml',
        'experiments/scripts/evo_readiness_batch_benchmark.py','pyproject.toml','uv.lock']
    files+=sorted(str(p.relative_to(ROOT)) for p in (ROOT/'experiments/helpers').glob('*.py'))
    subprocess.run(['git','ls-files','--error-unmatch','--',*files],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','diff','--quiet','HEAD','--',*files],cwd=ROOT,check=True)
    hashes={name:file_sha256(ROOT/name) for name in files}
    started=time.time()
    run=EvoRun.create(OmegaConf.create(cfg),'s5-batch-benchmark',args.run_id)
    for name in files:
        dest=run.run_dir/'source'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,dest)
    run.write_json('declaration.json',dict(config=cfg,source_sha256=hashes,started_at=started,
        repeated_existing_features_not_independent_episodes=True))
    charged=dict(predictor_rows=0,real_steps=0,diagnostic_policy_calls=0)
    model=original=None
    try:
        runtime=load_frozen_runtime(candidate,ROOT,require_archived=False)
        model=runtime['model']
        original=model.predict
        def predict(z,a,*args,**kwargs):
            if charged['predictor_rows']+len(z)>expected:
                raise RuntimeError('declared predictor budget exceeded before next call')
            charged['predictor_rows']+=len(z)
            return original(z,a,*args,**kwargs)
        model.predict=predict
        im=HistoryImaginer(model,runtime['scaler'],runtime['probe'],device='cuda',sigma=runtime['sigma'])
        baseline=ROOT/'runs'/candidate['implementation']['baseline_history_run']
        with np.load(baseline/'history_latents.npz') as f:
            zh,hist=f['z_hist'],f['hist_blocks']
        with np.load(ROOT/'runs'/candidate['inputs']['noise_run']/candidate['inputs']['noise_file']) as f:
            fit_ids=f['fit_indices']
        if len(fit_ids)!=cfg['calibration_roots']:
            raise ValueError('existing calibration role count changed')
        identity=digest_json(dict(config=cfg,sources=hashes,assets=runtime['provenance']))
        store=QueryStore(run.run_dir/'queries',study_identity=identity)
        summaries=[]
        for shape in cfg['shapes']:
            role,k,r,n=shape['role'],shape['noise_k'],shape['roots'],shape['samples']
            declared_r={'fitness':candidate['search']['fitness_roots_per_generation'],
                'selection':candidate['bank']['selection_episodes'],'evaluation':candidate['bank']['evaluation_episodes']}[role]
            sample_cfg=candidate['evaluation']['audit_samples'] if role=='evaluation' else candidate['search'][role+'_samples']
            if r!=declared_r or n!=sample_cfg['zero_noise' if k==0 else 'positive_noise']:
                raise ValueError('benchmark shape differs from candidate main protocol')
            ids=np.resize(fit_ids,r)
            measurements=[]
            for rep in range(repeats):
                request=dict(**shape,root_indices_sha256=digest_array(ids),repetition=rep,
                    seed=cfg['seed']+rep,phase='warmup' if rep<cfg['warmup_repeats'] else 'measured',
                    theta_sha256=digest_array(runtime['zero']))
                def query():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    begin=time.perf_counter()
                    out=im.rollout_policies(runtime['policy'],runtime['zero'][None],zh[ids],hist[ids],
                        n_samples=n,k=k,seed=request['seed'],n_blocks=cfg['horizon_blocks'],max_batch=cfg['max_batch'])
                    torch.cuda.synchronize()
                    elapsed=time.perf_counter()-begin
                    return dict(readout=out['readout'][0],wall_s=np.asarray(elapsed),
                        peak_allocated_bytes=np.asarray(torch.cuda.max_memory_allocated()),
                        peak_reserved_bytes=np.asarray(torch.cuda.max_memory_reserved()))
                out,_=store.execute(f'{role}-k{k:g}-repeat{rep}',request=request,
                    expected_costs=dict(predictor_rows=r*n*cfg['horizon_blocks'],real_steps=0),
                    callback=query,read_costs=lambda:charged)
                measurements.append(dict(repetition=rep,phase=request['phase'],wall_s=float(out['wall_s']),
                    peak_allocated_bytes=int(out['peak_allocated_bytes']),peak_reserved_bytes=int(out['peak_reserved_bytes'])))
            measured=[m['wall_s'] for m in measurements if m['phase']=='measured']
            summary=dict(**shape,predictor_rows_per_query=r*n*cfg['horizon_blocks'],
                unique_existing_roots=len(set(ids.tolist())),batch_rows=r*n,measurements=measurements,
                median_wall_s=float(np.median(measured)),min_wall_s=min(measured),max_wall_s=max(measured),
                maximum_allocated_bytes=max(m['peak_allocated_bytes'] for m in measurements),
                maximum_reserved_bytes=max(m['peak_reserved_bytes'] for m in measurements))
            summaries.append(summary)
            print(json.dumps({key:summary[key] for key in ['role','noise_k','batch_rows','median_wall_s','maximum_allocated_bytes']}),flush=True)
        theta=np.zeros((cfg['initial_action_diagnostic_random_candidates']+1,1020))
        theta[1:]=np.random.default_rng(cfg['seed']).normal(0,candidate['policy']['initial_CMA_sigma'],size=theta[1:].shape)
        def policy_call():
            charged['diagnostic_policy_calls']+=1
        def diagnose():
            out=initial_residual_diagnostics(runtime['policy'],runtime['calibration_features'],theta,on_policy_call=policy_call)
            return dict(out,theta=theta)
        diagnostics,_=store.execute('initial-actions',request=dict(theta_sha256=digest_array(theta),
            features_sha256=runtime['provenance']['calibration_features_sha256']),
            expected_costs=dict(predictor_rows=0,real_steps=0,diagnostic_policy_calls=cfg['expected_diagnostic_policy_calls']),
            callback=diagnose,read_costs=lambda:charged)
        assert diagnostics['raw_residual_rms'][0]==0 and diagnostics['clipped_action_delta_rms'][0]==0
        projected=[]
        for row in summaries:
            specs=[s for s in plan['search_specs'] if s.noise_k==row['noise_k']]
            if row['role']=='fitness':
                calls=sum(s.config.population*s.config.generations for s in specs)
            elif row['role']=='selection':
                calls=sum(s.config.generations+1 for s in specs)
            else:
                calls=plan['counts']['selected_policy_slots']+len(candidate['search']['search_seeds'])
            projected.append(dict(role=row['role'],noise_k=row['noise_k'],candidate_queries=calls,
                projected_wall_s=calls*row['median_wall_s']))
        assert charged['predictor_rows']==expected and charged['real_steps']==0
        assert charged['diagnostic_policy_calls']==cfg['expected_diagnostic_policy_calls']
        costs=dict(charged,gradient_updates=0,wall_s=time.time()-started)
        report=dict(passed=True,costs=costs,measurements=summaries,projected_model_only=projected,
            projected_model_only_hours=sum(r['projected_wall_s'] for r in projected)/3600,
            projection_excludes='physics, collection, encoding, query archive compression, optimiser, analysis, startup and interruptions',
            initial_actions=dict(zero_residual_preserved=True,random_candidates=len(theta)-1,
                median_raw_rms=float(np.median(diagnostics['raw_residual_rms'][1:])),
                median_clipped_delta_rms=float(np.median(diagnostics['clipped_action_delta_rms'][1:])),
                candidate_distribution='independent Gaussian action check, not selected or evolved policies'),
            main_study_executed=False,rosarl_compared=False,new_independent_episodes=0,
            asset_provenance=runtime['provenance'],source_sha256=hashes,accounting=store.accounting())
        run.write_json('benchmark.json',report)
        readme=f'# Main batch throughput validation\n\nThe declared frozen-model timing check passed with {expected:,} charged predictor rows and zero real simulator steps. It repeats existing calibration features solely to reach proposed batch shapes. No new independent episodes, search, ROSARL comparison or model training occurred.\n'
        (run.results_dir/'README.md').write_text(readme)
        run.finish(build_manifest(run_id=run.run_id,kind='evo-readiness-batch-benchmark-v1',costs=costs,
            started_at=started,data=runtime['provenance'],seeds=dict(benchmark=cfg['seed']),
            extra=dict(source_sha256=hashes,main_stage5=False)),upload=False,readme=readme)
        print(json.dumps(dict(passed=True,costs=costs,projected_model_only_hours=report['projected_model_only_hours'])),flush=True)
    except BaseException as exc:
        run.write_json('failure.json',dict(error=type(exc).__name__,message=str(exc),costs=charged))
        raise
    finally:
        if original is not None:
            model.predict=original


if __name__=='__main__':
    main()
