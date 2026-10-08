#!/usr/bin/env python3
"""Locate prediction drift before failure using saved Walker development trajectories."""
import argparse
import ast
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
import torch
import yaml
from omegaconf import OmegaConf

from helpers.evoCandidateQuality import ActionLimitedPolicy
from helpers.evoDivergence import MODES, RESETS, refreshed_predictions, verified_archive, validate_cell
from helpers.evoFollowup import load_freeze, split_banks
from helpers.evoHistoryImagine import HistoryImaginer
from helpers.evoInputs import EvoRun, input_revisions
from helpers.evoReadinessBanks import load_encoded_banks
from helpers.evoReadinessQueries import QueryStore, digest_json
from helpers.evoReadinessRuntime import load_frozen_runtime
from helpers.evoReadinessStudy import checked_real_output
from helpers.runManifest import build_manifest, file_sha256



def imported_helpers(entry_points, repository=ROOT):
    """Snapshot the actual helper import closure, excluding unrelated concurrent work."""
    repository = Path(repository)
    pending = list(entry_points)
    visited, helpers = set(), set()
    while pending:
        path = Path(pending.pop()).resolve()
        if path in visited:
            continue
        visited.add(path)
        tree = ast.parse(path.read_text())
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules += [a.name for a in node.names if a.name.startswith('helpers.')]
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith('helpers.'):
                    modules.append(node.module)
                elif node.module == 'helpers':
                    modules += ['helpers.'+a.name for a in node.names]
        for name in modules:
            target = repository/'experiments'/Path(*name.split('.')).with_suffix('.py')
            if not target.is_file():
                raise ValueError(f'unresolved local import: {name}')
            helpers.add(target)
            pending.append(target)
    return sorted(helpers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'configs/evo/divergence_20261005.yaml')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--run-id')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(args.protocol.read_text())
    c = config['divergence']
    parent = ROOT/c['parent_run']
    pc = yaml.safe_load((parent/'config.yaml').read_text())
    declaration = json.loads((parent/'study/study.json').read_text())
    parent_identity = declaration['identity']
    if digest_json(declaration['declaration'])!=parent_identity:
        raise ValueError('parent declaration checksum mismatch')
    if not json.loads((parent/'study/completion.json').read_text())['complete']:
        raise ValueError('parent diagnostic must be complete')
    pool,theta = load_freeze(parent/'study','candidate_pool',identity=parent_identity)
    rows = pool['entries']
    if c['roles']!=['real_selection','evaluation'] or len(rows)!=c['expected_candidates']:
        raise ValueError('the declared complete development pool and both roles are required')
    if (c['reset_periods']!=list(RESETS) or c['noise_k']!=0 or c['horizon_blocks']!=10
            or c['height_divergence_threshold_m']!=.05 or c['pitch_divergence_threshold_rad']!=.1):
        raise ValueError('unsupported diagnostic modes')
    encoded,physical,parents = load_encoded_banks(ROOT/pc['candidate_quality']['reused_bank_path'])
    banks,roots = split_banks(parents,physical,pc['candidate_quality']['banks'])
    planned = sum(len(banks[r]) for r in c['roles'])*len(rows)*10*len(MODES)
    plan = dict(candidates=len(rows),roles={r:len(banks[r]) for r in c['roles']},
        predictor_rows=planned,real_steps=0,renders=0,encodes=0,max_hours=c['max_hours'])
    print(json.dumps(plan),flush=True)
    if not args.execute:
        return
    if not args.run_id or c['execution_enabled'] is not True:
        parser.error('an enabled protocol and unique run ID are required')
    source = [Path(__file__),args.protocol,ROOT/'experiments/scripts/evo_divergence_report.py',
        ROOT/'pyproject.toml',ROOT/'uv.lock',ROOT/'docs/evoPlan/protocols/divergence-20261005.md',
        ROOT/'scripts/managed/evo_divergence.sh']
    source += imported_helpers([Path(__file__),ROOT/'experiments/scripts/evo_divergence_report.py'])
    source += list((ROOT/'experiments/tests').glob('test_evo_divergence*.py'))
    names = sorted({str(p.resolve().relative_to(ROOT)) for p in source})
    subprocess.run(['git','ls-files','--error-unmatch','--',*names],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','diff','--quiet','HEAD','--',*names],cwd=ROOT,check=True)
    code = {name:file_sha256(ROOT/name) for name in names}
    inputs = [parent/'config.yaml',parent/'study/study.json',parent/'study/completion.json',
              parent/'study/candidate_pool.json',parent/'study/candidate_pool.npz']
    for role in c['roles']:
        for row in rows:
            inputs.append(parent/'study'/role/(row['id']+'.npz'))
            if role=='evaluation':
                inputs.append(parent/'study'/role/(row['id']+'-k0.npz'))
    parent_hashes = {str(p.relative_to(ROOT)):file_sha256(p) for p in inputs}
    identity = dict(source_sha256=code,parent_files_sha256=parent_hashes,
                    protocol_sha256=file_sha256(args.protocol),parent_bank_sha256=encoded['encoded_banks_sha256'])
    run = EvoRun.create(OmegaConf.create(config),'6-divergence',args.run_id,resume=args.resume)
    launch_path = run.run_dir/'launch.json'
    if launch_path.exists():
        launch = json.loads(launch_path.read_text())
        if launch['identity']!=identity:
            raise ValueError('source, inputs or protocol changed since launch')
    else:
        started = time.time()
        launch = dict(identity=identity,started_at=started,deadline=started+3600*c['max_hours'])
        if c.get('inherited_deadline_from'):
            previous = ROOT/c['inherited_deadline_from']
            failed = json.loads((previous/'failure.json').read_text())
            if any(failed['current_process_costs'].values()) or list((previous/'queries').rglob('*.npz')):
                raise ValueError('deadline inheritance here is restricted to zero-query failed attempts')
            journals = list((previous/'queries').rglob('*.pending.json'))
            if not journals or any(any(json.loads(p.read_text())['measured_costs'].values()) for p in journals):
                raise ValueError('failed attempt must have explicit zero-cost journals')
            old_launch = json.loads((previous/'launch.json').read_text())
            launch.update(started_at=old_launch['started_at'],deadline=old_launch['deadline'],
                          replacement_process_started_at=started,
                          retained_zero_query_attempt=str(previous.relative_to(ROOT)),
                          retained_failure_sha256=file_sha256(previous/'failure.json'))
        run.write_json('launch.json',launch)
        for name in names:
            target = run.run_dir/'source'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,target)
    def deadline():
        if time.time()>launch['deadline']:
            raise TimeoutError('absolute diagnostic wall cap reached')
    deadline()
    runtime = load_frozen_runtime(pc,ROOT,require_archived=True)
    model = runtime['model']
    original_predict = model.predict
    counter = dict(predictor_rows=0,real_steps=0,renders=0,encodes=0)
    def predict(z,a,*args,**kwargs):
        deadline()
        counter['predictor_rows']+=len(z)
        return original_predict(z,a,*args,**kwargs)
    model.predict = predict
    study_identity = digest_json(identity)
    stores = {role:QueryStore(run.run_dir/'queries'/role,study_identity=study_identity) for role in c['roles']}
    try:
        im = HistoryImaginer(model,runtime['scaler'],runtime['probe'],device='cuda')
        for role in c['roles']:
            bank = banks[role]
            for i,row in enumerate(rows):
                path = parent/'study'/role/(row['id']+'.npz')
                real,receipt = verified_archive(path,parent_identity)
                if receipt['request']['bank']!=bank.identity() or receipt['request']['candidate']!=row:
                    raise ValueError('wrong candidate or episode bank')
                checked_real_output(real,len(bank),10)
                if not np.array_equal(real['qpos'][:,0],np.stack([r.qpos for r in roots[role]])):
                    raise ValueError('saved simulator root mismatch')
                request = dict(role=role,candidate=row,bank=bank.identity(),
                    parent_archive_sha256=parent_hashes[str(path.relative_to(ROOT))],modes=list(MODES))
                def compute():
                    policy = ActionLimitedPolicy(runtime['policy'],row['cap'])
                    with torch.inference_mode():
                        predicted = [refreshed_predictions(im,bank.z_hist,real['z'],bank.hist_blocks,
                                                          real['actions'],r) for r in RESETS]
                        if not all(torch.equal(z[:,0],predicted[0][:,0]) for z in predicted):
                            raise ValueError('refresh modes disagree before any refresh')
                        closed = im.rollout_policies(policy,theta[i:i+1],bank.z_hist,bank.hist_blocks,
                            n_samples=1,k=0,seed=0,n_blocks=10,
                            max_batch=pc['implementation']['max_imagined_batch'],return_latents=True)
                        predicted.append(torch.as_tensor(closed['z'][0,:,0],device='cuda'))
                        ro = [im.probe(z).double().cpu().numpy() for z in predicted[:-1]]
                        ro.append(closed['readout'][0,:,0])
                        # Saved real probe calls were one ten-frame segment at a time.
                        replayed_ro = np.stack([runtime['probe'].predict(torch.as_tensor(z,device='cuda'))
                                                for z in real['z']])
                        if not np.array_equal(replayed_ro,real['readout']):
                            raise ValueError('real-image readout failed exact reproduction')
                        real_z = torch.as_tensor(real['z'],device='cuda')
                        latent_rms = np.stack([(z.double()-real_z.double()).square().mean(-1).sqrt().cpu().numpy()
                                               for z in predicted])
                        if role=='evaluation':
                            old,_ = verified_archive(parent/'study'/role/(row['id']+'-k0.npz'),parent_identity)
                            if not np.array_equal(closed['readout'][0],old['readout']):
                                raise ValueError('closed-loop readout differs from frozen parent audit')
                        first_delta = float(np.max(np.abs(closed['actions'][0,:,0,0]-real['actions'][:,0])))
                        if first_delta>c['initial_action_tolerance']:
                            raise ValueError('first-block policy discrepancy exceeds declared tolerance')
                        idx = np.arange(len(bank))
                        first = np.minimum(real['dense_first_step']+1,100)
                        q = real['qpos'][idx,first]
                        margins = np.stack([(q[:,1]-.8)/.2,(2-q[:,1])/.2,
                                            (q[:,2]+1)/.5,(1-q[:,2])/.5],1)
                        out = dict(predicted_z=np.stack([z.cpu().numpy() for z in predicted]),
                            predicted_readout=np.stack(ro),latent_rms=latent_rms,
                            real_readout=real['readout'],truth=real['endpoint_targets'],
                            dense_first_step=real['dense_first_step'],dense_violated=real['dense_violated'],
                            root_height_pitch=real['qpos'][:,0,1:3],
                            action_drift_rms=np.sqrt(np.mean((closed['actions'][0,:,0]-real['actions'])**2,-1)),
                            initial_action_max_difference=np.asarray(first_delta),
                            failure_constraint=np.where(real['dense_violated'],margins.argmin(1),-1))
                    validate_cell(out)
                    return out
                stores[role].execute(row['id'],request=request,
                    expected_costs=dict(predictor_rows=len(bank)*50,real_steps=0,renders=0,encodes=0),
                    callback=compute,read_costs=lambda:counter.copy())
                print(f'[divergence] {role} {i+1}/{len(rows)}',flush=True)
        accounting = {role:store.accounting() for role,store in stores.items()}
        if any(v['pending_queries'] for v in accounting.values()):
            raise ValueError('unresolved diagnostic queries')
        costs = {key:sum(v['completed_costs'][key] for v in accounting.values()) for key in counter}
        if costs['predictor_rows']!=planned:
            raise ValueError('predictor budget mismatch')
        costs.update(wall_s=time.time()-launch['started_at'],gradient_updates=0)
        run.write_json('costs.json',costs)
        run.write_json('completion.json',dict(complete=True,study_identity=study_identity,
            parent_identity=parent_identity,plan=plan,accounting=accounting,
            modes=list(MODES),pool_sha256=pool['freeze_sha256']))
        run.finish(build_manifest(run_id=run.run_id,kind='evo-divergence-v1',costs=costs,
            started_at=launch['started_at'],upstream_revisions=input_revisions(runtime['paths']),
            data=runtime['provenance'],seeds=dict(bootstrap=c['bootstrap_seed']),extra=identity),upload=False)
        print(json.dumps(dict(complete=True,costs=costs)),flush=True)
    except BaseException as exc:
        run.write_json('failure.json',dict(error=type(exc).__name__,message=str(exc),current_process_costs=counter,
            implicit_replay_forbidden=True))
        raise
    finally:
        model.predict = original_predict


if __name__=='__main__':
    main()
