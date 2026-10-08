#!/usr/bin/env python3
"""Bounded failure-rich Walker predictor/probe repair using archived development data."""
import argparse
import copy
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

from evo_divergence import imported_helpers
from helpers.evoCandidateQuality import ActionLimitedPolicy
from helpers.evoDivergence import refreshed_predictions, verified_archive
from helpers.evoFollowup import load_freeze, split_banks
from helpers.evoHistoryImagine import HistoryImaginer
from helpers.evoInputs import EvoRun
from helpers.evoReadinessBanks import load_encoded_banks
from helpers.evoReadinessQueries import QueryStore, digest_json
from helpers.evoReadinessRuntime import load_frozen_runtime
from helpers.evoRepair import (ARMS, split_episodes, training_arrays, tensor_state, fit_probe_copy,
    fit_predictor, summarize, select_repair, nominate, repair_gate)
from helpers.runManifest import file_sha256, build_manifest


def source_inventory(protocol):
    paths=[Path(__file__),ROOT/'experiments/scripts/evo_repair_report.py',
        ROOT/'experiments/scripts/evo_divergence.py',protocol,ROOT/'pyproject.toml',ROOT/'uv.lock',
        ROOT/'docs/evoPlan/protocols/repair-20261005.md',ROOT/'scripts/managed/evo_repair.sh']
    paths+=imported_helpers(paths[:3])
    paths+=list((ROOT/'experiments/tests').glob('test_evo_repair*.py'))
    names=sorted({str(p.resolve().relative_to(ROOT)) for p in paths})
    subprocess.run(['git','ls-files','--error-unmatch','--',*names],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','diff','--quiet','HEAD','--',*names],cwd=ROOT,check=True)
    return {name:file_sha256(ROOT/name) for name in names}


def load_data(c):
    parent=ROOT/c['parent_run']
    divergence=ROOT/c['divergence_run']
    pc=yaml.safe_load((parent/'config.yaml').read_text())
    declaration=json.loads((parent/'study/study.json').read_text())
    pid=declaration['identity']
    assert digest_json(declaration['declaration'])==pid
    dc=json.loads((divergence/'completion.json').read_text())
    assert dc['complete'] and dc['parent_identity']==pid
    pool,theta=load_freeze(parent/'study','candidate_pool',identity=pid)
    assert len(pool['entries'])==c['candidates']
    encoded,physical,parents=load_encoded_banks(ROOT/pc['candidate_quality']['reused_bank_path'])
    banks,_=split_banks(parents,physical,pc['candidate_quality']['banks'])
    roles=['real_selection','evaluation']
    keys=[key for r in roles for key in banks[r].episode_keys]
    history=np.concatenate([banks[r].z_hist for r in roles])
    actions=np.concatenate([banks[r].hist_blocks for r in roles])
    data={k:[] for k in ['z','actions','truth','first','progress','readout']}
    basez,basero=[],[]
    paths=[parent/'config.yaml',parent/'study/study.json',parent/'study/completion.json',
           parent/'study/candidate_pool.json',parent/'study/candidate_pool.npz',
           divergence/'completion.json',divergence/'launch.json']
    for row in pool['entries']:
        real,old=[],[]
        for role in roles:
            rp=parent/'study'/role/(row['id']+'.npz')
            dp=divergence/'queries'/role/(row['id']+'.npz')
            r,receipt=verified_archive(rp,pid)
            d,_=verified_archive(dp,dc['study_identity'])
            assert receipt['request']['bank']==banks[role].identity()
            assert receipt['request']['candidate']==row
            assert np.array_equal(r['endpoint_targets'],d['truth'])
            real.append(r)
            old.append(d)
            paths += [rp,dp]
        for key,source in dict(z='z',actions='actions',truth='endpoint_targets',first='dense_first_step',progress='progress',readout='readout').items():
            data[key].append(np.concatenate([r[source] for r in real]))
        basez.append(np.concatenate([d['predicted_z'][[0,3,4]] for d in old],axis=1))
        basero.append(np.concatenate([d['predicted_readout'][[0,3,4]] for d in old],axis=1))
    data={k:np.stack(v) for k,v in data.items()}
    return dict(config=pc,pool=pool,theta=theta,keys=list(keys),history=history,history_actions=actions,
        data=data,base_z=np.stack(basez,axis=1),base_readout=np.stack(basero,axis=1),
        input_hashes={str(p.relative_to(ROOT)):file_sha256(p) for p in paths},
        bank_sha256=encoded['encoded_banks_sha256'])


def read_probe(probe,z):
    shape=z.shape[:-1]
    return probe.predict(z.reshape(-1,z.shape[-1]),batch=4096).reshape(*shape,3).astype(np.float64)


def summarize_arms(pred,data,indices,c):
    return {arm:{mode:summarize(pred[arm][:,m],data['truth'][:,indices],data['first'][:,indices],
        seed=c['bootstrap_seed'],replicates=c['bootstrap_replicates']) for m,mode in enumerate(c['inference_modes'])} for arm in ARMS}


def controller_audit(decisions, data, indices, c):
    rng=np.random.default_rng(c['bootstrap_seed'])
    draws=rng.integers(len(indices),size=(c['bootstrap_replicates'],len(indices)))
    f=data['first'][:,indices]<100
    p=data['progress'][:,indices]
    def result(i,reference):
        delta=f[i].astype(float)-f[reference]
        means=delta[draws].mean(1)
        return dict(index=i,failures=int(f[i].sum()),episodes=len(indices),
            failure_rate=float(f[i].mean()),reference_index=reference,
            paired_failure_difference=dict(point=float(delta.mean()),lo=float(np.quantile(means,.025)),hi=float(np.quantile(means,.975))),
            mean_progress=float(p[i].mean()),baseline_progress=float(p[0].mean()),
            progress_ratio=None if p[0].mean()<=0 else float(p[i].mean()/p[0].mean()))
    out={name:result(d['index'],0) for name,d in decisions.items()}
    if 'repaired' in decisions:
        out['repaired_vs_original_selection']=result(decisions['repaired']['index'],decisions['original']['index'])
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol',type=Path,default=ROOT/'configs/evo/repair_20261005.yaml')
    ap.add_argument('--execute',action='store_true')
    ap.add_argument('--run-id')
    args=ap.parse_args()
    config=yaml.safe_load(args.protocol.read_text())
    c=config['repair']
    print(json.dumps(dict(config=c,real_steps=0,encoder_updates=0,scope='reused development repair')),flush=True)
    if not args.execute:
        return
    if c['execution_enabled'] is not True or not args.run_id:
        ap.error('enabled protocol and unique run ID required')
    assert c['inference_modes']==['teacher_forced','real_action_tape']
    assert c['physical_scale']==[.2,.5,5.] and c['fitting_failure_fraction']==.5
    assert c['weight_decay']==1e-4 and c['gradient_norm_clip']==1.
    sources=source_inventory(args.protocol)
    run=EvoRun.create(OmegaConf.create(config),'6-repair',args.run_id)
    started=time.time()
    deadline_at=started+c['max_hours']*3600
    counter=dict(predictor_training_rows=0,predictor_inference_rows=0,predictor_gradient_updates=0,
        probe_gradient_updates=0,probe_training_rows=0,real_steps=0,renders=0,encodes=0)
    def deadline():
        if time.time()>deadline_at:
            raise TimeoutError('absolute one-hour repair budget reached')
    run.write_json('launch.json',dict(started_at=started,deadline=deadline_at,source_sha256=sources))
    for name in sources:
        target=run.run_dir/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    try:
        bundle=load_data(c)
        data=bundle['data']
        split=split_episodes(bundle['keys'],c['split_seed'],c['episodes_per_role'])
        declaration=dict(source_sha256=sources,input_sha256=bundle['input_hashes'],bank_sha256=bundle['bank_sha256'],
            pool_sha256=bundle['pool']['freeze_sha256'],roles={r:dict(indices=ix.tolist(),episode_keys=[bundle['keys'][i] for i in ix]) for r,ix in split.items()},
            previous_exposure='All episodes were previously exposed; audit is not fresh confirmation')
        identity=digest_json(declaration)
        run.write_json('declaration.json',dict(identity=identity,declaration=declaration))
        runtime=load_frozen_runtime(bundle['config'],ROOT,require_archived=True)
        model,original_probe=runtime['model'],runtime['probe']
        model.requires_grad_(False)
        initial=tensor_state(model)
        original_probe_state=tensor_state(original_probe)
        phase='inference'
        original_predict=model.predict
        def measured(z,a):
            deadline()
            counter['predictor_training_rows' if phase=='training' else 'predictor_inference_rows']+=len(z)
            return original_predict(z,a)
        model.predict=measured
        arrays=training_arrays(data,bundle['history'],bundle['history_actions'],split['dynamics_fit'],model,runtime['scaler'])
        ev=np.concatenate([split['development'],split['audit']])
        ndev=len(split['development'])
        predictions={arm:[] for arm in ARMS[1:]}
        real_probes=[]
        training=[]
        checkpoint_hashes={}
        query=QueryStore(run.run_dir/'queries',study_identity=identity)
        im=HistoryImaginer(model,runtime['scaler'],original_probe,device='cuda')
        for seed in c['seeds']:
            deadline()
            torch.manual_seed(seed)
            model.load_state_dict(initial)
            probe=copy.deepcopy(original_probe)
            phase='training'
            ph=fit_probe_copy(probe,data,split['probe_fit'],seed=seed,steps=c['probe_steps'],batch=c['batch'],lr=c['probe_lr'],counter=counter,deadline=deadline)
            dh=fit_predictor(model,original_probe,arrays,seed=seed,steps=c['predictor_steps'],batch=c['batch'],lr=c['predictor_lr'],physical_weight=c['physical_loss_weight'],counter=counter,deadline=deadline)
            phase='inference'
            assert all(torch.equal(v.cpu(),original_probe_state[k]) for k,v in original_probe.state_dict().items())
            checkpoint=run.run_dir/f'repair-seed{seed}.pt'
            torch.save(dict(seed=seed,identity=identity,predictor=tensor_state(model.predictor),pred_proj=tensor_state(model.pred_proj),probe=tensor_state(probe)),checkpoint)
            checkpoint_hashes[checkpoint.name]=file_sha256(checkpoint)
            predictor_delta=sum(float((v.cpu().double()-initial[k].double()).square().sum()) for k,v in model.state_dict().items() if k.startswith(('predictor.','pred_proj.')))**.5
            probe_delta=sum(float((v.cpu().double()-original_probe_state[k].double()).square().sum()) for k,v in probe.state_dict().items())**.5
            assert predictor_delta>0 and probe_delta>0
            training.append(dict(seed=seed,probe=ph,predictor=dh,checkpoint=checkpoint.name,predictor_delta_l2=predictor_delta,probe_delta_l2=probe_delta,frozen_parameters_and_buffers_verified=True))
            run.write_json('training_history.json',training)
            run.write_json('current_costs.json',counter)
            newz=[]
            for i,row in enumerate(bundle['pool']['entries']):
                def compute(i=i):
                    z=[refreshed_predictions(im,bundle['history'][ev],data['z'][i,ev],bundle['history_actions'][ev],data['actions'][i,ev],r).cpu().numpy() for r in [1,10]]
                    return dict(z=np.stack(z))
                def query_cost():
                    return dict(predictor_rows=counter['predictor_inference_rows'],real_steps=0,renders=0,encodes=0)
                out,_=query.execute(f'seed{seed}-{row["id"]}',request=dict(seed=seed,candidate=row,episodes=[bundle['keys'][j] for j in ev],modes=c['inference_modes'],checkpoint_sha256=checkpoint_hashes[checkpoint.name]),expected_costs=dict(predictor_rows=len(ev)*20,real_steps=0,renders=0,encodes=0),callback=compute,read_costs=query_cost)
                newz.append(out['z'])
            z=np.stack(newz,axis=1)
            predictions['probe'].append(read_probe(probe,bundle['base_z'][:2,:,ev]))
            predictions['dynamics'].append(read_probe(original_probe,z))
            predictions['combined'].append(read_probe(probe,z))
            real_probes.append(read_probe(probe,data['z'][:,ev]))
            print(f'[repair] seed {seed} training and prediction complete',flush=True)
        predictions={k:np.stack(v) for k,v in predictions.items()}
        predictions['baseline']=bundle['base_readout'][:2,:,ev][None]
        devpred={k:v[:,:,:, :ndev] for k,v in predictions.items()}
        dev=summarize_arms(devpred,data,split['development'],c)
        choice=select_repair({a:dev[a]['teacher_forced'] for a in ARMS},c['gate'])
        run.write_json('development_choice.json',dict(summary=dev,choice=choice,checkpoint_sha256=checkpoint_hashes))
        # This freeze occurs before any numerical audit summary or controller audit access.
        nominations=dict(original=nominate(bundle['base_readout'][2][:,split['development']][None],c['minimum_progress_ratio']))
        chosen=choice['chosen']
        if chosen:
            ro=[]
            for seed in c['seeds']:
                cp=torch.load(run.run_dir/f'repair-seed{seed}.pt',map_location='cpu',weights_only=True)
                model.load_state_dict(initial)
                if chosen in ['dynamics','combined']:
                    model.predictor.load_state_dict(cp['predictor'])
                    model.pred_proj.load_state_dict(cp['pred_proj'])
                model.eval().requires_grad_(False)
                probe=copy.deepcopy(original_probe)
                if chosen in ['probe','combined']:
                    probe.load_state_dict(cp['probe'])
                if chosen=='probe':
                    ro.append(read_probe(probe,bundle['base_z'][2][:,split['development']]))
                    continue
                imagined=HistoryImaginer(model,runtime['scaler'],probe,device='cuda')
                cells=[]
                for i,row in enumerate(bundle['pool']['entries']):
                    def compute(i=i,row=row):
                        out=imagined.rollout_policies(ActionLimitedPolicy(runtime['policy'],row['cap']),bundle['theta'][i:i+1],bundle['history'][split['development']],bundle['history_actions'][split['development']],n_samples=1,k=0,seed=0,n_blocks=10,max_batch=16384)
                        return dict(readout=out['readout'][0,:,0])
                    out,_=query.execute(f'closed-seed{seed}-{row["id"]}',request=dict(seed=seed,arm=chosen,candidate=row,episodes=[bundle['keys'][j] for j in split['development']],checkpoint_sha256=checkpoint_hashes[f'repair-seed{seed}.pt']),expected_costs=dict(predictor_rows=ndev*10,real_steps=0,renders=0,encodes=0),callback=compute,read_costs=lambda:dict(predictor_rows=counter['predictor_inference_rows'],real_steps=0,renders=0,encodes=0))
                    cells.append(out['readout'])
                ro.append(np.stack(cells))
            repaired_closed=np.stack(ro)
            nominations['repaired']=nominate(repaired_closed,c['minimum_progress_ratio'])
            np.savez_compressed(run.run_dir/'repaired_development_closed.npz',readout=repaired_closed)
        run.write_json('controller_nominations.json',dict(chosen_repair=chosen,nominations=nominations,audit_not_yet_summarized=True))
        auditpred={k:v[:,:,:,ndev:] for k,v in predictions.items()}
        audit=summarize_arms(auditpred,data,split['audit'],c)
        gates={a:repair_gate(audit['baseline']['teacher_forced'],audit[a]['teacher_forced'],c['gate']) for a in ARMS[1:]}
        realprobe=np.stack(real_probes)
        probe_summary={role:dict(baseline=summarize(data['readout'][:,ix][None],data['truth'][:,ix],data['first'][:,ix],seed=c['bootstrap_seed'],replicates=c['bootstrap_replicates']),
            repaired=summarize(realprobe[:,:,slice(0,ndev) if role=='development' else slice(ndev,None)],data['truth'][:,ix],data['first'][:,ix],seed=c['bootstrap_seed'],replicates=c['bootstrap_replicates'])) for role,ix in [('development',split['development']),('audit',split['audit'])]}
        per_seed={role:{arm:[{mode:summarize(v[s:s+1,m],data['truth'][:,split[role]],data['first'][:,split[role]],seed=c['bootstrap_seed'],replicates=c['bootstrap_replicates']) for m,mode in enumerate(c['inference_modes'])} for s in range(len(v))] for arm,v in pred.items()} for role,pred in [('development',devpred),('audit',auditpred)]}
        result=dict(development=dev,audit=audit,choice=choice,audit_gates=gates,real_probe=probe_summary,per_seed=per_seed,
            controller_audit=controller_audit(nominations,data,split['audit'],c),
            prediction_repair_audit_passed=bool(chosen and gates[chosen]['passed']),
            claim='Exploratory reused-episode audit, not fresh confirmation or a safety guarantee')
        np.savez_compressed(run.run_dir/'evaluation.npz',**predictions,real_probe=realprobe,truth=data['truth'][:,ev],first=data['first'][:,ev],real_readout=data['readout'][:,ev],progress=data['progress'][:,ev],episode_indices=ev,original_closed=bundle['base_readout'][2][:,split['development']][None])
        run.write_json('analysis.json',result)
        assert counter['predictor_gradient_updates']==len(c['seeds'])*c['predictor_steps']
        assert counter['probe_gradient_updates']==len(c['seeds'])*c['probe_steps']
        assert counter['predictor_training_rows']==len(c['seeds'])*c['predictor_steps']*c['batch']
        account=query.accounting()
        assert not account['pending_queries']
        assert account['completed_costs']['predictor_rows']==counter['predictor_inference_rows']
        assert all(file_sha256(ROOT/p)==sha for p,sha in bundle['input_hashes'].items())
        counter['wall_s']=time.time()-started
        run.write_json('costs.json',counter)
        run.write_json('completion.json',dict(complete=True,identity=identity,checkpoints=checkpoint_hashes,accounting=account,evaluation_sha256=file_sha256(run.run_dir/'evaluation.npz'),frozen_encoder_projector_action_encoder_and_buffers_verified=True,optimizer_counts_verified=True,all_parent_hashes_unchanged=True))
        run.finish(build_manifest(run_id=run.run_id,kind='evo-repair-v1',costs=counter,started_at=started,data=runtime['provenance'],seeds=dict(split=c['split_seed'],training=c['seeds']),extra=dict(study_identity=identity)),upload=c['archive_private'])
        print(json.dumps(dict(complete=True,choice=choice,costs=counter)),flush=True)
    except BaseException as exc:
        run.write_json('failure.json',dict(error=type(exc).__name__,message=str(exc),costs=counter,implicit_retry_forbidden=True))
        raise


if __name__=='__main__':
    main()
