"""Verify complete frozen evaluation receipts before forming paired study outcomes."""
import hashlib
import json
from pathlib import Path

import numpy as np

from helpers.evoRanking import segment_metrics
from helpers.evoReadinessQueries import digest_array,digest_json
from helpers.evoReadinessStudy import checked_audit_output,checked_real_output,load_frozen
from helpers.evoRun import noise_seed
from helpers.walkerRules import health_clearance,rule_unsafe


def verified_query(path,*,study_identity,request,expected_costs):
    path=Path(path)
    with np.load(path,allow_pickle=False) as saved:
        receipt=json.loads(str(saved['__receipt__']))
        arrays={k:saved[k].copy() for k in saved.files if k!='__receipt__'}
    identity=digest_json(dict(study=study_identity,key=path.stem,request=request,expected_costs=expected_costs))
    if receipt['identity']!=identity or receipt['study_identity']!=study_identity or receipt['key']!=path.stem or receipt['request']!=request:
        raise ValueError('evaluation receipt does not match the declared inputs')
    if receipt['data_sha256']!={k:digest_array(v) for k,v in arrays.items()}:
        raise ValueError('evaluation numeric content hash mismatch')
    if any(not isinstance(v,int) or isinstance(v,bool) or v<0 for v in receipt['costs'].values()):
        raise ValueError('evaluation receipt has invalid measured cost counters')
    if any(receipt['costs'].get(k)!=v for k,v in expected_costs.items()):
        raise ValueError('evaluation receipt has incorrect charged costs')
    if any(v.dtype.kind not in 'biufc' or not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('evaluation contains invalid numeric output')
    return arrays,receipt


def load_study_outcomes(directory,*,expected_roots=None):
    directory=Path(directory)
    study=json.loads((directory/'study.json').read_text())
    identity=study['identity']
    if digest_json(study['declaration'])!=identity:
        raise ValueError('study declaration identity changed')
    frozen,theta,baseline_theta=load_frozen(directory,identity=identity)
    progress=json.loads((directory/'evaluation_progress.json').read_text())
    if not progress['complete'] or progress['study_identity']!=identity:
        raise ValueError('final evaluation is incomplete')
    folder=directory/'evaluation'
    if list(folder.glob('*.pending.json')):
        raise ValueError('unresolved evaluation queries must be reconciled before analysis')
    declaration=study['declaration']
    role=declaration['banks']['evaluation']
    episode_keys=role['episodes']
    roots=len(episode_keys)
    if expected_roots is not None and len(expected_roots)!=roots:
        raise ValueError('physical root count differs from declared evaluation bank')
    evaluation=declaration['evaluation']
    horizon=evaluation['horizon_blocks']
    levels=evaluation['noise_levels']
    if sorted(levels)!=[0.,1.]:
        raise ValueError('both imagined audits are required')
    seeds=list(dict.fromkeys(s['config']['seed'] for s in declaration['specs']))
    required={(s['noise_k'],s['config']['penalty_mode'],s['config']['population'],g,s['config']['seed'])
        for s in declaration['specs'] for g in s['config']['checkpoints'] if g!=0}
    if [p['slot'] for p in frozen['picks']]!=list(range(len(frozen['picks']))):
        raise ValueError('frozen selection slots differ from archive order')
    observed={(p['noise_k'],p['penalty_mode'],p['population'],p['generation'],p['seed']) for p in frozen['picks']}
    conditions={p[:4] for p in observed}
    if any({p[4] for p in observed if p[:4]==key}!=set(seeds) for key in conditions):
        raise ValueError('every condition must contain every declared paired search seed')
    if len(set(episode_keys))!=roots:
        raise ValueError('evaluation episodes are not independent unique sources')
    if observed!=required or len(observed)!=len(frozen['picks']):
        raise ValueError('frozen selections do not cover the complete declared grid')
    common=dict(selection_archive=frozen['parameters_sha256'],evaluation_bank=role,horizon_blocks=horizon)
    read_files,receipts={},[]
    def read(key,request,costs):
        path=folder/f'{key}.npz'
        out,receipt=verified_query(path,study_identity=identity,request=request,expected_costs=costs)
        read_files[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        receipts.append(receipt)
        return out
    def real(key,parameters,reference=False):
        request=dict(**common,kind='reference' if reference else 'real',theta=None if reference else digest_array(parameters))
        out=read(key,request,dict(real_steps=roots*horizon*10,predictor_rows=0))
        checked_real_output(out,roots,horizon,reference=reference)
        if expected_roots is not None:
            for field in ['qpos','qvel']:
                initial=np.stack([getattr(r,field) for r in expected_roots])
                if not np.array_equal(out[field][:,0],initial):
                    raise ValueError('evaluated initial physics states differ from declared root order')
        if not reference and 'pairs' in out and not np.array_equal(out['pairs'],np.column_stack([np.zeros(roots,int),np.arange(roots)])):
            raise ValueError('real executor task ordering differs from episode ordering')
        return out
    def audit(key,parameters,k,seed):
        n=evaluation['audit_samples']['zero_noise' if k==0 else 'positive_noise']
        request=dict(**common,kind='audit',theta=digest_array(parameters),noise_k=k,samples=n,noise_seed=noise_seed(seed,200000))
        out=read(key,request,dict(predictor_rows=roots*n*horizon,real_steps=0))
        checked_audit_output(out,roots,n,horizon)
        return segment_metrics(out['readout'])['violated'].mean(-1)
    real_base=real('baseline-real',baseline_theta)
    reference=real('reference-real',None,True)
    baseline=dict(real=np.broadcast_to(real_base['dense_violated'],(len(seeds),roots)).copy(),
        progress=np.broadcast_to(real_base['progress'],(len(seeds),roots)).copy(),imagined={})
    for k in levels:
        baseline['imagined'][k]=np.stack([audit(f'baseline-seed{seed}-k{k:g}',baseline_theta,k,seed) for seed in seeds])
    cells,diagnostics={},{}
    for pick,parameters in zip(frozen['picks'],theta):
        key=(pick['noise_k'],pick['penalty_mode'],pick['population'],pick['generation'])
        if key not in cells:
            cells[key]=dict(real=np.empty((len(seeds),roots),bool),progress=np.empty((len(seeds),roots)),
                imagined={k:np.empty((len(seeds),roots)) for k in levels})
            diagnostics[key]=dict(real_readout=np.empty((len(seeds),roots),bool),endpoint_truth=np.empty((len(seeds),roots),bool))
        si=seeds.index(pick['seed'])
        prefix=f'pick{pick["slot"]:04d}'
        result=real(prefix+'-real',parameters)
        cells[key]['real'][si]=result['dense_violated']
        cells[key]['progress'][si]=result['progress']
        for k in levels:
            cells[key]['imagined'][k][si]=audit(prefix+f'-k{k:g}',parameters,k,pick['seed'])
        diagnostics[key]['real_readout'][si]=segment_metrics(result['readout'])['violated']
        ends=result['qpos'][:,10::10]
        diagnostics[key]['endpoint_truth'][si]=rule_unsafe('health',health_clearance(ends[...,1],ends[...,2])).any(1)
    expected_files=set(read_files)
    if {p.name for p in folder.glob('*.npz')}!=expected_files or progress['expected_queries']!=len(expected_files):
        raise ValueError('unexpected, missing or undeclared final-evaluation query files')
    costs={}
    for receipt in receipts:
        for k,v in receipt['costs'].items():
            costs[k]=costs.get(k,0)+v
    if costs!=progress['accounting']['completed_costs']:
        raise ValueError('evaluation accounting summary differs from verified receipts')
    return dict(cells=cells,baseline=baseline,diagnostics=diagnostics,reference=reference,
        search_seeds=seeds,episode_keys=episode_keys,
        provenance=dict(study_identity=identity,frozen_selections_sha256=frozen['freeze_sha256'],
                        evaluation_file_sha256=read_files,evaluation_costs=costs))


def save_paired_outcomes(directory,outcomes):
    """Compact numeric export with explicit condition, seed and episode axes."""
    from helpers.evoReadinessQueries import atomic_json
    from helpers.evoRun import _atomic
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    keys=sorted(outcomes['cells'])
    arrays={name:np.stack([outcomes['cells'][key][name] for key in keys]) for name in ['real','progress']}
    for k in [0.,1.]:
        arrays[f'imagined_k{k:g}']=np.stack([outcomes['cells'][key]['imagined'][k] for key in keys])
        arrays[f'baseline_imagined_k{k:g}']=outcomes['baseline']['imagined'][k]
    arrays.update(baseline_real=outcomes['baseline']['real'],baseline_progress=outcomes['baseline']['progress'],
        reference_real=outcomes['reference']['dense_violated'],reference_progress=outcomes['reference']['progress'])
    for name in ['real_readout','endpoint_truth']:
        arrays[name]=np.stack([outcomes['diagnostics'][key][name] for key in keys])
    _atomic(directory/'paired_outcomes.npz',lambda f:np.savez_compressed(f,**arrays))
    atomic_json(directory/'paired_axes.json',dict(conditions=[dict(noise_k=k,penalty_mode=m,population=p,generation=g) for k,m,p,g in keys],
        search_seeds=outcomes['search_seeds'],episode_keys=outcomes['episode_keys'],provenance=outcomes['provenance']))
