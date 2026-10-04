"""Main-study search scheduling, immutable selections and paired evaluation jobs.

Execution authorization and fresh-bank collection belong to the entry point. This
core also supports small, explicitly declared development integration checks.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from helpers.evoReadinessCMA import ReadinessCMAConfig, run_readiness_cma
from helpers.evoReadinessQueries import QueryStore, atomic_json, digest_array, digest_json, validate_roles
from helpers.evoRun import _atomic, noise_seed
from helpers.walkerRules import health_clearance, rule_unsafe


@dataclass(frozen=True)
class SearchSpec:
    noise_k: float
    config: ReadinessCMAConfig

    @property
    def name(self):
        c=self.config
        return f'k{self.noise_k:g}-{c.penalty_mode}-pop{c.population}-seed{c.seed}'


def prepare_study(directory,specs,banks,theta0,*,provenance,evaluation_plan):
    """Create or verify an identity before constructing any model-query callbacks."""
    if not specs or len({s.name for s in specs})!=len(specs):
        raise ValueError('nonempty unique search specifications required')
    if any(not np.isfinite(s.noise_k) or s.noise_k<0 for s in specs):
        raise ValueError('finite nonnegative noise required')
    if not provenance:
        raise ValueError('frozen input provenance required')
    roles=validate_roles(banks)
    if any(s.config.n_fitness_roots!=len(banks['fitness']) or
           s.config.selection_roots!=len(banks['selection']) for s in specs):
        raise ValueError('search sizes must match declared banks')
    theta0=np.asarray(theta0,np.float64)
    if theta0.ndim!=1 or len(theta0)<2 or not np.isfinite(theta0).all():
        raise ValueError('finite initial parameter vector required')
    declaration=dict(schema=1,specs=[asdict(s) for s in specs],banks=roles,evaluation=evaluation_plan,
        theta0_sha256=digest_array(theta0),provenance=provenance)
    identity=digest_json(declaration)
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/'study.json'
    if path.exists():
        saved=json.loads(path.read_text())
        if saved.get('identity')!=identity or digest_json(saved['declaration'])!=identity:
            raise ValueError('study inputs changed; cannot resume')
    else:
        if any(directory.iterdir()):
            raise ValueError('unrecognized nonempty study directory')
        atomic_json(path,dict(identity=identity,declaration=declaration))
    return identity


def load_frozen(directory,*,identity):
    directory=Path(directory)
    path=directory/'frozen_selections.json'
    if not path.exists():
        raise ValueError('all search selections must be frozen before evaluation')
    frozen=json.loads(path.read_text())
    content={k:v for k,v in frozen.items() if k!='freeze_sha256'}
    if frozen.get('freeze_sha256')!=digest_json(content):
        raise ValueError('selection freeze metadata changed')
    if frozen['study_identity']!=identity:
        raise ValueError('selection freeze belongs to a different study')
    weights=directory/'selected_policies.npz'
    if hashlib.sha256(weights.read_bytes()).hexdigest()!=frozen['parameters_sha256']:
        raise ValueError('frozen parameters changed')
    with np.load(weights,allow_pickle=False) as f:
        theta,baseline=f['theta'].copy(),f['baseline'].copy()
    if len(theta)!=len(frozen['picks']) or not np.isfinite(theta).all() or not np.isfinite(baseline).all():
        raise ValueError('invalid frozen selection archive')
    initial=json.loads((directory/'study.json').read_text())['declaration']['theta0_sha256']
    if digest_array(baseline)!=initial or any(digest_array(t)!=p['theta_sha256'] for t,p in zip(theta,frozen['picks'])):
        raise ValueError('frozen baseline or nominee identity differs')
    return frozen,theta,baseline


def run_searches(directory,specs,banks,theta0,*,provenance,evaluation_plan,scorer_factory,
                 max_generations_this_call=None):
    """Search every declared arm, then freeze all picks before permitting evaluation.

    scorer_factory receives (spec, search_dir, identity) and returns an object with
    fitness/selection callbacks. It must have access only to those episode roles.
    The optional limit is a scheduling pause at a complete generation, not a change
    to the scientific generation budget.
    """
    if max_generations_this_call is not None and max_generations_this_call<1:
        raise ValueError('scheduling allowance must be positive')
    directory=Path(directory)
    identity=prepare_study(directory,specs,banks,theta0,provenance=provenance,evaluation_plan=evaluation_plan)
    if (directory/'frozen_selections.json').exists():
        frozen,_,_=load_frozen(directory,identity=identity)
        return dict(complete=True,identity=identity,frozen=frozen)
    remaining=max_generations_this_call
    states=[]
    for spec in specs:
        folder=directory/'searches'/spec.name
        checkpoint=folder/'cma/checkpoint.pkl'
        before=0
        if checkpoint.exists():
            with checkpoint.open('rb') as f:
                before=pickle.load(f)['generation']
        # A pause must not initialize a new search and spend its baseline queries.
        if remaining==0 and before<spec.config.generations:
            break
        scorer=scorer_factory(spec,folder,identity)
        state=run_readiness_cma(spec.config,theta0,scorer.fitness,scorer.selection,folder/'cma',
            provenance=dict(study_identity=identity,search=spec.name),
            max_generations_this_call=remaining)
        states.append((spec,state))
        if remaining is not None:
            remaining-=state['generation']-before
        progress=dict(identity=identity,searches=[dict(name=s.name,generation=v['generation'],
            complete=v['generation']==s.config.generations,costs=v['costs']) for s,v in states])
        atomic_json(directory/'search_progress.json',progress)
        if state['generation']<spec.config.generations:
            break
    if len(states)!=len(specs) or any(v['generation']!=s.config.generations for s,v in states):
        return dict(complete=False,identity=identity)
    picks,theta=[],[]
    for spec,state in states:
        for generation in spec.config.checkpoints:
            if generation==0:
                continue
            pick=state['snapshots'][generation]
            picks.append(dict(slot=len(theta),search=spec.name,seed=spec.config.seed,
                population=spec.config.population,noise_k=spec.noise_k,
                penalty_mode=spec.config.penalty_mode,generation=generation,
                nominee_generation=pick['nominee_generation'],penalty=pick['penalty'],
                theta_sha256=digest_array(pick['theta'])))
            theta.append(pick['theta'])
    if not theta:
        raise ValueError('at least one nonzero selection checkpoint required')
    weights=directory/'selected_policies.npz'
    payload=dict(theta=np.stack(theta),baseline=np.asarray(theta0,np.float64))
    if weights.exists():
        # The parameter archive committed before a crash writing the JSON freeze.
        with np.load(weights,allow_pickle=False) as old:
            if set(old.files)!=set(payload) or any(not np.array_equal(old[k],v) for k,v in payload.items()):
                raise ValueError('uncommitted selection archive differs from completed searches')
    else:
        _atomic(weights,lambda f:np.savez_compressed(f,**payload))
    frozen=dict(study_identity=identity,picks=picks,
        parameters_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        phase='all_selections_frozen_before_evaluation',
        search_costs={s.name:v['costs'] for s,v in states})
    frozen['freeze_sha256']=digest_json(frozen)
    atomic_json(directory/'frozen_selections.json',frozen)
    return dict(complete=True,identity=identity,frozen=frozen)



def checked_real_output(output,roots,horizon_blocks,*,reference=False):
    steps=horizon_blocks*10
    shapes=dict(qpos=(roots,steps+1,9),qvel=(roots,steps+1,9),progress=(roots,),
        dense_violated=(roots,),actions=(roots,steps,6) if reference else (roots,horizon_blocks,60))
    if not reference:
        shapes.update(readout=(roots,horizon_blocks,3),dense_clearance=(roots,steps),
                      endpoint_targets=(roots,horizon_blocks,3))
    if any(name not in output or np.asarray(output[name]).shape!=shape for name,shape in shapes.items()):
        raise ValueError('incomplete or incorrectly shaped real evaluation')
    qp=np.asarray(output['qpos'])
    truth=rule_unsafe('health',health_clearance(qp[:,1:,1],qp[:,1:,2])).any(1)
    if not np.array_equal(output['dense_violated'],truth):
        raise ValueError('reported real failures differ from dense physics truth')
    if not np.array_equal(output['progress'],qp[:,-1,0]-qp[:,0,0]):
        raise ValueError('reported progress differs from displacement')
    return output


def checked_audit_output(output,roots,samples,horizon_blocks):
    if 'readout' not in output or np.asarray(output['readout']).shape!=(roots,samples,horizon_blocks,3):
        raise ValueError('incomplete or incorrectly shaped imagined audit')
    return output


def evaluate_frozen(directory,*,identity,evaluation_bank,real_callback,reference_callback,
                    audit_callback,read_costs,audit_samples,noise_levels=(0.,1.),
                    horizon_blocks=10,max_new_queries=None):
    """Callbacks run only after complete selection freeze; complete jobs are reused.

    real_callback(theta), reference_callback(), audit_callback(theta,k,n,seed)
    return numeric diagnostic arrays. read_costs counts predictor_rows and real_steps
    at their call sites. Interrupted requests retain a journal and refuse replay.
    """
    directory=Path(directory)
    frozen,theta,baseline=load_frozen(directory,identity=identity)
    study=json.loads((directory/'study.json').read_text())
    if study['identity']!=identity or study['declaration']['banks']['evaluation']!=evaluation_bank.identity():
        raise ValueError('evaluation episodes differ from the frozen study')
    requested_plan=dict(audit_samples=audit_samples,noise_levels=list(noise_levels),horizon_blocks=horizon_blocks)
    if digest_json(requested_plan)!=digest_json(study['declaration']['evaluation']):
        raise ValueError('evaluation settings differ from the frozen declaration')
    if set(audit_samples)!={'zero_noise','positive_noise'} or audit_samples['zero_noise']!=1:
        raise ValueError('one zero-noise sample and declared positive-noise samples required')
    if any(not isinstance(n,int) or n<1 for n in audit_samples.values()) or horizon_blocks<1:
        raise ValueError('positive integer evaluation sample counts and horizon required')
    if not noise_levels or len(set(noise_levels))!=len(noise_levels) or any(not np.isfinite(k) or k<0 for k in noise_levels):
        raise ValueError('finite nonnegative audit noise levels required')
    if max_new_queries is not None and max_new_queries<1:
        raise ValueError('positive query scheduling allowance required')
    store=QueryStore(directory/'evaluation',study_identity=identity)
    roots=len(evaluation_bank)
    common=dict(selection_archive=frozen['parameters_sha256'],evaluation_bank=evaluation_bank.identity(),
                horizon_blocks=horizon_blocks)
    jobs=[]
    def real_job(key,parameters,reference=False):
        request=dict(**common,kind='reference' if reference else 'real',
                     theta=None if reference else digest_array(parameters))
        def callback():
            output=reference_callback() if reference else real_callback(parameters)
            return checked_real_output(output,roots,horizon_blocks,reference=reference)
        jobs.append((key,request,dict(real_steps=roots*horizon_blocks*10,predictor_rows=0),callback))
    def audit_job(key,parameters,k,seed):
        n=audit_samples['zero_noise' if k==0 else 'positive_noise']
        stream=noise_seed(seed,200000)
        request=dict(**common,kind='audit',theta=digest_array(parameters),noise_k=k,samples=n,noise_seed=stream)
        jobs.append((key,request,dict(predictor_rows=roots*n*horizon_blocks,real_steps=0),
                     lambda:checked_audit_output(audit_callback(parameters,k,n,stream),roots,n,horizon_blocks)))
    real_job('baseline-real',baseline)
    real_job('reference-real',None,reference=True)
    for seed in sorted({p['seed'] for p in frozen['picks']}):
        for k in noise_levels:
            audit_job(f'baseline-seed{seed}-k{k:g}',baseline,k,seed)
    for pick,parameters in zip(frozen['picks'],theta):
        real_job(f'pick{pick["slot"]:04d}-real',parameters)
        for k in noise_levels:
            audit_job(f'pick{pick["slot"]:04d}-k{k:g}',parameters,k,pick['seed'])
    completed,new=0,0
    for key,request,costs,callback in jobs:
        already=(store.directory/f'{key}.npz').exists()
        if not already and max_new_queries is not None and new>=max_new_queries:
            break
        store.execute(key,request=request,expected_costs=costs,callback=callback,read_costs=read_costs)
        completed+=1
        new+=not already
    summary=dict(study_identity=identity,complete=completed==len(jobs),
        expected_queries=len(jobs),completed_queries=completed,accounting=store.accounting())
    atomic_json(directory/'evaluation_progress.json',summary)
    return summary
