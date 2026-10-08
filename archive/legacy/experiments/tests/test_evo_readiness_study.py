"""Study boundaries, interrupted-query accounting and multi-arm resume invariants."""
import json
import sys
from pathlib import Path
from dataclasses import replace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from helpers.evoReadinessCMA import ReadinessCMAConfig
from helpers.evoReadinessQueries import (
    ImaginedSearchScorer, IncompleteQueryError, LatentBank, QueryStore, validate_roles,
)
from helpers.evoReadinessStudy import SearchSpec, evaluate_frozen, load_frozen, run_searches


def test_completed_query_reuse_and_partial_failure_costs(tmp_path):
    counts=dict(predictor_rows=0,real_steps=0)
    store=QueryStore(tmp_path,study_identity='declared')
    def query():
        counts['predictor_rows']+=20
        return dict(readout=np.arange(6).reshape(2,3))
    args=dict(request={'seed':3},expected_costs={'predictor_rows':20,'real_steps':0},
              callback=query,read_costs=lambda:counts)
    out,receipt=store.execute('ok',**args)
    again,_=store.execute('ok',**args)
    assert np.array_equal(out['readout'],again['readout']) and counts['predictor_rows']==20
    # Simulate termination after atomic result replacement but before journal deletion.
    (tmp_path/'ok.pending.json').write_text(json.dumps(dict(identity=receipt['identity'],key='ok')))
    store.execute('ok',**args)
    assert not (tmp_path/'ok.pending.json').exists() and counts['predictor_rows']==20
    with pytest.raises(ValueError,match='different inputs'):
        store.execute('ok',**{**args,'request':{'seed':4}})
    def partial():
        counts['real_steps']+=3
        raise RuntimeError('renderer stopped after physics')
    bad=dict(request={'theta':'fixed'},expected_costs={'predictor_rows':0,'real_steps':100},
             callback=partial,read_costs=lambda:counts)
    with pytest.raises(RuntimeError,match='renderer stopped'):
        store.execute('partial',**bad)
    journal=json.loads((tmp_path/'partial.pending.json').read_text())
    assert journal['measured_costs']=={'predictor_rows':0,'real_steps':3}
    with pytest.raises(IncompleteQueryError):
        store.execute('partial',**bad)
    assert counts['real_steps']==3
    accounting=store.accounting()
    assert accounting['completed_costs']=={'predictor_rows':20,'real_steps':0}
    assert len(accounting['pending_queries'])==1
    assert not accounting['pending_queries'][0]['archive_committed']


def test_saved_query_corruption_is_not_reused(tmp_path):
    counts={'predictor_rows':0,'real_steps':0}
    store=QueryStore(tmp_path,study_identity='fixed')
    args=dict(request={},expected_costs=counts.copy(),callback=lambda:{'x':np.arange(2)},read_costs=lambda:counts)
    store.execute('q',**args)
    with np.load(tmp_path/'q.npz',allow_pickle=False) as saved:
        receipt=saved['__receipt__'].copy()
    np.savez(tmp_path/'q.npz',x=np.array([4,5]),__receipt__=receipt)
    with pytest.raises(ValueError,match='hash mismatch'):
        store.execute('q',**args)


def bank(prefix,n,offset):
    z=np.zeros((n,3,192),np.float32)
    z[:,:,0]=np.arange(n)[:,None]+offset
    return LatentBank(tuple(f'{prefix}-{i}' for i in range(n)),z,np.zeros((n,2,10,6),np.float32))


class FakeImaginer:
    def __init__(self,counts):
        self.counts=counts
        self.seen=[]

    def rollout_policies(self,policy,theta,zh,hist,*,n_samples,k,seed,n_blocks,max_batch):
        self.counts['predictor_rows']+=len(zh)*n_samples*n_blocks
        self.seen.append((zh[:,0,0].copy(),seed))
        rng=np.random.default_rng(seed)
        ro=np.zeros((1,len(zh),n_samples,n_blocks,3))
        ro[...,0]=1.2
        ro[...,2]=2+float(theta[0,0])+k*.01*rng.normal(size=(len(zh),n_samples,n_blocks))
        if theta[0,1]>.04:
            ro[:,::2,:,5:,0]=.6
        return dict(readout=ro)



def real_output(progress,reference=False):
    qp=np.zeros((4,101,9))
    qp[:,:,1]=1.2
    qp[:,:,0]=np.linspace(0,progress,101)
    out=dict(qpos=qp,qvel=np.zeros_like(qp),progress=qp[:,-1,0],dense_violated=np.zeros(4,bool),
        actions=np.zeros((4,100,6) if reference else (4,10,60)))
    if not reference:
        out.update(readout=np.zeros((4,10,3)),dense_clearance=np.zeros((4,100)),
                   endpoint_targets=np.zeros((4,10,3)))
    return out


def study_setup():
    banks=dict(fitness=bank('fit',6,0),selection=bank('select',3,10),evaluation=bank('eval',4,20))
    c=ReadinessCMAConfig(population=4,generations=2,sigma0=.1,seed=31,n_fitness_roots=6,
        fitness_roots_per_generation=4,fitness_samples=1,selection_roots=3,selection_samples=1,
        checkpoints=(0,1,2))
    specs=[SearchSpec(k,replace(c,seed=seed,fitness_samples=1 if k==0 else 2,
                              selection_samples=1 if k==0 else 2)) for seed in [31,32] for k in [0.,1.]]
    evaluation=dict(audit_samples={'zero_noise':1,'positive_noise':2},noise_levels=[0.,1.],horizon_blocks=10)
    return banks,specs,evaluation


def test_multiarm_search_resume_freeze_and_paired_evaluation(tmp_path):
    banks,specs,evaluation=study_setup()
    states={}
    for name in ['full','resumed']:
        counts=dict(predictor_rows=0,real_steps=0)
        imaginer=FakeImaginer(counts)
        def factory(spec,folder,identity):
            return ImaginedSearchScorer(imaginer,None,banks['fitness'],banks['selection'],
                config=spec.config,noise_k=spec.noise_k,
                store=QueryStore(folder/'queries',study_identity=identity),read_costs=lambda:counts)
        path=tmp_path/name
        args=dict(provenance={'assets':'frozen'},evaluation_plan=evaluation,scorer_factory=factory)
        if name=='resumed':
            partial=run_searches(path,specs,banks,np.zeros(2),**args,max_generations_this_call=1)
            assert not partial['complete']
            assert not (path/'frozen_selections.json').exists()
            with pytest.raises(ValueError,match='frozen before evaluation'):
                load_frozen(path,identity=partial['identity'])
            # No baseline query for a later search after the scheduling allowance ends.
            assert len(list((path/'searches').iterdir()))==1
        done=run_searches(path,specs,banks,np.zeros(2),**args)
        assert done['complete'] and counts['predictor_rows']==2460 and counts['real_steps']==0
        assert all(np.max(ids)<20 for ids,_ in imaginer.seen)
        frozen,theta,baseline=load_frozen(path,identity=done['identity'])
        assert len(frozen['picks'])==8
        states[name]=(frozen,theta,counts.copy())
        # Already frozen search scheduling spends no further model queries.
        run_searches(path,specs,banks,np.zeros(2),**args)
        assert counts['predictor_rows']==2460
        calls=[]
        def real(theta):
            calls.append('real')
            counts['real_steps']+=400
            return real_output(float(theta[0])+1)
        def reference():
            calls.append('reference')
            counts['real_steps']+=400
            return real_output(1.,reference=True)
        def audit(theta,k,n,seed):
            calls.append('audit')
            counts['predictor_rows']+=4*n*10
            return {'readout':np.zeros((4,n,10,3))}
        kwargs=dict(identity=done['identity'],evaluation_bank=banks['evaluation'],
            real_callback=real,reference_callback=reference,audit_callback=audit,
            read_costs=lambda:counts,**evaluation)
        first=evaluate_frozen(path,**kwargs,max_new_queries=3)
        assert not first['complete'] and len(calls)==3
        before=counts.copy()
        with pytest.raises(ValueError,match='settings differ'):
            evaluate_frozen(path,**{**kwargs,'audit_samples':{'zero_noise':1,'positive_noise':3}})
        assert counts==before
        final=evaluate_frozen(path,**kwargs)
        assert final['complete'] and len(calls)==30
        assert counts=={'predictor_rows':3660,'real_steps':4000}
        assert final['accounting']['completed_costs']=={'predictor_rows':1200,'real_steps':4000}
        evaluate_frozen(path,**kwargs)
        assert len(calls)==30
    assert np.array_equal(states['full'][1],states['resumed'][1])
    assert states['full'][0]['picks']==states['resumed'][0]['picks']
    assert states['full'][2]==states['resumed'][2]


def test_episode_leakage_and_changed_inputs_rejected_before_queries(tmp_path):
    banks,specs,evaluation=study_setup()
    overlap=dict(banks,selection=LatentBank(banks['fitness'].episode_keys[:3],
        banks['selection'].z_hist,banks['selection'].hist_blocks))
    with pytest.raises(ValueError,match='overlap'):
        validate_roles(overlap)
    def forbidden(*args):
        raise AssertionError('must reject before constructing query callbacks')
    with pytest.raises(ValueError,match='overlap'):
        run_searches(tmp_path/'bad',specs,overlap,np.zeros(2),provenance={'x':1},
            evaluation_plan=evaluation,scorer_factory=forbidden)
    from helpers.evoReadinessStudy import prepare_study
    prepare_study(tmp_path/'declared',specs,banks,np.zeros(2),provenance={'x':1},evaluation_plan=evaluation)
    with pytest.raises(ValueError,match='inputs changed'):
        run_searches(tmp_path/'declared',specs,banks,np.zeros(2),provenance={'x':2},
            evaluation_plan=evaluation,scorer_factory=forbidden)


def test_real_receipt_requires_consistent_dense_truth_and_progress():
    from helpers.evoReadinessStudy import checked_real_output
    valid=real_output(2.)
    checked_real_output(valid,4,10)
    with pytest.raises(ValueError,match='dense physics truth'):
        checked_real_output(dict(valid,dense_violated=np.ones(4,bool)),4,10)
    with pytest.raises(ValueError,match='displacement'):
        checked_real_output(dict(valid,progress=np.zeros(4)),4,10)
    with pytest.raises(ValueError,match='incorrectly shaped'):
        checked_real_output(dict(valid,readout=np.zeros((4,9,3))),4,10)
