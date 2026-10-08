"""Follow-up budget, selection leakage, fixed scoring, and replay invariants."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from helpers.evoFollowup import fixed_metrics, freeze, load_freeze, real_selection, shortlist
from helpers.evoFollowup import split_banks, study_plan, validate_banks
from helpers.evoFollowupStudy import DualAuditScorer, execute_study
from helpers.evoRanking import rosarl_scores
from helpers.evoReadinessQueries import IncompleteQueryError
from test_evo_readiness_study import FakeImaginer, bank, real_output

ROOT = Path(__file__).resolve().parents[2]


def config():
    c = yaml.safe_load((ROOT/'configs/evo/stage6_engineering_20261005.yaml').read_text())
    c['followup']['banks'] = dict(fitness=6,imagined_selection=4,real_selection=4,evaluation=4)
    c['followup']['search'].update(generations=2,noisy_population=3,noisy_fitness_samples=2,
                                  noisy_selection_samples=2,final_audit_samples=2,sigma0=.1)
    c['policy']['initial_CMA_sigma'] = .1
    return c


def test_exact_matching_and_development_guard():
    c = config()
    p = study_plan(c)
    assert not p['blockers']
    assert {a['population']*a['samples'] for a in p['arms']} == {6}
    assert p['counts']['fitness_rows_per_search']==2*6*4*10
    assert p['counts']['selection_audit_rows_per_search']==3*4*3*10
    c['followup']['phase']='confirmation'
    assert len(study_plan(c)['blockers'])==2


@pytest.mark.parametrize('field,value',[('shortlist_size',1),('shortlist_size',2.5),
    ('fixed_penalties',[0]),('fixed_penalties',[2,2]),('seeds',[1,1]),
    ('bank_source','unspecified'),('bootstrap_replicates',0)])
def test_invalid_protocol(field,value):
    c=config()
    c['followup'][field]=value
    with pytest.raises(ValueError):
        study_plan(c)


def test_fixed_penalty_uses_only_progress_before_failure():
    raw=dict(violated=np.array([[0,1]],bool),ret=np.array([[3.,100.]]),
             ret_pre=np.array([[3.,2.]]),rows=20)
    transformed=fixed_metrics(raw,4.)
    # The high post-failure reward must not rescue the unsafe segment.
    assert rosarl_scores(transformed['violated'],transformed['ret'],transformed['ret_pre'],0)==.5
    assert raw['ret'][0,1]==100 and raw['ret_pre'][0,1]==2


def test_roles_split_without_episode_leakage():
    parents=dict(fitness=bank('f',6,0),selection=bank('s',8,10),evaluation=bank('e',4,20))
    physical={key:list(range(len(b))) for key,b in parents.items()}
    banks,roots=split_banks(parents,physical,config()['followup']['banks'])
    assert roots['imagined_selection']==[0,1,2,3] and roots['real_selection']==[4,5,6,7]
    banks['real_selection']=banks['imagined_selection']
    with pytest.raises(ValueError,match='leakage'):
        validate_banks(banks)


def test_selection_prefers_risk_and_requires_progress_or_falls_back():
    f=np.array([[1,1,0,0],[1,0,0,0],[0,0,0,0],[1,0,0,0]])
    p=np.array([[10]*4,[9]*4,[8]*4,[9.5]*4])
    decision=real_selection(f,p,minimum_progress_ratio=.9)
    assert decision['slot']==3 and decision['eligible_slots']==[1,3]
    assert not decision['guarantee']
    assert real_selection(f,np.zeros_like(p),minimum_progress_ratio=.9)['slot']==0
    assert real_selection(np.zeros_like(f),p,minimum_progress_ratio=.9)['slot']==0
    with pytest.raises(ValueError):
        real_selection(np.empty((0,4)),np.empty((0,4)),minimum_progress_ratio=.9)


def test_shortlist_includes_baseline_and_deduplicates_best_nominee():
    state=dict(generation=3,nominees=[dict(theta=np.array(x,float)) for x in [[0,0],[1,1],[1,1],[2,2]]],
        snapshots={3:dict(scores=[0,3,3,2],nominee_index=1)})
    out=shortlist(state,4)
    assert out['nominee_indices']==[0,1,3] and out['imagined_slot']==1


def test_freeze_tampering_rejected(tmp_path):
    freeze(tmp_path,'choices',[dict(slot=0)],np.zeros((1,2)),identity='test')
    with pytest.raises(ValueError,match='changed'):
        freeze(tmp_path,'choices',[dict(slot=1)],np.zeros((1,2)),identity='test')
    np.savez(tmp_path/'choices.npz',theta=np.ones((1,2)))
    with pytest.raises(ValueError,match='parameters changed'):
        load_freeze(tmp_path,'choices',identity='test')


def setup_study(tmp_path):
    c=config()
    banks={r:bank(r,n,10*i) for i,(r,n) in enumerate(c['followup']['banks'].items())}
    counts=dict(predictor_rows=0,real_steps=0)
    im=FakeImaginer(counts)
    calls=[]
    def factory(cfg,arm,folder,identity):
        return DualAuditScorer(im,None,banks,cfg,arm,folder,identity,lambda:counts,2,16384)
    def real(role,theta):
        assert (tmp_path/'shortlists.json').exists()
        if role=='evaluation':
            assert (tmp_path/'selected.json').exists()
        calls.append(role)
        counts['real_steps']+=400
        return real_output(2+float(theta[0]))
    def audit(theta,k,n,seed):
        assert (tmp_path/'selected.json').exists()
        out=im.rollout_policies(None,theta[None],banks['evaluation'].z_hist,
            banks['evaluation'].hist_blocks,n_samples=n,k=k,seed=seed,n_blocks=10,max_batch=16384)
        return dict(readout=out['readout'][0])
    args=dict(provenance={'fixture':'frozen'},scorer_factory=factory,real_callback=real,
              audit_callback=audit,read_costs=lambda:counts)
    return c,banks,counts,calls,args


def test_global_barriers_exact_costs_and_resume(tmp_path):
    c,banks,counts,calls,args=setup_study(tmp_path)
    def run(**kw):
        return execute_study(tmp_path,c,banks,np.zeros(2),**args,**kw)
    assert not run(stop_after='search')['complete']
    assert not calls and not (tmp_path/'selected.json').exists()
    assert not run(stop_after='selection')['complete']
    assert calls and set(calls)=={'real_selection'}
    assert (tmp_path/'selected.json').exists()
    result=run()
    assert result['complete'] and not result['analysis']['confirmatory']
    assert counts=={k:result['actual_costs'][k] for k in counts}
    assert counts['predictor_rows']==study_plan(c)['counts']['total_predictor_rows']
    assert calls.index('evaluation')==calls.count('real_selection')
    saved=counts.copy()
    assert run()['actual_costs']==result['actual_costs']
    assert counts==saved
    changed=deepcopy(c)
    changed['followup']['fixed_penalties']=[4.]
    with pytest.raises(ValueError,match='declaration changed'):
        execute_study(tmp_path,changed,banks,np.zeros(2),**args)


def test_partial_real_query_cannot_be_replayed(tmp_path):
    c,banks,counts,calls,args=setup_study(tmp_path)
    def fail(role,theta):
        counts['real_steps']+=3
        raise RuntimeError('partial simulator failure')
    args['real_callback']=fail
    with pytest.raises(RuntimeError,match='partial simulator'):
        execute_study(tmp_path,c,banks,np.zeros(2),**args)
    assert not (tmp_path/'selected.json').exists()
    measured=counts.copy()
    with pytest.raises(IncompleteQueryError):
        execute_study(tmp_path,c,banks,np.zeros(2),**args)
    assert counts==measured
    pending=list((tmp_path/'real_selection').glob('*.pending.json'))
    assert json.loads(pending[0].read_text())['measured_costs']['real_steps']==3
