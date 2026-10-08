"""Action constraints, honest sampling, frozen selectors, costs and exact resume."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from helpers.evoCandidateQuality import ActionLimitedPolicy, build_pool, choose_controllers
from helpers.evoCandidateQuality import diagnostic_analysis, execute_diagnostic
from helpers.evoFollowup import freeze
from helpers.evoReadinessQueries import QueryStore, atomic_json, digest_json
from helpers.evoReadinessQueries import IncompleteQueryError
from test_evo_readiness_study import bank, real_output

ROOT = Path(__file__).resolve().parents[2]


class SimplePolicy:
    device = 'cpu'
    n_params = 2
    action_mean = torch.zeros(6)
    action_std = torch.ones(6)

    def act(self, theta, features):
        return (features + torch.as_tensor(theta)[:,None,:1]).clamp(-1,1)


@pytest.mark.parametrize('cap',[None,.025,.1])
def test_cap_holds_at_every_visited_input_and_uncapped_is_exact(cap):
    base = SimplePolicy()
    policy = ActionLimitedPolicy(base,cap)
    theta = torch.tensor([[.8,0],[-.8,0]])
    feats = torch.linspace(-1,1,120).reshape(2,1,60)
    out = policy.act(theta,feats)
    baseline = base.act(torch.zeros_like(theta),feats)
    assert out.abs().max()<=1
    if cap is None:
        assert torch.equal(out,base.act(theta,feats))
    else:
        assert (out-baseline).abs().max()<=cap+1e-7
    stats = policy.statistics()
    assert stats['action_elements']==120 and stats['policy_calls']==2
    assert np.isclose(stats['action_delta_rms'],float((out-baseline).double().square().mean().sqrt()))
    assert torch.equal(policy.act(torch.zeros_like(theta),feats),baseline)


def test_uniform_population_sampling_and_paired_scales(tmp_path):
    c = yaml.safe_load((ROOT/'configs/evo/candidate_quality_20261005.yaml').read_text())
    cfg = c['candidate_quality']
    cfg.update(parent_run='parent',parent_searches=['one'],sample_generations=[0],
               random_directions=2)
    folder = tmp_path/'parent/study'
    folder.mkdir(parents=True)
    declaration = {'frozen':'parent'}
    identity = digest_json(declaration)
    atomic_json(folder/'study.json',dict(identity=identity,declaration=declaration))
    theta = np.ones((2,1020))
    theta[0]=0
    freeze(folder,'shortlists',[dict(name='one',offset=0,imagined_slot=1,noise_k=0)],theta,identity=identity)
    store = QueryStore(folder/'searches/one/queries/k0',study_identity=identity)
    store.execute('fitness-g0000',request={},expected_costs={},
        callback=lambda:dict(theta=np.arange(8*1020).reshape(8,1020)),read_costs=lambda:{})
    rows,params,inputs = build_pool(c,tmp_path)
    # No simulator-selection or evaluation files even exist in this parent fixture.
    again = build_pool(c,tmp_path)
    assert rows==again[0] and np.array_equal(params,again[1]) and inputs==again[2]
    for direction in range(2):
        pairs=[(i,r['origins'][0]['scale']) for i,r in enumerate(rows)
               if r['origins'][0].get('direction')==direction]
        reference=params[pairs[0][0]]/pairs[0][1]
        assert all(np.array_equal(params[i]/s,reference) for i,s in pairs)


def fixture(tmp_path):
    config={'candidate_quality':dict(noise_samples=2,diagnostic_fixed_penalty=.5,
        minimum_progress_ratio=.9,audit_seed=12,bootstrap_seed=13,bootstrap_replicates=100,
        banks=dict(imagined_selection=4,real_selection=4,evaluation=4),max_hours=1)}
    banks={role:bank(role,4,20*i) for i,role in enumerate(config['candidate_quality']['banks'])}
    rows=[dict(id=f'candidate-{i:03}',cap=None,origins=[dict(kind=kind)])
          for i,kind in enumerate(['baseline','random_direction','imagined_winner'])]
    theta=np.array([[0.,0.],[1.,0.],[2.,0.]])
    counts=dict(real_steps=0,predictor_rows=0)
    calls=[]
    def real(role,row,th):
        assert (tmp_path/'candidate_pool.json').exists()
        if role=='evaluation':
            assert (tmp_path/'diagnostic_choices.json').exists()
        calls.append(role)
        counts['real_steps']+=400
        out=real_output(2+th[0]/10)
        out.update(action_delta_rms=np.asarray(th[0]/100),endpoint_clearance=np.ones((4,10)))
        out['readout'][...,0]=1.2
        return out
    def imagined(role,row,th,k,n):
        if role=='evaluation':
            assert (tmp_path/'diagnostic_choices.json').exists()
        counts['predictor_rows']+=4*n*10
        readout=np.zeros((4,n,10,3))
        readout[...,0]=1.2
        readout[...,2]=2+th[0]
        return dict(readout=readout)
    args=dict(provenance={'synthetic':True},real_callback=real,imagined_callback=imagined,
              read_costs=lambda:counts)
    return config,rows,theta,banks,counts,calls,args


def test_freezes_accounting_resume_and_changed_pool_rejected(tmp_path):
    c,rows,theta,banks,counts,calls,args=fixture(tmp_path)
    def run(**kw):
        return execute_diagnostic(tmp_path,c,rows,theta,banks,**args,**kw)
    assert not run(stop_after='selection')['complete']
    assert set(calls)=={'real_selection'}
    result=run()
    assert counts=={'real_steps':2400,'predictor_rows':720}
    assert result['complete'] and result['costs']==counts
    assert result['analysis']['frozen_choices']['simulator/all']['baseline_fallback']
    saved=counts.copy()
    assert run()==result and counts==saved
    theta[1,0]=3
    with pytest.raises(ValueError,match='changed'):
        run()


def test_interrupted_query_cannot_be_implicitly_replayed(tmp_path):
    c,rows,theta,banks,counts,calls,args=fixture(tmp_path)
    def fail(*a):
        counts['real_steps']+=3
        raise RuntimeError('partial')
    args['real_callback']=fail
    with pytest.raises(RuntimeError,match='partial'):
        execute_diagnostic(tmp_path,c,rows,theta,banks,**args)
    saved=counts.copy()
    with pytest.raises(IncompleteQueryError):
        execute_diagnostic(tmp_path,c,rows,theta,banks,**args)
    assert counts==saved


def test_selection_uses_failure_progress_and_baseline_not_check_outcomes():
    rows=[dict(origins=[dict(kind='baseline')]),dict(origins=[dict(kind='random_direction')]),
          dict(origins=[dict(kind='imagined_winner')])]
    f=np.array([[1,1,0,0],[1,0,0,0],[0,0,0,0]])
    p=np.array([[10]*4,[9]*4,[8]*4])
    choices=choose_controllers(rows,f,p,{'im':[0,1,2]},.9)
    assert choices['simulator/all']['index']==1
    assert choices['im/all']['index']==2
    assert choices['simulator/imagined_winner']['index']==0


def test_check_envelope_kept_separate_from_frozen_choice():
    rows=[dict(id=str(i),origins=[dict(kind='x')]) for i in range(2)]
    selected=dict(failure=np.array([[1,0,0,0],[1,1,0,0]]),progress=np.ones((2,4)))
    check=deepcopy(selected)
    check.update(failure=np.array([[1,0,0,0],[0,0,0,0]]),action_rms=np.zeros(2),
        imagined_failure={'0':np.zeros((2,4))},readout_failure=np.zeros((2,4)),
        endpoint_failure=np.zeros((2,4)))
    out=diagnostic_analysis(rows,selected,check,{'frozen':dict(index=0)},
                            progress_floor=.9,seed=1,replicates=100)
    assert out['check_eligible_indices']==[1]
    assert out['frozen_choices']['frozen']['baseline_fallback']
    assert out['descriptive_rank_correlations']['0'] is None
    assert not out['eligible_on_both']
