"""Tests of pairing, multiplicity, real-risk claims and undefined progress ratios."""
import sys
from pathlib import Path
from itertools import product

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from helpers.evoReadinessAnalysis import CrossedBootstrap, analyze_readiness


def test_crossed_weights_match_explicit_resampling_and_preserve_pairing():
    b=CrossedBootstrap(2,3,replicates=3,seed=4)
    seed_ids=[[0,0],[0,1],[1,1]]
    episode_ids=[[0,1,1],[2,1,0],[2,2,2]]
    b.seed_weights=np.array([[1,0],[.5,.5],[0,1]])
    b.episode_weights=np.array([[1/3,2/3,0],[1/3,1/3,1/3],[0,0,1]])
    x=np.array([[0.,2.,5.],[4.,1.,3.]])
    explicit=np.array([x[np.ix_(s,r)].mean() for s,r in zip(seed_ids,episode_ids)])
    assert np.allclose(b.draws(x),explicit,atol=1e-15)
    assert np.allclose(b.draws(x+7)-b.draws(x),7)
    assert b.interval(np.ones((2,3)),coverage=.975)['lo']==1
    assert b.ratio(x+1,x+1)['lo']==1


def fixture_cells():
    s,r=3,4
    baseline=dict(real=np.zeros((s,r)),progress=np.ones((s,r))*2,
                  imagined={0.:np.zeros((s,r)),1.:np.zeros((s,r))})
    cells={}
    for k,mode,pop,g in product([0.,1.],['zero','rosarl_style'],[16,256],[1,16]):
        high=pop==256 and g==16
        fixed=k==1 and mode=='rosarl_style'
        # The combined arm can inflate imagined risk without reducing real risk.
        cells[(k,mode,pop,g)]=dict(real=np.full((s,r),int(high)),progress=np.ones((s,r))*2,
            imagined={0.:np.zeros((s,r)),1.:np.full((s,r),float(high and fixed))})
    kwargs=dict(search_seeds=[1,2,3],episode_keys=['a','b','c','d'],population_sizes=[16,256],
                checkpoints=[1,16],replicates=100,bootstrap_seed=12)
    return cells,baseline,kwargs


def test_smaller_gap_alone_does_not_establish_real_safety_improvement():
    cells,base,kw=fixture_cells()
    out=analyze_readiness(cells,base,**kw)
    assert out['primary']['k0_zero_gap_amplification']['point']==1
    assert out['primary']['combined_method_minus_k0_zero_amplification']['point']==-1
    assert out['primary']['individual_coverage']==.975
    assert out['primary']['family_coverage']==.95
    assert not out['practical_benefit']['real_reduction_supported']
    assert not out['practical_benefit']['both_practical_conditions_supported']
    curve=next(c for c in out['curves'] if c['noise_k']==1 and c['penalty_mode']=='rosarl_style' and c['population']==256 and c['generation']==16)
    assert curve['gap_own_noise']['point']==0 and curve['gap_common_k0']['point']==1
    assert out['baseline']['real_failure']['degenerate_empirical_bootstrap']
    assert out['baseline']['fixed_policy_real_rate']['wilson_95'][1]>0
    assert out['sampling']['independent_episode_count']==4
    assert out['sampling']['total_pairs_are_not_independent_episodes']==12


def test_real_failure_reduction_requires_retained_progress():
    cells,base,kw=fixture_cells()
    high=cells[(1.,'rosarl_style',256,16)]
    high['real'][:]=0
    high['progress'][:]=1
    out=analyze_readiness(cells,base,**kw)
    assert out['practical_benefit']['real_reduction_supported']
    assert not out['practical_benefit']['progress_retention_supported']
    high['progress'][:]=2
    out=analyze_readiness(cells,base,**kw)
    assert out['practical_benefit']['both_practical_conditions_supported']


def test_nonpositive_progress_denominator_is_undefined():
    b=CrossedBootstrap(2,2,replicates=100,seed=19)
    result=b.ratio(np.ones((2,2)),np.array([[1.,-1.],[1.,-1.]]))
    assert not result['defined'] and result['point'] is None and result['lo'] is None
    assert result['nonpositive_bootstrap_denominators']>0


def test_incomplete_arms_or_duplicate_source_episodes_cannot_be_analyzed():
    cells,base,kw=fixture_cells()
    cells.pop((1.,'rosarl_style',256,16))
    with pytest.raises(ValueError,match='complete declared four-arm'):
        analyze_readiness(cells,base,**kw)
    cells,base,kw=fixture_cells()
    with pytest.raises(ValueError,match='unique search seeds'):
        analyze_readiness(cells,base,**{**kw,'episode_keys':['a','b','b','d']})
    base['real'][1,0]=1
    with pytest.raises(ValueError,match='shared real baseline'):
        analyze_readiness(cells,base,**kw)
