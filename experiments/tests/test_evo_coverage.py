"""Training coverage isolation, equal budgets, and family-balanced checkpoint selection."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
from helpers.evoCoverage import coverage_indices,fit_coverage_seed  # noqa: E402
from helpers.evoInputs import load_stage_config  # noqa: E402
from helpers.evoMlpPolicy import MLPBlockPolicy  # noqa: E402


@pytest.fixture(scope='module',autouse=True)
def one_thread():
    old=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_matched_sampling_never_crosses_episode_roles_and_is_seeded():
    lag=np.repeat(np.arange(10),10)
    ppo=np.repeat(np.arange(100,110),3)
    idx,w,meta=coverage_indices(lag,[8,9],ppo,[108,109],seed=123,fit_episodes_per_family=4)
    assert len(idx['baseline_lag'])==len(idx['mixed_lag'])+len(idx['mixed_ppo'])==24
    assert len(idx['mixed_lag'])==len(idx['mixed_ppo'])==12
    for key in ['baseline_lag','mixed_lag']:
        assert not np.isin(lag[idx[key]],[8,9]).any()
        assert len(np.unique(idx[key]))==len(idx[key])
    assert not np.isin(ppo[idx['mixed_ppo']],[108,109]).any()
    assert set(lag[idx['validation_lag']])=={8,9}
    assert set(ppo[idx['validation_ppo']])=={108,109}
    assert w[:20].sum()==pytest.approx(.5) and w[20:].sum()==pytest.approx(.5)
    repeat,ww,mm=coverage_indices(lag,[8,9],ppo,[108,109],seed=123,fit_episodes_per_family=4)
    assert meta==mm and np.array_equal(w,ww)
    for key in idx:
        assert np.array_equal(idx[key],repeat[key])


def test_invalid_roles_and_insufficient_episodes_fail_before_training():
    with pytest.raises(ValueError,match='disjoint'):
        coverage_indices(np.arange(10),[9],np.arange(10),[9],seed=0,fit_episodes_per_family=1)
    with pytest.raises(ValueError,match='insufficient'):
        coverage_indices(np.arange(10),[9],np.arange(100,110),[109],seed=0,fit_episodes_per_family=10)
    with pytest.raises(ValueError,match='nonempty'):
        coverage_indices(np.arange(10),[],np.arange(100,110),[109],seed=0,fit_episodes_per_family=1)


def test_checkpoint_score_weights_families_not_sample_counts():
    torch.manual_seed(7)
    p=MLPBlockPolicy((np.zeros(6),np.ones(6)),hidden=(8,))
    x=torch.randn(32,384)
    y=torch.zeros(32,60)
    xv=torch.randn(8,384)
    yv=torch.cat([torch.zeros(2,60),torch.full((6,60),.8)]).requires_grad_()
    weights=np.r_[np.full(2,.25),np.full(6,1/12)]
    best,hist,updates=fit_coverage_seed(p,x,y,xv,yv,seed=3,epochs=4,batch_size=16,
        lr=.01,weight_decay=0.,val_weights=weights)
    pred=p.act(best['theta'][None],xv[None])[0]
    delta=(pred.double()-yv.detach().double()).square()
    independent=.5*float(delta[:2].mean())+.5*float(delta[2:].mean())
    assert best['val_mse']==pytest.approx(independent,abs=1e-7)
    assert abs(best['val_mse']-float(delta.mean()))>.01
    assert updates==8 and best['val_mse']==min(r['val_mse'] for r in hist)
    assert yv.grad is None
    with pytest.raises(ValueError,match='weights'):
        fit_coverage_seed(p,x,y,xv,yv,seed=3,epochs=1,batch_size=16,
            lr=.01,weight_decay=0.,val_weights=np.zeros(8))


def test_coverage_protocol_preserves_architecture_optimizer_and_development_role():
    old,new=load_stage_config('stage2_mlp'),load_stage_config('stage2_coverage')
    assert old.policy==new.policy and old.cache==new.cache and old.inputs==new.inputs
    for key in ['seeds','epochs','batch_size','optimizer','learning_rate','weight_decay','loss','refit_on_all']:
        assert old.training[key]==new.training[key]
    assert new.real.roots=='S4_development_only' and new.real.expected_roots==24
    assert new.interpretation.gate0=='not_run'
    assert new.interpretation.model_selection_uses_real_outcomes is False
