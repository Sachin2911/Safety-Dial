"""Frozen-controller identity, independent nominee selection and crossed uncertainty."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments'))
from helpers.evoReadinessSearch import FixedGainResidualPolicy, NomineeSelector, pilot_uncertainty, task_returns
from test_evo_transfer_pilot import policy_fixture


def test_preconditioned_residual_preserves_base_and_fixed_gain():
    residual = policy_fixture()
    scales = np.linspace(.3,2.,17).astype(np.float32)
    policy = FixedGainResidualPolicy(residual,scales)
    features = torch.randn(1,7,444)
    theta = np.zeros((1,1020))
    assert torch.equal(policy.act(theta,features),residual.act(np.zeros((1,1021)),features))
    theta = np.arange(1020,dtype=np.float32)[None]*1e-6
    raw = np.concatenate([(theta.reshape(1,60,17)/scales).reshape(1,1020),np.zeros((1,1))],axis=1)
    assert np.array_equal(policy.raw_theta(theta).numpy(),raw)
    assert torch.equal(policy.act(theta,features),residual.act(raw,features))
    assert policy.n_params==1020 and policy.raw_theta(theta)[0,-1]==0
    with pytest.raises(ValueError):
        FixedGainResidualPolicy(residual,np.zeros(17))


def test_matched_termination_selection_uses_only_available_nominees():
    metrics=dict(violated=np.array([False,True]),ret=np.array([2.,100.]),ret_pre=np.array([2.,1.]))
    assert np.array_equal(task_returns(metrics),[2.,1.])
    select=NomineeSelector()
    select.add([0],metrics)
    select.add([1],dict(violated=[False],ret=[1.],ret_pre=[1.]))
    select.add([2],dict(violated=[False],ret=[2.],ret_pre=[2.]))
    select.add([3],dict(violated=[False],ret=[2.],ret_pre=[2.]))
    assert select.best(1)[0]==0  # later outcomes cannot change an earlier checkpoint
    assert select.best(2)[0]==2 and select.best(3)[0]==2  # earliest nominee wins ties
    with pytest.raises(ValueError):
        select.best(4)


def test_crossed_variance_recovers_additive_seed_and_episode_components():
    seed=np.array([-1.,1.,-1.,1.])
    episode=np.array([-2.,0.,2.])
    out=pilot_uncertainty(seed[:,None]+episode[None],n_boot=200,seed=3)
    assert out['point']==0 and out['search_seeds']==4 and out['independent_episodes']==3
    assert out['variance_components']['seed']==pytest.approx(np.var(seed,ddof=1))
    assert out['variance_components']['episode']==pytest.approx(np.var(episode,ddof=1))
    assert out['variance_components']['interaction']==0
    again=pilot_uncertainty(seed[:,None]+episode[None],n_boot=200,seed=3)
    assert out==again
    flat=pilot_uncertainty(np.zeros((4,96)),n_boot=20)
    assert all(x['plug_in_power'] is None for x in flat['planning_scenarios'])
