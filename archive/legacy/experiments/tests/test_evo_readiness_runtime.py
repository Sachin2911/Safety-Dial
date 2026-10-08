"""Frozen residual diagnostics preserve zero and report clipping in actual action units."""
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from helpers.evoReadinessRuntime import initial_residual_diagnostics
from helpers.evoReadinessSearch import FixedGainResidualPolicy
from test_evo_transfer_pilot import policy_fixture


def test_initial_action_diagnostics_measure_zero_and_saturated_bias():
    policy=FixedGainResidualPolicy(policy_fixture(),np.ones(17))
    features=torch.zeros((64,444))
    theta=np.zeros((2,1020))
    theta[1].reshape(60,17)[:,-1]=3
    calls=[]
    before=theta.copy()
    result=initial_residual_diagnostics(policy,features,theta,on_policy_call=lambda:calls.append(1))
    assert len(calls)==3 and np.array_equal(theta,before)
    assert np.array_equal(result['raw_residual_rms'],[0,3])
    assert result['clipped_action_delta_rms'][0]==0
    assert 0<result['clipped_action_delta_rms'][1]<=2
    assert result['clipped_action_fraction'][1]==1
