"""State reference is explicit privileged information and validation selection is fixed."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments'))
from helpers.evoInformation import observations,StateReferencePolicy,evaluate_state,candidate_key,fit_candidates  # noqa: E402
from helpers.walkerRules import execute_branch  # noqa: E402
import test_evo_real as fixtures  # noqa: E402
roots=fixtures.roots


@pytest.fixture(scope='module',autouse=True)
def one_thread():
    old=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_state_observation_matches_environment_and_exact_action_replay(roots):
    env=fixtures.physics_env()
    try:
        np.testing.assert_array_equal(observations(env.unwrapped.data.qpos,env.unwrapped.data.qvel),
                                      env.unwrapped._get_obs().astype(np.float32))
        p=StateReferencePolicy(fixtures.SCALER,hidden=(8,),state_mean=np.zeros(34),state_std=np.ones(34))
        theta=p.network_theta(p.network())
        out=evaluate_state(p,theta,roots)
        for i,r in enumerate(roots):
            ref=execute_branch(env,r.qpos,r.qvel,out['actions'][i].reshape(100,6))
            assert np.array_equal(out['qpos'][i],ref.qpos)
        restored=StateReferencePolicy.from_state(p.state())
        assert np.array_equal(restored.state()['state_mean'],p.state()['state_mean'])
    finally:
        env.close()


def test_validation_ranking_requires_progress_then_safety():
    rows=[dict(violations=0,progress_mean=.1,val_mse=.01,seed=1,epoch=20),
          dict(violations=2,progress_mean=1.,val_mse=.2,seed=1,epoch=20),
          dict(violations=1,progress_mean=.8,val_mse=.3,seed=1,epoch=20)]
    assert min(rows,key=lambda r:candidate_key(r,.5)) is rows[2]


def test_candidate_pool_only_declared_epochs_and_offline_best():
    p=StateReferencePolicy(fixtures.SCALER,hidden=(8,),state_mean=np.zeros(34),state_std=np.ones(34))
    x,y=torch.randn(12,444),torch.randn(12,60)
    candidates,hist,updates=fit_candidates(p,x,y,x,y,np.ones(12),seed=4,epochs=4,
                                         checkpoint_epochs=[2,4],batch_size=5)
    best=min(hist,key=lambda r:r['val_mse'])
    assert {r['epoch'] for r in candidates}=={2,4,best['epoch']}
    assert updates==12
