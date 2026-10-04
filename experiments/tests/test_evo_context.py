"""Full context respects episode boundaries and uses only its own prior decisions."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments'))
from helpers.evoContextPolicy import ContextBlockPolicy,context_training_features  # noqa: E402
from helpers.evoContextImagine import ContextImaginer  # noqa: E402
from helpers.evoContextReal import ContextRealExecutor  # noqa: E402
from helpers.walkerRules import execute_branch  # noqa: E402
import test_evo_real as fixtures  # noqa: E402
roots,ctx,tiny=fixtures.roots,fixtures.ctx,fixtures.tiny


@pytest.fixture(scope='module',autouse=True)
def one_thread():
    old=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_context_cache_excludes_episode_crossings_and_masks_control():
    x=np.arange(8*384,dtype=np.float32).reshape(8,384)
    a=np.arange(8*60,dtype=np.float32).reshape(8,60)
    e=np.array([0]*4+[1]*4)
    full,valid=context_training_features(x,a,e,use_context=True)
    short,other=context_training_features(x,a,e,use_context=False)
    assert np.array_equal(valid,[False,False,True,True,False,False,True,True])
    assert np.array_equal(valid,other)
    assert np.array_equal(full[2,384:576],x[0,192:])
    assert np.array_equal(full[2,576:636],a[0])
    assert np.array_equal(full[:,636:],a)
    assert np.count_nonzero(short[:,384:636])==0


def test_full_context_real_feedback_and_fixed_tape_replay(tiny,ctx,roots):
    model,probe,_=tiny
    p=ContextBlockPolicy(fixtures.SCALER,hidden=(16,8))
    theta=p.network_theta(p.network())[None]
    ex=ContextRealExecutor(model,fixtures.SCALER,probe,p,device='cpu',ctx=ctx,max_envs=2,encode_batch=4)
    try:
        hist=ex.history_latents(roots).numpy()
        out=ex.run(theta,roots,z_hist=hist,record_qpos=True)
        ex.max_envs=1
        again=ex.run(theta,roots,z_hist=hist,record_qpos=True)
        assert np.array_equal(out['qpos'],again['qpos'])
        for i,r in enumerate(roots):
            tape=out['actions'][i].reshape(100,6)
            ref=execute_branch(ex._physics(1)[0],r.qpos,r.qvel,tape)
            assert np.array_equal(ref.qpos,out['qpos'][i])
            z=np.concatenate([hist[i],out['z'][i]])
            past=np.concatenate([r.history_actions.reshape(2,60),out['actions'][i]])
            for b in range(10):
                feat=p.features(*[torch.tensor(z[b+j:b+j+1]) for j in [2,1,0]],
                                past[b:b+2].reshape(1,120))
                expected=p.act(theta,feat[:,None])[0,0].numpy()
                np.testing.assert_allclose(expected,out['actions'][i,b],atol=1e-6)
    finally:
        ex.close()


def test_full_context_imagination_and_saved_policy(tiny):
    model,probe,_=tiny
    p=ContextBlockPolicy(fixtures.SCALER,hidden=(16,8))
    theta=p.network_theta(p.network())[None]
    restored=ContextBlockPolicy.from_state(p.state())
    zh=torch.randn(2,3,192)
    past=torch.randn(2,2,10,6).clamp(-1,1)
    im=ContextImaginer(model,fixtures.SCALER,probe,device='cpu')
    out=im.rollout_policies(restored,theta,zh,past,return_latents=True)
    z=np.concatenate([zh.numpy(),out['z'][0,:,0]],axis=1)
    actions=np.concatenate([past.numpy().reshape(2,2,60),out['actions'][0,:,0]],axis=1)
    for b in range(10):
        feat=p.features(*[torch.tensor(z[:,b+j]) for j in [2,1,0]],actions[:,b:b+2].reshape(2,120))
        expected=p.act(theta,feat[None])[0].numpy()
        np.testing.assert_allclose(expected,out['actions'][0,:,0,b],atol=1e-6)
    im.sigma=np.ones(192)*.01
    noisy=im.rollout_policies(p,np.repeat(theta,2,axis=0),zh,past,k=.5,n_samples=2)
    assert noisy['actions'].shape==(2,2,2,10,60)
