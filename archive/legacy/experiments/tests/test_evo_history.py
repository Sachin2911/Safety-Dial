"""History is past-only and identical in real, imagined and trainable policy paths."""
import sys
from pathlib import Path
import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments'))
from helpers.evoHistoryData import cached_previous_actions  # noqa: E402
from helpers.evoHistoryPolicy import HistoryBlockPolicy  # noqa: E402
from helpers.evoHistoryReal import HistoryRealExecutor  # noqa: E402
from helpers.evoHistoryImagine import HistoryImaginer  # noqa: E402
from helpers.walkerRules import execute_branch  # noqa: E402
import test_evo_real as fixtures  # noqa: E402

roots,ctx,tiny=fixtures.roots,fixtures.ctx,fixtures.tiny


@pytest.fixture(scope='module',autouse=True)
def one_thread():
    before=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_cached_history_alignment_excludes_future_actions(tmp_path):
    path=tmp_path/'actions.h5'
    actions=np.arange(30*6,dtype=np.float32).reshape(30,6)/180
    with h5py.File(path,'w') as f:
        f['ep_offset']=[0]
        f['ep_len']=[30]
        f['action']=actions
    targets=np.stack([actions[t:t+10].reshape(60) for t in [10,15,20]])
    past=cached_previous_actions(path,[0,0,0],targets)
    assert np.array_equal(past,np.stack([actions[t-10:t].reshape(60) for t in [10,15,20]]))
    with pytest.raises(ValueError,match='temporal alignment'):
        cached_previous_actions(path,[0,0,0],targets[::-1])


def test_policy_state_features_and_trainable_network():
    policy=HistoryBlockPolicy(fixtures.SCALER)
    history=torch.randn(2,60)
    z=torch.randn(2,192)
    feats=policy.features(z,z*.5,history)
    net=policy.network()
    theta=policy.network_theta(net)
    assert policy.n_params==154556
    with torch.inference_mode():
        np.testing.assert_allclose(policy.act(theta[None],feats[None])[0],
                                   net(feats).clamp(-1,1),atol=1e-6)
    restored=HistoryBlockPolicy.from_state(policy.state())
    assert torch.equal(restored.features(z,z*.5,history),feats)
    zero=HistoryBlockPolicy(fixtures.SCALER,use_history=False)
    assert torch.count_nonzero(zero.features(z,z,history)[:,-60:])==0
    assert not torch.equal(feats,policy.features(z,z*.5,history+1))


def test_history_real_feedback_preserves_physics_and_grouping(tiny,ctx,roots):
    model,probe,_=tiny
    policy=HistoryBlockPolicy(fixtures.SCALER)
    torch.manual_seed(23)
    theta=policy.network_theta(policy.network())[None]
    ex=HistoryRealExecutor(model,fixtures.SCALER,probe,policy,device='cpu',ctx=ctx,max_envs=2,encode_batch=4)
    try:
        history=ex.history_latents(roots).numpy()
        out=ex.run(theta,roots,record_qpos=True,z_hist=history)
        ex.max_envs=1
        again=ex.run(theta,roots,record_qpos=True,z_hist=history)
        assert np.array_equal(out['qpos'],again['qpos'])
        for i,r in enumerate(roots):
            replay=execute_branch(ex._physics(1)[0],r.qpos,r.qvel,out['actions'][i].reshape(100,6))
            assert np.array_equal(replay.qpos,out['qpos'][i])
            for b in range(10):
                zt=history[i,-1] if b==0 else out['z'][i,b-1]
                zp=history[i,-2] if b==0 else history[i,-1] if b==1 else out['z'][i,b-2]
                past=r.history_actions[-1].reshape(60) if b==0 else out['actions'][i,b-1]
                feats=policy.features(torch.tensor(zt[None]),torch.tensor(zp[None]),past[None])
                expected=policy.act(theta,feats[:,None])[0,0].numpy()
                np.testing.assert_allclose(expected,out['actions'][i,b],atol=1e-6)
    finally:
        ex.close()


def test_history_imagination_uses_its_own_last_action(tiny):
    model,probe,_=tiny
    policy=HistoryBlockPolicy(fixtures.SCALER)
    theta=policy.network_theta(policy.network())[None]
    im=HistoryImaginer(model,fixtures.SCALER,probe,device='cpu')
    zh=torch.randn(2,3,192)
    hist=torch.randn(2,2,10,6).clamp(-1,1)
    out=im.rollout_policies(policy,theta,zh,hist,return_latents=True)
    again=im.rollout_policies(policy,theta,zh,hist,return_latents=True)
    assert np.array_equal(out['z'],again['z'])
    for b in range(10):
        zt=zh[:,-1] if b==0 else torch.tensor(out['z'][0,:,0,b-1])
        zp=zh[:,-2] if b==0 else zh[:,-1] if b==1 else torch.tensor(out['z'][0,:,0,b-2])
        past=hist[:,-1].reshape(2,60) if b==0 else torch.tensor(out['actions'][0,:,0,b-1])
        expected=policy.act(theta,policy.features(zt,zp,past)[None]).numpy()[0]
        np.testing.assert_allclose(expected,out['actions'][0,:,0,b],atol=1e-6)
    im.sigma=np.ones(192)*.01
    noisy=im.rollout_policies(policy,np.repeat(theta,2,axis=0),zh,hist,k=.5,n_samples=2)
    assert noisy['actions'].shape==(2,2,2,10,60)
