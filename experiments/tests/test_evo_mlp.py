"""MLP training/deployment parity, checkpoint selection and exact real execution."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))
from helpers.evoInputs import load_stage_config  # noqa: E402
from helpers.evoImagine import ClosedLoopImaginer  # noqa: E402
from helpers.evoMlpFit import fit_seed, select_fit  # noqa: E402
from helpers.evoMlpPolicy import MLPBlockPolicy  # noqa: E402
from helpers.evoMlpReal import MLPRealExecutor  # noqa: E402
from helpers.evoRoots import load_roots_json  # noqa: E402
from helpers.locoData import RenderContext  # noqa: E402
from helpers.poseProbes import load_probe  # noqa: E402
from helpers.walkerLewm import load_walker_model  # noqa: E402
from helpers.walkerRules import execute_branch  # noqa: E402
from test_evo_real import MODEL_DIR, PROBE, S4_DEV, SCALER, physics_env  # noqa: E402


@pytest.fixture(scope='module',autouse=True)
def one_thread():
    before=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_trainable_network_matches_stateless_deployment_and_save():
    torch.manual_seed(12)
    p=MLPBlockPolicy(SCALER,hidden=(16,8))
    net=p.network()
    x=torch.randn(17,384)
    theta=p.network_theta(net)
    actual=p.act(theta[None],x[None])[0]
    torch.testing.assert_close(actual,net(x).clamp(-1,1),atol=1e-6,rtol=1e-5)
    assert np.array_equal(p.pack(p.unpack(theta)),theta)
    tt=torch.tensor(theta)
    assert torch.equal(p.pack(p.unpack(tt)),tt)
    restored=MLPBlockPolicy.from_state(p.state())
    assert restored.n_params==p.n_params
    assert torch.equal(restored.act(theta[None],x[None]),actual[None])
    with pytest.raises(ValueError):
        MLPBlockPolicy.from_state({**p.state(),'policy_kind':'linear_sequential_block_v1'})
    with pytest.raises(ValueError):
        p.act(np.zeros((1,23100)),x[None])
    batched=p.act(np.stack([theta,theta]),x[None].expand(2,-1,-1))
    torch.testing.assert_close(batched,actual[None].expand(2,-1,-1),atol=1e-6,rtol=1e-5)


def test_fixed_budget_training_selects_saved_validation_checkpoint_without_gradients():
    torch.manual_seed(11)
    p=MLPBlockPolicy(SCALER,hidden=(8,))
    x=torch.randn(96,384)
    y=(x[:,:1]*0.2).expand(-1,60).clone()
    yval=y[64:].clone().requires_grad_()
    args=dict(seed=7,epochs=12,batch_size=32,lr=.01,weight_decay=0.)
    best,hist,n=fit_seed(p,x[:64],y[:64],x[64:],yval,**args)
    assert n==24 and len(hist)==12 and yval.grad is None
    assert best['val_mse']==min(row['val_mse'] for row in hist)
    assert hist[-1]['fit_raw_mse']<hist[0]['fit_raw_mse']
    pred=p.act(best['theta'][None],x[64:][None])[0]
    mse=float((pred.double()-yval.detach().double()).square().mean())
    assert mse==pytest.approx(best['val_mse'],abs=1e-7)
    repeat,_,_=fit_seed(p,x[:64],y[:64],x[64:],yval,**args)
    assert np.array_equal(repeat['theta'],best['theta'])
    chosen=select_fit([dict(val_mse=.1,seed=2,epoch=1),dict(val_mse=.1,seed=1,epoch=3),
                       dict(val_mse=.1,seed=1,epoch=2)])
    assert chosen['seed']==1 and chosen['epoch']==2


def test_development_protocol_does_not_select_on_gate_roots():
    cfg=load_stage_config('stage2_mlp')
    assert cfg.real.roots=='S4_development_only' and cfg.real.expected_roots==24
    assert cfg.interpretation.gate0=='not_run'
    assert cfg.interpretation.model_selection_uses_real_outcomes is False
    assert cfg.training.refit_on_all is False
    assert cfg.training.linear_baseline_fit=='same_fit_episodes_only'
    assert len(cfg.cache.archive_revision)==40


def test_cuda_mlp_sequential_physics_feedback_and_group_invariance():
    if not torch.cuda.is_available() or not (MODEL_DIR/'weights.pt').exists():
        pytest.skip('needs CUDA and pinned assets')
    model,scaler=load_walker_model(MODEL_DIR,'cuda')
    probe,_=load_probe(PROBE,'cuda')
    p=MLPBlockPolicy(scaler,device='cuda')
    torch.manual_seed(5)
    theta=np.stack([p.network_theta(p.network()),p.network_theta(p.network())])
    roots=load_roots_json(S4_DEV/'roots.json')[:2]
    ctx=RenderContext()
    ex=MLPRealExecutor(model,scaler,probe,p,device='cuda',ctx=ctx,max_envs=4,encode_batch=64)
    one=MLPRealExecutor(model,scaler,probe,p,device='cuda',ctx=ctx,max_envs=1,encode_batch=64)
    env=physics_env()
    try:
        out=ex.run(theta,roots,record_qpos=True)
        again=one.run(theta,roots,record_qpos=True)
        for key in out:
            assert np.array_equal(out[key],again[key]),key
        for i,(cand,r) in enumerate(out['pairs']):
            tape=out['actions'][i].reshape(100,6)
            log=execute_branch(env,roots[r].qpos,roots[r].qvel,tape)
            for key in ['qpos','qvel','x_velocity']:
                assert np.array_equal(out[key][i],getattr(log,key)),key
            zh=ex.history_latents([roots[r]])
            zt=zh[:,-1]
            zp=zh[:,-2]
            for b in range(10):
                a=p.act(theta[[cand]],p.features(zt,zp)[:,None])[0,0].cpu().numpy()
                assert np.array_equal(a,out['actions'][i,b])
                zp,zt=zt,torch.tensor(out['z'][i,b:b+1],device='cuda')
        assert ex.counters.real_steps==400
        im=ClosedLoopImaginer(model,scaler,probe,device='cuda')
        zh=ex.history_latents(roots)
        hist=np.stack([r.history_actions for r in roots])
        imagined=im.rollout_policies(p,theta,zh,hist,return_latents=True)
        zt=zh[None,:,-1].expand(2,-1,-1)
        zp=zh[None,:,-2].expand(2,-1,-1)
        for b in range(10):
            expected=p.act(theta,p.features(zt,zp)).cpu().numpy()
            assert np.array_equal(expected,imagined['actions'][:,:,0,b])
            zp,zt=zt,torch.tensor(imagined['z'][:,:,0,b],device='cuda')
    finally:
        ex.close()
        one.close()
        ctx.close()
        env.close()
