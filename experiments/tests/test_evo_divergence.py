"""Causal history windows, accumulation controls, event censoring and clustered counts."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from helpers.evoDivergence import refreshed_predictions, onset_metrics, event_sufficient_statistics
from helpers.evoDivergence import summarize_cells, validate_cell


class Transition(torch.nn.Module):
    def __init__(self,gain=1.):
        super().__init__()
        self.gain=gain
        self.predictor=SimpleNamespace(num_frames=3)
        self.rows=0

    def action_encoder(self,a):
        return a[...,:1].expand(*a.shape[:-1],192)

    def predict(self,z,a):
        self.rows+=len(z)
        return z+self.gain*a


def setup(gain):
    model=Transition(gain).eval()
    im=SimpleNamespace(model=model,device='cpu',flat=lambda a:torch.as_tensor(a,dtype=torch.float32).reshape(*np.shape(a)[:-2],60))
    zh=np.broadcast_to(np.arange(3,dtype=np.float32)[None,:,None],(2,3,192)).copy()
    real=np.broadcast_to(np.arange(3,13,dtype=np.float32)[None,:,None],(2,10,192)).copy()
    hist=np.ones((2,2,10,6),np.float32)
    actions=np.ones((2,10,60),np.float32)
    return im,zh,real,hist,actions


@pytest.mark.parametrize('period',[1,2,5,10])
def test_exact_dynamics_and_accounting(period):
    im,zh,real,hist,actions=setup(1.)
    out=refreshed_predictions(im,zh,real,hist,actions,period)
    assert np.array_equal(out.numpy(),real)
    assert im.model.rows==20


@pytest.mark.parametrize('period',[1,2,5,10])
def test_known_one_step_bias_accumulates_only_since_last_refresh(period):
    im,zh,real,hist,actions=setup(.9)
    out=refreshed_predictions(im,zh,real,hist,actions,period).numpy()
    expected=.1*(1+np.arange(10)%period)
    np.testing.assert_allclose((real-out)[0,:,0],expected,atol=2e-6)


def test_future_real_frames_do_not_leak_into_earlier_predictions():
    im,zh,real,hist,actions=setup(.9)
    first=refreshed_predictions(im,zh,real,hist,actions,1)
    changed=real.copy()
    changed[:,5:]+=100
    second=refreshed_predictions(im,zh,changed,hist,actions,1)
    assert torch.equal(first[:,:6],second[:,:6])
    assert not torch.equal(first[:,6:],second[:,6:])
    altered=actions.copy()
    altered[:,5:]*=2
    third=refreshed_predictions(im,zh,real,hist,altered,10)
    base=refreshed_predictions(im,zh,real,hist,actions,10)
    assert torch.equal(base[:,:5],third[:,:5])


def cell():
    real=np.zeros((3,10,3))
    real[...,0]=1.2
    real[0,4:,0]=.7
    real[1,6:,0]=.7
    pred=np.broadcast_to(real,(5,*real.shape)).copy()
    # Row 0's error arises at the endpoint after the first dense failure.
    pred[:,0,4:,0]=1.2
    # Row 1's error starts at endpoint 30, before event step 70.
    pred[:,1,2:,0]+=.1
    return dict(real_readout=real,truth=real.copy(),predicted_readout=pred,
        latent_rms=np.zeros((5,3,10)),dense_violated=np.array([1,1,0],bool),
        dense_first_step=np.array([45,69,100]),failure_constraint=np.array([0,0,-1]))


def test_failure_alignment_excludes_event_endpoint_and_post_failure_error():
    c=cell()
    out=onset_metrics(c)
    assert np.array_equal(out['before_failure'][0],[False,True,False])
    assert np.isnan(out['lead_ms'][0,0])
    assert out['lead_ms'][0,1]==320
    assert not out['healthy_prefix_mask'][1,6]
    assert not out['healthy_prefix_mask'][2].any()


def test_event_detection_and_clustered_denominators_do_not_multiply_episodes():
    c=cell()
    s=event_sufficient_statistics(c)
    assert s['event_endpoint'].sum()==2
    assert s['real_readout_event_detected'].sum()==2
    summary=summarize_cells([c,c],episode_keys=['a','b','c'],replicates=100)
    assert summary['independent_source_episodes']==3
    assert summary['candidate_episode_pairs']==6 and summary['failed_pairs']==4
    assert summary['failing_source_episodes']==2
    ratio=summary['modes']['teacher_forced']['divergence_before_dense_failure']
    assert ratio['numerator']==2 and ratio['denominator']==4 and ratio['point']==.5
    assert summary['failure_constraints']['low_height']==4


def test_censored_no_failure_outputs_are_undefined_not_zero_evidence():
    c=cell()
    c['dense_violated'][:]=False
    c['dense_first_step'][:]=100
    c['failure_constraint'][:]=-1
    s=summarize_cells([c],episode_keys=['a','b','c'],replicates=100)
    assert s['failed_pairs']==0
    for m in s['modes'].values():
        assert m['divergence_before_dense_failure']['point'] is None
        assert m['median_lead_ms'] is None


def test_bad_shapes_failure_flags_and_training_mode_rejected():
    c=cell()
    c['dense_first_step'][0]=100
    with pytest.raises(ValueError,match='mismatch'):
        validate_cell(c)
    im,zh,real,hist,actions=setup(1.)
    im.model.train()
    with pytest.raises(ValueError,match='eval mode'):
        refreshed_predictions(im,zh,real,hist,actions,1)
    with pytest.raises(ValueError,match='supported'):
        refreshed_predictions(im,zh,real,hist,actions,3)


def test_full_three_frame_and_three_action_windows_at_refresh_boundaries():
    im,zh,real,hist,actions=setup(.9)
    histories=[]
    windows=[]
    buffers=[]
    original_predict=im.model.predict
    original_encoder=im.model.action_encoder
    def encoder(a):
        buffers.append(a.clone())
        return original_encoder(a)
    def predict(z,a):
        histories.append(z.clone())
        windows.append(a.clone())
        return original_predict(z,a)
    im.model.predict=predict
    im.model.action_encoder=encoder
    hist[:,0]=.3
    hist[:,1]=.4
    actions[:]=np.arange(1,11)[None,:,None]
    out=refreshed_predictions(im,zh,real,hist,actions,2)
    assert all(b.shape==(2,12,60) for b in buffers)
    assert torch.equal(histories[0],torch.as_tensor(zh))
    assert torch.equal(histories[1][:,0],torch.as_tensor(zh[:,1]))
    assert torch.equal(histories[1][:,1],torch.as_tensor(zh[:,2]))
    assert torch.equal(histories[1][:,2],out[:,0])
    assert torch.equal(histories[2],torch.as_tensor(np.concatenate([zh[:,2:],real[:,:2]],1)))
    torch.testing.assert_close(windows[0][0,:,0],torch.tensor([.3,.4,1.]))
    torch.testing.assert_close(windows[1][0,:,0],torch.tensor([.4,1.,2.]))
    torch.testing.assert_close(windows[2][0,:,0],torch.tensor([1.,2.,3.]))
    assert not buffers[0][:,3:].any()
