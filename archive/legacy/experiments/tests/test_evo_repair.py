"""Repair invariants: whole episodes, temporal alignment, fitting isolation, and gates."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch

from helpers.evoRepair import (split_episodes, fitting_masks, balanced_indices, training_arrays,
    fit_predictor, fit_probe_copy, metric_arrays, summarize, select_repair, nominate, tensor_state)


def test_split_is_unique_episode_based_and_order_independent():
    keys=[f'episode:{i}' for i in range(96)]
    a=split_episodes(keys,4)
    b=split_episodes(keys[::-1],4)
    assert all(len(v)==24 for v in a.values())
    assert len(set(np.concatenate(list(a.values()))))==96
    assert all([keys[i] for i in a[r]]==[keys[::-1][i] for i in b[r]] for r in a)
    with pytest.raises(ValueError):
        split_episodes(keys[:-1]+keys[:1],4)


def test_fitting_stops_at_first_event_block_and_includes_two_failure_rich_endpoints():
    first=np.array([[0,9,10,25,99,100]])
    allowed,rich=fitting_masks(first)
    assert allowed.sum(-1).tolist()==[[1,1,2,3,10,10]]
    assert rich.sum(-1).tolist()==[[1,1,2,2,2,0]]
    assert not rich[0,3,0] and rich[0,3,1] and rich[0,3,2]
    assert not allowed[0,3,3:].any()


def test_balanced_sampling_never_leaks_other_rows():
    rich=np.array([True,False,False,True,False])
    ix=balanced_indices(rich,100,np.random.default_rng(1))
    assert rich[ix].sum()==50 and len(ix)==100
    with pytest.raises(ValueError):
        balanced_indices(np.ones(3,bool),4,np.random.default_rng(1))


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=torch.nn.Linear(2,2)
        self.projector=torch.nn.Linear(2,2)
        self.action_encoder=torch.nn.Identity()
        self.predictor=torch.nn.Linear(4,2)
        self.pred_proj=torch.nn.Sequential(torch.nn.BatchNorm1d(2),torch.nn.Linear(2,2))
    def predict(self,z,a):
        v=self.predictor(torch.cat([z,a[...,:2]],-1))
        return self.pred_proj(v.reshape(-1,2)).reshape_as(v)


def sample_data():
    z=np.arange(2*4*10*2,dtype=np.float32).reshape(2,4,10,2)/50
    truth=np.stack([z[...,0]*.1+1,z[...,1]*.1,z[...,0]],-1)
    first=np.array([[25,100,100,99],[100,10,100,100]])
    return dict(z=z,truth=truth,first=first,actions=np.arange(2*4*10*60,dtype=np.float32).reshape(2,4,10,60)),np.arange(4*3*2,dtype=np.float32).reshape(4,3,2),np.ones((4,2,10,6),np.float64)


def test_cached_context_and_actions_use_exact_past_windows_and_ignore_other_roles():
    d,h,a=sample_data()
    model=Toy().eval()
    out=training_arrays(d,h,a,[0],model,(np.zeros(6),np.ones(6)),device='cpu')
    np.testing.assert_array_equal(out['context'][0],h[0])
    np.testing.assert_array_equal(out['context'][1],np.concatenate([h[0,1:],d['z'][0,0,:1]]))
    np.testing.assert_array_equal(out['action'][0,0],a[0,0].reshape(-1))
    np.testing.assert_array_equal(out['action'][0,2],d['actions'][0,0,0])
    np.testing.assert_array_equal(out['action'][1,2],d['actions'][0,0,1])
    d['z'][:,1:]+=900
    d['truth'][:,1:]+=900
    again=training_arrays(d,h,a,[0],model,(np.zeros(6),np.ones(6)),device='cpu')
    assert all(np.array_equal(out[k],again[k]) for k in out)


def test_predictor_update_preserves_frozen_parameters_and_all_running_statistics():
    torch.manual_seed(5)
    model=Toy().eval()
    old=tensor_state(model)
    probe=torch.nn.Linear(2,3).eval()
    probe_old=tensor_state(probe)
    arrays=dict(context=np.ones((8,3,2),np.float32),action=np.ones((8,3,60),np.float32),
        target=np.zeros((8,2),np.float32),physical=np.zeros((8,3),np.float32),rich=np.arange(8)%2==0)
    counter=dict(predictor_gradient_updates=0)
    fit_predictor(model,probe,arrays,seed=5,steps=3,batch=4,lr=.001,physical_weight=.1,counter=counter,deadline=lambda:None)
    new=tensor_state(model)
    assert counter['predictor_gradient_updates']==3
    assert any(not torch.equal(old[k],new[k]) for k in old if k.startswith('predictor.'))
    assert all(torch.equal(old[k],new[k]) for k in old if k.startswith(('encoder.','projector.')) or 'running_' in k or 'num_batches' in k)
    assert all(torch.equal(probe_old[k],v) for k,v in probe.state_dict().items())


def test_probe_fit_uses_own_episodes_and_changes_parameters():
    torch.manual_seed(6)
    probe=torch.nn.Linear(2,3).eval()
    original=tensor_state(probe)
    d,_,_=sample_data()
    counter=dict(probe_gradient_updates=0,probe_training_rows=0)
    fit_probe_copy(probe,d,[0],seed=6,steps=3,batch=4,lr=.001,counter=counter,deadline=lambda:None)
    assert counter==dict(probe_gradient_updates=3,probe_training_rows=12)
    assert any(not torch.equal(original[k],v) for k,v in probe.state_dict().items())


def test_event_alignment_counts_transients_and_matched_healthy_endpoints_correctly():
    truth=np.zeros((1,3,10,3))
    truth[...,0]=1.2
    truth[0,0,2,0]=.7
    first=np.array([[25,25,100]])
    pred=truth[None].copy()
    pred[0,0,0,0,0]=.7
    a=metric_arrays(pred,truth,first)
    assert a['endpoint_pairs']==1 and a['failed_pairs']==2
    assert a['event_num'].tolist()==[1,0,0]
    assert a['false_num'].tolist()==[1,0,0]
    assert a['false_den'].tolist()==[2,2,10]
    s=summarize(pred,truth,first,replicates=100)
    assert s['event_detection']['point']==1
    assert s['false_alarm']['point']==1/14


def test_seed_duplicates_do_not_shrink_episode_intervals():
    truth=np.zeros((2,4,10,3))
    truth[...,0]=1.2
    truth[:,:,3,0]=.7
    first=np.full((2,4),35)
    pred=truth[None].copy()
    pred[:,:,0,3,0]=1.2
    a=summarize(pred,truth,first,replicates=100)
    b=summarize(np.repeat(pred,3,axis=0),truth,first,replicates=100)
    assert [a['event_detection'][k] for k in ['point','lo','hi']]==[b['event_detection'][k] for k in ['point','lo','hi']]


def test_no_failure_denominator_is_undefined():
    truth=np.zeros((1,3,10,3))
    truth[...,0]=1.2
    s=summarize(truth[None],truth,np.full((1,3),100),replicates=100)
    assert s['event_detection']['point'] is None


def test_gate_requires_detection_false_alarm_and_all_retention_conditions():
    def metric(detection=.1,false=.01,mae=.1):
        return {k:dict(point=v) for k,v in dict(event_detection=detection,false_alarm=false,
            healthy_height_mae=mae,healthy_pitch_mae=mae,healthy_speed_mae=mae).items()}
    gate=dict(minimum_detection_gain=.1,maximum_false_alarm_increase=.02,maximum_healthy_mae_ratio=1.25)
    summaries=dict(baseline=metric(),probe=metric(.5,.5),dynamics=metric(.4,.02,.2),combined=metric(.3,.02))
    assert select_repair(summaries,gate)['chosen']=='combined'
    summaries['combined']=metric(.15)
    assert select_repair(summaries,gate)['chosen'] is None


def test_controller_nomination_uses_imagined_risk_and_progress_with_fallback():
    ro=np.zeros((3,3,4,10,3))
    ro[...,0]=1.2
    ro[...,2]=1
    ro[:,0,:2,5,0]=.7
    ro[:,1,:1,5,0]=.7
    ro[:,2,...,2]=.1
    assert nominate(ro)['index']==1
    ro[:,1,...,2]=.1
    assert nominate(ro)['index']==0


def test_real_cuda_predictor_fits_with_frozen_encoder_and_batchnorm():
    from helpers.walkerLewm import load_walker_model
    root=Path(__file__).resolve().parents[2]
    path=root/'data/hf/safetydial-walker2d/lewm-a/walker2d-lewm-a-recovery-20260927-1'
    if not torch.cuda.is_available() or not (path/'weights.pt').is_file():
        pytest.skip('local Walker checkpoint and CUDA required')
    model,_=load_walker_model(path,'cuda')
    model.requires_grad_(False)
    torch.manual_seed(16)
    with torch.no_grad():
        action=model.action_encoder(torch.zeros((8,12,60),device='cuda'))[:,:3].cpu().numpy()
    arrays=dict(context=np.zeros((8,3,192),np.float32),action=action,
        target=np.ones((8,192),np.float32)*.01,physical=np.zeros((8,3),np.float32),rich=np.arange(8)%2==0)
    probe=torch.nn.Linear(192,3).cuda().eval()
    before=tensor_state(model.pred_proj)
    fit_predictor(model,probe,arrays,seed=16,steps=1,batch=8,lr=.00001,physical_weight=.1,
        counter=dict(predictor_gradient_updates=0),deadline=lambda:None)
    assert any(not torch.equal(before[k],v.cpu()) for k,v in model.pred_proj.state_dict().items())
