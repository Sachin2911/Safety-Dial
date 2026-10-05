"""Retrospective dynamics/readout decomposition on immutable real controller traces."""
import json
from pathlib import Path

import numpy as np
import torch

from helpers.evoImagine import _check_model
from helpers.evoReadinessQueries import QueryStore, digest_array
from helpers.walkerRules import health_clearance

MODES = ('teacher_forced', 'refresh_2', 'refresh_5', 'real_action_tape', 'closed_loop')
RESETS = (1, 2, 5, 10)


def verified_archive(path, identity):
    path = Path(path)
    with np.load(path, allow_pickle=False) as f:
        receipt = json.loads(str(f['__receipt__']))
        arrays = {k:f[k].copy() for k in f.files if k!='__receipt__'}
    QueryStore(path.parent,study_identity=identity)._validate_receipt(receipt,path.stem)
    if receipt['data_sha256'] != {k:digest_array(v) for k,v in arrays.items()}:
        raise ValueError('parent numeric archive changed')
    return arrays,receipt


@torch.inference_mode()
def refreshed_predictions(imaginer, z_history, real_z, history_actions, real_actions, reset_every):
    """Use real executed actions; refresh the THREE-frame context at declared starts.

    At block b the most recent observed real frame is b, the target is b+1.
    Full-length action buffers preserve the established predictor kernel layout.
    Real future frames are used only when their refresh time has arrived.
    """
    if reset_every not in RESETS:
        raise ValueError('supported refresh periods are 1, 2, 5, 10 blocks')
    _check_model(imaginer.model)
    zh = torch.as_tensor(z_history,dtype=torch.float32,device=imaginer.device)
    real = torch.as_tensor(real_z,dtype=torch.float32,device=imaginer.device)
    actions = torch.as_tensor(real_actions,dtype=torch.float32,device=imaginer.device)
    if zh.ndim!=3 or zh.shape[1:]!=(3,192) or real.shape!=(len(zh),10,192):
        raise ValueError('three real history frames and ten real target frames required')
    if actions.shape!=(len(zh),10,60):
        raise ValueError('ten sequential raw action blocks required')
    if not torch.isfinite(real).all() or not torch.isfinite(zh).all() or not torch.isfinite(actions).all():
        raise ValueError('finite latent histories and actions required')
    full_real = torch.cat([zh,real],1)
    hist = imaginer.flat(history_actions)
    future = imaginer.flat(actions.cpu().numpy().reshape(len(zh),10,10,6))
    buffer = torch.zeros((len(zh),12,60),device=imaginer.device)
    buffer[:,:2] = hist
    predictions = []
    context = None
    for b in range(10):
        if b%reset_every==0:
            context = list(full_real[:,b:b+3].unbind(1))
        buffer[:,b+2] = future[:,b]
        encoded = imaginer.model.action_encoder(buffer)
        pred = imaginer.model.predict(torch.stack(context[-3:],1),encoded[:,b:b+3])[:,-1]
        context.append(pred)
        predictions.append(pred)
    return torch.stack(predictions,1)


def validate_cell(out):
    n = len(out['dense_first_step'])
    if out['predicted_readout'].shape!=(5,n,10,3) or out['real_readout'].shape!=(n,10,3):
        raise ValueError('wrong mode or readout shapes')
    if out['truth'].shape!=(n,10,3) or out['latent_rms'].shape!=(5,n,10):
        raise ValueError('wrong truth or latent-error shape')
    if not np.all(np.isfinite(out['predicted_readout'])):
        raise ValueError('nonfinite predictions')
    first = out['dense_first_step']
    if not np.issubdtype(first.dtype,np.integer) or ((first<0)|(first>100)).any():
        raise ValueError('dense failure step must be 0..99 or censored at 100')
    if not np.array_equal(first<100,out['dense_violated']):
        raise ValueError('failure time/flag mismatch')


def onset_metrics(out, height_threshold=.05, pitch_threshold=.1):
    """Error against REAL readouts: avoid counting probe error as dynamics error.

    A strictly preceding endpoint must be earlier than the first dense event.
    Errors arising only at/after the event are never called advance divergence.
    """
    validate_cell(out)
    err = np.abs(out['predicted_readout'][...,:2]-out['real_readout'][None,...,:2])
    exceeded = (err[...,0]>height_threshold)|(err[...,1]>pitch_threshold)
    first_error = np.where(exceeded.any(-1),exceeded.argmax(-1),10)
    endpoint_steps = (np.arange(10)+1)*10
    failed = out['dense_violated']
    event_step = out['dense_first_step']+1
    valid = endpoint_steps[None,:] < event_step[:,None]
    valid &= failed[:,None]
    before = (exceeded & valid[None]).any(-1)
    leads_ms = np.where(before,(event_step[None]-(first_error+1)*10)*8,np.nan)
    return dict(first_error_block=first_error,before_failure=before,lead_ms=leads_ms,
                exceeded=exceeded,healthy_prefix_mask=valid)


def event_sufficient_statistics(out):
    """Candidate-local episode statistics; sum by episode before bootstrapping."""
    validate_cell(out)
    pred = out['predicted_readout']
    actual = out['truth']
    ro = out['real_readout']
    unsafe = health_clearance(pred[...,0],pred[...,1])<=0
    real_unsafe = health_clearance(ro[...,0],ro[...,1])<=0
    endpoint_unsafe = health_clearance(actual[...,0],actual[...,1])<=0
    failed = out['dense_violated']
    first_block = np.minimum(out['dense_first_step']//10,9)
    n = len(failed)
    idx = np.arange(n)
    event_endpoint = endpoint_unsafe[idx,first_block] & failed
    through = np.arange(10)[None,:]<=first_block[:,None]
    alarm = (unsafe & through[None]).any(-1)
    onset = onset_metrics(out)
    at_event = pred[:,idx,first_block,:]-actual[idx,first_block][None]
    ro_error = ro[idx,first_block]-actual[idx,first_block]
    return dict(failed=failed,event_endpoint=event_endpoint,
        real_readout_event_detected=real_unsafe[idx,first_block]&event_endpoint,
        mode_event_detected=unsafe[:,idx,first_block]&event_endpoint[None],
        mode_alarm_through_event=alarm&failed[None],
        mode_ever_alarm=unsafe.any(-1),endpoint_ever=endpoint_unsafe.any(-1),
        readout_ever=real_unsafe.any(-1),before_failure=onset['before_failure'],
        lead_ms=onset['lead_ms'],at_event_error=at_event,readout_at_event_error=ro_error,
        failure_constraint=out['failure_constraint'])


def summarize_cells(cells, *, episode_keys, seed=20261503, replicates=2000):
    """Uncertainty resamples source episodes, conditional on the whole fixed pool.

    Failed candidate/episode pairs are correlated and not independent trials.
    Episode-ratio intervals keep zero-event episodes in the resampling frame.
    """
    if len(set(episode_keys))!=len(episode_keys) or not cells:
        raise ValueError('nonempty cells and unique episode keys required')
    n = len(episode_keys)
    if any(len(c['dense_first_step'])!=n for c in cells):
        raise ValueError('episode count mismatch')
    stats = [event_sufficient_statistics(c) for c in cells]
    fail = np.stack([s['failed'] for s in stats])
    endpoint = np.stack([s['event_endpoint'] for s in stats])
    draws = np.random.default_rng(seed).integers(n,size=(replicates,n))
    def ratio(num,den):
        num,den = np.asarray(num,float).sum(0),np.asarray(den,float).sum(0)
        if not den.sum():
            return dict(point=None,lo=None,hi=None,numerator=0,denominator=0,defined_draws=0)
        d = den[draws].sum(1)
        values = num[draws].sum(1)[d>0]/d[d>0]
        return dict(point=float(num.sum()/den.sum()),lo=float(np.quantile(values,.025)),
                    hi=float(np.quantile(values,.975)),numerator=float(num.sum()),
                    denominator=float(den.sum()),defined_draws=int(len(values)))
    modes = {}
    for m,name in enumerate(MODES):
        before = np.stack([s['before_failure'][m] for s in stats])
        error = np.stack([s['at_event_error'][m] for s in stats])
        leads = np.concatenate([s['lead_ms'][m][s['before_failure'][m]] for s in stats])
        trajectories = np.stack([c['predicted_readout'][m]-c['truth'] for c in cells])
        dynamics = np.stack([c['predicted_readout'][m]-c['real_readout'] for c in cells])
        prefix = (np.arange(10)[None,None,:]+1)*10 < np.stack([c['dense_first_step']+1 for c in cells])[...,None]
        # All still-healthy endpoints, including censored healthy trajectories.
        prefix &= np.stack([health_clearance(c['truth'][...,0],c['truth'][...,1])>0 for c in cells])
        def curve(arr,mask):
            count = mask.sum((0,1))
            numerator = np.where(mask[...,None],arr,0).sum((0,1))
            return [None if count[b]==0 else (numerator[b]/count[b]).tolist() for b in range(10)]
        modes[name] = dict(
            segment_alarm_rate=ratio(np.stack([s['mode_ever_alarm'][m] for s in stats]),np.ones_like(fail)),
            event_endpoint_detection=ratio(np.stack([s['mode_event_detected'][m] for s in stats]),endpoint),
            alarm_through_failure_block=ratio(np.stack([s['mode_alarm_through_event'][m] for s in stats]),fail),
            divergence_before_dense_failure=ratio(before,fail),
            median_lead_ms=None if not len(leads) else float(np.median(leads)),
            event_height_bias=ratio(error[...,0]*fail,fail),
            event_pitch_bias=ratio(error[...,1]*fail,fail),
            healthy_prefix_physical_mae=curve(np.abs(trajectories),prefix),
            healthy_prefix_dynamics_mae=curve(np.abs(dynamics),prefix),
            healthy_prefix_counts=prefix.sum((0,1)).tolist(),
            all_latent_rms_by_block=np.mean([c['latent_rms'][m] for c in cells],axis=(0,1)).tolist())
    sensitivity = {}
    for h,p in [(.025,.05),(.05,.1),(.1,.2)]:
        masks = [onset_metrics(c,h,p)['before_failure'] for c in cells]
        sensitivity[f'height_{h:g}_pitch_{p:g}'] = {
            name:ratio(np.stack([v[m] for v in masks]),fail) for m,name in enumerate(MODES)}
    return dict(candidates=len(cells),independent_source_episodes=n,candidate_episode_pairs=len(cells)*n,
        failed_pairs=int(fail.sum()),failing_source_episodes=int(fail.any(0).sum()),
        baseline_failures=int(fail[0].sum()),
        event_endpoint_visible=int(endpoint.sum()),
        dense_event_not_visible_at_containing_block_end=int((fail&~endpoint).sum()),
        real_readout_event_detection=ratio(np.stack([s['real_readout_event_detected'] for s in stats]),endpoint),
        real_readout_segment_alarm_rate=ratio(np.stack([s['readout_ever'] for s in stats]),np.ones_like(fail)),
        failure_constraints={name:int(sum(np.sum(s['failure_constraint'][s['failed']]==i) for s in stats))
            for i,name in enumerate(['low_height','high_height','negative_pitch','positive_pitch'])},
        modes=modes,onset_threshold_sensitivity=sensitivity,
        uncertainty='Descriptive 95% episode-cluster bootstrap, conditional on the fixed correlated '
        'candidate pool; reused development episodes. Resets access real history only at declared '
        'times and are privileged diagnostics, not root-time policy evaluations.')
