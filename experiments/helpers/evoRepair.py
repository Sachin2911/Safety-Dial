"""Frozen-encoder repair controls and episode-separated retrospective evaluation."""
import hashlib

import numpy as np
import torch

from helpers.walkerRules import health_clearance

ARMS = ('baseline', 'probe', 'dynamics', 'combined')
ROLES = ('probe_fit', 'dynamics_fit', 'development', 'audit')


def split_episodes(keys, seed, size=24):
    if len(keys)!=4*size or len(set(keys))!=len(keys):
        raise ValueError('four equal groups of unique whole source episodes required')
    order = sorted(range(len(keys)), key=lambda i: hashlib.sha256(f'{seed}:{keys[i]}'.encode()).hexdigest())
    return {name:np.asarray(order[j*size:(j+1)*size],int) for j,name in enumerate(ROLES)}


def fitting_masks(first):
    first = np.asarray(first)
    block = np.minimum(first//10,9)
    b = np.arange(10)
    allowed = b<=block[...,None]
    rich = (first<100)[...,None] & (b>=np.maximum(block-1,0)[...,None]) & allowed
    return allowed,rich


def balanced_indices(rich, batch, rng):
    yes,no = np.flatnonzero(rich),np.flatnonzero(~rich)
    if not len(yes) or not len(no) or batch%2:
        raise ValueError('both failure-rich and ordinary fitting rows and an even batch required')
    return np.concatenate([rng.choice(yes,batch//2),rng.choice(no,batch//2)])


def training_arrays(data, history, hist_actions, indices, model, scaler, device='cuda'):
    """Cache detached three-real-frame contexts and past/executed action windows only."""
    z = data['z'][:,indices]
    c,e,t,d = z.shape
    full = np.concatenate([np.broadcast_to(history[indices],(c,e,3,d)),z],axis=2)
    contexts = np.stack([full[:,:,b:b+3] for b in range(t)],axis=2)
    raw = np.concatenate([np.broadcast_to(hist_actions[indices].reshape(e,2,60),(c,e,2,60)),
                          data['actions'][:,indices]],axis=2).reshape(c*e,12,10,6)
    mean,std = [torch.as_tensor(x,dtype=torch.float32,device=device) for x in scaler]
    flat = ((torch.as_tensor(raw,device=device,dtype=torch.float32)-mean)/std).reshape(c*e,12,60)
    encoded = []
    with torch.no_grad():
        for b in range(t):
            buffer = torch.zeros_like(flat)
            buffer[:,:b+3] = flat[:,:b+3]
            encoded.append(model.action_encoder(buffer)[:,b:b+3].detach().cpu().numpy())
    act = np.stack(encoded,axis=1).reshape(c,e,t,3,-1)
    mask,rich = fitting_masks(data['first'][:,indices])
    return dict(context=contexts[mask],action=act[mask],target=z[mask],
                physical=data['truth'][:,indices][mask],rich=rich[mask])


def freeze_for_predictor(model):
    model.eval().requires_grad_(False)
    for module in [model.predictor,model.pred_proj]:
        module.requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad]


def tensor_state(module):
    return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}


def assert_frozen(model, before):
    for name,state in before.items():
        now = getattr(model,name).state_dict()
        if any(not torch.equal(v.cpu(),state[k]) for k,v in now.items()):
            raise ValueError(f'frozen module changed: {name}')
    if any(m.training for m in model.modules()):
        raise ValueError('model mode or running statistics were not frozen')


def fit_probe_copy(probe, data, indices, *, seed, steps, batch, lr, counter, deadline):
    probe.eval().requires_grad_(True)
    before = {k:v.detach().cpu().clone() for k,v in probe.named_buffers()}
    mask,rich = fitting_masks(data['first'][:,indices])
    device = next(probe.parameters()).device
    z = torch.as_tensor(data['z'][:,indices][mask],device=device)
    y = torch.as_tensor(data['truth'][:,indices][mask],dtype=torch.float32,device=device)
    scale = torch.tensor([.2,.5,5.],device=device)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(probe.parameters(),lr=lr,weight_decay=1e-4)
    history = []
    for step in range(steps):
        deadline()
        ix = balanced_indices(rich[mask],batch,rng)
        opt.zero_grad(set_to_none=True)
        loss = ((probe(z[ix])-y[ix])/scale).square().mean()
        if not torch.isfinite(loss):
            raise ValueError('nonfinite probe fitting loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(),1.)
        opt.step()
        counter['probe_gradient_updates']+=1
        counter['probe_training_rows']+=batch
        if (step+1)%100==0:
            history.append(dict(step=step+1,loss=float(loss.detach())))
    assert all(torch.equal(v.cpu(),before[k]) for k,v in probe.named_buffers())
    probe.requires_grad_(False)
    return history


def fit_predictor(model, probe, arrays, *, seed, steps, batch, lr, physical_weight, counter, deadline):
    before = {name:tensor_state(getattr(model,name)) for name in ['encoder','projector','action_encoder']}
    buffers = {k:v.detach().cpu().clone() for k,v in model.named_buffers()}
    params = freeze_for_predictor(model)
    probe.eval().requires_grad_(False)
    device = next(model.parameters()).device
    a = {k:torch.as_tensor(v,device=device,dtype=torch.float32) for k,v in arrays.items() if k!='rich'}
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(params,lr=lr,weight_decay=1e-4)
    scale = torch.tensor([.2,.5,5.],device=device)
    history = []
    for step in range(steps):
        deadline()
        ix = balanced_indices(arrays['rich'],batch,rng)
        opt.zero_grad(set_to_none=True)
        pred = model.predict(a['context'][ix],a['action'][ix])[:,-1]
        latent = (pred-a['target'][ix]).square().mean()
        physical = ((probe(pred)-a['physical'][ix])/scale).square().mean()
        loss = latent+physical_weight*physical
        if not torch.isfinite(loss):
            raise ValueError('nonfinite predictor fitting loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params,1.)
        opt.step()
        counter['predictor_gradient_updates']+=1
        if (step+1)%100==0:
            history.append(dict(step=step+1,loss=float(loss.detach()),latent=float(latent.detach()),physical=float(physical.detach())))
    assert_frozen(model,before)
    assert all(torch.equal(v.cpu(),buffers[k]) for k,v in model.named_buffers())
    model.requires_grad_(False)
    return history


def metric_arrays(readout, truth, first):
    """Input S,C,E,T,3; one source episode remains one bootstrap cluster."""
    ro = np.asarray(readout)
    if ro.ndim!=5 or ro.shape[1:]!=truth.shape or first.shape!=truth.shape[:2]:
        raise ValueError('seed,candidate,episode,time,physical arrays required')
    if not np.isfinite(ro).all():
        raise ValueError('nonfinite evaluation')
    unsafe = health_clearance(ro[...,0],ro[...,1])<=0
    actual = health_clearance(truth[...,0],truth[...,1])<=0
    ci,ei = np.indices(first.shape)
    block = np.minimum(first//10,9)
    failed = first<100
    endpoint = actual[ci,ei,block]&failed
    healthy = ((np.arange(10)+1)*10 < (first+1)[...,None])&~actual
    error = ro-truth[None]
    # Sum seeds and candidates, preserving episode as the sole resampling unit.
    s = len(ro)
    return dict(event_num=(unsafe[:,ci,ei,block]&endpoint[None]).sum((0,1)),
        event_den=endpoint.sum(0)*s,
        false_num=(unsafe&healthy[None]).sum((0,1,3)),false_den=healthy.sum((0,2))*s,
        height_num=(np.abs(error[...,0])*healthy[None]).sum((0,1,3)),
        pitch_num=(np.abs(error[...,1])*healthy[None]).sum((0,1,3)),
        speed_num=(np.abs(error[...,2])*healthy[None]).sum((0,1,3)),
        height_bias_num=(error[:,ci,ei,block,0]*failed[None]).sum((0,1)),
        fail_den=failed.sum(0)*s,failed_pairs=int(failed.sum()),endpoint_pairs=int(endpoint.sum()))


def summarize(readout, truth, first, *, seed=20261609, replicates=2000):
    a = metric_arrays(readout,truth,first)
    e = truth.shape[1]
    draws = np.random.default_rng(seed).integers(e,size=(replicates,e))
    def ratio(num,den):
        n,d = a[num],a[den]
        ds = d[draws].sum(1)
        if not d.sum():
            return dict(point=None,lo=None,hi=None,numerator=0,denominator=0)
        x = n[draws].sum(1)[ds>0]/ds[ds>0]
        return dict(point=float(n.sum()/d.sum()),lo=float(np.quantile(x,.025)),hi=float(np.quantile(x,.975)),
                    numerator=float(n.sum()),denominator=int(d.sum()))
    return dict(event_detection=ratio('event_num','event_den'),false_alarm=ratio('false_num','false_den'),
        healthy_height_mae=ratio('height_num','false_den'),healthy_pitch_mae=ratio('pitch_num','false_den'),
        healthy_speed_mae=ratio('speed_num','false_den'),event_height_bias=ratio('height_bias_num','fail_den'),
        failed_pairs=a['failed_pairs'],endpoint_pairs=a['endpoint_pairs'],episodes=e,seeds=len(readout))


def repair_gate(base, repaired, config):
    def point(m,key):
        return m[key]['point']
    needed=['event_detection','false_alarm','healthy_height_mae','healthy_pitch_mae','healthy_speed_mae']
    if any(point(m,k) is None for m in [base,repaired] for k in needed):
        return dict(passed=False,reason='undefined denominator')
    tests=dict(detection=point(repaired,'event_detection')>=point(base,'event_detection')+config['minimum_detection_gain'],
        false_alarms=point(repaired,'false_alarm')<=point(base,'false_alarm')+config['maximum_false_alarm_increase'])
    for key in needed[2:]:
        tests[key]=point(repaired,key)<=config['maximum_healthy_mae_ratio']*point(base,key)
    return dict(passed=all(tests.values()),tests=tests)


def select_repair(summaries, gate):
    checks = {arm:repair_gate(summaries['baseline'],summaries[arm],gate) for arm in ARMS[1:]}
    passed = [arm for arm in ARMS[1:] if checks[arm]['passed']]
    chosen = min(passed,key=lambda k:(-summaries[k]['event_detection']['point'],
        summaries[k]['false_alarm']['point'],ARMS.index(k))) if passed else None
    return dict(chosen=chosen,gates=checks)


def nominate(readout, progress_floor=.9):
    """Frozen imagined-only selection: lower risk, retained progress, baseline fallback."""
    ro=np.asarray(readout)
    unsafe=health_clearance(ro[...,0],ro[...,1])<=0
    risk=unsafe.any(-1).mean((0,2))
    progress=(ro[...,2].sum(-1)*.08).mean((0,2))
    eligible=[i for i in range(1,len(risk)) if risk[i]<risk[0] and progress[0]>0 and progress[i]>=progress_floor*progress[0]]
    chosen=min(eligible,key=lambda i:(risk[i],-progress[i],i)) if eligible else 0
    return dict(index=chosen,eligible=eligible,imagined_risk=risk.tolist(),imagined_progress=progress.tolist())
