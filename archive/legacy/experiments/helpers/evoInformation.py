"""Privileged diagnostic reference, exact state features, and checkpoint training."""
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

from helpers.evoHistoryPolicy import HistoryBlockPolicy
from helpers.evoReal import _physics_env, _start, _step
from helpers.walkerRules import health_clearance, rule_unsafe


def observations(qpos, qvel):
    return np.concatenate([np.asarray(qpos)[..., 1:], np.clip(qvel, -10, 10)], axis=-1).astype(np.float32)


def cached_state_features(path, episodes):
    out = np.empty((len(episodes), 34), np.float32)
    with h5py.File(path) as f:
        offsets, lengths = f['ep_offset'][:], f['ep_len'][:]
        for e in np.unique(episodes):
            lo, n = int(offsets[e]), int(lengths[e])
            obs = observations(f['qpos'][lo:lo+n], f['qvel'][lo:lo+n])
            t = np.arange(0, n, 5)
            t = t[(t >= 10) & (t + 10 <= n)]
            idx = np.flatnonzero(episodes == e)
            if len(idx) != len(t):
                raise ValueError('state feature rows do not match cached episode rows')
            out[idx] = np.concatenate([obs[t], obs[t]-obs[t-10]], axis=1)
    return out


class StateReferencePolicy(HistoryBlockPolicy):
    def __init__(self, *args, state_mean, state_std, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_mean = torch.as_tensor(state_mean, dtype=torch.float32, device=self.device)
        self.state_std = torch.as_tensor(state_std, dtype=torch.float32, device=self.device)

    def state_features(self, current, previous, past):
        current, previous = [torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in (current, previous)]
        x = (torch.cat([current, current-previous], -1) - self.state_mean) / self.state_std
        past = torch.as_tensor(past, dtype=torch.float32, device=self.device).reshape(-1, 10, 6)
        past = ((past-self.action_mean)/self.action_std).reshape(-1, 60)
        return torch.cat([x, torch.zeros((len(x), 350), device=self.device), past], -1)

    def state(self):
        return {**super().state(), 'policy_kind': 'state_reference_history_v1',
                'state_mean': self.state_mean.cpu().numpy(), 'state_std': self.state_std.cpu().numpy()}

    @classmethod
    def from_state(cls, state, *, device='cpu'):
        if str(state.get('policy_kind','')) != 'state_reference_history_v1':
            raise ValueError('not a privileged state reference')
        return cls((state['action_mean'],state['action_std']),state['feature_mean'],
                   state['feature_std'],hidden=state['hidden'],state_mean=state['state_mean'],
                   state_std=state['state_std'],device=device)


def evaluate_state(policy, theta, roots):
    env = _physics_env()
    logs = []
    try:
        with torch.inference_mode():
            for root in roots:
                qp, qv, xv = np.empty((101, 9)), np.empty((101, 9)), np.empty(100)
                _start(env, root, qp, qv)
                previous = observations(root.history_qpos[-2], root.history_qvel[-2])[None]
                past = root.history_actions[-1].reshape(1, 60)
                actions = []
                for b in range(10):
                    current = observations(qp[b*10], qv[b*10])[None]
                    feats = policy.state_features(current, previous, past)
                    a = policy.act(np.asarray(theta)[None], feats[:, None])[0, 0].cpu().numpy()
                    actions.append(a)
                    for t in range(b*10, b*10+10):
                        _step(env.unwrapped, a.reshape(10, 6)[t-b*10], qp, qv, xv, t)
                    previous, past = current, a[None]
                clearance = health_clearance(qp[1:, 1], qp[1:, 2])
                unsafe = rule_unsafe('health', clearance)
                logs.append(dict(qpos=qp,qvel=qv,actions=np.asarray(actions),
                    dense_clearance=clearance,dense_violated=bool(unsafe.any()),
                    dense_first_step=int(np.flatnonzero(unsafe)[0]) if unsafe.any() else 100,
                    progress=float(qp[-1,0]-qp[0,0])))
    finally:
        env.close()
    return {k:np.stack([r[k] for r in logs]) for k in logs[0]}


def fit_candidates(policy, Xfit, Yfit, Xval, Yval, weights, *, seed, epochs,
                   checkpoint_epochs, batch_size=1024, lr=.001, weight_decay=.0001, progress=None):
    """Original AdamW fit loop, retaining only declared epochs and offline best."""
    torch.manual_seed(seed)
    net=policy.network()
    opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=weight_decay)
    gen=torch.Generator(device=Xfit.device).manual_seed(seed)
    weights=torch.as_tensor(weights,device=Xval.device,dtype=torch.float64)
    weights=weights/weights.sum()
    best,kept,hist,updates=None,{},[],0
    for epoch in range(1,epochs+1):
        net.train()
        order=torch.randperm(len(Xfit),generator=gen,device=Xfit.device)
        sse=0.
        for lo in range(0,len(order),batch_size):
            idx=order[lo:lo+batch_size]
            loss=(net(Xfit[idx])-Yfit[idx]).square().mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('nonfinite fit loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sse+=float(loss.detach())*len(idx)
            updates+=1
        net.eval()
        with torch.inference_mode():
            pred=torch.cat([net(Xval[i:i+4096]).clamp(-1,1) for i in range(0,len(Xval),4096)])
            mse=float(((pred.double()-Yval.double()).square().mean(-1)*weights).sum())
        row=dict(seed=seed,epoch=epoch,val_mse=mse,fit_raw_mse=sse/len(Xfit))
        hist.append(row)
        if best is None or mse<best['val_mse'] or epoch in checkpoint_epochs:
            snapshot=dict(**row,theta=policy.network_theta(net).copy())
            if best is None or mse<best['val_mse']:
                best=snapshot
            if epoch in checkpoint_epochs:
                kept[epoch]=snapshot
        if progress is not None and (epoch==1 or epoch%10==0):
            progress(row)
    kept[best['epoch']]=best
    return list(kept.values()),hist,updates


def candidate_key(row, minimum_progress):
    return (row['progress_mean']<minimum_progress,row['violations'],
            -row['progress_mean'],row['val_mse'],row['seed'],row['epoch'])
