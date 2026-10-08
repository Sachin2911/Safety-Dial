"""Privileged coordinate-dynamics reference for Push-T (pushT.md, E1 references).

A small MLP on expert STATE transitions at block granularity: given the true 7-d state
and the next `ACTION_BLOCK` raw actions, predict the state at the block endpoint. It uses
privileged simulator state that the latent method never sees, so it is a reference, not a
competitor: if it solves the constraint decision where the latent model fails, that is
evidence against making the latent route central.

Trained on whole expert episodes of the `replay` split; validated on `retention`.
Rolled out autoregressively over the horizon from the true root state.
"""

from __future__ import annotations

import numpy as np
import torch

from helpers.pushtAssets import ACTION_BLOCK

FEAT_DIM = 6  # pusher x, y, block x, y, sin, cos


def state_to_feat(states: np.ndarray) -> np.ndarray:
    s = np.asarray(states, float)
    return np.stack([s[..., 0], s[..., 1], s[..., 2], s[..., 3], np.sin(s[..., 4]), np.cos(s[..., 4])], axis=-1).astype(np.float32)


def feat_to_state(feat: np.ndarray, pusher_vel=None) -> np.ndarray:
    f = np.asarray(feat, float)
    th = np.arctan2(f[..., 4], f[..., 5]) % (2 * np.pi)
    out = np.zeros((*f.shape[:-1], 7))
    out[..., 0:4] = f[..., 0:4]
    out[..., 4] = th
    return out


class CoordDynamics(torch.nn.Module):
    def __init__(self, hidden=(256, 256), feat_mean=None, feat_std=None, act_std=0.2):
        super().__init__()
        d_in = 6 + ACTION_BLOCK * 2
        layers, d = [], d_in
        for h in hidden:
            layers += [torch.nn.Linear(d, h), torch.nn.ReLU()]
            d = h
        layers.append(torch.nn.Linear(d, 6))
        self.net = torch.nn.Sequential(*layers)
        self.register_buffer("feat_mean", torch.as_tensor(feat_mean if feat_mean is not None else np.zeros(6), dtype=torch.float32))
        self.register_buffer("feat_std", torch.as_tensor(feat_std if feat_std is not None else np.ones(6), dtype=torch.float32))
        self.act_std = float(act_std)
        self.register_buffer("delta_scale", torch.tensor([20.0, 20.0, 10.0, 10.0, 0.2, 0.2]))

    def forward(self, feat, act_flat):
        x = torch.cat([(feat - self.feat_mean) / self.feat_std, act_flat / self.act_std], dim=-1)
        return feat + self.net(x) * self.delta_scale

    @torch.no_grad()
    def rollout(self, state0: np.ndarray, tapes: np.ndarray) -> np.ndarray:
        """state0 (7,), tapes (S, K, ACTION_BLOCK, 2) raw -> endpoint states (S, K+1, 7)."""
        dev = self.feat_mean.device
        tapes = np.asarray(tapes, float)
        S, K = tapes.shape[:2]
        f = torch.as_tensor(state_to_feat(np.repeat(state0[None], S, 0)), device=dev)
        a = torch.as_tensor(tapes.reshape(S, K, -1), dtype=torch.float32, device=dev)
        out = [f]
        for k in range(K):
            f = self(f, a[:, k])
            out.append(f)
        feats = torch.stack(out, 1).cpu().numpy()
        return feat_to_state(feats)


def expert_block_transitions(h5_path, episodes, max_steps: int | None = None):
    """(feat_t, actions t..t+B-1 flat, feat_{t+B}) for every block-aligned t in the episodes."""
    import h5py
    import hdf5plugin  # noqa: F401

    X, A, Y = [], [], []
    n = 0
    with h5py.File(h5_path, "r") as fh:
        ep_len, ep_off = fh["ep_len"][:], fh["ep_offset"][:]
        for ep in episodes:
            a0, L = int(ep_off[ep]), int(ep_len[ep])
            st = fh["state"][a0 : a0 + L]
            ac = fh["action"][a0 : a0 + L]
            for t in range(0, L - ACTION_BLOCK, 1):
                X.append(state_to_feat(st[t]))
                A.append(ac[t : t + ACTION_BLOCK].reshape(-1))
                Y.append(state_to_feat(st[t + ACTION_BLOCK]))
            n += L
            if max_steps and n >= max_steps:
                break
    return np.asarray(X, np.float32), np.asarray(A, np.float32), np.asarray(Y, np.float32)


def fit_coord_dynamics(X, A, Y, Xv, Av, Yv, *, device="cuda", epochs=30, lr=1e-3, batch=2048, seed=0, verbose=True):
    torch.manual_seed(seed)
    model = CoordDynamics(feat_mean=X.mean(0), feat_std=X.std(0) + 1e-6, act_std=float(A.std())).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    xt, at, yt = (torch.from_numpy(v).to(device) for v in (X, A, Y))
    xv, av, yv = (torch.from_numpy(v).to(device) for v in (Xv, Av, Yv))
    w = 1.0 / model.delta_scale
    best, best_state = np.inf, None
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(len(xt), device=device)
        for i in range(0, len(xt), batch):
            b = idx[i : i + batch]
            loss = (((model(xt[b], at[b]) - yt[b]) * w) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = model(xv, av)
            vloss = float((((pv - yv) * w) ** 2).mean())
            centre = float(torch.linalg.norm(pv[:, 2:4] - yv[:, 2:4], dim=1).median())
        if vloss < best:
            best, best_state = vloss, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose and (ep + 1) % 10 == 0:
            print(f"[coord] epoch {ep + 1}/{epochs} val loss {vloss:.4f} block-centre median err {centre:.2f}px")
    model.load_state_dict(best_state)
    model.eval()
    return model, {"val_loss": best}
