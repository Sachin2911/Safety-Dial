"""Matched imitation-data budgets and family-balanced validation for coverage diagnosis."""
from __future__ import annotations

import numpy as np
import torch


def coverage_indices(lag_episode, lag_validation, ppo_episode, ppo_validation, *, seed,
                     fit_episodes_per_family=120):
    """Whole-episode roles, matched sample budgets, and 50/50 mixed training.

    The control draws from all original fit episodes; the mixed arm draws from a
    fixed random half of each family's fit episodes. Short PPO episodes reduce both
    arms' sample budgets equally. Validation uses ALL held-out samples, with equal
    total weight per family regardless of episode length. No outcomes enter here.
    """
    lag_episode, ppo_episode = np.asarray(lag_episode), np.asarray(ppo_episode)
    lag_validation, ppo_validation = np.asarray(lag_validation), np.asarray(ppo_validation)
    if set(lag_episode).intersection(ppo_episode):
        raise ValueError("source policy families must have disjoint episodes")
    rng = np.random.default_rng(seed)
    fit_lag = np.setdiff1d(np.unique(lag_episode), lag_validation)
    fit_ppo = np.setdiff1d(np.unique(ppo_episode), ppo_validation)
    n = int(fit_episodes_per_family)
    if n < 1 or min(len(fit_lag), len(fit_ppo)) < n:
        raise ValueError("insufficient fit episodes for the declared coverage design")
    selected_lag = np.sort(rng.choice(fit_lag, n, replace=False))
    selected_ppo = np.sort(rng.choice(fit_ppo, n, replace=False))
    all_lag = np.flatnonzero(np.isin(lag_episode, fit_lag))
    pool_lag = np.flatnonzero(np.isin(lag_episode, selected_lag))
    pool_ppo = np.flatnonzero(np.isin(ppo_episode, selected_ppo))
    per_family = min(len(pool_lag), len(pool_ppo), len(all_lag)//2)
    val_lag = np.flatnonzero(np.isin(lag_episode, lag_validation))
    val_ppo = np.flatnonzero(np.isin(ppo_episode, ppo_validation))
    if min(per_family, len(val_lag), len(val_ppo)) < 1:
        raise ValueError("need nonempty fit and validation samples in both families")
    idx = {
        "baseline_lag": np.sort(rng.choice(all_lag, 2*per_family, replace=False)),
        "mixed_lag": np.sort(rng.choice(pool_lag, per_family, replace=False)),
        "mixed_ppo": np.sort(rng.choice(pool_ppo, per_family, replace=False)),
        "validation_lag": val_lag, "validation_ppo": val_ppo,
    }
    weights = np.r_[np.full(len(val_lag), .5/len(val_lag)),
                    np.full(len(val_ppo), .5/len(val_ppo))]
    meta = {"fit_episodes_selected_lag":selected_lag.tolist(),
            "fit_episodes_selected_ppo":selected_ppo.tolist(),
            "training_samples_per_arm":2*per_family,"mixed_samples_per_family":per_family,
            "validation_samples_lag":len(val_lag),"validation_samples_ppo":len(val_ppo),
            "validation_weight_per_family":.5,"seed":int(seed)}
    return idx, weights, meta


# Fit loop adapted from evoMlpFit at e8f650b; only validation weighting changes.
def fit_coverage_seed(policy, Xfit, Yfit, Xval, Yval, *, seed, epochs, batch_size, lr, weight_decay,
             val_weights, progress=None):
    """Return best validation checkpoint, history and selected epoch for one seed.

    All tensors must already be on the policy device. The final deployment checkpoint
    is the selected fit-only model: validation targets never receive gradient updates.
    """
    if len(Xfit) < 1 or len(Xval) < 1 or epochs < 1 or batch_size < 1:
        raise ValueError("need nonempty fit/validation sets and positive run sizes")
    val_weights = torch.as_tensor(val_weights, device=Xval.device, dtype=torch.float64)
    if val_weights.shape != (len(Xval),) or not bool(torch.isfinite(val_weights).all()) \
            or bool((val_weights < 0).any()) or not bool(val_weights.sum() > 0):
        raise ValueError("validation weights must be finite nonnegative sample weights")
    val_weights = val_weights / val_weights.sum()
    torch.manual_seed(int(seed))
    net = policy.network()
    opt = torch.optim.AdamW(net.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    gen = torch.Generator(device=Xfit.device).manual_seed(int(seed))
    best, history, updates = None, [], 0
    for epoch in range(1, int(epochs) + 1):
        net.train()
        order = torch.randperm(len(Xfit), generator=gen, device=Xfit.device)
        sse = 0.0
        for lo in range(0, len(order), int(batch_size)):
            idx = order[lo:lo + int(batch_size)]
            pred = net(Xfit[idx])
            loss = (pred - Yfit[idx]).square().mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite imitation loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sse += float(loss.detach()) * len(idx)
            updates += 1
        net.eval()
        with torch.inference_mode():
            pred = torch.cat([net(Xval[i:i + 4096]).clamp(-1, 1) for i in range(0, len(Xval), 4096)])
            vmse = float(((pred.double() - Yval.double()).square().mean(-1) * val_weights).sum())
        if not np.isfinite(vmse):
            raise FloatingPointError("nonfinite validation loss")
        row = {"seed": int(seed), "epoch": epoch, "fit_raw_mse": sse / len(Xfit), "val_mse": vmse}
        history.append(row)
        if best is None or vmse < best["val_mse"]:
            best = {**row, "theta": policy.network_theta(net).copy()}
        if progress is not None and (epoch == 1 or epoch % 10 == 0):
            progress(row)
    return best, history, updates

