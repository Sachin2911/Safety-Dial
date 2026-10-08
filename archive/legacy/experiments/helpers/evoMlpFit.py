"""Fixed-budget MLP imitation fit; selection sees only offline validation frames."""
from __future__ import annotations

import numpy as np
import torch


def fit_seed(policy, Xfit, Yfit, Xval, Yval, *, seed, epochs, batch_size, lr, weight_decay,
             progress=None):
    """Return best validation checkpoint, history and selected epoch for one seed.

    All tensors must already be on the policy device. The final deployment checkpoint
    is the selected fit-only model: validation targets never receive gradient updates.
    """
    if len(Xfit) < 1 or len(Xval) < 1 or epochs < 1 or batch_size < 1:
        raise ValueError("need nonempty fit/validation sets and positive run sizes")
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
            vmse = float((pred.double() - Yval.double()).square().mean())
        if not np.isfinite(vmse):
            raise FloatingPointError("nonfinite validation loss")
        row = {"seed": int(seed), "epoch": epoch, "fit_raw_mse": sse / len(Xfit), "val_mse": vmse}
        history.append(row)
        if best is None or vmse < best["val_mse"]:
            best = {**row, "theta": policy.network_theta(net).copy()}
        if progress is not None and (epoch == 1 or epoch % 10 == 0):
            progress(row)
    return best, history, updates


def select_fit(results):
    """Predeclared deterministic tie order, with no access to real outcomes."""
    return min(results, key=lambda row: (row["val_mse"], row["seed"], row["epoch"]))
