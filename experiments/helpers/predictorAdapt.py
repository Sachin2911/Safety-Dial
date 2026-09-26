"""Predictor-side adaptation of a LeWM with a frozen encoder (protocol.md).

Frozen: encoder and observation projector (gradients AND running statistics: they stay
in eval() after any parent train() call), scalers, probes. Trainable ("predictor-side"):
action encoder, predictor and prediction projector; "predictor-only" trains the
predictor alone. Targets are cached, detached latents of the real frames from the
frozen encoder, so training touches no images.

Loss: teacher-forced latent MSE over sliding windows of the predictor's context length
(the upstream training loss, `third_party/le-wm/train.py`), optionally plus a short
autoregressive rollout term. SIGReg is omitted: frozen targets give it no gradient.
Every batch mixes the original-data replay subset with acquired data (half and half by
default). Each run restarts from the released weights.

Clips: a clip is `T` block-endpoint frames and the `T-1` action blocks between them,
stored as latents (T, D) and normalised flat actions (T-1, 10).
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from helpers.imagination import blocks_to_model
from helpers.pushtAssets import ACTION_BLOCK, HISTORY_FRAMES

PREDICTOR_SIDE = ("action_encoder", "predictor", "pred_proj")
PREDICTOR_ONLY = ("predictor",)


@dataclass
class AdaptConfig:
    modules: str = "predictor_side"  # "predictor_side" | "predictor_only"
    loss: str = "teacher_forced"  # "teacher_forced" | "tf_plus_rollout"
    rollout_weight: float = 0.5
    lr: float = 5e-5
    weight_decay: float = 1e-3
    steps: int = 1500
    batch_size: int = 64
    replay_frac: float = 0.5
    grad_clip: float = 1.0
    seed: int = 0
    ctx: int = HISTORY_FRAMES

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------------------
# clips
# --------------------------------------------------------------------------------------
@dataclass
class ClipSet:
    latents: np.ndarray  # (N, T, D) float32
    actions: np.ndarray  # (N, T-1, 10) float32 normalised flat
    meta: dict

    def __len__(self) -> int:
        return len(self.latents)

    def subset(self, idx) -> "ClipSet":
        return ClipSet(self.latents[idx], self.actions[idx], dict(self.meta))

    @staticmethod
    def concat(sets: list["ClipSet"]) -> "ClipSet":
        return ClipSet(np.concatenate([s.latents for s in sets]), np.concatenate([s.actions for s in sets]), {"concat": [s.meta for s in sets]})

    def save(self, path) -> None:
        np.savez_compressed(path, latents=self.latents, actions=self.actions, meta=str(self.meta))

    @staticmethod
    def load(path) -> "ClipSet":
        z = np.load(path, allow_pickle=False)
        return ClipSet(z["latents"], z["actions"], {"loaded_from": str(path)})


def expert_replay_clips(model, process, h5_path, episodes, n_clips: int, rng, *, T: int = HISTORY_FRAMES + 5, device="cuda", verbose=True) -> ClipSet:
    """Windows of T endpoint frames (stride ACTION_BLOCK) from whole expert episodes."""
    import h5py
    import hdf5plugin  # noqa: F401

    from helpers.imagination import encode

    span = (T - 1) * ACTION_BLOCK + 1
    eps = list(rng.permutation(np.asarray(episodes)))
    Z, A = [], []
    t0 = time.time()
    with h5py.File(h5_path, "r") as f:
        ep_len, ep_off = f["ep_len"][:], f["ep_offset"][:]
        for ep in eps:
            ep = int(ep)
            L = int(ep_len[ep])
            if L < span:
                continue
            a0 = int(ep_off[ep])
            n_here = max(1, min(4, (L - span) // 10 + 1))
            starts = rng.choice(np.arange(0, L - span + 1), size=n_here, replace=False)
            lo, hi = int(starts.min()), int(starts.max() + span)
            frames = f["pixels"][a0 + lo : a0 + hi]
            acts = f["action"][a0 + lo : a0 + hi].astype(np.float64)
            for s in starts:
                s = int(s - lo)
                idx = s + np.arange(T) * ACTION_BLOCK
                z = encode(model, frames[idx], device).cpu().numpy()
                blocks = acts[s : s + (T - 1) * ACTION_BLOCK].reshape(T - 1, ACTION_BLOCK, 2)
                Z.append(z)
                A.append(blocks_to_model(process, blocks).numpy())
                if len(Z) >= n_clips:
                    break
            if len(Z) >= n_clips:
                break
            if verbose and len(Z) % 500 < n_here:
                print(f"[adapt] replay clips {len(Z)}/{n_clips} ({time.time() - t0:.0f}s)")
    return ClipSet(np.stack(Z).astype(np.float32), np.stack(A).astype(np.float32), {"source": "expert_replay", "n": len(Z), "T": T})


def branch_clips(imaginer, bank, indices, *, env=None) -> ClipSet:
    """Clips from executed branches: root history (ctx-1 frames) + the 6 branch endpoint frames."""
    from helpers.pushtReplay import make_env, reset_root

    env = env or make_env()
    cache = {}
    Z, A = [], []
    for j in indices:
        b = bank.branch(int(j), frames=True)
        ri = int(b["root_index"])
        if ri not in cache:
            ctx = reset_root(env, bank.roots[ri])
            cache[ri] = (imaginer.encode(ctx.frames[:-1]).cpu().numpy(), ctx.history_actions.copy())
        z_prev, hist_blocks = cache[ri]
        z_branch = imaginer.encode(b["frames"]).cpu().numpy()
        blocks = np.concatenate([hist_blocks, b["tape"].astype(np.float64)], 0)
        Z.append(np.concatenate([z_prev, z_branch], 0))
        A.append(blocks_to_model(imaginer.process, blocks).numpy())
    return ClipSet(np.stack(Z).astype(np.float32), np.stack(A).astype(np.float32), {"source": "branches", "n": len(Z)})


# --------------------------------------------------------------------------------------
# freezing and losses
# --------------------------------------------------------------------------------------
def freeze_for_adaptation(model, modules: str) -> list[torch.nn.Parameter]:
    """Freeze everything except the named modules; keep frozen modules in eval()."""
    train_names = PREDICTOR_SIDE if modules == "predictor_side" else PREDICTOR_ONLY
    params = []
    for name, mod in model.named_children():
        trainable = name in train_names
        for p in mod.parameters():
            p.requires_grad_(trainable)
        if trainable:
            mod.train()
            params += list(mod.parameters())
        else:
            mod.eval()
    return params


def keep_frozen_eval(model, modules: str) -> None:
    """Frozen modules stay in eval(); every BatchNorm layer stays in eval() as well.

    The prediction projector contains BatchNorm1d(2048). In train mode its running
    statistics drift toward our small mixed batches and the released model's loss on
    held-out expert clips rises from 0.0087 to 0.0103 even with replay data ONLY (measured
    26 September 2026, 300 steps, lr 5e-5); with the running statistics frozen the same
    run lowers it to 0.0060. The affine BatchNorm weights remain trainable. This extends
    the protocol's "freeze running statistics" rule to the trainable modules' normalisers.
    """
    train_names = PREDICTOR_SIDE if modules == "predictor_side" else PREDICTOR_ONLY
    for name, mod in model.named_children():
        if name not in train_names:
            mod.eval()
    for mod in model.modules():
        if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
            mod.eval()


def teacher_forced_loss(model, z: torch.Tensor, a: torch.Tensor, ctx: int) -> torch.Tensor:
    """Sliding windows of `ctx` frames; predict the next frame at every window position."""
    T = z.shape[1]
    act_emb = model.action_encoder(a)  # (B, T-1, E)
    losses = []
    for s in range(0, T - ctx):
        pred = model.predict(z[:, s : s + ctx], act_emb[:, s : s + ctx])  # (B, ctx, D)
        losses.append((pred - z[:, s + 1 : s + ctx + 1]).pow(2).mean())
    return torch.stack(losses).mean()


def rollout_loss(model, z: torch.Tensor, a: torch.Tensor, ctx: int) -> torch.Tensor:
    """Autoregressive from the first `ctx` real frames; MSE at every imagined step."""
    T = z.shape[1]
    act_emb = model.action_encoder(a)
    emb_list = list(z[:, :ctx].unbind(1))
    losses = []
    for t in range(ctx, T):
        lo = t - ctx
        pred = model.predict(torch.stack(emb_list[lo:t], 1), act_emb[:, lo:t])[:, -1]
        losses.append((pred - z[:, t]).pow(2).mean())
        emb_list.append(pred)
    return torch.stack(losses).mean()


# --------------------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------------------
def adapt(base_state_dict: dict, model_template, acquired: ClipSet | None, replay: ClipSet, cfg: AdaptConfig, *, device="cuda", verbose=True, eval_fn=None, eval_every=0):
    """Fresh copy of the released weights, adapted on acquired+replay clips. Returns (model, log)."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = copy.deepcopy(model_template).to(device)
    model.load_state_dict(base_state_dict, strict=True)
    params = freeze_for_adaptation(model, cfg.modules)
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps)
    rng = np.random.default_rng(cfg.seed)
    zr = torch.from_numpy(replay.latents).to(device)
    ar = torch.from_numpy(replay.actions).to(device)
    have_acq = acquired is not None and len(acquired) > 0
    if have_acq:
        za = torch.from_numpy(acquired.latents).to(device)
        aa = torch.from_numpy(acquired.actions).to(device)
    n_rep = int(round(cfg.batch_size * (cfg.replay_frac if have_acq else 1.0)))
    n_acq = cfg.batch_size - n_rep if have_acq else 0
    log = {"cfg": cfg.to_dict(), "steps": [], "n_acquired": int(len(acquired)) if have_acq else 0, "n_replay": int(len(replay))}
    t0 = time.time()
    for step in range(cfg.steps):
        keep_frozen_eval(model, cfg.modules)
        ir = torch.as_tensor(rng.integers(len(zr), size=n_rep), device=device)
        z, a = zr[ir], ar[ir]
        if n_acq:
            ia = torch.as_tensor(rng.integers(len(za), size=n_acq), device=device)
            z, a = torch.cat([z, za[ia]]), torch.cat([a, aa[ia]])
        loss_tf = teacher_forced_loss(model, z, a, cfg.ctx)
        loss = loss_tf
        loss_ro = torch.zeros(())
        if cfg.loss == "tf_plus_rollout":
            loss_ro = rollout_loss(model, z, a, cfg.ctx)
            loss = loss_tf + cfg.rollout_weight * loss_ro
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()
        sched.step()
        if (step + 1) % 100 == 0 or step == 0:
            rec = {"step": step + 1, "loss": float(loss.detach()), "loss_tf": float(loss_tf.detach()), "loss_rollout": float(loss_ro.detach()), "t": time.time() - t0}
            if eval_fn is not None and eval_every and (step + 1) % eval_every == 0:
                model.eval()
                rec["eval"] = eval_fn(model)
            log["steps"].append(rec)
            if verbose and (step + 1) % 500 == 0:
                print(f"[adapt] step {step + 1}/{cfg.steps} loss {float(loss):.4f} (tf {float(loss_tf):.4f} ro {float(loss_ro):.4f}) {time.time() - t0:.0f}s")
    model.eval()
    log["wall_clock_s"] = time.time() - t0
    return model, log


def predictor_side_state(model, modules: str = "predictor_side") -> dict:
    names = PREDICTOR_SIDE if modules == "predictor_side" else PREDICTOR_ONLY
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k.split(".")[0] in names}


class ReadoutCorrection(torch.nn.Module):
    """Control: a residual MLP on IMAGINED latents (plus horizon index) -> pose correction.

    Trained on the same new data as the predictor. If only this helps, the gain is better
    interpretation of imagined latents, not repaired dynamics.
    """

    def __init__(self, in_dim: int = 192, hidden: int = 128, out_dim: int = 4, n_steps: int = 5):
        super().__init__()
        self.step_emb = torch.nn.Embedding(n_steps + 1, 16)
        self.net = torch.nn.Sequential(torch.nn.Linear(in_dim + 16, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, out_dim))
        self.register_buffer("scale", torch.tensor([20.0, 20.0, 0.2, 0.2]))

    def forward(self, z, step_idx):
        return self.net(torch.cat([z, self.step_emb(step_idx)], -1)) * self.scale


def fit_readout_correction(probe, z_imag: np.ndarray, targets: np.ndarray, *, device="cuda", steps=1500, lr=1e-3, seed=0, batch=256) -> ReadoutCorrection:
    """z_imag (N, K, D) imagined latents; targets (N, K, 4) true block target (x, y, sin, cos)."""
    torch.manual_seed(seed)
    N, K, D = z_imag.shape
    corr = ReadoutCorrection(D, n_steps=K).to(device)
    opt = torch.optim.Adam(corr.parameters(), lr=lr, weight_decay=1e-4)
    z = torch.from_numpy(z_imag.astype(np.float32)).to(device)
    y = torch.from_numpy(targets.astype(np.float32)).to(device)
    steps_idx = torch.arange(1, K + 1, device=device).expand(N, K)
    with torch.no_grad():
        base = probe(z.reshape(-1, D)).reshape(N, K, 4)
    resid = (y - base) / corr.scale
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        i = torch.as_tensor(rng.integers(N, size=batch), device=device)
        pred = corr(z[i], steps_idx[i]) / corr.scale
        loss = (pred - resid[i]).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return corr.eval()
