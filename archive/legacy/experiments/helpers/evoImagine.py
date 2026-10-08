"""Closed-loop imagination in the Walker LeWM: the policy reacts to every imagined latent.

Evolution (docs/evoPlan/README.md) scores linear latent policies on imagined 0.8 s segments, so
the action blocks are not known before a rollout starts: block t is chosen from the latest two
latents, real or imagined. `rollout_core` is `LeWM.rollout` (three-frame context window, one
predictor call per block, keep the last output) with that one change. The FULL action buffer
(B, 2 + K, 60) is re-encoded at every step with the undecided blocks still zero: the action
encoder works per position, and a call of exactly the shape of `LeWM.rollout`'s single call
runs the same kernels, so every decided position is bitwise equal to the reference. Fed a fixed
tape with k = 0, `rollout_tapes` is therefore `torch.equal` to `WalkerImaginer.rollout` (tested
with a tiny model on CPU and CUDA and with the real model on CUDA), which is what catches errors
in the action window. Every module must stay in eval mode: dropout or BatchNorm batch statistics
would make rows depend on chance or on each other. Each call checks all submodules, since
`model.training` alone misses a predictor put back in train mode.

Exactness holds per configuration. Kernels are picked by batch shape (on CUDA, and on CPU with
many threads), so results depend on the chunk shapes: on `chunk` for `rollout_tapes` (as for
`WalkerImaginer.rollout`) and on (P, R, n, max_batch) for `rollout_policies`. With the real
model on CUDA (24 dev roots, P = 16), changing `max_batch` or P moved readouts by up to about
3e-4 at k = 0 and 4e-3 at k = 1, n = 4: last-bit kernel differences compound through the
predictor layers and the closed loop, so a segment that close to the safety boundary can flip
its violation flag. A fixed configuration is bitwise repeatable on the same machine and
software stack, also across processes, which is what resumed runs need. Keep `max_batch` fixed
and record it per study, and do not compare imagined numbers across population sizes or
`max_batch` values (re-scoring one candidate alone is another population size) at a tighter
tolerance than that.

Noise: each predicted latent becomes `z + k * sigma * eps` before anything reads it, so the
noisy latent is what the policy sees next and what enters later context windows. `sigma` is
the per-dimension std of one-step teacher-forced residuals (`one_step_residuals`,
`sigma_from_residuals`); `open_loop_errors` records the multi-step error growth for reference.
Populations use common random numbers: one eps (R, n, K, D) per call from a seeded generator,
shared by every candidate, so candidates differ only through their parameters.

    im = ClosedLoopImaginer(model, scaler, probe, device="cuda", sigma=sigma)
    out = im.rollout_policies(policy, theta, z_hist, hist_blocks, n_samples=4, k=1.0, seed=7)
    out["readout"]          # (P, R, n, K, 3): probe height, pitch, speed at every block end
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields

import numpy as np
import torch

from helpers.evoPolicy import FEATURE_DIM
from helpers.walkerLewm import WalkerImaginer
from helpers.walkerRules import HISTORY, HORIZON_BLOCKS

N_HIST = HISTORY - 1  # action blocks between the three context frames


@dataclass
class Counters:
    """Cost accounting shared by the imagined and real paths (evoReal and evoRun import it)."""

    imagined_rows: int = 0  # predictor rows: batch rows x predicted blocks
    real_steps: int = 0  # environment steps actually simulated
    renders: int = 0
    encodes: int = 0  # frames through the image encoder
    wall_s: float = 0.0

    def add(self, other) -> None:
        """Add another `Counters` (or a dict holding some of the fields) in place."""
        vals = other.as_dict() if isinstance(other, Counters) else dict(other)
        unknown = set(vals) - {f.name for f in fields(self)}
        if unknown:
            raise ValueError(f"unknown counters: {sorted(unknown)}")
        for name, v in vals.items():
            setattr(self, name, getattr(self, name) + (float(v) if name == "wall_s" else int(v)))

    def as_dict(self) -> dict:
        return asdict(self)


def _check_model(model) -> None:
    if any(m.training for m in model.modules()):  # model.training alone misses a submodule put back in train mode
        raise ValueError("every LeWM module must be in eval mode (dropout and BatchNorm batch statistics break exactness)")
    if getattr(model.predictor, "num_frames", HISTORY) != HISTORY:
        raise ValueError(f"expected a predictor with a {HISTORY}-frame window")


def _tape_fn(blocks: torch.Tensor):
    """action_fn that replays fixed normalised blocks (B, K, 60)."""

    def fn(t, z_t, z_prev):
        return blocks[:, t]

    return fn


def _repeat(x: torch.Tensor, c: int) -> torch.Tensor:
    """(M, ...) -> (c * M, ...), the M rows repeated for each of c candidates (row p * M + m)."""
    return x.unsqueeze(0).expand(c, *x.shape).reshape(c * x.shape[0], *x.shape[1:])


def _rollout(model, z_hist, hist_norm, action_fn, n_blocks: int, k: float = 0.0, sigma=None, eps=None):
    """`LeWM.rollout` with block t chosen by `action_fn` from the latest two latents (module doc)."""
    _check_model(model)
    B = z_hist.shape[0]
    emb = list(z_hist.unbind(1))
    buf = torch.zeros((B, N_HIST + n_blocks, hist_norm.shape[-1]), dtype=torch.float32, device=z_hist.device)
    buf[:, :N_HIST] = hist_norm
    for t in range(n_blocks):
        blk = action_fn(t, emb[-1], emb[-2])
        if tuple(blk.shape) != (B, buf.shape[-1]):
            raise ValueError(f"action_fn must return ({B}, {buf.shape[-1]}), got {tuple(blk.shape)}")
        buf[:, N_HIST + t] = blk
        ae = model.action_encoder(buf)  # full buffer: the shape of LeWM.rollout's single call
        z = model.predict(torch.stack(emb[-HISTORY:], 1), ae[:, t : t + HISTORY])[:, -1]
        if k > 0:
            z = z + k * sigma * eps[:, t]
        emb.append(z)
    return torch.stack(emb[HISTORY:], 1), buf[:, N_HIST:]


class ClosedLoopImaginer:
    """Batched closed-loop rollouts with exact `WalkerImaginer` encoding and block normalisation."""

    def __init__(self, model, scaler, probe, *, device: str = "cuda", sigma=None):
        self.device = device
        self._ref = WalkerImaginer(model, scaler, device)  # puts the model on `device` in eval mode
        self.model = self._ref.model
        _check_model(self.model)
        self.mean, self.std = self._ref.mean, self._ref.std
        self.probe = probe.to(device).eval() if isinstance(probe, torch.nn.Module) else probe
        self.counters = Counters()
        self.sigma = sigma
        self._depth = 0

    @property
    def sigma(self) -> torch.Tensor | None:
        """Per-dimension noise scale (D,) float32 on the device, or None (k must then be 0)."""
        return self._sigma

    @sigma.setter
    def sigma(self, value) -> None:
        if value is None:
            self._sigma = None
            return
        s = value.detach() if isinstance(value, torch.Tensor) else np.asarray(value)
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device).clone()
        if s.ndim != 1 or not bool(torch.isfinite(s).all()) or bool((s < 0).any()):
            raise ValueError("sigma must be a finite nonnegative (D,) vector")
        self._sigma = s

    @contextmanager
    def _clock(self):
        """Adds the wall time of the outermost public call only, so nested calls count once."""
        self._depth += 1
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._depth -= 1
            if self._depth == 0:
                if torch.device(self.device).type == "cuda":
                    torch.cuda.synchronize(self.device)
                self.counters.wall_s += time.perf_counter() - t0

    def flat(self, blocks) -> torch.Tensor:
        """Raw blocks (..., 10, 6) -> normalised (..., 60), exactly `WalkerImaginer._flat`."""
        return self._ref._flat(blocks)

    def encode(self, frames) -> torch.Tensor:
        """uint8 frames (N, H, W, 3) -> latents (N, D), exactly `WalkerImaginer.encode`."""
        _check_model(self.model)  # the projector's BatchNorm in train mode would mix frames (and update its stats)
        with self._clock():
            z = self._ref.encode(frames)
        self.counters.encodes += int(z.shape[0])
        return z

    def rollout_core(self, z_hist, hist_norm, action_fn, *, n_blocks: int = HORIZON_BLOCKS, k: float = 0.0, eps=None):
        """z_hist (B, 3, D) real latents at t-20, t-10, t; hist_norm (B, 2, 60) normalised history
        blocks; `action_fn(t, z_t, z_prev) -> (B, 60)` normalised block t from the latest and the
        previous latent (B, D). With k > 0, eps (B, K, D) standard normal is required. Returns the
        imagined latents (B, K, D) and the normalised blocks used (B, K, 60), float32."""
        k, n_blocks = float(k), int(n_blocks)
        if not (math.isfinite(k) and k >= 0) or n_blocks < 1:
            raise ValueError("need a finite k >= 0 and n_blocks >= 1")
        with self._clock(), torch.inference_mode():
            z_hist = torch.as_tensor(z_hist, dtype=torch.float32, device=self.device)
            hist_norm = torch.as_tensor(hist_norm, dtype=torch.float32, device=self.device)
            B, D = z_hist.shape[0], z_hist.shape[-1]
            if z_hist.shape != (B, HISTORY, D) or hist_norm.ndim != 3 or hist_norm.shape[:2] != (B, N_HIST):
                raise ValueError(f"z_hist must be (B, {HISTORY}, D) and hist_norm (B, {N_HIST}, A)")
            if k > 0:
                if self._sigma is None or self._sigma.shape != (D,):
                    raise ValueError(f"k > 0 needs sigma of shape ({D},)")
                if eps is None:
                    raise ValueError("k > 0 needs eps (B, K, D)")
                eps = torch.as_tensor(eps, dtype=torch.float32, device=self.device)
                if eps.shape != (B, n_blocks, D):
                    raise ValueError(f"eps must be {(B, n_blocks, D)}, got {tuple(eps.shape)}")
            z, blocks = _rollout(self.model, z_hist, hist_norm, action_fn, n_blocks, k, self._sigma, eps)
            self.counters.imagined_rows += B * n_blocks
        return z, blocks

    def rollout_tapes(self, z_hist, hist_blocks, tapes, chunk: int = 512) -> torch.Tensor:
        """The contract of `WalkerImaginer.rollout`, chunking included: z_hist (3, D) for ONE root,
        hist_blocks (2, 10, 6) and tapes (S, K, 10, 6) raw -> imagined latents (S, K, D)."""
        with self._clock(), torch.inference_mode():
            z_hist = torch.as_tensor(z_hist, dtype=torch.float32, device=self.device)
            hist, fut = self.flat(hist_blocks), self.flat(tapes)
            if z_hist.ndim != 2 or z_hist.shape[0] != HISTORY or hist.ndim != 2 or hist.shape[0] != N_HIST \
                    or fut.ndim != 3:
                raise ValueError(f"need z_hist ({HISTORY}, D), hist_blocks ({N_HIST}, F, 6) and tapes (S, K, F, 6)")
            S, K = fut.shape[:2]
            out = torch.empty((S, K, z_hist.shape[1]), device=self.device)
            for i in range(0, S, chunk):
                f = fut[i : i + chunk]
                s = f.shape[0]
                zh, hn = z_hist.expand(s, -1, -1), hist.expand(s, -1, -1)
                out[i : i + s] = self.rollout_core(zh, hn, _tape_fn(f), n_blocks=K)[0]
        return out

    def rollout_policies(self, policy, theta, z_hist, hist_blocks, *, n_samples: int = 1, k: float = 0.0, seed: int = 0,
                         n_blocks: int = HORIZON_BLOCKS, max_batch: int = 16384, return_latents: bool = False) -> dict:
        """Closed-loop rollouts of P candidates from R roots with n noise samples each.

        theta (P, 2310) numpy or tensor; z_hist (R, 3, D); hist_blocks (R, 2, 10, 6) raw. Rows are
        laid out (p, r, s) row-major and processed in chunks of whole candidates of at most
        `max_batch` rows. With k > 0, eps (R, n, K, D) is drawn once from
        `torch.Generator(device).manual_seed(seed)` and row (p, r, s) uses eps[r, s]; with k == 0
        nothing is drawn and n_samples must be 1. Returns numpy arrays: `readout` (P, R, n, K, 3)
        float64 (probe at every imagined block end), `actions` (P, R, n, K, 6) float32 raw clipped,
        `root_readout` (R, 3) float64 (probe on z_hist[:, -1]) and, if `return_latents`, `z`
        (P, R, n, K, D) float32.
        """
        if self.probe is None:
            raise ValueError("rollout_policies needs a probe")
        n, K, k = int(n_samples), int(n_blocks), float(k)
        if not (math.isfinite(k) and k >= 0) or n < 1 or (k == 0 and n != 1):
            raise ValueError("need a finite k >= 0 and n_samples >= 1, with n_samples == 1 when k == 0")
        with self._clock(), torch.inference_mode():
            theta = torch.as_tensor(theta, dtype=torch.float32, device=self.device)
            z_hist = torch.as_tensor(z_hist, dtype=torch.float32, device=self.device)
            hist = self.flat(hist_blocks)
            P, R, D = theta.shape[0], z_hist.shape[0], z_hist.shape[-1]
            if theta.ndim != 2 or P < 1 or z_hist.shape != (R, HISTORY, D) or hist.ndim != 3 \
                    or hist.shape[:2] != (R, N_HIST):
                raise ValueError(f"need theta (P, n_params), z_hist (R, {HISTORY}, D) and hist_blocks (R, {N_HIST}, F, 6)")
            pol = (getattr(policy, "action_mean", None), getattr(policy, "action_std", None))
            if pol[0] is not None and not all(torch.equal(a.to(self.device), b) for a, b in zip(pol, (self.mean, self.std))):
                raise ValueError("the policy and the imaginer use different action scalers")  # blocks would not match
            M = R * n
            if M > max_batch:
                raise ValueError(f"one candidate needs {M} rows (roots x samples), more than max_batch={max_batch}")
            per = max_batch // M
            z_rows = z_hist.unsqueeze(1).expand(R, n, HISTORY, D).reshape(M, HISTORY, D)
            h_rows = hist.unsqueeze(1).expand(R, n, N_HIST, hist.shape[-1]).reshape(M, N_HIST, hist.shape[-1])
            e_rows = None
            if k > 0:
                gen = torch.Generator(device=self.device).manual_seed(int(seed))
                e_rows = torch.randn((R, n, K, D), generator=gen, device=self.device, dtype=torch.float32).reshape(M, K, D)
            readout = actions = zs = None
            for p0 in range(0, P, per):
                th = theta[p0 : p0 + per]
                c = th.shape[0]
                taken = []

                def act_fn(t, z_t, z_prev, th=th, c=c, taken=taken):
                    a = policy.act(th, policy.features(z_t, z_prev).view(c, M, FEATURE_DIM))
                    taken.append(a)
                    return policy.model_block(a).reshape(c * M, -1)

                eps = None if e_rows is None else _repeat(e_rows, c)
                z, _ = self.rollout_core(_repeat(z_rows, c), _repeat(h_rows, c), act_fn, n_blocks=K, k=k, eps=eps)
                ro = self.probe(z).double().cpu().numpy().reshape(c, M, K, -1)
                acts = torch.stack(taken, 2).cpu().numpy()  # (c, M, K, 6)
                if readout is None:
                    readout = np.empty((P, *ro.shape[1:]), np.float64)
                    actions = np.empty((P, *acts.shape[1:]), np.float32)
                    zs = np.empty((P, M, K, D), np.float32) if return_latents else None
                readout[p0 : p0 + c], actions[p0 : p0 + c] = ro, acts
                if zs is not None:
                    zs[p0 : p0 + c] = z.cpu().numpy().reshape(c, M, K, D)
            root = self.probe(z_hist[:, -1]).double().cpu().numpy()
        out = {"readout": readout.reshape(P, R, n, K, -1), "actions": actions.reshape(P, R, n, K, -1), "root_readout": root}
        if zs is not None:
            out["z"] = zs.reshape(P, R, n, K, D)
        return out


# --------------------------------------------------------------------------------------
# noise calibration
# --------------------------------------------------------------------------------------
@torch.inference_mode()
def one_step_residuals(model, z_win, a_win, *, batch: int = 4096) -> torch.Tensor:
    """Teacher-forced one-step residuals, predicted minus real z[3]: z_win (N, 4, D) real latents at
    four consecutive block ends, a_win (N, 3, 60) normalised blocks between them -> (N, D)."""
    _check_model(model)
    dev = next(model.parameters()).device
    z_win = torch.as_tensor(z_win, dtype=torch.float32, device=dev)
    a_win = torch.as_tensor(a_win, dtype=torch.float32, device=dev)
    N = z_win.shape[0]
    if N < 1 or z_win.ndim != 3 or z_win.shape[1] != HISTORY + 1 or a_win.ndim != 3 \
            or a_win.shape[:2] != (N, HISTORY):
        raise ValueError(f"need z_win (N, {HISTORY + 1}, D) and a_win (N, {HISTORY}, A) with N >= 1")
    out = []
    for i in range(0, N, batch):
        z, a = z_win[i : i + batch], a_win[i : i + batch]
        out.append(model.predict(z[:, :HISTORY], model.action_encoder(a))[:, -1] - z[:, HISTORY])
    return torch.cat(out)


def sigma_from_residuals(res) -> np.ndarray:
    """Per-dimension std (ddof=1) of residuals (..., D) -> (D,) float64."""
    r = res.detach().double().cpu().numpy() if isinstance(res, torch.Tensor) else np.asarray(res, dtype=np.float64)
    r = r.reshape(-1, r.shape[-1])
    if len(r) < 2 or not np.isfinite(r).all():
        raise ValueError("need at least two finite residual rows")
    return r.std(0, ddof=1)


@torch.inference_mode()
def open_loop_errors(model, z_seq, a_seq, n_blocks: int, *, batch: int = 4096) -> np.ndarray:
    """Multi-step error growth: z_seq (N, 3 + K, D) real latents and a_seq (N, 2 + K, 60) normalised
    blocks (longer sequences use their first 3 + K and 2 + K). Rolls the model open loop from the
    three real context latents with the real blocks (the `rollout_core` recursion with a tape) and
    returns the RMS error over samples and dimensions at each horizon, (K,) float64."""
    _check_model(model)
    K = int(n_blocks)
    dev = next(model.parameters()).device
    z_seq = torch.as_tensor(z_seq, dtype=torch.float32, device=dev)
    a_seq = torch.as_tensor(a_seq, dtype=torch.float32, device=dev)
    N = z_seq.shape[0]
    if K < 1 or N < 1 or z_seq.ndim != 3 or z_seq.shape[1] < HISTORY + K or a_seq.ndim != 3 \
            or a_seq.shape[0] != N or a_seq.shape[1] < N_HIST + K:
        raise ValueError(f"need z_seq (N, >= {HISTORY + K}, D) and a_seq (N, >= {N_HIST + K}, A) with N >= 1")
    sq = torch.zeros(K, dtype=torch.float64, device=dev)
    for i in range(0, N, batch):
        z, a = z_seq[i : i + batch], a_seq[i : i + batch]
        pred, _ = _rollout(model, z[:, :HISTORY], a[:, :N_HIST], _tape_fn(a[:, N_HIST:]), K)
        sq += (pred.double() - z[:, HISTORY : HISTORY + K].double()).square().sum((0, 2))
    return torch.sqrt(sq / (N * z_seq.shape[-1])).cpu().numpy()
