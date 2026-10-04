"""Full-context imagination, adapted from evoHistoryImagine at 63ca137.

The validated rollout core and candidate/noise layout stay unchanged; the action
callback also receives the two preceding blocks and older latent, initially from root history.
"""
import math
import numpy as np
import torch
from helpers.evoImagine import ClosedLoopImaginer, _repeat, N_HIST
from helpers.evoContextPolicy import CONTEXT_FEATURE_DIM
from helpers.walkerRules import HISTORY, HORIZON_BLOCKS


class ContextImaginer(ClosedLoopImaginer):
    def rollout_policies(self, policy, theta, z_hist, hist_blocks, *, n_samples: int = 1, k: float = 0.0, seed: int = 0,
                         n_blocks: int = HORIZON_BLOCKS, max_batch: int = 16384, return_latents: bool = False) -> dict:
        """Closed-loop rollouts of P candidates from R roots with n noise samples each.

        theta (P, n_params) numpy or tensor; z_hist (R, 3, D); hist_blocks (R, 2, 10, 6) raw. Rows are
        laid out (p, r, s) row-major and processed in chunks of whole candidates of at most
        `max_batch` rows. With k > 0, eps (R, n, K, D) is drawn once from
        `torch.Generator(device).manual_seed(seed)` and row (p, r, s) uses eps[r, s]; with k == 0
        nothing is drawn and n_samples must be 1. Returns numpy arrays: `readout` (P, R, n, K, 3)
        float64 (probe at every imagined block end), `actions` (P, R, n, K, 60) float32 raw clipped,
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
            raw_rows = torch.as_tensor(hist_blocks, dtype=torch.float32, device=self.device).reshape(R, 120).repeat_interleave(n, dim=0)
            e_rows = None
            if k > 0:
                gen = torch.Generator(device=self.device).manual_seed(int(seed))
                e_rows = torch.randn((R, n, K, D), generator=gen, device=self.device, dtype=torch.float32).reshape(M, K, D)
            readout = actions = zs = None
            for p0 in range(0, P, per):
                th = theta[p0 : p0 + per]
                c = th.shape[0]
                taken = []
                past = _repeat(raw_rows, c)
                older = _repeat(z_rows[:, 0], c)

                def act_fn(t, z_t, z_prev, th=th, c=c, taken=taken):
                    nonlocal past, older
                    a = policy.act(th, policy.features(z_t, z_prev, older, past).view(c, M, CONTEXT_FEATURE_DIM))
                    past = torch.cat([past[:, 60:], a.reshape(c * M, 60)], -1)
                    older = z_prev
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
