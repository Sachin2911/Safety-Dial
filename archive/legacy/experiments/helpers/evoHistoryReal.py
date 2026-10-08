"""Action-history real execution, using the validated sequential physics protocol."""
import numpy as np
import torch
from helpers.evoMlpReal import MLPRealExecutor
from helpers.evoReal import _start, _step
from helpers.walkerRules import HISTORY, HORIZON_BLOCKS, FRAMESKIP


class HistoryRealExecutor(MLPRealExecutor):
    def _group(self, theta, roots, zh, tasks, out) -> None:
        """One lockstep group: each task restored into its own env, then per block act on
        (z_t, z_prev), execute 10 sequential actions, render every block-end state and encode them."""
        envs = self._physics(len(tasks))
        qp, qv, xv = out["qpos"], out["qvel"], out["x_velocity"]
        for i, (env, r) in enumerate(zip(envs, tasks[:, 1])):
            _start(env, roots[r], qp[i], qv[i])
        rows = torch.as_tensor(tasks[:, 1], device=self.device)
        th = [theta[[p]] for p in tasks[:, 0].tolist()]  # a fresh (1, n_params) tensor per task
        z_t, z_prev, zs = zh[rows, HISTORY - 1], zh[rows, HISTORY - 2], []
        past = torch.as_tensor(np.stack([roots[r].history_actions[-1].reshape(60) for r in tasks[:, 1]]), dtype=torch.float32, device=self.device)
        for b in range(HORIZON_BLOCKS):
            acts = [self.policy.act(t, self.policy.features(z_t[i:i+1], z_prev[i:i+1], past[i:i+1])[:, None]) for i, t in enumerate(th)]
            a = torch.cat(acts)[:, 0].double().cpu().numpy()
            past = torch.as_tensor(a, dtype=torch.float32, device=self.device)
            out["actions"][:, b] = a
            blocks = a.reshape(len(tasks), FRAMESKIP, 6)
            for i, env in enumerate(envs):
                for t in range(b * FRAMESKIP, (b + 1) * FRAMESKIP):
                    _step(env.unwrapped, blocks[i, t - b * FRAMESKIP], qp[i], qv[i], xv[i], t)
            end = (b + 1) * FRAMESKIP
            z_prev, z_t = z_t, self._encode_ends(self._render(qp[:, end], qv[:, end]))
            zs.append(z_t)
        z = torch.stack(zs, 1)
        out["z"][:] = z.cpu().numpy()
        out["readout"][:] = np.stack([self.probe.predict(seg) for seg in z])  # one segment (10, D) per call
