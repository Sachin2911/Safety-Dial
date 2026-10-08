"""Small linear residual over fixed random visual-history features around a frozen ensemble."""
import numpy as np
import torch

from helpers.evoEnsemblePolicy import EnsembleHistoryPolicy


class ResidualHistoryPolicy:
    def __init__(self, base, base_theta, projection):
        self.base, self.device = base, base.device
        self.base_theta = torch.as_tensor(base_theta, dtype=torch.float32, device=self.device)
        self.projection = torch.as_tensor(projection, dtype=torch.float32, device=self.device)
        if self.base_theta.shape != (base.n_params,) or self.projection.ndim != 2 or self.projection.shape[0] != 444:
            raise ValueError('frozen base parameters and a 444-by-rank projection required')
        self.rank = self.projection.shape[1]
        self.n_params = 60*(self.rank+1)+1
        self.action_mean, self.action_std = base.action_mean, base.action_std

    def features(self, z_t, z_prev, previous_actions):
        return self.base.features(z_t, z_prev, previous_actions)

    def residual_features(self, feats):
        # Task-local projection avoids BLAS batch-shape changes in the real executor.
        projected = feats @ self.projection
        return torch.cat([projected, torch.ones_like(projected[..., :1])], -1)

    def residual(self, theta, feats):
        theta = torch.as_tensor(theta, dtype=torch.float32, device=feats.device)
        if theta.ndim != 2 or theta.shape != (feats.shape[0], self.n_params):
            raise ValueError('candidate-major residual parameters required')
        phi = self.residual_features(feats).reshape(theta.shape[0], -1, self.rank+1)
        weights = theta[:, :-1].reshape(theta.shape[0], 60, self.rank+1)
        return torch.einsum('pof,pmf->pmo', weights, phi).reshape(*feats.shape[:-1], 60)

    def act(self, theta, feats):
        theta = torch.as_tensor(theta, dtype=torch.float32, device=feats.device)
        residual = self.residual(theta, feats)
        base = self.base.act(self.base_theta[None].expand(len(theta), -1), feats)
        gain = (1+theta[:, -1]).reshape(len(theta), *([1]*(feats.ndim-1)))
        return (gain*base+residual).clamp(-1, 1)

    def model_block(self, action):
        return self.base.model_block(action)

    def state(self):
        return {**self.base.state(), 'policy_kind': 'residual_history_sequential_block_v1',
                'base_theta': self.base_theta.cpu().numpy(), 'projection': self.projection.cpu().numpy()}

    @classmethod
    def from_state(cls, state, *, device='cpu'):
        if str(state.get('policy_kind', '')) != 'residual_history_sequential_block_v1':
            raise ValueError('not a residual visual-history policy')
        base = EnsembleHistoryPolicy.from_state(dict(state, policy_kind='ensemble_history_sequential_block_v1'), device=device)
        return cls(base, state['base_theta'], state['projection'])


def residual_candidates(policy, calibration_features, *, directions, rms_scales, gains, seed):
    """Paired directions, calibrated in raw action units on selection-root inputs only."""
    rng = np.random.default_rng(seed)
    thetas, meta = [np.zeros(policy.n_params, np.float32)], [dict(kind='baseline')]
    for direction in range(directions):
        raw = rng.standard_normal(policy.n_params).astype(np.float32)
        raw[-1] = 0
        with torch.inference_mode():
            rms = float(policy.residual(raw[None], calibration_features[None]).square().mean().sqrt())
        if not np.isfinite(rms) or rms <= 0:
            raise ValueError('degenerate residual calibration')
        for scale in rms_scales:
            for sign in [-1, 1]:
                thetas.append(raw*(float(scale)*sign/rms))
                meta.append(dict(kind='residual', direction=direction, target_raw_rms=float(scale), sign=sign))
    for gain in gains:
        theta = np.zeros(policy.n_params, np.float32)
        theta[-1] = float(gain)-1
        thetas.append(theta)
        meta.append(dict(kind='attenuation', gain=float(gain)))
    return np.stack(thetas), meta
