"""Frozen-gain residual search, independent nominee selection and pilot uncertainty."""
import math

import numpy as np
import torch
from scipy.stats import norm


class FixedGainResidualPolicy:
    """Search 60x17 preconditioned residual weights; retain base gain exactly one."""
    def __init__(self, residual_policy, feature_rms):
        self.residual_policy = residual_policy
        self.device = residual_policy.device
        self.feature_rms = torch.as_tensor(feature_rms, dtype=torch.float32, device=self.device)
        if self.feature_rms.shape != (residual_policy.rank+1,) or not torch.isfinite(self.feature_rms).all() or (self.feature_rms <= 0).any():
            raise ValueError('finite positive residual-feature scales required')
        self.n_params = residual_policy.n_params-1
        self.action_mean, self.action_std = residual_policy.action_mean, residual_policy.action_std

    def features(self, z_t, z_prev, previous_actions):
        return self.residual_policy.features(z_t, z_prev, previous_actions)

    def raw_theta(self, theta):
        t = torch.as_tensor(theta, dtype=torch.float32, device=self.device)
        if t.ndim != 2 or t.shape[1] != self.n_params or not torch.isfinite(t).all():
            raise ValueError('finite candidate-major residual weights required')
        raw = (t.reshape(len(t), 60, -1)/self.feature_rms).reshape(len(t), -1)
        return torch.cat([raw, torch.zeros((len(t), 1), device=self.device)], dim=1)

    def act(self, theta, features):
        return self.residual_policy.act(self.raw_theta(theta), features)

    def model_block(self, actions):
        return self.residual_policy.model_block(actions)


def task_returns(metrics):
    """Same first-violation truncation as ROSARL, with no additional unsafe penalty."""
    v, full, pre = [np.asarray(metrics[k]) for k in ['violated', 'ret', 'ret_pre']]
    if v.shape != full.shape or v.shape != pre.shape or not np.isfinite(full).all() or not np.isfinite(pre).all():
        raise ValueError('finite matching segment metrics required')
    return np.where(v, pre, full)


class NomineeSelector:
    """Select only from fixed-bank outcomes, never compare generation fitness values."""
    def __init__(self):
        self.theta, self.scores = [], []

    def add(self, theta, metrics):
        u = task_returns(metrics).reshape(-1)
        self.theta.append(np.asarray(theta, np.float64).copy())
        self.scores.append(math.fsum(u.tolist())/len(u))

    def best(self, generation):
        if not 0 <= generation < len(self.theta):
            raise ValueError('one baseline and one nominee per completed generation required')
        j = int(np.argmax(self.scores[:generation+1]))
        return j, self.theta[j].copy()


def pilot_uncertainty(values, *, seed=20261220, n_boot=2000):
    """Crossed seed/episode resampling and exploratory random-effects variance components.

    Four search seeds only support a pilot estimate, not a final power guarantee.
    """
    x = np.asarray(values, np.float64)
    if x.ndim != 2 or min(x.shape) < 2 or not np.isfinite(x).all():
        raise ValueError('finite seed-by-episode contrast with both axes >=2 required')
    s, r = x.shape
    grand, sm, rm = x.mean(), x.mean(1), x.mean(0)
    error = x-sm[:, None]-rm[None, :]+grand
    interaction = float(np.square(error).sum()/((s-1)*(r-1)))
    vs = max(0., float(np.var(sm, ddof=1)-interaction/r))
    vr = max(0., float(np.var(rm, ddof=1)-interaction/s))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        si, ri = rng.integers(s, size=s), rng.integers(r, size=r)
        draws.append(x[si[:, None], ri[None, :]].mean())
    scenarios = []
    for ns in [10, 20, 40]:
        for nr in [320, 640, 1280]:
            variance = vs/ns+vr/nr+interaction/(ns*nr)
            if variance == 0:
                power = None
            else:
                effect_z = .1/np.sqrt(variance)
                critical = norm.ppf(1-.025/2)
                power = float(norm.sf(critical-effect_z)+norm.cdf(-critical-effect_z))
            scenarios.append(dict(search_seeds=ns, evaluation_episodes=nr,
                nominal_alpha=.025, absolute_effect=.1, plug_in_power=power))
    return dict(point=float(grand), lo=float(np.percentile(draws, 2.5)), hi=float(np.percentile(draws, 97.5)),
        search_seeds=s, independent_episodes=r, seed_means=sm.tolist(), n_boot=n_boot,
        variance_components=dict(seed=vs, episode=vr, interaction=interaction), planning_scenarios=scenarios,
        warning='Development pilot only; few seeds, truncated variance estimates and no ROSARL-arm variance. Not final study power.')
