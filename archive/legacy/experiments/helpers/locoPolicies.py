"""Uniform policy adapters for the locomotion velocity tasks.

Every policy exposes the same `act(obs) -> np.ndarray` contract, so the collector, Gate 0 and
the filtered-evaluation loop never branch on which kind of policy they were handed.

`policy_id` values are written into the dataset as a column and are the basis of the Stage 3
dataset-composition check, so they are fixed here rather than assigned at each call site.
"""

from __future__ import annotations

import numpy as np

#: Stable integer codes written to the `policy_id` dataset column. Append only.
POLICY_IDS = {
    "random": 0,
    "scripted_forward": 1,
    "sb3_ppo": 2,
    "omnisafe_ppolag": 3,
    "omnisafe_cpo": 4,
    "zero": 5,
}


class BasePolicy:
    """`act(obs) -> action`, plus a stable id and name for provenance."""

    name = "base"

    def __init__(self, action_space, seed: int | None = None):
        self.action_space = action_space
        self.rng = np.random.default_rng(seed)
        self.low = np.asarray(action_space.low, dtype=np.float32)
        self.high = np.asarray(action_space.high, dtype=np.float32)
        self.dim = int(action_space.shape[0])

    @property
    def policy_id(self) -> int:
        return POLICY_IDS[self.name]

    def reset(self) -> None:
        """Called at the start of each episode. Stateful policies override."""

    def act(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class RandomActionPolicy(BasePolicy):
    """Uniform over the action box. Falls almost immediately, which is the point."""

    name = "random"

    def act(self, obs):
        return self.rng.uniform(self.low, self.high).astype(np.float32)


class ZeroActionPolicy(BasePolicy):
    """Zero torque. The limp-collapse baseline."""

    name = "zero"

    def act(self, obs):
        return np.zeros(self.dim, dtype=np.float32)


class ScriptedForwardPolicy(BasePolicy):
    """Maximum forward torque, optionally low-pass filtered.

    The proposal's "full forward" probe policy. With `smooth=0` this is the literal
    max-torque drive; with `smooth>0` it becomes a crude but less degenerate gait driver, which
    is what the Phase 0 triage wants: it needs falls from something other than pure flailing.
    """

    name = "scripted_forward"

    def __init__(self, action_space, seed=None, *, smooth: float = 0.0, noise: float = 0.0):
        super().__init__(action_space, seed)
        self.smooth = float(smooth)
        self.noise = float(noise)
        self._prev = np.zeros(self.dim, dtype=np.float32)

    def reset(self):
        self._prev = np.zeros(self.dim, dtype=np.float32)

    def act(self, obs):
        target = self.high.copy()
        if self.noise:
            target = target + self.rng.normal(0.0, self.noise, self.dim).astype(np.float32)
        action = self.smooth * self._prev + (1.0 - self.smooth) * target
        action = np.clip(action, self.low, self.high).astype(np.float32)
        self._prev = action
        return action


class MixedPolicy(BasePolicy):
    """Pick one member per episode, by weight. Used for the Phase 0 triage collection.

    Reports the *selected* member's `policy_id`, so provenance survives the mixing.
    """

    name = "random"  # overridden per episode via `policy_id`

    def __init__(self, members, weights=None, seed=None):
        self.members = list(members)
        n = len(self.members)
        self.weights = np.ones(n) / n if weights is None else np.asarray(weights, float)
        self.weights = self.weights / self.weights.sum()
        self.rng = np.random.default_rng(seed)
        self.current = self.members[0]

    @property
    def policy_id(self) -> int:
        return self.current.policy_id

    def reset(self):
        idx = self.rng.choice(len(self.members), p=self.weights)
        self.current = self.members[idx]
        self.current.reset()

    def act(self, obs):
        return self.current.act(obs)


class TorchActorPolicy(BasePolicy):
    """A trained actor network, with optional observation normalisation and action noise.

    Covers both the stable-baselines3 and the OmniSafe cases: both reduce to a deterministic
    MLP over a normalised observation. `obs_mean` / `obs_var` come from the training-time
    running normaliser and must be carried with the weights, since a policy evaluated on
    unnormalised observations silently produces garbage rather than failing.

    `action_noise` is the Phase 4 evaluation perturbation, sigma in units of the action range.
    """

    name = "sb3_ppo"

    def __init__(
        self,
        action_space,
        net,
        *,
        seed=None,
        obs_mean=None,
        obs_var=None,
        action_noise: float = 0.0,
        name: str = "sb3_ppo",
        device: str = "cpu",
    ):
        super().__init__(action_space, seed)
        import torch

        self.torch = torch
        self.net = net.to(device).eval()
        self.device = device
        self.name = name
        self.obs_mean = None if obs_mean is None else np.asarray(obs_mean, np.float32)
        self.obs_var = None if obs_var is None else np.asarray(obs_var, np.float32)
        self.action_noise = float(action_noise)

    def act(self, obs):
        x = np.asarray(obs, np.float32)
        if self.obs_mean is not None:
            x = (x - self.obs_mean) / np.sqrt(self.obs_var + 1e-8)
            x = np.clip(x, -10.0, 10.0)
        with self.torch.no_grad():
            t = self.torch.as_tensor(x, device=self.device).unsqueeze(0)
            action = self.net(t).squeeze(0).cpu().numpy().astype(np.float32)
        if self.action_noise:
            action = action + self.rng.normal(0.0, self.action_noise, self.dim).astype(np.float32)
        return np.clip(action, self.low, self.high).astype(np.float32)


def make_policy(spec: str, action_space, seed=None, **kwargs) -> BasePolicy:
    """Build a policy from a short name. Used by the Hydra entry points."""
    table = {
        "random": RandomActionPolicy,
        "zero": ZeroActionPolicy,
        "scripted_forward": ScriptedForwardPolicy,
    }
    if spec not in table:
        raise KeyError(f"unknown policy {spec!r}; have {sorted(table)} plus TorchActorPolicy")
    return table[spec](action_space, seed=seed, **kwargs)
