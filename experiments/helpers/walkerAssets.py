"""Walker2d behaviour policies exported from OmniSafe, loaded into the vendored env.

Only policy weights cross the OmniSafe boundary (AGENTS.md). An export is a plain dict:
`hidden_sizes`, `activation`, `state_dict` (the actor's mean MLP), `obs_mean`, `obs_var`,
`algo`, `epoch`, written by `walker_s0_export_policies.py` (OmniSafe venv) and consumed
here with torch.nn only. Measured return, speed and fall rate come from the vendored
`SafetyWalker2dVelocity-v1`, never from OmniSafe's own logs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from helpers.locoPolicies import TorchActorPolicy

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_DIR = REPO_ROOT / "data" / "study" / "walker2d" / "policies"

_ACT = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU}


def build_mlp(in_dim: int, hidden_sizes, out_dim: int, activation: str = "tanh") -> torch.nn.Sequential:
    layers, d = [], in_dim
    for h in hidden_sizes:
        layers += [torch.nn.Linear(d, h), _ACT[activation]()]
        d = h
    layers.append(torch.nn.Linear(d, out_dim))
    return torch.nn.Sequential(*layers)


def load_exported_actor(path: Path, action_space, *, seed=None, action_noise: float = 0.0, device="cpu") -> TorchActorPolicy:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sd = blob["state_dict"]
    in_dim = sd["0.weight"].shape[1]
    out_dim = [v for k, v in sd.items() if k.endswith("weight")][-1].shape[0]
    net = build_mlp(in_dim, blob["hidden_sizes"], out_dim, blob["activation"])
    net.load_state_dict(sd)
    name = f"{blob['algo']}-e{blob['epoch']}"
    return TorchActorPolicy(action_space, net, seed=seed, obs_mean=blob["obs_mean"], obs_var=blob["obs_var"], action_noise=action_noise, name=name, device=device)


def list_exports(policy_dir: Path = POLICY_DIR) -> list[Path]:
    return sorted(Path(policy_dir).glob("*.pt"))


def evaluate_policy(env, policy, n_episodes: int = 5, seed: int = 0, max_steps: int = 1000) -> dict:
    """Return, mean speed, speed-cost rate and fall rate in the vendored env (five-tuple API)."""
    rets, speeds, costs, falls, lengths = [], [], [], [], []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        policy.reset(seed=seed + ep) if hasattr(policy, "reset") else None
        ret, cost, vs = 0.0, 0.0, []
        for t in range(max_steps):
            obs, r, term, trunc, info = env.step(policy.act(obs))
            ret += float(r)
            cost += float(info.get("cost", 0.0))
            vs.append(float(info.get("x_velocity", np.nan)))
            if term or trunc:
                break
        rets.append(ret)
        speeds.append(float(np.nanmean(vs)))
        costs.append(cost / (t + 1))
        falls.append(bool(term))
        lengths.append(t + 1)
    return {"return_mean": float(np.mean(rets)), "speed_mean": float(np.mean(speeds)), "speed_p90": float(np.percentile(speeds, 90)),
            "cost_rate": float(np.mean(costs)), "fall_rate": float(np.mean(falls)), "len_mean": float(np.mean(lengths)), "n": n_episodes}
