"""PPO in the same frozen adapter coordinates, with raw-logit action likelihoods."""

import math

import numpy as np
import torch
from torch import nn
from torch.func import functional_call

from helpers.recurrent_adapter import AdapterSpec, RecurrentActionAdapter


class LogitAdapter(RecurrentActionAdapter):
    def forward(self, observations, state):
        state = self.memory(torch.tanh(self.features(observations)), state)
        return self.head(state), state


class CoordinatePolicy(nn.Module):
    def __init__(self, mapping, seed):
        super().__init__()
        self.representation = mapping.representation
        self.register_buffer("neutral", torch.from_numpy(mapping.neutral.copy()))
        self.register_buffer("scales", torch.from_numpy(mapping.scales.copy()))
        self.register_buffer(
            "projection",
            torch.from_numpy(mapping.projection.copy())
            if mapping.representation == "subspace"
            else torch.empty(0),
        )
        self.coordinates = nn.Parameter(torch.zeros(mapping.dimension))
        self.log_std = nn.Parameter(torch.full((4,), math.log(0.5)))
        self.actor = LogitAdapter(AdapterSpec(396, 2))
        self.actor.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.critic = nn.Sequential(nn.Linear(428, 64), nn.Tanh(), nn.Linear(64, 1))
        self.shapes = [(name, p.shape, p.numel()) for name, p in self.actor.named_parameters()]

    def vector(self):
        if self.representation == "full":
            change = self.coordinates
        elif self.representation == "head":
            change = torch.cat((self.coordinates.new_zeros(19172 - 132), self.coordinates))
        else:
            change = self.projection @ self.coordinates
        return self.neutral + self.scales * change

    def parameter_dict(self):
        vector, offset, result = self.vector(), 0, {}
        for name, shape, n in self.shapes:
            result[name] = vector[offset : offset + n].reshape(shape)
            offset += n
        assert offset == 19172
        return result

    def initial_state(self, batch_size=1):
        return self.neutral.new_zeros(batch_size, 32)

    def distribution(self, inputs, hidden, parameters=None):
        parameters = self.parameter_dict() if parameters is None else parameters
        mean, hidden = functional_call(self.actor, parameters, (inputs, hidden))
        distribution = torch.distributions.Normal(mean, self.log_std.clamp(-3, 1).exp())
        value = self.critic(torch.cat((inputs, hidden.detach()), dim=-1)).squeeze(-1)
        return distribution, hidden, value

    def sample(self, inputs, hidden, base, generator):
        distribution, hidden, value = self.distribution(inputs, hidden)
        raw = distribution.loc + distribution.scale * torch.randn(
            distribution.loc.shape, generator=generator, device=inputs.device
        )
        gain, residual = 1 + raw[:, :2].tanh(), raw[:, 2:].tanh()
        proposed = gain * base + residual
        detail = dict(
            gain=gain,
            residual=residual,
            proposed=proposed,
            raw_logits=raw,
            log_probability=distribution.log_prob(raw).sum(-1),
            values=value,
        )
        return proposed.clamp(-1, 1), hidden, detail

    def sequence(self, inputs, raw):
        hidden, parameters = self.initial_state(), self.parameter_dict()
        logps, values = [], []
        for x, action in zip(inputs, raw):
            distribution, hidden, value = self.distribution(x[None], hidden, parameters)
            logps.append(distribution.log_prob(action[None]).sum(-1)[0])
            values.append(value[0])
        return torch.stack(logps), torch.stack(values)

    @torch.no_grad()
    def export(self):
        return self.vector().numpy().copy()


def terminal_utility(success, violation):
    return (81 * (success and not violation) + 9 * success - violation) / 90


def advantages(values, terminal_reward, gae_lambda=0.95):
    """Complete finite-horizon episodes only; no quota-cut pseudo-terminal reward."""
    values = np.asarray(values, dtype=np.float32)
    result = np.empty_like(values)
    tail = 0.0
    for i in range(len(values) - 1, -1, -1):
        next_value = values[i + 1] if i + 1 < len(values) else 0.0
        reward = terminal_reward if i == len(values) - 1 else 0.0
        tail = reward + next_value - values[i] + gae_lambda * tail
        result[i] = tail
    return result, result + values


def prepare_episodes(rows, output):
    episodes = []
    for row in rows:
        if row["status"] != "complete":
            continue
        with np.load(output / "episodes" / row["id"] / "trace.npz") as trace:
            advantage, returns = advantages(
                trace["values"], terminal_utility(row["success"], row["violation"])
            )
            episodes.append(
                dict(
                    id=row["id"],
                    inputs=torch.from_numpy(trace["inputs"].copy()),
                    raw=torch.from_numpy(trace["raw_logits"].copy()),
                    old_logp=torch.from_numpy(trace["log_probability"].copy()),
                    advantage=torch.from_numpy(advantage),
                    returns=torch.from_numpy(returns),
                )
            )
    if episodes:
        all_advantages = torch.cat([e["advantage"] for e in episodes])
        mean, std = all_advantages.mean(), all_advantages.std(unbiased=False)
        for episode in episodes:
            episode["advantage"] = (episode["advantage"] - mean) / (std + 1e-8)
    return episodes


def update(policy, optimizer, episodes, rng, epochs=4):
    """Full recurrent unroll per episode, with fixed old likelihoods and GAE targets."""
    logs = []
    for epoch in range(epochs):
        order = rng.permutation(len(episodes))
        for offset in range(0, len(order), 4):
            terms = []
            for index in order[offset : offset + 4]:
                episode = episodes[index]
                logp, value = policy.sequence(episode["inputs"], episode["raw"])
                ratio = (logp - episode["old_logp"]).exp()
                advantage = episode["advantage"]
                actor_loss = -torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage)
                critic_loss = (value - episode["returns"]).square()
                terms.append(actor_loss + 0.5 * critic_loss)
            loss = torch.cat(terms).mean()
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad], 0.5, error_if_nonfinite=True
            )
            optimizer.step()
            logs.append(dict(epoch=epoch, loss=float(loss.detach()), gradient_norm=float(norm)))
    return logs
