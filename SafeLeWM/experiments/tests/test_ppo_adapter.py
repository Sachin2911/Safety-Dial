"""Rented-host PPO gradient, likelihood, recurrence and accounting checks."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "experiments/scripts")]
from helpers.mechanism import SearchMap, projection_matrix  # noqa: E402
from helpers.ppo_adapter import (  # noqa: E402
    CoordinatePolicy,
    advantages,
    terminal_utility,
    prepare_episodes,
    update,
)
from helpers.recurrent_adapter import AdapterSpec, RecurrentActionAdapter  # noqa: E402


def make_policy(rep):
    actor = RecurrentActionAdapter(AdapterSpec(396, 2, initialization_seed=12101))
    mapping = SearchMap(
        actor.parameters_vector(),
        actor.coordinate_scales(),
        rep,
        projection_matrix(12105) if rep == "subspace" else None,
    )
    return CoordinatePolicy(mapping, 12106), actor


@pytest.mark.parametrize("rep", ["full", "head", "subspace"])
def test_deterministic_export_gradient_and_recurrent_likelihood(rep):
    torch.set_num_threads(4)
    policy, actor = make_policy(rep)
    x = torch.randn(13, 396, generator=torch.Generator().manual_seed(1))
    base = torch.tensor([[0.9, -0.9]])
    state, native_state = policy.initial_state(), actor.initial_state()
    with torch.no_grad():
        policy.coordinates.add_(
            torch.randn(policy.coordinates.shape, generator=torch.Generator().manual_seed(2)) * 0.1
        )
        actor.load_vector(policy.export())
        for row in x:
            distribution, state, _ = policy.distribution(row[None], state)
            logits = distribution.loc
            expected = ((1 + logits[:, :2].tanh()) * base + logits[:, 2:].tanh()).clamp(-1, 1)
            action, native_state, _ = actor(row[None], native_state, base)
            torch.testing.assert_close(expected, action, rtol=0, atol=0)
            torch.testing.assert_close(state, native_state, rtol=0, atol=0)
        state = policy.initial_state()
        raw, logp, values = [], [], []
        generator = torch.Generator().manual_seed(4)
        for row in x:
            _, state, detail = policy.sample(row[None], state, base, generator)
            raw.append(detail["raw_logits"][0])
            logp.append(detail["log_probability"][0])
            values.append(detail["values"][0])
    raw, old = torch.stack(raw), torch.stack(logp)
    actual, actual_values = policy.sequence(x, raw)
    torch.testing.assert_close(actual, old, rtol=0, atol=0)
    torch.testing.assert_close(actual_values, torch.stack(values), rtol=0, atol=0)
    actual.sum().backward()
    assert policy.coordinates.grad is not None and torch.isfinite(policy.coordinates.grad).all()
    assert policy.coordinates.grad.abs().sum() > 0
    assert policy.log_std.grad.abs().sum() > 0
    assert all(p.grad is None for p in policy.actor.parameters())


def test_terminal_utility_and_gae_do_not_reward_inactivity():
    assert terminal_utility(True, False) == 1
    assert terminal_utility(True, True) == 8 / 90
    assert terminal_utility(False, True) == -1 / 90
    assert terminal_utility(False, False) == 0
    advantage, returns = advantages(np.zeros(3), 1, gae_lambda=1)
    np.testing.assert_array_equal(advantage, np.ones(3))
    np.testing.assert_array_equal(returns, np.ones(3))
    a, b = advantages(np.array([0.1, 0.2, 0.3]), 0, gae_lambda=1)
    np.testing.assert_allclose(b, 0, atol=1e-7)
    np.testing.assert_allclose(a, [-0.1, -0.2, -0.3], atol=1e-7)


def test_clipped_actions_are_not_used_as_likelihood_observations():
    policy, _ = make_policy("head")
    with torch.no_grad():
        policy.coordinates[-4:] = 100
        _, _, detail = policy.sample(
            torch.zeros(1, 396),
            policy.initial_state(),
            torch.ones(1, 2),
            torch.Generator().manual_seed(1),
        )
        raw = detail["raw_logits"]
        assert raw.max() > 1
        d, _, _ = policy.distribution(torch.zeros(1, 396), policy.initial_state())
        torch.testing.assert_close(detail["log_probability"], d.log_prob(raw).sum(-1))
        assert not torch.allclose(detail["log_probability"], d.log_prob(raw.clamp(-1, 1)).sum(-1))


def test_partial_trajectories_excluded_and_update_changes_parameters(tmp_path):
    policy, _ = make_policy("head")
    rows = []
    for i in range(3):
        folder = tmp_path / "episodes" / str(i)
        folder.mkdir(parents=True)
        x = torch.randn(7, 396, generator=torch.Generator().manual_seed(i))
        state = policy.initial_state()
        raw = []
        logp = []
        values = []
        with torch.no_grad():
            for row in x:
                _, state, detail = policy.sample(
                    row[None], state, torch.zeros(1, 2), torch.Generator().manual_seed(i)
                )
                raw.append(detail["raw_logits"][0].numpy())
                logp.append(detail["log_probability"][0].numpy())
                values.append(detail["values"][0].numpy())
        np.savez(
            folder / "trace.npz",
            inputs=x.numpy(),
            raw_logits=raw,
            log_probability=logp,
            values=values,
        )
        rows.append(
            dict(
                id=str(i),
                status="budget_exhausted" if i == 2 else "complete",
                success=i == 0,
                violation=i == 1,
            )
        )
    episodes = prepare_episodes(rows, tmp_path)
    assert [e["id"] for e in episodes] == ["0", "1"]
    previous = policy.export()
    optimizer = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=3e-4)
    logs = update(policy, optimizer, episodes, np.random.default_rng(42))
    assert len(logs) == 4 and not np.array_equal(previous, policy.export())
    assert all(np.isfinite(r["loss"]) for r in logs)


def test_actual_ppo_scheduler_exact_quota_and_partial_selection(tmp_path, monkeypatch):
    import run_matched_ppo as runner
    from helpers.replication_budget import episode_status

    monkeypatch.setattr(runner, "OUT", tmp_path)
    monkeypatch.setattr(
        runner,
        "prepare_episodes",
        lambda rows, _: [dict(id=r["id"]) for r in rows if r["status"] == "complete"],
    )
    monkeypatch.setattr(runner, "update", lambda *args, **kwargs: [dict(loss=0)])
    (tmp_path / "candidates").mkdir()
    spent = 0
    recorded = []

    class Pool:
        def episode(self, spec, parameters=None, limit=200, **kwargs):
            nonlocal spent
            n = min(limit, 137)
            row = dict(
                id=spec["id"],
                status=episode_status(n, n == 137, limit),
                success=n == 137,
                safe_completion=n == 137,
                violation=False,
                resources=dict(environment_actions=n),
            )
            spent += n
            assert spent <= 40000
            return row

        def batch(self, jobs):
            results = {}
            for job in jobs:
                quota = job["action_quota"]
                rows = []
                for spec in job["episodes"]:
                    if quota == 0:
                        break
                    row = self.episode(spec, limit=min(200, quota))
                    quota -= row["resources"]["environment_actions"]
                    rows.append(row)
                results[job["id"]] = rows
            return results

    actor = RecurrentActionAdapter(AdapterSpec(396, 2))
    mapping = SearchMap(actor.parameters_vector(), actor.coordinate_scales(), "head")
    summary = dict(candidates=[], searches={})
    best = runner.search_ppo(
        Pool(),
        dict(id="R0", ppo_seed=3, shuffle_seed=4),
        mapping,
        dict(
            training_actions_per_search=40000, rollout_actions=3200, learning_rate=0.0003, epochs=4
        ),
        dict(training=[dict(id=str(i)) for i in range(8)], training_seed=8201),
        summary,
        recorded.extend,
    )
    assert spent == sum(r["resources"]["environment_actions"] for r in recorded) == 40000
    assert best["eligible"]
    assert best == max(
        (c for c in summary["candidates"] if c["eligible"]), key=lambda c: c["score"]
    )
    assert any(r["status"] == "budget_exhausted" for r in recorded)
    assert sum(s["actions"] for s in summary["searches"].values()) == 40000
