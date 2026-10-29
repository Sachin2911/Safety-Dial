"""Condition isolation, replay mixture, calibration gates and lifecycle with a fake agent."""

import base64
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from helpers.dsrl11_learner import (
    Dsrl11LearnerBridge,
    Dsrl11LearnerConfig,
    FrozenResidualActors,
    PauseComparison,
    calibration_summary,
    condition_streams,
    load_residual_episode,
    monte_carlo_returns,
    pause_status,
)
from helpers.dsrl11_protocol import CONDITIONS

CONFIG = Dsrl11LearnerConfig(updates_per_transition=5, critic_reduction="mean")
BANK, ONLINE = 1.0, 2.0


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeBackend:
    """Records every batch; critic values follow ``q_value`` for the gate."""

    q_value = 0.4
    initial = 0.0

    def __init__(self, source, config, *, rng_seed):
        import jax

        self.jax = jax
        self.config, self.rng_seed = config, rng_seed
        self.agent = SimpleNamespace(
            _save_dict={"critic": np.full(3, self.initial, np.float32), "actor": np.zeros(2)}
        )
        self.parameter_counts = {"actor": 2, "critic": 3}
        self.batches, self.published, self.calibration_keys = [], 0, []

    def update(self, batch):
        self.batches.append(
            (batch["observations"]["state"][:, 0, 0].copy(), batch["rewards"].copy())
        )
        self.agent._save_dict["critic"] = self.agent._save_dict["critic"] + 1
        return {"critic_loss": 0.1, "actor_loss": -0.2, "temperature": 0.9}

    def calibrate(self, data, key_seed):
        self.calibration_keys.append(key_seed)
        count = len(data["rewards"])
        heads = np.full((10, count), self.q_value) + np.linspace(-0.2, 0.2, 10)[:, None]
        return {
            "qs_recorded": heads,
            "qs_sampled": heads,
            "targets": np.zeros(count),
            "log_probs": np.zeros(count),
            "loc": np.zeros((count, 32)),
            "scale": np.ones((count, 32)),
        }

    def sample(self, pixels, state, seed, index):
        rng = np.random.default_rng([seed, index, self.published])
        return rng.uniform(-1, 1, 32).astype(np.float32)

    def publish(self):
        self.published += 1

    def dumps(self):
        return json.dumps(
            {"critic": self.agent._save_dict["critic"].tolist(), "published": self.published}
        ).encode()

    def loads(self, content):
        saved = json.loads(content)
        self.agent._save_dict["critic"] = np.asarray(saved["critic"], np.float32)
        self.published = saved["published"]

    def actor_dumps(self):
        return json.dumps({"published": self.published}).encode()

    def reference_parity(self, batch, reserve):
        reserve("author_reference")
        reserve("bridge_parity")
        self.update(batch)
        self.update(batch)
        return {"maximum_state_difference": 0.0, "updates_charged": 2}


class FakeActor:
    def __init__(self, source, config, content):
        self.published = json.loads(content)["published"]

    def sample(self, pixels, state, seed, index):
        rng = np.random.default_rng([seed, index, self.published])
        return rng.uniform(-1, 1, 32).astype(np.float32)


def episode(root, job_id, *, version=-1, condition=None, marker=ONLINE, count=3):
    durations = np.array([8] * (count - 1) + [5], np.int32)
    states = np.zeros((count + 1, 8), np.float32)
    states[:, 0] = marker
    arrays = {
        "pixels": np.full((count + 1, 64, 64, 3), 7, np.uint8),
        "states": states,
        "residual": np.linspace(-1, 1, 32 * count, dtype=np.float32).reshape(count, 32),
        "rewards_r0": np.full(count, 0.1, np.float32),
        "rewards_r1": np.full(count, 0.2, np.float32),
        "discounts": (0.9998**durations).astype(np.float32),
        "masks": np.r_[np.ones(count - 1), 0].astype(np.float32),
        "durations": durations,
        "query_steps": np.arange(count, dtype=np.int32) * 8,
        "actor_versions": np.full(count, version, np.int32),
    }
    path = root / "episodes" / f"{job_id}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return {
        "job_id": job_id,
        "path": str(path),
        "sha256": sha(path),
        "actor_version": version,
        "transitions": count,
        "condition": condition,
    }


def bank(root, successes=2):
    episodes = [episode(root, f"B{i:02d}", marker=BANK) for i in range(12)]
    return {"episodes": episodes, "strict_safe_completions": successes}


def online(root, condition, phase):
    first, version = {"warmup": (0, -1), "batch_a": (3, 0), "batch_b": (6, 1)}[phase]
    return {
        "condition": condition,
        "phase": phase,
        "round_id": f"{condition}:{phase}",
        "episodes": [
            episode(root, f"T_{condition}_{k}", version=version, condition=condition)
            for k in range(first, first + 3)
        ],
    }


def learner(tmp_path, config=CONFIG, factory=FakeBackend):
    bridge = Dsrl11LearnerBridge(
        tmp_path / "learner", tmp_path / "source", config, backend_factory=factory
    )
    bridge.initialize("learning")
    return bridge


def run_round(bridge, root, condition, phase):
    appended = bridge.append(online(root, condition, phase))
    trained = bridge.train(
        {
            "condition": condition,
            "round_id": appended["round_id"],
            "expected_new_transitions": appended["update_credit"],
        }
    )
    published = bridge.publish({"condition": condition, "round_id": appended["round_id"]})
    return appended, trained, published


def test_config_admits_only_the_four_frozen_cells():
    for utd in (5, 20):
        for reduction in ("mean", "min"):
            resolved = Dsrl11LearnerConfig(
                updates_per_transition=utd, critic_reduction=reduction
            ).resolved()
            assert resolved["author_kwargs"]["critic_reduction"] == reduction
            assert resolved["changes_from_task7"]["seed"] == [6060, 11060]
    for kwargs in ({"updates_per_transition": 10}, {"critic_reduction": "max"}, {"capacity": 10}):
        with pytest.raises(ValueError):
            Dsrl11LearnerConfig(**kwargs)


def test_condition_streams_are_distinct_named_seeds():
    seeds = [value for c in CONDITIONS for value in condition_streams(c).values()]
    assert len(set(seeds)) == len(seeds)
    with pytest.raises(ValueError):
        condition_streams("r2_prior")


def test_episode_loader_selects_one_reward_and_checks_the_layout(tmp_path):
    descriptor = episode(tmp_path, "B00")
    r0 = load_residual_episode(descriptor, CONFIG, "r0")
    r1 = load_residual_episode(descriptor, CONFIG, "r1")
    assert np.all(r0["rewards"] == np.float32(0.1)) and np.all(r1["rewards"] == np.float32(0.2))
    assert np.array_equal(r0["noise"], r1["noise"]) and r0["noise"].dtype == np.float32
    with pytest.raises(ValueError, match="hash"):
        load_residual_episode({**descriptor, "sha256": "0" * 64}, CONFIG, "r1")
    with pytest.raises(ValueError, match="actor version"):
        load_residual_episode({**descriptor, "actor_version": 0}, CONFIG, "r1")
    with pytest.raises(ValueError, match="r0 or r1"):
        load_residual_episode(descriptor, CONFIG, "r2")
    path = tmp_path / "episodes/B00.npz"
    with np.load(path) as source:
        arrays = {k: source[k].copy() for k in source.files}
    arrays["residual"][0, 0] = 1.5
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="domain"):
        load_residual_episode({**descriptor, "sha256": sha(path)}, CONFIG, "r1")


def test_monte_carlo_returns_follow_recorded_discounts():
    returns = monte_carlo_returns(np.array([1.0, 0.0, 2.0]), np.array([0.5, 0.5, 0.25]))
    np.testing.assert_allclose(returns, [1.0 + 0.5 * 0.0 + 0.25 * 2.0, 1.0, 2.0])


def summary(heads, reduction="mean", terminal=(False, False, True, False)):
    count = heads.shape[1]
    raw = {
        "qs_recorded": heads,
        "qs_sampled": heads,
        "targets": np.zeros(count),
        "log_probs": np.zeros(count),
        "loc": np.zeros((count, 32)),
        "scale": np.ones((count, 32)),
    }
    data = {"terminal": np.array(terminal), "mc_returns": np.zeros(count)}
    return calibration_summary(raw, data, reduction)


def test_gate_uses_the_learners_reduced_estimate_not_single_heads():
    # Heads scatter widely around 0.4, as a fresh random ensemble does.
    heads = 0.4 + np.linspace(-1.5, 1.5, 10)[:, None] * np.ones((1, 4))
    mean = summary(heads, "mean")
    assert mean["recorded"]["heads"]["fraction_outside_gate"] > 0.3
    assert mean["gate_fractions"]["recorded"] == 0.0 and not mean["gate_failed"]["recorded"]
    minimum = summary(heads, "min")
    assert minimum["gate_fractions"]["recorded"] == 1.0 and minimum["gate_failed"]["recorded"]
    assert minimum["gate_quantity"] == "min of ten critic heads"


def test_nonfinite_calibration_fails_closed_and_stays_json_safe():
    heads = np.full((10, 4), 0.2)
    heads[3, 1] = np.nan
    result = summary(heads)
    assert not result["finite"] and result["gate_fractions"]["recorded"] is None
    assert result["gate_failed"] == {"recorded": True, "sampled": True}
    status = pause_status([{**result, "scheduled": True}])
    assert status["pause"] and status["nonfinite"]


def test_pause_series_excludes_the_initialization_baseline():
    bad = {"gate_fractions": {"recorded": 0.5, "sampled": 0.0}, "finite": True}
    good = {"gate_fractions": {"recorded": 0.0, "sampled": 0.0}, "finite": True}
    history = [{**bad, "scheduled": False}] + [{**bad, "scheduled": True}] * 2
    assert not pause_status(history)["pause"]
    assert pause_status(history + [{**bad, "scheduled": True}])["fired"]["recorded"]
    assert not pause_status(history + [{**good, "scheduled": True}])["pause"]


def test_full_learning_lifecycle_isolates_conditions(tmp_path):
    bridge = learner(tmp_path)
    loaded = bridge.load_bank(bank(tmp_path))
    assert loaded["bank_transitions"] == {
        "r0_none": 0,
        "r1_none": 0,
        "r0_prior": 36,
        "r1_prior": 36,
    }
    seeds = {name: state.backend.rng_seed for name, state in bridge.conditions.items()}
    assert len(set(seeds.values())) == 4
    for condition in CONDITIONS:
        warm = bridge.append(online(tmp_path, condition, "warmup"))
        assert warm["update_credit"] == 0
        with pytest.raises(ValueError, match="No eligible"):
            bridge.train({"condition": condition, "round_id": warm["round_id"]})
        appended, trained, published = run_round(bridge, tmp_path, condition, "batch_a")
        assert appended["update_credit"] == 9 and trained["updates"] == 45
        assert published["actor_version"] == 1
        _, trained, published = run_round(bridge, tmp_path, condition, "batch_b")
        assert trained["condition_updates"] == 90 and published["actor_version"] == 2
    assert bridge.counts["learning"] == 360
    for condition, state in bridge.conditions.items():
        markers = np.stack([m for m, _ in state.backend.batches])
        rewards = np.stack([r for _, r in state.backend.batches])
        assert markers.shape == (90, 256)
        if state.prior:
            assert np.all(markers[:, :128] == BANK) and np.all(markers[:, 128:] == ONLINE)
        else:
            assert np.all(markers == ONLINE)
        assert np.all(rewards == np.float32(0.2 if state.reward == "r1" else 0.1))
        # Initialization baseline, then scheduled or round-end measurements only.
        calibrations = state.calibrations
        assert calibrations[0]["label"] == "initialization" and not calibrations[0]["scheduled"]
        assert len(set(state.backend.calibration_keys)) == 1
        assert state.online.size == 27 and state.actor_version == 2


def test_bank_gate_order_and_single_load_are_enforced(tmp_path):
    bridge = learner(tmp_path)
    with pytest.raises(ValueError, match="strict-safe"):
        bridge.load_bank(bank(tmp_path, successes=1))
    request = bank(tmp_path)
    shuffled = {**request, "episodes": request["episodes"][::-1]}
    with pytest.raises(ValueError, match="B00 through B11"):
        bridge.load_bank(shuffled)
    assert bridge.load_bank(request) == bridge.load_bank(request)
    with pytest.raises(ValueError, match="different prior bank"):
        bridge.load_bank({**request, "strict_safe_completions": 3})


def test_rounds_follow_frozen_phase_order_versions_and_identity(tmp_path):
    bridge = learner(tmp_path)
    with pytest.raises(ValueError, match="prior bank"):
        bridge.append(online(tmp_path, "r1_prior", "warmup"))
    bridge.load_bank(bank(tmp_path))
    with pytest.raises(ValueError, match="frozen phase order"):
        bridge.append(online(tmp_path, "r1_prior", "batch_a"))
    first = bridge.append(online(tmp_path, "r1_prior", "warmup"))
    assert bridge.append(online(tmp_path, "r1_prior", "warmup")) == first
    request = online(tmp_path, "r1_prior", "batch_a")
    stale = {**request, "episodes": [{**e, "actor_version": 1} for e in request["episodes"]]}
    with pytest.raises(ValueError, match="actor version"):
        bridge.append(stale)
    foreign = online(tmp_path, "r0_prior", "batch_a")
    with pytest.raises(ValueError, match="frozen job order"):
        bridge.append({**request, "episodes": foreign["episodes"]})
    appended = bridge.append(request)
    with pytest.raises(ValueError, match="disagree"):
        bridge.train(
            {
                "condition": "r1_prior",
                "round_id": appended["round_id"],
                "expected_new_transitions": 8,
            }
        )
    with pytest.raises(ValueError, match="unfinished"):
        bridge.publish({"condition": "r1_prior", "round_id": appended["round_id"]})
    with pytest.raises(ValueError, match="credit must finish"):
        bridge.append(online(tmp_path, "r1_prior", "batch_b"))


class InflatingBackend(FakeBackend):
    q_value = 5.0


def test_pause_rule_stops_every_condition(tmp_path):
    config = Dsrl11LearnerConfig(
        updates_per_transition=5, critic_reduction="mean", calibration_interval=10
    )
    bridge = learner(tmp_path, config, InflatingBackend)
    bridge.load_bank(bank(tmp_path))
    bridge.append(online(tmp_path, "r0_none", "warmup"))
    appended = bridge.append(online(tmp_path, "r0_none", "batch_a"))
    with pytest.raises(PauseComparison):
        bridge.train(
            {
                "condition": "r0_none",
                "round_id": appended["round_id"],
                "expected_new_transitions": 9,
            }
        )
    state = bridge.conditions["r0_none"]
    # Baseline plus three scheduled failures (updates 10, 20, 30); then nothing else.
    assert [c["label"] for c in state.calibrations] == [
        "initialization",
        "update10",
        "update20",
        "update30",
    ]
    assert state.updates == 30 and bridge.paused["condition"] == "r0_none"
    assert state.interrupted == {appended["round_id"]: "PauseComparison"}
    with pytest.raises(ValueError, match="paused"):
        bridge.append(online(tmp_path, "r1_prior", "warmup"))
    events = [json.loads(line)["event"] for line in bridge.ledger_path.read_text().splitlines()]
    assert events.count("comparison_paused") == 1 and "train_completed" not in events


def test_identical_initialization_is_required_across_conditions(tmp_path):
    class Diverging(FakeBackend):
        def __init__(self, source, config, *, rng_seed):
            super().__init__(source, config, rng_seed=rng_seed)
            self.agent._save_dict["critic"] = self.agent._save_dict["critic"] + rng_seed % 7

    with pytest.raises(ValueError, match="identical network initialization"):
        learner(tmp_path, factory=Diverging)


def test_checkpoint_round_trip_and_ambiguous_resume_is_refused(tmp_path):
    bridge = learner(tmp_path)
    bridge.load_bank(bank(tmp_path))
    bridge.append(online(tmp_path, "r1_prior", "warmup"))
    run_round(bridge, tmp_path, "r1_prior", "batch_a")
    saved = bridge.save("after-batch-a")
    restored = Dsrl11LearnerBridge(
        tmp_path / "learner", tmp_path / "source", CONFIG, backend_factory=FakeBackend
    )
    status = restored.load(saved["path"])
    state = restored.conditions["r1_prior"]
    assert status["conditions"]["r1_prior"]["actor_version"] == 1
    assert state.online.size == 18 and state.bank.size == 36 and state.updates == 45
    np.testing.assert_array_equal(
        state.backend.agent._save_dict["critic"],
        bridge.conditions["r1_prior"].backend.agent._save_dict["critic"],
    )
    # The restored learner continues the frozen order with its rebuilt calibration set.
    run_round(restored, tmp_path, "r1_prior", "batch_b")
    with pytest.raises(ValueError, match="immutable"):
        restored.save("after-batch-a")
    # An update reserved after the checkpoint makes that checkpoint ambiguous.
    restored.reserve_update("learning", "r1_prior", "orphan")
    fresh = Dsrl11LearnerBridge(
        tmp_path / "learner", tmp_path / "source", CONFIG, backend_factory=FakeBackend
    )
    with pytest.raises(ValueError, match="ambiguous resume"):
        fresh.load(saved["path"])


def test_update_cap_and_stop_file_block_updates(tmp_path):
    config = Dsrl11LearnerConfig(
        updates_per_transition=5, critic_reduction="mean", learning_update_cap=40
    )
    bridge = learner(tmp_path, config)
    bridge.load_bank(bank(tmp_path))
    bridge.append(online(tmp_path, "r0_none", "warmup"))
    appended = bridge.append(online(tmp_path, "r0_none", "batch_a"))
    with pytest.raises(ValueError, match="online update budget"):
        bridge.train(
            {
                "condition": "r0_none",
                "round_id": appended["round_id"],
                "expected_new_transitions": 9,
            }
        )
    (tmp_path / "STOP").write_text("stop")
    with pytest.raises(RuntimeError, match="stop file"):
        bridge.reserve_update("learning", "r0_none", "manual")


def actor_request(condition, version, index=0):
    return {
        "condition": condition,
        "actor_version": version,
        "job_id": f"T_{condition}_3",
        "query_index": index,
        "rng_seed": 99,
        "pixels_b64": base64.b64encode(np.zeros((64, 64, 3), np.uint8).tobytes()).decode(),
        "state": [0.0] * 8,
    }


def test_act_export_and_cpu_actors_share_one_versioned_contract(tmp_path):
    bridge = learner(tmp_path)
    bridge.load_bank(bank(tmp_path))
    with pytest.raises(ValueError, match="version mismatch"):
        bridge.act(actor_request("r1_none", 1))
    first = bridge.act(actor_request("r1_none", 0))
    assert first["condition"] == "r1_none" and len(first["residual"]) == 32
    with pytest.raises(ValueError, match="horizon"):
        bridge.act(actor_request("r1_none", 0, index=250))
    bridge.append(online(tmp_path, "r1_none", "warmup"))
    run_round(bridge, tmp_path, "r1_none", "batch_a")
    run_round(bridge, tmp_path, "r1_none", "batch_b")
    with pytest.raises(ValueError, match="published actor version"):
        bridge.export_actor({"condition": "r1_none", "name": "final", "actor_version": 1})
    exported = bridge.export_actor({"condition": "r1_none", "name": "final", "actor_version": 2})
    live = bridge.act(actor_request("r1_none", 2))
    actors = FrozenResidualActors(
        tmp_path / "cpu", tmp_path / "source", {"r1_none": exported["path"]}, factory=FakeActor
    )
    assert actors.act(actor_request("r1_none", 2)) == live
    with pytest.raises(ValueError, match="Unknown condition or actor version"):
        actors.act(actor_request("r1_none", 1))
    with pytest.raises(ValueError, match="Unknown condition"):
        actors.act(actor_request("r0_none", 2))


def test_engineering_profile_uses_its_own_state_and_charges_parity(tmp_path):
    bridge = Dsrl11LearnerBridge(
        tmp_path / "learner", tmp_path / "source", CONFIG, backend_factory=FakeBackend
    )
    bridge.initialize("engineering")
    episodes = [episode(tmp_path, f"E{i}") for i in range(4)]
    result = bridge.profile({"updates": 6, "episodes": episodes})
    assert result["updates_charged"] == 8 and bridge.counts["engineering"] == 8
    assert result["published_actor_version"] == 1 and result["live_transitions"] == 12
    with pytest.raises(ValueError, match="budget"):
        bridge.profile({"updates": 1_991, "episodes": episodes})
    with pytest.raises(ValueError, match="initialized once"):
        bridge.initialize("learning")
