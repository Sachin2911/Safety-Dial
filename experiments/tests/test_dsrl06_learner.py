"""Boundary, credit, publication and resume checks without importing JAX."""

import base64
from collections import namedtuple
import contextlib
import importlib.util
import json
from pathlib import Path
import socket
import sys
import threading
import time
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.dsrl06_learner import (
    AuthorActorBackend,
    AuthorBackend,
    FrozenActorBridge,
    LearnerBridge,
    LearnerConfig,
    ReplayStore,
    actor_seed,
    json_hash,
    load_episode,
    pixels_from_base64,
    prepare_observation,
    sha256,
)


class FakeBackend:
    def __init__(self, source, config):
        self.weight = 0
        self.published = 0
        self.parameter_counts = {"actor": 1, "critic": 1}

    def sample(self, pixels, state, seed, index):
        return np.random.default_rng(seed + index + self.published).uniform(-1, 1, 32)

    def update(self, batch):
        assert set(batch["observations"]) == {"pixels", "state"}
        assert batch["observations"]["pixels"].shape[1:] == (64, 64, 3, 1)
        assert batch["observations"]["state"].shape[1:] == (8, 1)
        assert batch["actions"].shape[1:] == (1, 32)
        self.weight += 1
        return {"critic_loss": 0.5, "q": float(self.weight)}

    def publish(self):
        self.published = self.weight

    def dumps(self):
        return json.dumps({"weight": self.weight, "published": self.published}).encode()

    def actor_dumps(self):
        return json.dumps({"published": self.published}).encode()

    def loads(self, content):
        value = json.loads(content)
        self.weight = value["weight"]
        self.published = value["published"]

    def reference_parity(self, batch, reserve):
        reserve("author_reference")
        reserve("bridge_parity")
        self.update(batch)
        return {"maximum_state_difference": 0.0, "updates_charged": 2}


class FakeActorBackend:
    def __init__(self, source, config, content):
        self.published = json.loads(content)["published"]
        self.parameter_counts = {"actor": 1}

    sample = FakeBackend.sample


def bridge(tmp_path, phase="training", **kwargs):
    config = LearnerConfig(capacity=1000, batch_size=8, **kwargs)
    result = LearnerBridge(
        tmp_path / "learner",
        Path("unused"),
        config=config,
        data_root=tmp_path,
        backend_factory=FakeBackend,
    )
    result.initialize(phase)
    return result


def episode(tmp_path, order, count=100, warmup=True, version=0):
    durations = np.full(count, 8, np.int32)
    values = {
        "pixels": np.full((count + 1, 64, 64, 3), order, np.uint8),
        "states": np.full((count + 1, 8), order / 10, np.float32),
        "noise": np.full((count, 32), 2.0 if warmup else 0.3, np.float32),
        "rewards": np.zeros(count, np.float32),
        "discounts": (0.9998**durations).astype(np.float32),
        "masks": np.r_[np.ones(count - 1), 0].astype(np.float32),
        "durations": durations,
        "query_steps": np.arange(count, dtype=np.int32) * 8,
        "actor_versions": np.full(count, version, np.int64),
    }
    path = tmp_path / f"episode{order}.npz"
    np.savez_compressed(path, **values)
    return {
        "job_id": f"T{order:02}",
        "order_index": order,
        "path": str(path),
        "sha256": sha256(path),
        "actor_version": version,
    }, values


def append_round(learner, tmp_path, start, count=100, warmup=True):
    descriptors = [
        episode(tmp_path, x, count, warmup, learner.actor_version)[0]
        for x in range(start, start + 3)
    ]
    request = {"round_id": str(start // 3), "warmup": warmup, "episodes": descriptors}
    return request, learner.append(request)


def ready_learner(tmp_path):
    learner = bridge(tmp_path)
    append_round(learner, tmp_path, 0)
    append_round(learner, tmp_path, 3)
    assert learner.warmup_complete
    return learner


def act_request(version=0, job="test", query=0):
    pixels = np.zeros((64, 64, 3), np.uint8)
    return {
        "job_id": job,
        "query_index": query,
        "rng_seed": 123,
        "actor_version": version,
        "pixels_b64": base64.b64encode(pixels).decode(),
        "state": list(range(8)),
    }


def test_observation_boundary_is_exact_bilinear_and_excludes_extra_state():
    image = np.random.default_rng(3).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    result, state = prepare_observation(image, np.arange(8))
    expected = np.asarray(Image.fromarray(image).resize((64, 64), Image.Resampling.BILINEAR))
    np.testing.assert_array_equal(result, expected)
    assert state.dtype == np.float32
    with pytest.raises(ValueError, match="eight finite"):
        prepare_observation(image, np.arange(9))
    with pytest.raises(ValueError):
        pixels_from_base64(base64.b64encode(b"wrong length").decode())


def test_episode_validation_rejects_partial_masks_hashes_and_unknown_actor(tmp_path):
    descriptor, values = episode(tmp_path, 0)
    loaded = load_episode(descriptor, LearnerConfig(), warmup=True)
    assert loaded["noise"].max() == 2.0  # Author Gaussian warm-up is not clipped.
    with pytest.raises(ValueError, match="magnitude-1"):
        load_episode(descriptor, LearnerConfig(), warmup=False)
    descriptor["actor_version"] = 5
    with pytest.raises(ValueError, match="actor versions"):
        load_episode(descriptor, LearnerConfig(), warmup=True)
    descriptor["actor_version"] = 0
    values["masks"][-1] = 1
    np.savez_compressed(descriptor["path"], **values)
    with pytest.raises(ValueError, match="hash"):
        load_episode(descriptor, LearnerConfig(), warmup=True)
    descriptor["sha256"] = sha256(descriptor["path"])
    with pytest.raises(ValueError, match="final zero mask"):
        load_episode(descriptor, LearnerConfig(), warmup=True)


def test_replay_preserves_native_state_and_actual_duration(tmp_path):
    descriptor, values = episode(tmp_path, 0, count=2, warmup=False)
    values["durations"][-1] = 3
    values["discounts"][-1] = 0.9998**3
    np.savez_compressed(descriptor["path"], **values)
    descriptor["sha256"] = sha256(descriptor["path"])
    data = load_episode(descriptor, LearnerConfig(), warmup=False)
    replay = ReplayStore(2, 6060)
    replay.append(data)
    batch = replay.sample(2, np.array([0, 1]))
    assert batch["discount"][1] == np.float32(0.9998**3)
    np.testing.assert_array_equal(batch["actions"][:, 0], values["noise"])
    assert batch["masks"].tolist() == [1, 0]
    with pytest.raises(ValueError, match="eviction"):
        replay.append(data)


def test_warmup_has_no_update_credit_and_round_append_is_idempotent(tmp_path):
    learner = bridge(tmp_path)
    request, _ = append_round(learner, tmp_path, 0)
    learner.append(request)
    assert learner.replay.size == 300 and len(learner.episodes) == 3
    assert learner.pending_transitions == 0 and not learner.warmup_complete
    with pytest.raises(ValueError, match="No eligible"):
        learner.train({"round_id": "0"})
    append_round(learner, tmp_path, 3)
    assert learner.warmup_complete and learner.replay.size == 600
    with pytest.raises(ValueError, match="Warm-up flag"):
        append_round(learner, tmp_path, 6, count=1, warmup=True)


def test_updates_do_not_change_published_actor_until_explicit_publish(tmp_path):
    learner = ready_learner(tmp_path)
    before = learner.act(act_request())
    append_round(learner, tmp_path, 6, count=1, warmup=False)
    with pytest.raises(ValueError, match="unfinished"):
        learner.publish("2")
    result = learner.train({"round_id": "2", "expected_new_transitions": 3})
    assert result["updates"] == 60 and learner.counts["training"] == 60
    assert learner.train({"round_id": "2"}) == result
    assert learner.act(act_request()) == before
    assert learner.publish("2")["actor_version"] == 1
    assert learner.publish("2")["actor_version"] == 1
    assert learner.act(act_request(1))["noise"] != before["noise"]
    with pytest.raises(ValueError, match="version mismatch"):
        learner.act(act_request())


def test_checkpoint_restores_optimizer_replay_rng_and_publication(tmp_path):
    learner = ready_learner(tmp_path)
    append_round(learner, tmp_path, 6, count=1, warmup=False)
    learner.train({"round_id": "2"})
    learner.publish("2")
    saved = learner.save("round2")
    expected = learner.replay.sample(7)
    restored = LearnerBridge(
        learner.work_dir,
        Path("unused"),
        config=learner.config,
        data_root=tmp_path,
        backend_factory=FakeBackend,
    )
    with pytest.raises(ValueError, match="checkpoint resume"):
        restored.initialize("training")
    restored.load(saved["path"])
    assert restored.status()["training_updates"] == 60
    np.testing.assert_array_equal(restored.replay.sample(7)["actions"], expected["actions"])
    assert restored.act(act_request(1)) == learner.act(act_request(1))
    assert restored.backend.weight == learner.backend.weight
    with pytest.raises(ValueError, match="immutable"):
        restored.save("round2")


def test_profile_counts_both_parity_updates_and_preserves_untrained_actor(tmp_path):
    learner = bridge(tmp_path, "engineering", engineering_update_cap=5)
    initial = learner.act(act_request())
    path = tmp_path / "profile.npz"
    np.savez(path, pixels=np.zeros((2, 64, 64, 3), np.uint8), states=np.zeros((2, 8)))
    request = {
        "path": str(path),
        "sha256": sha256(path),
        "updates": 3,
        "operation_id": "profile-01",
    }
    result = learner.profile(request)
    assert result["status"]["engineering_updates"] == 5
    assert result["request_sha256"] == json_hash(request)
    completed = json.loads(learner.ledger_path.read_text().splitlines()[-1])
    assert completed["event"] == "profile_completed"
    assert completed["request_sha256"] == json_hash(request)
    assert json_hash({**request, "operation_id": "profile-02"}) != result["request_sha256"]
    assert learner.act(act_request()) == initial
    with pytest.raises(ValueError, match="budget"):
        learner.profile({"path": str(path), "sha256": sha256(path), "updates": 1})
    learner.initialize("training")
    assert learner.counts["engineering"] == 5 and learner.replay.size == 0
    assert learner.backend.weight == 0


def test_actor_rng_is_stable_per_job_and_independent_of_other_requests(tmp_path):
    learner = bridge(tmp_path, "engineering")
    first = learner.act(act_request())
    learner.act(act_request(job="other", query=17))
    assert learner.act(act_request()) == first
    assert actor_seed(123, "test") != actor_seed(123, "other")
    with pytest.raises(ValueError):
        actor_seed(-1, "test")


def test_actor_precision_scope_restores_state_and_preserves_author_arguments():
    backend = object.__new__(AuthorBackend)
    precision = {"value": None}

    @contextlib.contextmanager
    def highest(value):
        assert value == "highest"
        previous = precision["value"]
        precision["value"] = value
        try:
            yield
        finally:
            precision["value"] = previous

    backend.jax = SimpleNamespace(
        random=SimpleNamespace(PRNGKey=lambda seed: (seed,), fold_in=lambda key, i: (*key, i)),
        default_matmul_precision=highest,
    )
    backend.published_actor = SimpleNamespace(
        apply_fn=object(), params=object(), batch_stats=object()
    )
    pixels = np.zeros((64, 64, 3), np.uint8)
    state = np.arange(8, dtype=np.float32)

    def sample(key, apply, params, observations, stats):
        assert precision["value"] == "highest" and key == (11, 3)
        assert apply is backend.published_actor.apply_fn
        assert (
            params is backend.published_actor.params
            and stats is backend.published_actor.batch_stats
        )
        np.testing.assert_array_equal(observations["pixels"], pixels[None, ..., None])
        np.testing.assert_array_equal(observations["state"], state[None, ..., None])
        return None, np.full((1, 1, 32), 0.2, np.float32)

    backend.sample_function = sample
    np.testing.assert_array_equal(
        backend.sample(pixels, state, 11, 3), np.full(32, 0.2, np.float32)
    )
    assert precision["value"] is None
    assert AuthorActorBackend.sample is AuthorBackend.sample

    def fail(*args):
        assert precision["value"] == "highest"
        raise RuntimeError("sampling failed")

    backend.sample_function = fail
    with pytest.raises(RuntimeError, match="sampling failed"):
        backend.sample(pixels, state, 11, 3)
    assert precision["value"] is None


@pytest.mark.parametrize("difference", [0.0, 3e-6])
def test_reference_parity_shares_static_state_but_rebinds_agents_independently(difference):
    State = namedtuple("ImmutableState", "value apply_fn tx")
    static_apply, static_tx = object(), object()
    calls, charges = [], []
    backend = object.__new__(AuthorBackend)

    def immutable(value):
        result = np.asarray([value], dtype=np.float64)
        result.flags.writeable = False
        return result

    class Agent:
        def __init__(self):
            self._actor = State(immutable(0), static_apply, static_tx)
            self._critic = State(immutable(0), static_apply, static_tx)
            self._temp = State(immutable(0), static_apply, static_tx)
            self._target_critic_params = immutable(0)
            self._rng = immutable(0)

        @property
        def _save_dict(self):
            return {
                "actor": self._actor,
                "critic": self._critic,
                "temp": self._temp,
                "target_critic_params": self._target_critic_params,
            }

        def update(self, batch):
            assert batch == {"fixed": True}
            if not calls:
                assert self is not backend.agent
                assert self._actor is backend.agent._actor
                assert self._critic is backend.agent._critic
                assert self._rng is backend.agent._rng
            else:
                assert self is backend.agent
                assert self._actor.value[0] == 0 and self._rng[0] == 0
            assert not self._actor.value.flags.writeable
            assert self._actor.apply_fn is static_apply and self._actor.tx is static_tx
            calls.append(self)
            self._actor = self._actor._replace(
                value=immutable(1 + (difference if self is backend.agent else 0))
            )
            self._critic = self._critic._replace(value=immutable(1))
            self._temp = self._temp._replace(value=immutable(1))
            self._target_critic_params = immutable(1)
            self._rng = immutable(1)
            return {"critic_loss": 0.5}

    def flatten(values):
        leaves = []
        for name, value in sorted(values.items()):
            path = f"['{name}']"
            leaves.append(
                (path + ".value", value.value) if isinstance(value, State) else (path, value)
            )
        return leaves, tuple(path for path, _ in leaves)

    backend.agent = Agent()
    backend.freeze = lambda value: value
    backend.jax = SimpleNamespace(
        block_until_ready=lambda value: value,
        device_get=lambda value: value,
        tree_util=SimpleNamespace(tree_flatten_with_path=flatten, keystr=lambda value: value),
    )
    if difference:
        with pytest.raises(ValueError, match=r"e-06 at \['actor'\].value"):
            backend.reference_parity({"fixed": True}, charges.append)
    else:
        assert (
            backend.reference_parity({"fixed": True}, charges.append)["maximum_state_difference"]
            == 0
        )
    assert charges == ["author_reference", "bridge_parity"] and len(calls) == 2
    assert calls[0]._actor is not backend.agent._actor
    assert calls[0]._actor.apply_fn is backend.agent._actor.apply_fn
    assert calls[0]._actor.tx is backend.agent._actor.tx


def test_uncheckpointed_updates_refuse_ambiguous_resume(tmp_path):
    learner = ready_learner(tmp_path)
    saved = learner.save("before")
    learner.reserve_update("engineering", "simulated-interrupted-check")
    with pytest.raises(ValueError, match="ambiguous resume"):
        learner.load(saved["path"])


def test_workstation_updates_import_once_and_survive_reinitialization(tmp_path):
    config = LearnerConfig(capacity=10)
    options = dict(config=config, backend_factory=FakeBackend, prior_engineering_updates=2)
    learner = LearnerBridge(tmp_path / "learner", Path("unused"), **options)
    learner.initialize("engineering")
    learner.initialize("training")
    assert learner.counts["engineering"] == 2
    reopened = LearnerBridge(tmp_path / "learner", Path("unused"), **options)
    assert reopened.counts["engineering"] == 2
    with pytest.raises(ValueError, match="fresh ledger"):
        LearnerBridge(
            tmp_path / "learner",
            Path("unused"),
            config=config,
            backend_factory=FakeBackend,
            prior_engineering_updates=3,
        )


def test_interrupted_round_cannot_silently_repeat_full_update_credit(tmp_path):
    learner = ready_learner(tmp_path)
    append_round(learner, tmp_path, 6, count=1, warmup=False)
    original = learner.backend.update

    def fail_after_one(batch):
        if learner.backend.weight:
            raise RuntimeError("synthetic kernel failure")
        return original(batch)

    learner.backend.update = fail_after_one
    with pytest.raises(RuntimeError, match="kernel failure"):
        learner.train({"round_id": "2"})
    assert learner.counts["training"] == 2
    with pytest.raises(ValueError, match="credit cannot reset"):
        learner.train({"round_id": "2"})
    assert learner.counts["training"] == 2


def test_cpu_export_uses_published_weights_and_preserves_sampling(tmp_path):
    learner = ready_learner(tmp_path)
    append_round(learner, tmp_path, 6, count=1, warmup=False)
    with pytest.raises(ValueError, match="idle published"):
        learner.export_actor("before-updates")
    learner.train({"round_id": "2"})
    exported = learner.export_actor("before-publish")
    cpu = FrozenActorBridge(
        tmp_path / "cpu-before", Path("unused"), exported["path"], backend_factory=FakeActorBackend
    )
    assert cpu.act(act_request()) == learner.act(act_request())
    assert cpu.backend.published == 0 and learner.backend.weight == 60
    learner.publish("2")
    exported = learner.export_actor("final")
    cpu = FrozenActorBridge(
        tmp_path / "cpu-final", Path("unused"), exported["path"], backend_factory=FakeActorBackend
    )
    assert cpu.act(act_request(1)) == learner.act(act_request(1))
    assert cpu.status()["actor_evaluation_mode"] == "cpu_published_actor"
    assert cpu.backend.published == 60 and not hasattr(cpu, "replay")
    with pytest.raises(ValueError, match="immutable"):
        learner.export_actor("final")
    with pytest.raises(ValueError, match="different weights"):
        FrozenActorBridge(
            tmp_path / "cpu-final",
            Path("unused"),
            learner.work_dir / "actor_exports/before-publish",
            backend_factory=FakeActorBackend,
        )
    (Path(exported["path"]) / "actor.msgpack").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        FrozenActorBridge(
            tmp_path / "tampered",
            Path("unused"),
            exported["path"],
            backend_factory=FakeActorBackend,
        )


@pytest.mark.parametrize("actor_only", [False, True])
def test_http_service_preserves_version_errors_and_shutdown(tmp_path, actor_only):
    path = Path(__file__).resolve().parents[1] / "scripts/dsrl06_learner_service.py"
    spec = importlib.util.spec_from_file_location("dsrl06_learner_service", path)
    service = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(service)
    learner = bridge(tmp_path, "engineering")
    if actor_only:
        exported = learner.export_actor("http")
        learner = FrozenActorBridge(
            tmp_path / "cpu-http",
            Path("unused"),
            exported["path"],
            backend_factory=FakeActorBackend,
        )
    try:
        available = socket.socket()
    except PermissionError:
        pytest.skip("This sandbox blocks localhost sockets")
    with available:
        available.bind(("127.0.0.1", 0))
        port = available.getsockname()[1]
    thread = threading.Thread(
        target=service.serve,
        args=(learner, "127.0.0.1", port),
        kwargs={"actor_only": actor_only},
    )
    thread.start()
    base = f"http://127.0.0.1:{port}"

    def post(route, value):
        request = Request(
            base + route,
            data=json.dumps(value).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=3) as response:
            return json.load(response)

    try:
        deadline = time.monotonic() + 3
        while True:
            try:
                with urlopen(base + "/status", timeout=1) as response:
                    assert json.load(response)["phase"] == (
                        "evaluation" if actor_only else "engineering"
                    )
                break
            except URLError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)
        assert len(post("/act", act_request())["noise"]) == 32
        with pytest.raises(HTTPError) as failed:
            post("/act", act_request(version=1))
        assert failed.value.code == 409
        assert "version mismatch" in json.load(failed.value)["message"]
        if actor_only:
            with pytest.raises(HTTPError) as failed:
                post("/initialize", {"phase": "training"})
            assert failed.value.code == 404
        else:
            assert post("/initialize", {"phase": "training"})["replay_transitions"] == 0
    finally:
        assert post("/shutdown", {})["shutdown"]
        thread.join(timeout=3)
        assert not thread.is_alive()
