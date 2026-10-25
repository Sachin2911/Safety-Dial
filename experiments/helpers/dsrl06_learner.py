"""Pinned author-SAC bridge and deterministic replay for the DSRL task 6 pilot.

Only ``AuthorBackend`` imports the separate JAX learner environment. The replay,
input validation and orchestration layer remain importable by the workstation venv.
No simulator object, monitor flag or reward history is an actor/critic observation.
"""

from __future__ import annotations

import base64
import contextlib
import copy
from dataclasses import asdict, dataclass
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

SOURCE_REVISION = "7f48937d4553e95244cd81c79236a3256df80597"
EPISODE_KEYS = {
    "pixels",
    "states",
    "noise",
    "rewards",
    "discounts",
    "masks",
    "durations",
    "query_steps",
    "actor_versions",
}


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@dataclass(frozen=True)
class LearnerConfig:
    seed: int = 6060
    gamma: float = 0.9998
    batch_size: int = 256
    capacity: int = 6000
    updates_per_transition: int = 20
    global_update_cap: int = 120000
    engineering_update_cap: int = 2000
    training_update_cap: int = 90000

    def author_kwargs(self) -> dict:
        # Explicitly resolves the launcher and constructor defaults at SOURCE_REVISION.
        return dict(
            actor_lr=1e-4,
            critic_lr=3e-4,
            temp_lr=3e-4,
            hidden_dims=(128, 128, 128),
            cnn_features=(32, 32, 32, 32),
            cnn_strides=(2, 1, 1, 1),
            cnn_padding="VALID",
            latent_dim=50,
            discount=self.gamma,
            tau=0.005,
            critic_reduction="mean",
            dropout_rate=0.0,
            aug_next=True,
            use_bottleneck=True,
            encoder_type="small",
            encoder_norm="group",
            use_spatial_softmax=True,
            softmax_temperature=-1,
            target_entropy="auto",
            num_qs=10,
            action_magnitude=1.0,
            num_cameras=1,
            color_jitter=True,
            init_temperature=1.0,
            decay_steps=None,
        )

    def resolved(self) -> dict:
        return {
            **asdict(self),
            "source_revision": SOURCE_REVISION,
            "author_kwargs": self.author_kwargs(),
            "target_entropy_resolved": -16.0,
            "image_transform": "RGB uint8 PIL bilinear resize to 64x64; no flip or crop",
            "actor_input": ["overview_rgb64", "native_robot_state8"],
            "action_chunk_shape": [1, 32],
            "sampling_seed_rule": (
                "uint32 little-endian first four SHA256 bytes of "
                "dsrl06-actor:<rng_seed>:<job_id>, then JAX fold_in(query_index)"
            ),
        }


def prepare_observation(image: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(image)
    state = np.asarray(state, dtype=np.float32)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Actor image must be an H by W by 3 uint8 RGB array")
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("Actor state must be eight finite native robot coordinates")
    if image.shape != (64, 64, 3):
        image = np.asarray(Image.fromarray(image).resize((64, 64), Image.Resampling.BILINEAR))
    return np.ascontiguousarray(image), np.ascontiguousarray(state)


def pixels_from_base64(value: str) -> np.ndarray:
    raw = base64.b64decode(value, validate=True)
    if len(raw) != 64 * 64 * 3:
        raise ValueError("pixels_b64 must contain exactly 64*64*3 uint8 RGB bytes")
    return np.frombuffer(raw, dtype=np.uint8).reshape(64, 64, 3).copy()


def actor_seed(rng_seed: int, job_id: str) -> int:
    if not isinstance(rng_seed, int) or not 0 <= rng_seed < 2**32 or not job_id:
        raise ValueError("A uint32 episode RNG seed and nonempty job id are required")
    digest = hashlib.sha256(f"dsrl06-actor:{rng_seed}:{job_id}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def resolve_data_path(path: str | Path, data_root: Path | None = None) -> Path:
    result = Path(path).resolve(strict=True)
    if data_root is not None and not result.is_relative_to(data_root.resolve()):
        raise ValueError("Input is outside the declared experiment data root")
    if not result.is_file():
        raise ValueError("Input must be a regular file")
    return result


def import_author_source(source: Path) -> Path:
    source = source.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != SOURCE_REVISION:
        raise ValueError("DSRL source revision is not the frozen author revision")
    if subprocess.run(["git", "-C", str(source), "diff", "--quiet", "HEAD"]).returncode:
        raise ValueError("Tracked author source differs from its frozen revision")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    return source


def load_episode(
    descriptor: dict,
    config: LearnerConfig,
    *,
    warmup: bool,
    data_root: Path | None = None,
) -> dict[str, np.ndarray]:
    path = resolve_data_path(descriptor["path"], data_root)
    if sha256(path) != descriptor["sha256"]:
        raise ValueError("Episode NPZ hash differs from its admitted descriptor")
    with np.load(path, allow_pickle=False) as archive:
        if not EPISODE_KEYS <= set(archive.files):
            raise ValueError(f"Missing episode arrays: {EPISODE_KEYS - set(archive.files)}")
        arrays = {name: archive[name].copy() for name in EPISODE_KEYS}
    count = len(arrays["noise"])
    if not 1 <= count <= 250:
        raise ValueError("A completed episode must contain 1 through 250 latent transitions")
    expected = {
        "pixels": (count + 1, 64, 64, 3),
        "states": (count + 1, 8),
        "noise": (count, 32),
        "rewards": (count,),
        "discounts": (count,),
        "masks": (count,),
        "durations": (count,),
        "query_steps": (count,),
        "actor_versions": (count,),
    }
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(f"Wrong shape for {name}: {arrays[name].shape}, expected {shape}")
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"Nonfinite {name}")
    if arrays["pixels"].dtype != np.uint8:
        raise ValueError("Replay images must remain lossless uint8")
    for name in ("durations", "query_steps", "actor_versions"):
        if not np.issubdtype(arrays[name].dtype, np.integer):
            raise ValueError(f"{name} must use an integer dtype")
    durations = arrays["durations"]
    if np.any((durations < 1) | (durations > 8)) or np.any(durations[:-1] != 8):
        raise ValueError("Nonterminal durations must be eight controls")
    if durations.sum() > 2000:
        raise ValueError("Episode exceeds the native horizon")
    if not np.array_equal(arrays["query_steps"], np.r_[0, np.cumsum(durations[:-1])]):
        raise ValueError("Query starts do not match executed control durations")
    if np.any(arrays["masks"][:-1] != 1) or arrays["masks"][-1] != 0:
        raise ValueError("A complete finite episode must have exactly one final zero mask")
    if not np.allclose(arrays["discounts"], config.gamma**durations, atol=3e-7, rtol=1e-7):
        raise ValueError("Discounts do not use actual executed-control duration")
    if not np.all(arrays["actor_versions"] == descriptor["actor_version"]):
        raise ValueError("Episode mixed actor versions or differs from its descriptor")
    if not warmup and np.max(np.abs(arrays["noise"])) > 1.00001:
        raise ValueError("Learned latent actions exceed the frozen magnitude-1 domain")
    return arrays


class ReplayStore:
    """No-eviction host replay; sampling RNG is isolated and checkpointed."""

    def __init__(self, capacity: int, seed: int):
        if capacity <= 0:
            raise ValueError("Replay capacity must be positive")
        self.capacity = capacity
        self.size = 0
        self.rng = np.random.default_rng(seed)
        self.data = {
            "pixels": np.empty((capacity, 64, 64, 3), np.uint8),
            "next_pixels": np.empty((capacity, 64, 64, 3), np.uint8),
            "states": np.empty((capacity, 8), np.float32),
            "next_states": np.empty((capacity, 8), np.float32),
            "noise": np.empty((capacity, 32), np.float32),
            "next_noise": np.empty((capacity, 32), np.float32),
            "rewards": np.empty(capacity, np.float32),
            "masks": np.empty(capacity, np.float32),
            "discounts": np.empty(capacity, np.float32),
        }

    def append(self, episode: dict[str, np.ndarray]) -> None:
        n = len(episode["noise"])
        if self.size + n > self.capacity:
            raise ValueError("Replay capacity exhausted; eviction is forbidden")
        target = slice(self.size, self.size + n)
        values = {
            "pixels": episode["pixels"][:-1],
            "next_pixels": episode["pixels"][1:],
            "states": episode["states"][:-1],
            "next_states": episode["states"][1:],
            "noise": episode["noise"],
            "next_noise": np.concatenate([episode["noise"][1:], episode["noise"][-1:]]),
            "rewards": episode["rewards"],
            "masks": episode["masks"],
            "discounts": episode["discounts"],
        }
        for name, value in values.items():
            self.data[name][target] = value
        self.size += n

    def sample(self, size: int, indices: np.ndarray | None = None) -> dict:
        if not self.size:
            raise ValueError("Cannot sample empty replay")
        if indices is None:
            indices = self.rng.integers(0, self.size, size=size)
        return self.batch_at(indices)

    def batch_at(self, indices: np.ndarray) -> dict:
        a = {name: values[indices] for name, values in self.data.items()}
        return {
            "observations": {"pixels": a["pixels"][..., None], "state": a["states"][..., None]},
            "next_observations": {
                "pixels": a["next_pixels"][..., None],
                "state": a["next_states"][..., None],
            },
            "actions": a["noise"][:, None, :],
            "next_actions": a["next_noise"][:, None, :],
            "rewards": a["rewards"],
            "masks": a["masks"],
            "discount": a["discounts"],
        }

    def save(self, path: Path) -> dict:
        np.savez_compressed(path, **{name: value[: self.size] for name, value in self.data.items()})
        return {"size": self.size, "capacity": self.capacity, "rng": self.rng.bit_generator.state}

    def restore(self, path: Path, metadata: dict) -> None:
        if metadata["capacity"] != self.capacity or metadata["size"] > self.capacity:
            raise ValueError("Checkpoint replay capacity mismatch")
        with np.load(path, allow_pickle=False) as source:
            for name, destination in self.data.items():
                value = source[name]
                if value.shape != destination[: metadata["size"]].shape:
                    raise ValueError(f"Checkpoint replay shape mismatch: {name}")
                destination[: len(value)] = value
        self.size = metadata["size"]
        self.rng.bit_generator.state = metadata["rng"]


class AuthorBackend:
    """Calls the pinned author's networks, actor sampler and unmodified SAC update."""

    def __init__(self, source: Path, config: LearnerConfig):
        import_author_source(source)
        import jax
        from flax import serialization
        from flax.core import freeze
        from jaxrl2.agents.common import sample_actions_jit
        from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner

        self.jax, self.serialization, self.freeze = jax, serialization, freeze
        self.learner_type = PixelSACLearner
        self.sample_function = sample_actions_jit
        self.config = config
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            self.agent = self.new_agent()
        self.initialization_log = stream.getvalue()
        self.published_actor = self.agent._actor
        self.parameter_counts = {
            "actor": sum(int(x.size) for x in jax.tree.leaves(self.agent._actor.params)),
            "critic": sum(int(x.size) for x in jax.tree.leaves(self.agent._critic.params)),
        }

    def new_agent(self):
        return self.learner_type(
            self.config.seed,
            {
                "pixels": np.zeros((1, 64, 64, 3, 1), np.uint8),
                "state": np.zeros((1, 8, 1), np.float32),
            },
            np.zeros((1, 1, 32), np.float32),
            **self.config.author_kwargs(),
        )

    def sample(self, pixels: np.ndarray, state: np.ndarray, seed: int, index: int) -> np.ndarray:
        key = self.jax.random.fold_in(self.jax.random.PRNGKey(seed), index)
        actor = self.published_actor
        # Sampling has one cross-device float32 contract; source SAC updates retain defaults.
        with self.jax.default_matmul_precision("highest"):
            _, action = self.sample_function(
                key,
                actor.apply_fn,
                actor.params,
                {"pixels": pixels[None, ..., None], "state": state[None, ..., None]},
                actor.batch_stats,
            )
        return np.asarray(action, dtype=np.float32).reshape(32)

    def update(self, batch: dict) -> dict[str, float]:
        info = self.agent.update(self.freeze(batch))
        self.jax.block_until_ready(info)
        return {name: float(np.asarray(x)) for name, x in self.jax.device_get(info).items()}

    def publish(self) -> None:
        self.published_actor = self.agent._actor

    def state(self) -> dict:
        return {
            **self.agent._save_dict,
            "rng": self.agent._rng,
            "published_actor": self.published_actor,
        }

    def dumps(self) -> bytes:
        return self.serialization.to_bytes(self.state())

    def actor_dumps(self) -> bytes:
        return self.serialization.to_bytes(
            {
                "params": self.published_actor.params,
                "batch_stats": self.published_actor.batch_stats,
            }
        )

    def loads(self, content: bytes) -> None:
        restored = self.serialization.from_bytes(self.state(), content)
        self.agent._actor = restored["actor"]
        self.agent._critic = restored["critic"]
        self.agent._target_critic_params = restored["target_critic_params"]
        self.agent._temp = restored["temp"]
        self.agent._rng = restored["rng"]
        self.published_actor = restored["published_actor"]

    def reference_parity(self, batch: dict, reserve) -> dict:
        """Two charged updates: direct author clone versus the bridge path."""
        # Source updates rebind immutable TrainStates and arrays. Preserve their
        # exact static apply_fn/Optax closures and initial device buffers so both
        # paths use the same JIT specialization, without mutating the other agent.
        reference = copy.copy(self.agent)
        reserve("author_reference")
        info = reference.update(self.freeze(batch))
        self.jax.block_until_ready(info)
        reserve("bridge_parity")
        self.update(batch)
        expected = {**reference._save_dict, "rng": reference._rng}
        actual = {**self.agent._save_dict, "rng": self.agent._rng}
        expected_leaves, expected_tree = self.jax.tree_util.tree_flatten_with_path(expected)
        actual_leaves, actual_tree = self.jax.tree_util.tree_flatten_with_path(actual)
        if expected_tree != actual_tree:
            raise ValueError("Author parity mutable-state tree structures differ")
        maximum, maximum_path = 0.0, None
        for (path, left), (_, right) in zip(expected_leaves, actual_leaves):
            a, b = np.asarray(left), np.asarray(right)
            if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError(
                    "Nonfinite or mismatched author parity state at "
                    + self.jax.tree_util.keystr(path)
                )
            difference = float(np.max(np.abs(a.astype(float) - b.astype(float))))
            if difference > maximum:
                maximum = difference
                maximum_path = self.jax.tree_util.keystr(path)
        if maximum > 1e-6:
            raise ValueError(
                f"Bridge differs from direct author update: {maximum} at {maximum_path}"
            )
        return {
            "maximum_state_difference": maximum,
            "maximum_state_difference_path": maximum_path,
            "updates_charged": 2,
        }


class AuthorActorBackend:
    """CPU-only author actor, with no critic, temperature, optimizer or replay."""

    def __init__(self, source: Path, config: LearnerConfig, content: bytes):
        import_author_source(source)
        import jax
        from flax import serialization
        from jaxrl2.agents.common import sample_actions_jit
        from jaxrl2.networks.encoders.networks import Encoder, PixelMultiplexer
        from jaxrl2.networks.learned_std_normal_policy import LearnedStdTanhNormalPolicy

        if jax.default_backend() != "cpu":
            raise ValueError("Published actor evaluation requires the CPU JAX backend")
        settings = config.author_kwargs()
        # These are the exact actor modules and constructor arguments in the pinned learner.
        encoder = Encoder(
            settings["cnn_features"], settings["cnn_strides"], settings["cnn_padding"]
        )
        policy = LearnedStdTanhNormalPolicy(
            settings["hidden_dims"],
            32,
            dropout_rate=settings["dropout_rate"],
            low=-settings["action_magnitude"],
            high=settings["action_magnitude"],
        )
        actor = PixelMultiplexer(
            encoder=encoder,
            network=policy,
            latent_dim=settings["latent_dim"],
            use_bottleneck=settings["use_bottleneck"],
        )
        _, actor_key, _, _ = jax.random.split(jax.random.PRNGKey(config.seed), 4)
        variables = actor.init(
            actor_key,
            {
                "pixels": np.zeros((1, 64, 64, 3, 1), np.uint8),
                "state": np.zeros((1, 8, 1), np.float32),
            },
        )
        restored = serialization.from_bytes(
            {"params": variables["params"], "batch_stats": variables.get("batch_stats")},
            content,
        )
        if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(restored)):
            raise ValueError("Exported actor contains nonfinite weights")
        self.jax = jax
        self.sample_function = sample_actions_jit
        self.published_actor = SimpleNamespace(apply_fn=actor.apply, **restored)
        self.parameter_counts = {
            "actor": sum(int(x.size) for x in jax.tree.leaves(restored["params"]))
        }

    sample = AuthorBackend.sample


class LearnerBridge:
    """Serial training, immutable publication, idempotent rounds and full snapshots."""

    def __init__(
        self,
        work_dir: Path,
        source: Path,
        *,
        config: LearnerConfig | None = None,
        data_root: Path | None = None,
        backend_factory=AuthorBackend,
        prior_engineering_updates: int = 0,
        prior_engineering_receipt: str | None = None,
    ):
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.source, self.config = Path(source), config or LearnerConfig()
        self.data_root = data_root
        self.backend_factory = backend_factory
        self.lock = threading.RLock()
        self.training_lock = threading.Lock()
        self.ledger_path = self.work_dir / "learner-ledger.jsonl"
        self.counts = {"engineering": 0, "training": 0, "actor": 0}
        self.history_has_training_data = False
        imported = 0
        had_events = False
        if self.ledger_path.exists():
            for line in self.ledger_path.read_text().splitlines():
                event = json.loads(line)
                had_events = True
                if event["event"] == "update_reserved":
                    self.counts[event["phase"]] += 1
                    imported += event.get("origin") == "workstation_cpu_preflight"
                elif event["event"] == "actor_request":
                    self.counts["actor"] += 1
                elif event["event"] == "round_appended":
                    self.history_has_training_data = True
        if not isinstance(prior_engineering_updates, int) or not (
            0 <= prior_engineering_updates <= self.config.engineering_update_cap
        ):
            raise ValueError("Invalid prior engineering update count")
        if prior_engineering_updates and imported != prior_engineering_updates:
            if had_events:
                raise ValueError("Prior engineering updates must be imported into a fresh ledger")
            for index in range(prior_engineering_updates):
                self.event(
                    "update_reserved",
                    phase="engineering",
                    label=f"workstation_cpu_preflight:{index + 1}",
                    origin="workstation_cpu_preflight",
                    receipt=prior_engineering_receipt,
                    phase_update=index + 1,
                )
            self.counts["engineering"] = prior_engineering_updates
        self.active_training = False
        self.backend = None
        self.phase = "engineering"
        self._fresh_state()

    def _fresh_state(self) -> None:
        self.replay = ReplayStore(self.config.capacity, self.config.seed)
        self.episodes: list[dict] = []
        self.appended_rounds: dict[str, dict] = {}
        self.trained_rounds: dict[str, dict] = {}
        self.interrupted_rounds: dict[str, str] = {}
        self.published_rounds: dict[str, int] = {}
        self.pending_transitions = 0
        self.pending_round = None
        self.warmup_complete = False
        self.actor_version = 0

    def event(self, name: str, **data) -> None:
        row = {"utc": utc(), "event": name, **data}
        with self.lock, self.ledger_path.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def initialize(self, phase: str) -> dict:
        if phase not in ("engineering", "training"):
            raise ValueError("Phase must be engineering or training")
        with self.lock:
            if self.active_training or (self.phase == "training" and self.episodes):
                raise ValueError("Cannot reset an active or already collecting training learner")
            if self.backend is None and self.history_has_training_data:
                raise ValueError("Existing training data require a checkpoint resume, not a reset")
            self.phase = phase
            self._fresh_state()
            self.backend = self.backend_factory(self.source, self.config)
            self.event("initialized", phase=phase, seed=self.config.seed)
            return self.status()

    def status(self) -> dict:
        with self.lock:
            return {
                "phase": self.phase,
                "initialized": self.backend is not None,
                "actor_version": self.actor_version,
                "episodes": len(self.episodes),
                "replay_transitions": self.replay.size,
                "warmup_complete": self.warmup_complete,
                "warmup_exhausted": len(self.episodes) >= 12 and not self.warmup_complete,
                "pending_transitions": self.pending_transitions,
                "pending_round": self.pending_round,
                "active_training": self.active_training,
                "interrupted_rounds": dict(self.interrupted_rounds),
                "engineering_updates": self.counts["engineering"],
                "training_updates": self.counts["training"],
                "global_updates": self.counts["engineering"] + self.counts["training"],
                "actor_requests": self.counts["actor"],
                "parameter_counts": getattr(self.backend, "parameter_counts", {}),
                "resolved_config": self.config.resolved(),
            }

    def act(self, request: dict) -> dict:
        pixels, state = prepare_observation(
            pixels_from_base64(request["pixels_b64"]),
            request["state"],
        )
        index = request["query_index"]
        if not isinstance(index, int) or not 0 <= index < 250:
            raise ValueError("Query index is outside the native episode horizon")
        with self.lock:
            if self.backend is None or request["actor_version"] != self.actor_version:
                raise ValueError("Actor version mismatch or uninitialized learner")
            seed = actor_seed(request["rng_seed"], request["job_id"])
            self.event(
                "actor_request",
                job_id=request["job_id"],
                query_index=index,
                actor_version=self.actor_version,
            )
            self.counts["actor"] += 1
            noise = self.backend.sample(pixels, state, seed, index)
            if noise.shape != (32,) or not np.isfinite(noise).all() or np.max(abs(noise)) > 1.00001:
                raise ValueError("Author actor returned invalid bounded noise")
            return {"noise": noise.tolist(), "actor_version": self.actor_version}

    def append(self, request: dict) -> dict:
        round_id = str(request["round_id"])
        descriptors = request["episodes"]
        warmup = request["warmup"]
        if type(warmup) is not bool or len(descriptors) != 3:
            raise ValueError("Append requires one complete three-episode round")
        fingerprint = json_hash({"warmup": warmup, "episodes": descriptors})
        with self.lock:
            if self.phase != "training" or self.active_training or self.backend is None:
                raise ValueError("Only an idle initialized training learner accepts replay")
            if round_id in self.appended_rounds:
                existing = self.appended_rounds[round_id]
                if existing["fingerprint"] != fingerprint:
                    raise ValueError("Round identity reused for different episode data")
                return existing["response"]
            if self.pending_transitions:
                raise ValueError("Prior round update credit must finish before appending")
            if len(self.episodes) + len(descriptors) > 24:
                raise ValueError("Training episode budget exhausted")
            if warmup != (not self.warmup_complete):
                raise ValueError("Warm-up flag disagrees with the frozen six-task cycle rule")
            if warmup and len(self.episodes) >= 12:
                raise ValueError("Warm-up transition threshold was not met within twelve episodes")
            arrays = []
            known = {row["job_id"] for row in self.episodes}
            for offset, descriptor in enumerate(descriptors):
                if descriptor["order_index"] != len(self.episodes) + offset:
                    raise ValueError("Episodes must be appended in frozen job order")
                if descriptor["job_id"] in known:
                    raise ValueError("Episode already entered replay")
                known.add(descriptor["job_id"])
                if descriptor["actor_version"] != self.actor_version:
                    raise ValueError("Collected episode did not use the published actor version")
                arrays.append(
                    load_episode(
                        descriptor,
                        self.config,
                        warmup=warmup,
                        data_root=self.data_root,
                    )
                )
            new_count = sum(len(a["noise"]) for a in arrays)
            if self.replay.size + new_count > self.config.capacity:
                raise ValueError("Round would overflow no-eviction replay")
            for descriptor, episode in zip(descriptors, arrays):
                self.replay.append(episode)
                self.episodes.append(
                    {**descriptor, "warmup": warmup, "transitions": len(episode["noise"])}
                )
            if warmup:
                if len(self.episodes) % 6 == 0 and self.replay.size > 500:
                    self.warmup_complete = True
            else:
                self.pending_transitions = new_count
                self.pending_round = round_id
            response = {
                "round_id": round_id,
                "new_transitions": new_count,
                "warmup": warmup,
                "status": self.status(),
            }
            self.appended_rounds[round_id] = {"fingerprint": fingerprint, "response": response}
            self.history_has_training_data = True
            self.event(
                "round_appended",
                round_id=round_id,
                warmup=warmup,
                transitions=new_count,
                jobs=[d["job_id"] for d in descriptors],
            )
            return response

    def reserve_update(self, phase: str, label: str) -> None:
        with self.lock:
            cap = (
                self.config.engineering_update_cap
                if phase == "engineering"
                else self.config.training_update_cap
            )
            if (
                self.counts[phase] >= cap
                or sum(self.counts[p] for p in ("engineering", "training"))
                >= self.config.global_update_cap
            ):
                raise ValueError("Learner update cap exhausted")
            if (self.work_dir.parent / "STOP").exists():
                raise RuntimeError("Experiment admission stop file exists")
            self.event(
                "update_reserved", phase=phase, label=label, phase_update=self.counts[phase] + 1
            )
            self.counts[phase] += 1

    def train(self, request: dict) -> dict:
        round_id = str(request["round_id"])
        with self.training_lock:
            with self.lock:
                if round_id in self.trained_rounds:
                    return self.trained_rounds[round_id]
                if round_id in self.interrupted_rounds:
                    raise ValueError(
                        "Interrupted round requires diagnosis; update credit cannot reset"
                    )
                if self.phase != "training" or self.pending_round != round_id:
                    raise ValueError("No eligible transition credit for this round")
                count = self.pending_transitions * self.config.updates_per_transition
                if (
                    request.get("expected_new_transitions", self.pending_transitions)
                    != self.pending_transitions
                ):
                    raise ValueError("Caller and replay disagree on eligible update credit")
                if self.counts["training"] + count > self.config.training_update_cap:
                    raise ValueError("Round exceeds frozen post-warm-up update budget")
                if (
                    self.counts["training"] + self.counts["engineering"] + count
                    > self.config.global_update_cap
                ):
                    raise ValueError("Round exceeds global update budget")
                self.active_training = True
                self.event("train_start", round_id=round_id, updates=count)
            begin = time.monotonic()
            last = {}
            try:
                for index in range(count):
                    self.reserve_update("training", f"{round_id}:{index}")
                    last = self.backend.update(self.replay.sample(self.config.batch_size))
                    if not all(np.isfinite(value) for value in last.values()):
                        raise ValueError("Nonfinite source learner metrics; admission must stop")
                    if index % 100 == 0 or index + 1 == count:
                        self.event("training_metrics", round_id=round_id, index=index, **last)
                with self.lock:
                    self.pending_transitions = 0
                    self.pending_round = None
                    result = {
                        "round_id": round_id,
                        "updates": count,
                        "elapsed_seconds": time.monotonic() - begin,
                        "last_metrics": last,
                    }
                    self.trained_rounds[round_id] = result
                    self.event("train_completed", **result)
                    return result
            except Exception as error:
                self.interrupted_rounds[round_id] = type(error).__name__
                self.event("train_interrupted", round_id=round_id, error=type(error).__name__)
                raise
            finally:
                self.active_training = False

    def publish(self, round_id: str) -> dict:
        round_id = str(round_id)
        with self.lock:
            if self.active_training or self.pending_transitions:
                raise ValueError("Cannot publish while updates are unfinished")
            if round_id in self.published_rounds:
                return {"round_id": round_id, "actor_version": self.published_rounds[round_id]}
            if round_id not in self.trained_rounds:
                raise ValueError("Only a completed learning round can publish a new actor")
            self.backend.publish()
            self.actor_version += 1
            self.published_rounds[round_id] = self.actor_version
            self.event("actor_published", round_id=round_id, actor_version=self.actor_version)
            return {"round_id": round_id, "actor_version": self.actor_version}

    def profile(self, request: dict) -> dict:
        with self.lock:
            if self.active_training:
                raise ValueError("Wait for the active learner update before profiling")
            self.active_training = True
        try:
            result = self._profile(request)
        finally:
            with self.lock:
                self.active_training = False
        result["status"] = self.status()
        return result

    def _profile(self, request: dict) -> dict:
        with self.training_lock:
            if self.phase != "engineering" or self.backend is None:
                raise ValueError("Fixed-batch profiling is engineering only")
            updates = int(request.get("updates", 0))
            required = updates + (2 if request.get("parity", True) else 0)
            if (
                updates < 0
                or self.counts["engineering"] + required > self.config.engineering_update_cap
            ):
                raise ValueError("Engineering profiling update budget exceeded")
            if (
                self.counts["engineering"] + self.counts["training"] + required
                > self.config.global_update_cap
            ):
                raise ValueError("Global update budget exceeded")
            path = resolve_data_path(request["path"], self.data_root)
            if sha256(path) != request["sha256"]:
                raise ValueError("Profile observation hash mismatch")
            with np.load(path, allow_pickle=False) as source:
                pixels = source["pixels"].copy()
                states = source["states"].copy()
            if pixels.dtype != np.uint8 or pixels.shape[1:] != (64, 64, 3):
                raise ValueError("Profile pixels must be recorded RGB64 observations")
            if states.shape != (len(pixels), 8) or not len(pixels) or not np.isfinite(states).all():
                raise ValueError("Profile native states have invalid shape or values")
            indices = np.arange(self.config.batch_size) % len(pixels)
            observation = {
                "pixels": pixels[indices, ..., None],
                "state": states[indices, ..., None].astype(np.float32),
            }
            batch = {
                "observations": observation,
                "next_observations": observation,
                "actions": np.zeros((self.config.batch_size, 1, 32), np.float32),
                "next_actions": np.zeros((self.config.batch_size, 1, 32), np.float32),
                "rewards": np.zeros(self.config.batch_size, np.float32),
                "masks": np.zeros(self.config.batch_size, np.float32),
                "discount": np.full(self.config.batch_size, self.config.gamma**8, np.float32),
            }
            result = {
                "request_sha256": json_hash(request),
                "synthetic_zero_actions_and_targets": True,
                "recorded_observations": len(pixels),
            }
            if request.get("parity", True):
                result["source_parity"] = self.backend.reference_parity(
                    batch,
                    lambda label: self.reserve_update("engineering", label),
                )
            durations = []
            for index in range(updates):
                self.reserve_update("engineering", f"profile:{index}")
                begin = time.monotonic()
                info = self.backend.update(batch)
                durations.append(time.monotonic() - begin)
                if not all(np.isfinite(value) for value in info.values()):
                    raise ValueError("Nonfinite profiling update")
            result.update(
                {
                    "timed_updates": updates,
                    "seconds": sum(durations),
                    "median_update_seconds": float(np.median(durations)) if durations else None,
                    "status": self.status(),
                }
            )
            self.event("profile_completed", **{k: v for k, v in result.items() if k != "status"})
            return result

    def save(self, name: str) -> dict:
        if not name or Path(name).name != name:
            raise ValueError("Checkpoint name must be a single path component")
        with self.lock:
            if self.backend is None or self.active_training:
                raise ValueError("Cannot checkpoint an uninitialized or actively updating learner")
            parent = self.work_dir / "checkpoints"
            parent.mkdir(exist_ok=True)
            destination = parent / name
            if destination.exists():
                raise ValueError("Checkpoint already exists; snapshots are immutable")
            temporary = parent / (name + ".partial")
            if temporary.exists():
                raise ValueError("Partial checkpoint exists and must be preserved")
            temporary.mkdir()
            (temporary / "learner.msgpack").write_bytes(self.backend.dumps())
            replay = self.replay.save(temporary / "replay.npz")
            metadata = {
                "schema": "dsrl06-learner-checkpoint-v1",
                "utc": utc(),
                "config": self.config.resolved(),
                "phase": self.phase,
                "actor_version": self.actor_version,
                "counts": dict(self.counts),
                "episodes": self.episodes,
                "append_rounds": self.appended_rounds,
                "trained_rounds": self.trained_rounds,
                "interrupted_rounds": self.interrupted_rounds,
                "published_rounds": self.published_rounds,
                "pending_transitions": self.pending_transitions,
                "pending_round": self.pending_round,
                "warmup_complete": self.warmup_complete,
                "replay": replay,
            }
            atomic_json(temporary / "state.json", metadata)
            manifest = {p.name: sha256(p) for p in temporary.iterdir() if p.is_file()}
            atomic_json(temporary / "manifest.json", manifest)
            temporary.rename(destination)
            result = {
                "path": str(destination),
                "manifest_sha256": sha256(destination / "manifest.json"),
            }
            self.event("checkpoint_saved", **result)
            return result

    def export_actor(self, name: str) -> dict:
        if not name or Path(name).name != name:
            raise ValueError("Actor export name must be a single path component")
        with self.lock:
            if self.backend is None or self.active_training or self.pending_transitions:
                raise ValueError("Actor export requires an idle published actor")
            parent = self.work_dir / "actor_exports"
            parent.mkdir(exist_ok=True)
            destination = parent / name
            temporary = parent / (name + ".partial")
            if destination.exists() or temporary.exists():
                raise ValueError("Actor exports are immutable")
            temporary.mkdir()
            (temporary / "actor.msgpack").write_bytes(self.backend.actor_dumps())
            atomic_json(
                temporary / "state.json",
                {
                    "schema": "dsrl06-published-actor-v1",
                    "utc": utc(),
                    "actor_evaluation_mode": "cpu_published_actor",
                    "config": self.config.resolved(),
                    "actor_version": self.actor_version,
                    "counts": dict(self.counts),
                },
            )
            manifest = {p.name: sha256(p) for p in temporary.iterdir() if p.is_file()}
            atomic_json(temporary / "manifest.json", manifest)
            temporary.rename(destination)
            result = {
                "path": str(destination),
                "manifest_sha256": sha256(destination / "manifest.json"),
                "actor_version": self.actor_version,
                "actor_evaluation_mode": "cpu_published_actor",
            }
            self.event("actor_exported", **result)
            return result

    def load(self, path: str | Path) -> dict:
        path = Path(path).resolve(strict=True)
        with self.lock:
            if self.active_training:
                raise ValueError("Cannot restore during a training update")
            manifest = json.loads((path / "manifest.json").read_text())
            for name, expected in manifest.items():
                if Path(name).name != name or sha256(path / name) != expected:
                    raise ValueError("Checkpoint file hash mismatch")
            metadata = json.loads((path / "state.json").read_text())
            if json_hash(metadata["config"]) != json_hash(self.config.resolved()):
                raise ValueError("Checkpoint learner configuration mismatch")
            for phase in ("engineering", "training"):
                if metadata["counts"][phase] != self.counts[phase]:
                    raise ValueError(
                        "Uncheckpointed update attempts exist; refusing ambiguous resume"
                    )
            if self.backend is None:
                self.backend = self.backend_factory(self.source, self.config)
            self.backend.loads((path / "learner.msgpack").read_bytes())
            self.replay.restore(path / "replay.npz", metadata["replay"])
            self.phase = metadata["phase"]
            self.actor_version = metadata["actor_version"]
            self.episodes = metadata["episodes"]
            self.appended_rounds = metadata["append_rounds"]
            self.trained_rounds = metadata["trained_rounds"]
            self.interrupted_rounds = metadata.get("interrupted_rounds", {})
            self.published_rounds = metadata["published_rounds"]
            self.pending_transitions = metadata["pending_transitions"]
            self.pending_round = metadata["pending_round"]
            self.warmup_complete = metadata["warmup_complete"]
            self.event("checkpoint_restored", path=str(path))
            return self.status()


class FrozenActorBridge:
    """Read-only published actor endpoint using the same request and RNG contract."""

    def __init__(
        self,
        work_dir: Path,
        source: Path,
        checkpoint: Path,
        *,
        backend_factory=AuthorActorBackend,
    ):
        checkpoint = Path(checkpoint).resolve(strict=True)
        manifest = json.loads((checkpoint / "manifest.json").read_text())
        if set(manifest) != {"actor.msgpack", "state.json"}:
            raise ValueError("CPU service requires an actor-only export directory")
        for name, expected in manifest.items():
            if sha256(checkpoint / name) != expected:
                raise ValueError("Published actor export hash mismatch")
        self.metadata = json.loads((checkpoint / "state.json").read_text())
        if self.metadata["schema"] != "dsrl06-published-actor-v1":
            raise ValueError("Unknown published actor export schema")
        self.config = LearnerConfig(
            **{name: self.metadata["config"][name] for name in asdict(LearnerConfig())}
        )
        if json_hash(self.config.resolved()) != json_hash(self.metadata["config"]):
            raise ValueError("Published actor export configuration mismatch")
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.work_dir / "actor-ledger.jsonl"
        self.lock = threading.RLock()
        self.active_training = False
        self.actor_version = self.metadata["actor_version"]
        self.manifest_sha256 = sha256(checkpoint / "manifest.json")
        self.counts = {"actor": 0}
        if self.ledger_path.exists():
            for line in self.ledger_path.read_text().splitlines():
                event = json.loads(line)
                if event["event"] == "cpu_actor_loaded":
                    if event["manifest_sha256"] != self.manifest_sha256:
                        raise ValueError("Actor worker ledger already names different weights")
                elif event["event"] == "actor_request":
                    self.counts["actor"] += 1
        self.backend = backend_factory(
            Path(source), self.config, (checkpoint / "actor.msgpack").read_bytes()
        )
        self.event(
            "cpu_actor_loaded",
            actor_version=self.actor_version,
            manifest_sha256=self.manifest_sha256,
        )

    event = LearnerBridge.event
    act = LearnerBridge.act

    def status(self) -> dict:
        with self.lock:
            return {
                "phase": "evaluation",
                "initialized": True,
                "actor_evaluation_mode": "cpu_published_actor",
                "actor_version": self.actor_version,
                "manifest_sha256": self.manifest_sha256,
                "actor_requests": self.counts["actor"],
                "parameter_counts": self.backend.parameter_counts,
                "active_training": False,
                "resolved_config": self.config.resolved(),
            }
