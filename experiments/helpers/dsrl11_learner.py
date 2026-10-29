"""Condition-isolated author-SAC learners for the task-11 residual pilot.

Each condition starts from the same seed-11060 network initialization and owns its
optimizer, target critic, temperature, online replay, prior-bank copy and RNG
streams. Only ``Dsrl11Backend`` imports the separate JAX learner environment; the
replay, calibration arithmetic and orchestration remain importable anywhere.
No simulator object, monitor flag, potential or epsilon is an actor/critic input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np

from helpers.dsrl06_learner import (
    SOURCE_REVISION,
    AuthorActorBackend,
    AuthorBackend,
    LearnerConfig,
    ReplayStore,
    actor_seed,
    atomic_json,
    json_hash,
    pixels_from_base64,
    prepare_observation,
    resolve_data_path,
    sha256,
    utc,
)
from helpers.dsrl11_protocol import (
    BANK_SUCCESS_MINIMUM,
    CALIBRATION_INTERVAL,
    CALIBRATION_RANGE,
    CONDITION_SPEC,
    CONDITIONS,
    LEARNER_SEED,
    PHASES,
    RETURN_RANGE,
    calibration_failure,
    derive_seed,
    pause_rule_fired,
)

RESIDUAL_EPISODE_KEYS = frozenset(
    {
        "pixels",
        "states",
        "residual",
        "rewards_r0",
        "rewards_r1",
        "discounts",
        "masks",
        "durations",
        "query_steps",
        "actor_versions",
    }
)
ENGINEERING = "engineering"
PHASE_VERSIONS = {"warmup": -1, "batch_a": 0, "batch_b": 1}


@dataclass(frozen=True)
class Dsrl11LearnerConfig:
    seed: int = LEARNER_SEED
    gamma: float = 0.9998
    batch_size: int = 256
    prior_batch_size: int = 128
    capacity: int = 6000
    updates_per_transition: int = 20
    critic_reduction: str = "min"
    learning_update_cap: int = 120_000
    engineering_update_cap: int = 2_000
    calibration_interval: int = CALIBRATION_INTERVAL

    def __post_init__(self):
        if self.updates_per_transition not in (5, 20) or self.critic_reduction not in (
            "mean",
            "min",
        ):
            raise ValueError("The learner must use one of the four frozen A1 settings")
        if self.prior_batch_size * 2 != self.batch_size or self.capacity < 6000:
            raise ValueError("The frozen 128/128 mixture and no-eviction capacity changed")

    def author_kwargs(self) -> dict:
        kwargs = LearnerConfig(seed=self.seed, gamma=self.gamma).author_kwargs()
        kwargs["critic_reduction"] = self.critic_reduction
        return kwargs

    def resolved(self) -> dict:
        return {
            **asdict(self),
            "schema": "dsrl11-learner-config-v1",
            "source_revision": SOURCE_REVISION,
            "author_kwargs": self.author_kwargs(),
            "target_entropy_resolved": -16.0,
            "image_transform": "RGB uint8 PIL bilinear resize to 64x64; no flip or crop",
            "actor_input": ["overview_rgb64", "native_robot_state8"],
            "action": "residual u in [-1, 1]^32; decoder noise epsilon + 0.25 * tile(u, 16)",
            "action_chunk_shape": [1, 32],
            "sampling_seed_rule": (
                "uint32 little-endian first four SHA256 bytes of "
                "dsrl06-actor:<rng_seed>:<job_id>, then JAX fold_in(query_index)"
            ),
            "condition_streams": (
                "agent RNG PRNGKey(derive_seed('learner-rng', condition)) after the shared "
                "seed-11060 initialization; replay default_rng(derive_seed('replay', "
                "condition, 'online' or 'bank')); calibration PRNGKey(derive_seed("
                "'calibration-actions', condition))"
            ),
            "prior_mixture": (
                "prior conditions: 128 uniform bank and 128 uniform online transitions per "
                "update, bank rows first; other conditions: 256 uniform online"
            ),
            "update_credit": "updates_per_transition per newly appended batch transition; "
            "warm-up appends earn no credit",
            "changes_from_task7": {
                "seed": [6060, self.seed],
                "critic_reduction": ["mean", self.critic_reduction],
                "updates_per_transition": [20, self.updates_per_transition],
                "action": "absolute tiled noise replaced by the native-preserving residual",
            },
        }


def condition_streams(condition: str) -> dict[str, int]:
    if condition not in (*CONDITIONS, ENGINEERING):
        raise ValueError("Unknown learner condition")
    return {
        "agent_rng": derive_seed("learner-rng", condition),
        "online_replay": derive_seed("replay", condition, "online"),
        "bank_replay": derive_seed("replay", condition, "bank"),
        "calibration_actions": derive_seed("calibration-actions", condition),
    }


def load_residual_episode(
    descriptor: dict,
    config: Dsrl11LearnerConfig,
    reward: str,
    *,
    data_root: Path | None = None,
) -> dict[str, np.ndarray]:
    """Load one admitted residual episode in the replay layout with one reward."""
    if reward not in ("r0", "r1"):
        raise ValueError("Reward must be r0 or r1")
    path = resolve_data_path(descriptor["path"], data_root)
    if sha256(path) != descriptor["sha256"]:
        raise ValueError("Episode NPZ hash differs from its admitted descriptor")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != RESIDUAL_EPISODE_KEYS:
            raise ValueError("Residual episode arrays differ from the frozen layout")
        arrays = {name: archive[name].copy() for name in archive.files}
    count = len(arrays["residual"])
    if not 1 <= count <= 250:
        raise ValueError("A completed episode must contain 1 through 250 latent transitions")
    expected = {
        "pixels": (count + 1, 64, 64, 3),
        "states": (count + 1, 8),
        "residual": (count, 32),
        "rewards_r0": (count,),
        "rewards_r1": (count,),
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
    if arrays["pixels"].dtype != np.uint8 or arrays["residual"].dtype != np.float32:
        raise ValueError("Replay images must stay uint8 and residuals float32")
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
    if descriptor.get("transitions", count) != count:
        raise ValueError("Descriptor transition count differs from the episode")
    if np.max(np.abs(arrays["residual"])) > 1.0:
        raise ValueError("Residual actions exceed the frozen [-1, 1] domain")
    return {
        "pixels": arrays["pixels"],
        "states": arrays["states"],
        "noise": arrays["residual"],
        "rewards": arrays[f"rewards_{reward}"],
        "discounts": arrays["discounts"],
        "masks": arrays["masks"],
        "durations": arrays["durations"],
        "query_steps": arrays["query_steps"],
        "actor_versions": arrays["actor_versions"],
    }


def concatenate_batches(first: dict, second: dict) -> dict:
    if isinstance(first, dict):
        if set(first) != set(second):
            raise ValueError("Batch structures differ")
        return {key: concatenate_batches(first[key], second[key]) for key in first}
    return np.concatenate([first, second], axis=0)


# ---------------------------------------------------------------------------
# Calibration arithmetic (host side, no JAX).


def monte_carlo_returns(rewards: np.ndarray, discounts: np.ndarray) -> np.ndarray:
    """Discounted shaped return from every transition of one complete episode."""
    result = np.zeros(len(rewards), np.float64)
    following = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        following = float(rewards[index]) + float(discounts[index]) * following
        result[index] = following
    return result


def calibration_set(episodes: list[dict[str, np.ndarray]], labels: list[str]) -> dict:
    """All transitions of complete episodes, in their recorded order."""
    if not episodes or len(episodes) != len(labels):
        raise ValueError("Calibration requires labelled complete episodes")
    parts: dict[str, list] = {
        key: []
        for key in (
            "pixels",
            "next_pixels",
            "states",
            "next_states",
            "actions",
            "rewards",
            "discounts",
            "masks",
            "mc_returns",
            "terminal",
            "episode",
        )
    }
    for label, episode in zip(labels, episodes, strict=True):
        count = len(episode["noise"])
        parts["pixels"].append(episode["pixels"][:-1])
        parts["next_pixels"].append(episode["pixels"][1:])
        parts["states"].append(episode["states"][:-1])
        parts["next_states"].append(episode["states"][1:])
        parts["actions"].append(episode["noise"])
        parts["rewards"].append(episode["rewards"])
        parts["discounts"].append(episode["discounts"])
        parts["masks"].append(episode["masks"])
        parts["mc_returns"].append(monte_carlo_returns(episode["rewards"], episode["discounts"]))
        parts["terminal"].append(np.arange(count) == count - 1)
        parts["episode"].extend([label] * count)
    result = {key: np.concatenate(value) for key, value in parts.items() if key != "episode"}
    result["episode"] = parts["episode"]
    return result


def _range_statistics(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, np.float64)
    if values.size == 0:
        return {"count": 0}
    finite = bool(np.isfinite(values).all())
    if not finite:
        return {"count": int(values.size), "finite": False}
    low, high = CALIBRATION_RANGE
    lower, upper = RETURN_RANGE
    excess = np.maximum(np.maximum(lower - values, values - upper), 0.0)
    return {
        "count": int(values.size),
        "finite": True,
        "fraction_outside_gate": float(np.mean((values < low) | (values > high))),
        "fraction_outside_return_range": float(np.mean(excess > 0)),
        "maximum_exceedance": float(excess.max()),
        "mean_exceedance": float(excess.mean()),
        "mean": float(values.mean()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def calibration_summary(raw: dict[str, np.ndarray], data: dict, reduction: str) -> dict:
    """The gate uses the learner's own estimate: its configured reduction of the ten
    heads, for every calibration pair. Per-head values are descriptive; a freshly
    initialized ensemble scatters individual heads widely (task 7: 26% outside)."""
    terminal = np.asarray(data["terminal"], bool)
    summary: dict[str, Any] = {"transitions": int(len(terminal)), "reduction": reduction}
    finite = True
    for name in ("recorded", "sampled"):
        heads = np.asarray(raw[f"qs_{name}"], np.float64)
        if heads.ndim != 2 or heads.shape[1] != len(terminal):
            raise ValueError("Calibration predictions have the wrong shape")
        reduced = heads.min(axis=0) if reduction == "min" else heads.mean(axis=0)
        statistics = _range_statistics(heads)
        finite &= statistics.get("finite", False)
        row = {
            "heads": statistics,
            "reduced": _range_statistics(reduced),
            "terminal_heads": _range_statistics(heads[:, terminal]),
            "nonterminal_heads": _range_statistics(heads[:, ~terminal]),
        }
        if statistics.get("finite"):
            row["head_spread_mean"] = float((heads.max(0) - heads.min(0)).mean())
            row["head_std_mean"] = float(heads.std(0).mean())
            if name == "recorded":
                error = reduced - np.asarray(data["mc_returns"], np.float64)
                row["monte_carlo_error_descriptive"] = {
                    "terminal_mean": float(error[terminal].mean()) if terminal.any() else None,
                    "nonterminal_mean": float(error[~terminal].mean())
                    if (~terminal).any()
                    else None,
                    "absolute_mean": float(np.abs(error).mean()),
                }
        summary[name] = row
    targets = np.asarray(raw["targets"], np.float64)
    summary["targets"] = {
        "all": _range_statistics(targets),
        "terminal": _range_statistics(targets[terminal]),
        "nonterminal": _range_statistics(targets[~terminal]),
    }
    finite &= summary["targets"]["all"].get("finite", False)
    log_probs = np.asarray(raw["log_probs"], np.float64)
    scale = np.asarray(raw["scale"], np.float64)
    loc = np.asarray(raw["loc"], np.float64)
    policy_finite = bool(
        np.isfinite(log_probs).all() and np.isfinite(scale).all() and np.isfinite(loc).all()
    )
    finite &= policy_finite
    summary["policy"] = (
        {
            "entropy_estimate": float(-log_probs.mean()),
            "pre_squash_mean_norm": float(np.linalg.norm(loc, axis=-1).mean()),
            "pre_squash_std_mean": float(scale.mean()),
            "pre_squash_std_min": float(scale.min()),
        }
        if policy_finite
        else {"finite": False}
    )
    summary["finite"] = bool(finite)
    # None marks a nonfinite measurement; it always fails the gate.
    summary["gate_quantity"] = f"{reduction} of ten critic heads"
    summary["gate_fractions"] = {
        name: summary[name]["reduced"]["fraction_outside_gate"]
        if summary[name]["heads"].get("finite")
        else None
        for name in ("recorded", "sampled")
    }
    summary["gate_failed"] = {
        name: value is None or calibration_failure(value)
        for name, value in summary["gate_fractions"].items()
    }
    return summary


def pause_status(history: list[dict]) -> dict[str, Any]:
    """Apply the three-successive rule separately to each action set.

    The series holds the measurements after every 500 updates. The initialization
    measurement is a descriptive baseline of the random ensemble, not of learning.
    """
    scheduled = [row for row in history if row.get("scheduled")]
    fired = {
        name: pause_rule_fired(
            [
                math.inf if row["gate_fractions"][name] is None else row["gate_fractions"][name]
                for row in scheduled
            ]
        )
        for name in ("recorded", "sampled")
    }
    nonfinite = any(not row.get("finite") for row in history)
    return {
        "measurements": len(scheduled),
        "fired": fired,
        "nonfinite": nonfinite,
        "pause": bool(nonfinite or any(fired.values())),
    }


# ---------------------------------------------------------------------------
# JAX backend.


class Dsrl11Backend(AuthorBackend):
    """The pinned author agent with a condition RNG and read-only calibration passes."""

    def __init__(self, source: Path, config: Dsrl11LearnerConfig, *, rng_seed: int | None):
        super().__init__(source, config)
        self.initial_rng_data = np.asarray(self.agent._rng).tolist()
        # None keeps the author's seed-derived stream (A1); conditions get their own.
        if rng_seed is not None:
            self.agent._rng = self.jax.random.PRNGKey(int(rng_seed))
        self.rng_seed = rng_seed
        self._forward = None

    def _calibration_function(self):
        if self._forward is None:
            jax, reduction = self.jax, self.config.critic_reduction

            def collections(state, params=None):
                values = {"params": state.params if params is None else params}
                if getattr(state, "batch_stats", None) is not None:
                    values["batch_stats"] = state.batch_stats
                return values

            def forward(actor, critic, target_params, batch, key):
                key_now, key_next = jax.random.split(key)
                observations = batch["observations"]
                qs_recorded = critic.apply_fn(collections(critic), observations, batch["actions"])
                dist = actor.apply_fn(collections(actor), observations)
                sampled, log_probs = dist.sample_and_log_prob(seed=key_now)
                qs_sampled = critic.apply_fn(collections(critic), observations, sampled)
                next_actions = actor.apply_fn(
                    collections(actor), batch["next_observations"]
                ).sample(seed=key_next)
                next_qs = critic.apply_fn(
                    collections(critic, target_params), batch["next_observations"], next_actions
                )
                next_q = next_qs.min(axis=0) if reduction == "min" else next_qs.mean(axis=0)
                return {
                    "qs_recorded": qs_recorded,
                    "qs_sampled": qs_sampled,
                    "targets": batch["rewards"] + batch["discount"] * batch["masks"] * next_q,
                    "log_probs": log_probs,
                    "loc": dist.distribution._loc,
                    "scale": dist.distribution._scale_diag,
                }

            self._forward = jax.jit(forward)
        return self._forward

    def calibrate(self, data: dict, key_seed: int, chunk: int = 256) -> dict[str, np.ndarray]:
        """Forward passes only: agent parameters, optimizer state and RNG are untouched."""
        forward = self._calibration_function()
        count = len(data["rewards"])
        outputs: dict[str, list] = {}
        base = self.jax.random.PRNGKey(int(key_seed))
        for chunk_index, start in enumerate(range(0, count, chunk)):
            index = np.arange(start, start + chunk) % count
            valid = min(chunk, count - start)
            batch = {
                "observations": {
                    "pixels": data["pixels"][index][..., None],
                    "state": data["states"][index][..., None],
                },
                "next_observations": {
                    "pixels": data["next_pixels"][index][..., None],
                    "state": data["next_states"][index][..., None],
                },
                "actions": data["actions"][index][:, None, :],
                "rewards": data["rewards"][index].astype(np.float32),
                "discount": data["discounts"][index].astype(np.float32),
                "masks": data["masks"][index].astype(np.float32),
            }
            result = forward(
                self.agent._actor,
                self.agent._critic,
                self.agent._target_critic_params,
                self.freeze(batch),
                self.jax.random.fold_in(base, chunk_index),
            )
            for name, value in self.jax.device_get(result).items():
                value = np.asarray(value)
                value = value[:, :valid] if name.startswith("qs_") else value[:valid]
                outputs.setdefault(name, []).append(value)
        return {
            name: np.concatenate(values, axis=1 if name.startswith("qs_") else 0)
            for name, values in outputs.items()
        }


# ---------------------------------------------------------------------------
# Condition state and bridge.


class PauseComparison(RuntimeError):
    """The frozen calibration or finiteness rule stopped all scientific updates."""


class ConditionLearner:
    def __init__(self, condition: str, config: Dsrl11LearnerConfig, backend):
        spec = CONDITION_SPEC[condition] if condition != ENGINEERING else {"reward": "r1"}
        self.condition = condition
        self.reward = spec["reward"]
        self.prior = bool(spec.get("prior", False))
        self.streams = condition_streams(condition)
        self.backend = backend
        self.online = ReplayStore(config.capacity, self.streams["online_replay"])
        self.bank = (
            ReplayStore(config.capacity, self.streams["bank_replay"]) if self.prior else None
        )
        self.episodes: list[dict] = []
        self.rounds: dict[str, dict] = {}
        self.trained: dict[str, dict] = {}
        self.published: dict[str, int] = {}
        self.interrupted: dict[str, str] = {}
        self.pending_transitions = 0
        self.pending_round: str | None = None
        self.actor_version = 0
        self.updates = 0
        self.calibration: dict | None = None
        self.calibration_labels: list[str] = []
        self.calibrations: list[dict] = []
        self.bank_calibration: list[dict] | None = None
        self.warmup_calibration: list[dict] | None = None

    def next_phase(self) -> str | None:
        done = [self.rounds[r]["phase"] for r in self.rounds]
        for phase in PHASES:
            if phase not in done:
                return phase
        return None

    def sample(self, config: Dsrl11LearnerConfig) -> dict:
        if not self.prior:
            return self.online.sample(config.batch_size)
        return concatenate_batches(
            self.bank.sample(config.prior_batch_size),
            self.online.sample(config.batch_size - config.prior_batch_size),
        )

    def metadata(self) -> dict:
        return {
            "condition": self.condition,
            "reward": self.reward,
            "prior": self.prior,
            "streams": self.streams,
            "episodes": self.episodes,
            "rounds": self.rounds,
            "trained": self.trained,
            "published": self.published,
            "interrupted": self.interrupted,
            "pending_transitions": self.pending_transitions,
            "pending_round": self.pending_round,
            "actor_version": self.actor_version,
            "updates": self.updates,
            "calibration_labels": self.calibration_labels,
            "calibrations": self.calibrations,
        }


class Dsrl11LearnerBridge:
    """Serial updates on one GPU, immutable publication and complete snapshots."""

    def __init__(
        self,
        work_dir: Path,
        source: Path,
        config: Dsrl11LearnerConfig,
        *,
        data_root: Path | None = None,
        backend_factory=Dsrl11Backend,
    ):
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.source, self.config = Path(source), config
        self.data_root = data_root
        self.backend_factory = backend_factory
        self.lock = threading.RLock()
        self.training_lock = threading.Lock()
        self.ledger_path = self.work_dir / "learner-ledger.jsonl"
        self.counts = {"engineering": 0, "learning": 0, "actor": 0}
        self.history_has_data = False
        if self.ledger_path.exists():
            for line in self.ledger_path.read_text().splitlines():
                event = json.loads(line)
                if event["event"] == "update_reserved":
                    self.counts[event["phase"]] += 1
                elif event["event"] == "actor_request":
                    self.counts["actor"] += 1
                elif event["event"] in ("round_appended", "bank_loaded"):
                    self.history_has_data = True
        self.phase = None
        self.conditions: dict[str, ConditionLearner] = {}
        self.bank: dict | None = None
        self.paused: dict | None = None
        self.active_training = False

    # -- journal and accounting ------------------------------------------------
    def event(self, name: str, **data) -> None:
        row = {"utc": utc(), "event": name, **data}
        with self.lock, self.ledger_path.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def reserve_update(self, phase: str, condition: str, label: str) -> None:
        with self.lock:
            cap = (
                self.config.engineering_update_cap
                if phase == "engineering"
                else self.config.learning_update_cap
            )
            if phase not in ("engineering", "learning") or self.counts[phase] >= cap:
                raise ValueError("Learner update cap exhausted")
            if (self.work_dir.parent / "STOP").exists():
                raise RuntimeError("Experiment admission stop file exists")
            self.event(
                "update_reserved",
                phase=phase,
                condition=condition,
                label=label,
                phase_update=self.counts[phase] + 1,
            )
            self.counts[phase] += 1

    # -- lifecycle -------------------------------------------------------------
    def _backend(self, condition: str):
        streams = condition_streams(condition)
        return self.backend_factory(self.source, self.config, rng_seed=streams["agent_rng"])

    def initialize(self, phase: str) -> dict:
        with self.lock:
            if phase not in ("engineering", "learning"):
                raise ValueError("Phase must be engineering or learning")
            if self.phase is not None or self.history_has_data:
                raise ValueError("A learner phase is initialized once; resume from a checkpoint")
            names = (ENGINEERING,) if phase == "engineering" else CONDITIONS
            self.conditions = {
                name: ConditionLearner(name, self.config, self._backend(name)) for name in names
            }
            self.phase = phase
            initial = {name: state_digest(state.backend) for name, state in self.conditions.items()}
            if phase == "learning" and len(set(initial.values())) != 1:
                raise ValueError("Conditions must share one identical network initialization")
            self.event(
                "initialized",
                phase=phase,
                seed=self.config.seed,
                conditions=list(names),
                initial_state_sha256=initial,
                config_sha256=json_hash(self.config.resolved()),
            )
            return self.status()

    def status(self) -> dict:
        with self.lock:
            return {
                "phase": self.phase,
                "initialized": bool(self.conditions),
                "paused": self.paused,
                "active_training": self.active_training,
                "bank_loaded": self.bank is not None,
                "counts": dict(self.counts),
                "conditions": {
                    name: {
                        "actor_version": state.actor_version,
                        "episodes": len(state.episodes),
                        "online_transitions": state.online.size,
                        "bank_transitions": state.bank.size if state.bank else 0,
                        "next_phase": state.next_phase(),
                        "pending_transitions": state.pending_transitions,
                        "pending_round": state.pending_round,
                        "updates": state.updates,
                        "calibrations": len(state.calibrations),
                        "interrupted": dict(state.interrupted),
                    }
                    for name, state in self.conditions.items()
                },
                "parameter_counts": next(
                    (s.backend.parameter_counts for s in self.conditions.values()), {}
                ),
                "resolved_config": self.config.resolved(),
            }

    def _state(self, condition) -> ConditionLearner:
        if condition not in self.conditions:
            raise ValueError("Unknown or uninitialized learner condition")
        return self.conditions[condition]

    def _check_open(self):
        if self.paused is not None:
            raise ValueError(f"Scientific comparison paused: {self.paused['reason']}")

    # -- actor -----------------------------------------------------------------
    def act(self, request: dict) -> dict:
        pixels, state = prepare_observation(
            pixels_from_base64(request["pixels_b64"]), request["state"]
        )
        index = request["query_index"]
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 250:
            raise ValueError("Query index is outside the native episode horizon")
        with self.lock:
            learner = self._state(request["condition"])
            if request["actor_version"] != learner.actor_version:
                raise ValueError("Actor version mismatch")
            seed = actor_seed(request["rng_seed"], request["job_id"])
            self.event(
                "actor_request",
                condition=learner.condition,
                job_id=request["job_id"],
                query_index=index,
                actor_version=learner.actor_version,
            )
            self.counts["actor"] += 1
            residual = learner.backend.sample(pixels, state, seed, index)
        if residual.shape != (32,) or not np.isfinite(residual).all():
            raise ValueError("Author actor returned an invalid residual")
        if np.max(np.abs(residual)) > 1.0:
            raise ValueError("Author actor residual exceeds [-1, 1]")
        return {
            "residual": residual.tolist(),
            "actor_version": learner.actor_version,
            "condition": learner.condition,
        }

    # -- replay ----------------------------------------------------------------
    def load_bank(self, request: dict) -> dict:
        descriptors = request["episodes"]
        fingerprint = json_hash(request)
        with self.lock:
            if self.phase != "learning" or self.active_training:
                raise ValueError("Only an idle learning bridge loads the prior bank")
            if self.bank is not None:
                if self.bank["fingerprint"] != fingerprint:
                    raise ValueError("A different prior bank was already loaded")
                return self.bank["response"]
            if any(state.episodes for state in self.conditions.values()):
                raise ValueError("The prior bank precedes all online replay")
            if len(descriptors) != 12 or [d["job_id"] for d in descriptors] != [
                f"B{i:02d}" for i in range(12)
            ]:
                raise ValueError("The frozen bank is exactly B00 through B11 in order")
            if int(request.get("strict_safe_completions", -1)) < BANK_SUCCESS_MINIMUM:
                raise ValueError("The bank did not meet its strict-safe completion gate")
            loaded = {}
            for name, state in self.conditions.items():
                episodes = [
                    load_residual_episode(d, self.config, state.reward, data_root=self.data_root)
                    for d in descriptors
                ]
                if state.prior:
                    for episode in episodes:
                        state.bank.append(episode)
                state.bank_calibration = [episodes[i] for i in (0, 4, 8)]
                loaded[name] = state.bank.size if state.bank else 0
            response = {
                "bank_episodes": [d["job_id"] for d in descriptors],
                "bank_transitions": loaded,
                "status": self.status(),
            }
            self.bank = {"fingerprint": fingerprint, "response": response, "request": request}
            self.history_has_data = True
            self.event("bank_loaded", fingerprint=fingerprint, transitions=loaded)
            return response

    def append(self, request: dict) -> dict:
        round_id = str(request["round_id"])
        descriptors = request["episodes"]
        phase = request["phase"]
        fingerprint = json_hash({k: request[k] for k in ("condition", "phase", "episodes")})
        with self.lock:
            self._check_open()
            learner = self._state(request["condition"])
            if self.phase != "learning" or self.bank is None or self.active_training:
                raise ValueError("Online replay requires an idle learner with its prior bank")
            if round_id in learner.rounds:
                if learner.rounds[round_id]["fingerprint"] != fingerprint:
                    raise ValueError("Round identity reused for different episode data")
                return learner.rounds[round_id]["response"]
            if phase != learner.next_phase() or len(descriptors) != 3:
                raise ValueError("Rounds append three episodes in frozen phase order")
            if learner.pending_transitions:
                raise ValueError("Prior round update credit must finish before appending")
            expected_version = PHASE_VERSIONS[phase]
            if phase != "warmup" and learner.actor_version != expected_version:
                raise ValueError("Collected batch did not use the published actor version")
            known = {row["job_id"] for row in learner.episodes}
            arrays = []
            for offset, descriptor in enumerate(descriptors):
                episode_index = len(learner.episodes) + offset
                if descriptor["job_id"] != f"T_{learner.condition}_{episode_index}":
                    raise ValueError("Episodes must be appended in frozen job order")
                if (
                    descriptor["job_id"] in known
                    or descriptor.get("condition") != learner.condition
                ):
                    raise ValueError("Episode is duplicated or belongs to another condition")
                if descriptor["actor_version"] != expected_version:
                    raise ValueError("Episode actor version differs from its frozen phase")
                arrays.append(
                    load_residual_episode(
                        descriptor, self.config, learner.reward, data_root=self.data_root
                    )
                )
            new_count = sum(len(a["noise"]) for a in arrays)
            if learner.online.size + new_count > self.config.capacity:
                raise ValueError("Round would overflow no-eviction replay")
            for descriptor, episode in zip(descriptors, arrays):
                learner.online.append(episode)
                learner.episodes.append({**descriptor, "phase": phase})
            if phase == "warmup":
                learner.warmup_calibration = arrays
            else:
                learner.pending_transitions = new_count
                learner.pending_round = round_id
            response = {
                "condition": learner.condition,
                "round_id": round_id,
                "phase": phase,
                "new_transitions": new_count,
                "update_credit": 0 if phase == "warmup" else new_count,
            }
            learner.rounds[round_id] = {
                "phase": phase,
                "fingerprint": fingerprint,
                "response": response,
            }
            self.history_has_data = True
            self.event(
                "round_appended",
                condition=learner.condition,
                round_id=round_id,
                phase=phase,
                transitions=new_count,
                jobs=[d["job_id"] for d in descriptors],
            )
            return response

    # -- calibration -----------------------------------------------------------
    def _calibration_data(self, learner: ConditionLearner) -> dict:
        if learner.calibration is None:
            bank, warmup = learner.bank_calibration, learner.warmup_calibration
            if bank is None or warmup is None:
                raise ValueError("Calibration needs bank B00/B04/B08 and the warm-up episodes")
            labels = ["B00", "B04", "B08"] + [
                f"T_{learner.condition}_{index}" for index in range(3)
            ]
            learner.calibration = calibration_set([*bank, *warmup], labels)
            learner.calibration_labels = labels
        return learner.calibration

    def _measure(self, learner: ConditionLearner, *, scheduled: bool, label: str) -> dict:
        data = self._calibration_data(learner)
        raw = learner.backend.calibrate(data, learner.streams["calibration_actions"])
        summary = calibration_summary(raw, data, self.config.critic_reduction)
        summary.update(
            condition=learner.condition,
            updates=learner.updates,
            scheduled=scheduled,
            label=label,
            utc=utc(),
        )
        summary = json_safe(summary)
        learner.calibrations.append(summary)
        self.event("calibration", **summary)
        return summary

    # -- updates ---------------------------------------------------------------
    def train(self, request: dict) -> dict:
        round_id = str(request["round_id"])
        with self.training_lock:
            with self.lock:
                self._check_open()
                learner = self._state(request["condition"])
                if round_id in learner.trained:
                    return learner.trained[round_id]
                if round_id in learner.interrupted:
                    raise ValueError("Interrupted round requires diagnosis; credit cannot reset")
                if learner.pending_round != round_id or not learner.pending_transitions:
                    raise ValueError("No eligible transition credit for this round")
                if request.get("expected_new_transitions") != learner.pending_transitions:
                    raise ValueError("Caller and replay disagree on eligible update credit")
                count = learner.pending_transitions * self.config.updates_per_transition
                if self.counts["learning"] + count > self.config.learning_update_cap:
                    raise ValueError("Round exceeds the frozen online update budget")
                self.active_training = True
                self.event(
                    "train_start", condition=learner.condition, round_id=round_id, updates=count
                )
            begin, last = time.monotonic(), {}
            try:
                interval = self.config.calibration_interval
                if not learner.calibrations:
                    self._measure(learner, scheduled=False, label="initialization")
                    self._pause_check(learner)
                for index in range(count):
                    self.reserve_update("learning", learner.condition, f"{round_id}:{index}")
                    last = learner.backend.update(learner.sample(self.config))
                    learner.updates += 1
                    if not all(np.isfinite(value) for value in last.values()):
                        self._pause(learner, "nonfinite source learner metrics", last)
                    if index % 100 == 0 or index + 1 == count:
                        self.event(
                            "training_metrics",
                            condition=learner.condition,
                            round_id=round_id,
                            index=index,
                            **json_safe(last),
                        )
                    if learner.updates % interval == 0:
                        self._measure(learner, scheduled=True, label=f"update{learner.updates}")
                        self._pause_check(learner)
                if learner.updates % interval:
                    self._measure(learner, scheduled=False, label=f"{round_id}:end")
                with self.lock:
                    learner.pending_transitions = 0
                    learner.pending_round = None
                    result = {
                        "condition": learner.condition,
                        "round_id": round_id,
                        "updates": count,
                        "condition_updates": learner.updates,
                        "elapsed_seconds": time.monotonic() - begin,
                        "last_metrics": last,
                        "last_calibration": learner.calibrations[-1],
                    }
                    learner.trained[round_id] = result
                    self.event("train_completed", **result)
                    return result
            except Exception as error:
                learner.interrupted[round_id] = type(error).__name__
                self.event(
                    "train_interrupted",
                    condition=learner.condition,
                    round_id=round_id,
                    error=type(error).__name__,
                    message=str(error),
                )
                raise
            finally:
                self.active_training = False

    def _pause_check(self, learner: ConditionLearner) -> None:
        status = pause_status(learner.calibrations)
        if status["pause"]:
            reason = (
                "nonfinite calibration state"
                if status["nonfinite"]
                else "calibration gate failed at three successive measurements"
            )
            self._pause(learner, reason, status)

    def _pause(self, learner: ConditionLearner, reason: str, detail: dict) -> None:
        with self.lock:
            self.paused = {
                "condition": learner.condition,
                "reason": reason,
                "updates": learner.updates,
                "detail": json_safe(detail),
                "utc": utc(),
            }
            self.event("comparison_paused", **self.paused)
        raise PauseComparison(reason)

    def publish(self, request: dict) -> dict:
        round_id = str(request["round_id"])
        with self.lock:
            self._check_open()
            learner = self._state(request["condition"])
            if self.active_training or learner.pending_transitions:
                raise ValueError("Cannot publish while updates are unfinished")
            if round_id in learner.published:
                return {
                    "condition": learner.condition,
                    "round_id": round_id,
                    "actor_version": learner.published[round_id],
                }
            if round_id not in learner.trained:
                raise ValueError("Only a completed learning round can publish a new actor")
            learner.backend.publish()
            learner.actor_version += 1
            learner.published[round_id] = learner.actor_version
            self.event(
                "actor_published",
                condition=learner.condition,
                round_id=round_id,
                actor_version=learner.actor_version,
                updates=learner.updates,
            )
            return {
                "condition": learner.condition,
                "round_id": round_id,
                "actor_version": learner.actor_version,
            }

    # -- engineering -----------------------------------------------------------
    def profile(self, request: dict) -> dict:
        with self.training_lock:
            with self.lock:
                if self.phase != "engineering":
                    raise ValueError("Live-batch profiling uses the separate engineering state")
                learner = self._state(ENGINEERING)
                updates = int(request["updates"])
                if not 0 <= updates or self.counts["engineering"] + updates + 2 > (
                    self.config.engineering_update_cap
                ):
                    raise ValueError("Engineering profiling update budget exceeded")
                self.active_training = True
            try:
                episodes = [
                    load_residual_episode(d, self.config, "r1", data_root=self.data_root)
                    for d in request["episodes"]
                ]
                replay = ReplayStore(self.config.capacity, derive_seed("profile-replay"))
                for episode in episodes:
                    replay.append(episode)
                batch = replay.sample(self.config.batch_size)
                result = {
                    "request_sha256": json_hash(request),
                    "live_episodes": [d["job_id"] for d in request["episodes"]],
                    "live_transitions": replay.size,
                    "source_parity": learner.backend.reference_parity(
                        batch,
                        lambda label: self.reserve_update("engineering", ENGINEERING, label),
                    ),
                }
                durations = []
                for index in range(updates):
                    self.reserve_update("engineering", ENGINEERING, f"profile:{index}")
                    begin = time.monotonic()
                    info = learner.backend.update(replay.sample(self.config.batch_size))
                    durations.append(time.monotonic() - begin)
                    if not all(np.isfinite(value) for value in info.values()):
                        raise ValueError("Nonfinite profiling update")
                learner.updates += updates + 2
                # The CPU/GPU parity check then uses trained rather than initial weights.
                learner.backend.publish()
                learner.actor_version += 1
                result.update(
                    timed_updates=updates,
                    seconds=sum(durations),
                    median_update_seconds=float(np.median(durations)) if durations else None,
                    updates_charged=updates + 2,
                    published_actor_version=learner.actor_version,
                )
                self.event("profile_completed", **result)
                return {**result, "status": self.status()}
            finally:
                self.active_training = False

    # -- snapshots -------------------------------------------------------------
    def save(self, name: str) -> dict:
        if not name or Path(name).name != name:
            raise ValueError("Checkpoint name must be a single path component")
        with self.lock:
            if not self.conditions or self.active_training:
                raise ValueError("Cannot checkpoint an uninitialized or actively updating learner")
            parent = self.work_dir / "checkpoints"
            parent.mkdir(exist_ok=True)
            destination, temporary = parent / name, parent / (name + ".partial")
            if destination.exists() or temporary.exists():
                raise ValueError("Checkpoints are immutable")
            temporary.mkdir()
            conditions = {}
            for condition, learner in self.conditions.items():
                (temporary / f"{condition}.learner.msgpack").write_bytes(learner.backend.dumps())
                online = learner.online.save(temporary / f"{condition}.online.npz")
                bank = (
                    learner.bank.save(temporary / f"{condition}.bank.npz") if learner.bank else None
                )
                conditions[condition] = {**learner.metadata(), "online": online, "bank": bank}
            atomic_json(
                temporary / "state.json",
                {
                    "schema": "dsrl11-learner-checkpoint-v1",
                    "utc": utc(),
                    "config": self.config.resolved(),
                    "phase": self.phase,
                    "counts": dict(self.counts),
                    "paused": self.paused,
                    "bank_request": self.bank["request"] if self.bank else None,
                    "conditions": conditions,
                },
            )
            manifest = {p.name: sha256(p) for p in temporary.iterdir() if p.is_file()}
            atomic_json(temporary / "manifest.json", manifest)
            temporary.rename(destination)
            result = {
                "path": str(destination),
                "manifest_sha256": sha256(destination / "manifest.json"),
            }
            self.event("checkpoint_saved", **result)
            return result

    def export_actor(self, request: dict) -> dict:
        name = request["name"]
        if not name or Path(name).name != name:
            raise ValueError("Actor export name must be a single path component")
        with self.lock:
            learner = self._state(request["condition"])
            if self.active_training or learner.pending_transitions:
                raise ValueError("Actor export requires an idle published actor")
            if int(request["actor_version"]) != learner.actor_version:
                raise ValueError("Requested export differs from the published actor version")
            parent = self.work_dir / "actor_exports"
            parent.mkdir(exist_ok=True)
            destination, temporary = parent / name, parent / (name + ".partial")
            if destination.exists() or temporary.exists():
                raise ValueError("Actor exports are immutable")
            temporary.mkdir()
            (temporary / "actor.msgpack").write_bytes(learner.backend.actor_dumps())
            atomic_json(
                temporary / "state.json",
                {
                    "schema": "dsrl11-published-actor-v1",
                    "utc": utc(),
                    "actor_evaluation_mode": "cpu_published_actor",
                    "config": self.config.resolved(),
                    "condition": learner.condition,
                    "actor_version": learner.actor_version,
                    "condition_updates": learner.updates,
                    "published_rounds": learner.published,
                },
            )
            manifest = {p.name: sha256(p) for p in temporary.iterdir() if p.is_file()}
            atomic_json(temporary / "manifest.json", manifest)
            temporary.rename(destination)
            result = {
                "path": str(destination),
                "manifest_sha256": sha256(destination / "manifest.json"),
                "condition": learner.condition,
                "actor_version": learner.actor_version,
                "actor_evaluation_mode": "cpu_published_actor",
            }
            self.event("actor_exported", **result)
            return result

    def load(self, path: str | Path) -> dict:
        path = Path(path).resolve(strict=True)
        with self.lock:
            if self.active_training or self.conditions:
                raise ValueError("Restore into a fresh learner process only")
            manifest = json.loads((path / "manifest.json").read_text())
            for name, expected in manifest.items():
                if Path(name).name != name or sha256(path / name) != expected:
                    raise ValueError("Checkpoint file hash mismatch")
            metadata = json.loads((path / "state.json").read_text())
            if json_hash(metadata["config"]) != json_hash(self.config.resolved()):
                raise ValueError("Checkpoint learner configuration mismatch")
            if (
                metadata["counts"]["learning"] != self.counts["learning"]
                or metadata["counts"]["engineering"] != self.counts["engineering"]
            ):
                raise ValueError("Uncheckpointed update attempts exist; refusing ambiguous resume")
            self.phase = metadata["phase"]
            for condition, saved in metadata["conditions"].items():
                learner = ConditionLearner(condition, self.config, self._backend(condition))
                learner.backend.loads((path / f"{condition}.learner.msgpack").read_bytes())
                learner.online.restore(path / f"{condition}.online.npz", saved["online"])
                if learner.bank is not None:
                    learner.bank.restore(path / f"{condition}.bank.npz", saved["bank"])
                for key in (
                    "episodes",
                    "rounds",
                    "trained",
                    "published",
                    "interrupted",
                    "pending_transitions",
                    "pending_round",
                    "actor_version",
                    "updates",
                    "calibration_labels",
                    "calibrations",
                ):
                    setattr(learner, key, saved[key])
                self.conditions[condition] = learner
            if metadata["bank_request"] is not None:
                self.bank = None
                request = metadata["bank_request"]
                # Calibration inputs are rebuilt from the same admitted, hash-checked files.
                for learner in self.conditions.values():
                    episodes = [
                        load_residual_episode(
                            d, self.config, learner.reward, data_root=self.data_root
                        )
                        for d in request["episodes"]
                    ]
                    learner.bank_calibration = [episodes[i] for i in (0, 4, 8)]
                    warmup = [e for e in learner.episodes if e["phase"] == "warmup"]
                    if warmup:
                        learner.warmup_calibration = [
                            load_residual_episode(
                                d, self.config, learner.reward, data_root=self.data_root
                            )
                            for d in warmup
                        ]
                self.bank = {
                    "fingerprint": json_hash(request),
                    "request": request,
                    "response": {"bank_episodes": [d["job_id"] for d in request["episodes"]]},
                }
            self.paused = metadata["paused"]
            self.event("checkpoint_restored", path=str(path))
            return self.status()


def state_digest(backend) -> str:
    """Hash of network parameters and optimizer state, excluding the condition RNG."""
    jax = backend.jax
    digest = hashlib.sha256()
    for path, value in jax.tree_util.tree_flatten_with_path(backend.agent._save_dict)[0]:
        array = np.asarray(value)
        digest.update(jax.tree_util.keystr(path).encode())
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def json_safe(value):
    """Journal-safe copy: nonfinite floats become strings instead of invalid JSON."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else repr(float(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    return value


class Dsrl11ActorBackend(AuthorActorBackend):
    """CPU author actor for an exported task-11 condition; no critic or optimizer."""


class FrozenResidualActors:
    """Read-only CPU actors for the four final exports, one request contract."""

    def __init__(self, work_dir: Path, source: Path, exports: dict[str, Path], *, factory=None):
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.work_dir / "actor-ledger.jsonl"
        self.lock = threading.RLock()
        self.active_training = False
        self.counts = {"actor": 0}
        self.actors: dict[str, dict] = {}
        factory = factory or Dsrl11ActorBackend
        if self.ledger_path.exists():
            for line in self.ledger_path.read_text().splitlines():
                event = json.loads(line)
                if event["event"] == "actor_request":
                    self.counts["actor"] += 1
        for condition, checkpoint in sorted(exports.items()):
            if condition not in (*CONDITIONS, ENGINEERING):
                raise ValueError("Unknown exported condition")
            checkpoint = Path(checkpoint).resolve(strict=True)
            manifest = json.loads((checkpoint / "manifest.json").read_text())
            if set(manifest) != {"actor.msgpack", "state.json"}:
                raise ValueError("CPU service requires actor-only export directories")
            for name, expected in manifest.items():
                if sha256(checkpoint / name) != expected:
                    raise ValueError("Published actor export hash mismatch")
            metadata = json.loads((checkpoint / "state.json").read_text())
            if (
                metadata["schema"] != "dsrl11-published-actor-v1"
                or metadata["condition"] != condition
            ):
                raise ValueError("Export schema or condition differs from its route")
            fields = {name: metadata["config"][name] for name in asdict(Dsrl11LearnerConfig())}
            config = Dsrl11LearnerConfig(**fields)
            if json_hash(config.resolved()) != json_hash(metadata["config"]):
                raise ValueError("Published actor export configuration mismatch")
            backend = factory(Path(source), config, (checkpoint / "actor.msgpack").read_bytes())
            self.actors[condition] = {
                "backend": backend,
                "actor_version": metadata["actor_version"],
                "manifest_sha256": sha256(checkpoint / "manifest.json"),
                "config": config,
            }
            self.event(
                "cpu_actor_loaded",
                condition=condition,
                actor_version=metadata["actor_version"],
                manifest_sha256=self.actors[condition]["manifest_sha256"],
            )

    event = Dsrl11LearnerBridge.event

    def act(self, request: dict) -> dict:
        pixels, state = prepare_observation(
            pixels_from_base64(request["pixels_b64"]), request["state"]
        )
        index = request["query_index"]
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 250:
            raise ValueError("Query index is outside the native episode horizon")
        with self.lock:
            actor = self.actors.get(request["condition"])
            if actor is None or request["actor_version"] != actor["actor_version"]:
                raise ValueError("Unknown condition or actor version")
            seed = actor_seed(request["rng_seed"], request["job_id"])
            self.event(
                "actor_request",
                condition=request["condition"],
                job_id=request["job_id"],
                query_index=index,
                actor_version=actor["actor_version"],
            )
            self.counts["actor"] += 1
            residual = actor["backend"].sample(pixels, state, seed, index)
        if residual.shape != (32,) or not np.isfinite(residual).all():
            raise ValueError("CPU actor returned an invalid residual")
        if np.max(np.abs(residual)) > 1.0:
            raise ValueError("CPU actor residual exceeds [-1, 1]")
        return {
            "residual": residual.tolist(),
            "actor_version": actor["actor_version"],
            "condition": request["condition"],
        }

    def status(self) -> dict:
        with self.lock:
            return {
                "phase": "evaluation",
                "actor_evaluation_mode": "cpu_published_actor",
                "actors": {
                    name: {
                        "actor_version": value["actor_version"],
                        "manifest_sha256": value["manifest_sha256"],
                    }
                    for name, value in self.actors.items()
                },
                "actor_requests": self.counts["actor"],
                "active_training": False,
            }
