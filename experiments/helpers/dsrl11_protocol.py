"""CPU protocol, schedules, rewards and accounting for the task-11 DSRL pilot.

This module imports neither a simulator nor a learning framework. It reuses the
task-6 scorer, transition rule and return identity unchanged and adds the residual
interface, the second (R1) potential and the fresh dsrl11 schedule and ceilings.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from helpers.dsrl06_protocol import (
    GAMMA,
    SHAPING_SCALE,
    EpisodeScorer,
    ProtocolLedger,
    benchmark_episode_seed,
    check_return_identity,
    reward_transition,
)

__all__ = [
    "GAMMA",
    "SHAPING_SCALE",
    "EpisodeScorer",
    "benchmark_episode_seed",
    "check_return_identity",
    "reward_transition",
]

NAMESPACE = "dsrl11-v1"
RESIDUAL_SCALE = 0.25
LATENT_SHAPE = (16, 32)
LEARNING_TASK = 3
ENGINEERING_TASKS = (1, 4, 10, 11)
CONDITIONS = ("r0_none", "r1_none", "r0_prior", "r1_prior")
CONDITION_SPEC = MappingProxyType(
    {
        "r0_none": MappingProxyType({"reward": "r0", "prior": False}),
        "r1_none": MappingProxyType({"reward": "r1", "prior": False}),
        "r0_prior": MappingProxyType({"reward": "r0", "prior": True}),
        "r1_prior": MappingProxyType({"reward": "r1", "prior": True}),
    }
)
PRIMARY_CONDITION = "r1_prior"
LEARNER_SEED = 11060
DIAGNOSTIC_SEEDS = (11061, 11062)
ENGINEERING_ROOTS = tuple(range(109000, 109008))
BANK_ROOTS = tuple(range(110000, 110012))
ONLINE_ROOTS = tuple(range(111000, 111009))
EVALUATION_ROOTS = tuple(range(112000, 112008))
PHASES = ("warmup", "batch_a", "batch_b")
COLLECTION_LANES = (0, 1, 2)
LEARNER_GPU = 3
BANK_SUCCESS_MINIMUM = 2
CAPS = MappingProxyType(
    {
        "starts": 98,
        "replacements": 2,
        "policy_controls": 196_000,
        "warmup_controls": 980,
        "model_queries": 27_000,
        "model_only_queries": 1_200,
        "learner_updates": 305_000,
        "diagnostic_updates": 180_000,
        "engineering_updates": 2_000,
        "learning_updates": 120_000,
    }
)
UPDATE_PHASES = ("diagnostic", "engineering", "learning")
# Reward geometry (metres, OmniGibson world frame, z up). See geometry_specification().
REACH_SCALE_M = 0.10
LIFT_SCALE_M = 0.10
GOAL_SCALE_M = 0.20
R1_WEIGHTS = MappingProxyType(
    {"reach": 0.10, "closed": 0.20, "grasp": 0.25, "lift": 0.20, "goal": 0.25}
)
CALIBRATION_RANGE = (-0.30, 1.05)
RETURN_RANGE = (-0.25, 1.0)
CALIBRATION_FRACTION = 0.01
CALIBRATION_INTERVAL = 500
PAUSE_SUCCESSIVE = 3


def derive_seed(stream: str, *parts: Any) -> int:
    """Stable 31-bit seed in the dsrl11 namespace, independent of Python's hash()."""
    if not stream:
        raise ValueError("A named RNG stream is required")
    payload = json.dumps([NAMESPACE, stream, *parts], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") & 0x7FFFFFFF


def uniform_residual(seed: int, query_index: int) -> np.ndarray:
    """Bounded uniform residual for warm-up and engineering, reconstructible per query."""
    sequence = np.random.SeedSequence([int(seed) & 0xFFFFFFFF, int(query_index)])
    rng = np.random.Generator(np.random.PCG64(sequence))
    return rng.uniform(-1.0, 1.0, 32).astype(np.float32)


def inject_residual(epsilon: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """Decoder noise epsilon + 0.25 * tile(u, 16), in float32 and never clipped."""
    epsilon = np.asarray(epsilon)
    residual = np.asarray(residual)
    if epsilon.shape != LATENT_SHAPE or epsilon.dtype != np.float32:
        raise ValueError("Native noise must be a float32 16 by 32 draw")
    if residual.shape != (32,) or residual.dtype != np.float32:
        raise ValueError("The residual action must be a float32 32-vector")
    if not (np.isfinite(epsilon).all() and np.isfinite(residual).all()):
        raise ValueError("Residual injection requires finite inputs")
    if np.max(np.abs(residual)) > 1.0:
        raise ValueError("The residual action lies outside [-1, 1]^32")
    injected = epsilon + np.float32(RESIDUAL_SCALE) * np.tile(residual, (LATENT_SHAPE[0], 1))
    injected = injected.astype(np.float32)
    if not np.any(residual) and not np.array_equal(injected, epsilon):
        raise ValueError("A zero residual must pass the native draw through exactly")
    return injected


def _job(stage, job_id, task, root, lane, source, **extra):
    episode_seed = benchmark_episode_seed(task, root)
    return {
        "job_id": job_id,
        "stage": stage,
        "task": task,
        "seed": root,
        "lane": lane,
        "mode": "residual",
        "residual_source": source,
        "episode_seed": episode_seed,
        "numpy_seed": episode_seed,
        "simulator_reset_seed": None,
        **extra,
    }


def condition_order() -> list[str]:
    """One seeded condition permutation, reused for every collection phase."""
    rng = np.random.Generator(np.random.PCG64(derive_seed("online-condition-order")))
    return [CONDITIONS[i] for i in rng.permutation(len(CONDITIONS))]


def frozen_jobs() -> dict[str, list[dict[str, Any]]]:
    engineering = []
    for index, task in enumerate(ENGINEERING_TASKS):
        job_id = f"E{index}"
        engineering.append(
            _job(
                "engineering",
                job_id,
                task,
                ENGINEERING_ROOTS[index],
                index,
                "zero",
                epsilon_seed=derive_seed("epsilon", "engineering", job_id),
                round=0,
            )
        )
    for index, task in enumerate(ENGINEERING_TASKS):
        job_id = f"E{index + 4}"
        engineering.append(
            _job(
                "engineering",
                job_id,
                task,
                ENGINEERING_ROOTS[index + 4],
                (index + 1) % 4,
                "uniform",
                epsilon_seed=derive_seed("epsilon", "engineering", job_id),
                uniform_seed=derive_seed("uniform", "engineering", job_id),
                round=1,
            )
        )
    bank = [
        _job(
            "bank",
            f"B{index:02d}",
            LEARNING_TASK,
            root,
            index % 4,
            "zero",
            epsilon_seed=derive_seed("epsilon", "bank", root),
            round=index // 4,
        )
        for index, root in enumerate(BANK_ROOTS)
    ]
    order = condition_order()
    online = []
    for phase_index, phase in enumerate(PHASES):
        for slot, condition in enumerate(order):
            round_index = phase_index * len(order) + slot
            for offset in range(3):
                episode = phase_index * 3 + offset
                root = ONLINE_ROOTS[episode]
                job_id = f"T_{condition}_{episode}"
                source = "uniform" if phase == "warmup" else "actor"
                extra = {
                    "condition": condition,
                    "phase": phase,
                    "episode_index": episode,
                    "round": round_index,
                    "epsilon_seed": derive_seed("epsilon", "online", condition, root),
                    "actor_version": {"warmup": -1, "batch_a": 0, "batch_b": 1}[phase],
                }
                if source == "uniform":
                    extra["uniform_seed"] = derive_seed("uniform", "online", condition, root)
                else:
                    extra["actor_rng_seed"] = derive_seed("actor", "online", condition, root)
                online.append(
                    _job(
                        "online",
                        job_id,
                        LEARNING_TASK,
                        root,
                        COLLECTION_LANES[(offset + phase_index) % 3],
                        source,
                        **extra,
                    )
                )
    arms = ("native", *order)
    evaluation = []
    for root_index, root in enumerate(EVALUATION_ROOTS):
        for arm in arms:
            index = len(evaluation)
            extra = {
                "arm": arm,
                "condition": None if arm == "native" else arm,
                "order": index,
                "epsilon_seed": derive_seed("epsilon", "evaluation", root),
                "actor_version": -1 if arm == "native" else 2,
            }
            if arm != "native":
                extra["actor_rng_seed"] = derive_seed("actor", "evaluation", arm, root)
            evaluation.append(
                _job(
                    "evaluation",
                    f"V{root_index}_{arm}",
                    LEARNING_TASK,
                    root,
                    index % 4,
                    "zero" if arm == "native" else "actor",
                    **extra,
                )
            )
    return {
        "engineering_jobs": engineering,
        "bank_jobs": bank,
        "online_jobs": online,
        "evaluation_jobs": evaluation,
    }


def all_jobs(schedule: dict[str, list[dict[str, Any]]] | None = None) -> dict[str, dict]:
    schedule = frozen_jobs() if schedule is None else schedule
    jobs = {job["job_id"]: job for group in schedule.values() for job in group}
    if len(jobs) != sum(len(group) for group in schedule.values()):
        raise ValueError("Frozen job identities must be unique")
    return jobs


def lane_sequences(jobs: list[dict[str, Any]]) -> dict[int, list[str]]:
    """Per-lane frozen order. A lane starts its next job once free and admitted."""
    lanes: dict[int, list[str]] = {}
    for job in jobs:
        lanes.setdefault(job["lane"], []).append(job["job_id"])
    return lanes


def residual_for(job: dict[str, Any], query_index: int) -> np.ndarray | None:
    """Return the reconstructible residual, or None for actor-sampled jobs."""
    source = job["residual_source"]
    if source == "zero":
        return np.zeros(32, np.float32)
    if source == "uniform":
        return uniform_residual(job["uniform_seed"], query_index)
    if source == "actor":
        return None
    raise ValueError("Unknown frozen residual source")


# ---------------------------------------------------------------------------
# R1 potential and its benchmark-grounded geometry.


def geometry_specification() -> dict[str, Any]:
    """Frozen definitions; every reference comes from pinned benchmark source."""
    return {
        "frame": "OmniGibson world frame, metres, z up",
        "reach": {
            "points": ["robot.get_eef_position() (Franka eef_link)", "jar root position"],
            "recorded_fields": ["eef_pos", "jar_pos"],
            "definition": "1 - tanh(||eef_pos - jar_pos|| / 0.10 m)",
            "source": "ManiGuard 6bf4c61 maniguard/eval/benchmark.py eef2target metric",
            "reason": "ManiGuard defines no jar grasp affordance; its own engagement metric "
            "measures the eef link to the target root, so reach uses exactly that pair",
            "finger_midpoint_note": "The plan's finger midpoint is represented by the pinned "
            "Franka eef_link grasp frame, used by the benchmark; finger link positions "
            "are recorded for audit only",
        },
        "closed": "native jar_closed proposition, latched at first observation",
        "grasp": "robot.is_grasping(jar) == IsGraspingState.TRUE (recorded enum == 1); "
        "current, not latched",
        "lift": "clip((jar_pos.z - jar_pos.z at step 0) / 0.10 m, 0, 1)",
        "goal": {
            "definition": "1 - tanh(d / 0.20 m), d = ||c - clamp(c, aabb_min, aabb_max)||",
            "goal_reference": "goal sphere centre_world from the native GoalRegionChecker",
            "jar_goal_reference": "closest point of the jar world AABB to the sphere centre",
            "source": "ManiGuard 6bf4c61 maniguard/utils/goal_region.py "
            "object_intersects_goal_region",
            "recorded_fields": ["jar_aabb_min", "jar_aabb_max", "goal_center_world"],
            "benchmark_success": "held and d <= radius_m",
        },
        "weights": dict(R1_WEIGHTS),
        "scales_m": {"reach": REACH_SCALE_M, "lift": LIFT_SCALE_M, "goal": GOAL_SCALE_M},
        "zeroing": "Phi1 = 0 after a raw violation, after a closure-order failure and at "
        "every terminal state",
        "information": "Privileged training reward only; never an actor or critic input",
    }


def _vector(value, name, shape=(3,)) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite array of shape {shape}")
    return array


def r1_components(
    *,
    eef_pos,
    jar_pos,
    initial_jar_pos,
    jar_aabb_min,
    jar_aabb_max,
    goal_center,
) -> dict[str, float]:
    eef = _vector(eef_pos, "eef_pos")
    jar = _vector(jar_pos, "jar_pos")
    initial = _vector(initial_jar_pos, "initial_jar_pos")
    low = _vector(jar_aabb_min, "jar_aabb_min")
    high = _vector(jar_aabb_max, "jar_aabb_max")
    center = _vector(goal_center, "goal_center")
    if np.any(low > high):
        raise ValueError("The jar AABB minimum exceeds its maximum")
    reach = 1.0 - math.tanh(float(np.linalg.norm(eef - jar)) / REACH_SCALE_M)
    lift = min(max((jar[2] - initial[2]) / LIFT_SCALE_M, 0.0), 1.0)
    closest = np.minimum(np.maximum(center, low), high)
    distance = float(np.linalg.norm(center - closest))
    goal = 1.0 - math.tanh(distance / GOAL_SCALE_M)
    return {"reach": reach, "lift": lift, "goal": goal, "goal_distance_m": distance}


def r1_potential(
    components: dict[str, float],
    *,
    closed: bool,
    grasp: bool,
    raw_violation: bool,
    order_failed: bool,
    terminal: bool = False,
) -> float:
    if terminal or raw_violation or order_failed:
        return 0.0
    for name in ("reach", "lift", "goal"):
        value = components[name]
        if not (math.isfinite(value) and 0.0 <= value <= 1.0):
            raise ValueError(f"R1 component {name} must lie in [0, 1]")
    c, g = float(bool(closed)), float(bool(grasp))
    weights = R1_WEIGHTS
    value = (
        weights["reach"] * components["reach"]
        + weights["closed"] * c
        + weights["grasp"] * c * g
        + weights["lift"] * c * g * components["lift"]
        + weights["goal"] * c * g * components["lift"] * components["goal"]
    )
    if not -1e-12 <= value <= 1 + 1e-12:
        raise ValueError("R1 weights and components must keep the potential in [0, 1]")
    # Only floating-point rounding of a mathematically bounded sum is absorbed here.
    return float(min(max(value, 0.0), 1.0))


def episode_rewards(potentials: list[float], steps: int, strict_safe: bool) -> list[dict]:
    """Build both-sided transitions from per-control potentials over a whole episode."""
    if len(potentials) != steps + 1:
        raise ValueError("One potential per control state, including state zero, is required")
    count = math.ceil(steps / 8)
    rows = []
    for index in range(count):
        start, end = index * 8, min((index + 1) * 8, steps)
        rows.append(
            reward_transition(
                potentials[start],
                potentials[end],
                end - start,
                terminal=index == count - 1,
                strict_safe_requested=bool(index == count - 1 and strict_safe),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Accounting.


class Dsrl11Ledger(ProtocolLedger):
    """Hash-chained reservations with dsrl11 identities and three update phases."""

    def __init__(self, path: str | Path, schedule: dict | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise ValueError("The accounting ledger cannot be a symlink")
        self.caps = MappingProxyType(dict(CAPS))
        self.job_ids = frozenset(all_jobs(schedule))
        self.summary()

    def _apply(self, state, event, data):
        usage, attempts = state["usage"], state["attempts"]
        if event == "start_reserved":
            job, attempt = data["job_id"], int(data["attempt"])
            if job not in self.job_ids:
                raise ValueError("Only frozen job identities may consume simulator starts")
            history = attempts.setdefault(job, [])
            if attempt != len(history) + 1 or attempt > 2:
                raise ValueError("Each frozen job has one original and at most one replacement")
            if history and history[-1].get("valid") is not False:
                raise ValueError("Only a recorded invalid attempt can be replaced")
            if history and not data.get("repair_receipt"):
                raise ValueError("A replacement requires its recorded repair receipt")
            usage["starts"] += 1
            usage["replacements"] += int(attempt > 1)
            history.append({"attempt": attempt, "valid": None})
        elif event == "start_finished":
            history = attempts.get(data["job_id"], [])
            if not history or history[-1]["attempt"] != data["attempt"]:
                raise ValueError("Cannot finish an unreserved attempt")
            if history[-1]["valid"] is not None:
                raise ValueError("Attempt has already been finished")
            if type(data["valid"]) is not bool:
                raise ValueError("An attempt verdict must be boolean")
            history[-1]["valid"] = data["valid"]
        elif event == "charge":
            kind, count = data["kind"], int(data["count"])
            if count < 0:
                raise ValueError("Charges cannot be negative")
            if kind == "learner_updates":
                phase = data.get("phase")
                if phase not in UPDATE_PHASES:
                    raise ValueError("Learner updates require diagnostic/engineering/learning")
                usage["learner_updates"] += count
                usage[f"{phase}_updates"] += count
            elif kind == "model_queries":
                usage["model_queries"] += count
                if data.get("model_only"):
                    usage["model_only_queries"] += count
            elif kind in {"policy_controls", "warmup_controls"}:
                usage[kind] += count
            else:
                raise ValueError("Unknown interaction counter")
        else:
            raise ValueError("Unknown accounting event")
        for kind, count in usage.items():
            if count > self.caps[kind]:
                raise ValueError(f"Frozen {kind} ceiling would be exceeded")


def calibration_failure(fraction_outside: float) -> bool:
    return not (math.isfinite(fraction_outside) and fraction_outside <= CALIBRATION_FRACTION)


def pause_rule_fired(fractions: list[float]) -> bool:
    """Stage-C pause: the 1% gate failing at three successive measurements."""
    run = 0
    for fraction in fractions:
        run = run + 1 if calibration_failure(fraction) else 0
        if run >= PAUSE_SUCCESSIVE:
            return True
    return False


def select_learner_setting(cells: dict[tuple[int, str], dict]) -> dict[str, Any]:
    """Apply the approved A1 rule: admission, then (20,min)>(20,mean)>(5,min)>(5,mean)."""
    preference = [(20, "min"), (20, "mean"), (5, "min"), (5, "mean")]
    if set(cells) != set(preference):
        raise ValueError("A1 must report all four grid cells")
    admitted = [key for key in preference if cells[key].get("admitted") is True]
    return {
        "rule": "admit iff finite and the stage-C pause rule never fires for both seeds and "
        "both action sets; prefer (20,min), (20,mean), (5,min), (5,mean)",
        "admitted": [list(key) for key in admitted],
        "selected": list(admitted[0]) if admitted else None,
    }
