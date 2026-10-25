"""CPU protocol, rewards, authority checks and interaction accounting for DSRL06.

This module imports neither a simulator nor a learning framework. Financial deadlines
and provider state are deliberately owned by the supervisor, separately from this ledger.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any
import zlib

import numpy as np


GAMMA = 0.9998
SHAPING_SCALE = 0.25
DEVELOPMENT_TASKS = (0, 3, 6, 9, 15, 17)
ENGINEERING_TASKS = (1, 4, 10, 11)
EVALUATION_ARMS = ("native", "fixed_zero", "random", "dsrl")
CAPS = MappingProxyType(
    {
        "starts": 80,
        "replacements": 2,
        "policy_controls": 160_000,
        "warmup_controls": 800,
        "model_queries": 21_000,
        "learner_updates": 120_000,
        "engineering_updates": 2_000,
        "learning_updates": 90_000,
    }
)


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _boolean(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    raise ValueError(f"{name} must be a recorded boolean predicate")


@dataclass
class EpisodeScorer:
    """History belongs to the scorer only, never to the actor observation.

    Observe the initial state at step 0, then every post-control state in order.
    The instantaneous goal must come from the unchanged native goal checker.
    """

    step: int = -1
    first_closed_step: int | None = None
    first_departure_step: int | None = None
    raw_violation: bool = False
    counted_violation: bool = False
    jar_closed_now: bool = False
    jar_on_support: bool = True
    grasped: bool = False
    instantaneous_goal: bool = False
    goal_hold_count: int = 0
    goal_confirmed: bool = False
    ever_grasped: bool = False

    @property
    def closed_once(self) -> bool:
        return self.first_closed_step is not None

    @property
    def order_failed(self) -> bool:
        return self.first_departure_step is not None and (
            self.first_closed_step is None or self.first_departure_step < self.first_closed_step
        )

    @property
    def requested_order(self) -> bool:
        return self.closed_once and not self.order_failed

    def observe(
        self,
        step: int,
        *,
        jar_closed: bool,
        jar_on_support: bool,
        grasped: bool,
        instantaneous_goal: bool,
        raw_violation: bool = False,
        counted_violation: bool = False,
    ) -> float:
        step = _integer(step, "state step")
        if step > 2000 or step != self.step + 1:
            raise ValueError("Observe the initial state and every control state exactly once")
        jar_closed = _boolean(jar_closed, "jar_closed")
        jar_on_support = _boolean(jar_on_support, "jar_on_support")
        grasped = _boolean(grasped, "grasped")
        instantaneous_goal = _boolean(instantaneous_goal, "instantaneous_goal")
        raw_violation = _boolean(raw_violation, "raw_violation")
        counted_violation = _boolean(counted_violation, "counted_violation")
        if counted_violation and not (self.raw_violation or raw_violation):
            raise ValueError("A counted violation cannot exist without a raw violation")
        self.step = step
        self.jar_closed_now = bool(jar_closed)
        self.jar_on_support = bool(jar_on_support)
        self.grasped = bool(grasped)
        self.instantaneous_goal = bool(instantaneous_goal)
        # Both predicates at this state are applied before testing order; equal steps are valid.
        if self.jar_closed_now and self.first_closed_step is None:
            self.first_closed_step = step
        if not self.jar_on_support and self.first_departure_step is None:
            self.first_departure_step = step
        self.raw_violation |= bool(raw_violation)
        self.counted_violation |= bool(counted_violation)
        self.ever_grasped |= self.grasped
        # The benchmark's ten-control goal hold excludes the initial, pre-control state.
        if step > 0:
            self.goal_hold_count = self.goal_hold_count + 1 if self.instantaneous_goal else 0
            self.goal_confirmed |= self.goal_hold_count >= 10
        return self.potential()

    def potential(self, terminal: bool = False) -> float:
        if terminal or self.raw_violation or self.order_failed or not self.closed_once:
            return 0.0
        if self.instantaneous_goal:
            return 1.0
        if self.grasped and not self.jar_on_support:
            return 0.75
        if self.grasped:
            return 0.50
        return 0.20

    def labels(
        self, success: bool, valid: bool = True, *, counted_violation: bool | None = None
    ) -> dict[str, Any]:
        valid = _boolean(valid, "valid")
        if counted_violation is not None:
            counted = _boolean(counted_violation, "final counted_violation")
            if counted and not self.raw_violation:
                raise ValueError("A final counted violation must have recorded raw evidence")
            if self.counted_violation and not counted:
                raise ValueError("The final counted violation cannot erase an observed count")
            self.counted_violation = counted
        if valid:
            success = _boolean(success, "success")
        if valid and self.step < 1:
            raise ValueError("A valid episode must execute at least one control")
        if valid and success != self.goal_confirmed:
            raise ValueError("Native goal success differs from the observed ten-control hold")
        requested = bool(success and self.requested_order) if valid else None
        return {
            "valid": bool(valid),
            "success": bool(success) if valid else None,
            "requested_success": requested,
            "strict_safe_requested": bool(requested and not self.raw_violation) if valid else None,
            "safe_requested": bool(requested and not self.counted_violation) if valid else None,
            "raw_violation": self.raw_violation,
            "counted_violation": self.counted_violation,
            "order_failed": self.order_failed,
            "closed_once": self.closed_once,
            "first_closed_step": self.first_closed_step,
            "first_departure_step": self.first_departure_step,
            "cost": int(self.raw_violation or self.order_failed) if valid else None,
            "unfinished": not bool(success) if valid else None,
            "ever_grasped": self.ever_grasped,
            "goal_hold_count": self.goal_hold_count,
            "goal_confirmed": self.goal_confirmed,
        }

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> EpisodeScorer:
        restored = cls(**state)
        restored.step = _integer(restored.step, "restored step", -1)
        restored.goal_hold_count = _integer(restored.goal_hold_count, "restored goal hold")
        for key in (
            "raw_violation",
            "counted_violation",
            "jar_closed_now",
            "jar_on_support",
            "grasped",
            "instantaneous_goal",
            "goal_confirmed",
            "ever_grasped",
        ):
            setattr(restored, key, _boolean(getattr(restored, key), f"restored {key}"))
        if not -1 <= restored.step <= 2000:
            raise ValueError("Invalid restored control state")
        for key in ("first_closed_step", "first_departure_step"):
            value = getattr(restored, key)
            if value is not None:
                value = _integer(value, f"restored {key}")
                setattr(restored, key, value)
            if value is not None and not 0 <= value <= restored.step:
                raise ValueError(f"Invalid restored {key}")
        if restored.counted_violation and not restored.raw_violation:
            raise ValueError("Restored counted violation has no raw violation")
        if restored.goal_hold_count < 0 or restored.goal_hold_count > max(restored.step, 0):
            raise ValueError("Invalid restored goal hold")
        if restored.goal_confirmed and restored.step < 10:
            raise ValueError("Invalid restored goal confirmation")
        if restored.goal_hold_count >= 10 and not restored.goal_confirmed:
            raise ValueError("Restored goal hold lacks its confirmation")
        return restored


def reward_transition(
    phi_before: float,
    phi_after: float,
    controls: int,
    *,
    terminal: bool,
    strict_safe_requested: bool = False,
    valid: bool = True,
) -> dict[str, Any]:
    """Construct one latent transition, forcing all valid terminal potentials to zero."""
    controls = _integer(controls, "executed controls", 1)
    terminal = _boolean(terminal, "terminal")
    strict_safe_requested = _boolean(strict_safe_requested, "strict_safe_requested")
    valid = _boolean(valid, "valid")
    if controls > 8 or (not terminal and controls != 8):
        raise ValueError(
            "Nonterminal transitions execute eight controls; terminal ones execute 1..8"
        )
    if not valid:
        raise ValueError("Interrupted attempts are not terminal replay data")
    if strict_safe_requested and not terminal:
        raise ValueError("Completion reward belongs only to the terminal transition")
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (phi_before, phi_after)):
        raise ValueError("Potential must be finite and bounded in [0, 1]")
    discount = GAMMA**controls
    final_phi = 0.0 if terminal else float(phi_after)
    base = float(terminal and strict_safe_requested)
    shaping = SHAPING_SCALE * (discount * final_phi - phi_before)
    return {
        "duration": controls,
        "discount": discount,
        "mask": float(not terminal),
        "terminal": bool(terminal),
        "phi_before": float(phi_before),
        "phi_after_raw": float(phi_after),
        "phi_after": final_phi,
        "base_reward": base,
        "shaping_reward": shaping,
        "reward": base + shaping,
    }


def check_return_identity(
    transitions: list[dict[str, Any]], *, atol: float = 1e-5
) -> dict[str, float]:
    """Verify the complete episode and return both discounted environmental returns."""
    if not transitions:
        raise ValueError("A complete episode has at least one transition")
    weight, shaped_return, base_return = 1.0, 0.0, 0.0
    for i, row in enumerate(transitions):
        terminal = i == len(transitions) - 1
        if bool(row["terminal"]) != terminal or row["mask"] != float(not terminal):
            raise ValueError("Only the final valid transition terminates bootstrap")
        expected = reward_transition(
            row["phi_before"],
            row["phi_after_raw"],
            row["duration"],
            terminal=terminal,
            strict_safe_requested=bool(row["base_reward"]),
        )
        for key in ("discount", "phi_after", "base_reward", "shaping_reward", "reward"):
            if not math.isclose(float(row[key]), expected[key], abs_tol=atol, rel_tol=0):
                raise ValueError(f"Invalid transition {key}")
        if i and not math.isclose(
            transitions[i - 1]["phi_after"], row["phi_before"], abs_tol=atol, rel_tol=0
        ):
            raise ValueError("Potential history is not contiguous across queries")
        shaped_return += weight * row["reward"]
        base_return += weight * row["base_reward"]
        weight *= row["discount"]
    corrected = shaped_return + SHAPING_SCALE * transitions[0]["phi_before"]
    error = corrected - base_return
    if not math.isclose(error, 0, abs_tol=atol, rel_tol=0):
        raise ValueError("Discounted shaped and base returns do not reconcile")
    return {
        "discounted_shaped_return": shaped_return,
        "discounted_base_return": base_return,
        "corrected_discounted_shaped_return": corrected,
        "return_identity_error": error,
    }


def derive_seed(job_id: str, stream: str, query_index: int | None = None) -> int:
    """Stable private stream seed; it is independent of Python's randomized hash()."""
    if not job_id or not stream:
        raise ValueError("Job identity and stream name are required")
    if query_index is not None:
        query_index = _integer(query_index, "query index")
    payload = json.dumps(["dsrl06-v1", job_id, stream, query_index], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") & 0x7FFFFFFF


def actor_query_seed(job_id: str, query_index: int) -> int:
    """Unused CPU stream helper, not the learner service's JAX sampling key.

    Runtime sampling uses the SHA256 base-key and JAX fold_in mapping declared by
    LearnerConfig.resolved; the server supplies the frozen job actor_rng_seed.
    """
    return derive_seed(job_id, "actor", query_index)


def benchmark_episode_seed(task: int, seed: int) -> int:
    """Match pinned ManiGuard's policy and numpy seed, not a nonexistent reset seed."""
    task, seed = _integer(task, "task"), _integer(seed, "seed")
    return zlib.crc32(f"{seed}:task_{task:04d}/base".encode()) & 0x7FFFFFFF


def frozen_jobs() -> dict[str, list[dict[str, Any]]]:
    def job(job_id, stage, task, seed, lane, mode, **extra):
        episode_seed = benchmark_episode_seed(task, seed)
        return {
            "job_id": job_id,
            "stage": stage,
            "task": task,
            "seed": seed,
            "lane": lane,
            "mode": mode,
            "episode_seed": episode_seed,
            "policy_rng_seed": episode_seed,
            "numpy_seed": episode_seed,
            "simulator_reset_seed": None,
            "actor_rng_seed": derive_seed(job_id, "actor"),
            **extra,
        }

    engineering = [
        job(f"E{i}", "engineering", task, 60000 + i, lane, mode)
        for i, (task, lane, mode) in enumerate(
            [
                (1, 0, "native"),
                (4, 1, "native"),
                (10, 2, "native"),
                (11, 3, "native"),
                (1, 0, "random"),
                (4, 1, "actor"),
            ]
        )
    ]
    for item in engineering:
        item["admission_requires"] = (
            ["E3_first_policy_response"]
            if item["job_id"] in ("E0", "E1", "E2")
            else ["authority_gate_passed"]
            if item["job_id"] in ("E4", "E5")
            else []
        )
    training = []
    for seed in range(61000, 61004):
        for task in DEVELOPMENT_TASKS:
            i = len(training)
            training.append(
                job(
                    f"T{i:02d}",
                    "training",
                    task,
                    seed,
                    i % 3,
                    "actor",
                    round=i // 3,
                    cycle=i // 6,
                    warmup_required=i < 6,
                    warmup_fallback_eligible=6 <= i < 12,
                )
            )
    evaluation = []
    for seed in (62000, 62001):
        for task in DEVELOPMENT_TASKS:
            block = len(evaluation) // 4
            for arm_index, arm in enumerate(EVALUATION_ARMS):
                mode = {"fixed_zero": "zero", "dsrl": "actor"}.get(arm, arm)
                evaluation.append(
                    job(
                        f"V{block:02d}_{arm}",
                        "evaluation",
                        task,
                        seed,
                        (arm_index + block) % 4,
                        mode,
                        arm=arm,
                        block=block,
                    )
                )
    return {
        "engineering_jobs": engineering,
        "training_jobs": training,
        "evaluation_jobs": evaluation,
    }


def decoder_vectors() -> tuple[list[str], np.ndarray]:
    labels = ["zero", "coordinate7_plus", "coordinate7_minus", "all_ones"]
    e7 = np.eye(32, dtype="<f4")[7]
    vectors = [np.zeros(32, dtype="<f4"), e7, -e7, np.ones(32, dtype="<f4")]
    for bit in range(5):
        row = np.asarray([1 if not (k >> bit & 1) else -1 for k in range(32)], dtype="<f4")
        labels.extend([f"walsh_bit{bit}_plus", f"walsh_bit{bit}_minus"])
        vectors.extend([row, -row])
    return labels, np.stack(vectors).astype("<f4")


def gripper_commands(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite gripper output invalidates the probe")
    return np.where(values > 0.01, 1.0, -1.0).astype(np.float32)


def select_probe_snapshots(episodes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Select by frozen causal rule before seeing any candidate decoder response.

    Each E0..E3 entry has task and queries. Query records have step, closed_once,
    raw_violation and order_failed; other fields (paths, hashes, snapshot_id) pass through.
    """
    selected = []
    for episode_id, task in zip(("E0", "E1", "E2", "E3"), ENGINEERING_TASKS, strict=True):
        episode = episodes[episode_id]
        if episode["task"] != task:
            raise ValueError("Probe pool task changed")
        queries = sorted(episode["queries"], key=lambda q: q["step"])
        if not queries or queries[0]["step"] != 0:
            raise ValueError("Missing initial policy query")
        if len({q["step"] for q in queries}) != len(queries):
            raise ValueError("Duplicate query step")
        eligible = [
            q for q in queries if q["step"] > 0 and not q["raw_violation"] and not q["order_failed"]
        ]
        after_closure = [q for q in eligible if q["closed_once"]]
        second = (
            after_closure[0]
            if after_closure
            else next((q for q in reversed(eligible) if q["step"] <= 800), None)
        )
        if second is None or queries[0]["raw_violation"] or queries[0]["order_failed"]:
            raise ValueError("Insufficient distinct nonviolating pool snapshots")
        for slot, query in enumerate((queries[0], second)):
            selected.append({**query, "task": task, "episode_id": episode_id, "slot": slot})
    return selected


def evaluate_authority(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate eight completed probes; no measurements are silently dropped.

    Each snapshot has task, snapshot_id, reference (16,8), native_replays (2,16,8),
    and candidates (14,16,8) in decoder_vectors order, all after native transforms.
    """
    result = {"valid": False, "admit": False, "snapshots": [], "reason": None}
    try:
        if len(snapshots) != 8:
            raise ValueError("Exactly eight frozen snapshots are required")
        if Counter(s["task"] for s in snapshots) != Counter({t: 2 for t in ENGINEERING_TASKS}):
            raise ValueError("Frozen pool-task representation changed")
        if len({s["snapshot_id"] for s in snapshots}) != 8:
            raise ValueError("Duplicate snapshot identity")
        for snapshot in snapshots:
            arrays = {}
            for key, shape in (
                ("reference", (16, 8)),
                ("native_replays", (2, 16, 8)),
                ("candidates", (14, 16, 8)),
            ):
                array = np.asarray(snapshot[key])
                if (
                    array.shape != shape
                    or not np.issubdtype(array.dtype, np.floating)
                    or not np.isfinite(array).all()
                ):
                    raise ValueError(f"Invalid probe {key}")
                arrays[key] = array
            reference, replays, candidates = (
                arrays["reference"],
                arrays["native_replays"],
                arrays["candidates"],
            )
            native_commands = gripper_commands(reference[:8, 7])
            if not np.allclose(replays, reference[None], atol=1e-5, rtol=1e-5):
                raise ValueError("Native decoded replay differs from the factual chunk")
            if not np.array_equal(
                gripper_commands(replays[:, :8, 7]), np.broadcast_to(native_commands, (2, 8))
            ):
                raise ValueError("Native replay changed a converted gripper command")
            grip = candidates[:, :8, 7]
            positions = np.flatnonzero((grip.max(axis=0) >= 0.02) & (grip.min(axis=0) <= 0))
            commands = gripper_commands(grip)
            result["snapshots"].append(
                {
                    "snapshot_id": snapshot["snapshot_id"],
                    "task": snapshot["task"],
                    "sensitive": bool(len(positions)),
                    "sensitive_prefix_positions": positions.tolist(),
                    "raw_gripper_min": float(grip.min()),
                    "raw_gripper_max": float(grip.max()),
                    "command_changes_against_zero": (commands != commands[0]).sum(axis=1).tolist(),
                    "command_changes_against_native": (commands != native_commands)
                    .sum(axis=1)
                    .tolist(),
                    "arm_rms_against_zero": np.sqrt(
                        np.mean((candidates[:, :8, :7] - candidates[0, :8, :7]) ** 2, axis=(1, 2))
                    ).tolist(),
                    "arm_rms_against_native": np.sqrt(
                        np.mean((candidates[:, :8, :7] - reference[:8, :7]) ** 2, axis=(1, 2))
                    ).tolist(),
                }
            )
        sensitive = [s for s in result["snapshots"] if s["sensitive"]]
        tasks = sorted({s["task"] for s in sensitive})
        result.update(
            valid=True,
            sensitive_snapshots=len(sensitive),
            sensitive_tasks=tasks,
            admit=len(sensitive) >= 2 and len(tasks) >= 2,
        )
        result["reason"] = "authority_present" if result["admit"] else "authority_below_minimum"
    except (KeyError, TypeError, ValueError) as error:
        result["reason"] = str(error)
    return result


class ProtocolLedger:
    """Append and fsync reservations before work; replay never resets consumed budget.

    One supervisor owns accounting. The lock also prevents accidental concurrent
    appends. Query-server journals remain separate and are aggregated by that owner.
    A truncated or modified ledger is rejected, not silently repaired or ignored.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise ValueError("The accounting ledger cannot be a symlink")
        self.caps = MappingProxyType(dict(CAPS))
        self.job_ids = frozenset(job["job_id"] for jobs in frozen_jobs().values() for job in jobs)
        self.summary()  # Validate existing state before a resume can be admitted.

    @staticmethod
    def _canonical(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

    def _read(self, handle):
        handle.seek(0)
        raw = handle.read()
        if raw and not raw.endswith("\n"):
            raise ValueError("Accounting ledger ends in an incomplete record")
        rows, previous = [], "0" * 64
        state = {"usage": {k: 0 for k in self.caps}, "attempts": {}, "event_ids": {}}
        for i, line in enumerate(raw.splitlines()):
            row = json.loads(line)
            payload = {k: v for k, v in row.items() if k != "sha256"}
            if row["sequence"] != i or row["previous_sha256"] != previous:
                raise ValueError("Accounting sequence or hash chain is broken")
            digest = hashlib.sha256(self._canonical(payload).encode()).hexdigest()
            if row["sha256"] != digest:
                raise ValueError("Accounting record was modified")
            self._apply(state, row["event"], row["data"])
            if row.get("event_id") is not None:
                event_id = row["event_id"]
                if event_id in state["event_ids"]:
                    raise ValueError("Duplicate accounting event identity")
                state["event_ids"][event_id] = (row["event"], row["data"])
            rows.append(row)
            previous = digest
        return state, rows, previous

    def _apply(self, state, event, data):
        usage, attempts = state["usage"], state["attempts"]
        if event == "start_reserved":
            job, attempt = data["job_id"], _integer(data["attempt"], "attempt", 1)
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
            kind, count = data["kind"], _integer(data["count"], "charged count")
            if kind not in {
                "policy_controls",
                "warmup_controls",
                "model_queries",
                "learner_updates",
            }:
                raise ValueError("Unknown interaction counter")
            usage[kind] += count
            if kind == "learner_updates":
                phase = data["phase"]
                if phase not in ("engineering", "learning"):
                    raise ValueError("Learner updates require engineering or learning phase")
                usage[f"{phase}_updates"] += count
        else:
            raise ValueError("Unknown accounting event")
        for kind, count in usage.items():
            if count > self.caps[kind]:
                raise ValueError(f"Frozen {kind} ceiling would be exceeded")

    def _operation(self, event=None, data=None, event_id=None):
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            state, rows, previous = self._read(handle)
            if event is not None:
                if event_id is not None and event_id in state["event_ids"]:
                    if state["event_ids"][event_id] != (event, data):
                        raise ValueError(
                            "An accounting event identity was reused with different data"
                        )
                    return state
                self._apply(state, event, data)
                payload = {
                    "schema": "dsrl06-ledger-v1",
                    "sequence": len(rows),
                    "previous_sha256": previous,
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "event": event,
                    "data": data,
                    "event_id": event_id,
                }
                digest = hashlib.sha256(self._canonical(payload).encode()).hexdigest()
                handle.seek(0, os.SEEK_END)
                handle.write(self._canonical({**payload, "sha256": digest}) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return state

    def summary(self):
        state = self._operation()
        return {"usage": state["usage"], "attempts": state["attempts"]}

    def reserve_start(self, job_id: str, attempt: int = 1, *, event_id=None, **details):
        if not job_id:
            raise ValueError("A frozen job identity is required")
        attempt = _integer(attempt, "attempt", 1)
        return self._operation(
            "start_reserved", {"job_id": job_id, "attempt": attempt, **details}, event_id
        )["usage"]

    def finish_start(self, job_id: str, attempt: int, *, valid: bool, event_id=None, **details):
        attempt = _integer(attempt, "attempt", 1)
        valid = _boolean(valid, "valid")
        return self._operation(
            "start_finished",
            {"job_id": job_id, "attempt": attempt, "valid": valid, **details},
            event_id,
        )["usage"]

    def charge(self, kind: str, count: int = 1, *, phase=None, event_id=None, **details):
        count = _integer(count, "charged count")
        if phase == "training":
            phase = "learning"
        return self._operation(
            "charge", {"kind": kind, "count": count, "phase": phase, **details}, event_id
        )["usage"]
