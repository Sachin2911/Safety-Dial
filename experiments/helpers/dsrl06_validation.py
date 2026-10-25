"""Offline admission of DSRL06 recordings, rewards and lossless learner episodes.

``build_episode`` never runs a policy or simulator. Native query captures remain
useful authority snapshots, but their untiled Gaussian tensors never enter replay.
Absolute paths in preserved receipts can be relocated only through explicit prefix
mapping, for example ``path_map={"/remote/run": "/local/copied-run"}``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from .dsrl06_protocol import (
    EpisodeScorer,
    check_return_identity,
    frozen_jobs,
    reward_transition,
)
from .dsrl06_server import ACTOR_TRANSFORM, actor_observation
from .tracking import load_recording, validate_recording


GRASP_VALIDITY = "native-enum-pm1-held-parity-v1"
GRASP_SOURCE_PROOF = {
    "behavior_revision": "88454bd04f75dc57c00ab1f1a00bcde1ff505950",
    "maniguard_revision": "6bf4c61a1f168cc46e02eb22c41bcb51815e9726",
    "controller_base.py": "855a338190d3794f4f078b25ba6588e39708729221c85af0f8b0b12d5538c12b",
    "manipulation_robot.py": "7a565f80c06a11f2eaa9d7514edbebf5772ee7b3aa73ad765cae4406c169c4ed",
    "goal_region.py": "e7e4bef5fee97000ed9df0b5d360e48ffb6eb32e1460d7707c451ae465748f82",
    "goal_checker.py": "ace47ce46cc0a540c759b0f5d0c89f050bdebbe83877e6052e770bf09acc74b0",
    "receipt": "progress/receipts/06-dsrl-grasp-repair.md",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def _resolve(value, directory: Path, path_map: Mapping[str, str] | None) -> Path:
    path = Path(value)
    for old, new in sorted((path_map or {}).items(), key=lambda item: -len(str(item[0]))):
        try:
            path = Path(new) / path.relative_to(Path(old))
            break
        except ValueError:
            continue
    if not path.is_absolute():
        path = directory / path
    if not path.is_file():
        raise ValueError(f"Missing recorded artifact: {path}")
    return path.resolve()


def _one(paths, label: str) -> Path:
    paths = sorted(set(Path(path).resolve() for path in paths))
    if len(paths) != 1:
        raise ValueError(f"Expected one {label}, found {len(paths)}; supply its explicit path")
    return paths[0]


def _recording(directory: Path) -> Path:
    candidates = []
    for path in directory.rglob("*.npz"):
        if path.name.endswith(".final_observation.npz"):
            continue
        with np.load(path, allow_pickle=False) as source:
            if "meta" not in source:
                continue
            meta = json.loads(str(source["meta"]))
            if meta.get("dsrl06_schema") == "dsrl06-recording-v1":
                candidates.append(path)
    return _one(candidates, "DSRL06 recording")


def _boolean_rows(data, key: str, count: int) -> np.ndarray:
    value = np.asarray(data[key])
    if value.shape != (count,) or value.dtype != np.bool_:
        raise ValueError(f"{key} must contain N+1 recorded booleans")
    return value


def _observation(arrays, expected_state, prompt: str) -> tuple[np.ndarray, np.ndarray]:
    for name in ("observation_image_left", "observation_wrist_image"):
        image = arrays[name]
        if image.shape != (256, 256, 3) or image.dtype != np.uint8:
            raise ValueError("Query observations must preserve both native RGB256 cameras")
    raw_state = arrays["observation_state"]
    if raw_state.shape != (8,) or not np.isfinite(raw_state).all():
        raise ValueError("Query observation has invalid native robot state")
    if arrays["prompt"].shape != () or str(arrays["prompt"]) != prompt:
        raise ValueError("Saved prompt differs from the native episode prompt")
    pixels, state = actor_observation(
        {"observation/image_left": arrays["observation_image_left"], "observation/state": raw_state}
    )
    if arrays["pixels"].dtype != np.uint8 or not np.array_equal(arrays["pixels"], pixels):
        raise ValueError("Actor pixels do not match the frozen lossless RGB64 transform")
    if arrays["state"].dtype != np.float32 or not np.array_equal(arrays["state"], state):
        raise ValueError("Actor state is not the native float32 robot state")
    if not np.array_equal(state, np.asarray(expected_state, dtype=np.float32)):
        raise ValueError("Query observation state is not aligned with its control boundary")
    return pixels, state


def _scorer_history(data, meta, row, journal):
    count = row["steps"] + 1
    for key in ("success", "ltl_violated", "counted_violation", "ever_grasped"):
        if not isinstance(row[key], bool):
            raise ValueError(f"Native result {key} is not a recorded boolean")
    flags = {
        key: _boolean_rows(data, key, count)
        for key in (
            "raw_violation",
            "counted_violation",
            "goal_instantaneous",
            "goal_checked",
            "goal_held",
        )
    }
    if not flags["goal_checked"].all() or not data["ap_valid"].all():
        raise ValueError("Scorer history contains an unchecked predicate")
    grasping = data["grasping"]
    if (
        grasping.shape != (count,)
        or not np.issubdtype(grasping.dtype, np.integer)
        or np.any(~np.isin(grasping, [-1, 1]))
    ):
        raise ValueError("Grasp enums must be native TRUE=1 or FALSE=-1; UNKNOWN=0 is ambiguous")
    for name in ("raw_violation_count", "raw_violation_step"):
        values = data[name]
        if values.shape != (count,) or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{name} must contain N+1 integers")
    raw = flags["raw_violation"]
    counts, steps = data["raw_violation_count"], data["raw_violation_step"]
    if np.any(np.diff(raw.astype(int)) < 0) or np.any(np.diff(counts) < 0):
        raise ValueError("Raw monitor history cannot erase a violation")
    first = int(np.flatnonzero(raw)[0]) if raw.any() else -1
    if np.any(counts < 0) or np.any(counts[~raw] != 0) or np.any(counts[raw] < 1):
        raise ValueError("Raw violation counts disagree with the native flag")
    if np.any(steps[~raw] != -1) or np.any(steps[raw] != first):
        raise ValueError("Raw violation step disagrees with its first observed state")
    expected_violation_step = None if first == -1 else first
    if (
        bool(row["ltl_violated"]) != bool(raw[-1])
        or row["ltl_violation_step"] != expected_violation_step
        or row["ltl_violation_count"] != int(counts[-1])
    ):
        raise ValueError("Raw monitor history differs from the native result")
    contact = row["first_contact_step"]
    counted = raw & (steps >= contact) if contact is not None else np.zeros(count, bool)
    if not np.array_equal(flags["counted_violation"], counted):
        raise ValueError("Counted violation history differs from the native contact gate")
    if bool(row["counted_violation"]) != bool(counted[-1]):
        raise ValueError("Counted violation differs from the native result")
    names = meta["ap_names"]
    if len(names) != len(set(names)) or not {"jar_closed", "jar_on_support"} <= set(names):
        raise ValueError("Jar scorer propositions are missing or ambiguous")
    propositions = {name: data["ap"][:, index] for index, name in enumerate(names)}
    states = [entry for entry in journal if entry["event"] == "scorer_state"]
    if [entry["step"] for entry in states] != list(range(count)):
        raise ValueError("Scorer journal must contain every state in order, including state zero")
    rule = meta.get("dsrl_grasp_validity")
    if rule not in (None, GRASP_VALIDITY):
        raise ValueError("Unknown recorded grasp-validity interpretation")
    old_validity = np.asarray(data["ap_valid"], bool) & (grasping >= 0)
    recorded_valid = meta.get("dsrl_recording_valid")
    if type(recorded_valid) is not bool:
        raise ValueError("Recorder validity must be an explicit boolean")
    expected_meta_valid = bool(old_validity.all()) if rule is None else True
    if recorded_valid is not expected_meta_valid:
        raise ValueError("Recorded episode validity cannot be explained by its declared grasp rule")
    scorer, histories, potentials, corrected_steps = EpisodeScorer(), [], [], []
    for step, entry in enumerate(states):
        fields = {
            "jar_closed": bool(propositions["jar_closed"][step]),
            "jar_on_support": bool(propositions["jar_on_support"][step]),
            "grasped": bool(data["grasping"][step] == 1),
            "instantaneous_goal": bool(flags["goal_instantaneous"][step]),
            "raw_violation": bool(raw[step]),
            "counted_violation": bool(counted[step]),
        }
        expected_valid = bool(old_validity[step]) if rule is None else True
        if entry.get("valid") is not expected_valid or any(
            entry.get(k) != v for k, v in fields.items()
        ):
            raise ValueError("Scorer journal disagrees with recorded predicate arrays")
        held = entry["goal_detail"].get("held")
        if (
            type(held) is not bool
            or held != bool(flags["goal_held"][step])
            or held != bool(grasping[step] == 1)
        ):
            raise ValueError(
                "Native held, recorded goal_held and raw TRUE enum must agree at every state"
            )
        if not expected_valid:
            corrected_steps.append(step)
        potentials.append(scorer.observe(step, **fields))
        histories.append(scorer.state_dict())
    labels = scorer.labels(bool(row["success"]), counted_violation=bool(row["counted_violation"]))
    first_hold = next((state["step"] for state in histories if state["goal_confirmed"]), None)
    if row["success"] and (first_hold != row["steps"] or row["success_step"] != first_hold):
        raise ValueError("Native rollout did not stop at its first ten-control goal hold")
    if labels["ever_grasped"] != bool(row["ever_grasped"]):
        raise ValueError("Native engagement label differs from recorded grasp history")
    # A native unsuccessful completion is the 2,000-control timeout, never a crash.
    if row.get("nan_terminated", False) or (not row["success"] and row["steps"] != 2000):
        raise ValueError("Incomplete or invalid episodes cannot enter the learning objective")
    audit = {
        "schema": "dsrl06-derived-grasp-audit-v1",
        "correction_applied": bool(corrected_steps),
        "recorded_validity_rule": rule or "legacy-ap-valid-and-raw-grasp-greater-or-equal-zero",
        "derived_validity_rule": GRASP_VALIDITY,
        "recorded_dsrl_recording_valid": recorded_valid,
        "derived_dsrl_recording_valid": True,
        "corrected_scorer_state_valid_steps": corrected_steps,
        "corrected_state_count": len(corrected_steps),
        "states_verified": count,
        "native_enums": {"TRUE": 1, "UNKNOWN": 0, "FALSE": -1},
        "canonical_grasp_predicate": "raw grasp enum == TRUE(1), equal to native goal_detail.held",
        "unknown_states": 0,
        "held_agreement_states": count,
        "raw_artifacts_modified": False,
        "source_proof": GRASP_SOURCE_PROOF,
    }
    return labels, histories, potentials, audit


def _frozen_job(meta, row, directory, job_schedule=None):
    settings = meta["dsrl_settings"]
    schedule = frozen_jobs() if job_schedule is None else job_schedule
    jobs = {item["job_id"]: item for group in schedule.values() for item in group}
    if settings.get("job_id") not in jobs:
        raise ValueError("Episode does not belong to the supplied frozen job list")
    job = jobs[settings["job_id"]]
    if (
        row["scene_name"] != f"task_{job['task']:04d}/base"
        or row["seed"] != job["seed"]
        or row["episode_seed"] != job["episode_seed"]
    ):
        raise ValueError("Native scene and seeds differ from the frozen job")
    modes = {job["mode"]}
    if job["stage"] == "training" and job.get("warmup_required", False):
        modes.add("gaussian")
    if job["stage"] == "training" and job.get("warmup_fallback_eligible", False):
        modes.add("gaussian")
    if settings.get("mode") not in modes or settings.get("rng_seed") != job["actor_rng_seed"]:
        raise ValueError("Episode mode or exploration seed differs from its frozen job")
    if settings.get("purpose", "policy") != "policy":
        raise ValueError("A rollout must use the policy query purpose")
    manifest_path = directory / "job.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for key in ("job_id", "task", "seed", "episode_seed", "actor_rng_seed"):
            if key in manifest and manifest[key] != job[key]:
                raise ValueError(f"Episode job manifest disagrees with frozen {key}")
    return job, settings


def _query_noise(arrays, mode, rng_seed, query_index):
    noise = arrays["noise"]
    if noise.shape != (16, 32) or noise.dtype != np.float32 or not np.isfinite(noise).all():
        raise ValueError("Saved decoder noise must be a finite float32 16 by 32 tensor")
    if mode == "native":
        return
    if noise.dtype != np.float32 or not np.array_equal(noise, np.repeat(noise[:1], 16, axis=0)):
        raise ValueError("Explicit rollout noise must be one float32 vector tiled over 16 steps")
    if mode != "gaussian" and np.max(np.abs(noise)) > 1.000001:
        raise ValueError("Bounded rollout noise exceeds magnitude one")
    if mode in {"zero", "random", "gaussian"}:
        rng = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence([rng_seed & 0xFFFFFFFF, query_index]))
        )
        if mode == "zero":
            expected = np.zeros(32, np.float32)
        elif mode == "random":
            expected = rng.uniform(-1, 1, 32).astype(np.float32)
        else:
            expected = rng.standard_normal(32).astype(np.float32)
        if not np.array_equal(noise[0], expected):
            raise ValueError("Saved exploration noise differs from its frozen RNG stream")


def _write_npz(path: Path, arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz.part", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
            if path.exists():
                previous = _load_npz(path)
                if set(previous) != set(arrays) or any(
                    not np.array_equal(previous[key], value) for key, value in arrays.items()
                ):
                    raise ValueError("Refusing to replace a different admitted learner episode")
                temporary.unlink()
            else:
                os.link(temporary, path)
                temporary.unlink()
        finally:
            temporary.unlink(missing_ok=True)


def build_episode(
    directory: str | Path,
    output_path: str | Path | None = None,
    *,
    recording_path: str | Path | None = None,
    row_path: str | Path | None = None,
    sidecar_path: str | Path | None = None,
    path_map: Mapping[str, str] | None = None,
    job_schedule: Mapping[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Validate one complete rollout and optionally admit its non-native replay.

    Returns ``validation`` (JSON-safe receipt), ``descriptor`` (learner input or
    None), and ``snapshots`` (query receipts plus permitted pixels/state arrays).
    The latter can supply both authority probes and engineering learner inputs.
    A caller reusing this validator supplies its verified frozen job schedule;
    omission retains the original task-6 schedule for historical analysis.
    """
    directory = Path(directory).resolve()
    recording_path = (
        _resolve(recording_path, directory, path_map) if recording_path else _recording(directory)
    )
    row_path = (
        _resolve(row_path, directory, path_map)
        if row_path
        else _one(directory.rglob("results.jsonl"), "native result file")
    )
    sidecar_path = (
        _resolve(sidecar_path, directory, path_map)
        if sidecar_path
        else _one(directory.rglob("*_ltl.json"), "native monitor sidecar")
    )
    legacy = validate_recording(recording_path, row_path, sidecar_path)
    data, meta = load_recording(recording_path)
    if meta.get("dsrl06_schema") != "dsrl06-recording-v1":
        raise ValueError("A complete DSRL06 recorder receipt is required")
    journal_path = recording_path.with_suffix(".journal.jsonl")
    journal = [json.loads(line) for line in journal_path.read_text().splitlines()]
    row, count = legacy["row"], legacy["queries"]
    job, settings = _frozen_job(meta, row, directory, job_schedule)
    labels, histories, potentials, scorer_audit = _scorer_history(data, meta, row, journal)
    scorer_audit["input_sha256"] = {
        "recording": _sha256(recording_path),
        "journal": _sha256(journal_path),
    }
    warmup_attempts, warmup_returned = meta["warmup_attempts"], meta["warmup_returned"]
    if warmup_attempts != 10 or warmup_returned != 10:
        raise ValueError("A valid rollout must retain all ten counted native warmup controls")
    if len([e for e in journal if e["event"] == "warmup_attempt"]) != warmup_attempts:
        raise ValueError("Warmup journal differs from attempted-control metadata")
    if len([e for e in journal if e["event"] == "warmup_returned"]) != warmup_returned:
        raise ValueError("Warmup journal differs from returned-control metadata")
    references = meta["dsrl_queries"]
    returned = [e for e in journal if e["event"] == "query_returned"]
    attempted = [e for e in journal if e["event"] == "query_attempt"]
    if len(references) != count or len(returned) != count or len(attempted) != count:
        raise ValueError("Every proposal must retain exactly one query attempt and response")
    images, states, noises, versions, snapshots, source_hashes = [], [], [], [], [], []
    previous_rng = None
    ids = set()
    prompt = row["prompt"]
    for index, reference in enumerate(references):
        step = index * 8
        path = _resolve(reference["record_path"], directory, path_map)
        arrays = _load_npz(path)
        query = json.loads(str(arrays["meta"]))
        mode = settings["mode"]
        expected = {
            "job_id": job["job_id"],
            "query_index": index,
            "step": step,
            "mode": mode,
            "purpose": "policy",
            "rng_seed": job["actor_rng_seed"],
            "episode_seed": job["episode_seed"],
            "actor_version": settings.get("actor_version", -1),
        }
        if query.get("status") != "returned" or query.get("actor_transform") != ACTOR_TRANSFORM:
            raise ValueError("Query capture is incomplete or uses an unfrozen actor transform")
        if any(query.get(key) != value for key, value in expected.items()):
            raise ValueError("Query metadata differs from its native rollout identity")
        for key in ("job_id", "query_index", "step", "mode", "actor_version", "query_id"):
            if reference.get(key) != query[key]:
                raise ValueError("Query response receipt disagrees with its server artifact")
        if query["query_id"] in ids:
            raise ValueError("One model query cannot supply two rollout proposals")
        ids.add(query["query_id"])
        if (
            any(
                returned[index].get(k) != reference[k]
                for k in ("step", "query_index", "query_id", "record_path", "mode", "actor_version")
            )
            or attempted[index].get("step") != step
            or attempted[index].get("query_index") != index
        ):
            raise ValueError("Query journal is not aligned with its response metadata")
        native = mode == "native"
        origin = (
            "same-key Gaussian reconstruction after native inference"
            if native
            else "explicit noise supplied before inference"
        )
        if query.get("noise_origin") != origin or reference.get("noise_origin") != origin:
            raise ValueError("Query does not identify native versus explicit noise honestly")
        for name in ("rng_before", "rng_after", "sample_rng"):
            if arrays[name].shape != (2,) or arrays[name].dtype != np.uint32:
                raise ValueError("Query must preserve both uint32 words of each policy RNG key")
        if previous_rng is not None and not np.array_equal(arrays["rng_before"], previous_rng):
            raise ValueError("Native policy RNG continuity is broken between queries")
        previous_rng = arrays["rng_after"]
        actions = arrays["actions"]
        if (
            actions.shape != (16, 8)
            or actions.dtype != np.float32
            or not np.isfinite(actions).all()
            or not np.array_equal(actions, arrays["actions_native"].astype(np.float32))
            or not np.array_equal(actions, data["proposal_chunk"][index])
        ):
            raise ValueError("Saved model output differs from the executed proposal")
        pixels, state = _observation(arrays, data["policy_state"][step], prompt)
        _query_noise(arrays, mode, job["actor_rng_seed"], index)
        images.append(pixels)
        states.append(state)
        noises.append(arrays["noise"][0])
        versions.append(query["actor_version"])
        digest = _sha256(path)
        source_hashes.append({"path": str(path), "sha256": digest, "query_id": query["query_id"]})
        history = histories[step]
        snapshots.append(
            {
                **reference,
                "record_path": str(path),
                "sha256": digest,
                "task": job["task"],
                "closed_once": history["first_closed_step"] is not None,
                "raw_violation": history["raw_violation"],
                "order_failed": (
                    history["first_departure_step"] is not None
                    and (
                        history["first_closed_step"] is None
                        or history["first_departure_step"] < history["first_closed_step"]
                    )
                ),
                "pixels": pixels,
                "state": state,
            }
        )
    if len(set(versions)) != 1 or (settings["mode"] == "actor" and versions[0] < 0):
        raise ValueError("Every rollout must use one published actor version")
    final_path = _resolve(meta["final_observation_path"], directory, path_map)
    final = _load_npz(final_path)
    if final["step"].shape != () or int(final["step"]) != row["steps"]:
        raise ValueError("Final observation is not at the terminal control boundary")
    finals = [e for e in journal if e["event"] == "final_observation"]
    if (
        len(finals) != 1
        or finals[0]["step"] != row["steps"]
        or finals[0]["path"] != meta["final_observation_path"]
    ):
        raise ValueError("Final observation journal disagrees with its metadata")
    pixels, state = _observation(final, data["policy_state"][-1], prompt)
    images.append(pixels)
    states.append(state)
    transitions = []
    for index in range(count):
        start, end = index * 8, min((index + 1) * 8, row["steps"])
        transitions.append(
            reward_transition(
                potentials[start],
                potentials[end],
                end - start,
                terminal=index == count - 1,
                strict_safe_requested=bool(index == count - 1 and labels["strict_safe_requested"]),
            )
        )
    identity = check_return_identity(transitions)
    eligible = settings["mode"] != "native"
    descriptor = None
    if output_path is not None and eligible:
        output = Path(output_path).resolve()
        arrays = {
            "pixels": np.stack(images).astype(np.uint8),
            "states": np.stack(states).astype(np.float32),
            "noise": np.stack(noises).astype(np.float32),
            "rewards": np.asarray([t["reward"] for t in transitions], np.float32),
            "discounts": np.asarray([t["discount"] for t in transitions], np.float32),
            "masks": np.asarray([t["mask"] for t in transitions], np.float32),
            "durations": np.asarray([t["duration"] for t in transitions], np.int32),
            "query_steps": np.arange(count, dtype=np.int32) * 8,
            "actor_versions": np.asarray(versions, np.int32),
        }
        _write_npz(output, arrays)
        descriptor = {
            "job_id": job["job_id"],
            "order_index": int(job["job_id"][1:]) if job["stage"] == "training" else -1,
            "path": str(output),
            "sha256": _sha256(output),
            "actor_version": versions[0],
        }
    validation = {
        "schema": "dsrl06-validation-v1",
        "valid": True,
        "job_id": job["job_id"],
        "steps": row["steps"],
        "queries": count,
        "warmup_controls": warmup_attempts,
        "mode": settings["mode"],
        "actor_version": versions[0],
        "learner_eligible": eligible,
        "labels": labels,
        "scorer_audit": scorer_audit,
        "return_identity": identity,
        "transitions": transitions,
        "source_queries": source_hashes,
        "recording": {"path": str(recording_path), "sha256": _sha256(recording_path)},
        "final_observation": {"path": str(final_path), "sha256": _sha256(final_path)},
        "source_files": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in (
                ("recording", recording_path),
                ("result", row_path),
                ("monitor", sidecar_path),
                ("journal", journal_path),
                ("final_observation", final_path),
            )
        },
        "episode_seed": job["episode_seed"],
        "simulator_reset_seed": None,
    }
    return {"validation": validation, "descriptor": descriptor, "snapshots": snapshots}
