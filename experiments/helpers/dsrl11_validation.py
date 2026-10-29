"""Offline admission of task-11 residual recordings with both reward definitions.

``build_residual_episode`` never runs a policy or simulator. It reuses the task-6
recording, scorer-history and observation checks unchanged, then verifies the
residual provenance of every query, computes R0 and R1 from the same complete
scorer trace and checks both return identities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from helpers.dsrl06_server import ACTOR_TRANSFORM
from helpers.dsrl06_validation import (
    _load_npz,
    _observation,
    _one,
    _recording,
    _resolve,
    _scorer_history,
    _sha256,
    _write_npz,
)
from helpers.dsrl11_protocol import (
    RESIDUAL_SCALE,
    all_jobs,
    check_return_identity,
    episode_rewards,
    inject_residual,
    r1_components,
    r1_potential,
    residual_for,
)
from helpers.dsrl11_server import NOISE_ORIGIN
from helpers.tracking import load_recording, validate_recording

LEARNER_KEYS = (
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
)


def _job(meta, row, directory, schedule):
    settings = meta["dsrl_settings"]
    jobs = all_jobs(schedule)
    job = jobs.get(settings.get("job_id"))
    if job is None:
        raise ValueError("Episode does not belong to the frozen task-11 schedule")
    if (
        row["scene_name"] != f"task_{job['task']:04d}/base"
        or row["seed"] != job["seed"]
        or row["episode_seed"] != job["episode_seed"]
    ):
        raise ValueError("Native scene and seeds differ from the frozen job")
    expected = {
        "mode": "residual",
        "purpose": "policy",
        "residual_source": job["residual_source"],
        "epsilon_seed": job["epsilon_seed"],
    }
    if job["residual_source"] == "uniform":
        expected["uniform_seed"] = job["uniform_seed"]
    if job["residual_source"] == "actor":
        expected.update(
            actor_condition=job["condition"],
            actor_version=job["actor_version"],
            rng_seed=job["actor_rng_seed"],
        )
    if any(settings.get(key) != value for key, value in expected.items()):
        raise ValueError("Episode routing differs from its frozen residual job")
    manifest = directory / "job.json"
    if manifest.exists():
        recorded = json.loads(manifest.read_text())
        for key in ("job_id", "task", "seed", "episode_seed", "epsilon_seed"):
            if recorded.get(key) != job[key]:
                raise ValueError(f"Episode job manifest disagrees with frozen {key}")
    return job, settings


def native_intersects(aabb_min, aabb_max, center, radius) -> bool:
    """The pinned object_intersects_goal_region arithmetic, operation for operation."""
    cx, cy, cz = (float(v) for v in center)
    low, high = [float(v) for v in aabb_min], [float(v) for v in aabb_max]
    dx = cx - min(max(cx, low[0]), high[0])
    dy = cy - min(max(cy, low[1]), high[1])
    dz = cz - min(max(cz, low[2]), high[2])
    return (dx * dx + dy * dy + dz * dz) <= float(radius * radius)


def _r1_history(data, meta, histories, steps, journal):
    region = meta.get("goal_region")
    if not region or region.get("mode") != "held_intersection":
        raise ValueError("R1 requires the native held-intersection goal region")
    center = np.asarray(region["center_world"], np.float64)
    radius = float(region["radius_m"])
    count = steps + 1
    for key in ("eef_pos", "jar_pos", "jar_aabb_min", "jar_aabb_max"):
        values = np.asarray(data[key])
        if values.shape != (count, 3) or not np.isfinite(values).all():
            raise ValueError(f"R1 geometry field {key} must hold N+1 finite 3-vectors")
    if np.any(data["jar_aabb_min"] > data["jar_aabb_max"]):
        raise ValueError("Recorded jar AABB is inverted")
    states = [entry for entry in journal if entry["event"] == "scorer_state"]
    for step, entry in enumerate(states):
        detail = entry["goal_detail"]
        if (
            detail.get("mode") != region["mode"]
            or detail.get("target_name") != region["target_name"]
            or [float(v) for v in detail.get("center_world", [])] != list(center)
            or float(detail.get("radius_m", -1)) != radius
        ):
            raise ValueError("Native goal reads disagree with the recorded goal region")
        recomputed = native_intersects(
            data["jar_aabb_min"][step], data["jar_aabb_max"][step], center, radius
        )
        if detail.get("intersects") is not recomputed:
            raise ValueError("Recorded jar AABB does not reproduce the native goal intersection")
    potentials, components = [], []
    for step in range(count):
        history = histories[step]
        parts = r1_components(
            eef_pos=data["eef_pos"][step],
            jar_pos=data["jar_pos"][step],
            initial_jar_pos=data["jar_pos"][0],
            jar_aabb_min=data["jar_aabb_min"][step],
            jar_aabb_max=data["jar_aabb_max"][step],
            goal_center=center,
        )
        closed = history["first_closed_step"] is not None
        departure = history["first_departure_step"]
        order_failed = departure is not None and (
            history["first_closed_step"] is None or departure < history["first_closed_step"]
        )
        potentials.append(
            r1_potential(
                parts,
                closed=closed,
                grasp=bool(data["grasping"][step] == 1),
                raw_violation=bool(history["raw_violation"]),
                order_failed=order_failed,
            )
        )
        components.append(parts)
    # The benchmark's eef2target metric, in its own float32 arithmetic, over controls.
    reach32 = np.linalg.norm(data["eef_pos"][1:] - data["jar_pos"][1:], axis=1)
    return potentials, {
        "goal_region": region,
        "intersection_states_reproduced": count,
        "minimum_reach_distance_m": float(np.min(reach32)) if steps else None,
        "minimum_goal_distance_m": float(min(c["goal_distance_m"] for c in components)),
        "states_intersecting_goal": int(
            sum(
                native_intersects(lo, hi, center, radius)
                for lo, hi in zip(data["jar_aabb_min"], data["jar_aabb_max"], strict=True)
            )
        ),
        "maximum_lift": float(max(c["lift"] for c in components)),
        "maximum_potential": float(max(potentials)),
    }


def _query_residual(arrays, job, index):
    for name in ("epsilon", "noise"):
        value = arrays[name]
        if value.shape != (16, 32) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"Saved {name} must be a finite float32 16 by 32 tensor")
    residual = arrays["residual_u"]
    if residual.shape != (32,) or residual.dtype != np.float32:
        raise ValueError("Saved residual must be a float32 32-vector")
    if not np.isfinite(residual).all() or np.max(np.abs(residual)) > 1:
        raise ValueError("Saved residual lies outside [-1, 1]^32")
    expected = residual_for(job, index)
    if expected is not None and not np.array_equal(residual, expected):
        raise ValueError("Saved residual differs from its frozen reconstructible stream")
    if not np.array_equal(arrays["noise"], inject_residual(arrays["epsilon"], residual)):
        raise ValueError("Decoder noise is not epsilon plus the 0.25 tiled residual")
    if job["residual_source"] == "zero" and not np.array_equal(arrays["noise"], arrays["epsilon"]):
        raise ValueError("A zero-residual query must receive the native draw unchanged")
    return residual


def build_residual_episode(
    directory: str | Path,
    output_path: str | Path | None = None,
    *,
    job_schedule: Mapping[str, list[dict[str, Any]]],
    path_map: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    directory = Path(directory).resolve()
    recording_path = _recording(directory)
    row_path = _one(directory.rglob("results.jsonl"), "native result file")
    sidecar_path = _one(directory.rglob("*_ltl.json"), "native monitor sidecar")
    legacy = validate_recording(recording_path, row_path, sidecar_path)
    data, meta = load_recording(recording_path)
    if meta.get("dsrl11_schema") != "dsrl11-recording-v1":
        raise ValueError("A complete DSRL11 recorder receipt is required")
    journal_path = recording_path.with_suffix(".journal.jsonl")
    journal = [json.loads(line) for line in journal_path.read_text().splitlines()]
    row, count = legacy["row"], legacy["queries"]
    job, settings = _job(meta, row, directory, job_schedule)
    labels, histories, potentials_r0, scorer_audit = _scorer_history(data, meta, row, journal)
    scorer_audit["input_sha256"] = {
        "recording": _sha256(recording_path),
        "journal": _sha256(journal_path),
    }
    if meta["warmup_attempts"] != 10 or meta["warmup_returned"] != 10:
        raise ValueError("A valid rollout must retain all ten counted native warmup controls")
    for event, expected in (("warmup_attempt", 10), ("warmup_returned", 10)):
        if len([e for e in journal if e["event"] == event]) != expected:
            raise ValueError("Warmup journal differs from attempted-control metadata")
    references = meta["dsrl_queries"]
    returned = [e for e in journal if e["event"] == "query_returned"]
    attempted = [e for e in journal if e["event"] == "query_attempt"]
    if len(references) != count or len(returned) != count or len(attempted) != count:
        raise ValueError("Every proposal must retain exactly one query attempt and response")
    images, states, residuals, versions, snapshots, sources = [], [], [], [], [], []
    previous_rng, ids = None, set()
    for index, reference in enumerate(references):
        step = index * 8
        path = _resolve(reference["record_path"], directory, path_map)
        arrays = _load_npz(path)
        query = json.loads(str(arrays["meta"]))
        expected = {
            "job_id": job["job_id"],
            "query_index": index,
            "step": step,
            "mode": "residual",
            "purpose": "policy",
            "residual_source": job["residual_source"],
            "residual_scale": RESIDUAL_SCALE,
            "epsilon_seed": job["epsilon_seed"],
            "actor_version": settings.get("actor_version", -1),
            "benchmark_episode_seed": job["episode_seed"],
            "noise_origin": NOISE_ORIGIN,
        }
        if job["residual_source"] == "actor":
            expected.update(actor_condition=job["condition"], rng_seed=job["actor_rng_seed"])
        if query.get("status") != "returned" or query.get("actor_transform") != ACTOR_TRANSFORM:
            raise ValueError("Query capture is incomplete or uses an unfrozen actor transform")
        if any(query.get(key) != value for key, value in expected.items()):
            raise ValueError("Query metadata differs from its residual rollout identity")
        expected_origin = (
            "seeded from the frozen epsilon_seed at query zero"
            if index == 0
            else "continued from the previous query of this job"
        )
        if query.get("rng_origin") != expected_origin:
            raise ValueError("Query does not record the frozen native-stream origin")
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
        pixels, state = _observation(arrays, data["policy_state"][step], row["prompt"])
        residuals.append(_query_residual(arrays, job, index))
        images.append(pixels)
        states.append(state)
        versions.append(query["actor_version"])
        digest = _sha256(path)
        sources.append({"path": str(path), "sha256": digest, "query_id": query["query_id"]})
        snapshots.append(
            {
                **reference,
                "record_path": str(path),
                "sha256": digest,
                "task": job["task"],
                "rng_before": arrays["rng_before"].tolist(),
                "sample_rng": arrays["sample_rng"].tolist(),
                "pixels": pixels,
                "state": state,
            }
        )
    if len(set(versions)) != 1 or versions[0] != job.get("actor_version", -1):
        raise ValueError("Every rollout must use its one frozen actor version")
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
    pixels, state = _observation(final, data["policy_state"][-1], row["prompt"])
    images.append(pixels)
    states.append(state)
    potentials_r1, geometry = _r1_history(data, meta, histories, row["steps"], journal)
    strict = bool(labels["strict_safe_requested"])
    transitions = {
        "r0": episode_rewards(potentials_r0, row["steps"], strict),
        "r1": episode_rewards(potentials_r1, row["steps"], strict),
    }
    if any(len(rows) != count for rows in transitions.values()):
        raise ValueError("Each policy query must own exactly one latent transition")
    identities = {name: check_return_identity(rows) for name, rows in transitions.items()}
    for name in ("discount", "mask", "duration"):
        if [t[name] for t in transitions["r0"]] != [t[name] for t in transitions["r1"]]:
            raise ValueError("Both rewards must share durations, discounts and masks")
    descriptor = None
    if output_path is not None:
        output = Path(output_path).resolve()
        arrays = {
            "pixels": np.stack(images).astype(np.uint8),
            "states": np.stack(states).astype(np.float32),
            "residual": np.stack(residuals).astype(np.float32),
            "rewards_r0": np.asarray([t["reward"] for t in transitions["r0"]], np.float32),
            "rewards_r1": np.asarray([t["reward"] for t in transitions["r1"]], np.float32),
            "discounts": np.asarray([t["discount"] for t in transitions["r0"]], np.float32),
            "masks": np.asarray([t["mask"] for t in transitions["r0"]], np.float32),
            "durations": np.asarray([t["duration"] for t in transitions["r0"]], np.int32),
            "query_steps": np.arange(count, dtype=np.int32) * 8,
            "actor_versions": np.asarray(versions, np.int32),
        }
        _write_npz(output, arrays)
        descriptor = {
            "job_id": job["job_id"],
            "stage": job["stage"],
            "condition": job.get("condition"),
            "path": str(output),
            "sha256": _sha256(output),
            "actor_version": versions[0],
            "transitions": count,
        }
    residual_array = np.stack(residuals)
    validation = {
        "schema": "dsrl11-validation-v1",
        "valid": True,
        "job_id": job["job_id"],
        "stage": job["stage"],
        "condition": job.get("condition"),
        "arm": job.get("arm"),
        "steps": row["steps"],
        "queries": count,
        "labels": labels,
        "native_result": {
            key: row.get(key)
            for key in (
                "success",
                "success_step",
                "ltl_violated",
                "ltl_violation_step",
                "counted_violation",
                "ever_contacted",
                "first_contact_step",
                "ever_grasped",
                "grasp_steps",
                "eef2target_min_dist",
            )
        },
        "returns": identities,
        "reward_counts": {
            name: {
                "nonzero": int(sum(t["reward"] != 0 for t in rows)),
                "positive": int(sum(t["reward"] > 0 for t in rows)),
                "negative": int(sum(t["reward"] < 0 for t in rows)),
            }
            for name, rows in transitions.items()
        },
        "geometry": geometry,
        "residual": {
            "source": job["residual_source"],
            "maximum_absolute": float(np.max(np.abs(residual_array))),
            "all_zero": bool(not np.any(residual_array)),
        },
        "scorer_audit": scorer_audit,
        "transitions": transitions,
        "query_sources": sources,
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
