"""CPU fixtures for residual provenance, R1 geometry and dual-reward learner admission."""

import hashlib
import json

import numpy as np
import pytest

from helpers.dsrl06_protocol import GAMMA
from helpers.dsrl06_server import ACTOR_TRANSFORM, actor_observation
from helpers.dsrl06_validation import GRASP_VALIDITY
from helpers.dsrl11_protocol import all_jobs, check_return_identity, inject_residual, residual_for
from helpers.dsrl11_server import NOISE_ORIGIN
from helpers.dsrl11_validation import LEARNER_KEYS, build_residual_episode, native_intersects
from helpers.tracking import SCORING, STATE_FIELDS

HALF_EXTENT = np.array([0.04, 0.04, 0.06])
JAR_START = np.array([0.5, 0.0, 0.10], np.float32)


def rewrite(path, transform):
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    transform(arrays)
    np.savez_compressed(path, **arrays)


def rewrite_journal(path, transform):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    transform(rows)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def intersects(low, high, center, radius):
    """Independent vector form of the native sphere-AABB test."""
    return float(np.sum((center - np.clip(center, low, high)) ** 2)) <= radius * radius


def fixture(directory, job_id="E4", *, controls=10, success=True, raw_step=None, contact_step=1):
    directory.mkdir()
    job = all_jobs()[job_id]
    count = (controls + 7) // 8
    scene = f"task_{job['task']:04d}/base"
    prompt = "Close the lid, then carry the jar."
    row = {key: None for key in SCORING}
    row.update(
        scene_name=scene,
        seed=job["seed"],
        episode_seed=job["episode_seed"],
        prompt=prompt,
        status="completed",
        steps=controls,
        success=success,
        success_step=controls if success else None,
        success_first_step=1 if success else None,
        ltl_violated=raw_step is not None,
        ltl_violation_step=raw_step,
        ltl_violation_count=int(raw_step is not None),
        first_contact_step=contact_step,
        ever_grasped=True,
        grasp_steps=controls,
        outcome="success" if success else "timeout",
        counted_violation=raw_step is not None
        and contact_step is not None
        and raw_step >= contact_step,
        safety_evaluated=contact_step is not None,
        ltl_monitored=True,
        nan_terminated=False,
    )
    result_path = directory / "results.jsonl"
    result_path.write_text(json.dumps(row) + "\n")
    size = controls + 1
    steps = np.arange(size)
    # The jar rises 2 cm per control after a grasp at step 1; the gripper approaches it.
    jar = np.repeat(JAR_START[None], size, axis=0)
    jar[:, 2] += (0.02 * steps).astype(np.float32)
    eef = (jar + np.stack([np.zeros(size), np.zeros(size), 0.3 / (steps + 1)], 1)).astype(
        np.float32
    )
    low = jar.astype(np.float64) - HALF_EXTENT
    high = jar.astype(np.float64) + HALF_EXTENT
    # The sphere touches the AABB from step 1 when the episode succeeds, never otherwise.
    center = np.array([0.5 if success else 1.5, 0.0, 0.21])
    radius = 0.04
    inside = np.array([intersects(low[k], high[k], center, radius) for k in range(size)])
    data = {key: np.zeros((size, 1)) for key in STATE_FIELDS}
    data.update(
        step=steps,
        q=np.zeros((size, 9)),
        dq=np.zeros((size, 9)),
        eef_pos=eef,
        jar_pos=jar,
        jar_aabb_min=low,
        jar_aabb_max=high,
        finger_pos=np.repeat(eef[:, None], 2, axis=1),
        policy_state=np.repeat(steps[:, None] / 10, 8, axis=1),
        grasping=np.r_[-1, np.ones(controls)].astype(int),
        ap=np.column_stack((steps == 0, steps == 0)),
        ap_valid=np.ones(size, bool),
    )
    held = steps > 0
    raw = steps >= raw_step if raw_step is not None else np.zeros(size, bool)
    data.update(
        raw_violation=raw,
        raw_violation_count=raw.astype(int),
        raw_violation_step=np.where(raw, raw_step, -1).astype(int)
        if raw_step is not None
        else np.full(size, -1),
        counted_violation=raw & bool(row["counted_violation"]),
        goal_instantaneous=held & inside,
        goal_checked=np.ones(size, bool),
        goal_held=held,
    )
    actions = np.repeat(np.arange(count)[:, None, None] / 100, 16 * 8, axis=1).reshape(count, 16, 8)
    actions = actions.astype(np.float32)
    data.update(
        proposal_chunk=actions,
        proposal_step=np.arange(count) * 8,
        exec_step=np.arange(controls),
        exec_proposal=np.arange(controls) // 8,
        exec_offset=np.arange(controls) % 8,
        exec_raw=actions[np.arange(controls) // 8, np.arange(controls) % 8],
    )
    data["exec_ctrl"] = data["exec_raw"].copy()
    data["exec_clipped"] = data["exec_raw"].copy()
    settings = {
        "mode": "residual",
        "job_id": job_id,
        "attempt_id": "1",
        "purpose": "policy",
        "residual_source": job["residual_source"],
        "epsilon_seed": job["epsilon_seed"],
    }
    if job["residual_source"] == "uniform":
        settings["uniform_seed"] = job["uniform_seed"]
    if job["residual_source"] == "actor":
        settings.update(
            actor_url="http://127.0.0.1:9011",
            actor_condition=job["condition"],
            actor_version=job["actor_version"],
            rng_seed=job["actor_rng_seed"],
        )
    region = {
        "mode": "held_intersection",
        "target_name": "jar",
        "marker_name": "goal_marker",
        "center_world": center.tolist(),
        "radius_m": radius,
    }
    meta = {
        "action_frequency": 20,
        "arm_control_idx": list(range(7)),
        "execute_horizon": 8,
        "config": {
            "state_mode": "joint",
            "controller_preset": "joint_position_raw",
            "ik_eef_to_joint": False,
            "camera_resolution": 256,
            "success_hold_steps": 10,
        },
        "result": row,
        "ap_names": ["jar_closed", "jar_on_support"],
        "recorder_overhead_s": 0.0,
        "dsrl06_schema": "dsrl06-recording-v1",
        "dsrl11_schema": "dsrl11-recording-v1",
        "dsrl_recording_valid": True,
        "dsrl_grasp_validity": GRASP_VALIDITY,
        "dsrl_settings": settings,
        "goal_region": region,
        "warmup_attempts": 10,
        "warmup_returned": 10,
        "dsrl_queries": [],
    }
    image = np.indices((256, 256)).sum(axis=0).astype(np.uint8)
    image = np.repeat(image[:, :, None], 3, axis=2)

    def observation(step):
        state = data["policy_state"][step]
        pixels, actor_state = actor_observation(
            {"observation/image_left": image, "observation/state": state}
        )
        return {
            "observation_image_left": image,
            "observation_wrist_image": image[::-1],
            "observation_state": state,
            "prompt": np.asarray(prompt),
            "pixels": pixels,
            "state": actor_state,
        }

    query_paths = []
    version = settings.get("actor_version", -1)
    for index in range(count):
        path = directory / f"query_{index + 1:08d}.npz"
        query = {
            "job_id": job_id,
            "query_index": index,
            "mode": "residual",
            "purpose": "policy",
            "actor_version": version,
            "packet_id": f"{job_id}:attempt1:query{index}",
            "attempt_id": "1",
            "step": index * 8,
            "residual_source": job["residual_source"],
            "residual_scale": 0.25,
            "epsilon_seed": job["epsilon_seed"],
            "benchmark_episode_seed": job["episode_seed"],
            "query_id": index + 1,
            "status": "returned",
            "actor_transform": ACTOR_TRANSFORM,
            "noise_origin": NOISE_ORIGIN,
            "rng_origin": "seeded from the frozen epsilon_seed at query zero"
            if index == 0
            else "continued from the previous query of this job",
        }
        if job["residual_source"] == "uniform":
            query["uniform_seed"] = job["uniform_seed"]
        if job["residual_source"] == "actor":
            query.update(actor_condition=job["condition"], rng_seed=job["actor_rng_seed"])
        residual = residual_for(job, index)
        if residual is None:
            residual = np.linspace(-0.9, 0.9, 32).astype(np.float32)
        epsilon = (
            np.random.default_rng([job["epsilon_seed"], index])
            .standard_normal((16, 32))
            .astype(np.float32)
        )
        arrays = observation(index * 8)
        arrays.update(
            epsilon=epsilon,
            residual_u=residual,
            noise=inject_residual(epsilon, residual),
            actions=actions[index],
            actions_native=actions[index].astype(float),
            rng_before=np.array([0, index], np.uint32),
            rng_after=np.array([0, index + 1], np.uint32),
            sample_rng=np.array([1, index], np.uint32),
            meta=np.asarray(json.dumps(query)),
        )
        np.savez_compressed(path, **arrays)
        query_paths.append(path)
        meta["dsrl_queries"].append({**query, "record_path": str(path)})
    final_path = directory / "scene.final_observation.npz"
    np.savez_compressed(final_path, **observation(controls), step=np.asarray(controls))
    meta["final_observation_path"] = str(final_path)
    recording_path = directory / "scene.npz"
    np.savez_compressed(recording_path, **data, meta=np.asarray(json.dumps(meta)))
    sidecar = {
        "log": [
            {
                "step": step,
                "ap": {name: bool(data["ap"][step, k]) for k, name in enumerate(meta["ap_names"])},
            }
            for step in range(size)
        ]
    }
    (directory / "scene_ltl.json").write_text(json.dumps(sidecar))
    journal = []
    for step in range(10):
        journal.extend(
            [
                {"event": "warmup_attempt", "index": step},
                {"event": "warmup_returned", "count": step + 1},
            ]
        )
    journal.append({"event": "goal_region", **region})
    for step in range(size):
        detail = {
            "mode": "held_intersection",
            "target_name": "jar",
            "marker_name": "goal_marker",
            "held": bool(held[step]),
            "held_via": "target" if held[step] else None,
            "intersects": bool(inside[step]),
            "radius_m": radius,
            "center_world": center.tolist(),
        }
        journal.append(
            {
                "event": "scorer_state",
                "step": step,
                "valid": True,
                "jar_closed": bool(data["ap"][step, 0]),
                "jar_on_support": bool(data["ap"][step, 1]),
                "grasped": bool(data["grasping"][step] == 1),
                "instantaneous_goal": bool(data["goal_instantaneous"][step]),
                "raw_violation": bool(raw[step]),
                "counted_violation": bool(data["counted_violation"][step]),
                "goal_detail": detail,
            }
        )
        if step < controls:
            if step % 8 == 0:
                reference = meta["dsrl_queries"][step // 8]
                journal.extend(
                    [
                        {"event": "query_attempt", "step": step, "query_index": step // 8},
                        {"event": "query_returned", **reference},
                        {"event": "proposal", "step": step},
                    ]
                )
            journal.append({"event": "applied", "step": step})
    journal.extend(
        [
            {"event": "final_observation", "step": controls, "path": str(final_path)},
            {"event": "finish", "result": row},
        ]
    )
    journal_path = recording_path.with_suffix(".journal.jsonl")
    journal_path.write_text("".join(json.dumps(entry) + "\n" for entry in journal))
    return {
        "recording": recording_path,
        "queries": query_paths,
        "final": final_path,
        "journal": journal_path,
        "data": data,
        "meta": meta,
        "inside": inside,
    }


def test_uniform_engineering_episode_admits_both_rewards(tmp_path):
    saved = fixture(tmp_path / "episode")
    output = tmp_path / "admitted.npz"
    result = build_residual_episode(tmp_path / "episode", output, job_schedule=None)
    validation, descriptor = result["validation"], result["descriptor"]
    assert validation["valid"] and validation["labels"]["strict_safe_requested"]
    assert validation["geometry"]["intersection_states_reproduced"] == 11
    assert validation["geometry"]["states_intersecting_goal"] == int(saved["inside"].sum()) == 10
    for name in ("r0", "r1"):
        identity = validation["returns"][name]
        assert identity["discounted_base_return"] == pytest.approx(GAMMA**8)
        assert abs(identity["return_identity_error"]) < 1e-9
    r0, r1 = validation["transitions"]["r0"], validation["transitions"]["r1"]
    assert [t["duration"] for t in r1] == [8, 2] and [t["mask"] for t in r1] == [1.0, 0.0]
    assert r1[0]["phi_before"] != r0[0]["phi_before"]
    assert descriptor["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert descriptor["transitions"] == 2 and descriptor["actor_version"] == -1
    with np.load(output, allow_pickle=False) as replay:
        assert set(replay.files) == set(LEARNER_KEYS)
        assert replay["pixels"].shape == (3, 64, 64, 3) and replay["states"].shape == (3, 8)
        np.testing.assert_array_equal(replay["residual"][0], residual_for(all_jobs()["E4"], 0))
        np.testing.assert_allclose(replay["rewards_r1"], [t["reward"] for t in r1], rtol=1e-6)
        np.testing.assert_array_equal(replay["durations"], [8, 2])
        np.testing.assert_array_equal(replay["actor_versions"], [-1, -1])
    # Deterministic: a rebuild produces the identical learner archive.
    assert build_residual_episode(tmp_path / "episode", output, job_schedule=None)[
        "descriptor"
    ] == (descriptor)


def test_r1_potential_trace_matches_the_frozen_formula(tmp_path):
    saved = fixture(tmp_path / "episode", controls=10)
    result = build_residual_episode(tmp_path / "episode", job_schedule=None)
    rows = result["validation"]["transitions"]["r1"]
    jar, eef = saved["data"]["jar_pos"].astype(float), saved["data"]["eef_pos"].astype(float)
    reach0 = 1 - np.tanh(np.linalg.norm(eef[0] - jar[0]) / 0.10)
    # Step 0: closed (latched), not yet grasped.
    assert rows[0]["phi_before"] == pytest.approx(0.10 * reach0 + 0.20)
    # Step 8: grasped, lifted 16 cm (clipped to one), intersecting the goal sphere.
    reach8 = 1 - np.tanh(np.linalg.norm(eef[8] - jar[8]) / 0.10)
    low, high = saved["data"]["jar_aabb_min"][8], saved["data"]["jar_aabb_max"][8]
    center = np.array(saved["meta"]["goal_region"]["center_world"])
    distance = np.linalg.norm(center - np.clip(center, low, high))
    goal = 1 - np.tanh(distance / 0.20)
    expected = 0.10 * reach8 + 0.20 + 0.25 + 0.20 + 0.25 * goal
    assert rows[0]["phi_after"] == pytest.approx(expected)
    assert rows[-1]["phi_after"] == 0.0 and rows[-1]["phi_after_raw"] > 0


def test_raw_violation_zeroes_r1_and_removes_completion(tmp_path):
    fixture(tmp_path / "episode", raw_step=2, contact_step=1)
    validation = build_residual_episode(tmp_path / "episode", job_schedule=None)["validation"]
    rows = validation["transitions"]["r1"]
    labels = validation["labels"]
    assert labels["raw_violation"] and not labels["strict_safe_requested"]
    assert rows[0]["phi_before"] > 0 and rows[0]["phi_after_raw"] == 0.0
    assert rows[1]["phi_before"] == 0.0 and all(t["base_reward"] == 0 for t in rows)
    check_return_identity(rows)


def test_full_length_actor_timeout_keeps_one_version_and_both_identities(tmp_path):
    fixture(tmp_path / "episode", "T_r1_prior_4", controls=2000, success=False)
    output = tmp_path / "r.npz"
    result = build_residual_episode(tmp_path / "episode", output, job_schedule=None)
    validation = result["validation"]
    assert validation["queries"] == 250 and result["descriptor"]["actor_version"] == 0
    assert not validation["labels"]["strict_safe_requested"]
    for name in ("r0", "r1"):
        assert validation["returns"][name]["discounted_base_return"] == 0.0
    with np.load(output, allow_pickle=False) as replay:
        np.testing.assert_array_equal(replay["actor_versions"], np.zeros(250))
        assert replay["masks"][-1] == 0 and replay["masks"][:-1].all()
        assert replay["pixels"].shape == (251, 64, 64, 3)


@pytest.mark.parametrize("job_id", ["B00", "T_r1_prior_4", "V3_native", "V3_r0_none"])
def test_zero_and_actor_sources_reconstruct_or_match_their_job(tmp_path, job_id):
    fixture(tmp_path / "episode", job_id)
    result = build_residual_episode(tmp_path / "episode", tmp_path / "r.npz", job_schedule=None)
    job = all_jobs()[job_id]
    validation = result["validation"]
    assert validation["residual"]["source"] == job["residual_source"]
    assert validation["residual"]["all_zero"] == (job["residual_source"] == "zero")
    assert result["descriptor"]["actor_version"] == job.get("actor_version", -1)
    assert validation["arm"] == job.get("arm") and validation["condition"] == job.get("condition")


def test_native_intersection_matches_the_pinned_arithmetic_at_the_boundary():
    low, high = np.array([0.0, 0.0, 0.0]), np.array([0.1, 0.1, 0.1])
    assert native_intersects(low, high, [0.15, 0.05, 0.05], 0.05)
    assert not native_intersects(low, high, [0.15 + 1e-9, 0.05, 0.05], 0.05)
    assert native_intersects(low, high, [0.05, 0.05, 0.05], 0.0)


def _flip_intersection(rows):
    state = next(r for r in rows if r["event"] == "scorer_state" and r["step"] == 4)
    state["goal_detail"]["intersects"] = not state["goal_detail"]["intersects"]


def _move_goal(rows):
    state = next(r for r in rows if r["event"] == "scorer_state" and r["step"] == 2)
    state["goal_detail"]["center_world"][0] += 0.01


@pytest.mark.parametrize(
    "mutation, match",
    [
        (_flip_intersection, "reproduce the native goal intersection"),
        (_move_goal, "disagree with the recorded goal region"),
    ],
)
def test_goal_geometry_inconsistencies_are_rejected(tmp_path, mutation, match):
    saved = fixture(tmp_path / "episode")
    rewrite_journal(saved["journal"], mutation)
    with pytest.raises(ValueError, match=match):
        build_residual_episode(tmp_path / "episode", job_schedule=None)


def _recording_change(change):
    def transform(arrays):
        meta = json.loads(str(arrays["meta"]))
        change(arrays, meta)
        arrays["meta"] = np.asarray(json.dumps(meta))

    return transform


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda a, m: m.pop("dsrl11_schema"), "DSRL11 recorder receipt"),
        (lambda a, m: m["goal_region"].update(mode="predicate"), "held-intersection"),
        (lambda a, m: m["dsrl_settings"].update(epsilon_seed=5), "routing differs"),
        (lambda a, m: m["dsrl_settings"].update(job_id="X9"), "frozen task-11 schedule"),
        (lambda a, m: a.__setitem__("jar_aabb_min", a["jar_aabb_min"][:-1]), "N\\+1"),
        (lambda a, m: a["jar_aabb_max"].__setitem__((3, 0), 0.0), "inverted"),
    ],
)
def test_recording_identity_and_geometry_fields_are_required(tmp_path, change, match):
    saved = fixture(tmp_path / "episode")
    rewrite(saved["recording"], _recording_change(change))
    with pytest.raises(ValueError, match=match):
        build_residual_episode(tmp_path / "episode", job_schedule=None)


def _query_meta(change):
    def transform(arrays):
        meta = json.loads(str(arrays["meta"]))
        change(meta)
        arrays["meta"] = np.asarray(json.dumps(meta))

    return transform


@pytest.mark.parametrize(
    "transform, match",
    [
        (lambda a: a["noise"].__setitem__((0, 0), a["noise"][0, 0] + 1e-3), "0.25 tiled"),
        (lambda a: a["residual_u"].__setitem__(0, 0.5), "frozen reconstructible stream"),
        (lambda a: a.__setitem__("epsilon", a["epsilon"].astype(np.float64)), "float32"),
        (lambda a: a["rng_before"].__setitem__(1, 7), "continuity"),
        (_query_meta(lambda m: m.update(noise_origin="other")), "residual rollout identity"),
        (_query_meta(lambda m: m.update(benchmark_episode_seed=1)), "residual rollout identity"),
        (_query_meta(lambda m: m.update(rng_origin="seeded")), "native-stream origin"),
        (_query_meta(lambda m: m.update(status="failed")), "incomplete"),
    ],
)
def test_corrupt_residual_queries_are_never_admitted(tmp_path, transform, match):
    saved = fixture(tmp_path / "episode")
    rewrite(saved["queries"][1], transform)
    with pytest.raises(ValueError, match=match):
        build_residual_episode(tmp_path / "episode", tmp_path / "r.npz", job_schedule=None)
    assert not (tmp_path / "r.npz").exists()


def test_a_zero_job_cannot_carry_a_nonzero_residual(tmp_path):
    saved = fixture(tmp_path / "episode", "B00")

    def perturb(arrays):
        arrays["residual_u"] = np.full(32, 0.1, np.float32)
        arrays["noise"] = inject_residual(arrays["epsilon"], arrays["residual_u"])

    rewrite(saved["queries"][0], perturb)
    with pytest.raises(ValueError, match="frozen reconstructible stream"):
        build_residual_episode(tmp_path / "episode", job_schedule=None)


def test_actor_rollouts_must_use_their_frozen_version_and_condition(tmp_path):
    saved = fixture(tmp_path / "episode", "T_r1_prior_4")

    def change(arrays, meta):
        meta["dsrl_settings"]["actor_condition"] = "r0_prior"

    rewrite(saved["recording"], _recording_change(change))
    with pytest.raises(ValueError, match="routing differs"):
        build_residual_episode(tmp_path / "episode", job_schedule=None)
