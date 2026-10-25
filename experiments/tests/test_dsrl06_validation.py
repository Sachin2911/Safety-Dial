"""End-to-end CPU fixtures for scorer replay and lossless learner admission."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.dsrl06_protocol import GAMMA, frozen_jobs  # noqa: E402
from helpers.dsrl06_learner import LearnerConfig, load_episode  # noqa: E402
from helpers.dsrl06_server import ACTOR_TRANSFORM, actor_observation  # noqa: E402
from helpers.dsrl06_validation import GRASP_SOURCE_PROOF, GRASP_VALIDITY, build_episode  # noqa: E402
from helpers.tracking import SCORING, STATE_FIELDS  # noqa: E402


def rewrite(path, transform):
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    transform(arrays)
    np.savez_compressed(path, **arrays)


def fixture(
    directory, job_id="E5", *, mode=None, controls=10, success=True, raw_step=None, contact_step=1
):
    directory.mkdir()
    job = next(j for jobs in frozen_jobs().values() for j in jobs if j["job_id"] == job_id)
    mode = mode or job["mode"]
    version = 0 if mode == "actor" else -1
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
    data = {key: np.zeros((size, 1)) for key in STATE_FIELDS}
    data.update(
        step=np.arange(size),
        q=np.zeros((size, 9)),
        dq=np.zeros((size, 9)),
        policy_state=np.repeat(np.arange(size)[:, None] / 10, 8, axis=1),
        grasping=np.r_[-1, np.ones(controls)].astype(int),
        ap=np.column_stack((np.arange(size) == 0, np.arange(size) == 0)),
        ap_valid=np.ones(size, bool),
    )
    raw = np.arange(size) >= raw_step if raw_step is not None else np.zeros(size, bool)
    data.update(
        raw_violation=raw,
        raw_violation_count=raw.astype(int),
        raw_violation_step=np.where(raw, raw_step, -1).astype(int)
        if raw_step is not None
        else np.full(size, -1),
        counted_violation=raw & bool(row["counted_violation"]),
        goal_instantaneous=(np.arange(size) > 0) & success,
        goal_checked=np.ones(size, bool),
        goal_held=np.arange(size) > 0,
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
        "job_id": job_id,
        "mode": mode,
        "rng_seed": job["actor_rng_seed"],
        "actor_version": version,
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
        "dsrl_recording_valid": True,
        "dsrl_grasp_validity": GRASP_VALIDITY,
        "dsrl_settings": settings,
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
    for index in range(count):
        path = directory / f"query_{index + 1:08d}.npz"
        origin = (
            "same-key Gaussian reconstruction after native inference"
            if mode == "native"
            else "explicit noise supplied before inference"
        )
        query = {
            **settings,
            "purpose": "policy",
            "episode_seed": job["episode_seed"],
            "query_index": index,
            "query_id": index + 1,
            "step": index * 8,
            "status": "returned",
            "actor_transform": ACTOR_TRANSFORM,
            "noise_origin": origin,
        }
        rng = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence([job["actor_rng_seed"], index]))
        )
        if mode == "native":
            noise = np.arange(512, dtype=np.float32).reshape(16, 32) / 100
        else:
            vector = {
                "zero": lambda: np.zeros(32),
                "random": lambda: rng.uniform(-1, 1, 32),
                "gaussian": lambda: rng.standard_normal(32),
                "actor": lambda: np.full(32, 0.3),
            }[mode]().astype(np.float32)
            noise = np.repeat(vector[None], 16, axis=0)
        arrays = observation(index * 8)
        arrays.update(
            noise=noise,
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
    for step in range(size):
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
                "goal_detail": {"held": bool(data["goal_held"][step])},
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
        "result": result_path,
        "data": data,
        "meta": meta,
    }


def test_full_recording_replays_latched_closure_and_partial_terminal(tmp_path):
    directory = tmp_path / "episode"
    fixture(directory)
    output = tmp_path / "admitted.npz"
    result = build_episode(directory, output)
    validation, descriptor = result["validation"], result["descriptor"]
    assert validation["labels"]["strict_safe_requested"]
    assert validation["transitions"][0]["phi_before"] == 0.2
    assert result["snapshots"][1]["closed_once"]  # closure has flapped false by query two
    assert validation["return_identity"]["corrected_discounted_shaped_return"] == pytest.approx(
        GAMMA**8
    )
    assert descriptor["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    admitted = load_episode(descriptor, LearnerConfig(), warmup=False, data_root=tmp_path)
    assert admitted["noise"].shape == (2, 32)
    with np.load(output, allow_pickle=False) as replay:
        assert replay["pixels"].shape == (3, 64, 64, 3)
        assert replay["pixels"].dtype == np.uint8
        np.testing.assert_array_equal(replay["durations"], [8, 2])
        np.testing.assert_array_equal(replay["query_steps"], [0, 8])
        np.testing.assert_array_equal(replay["masks"], [1, 0])
        np.testing.assert_allclose(replay["discounts"], GAMMA ** np.array([8, 2]))
        np.testing.assert_allclose(replay["states"][-1], np.ones(8))
    assert build_episode(directory, output)["descriptor"] == descriptor


def test_native_retains_untiled_noise_without_creating_learner_npz(tmp_path):
    directory = tmp_path / "native"
    fixture(directory, "E0")
    output = tmp_path / "native_replay.npz"
    result = build_episode(directory, output)
    assert result["descriptor"] is None
    assert not output.exists()
    assert not result["validation"]["learner_eligible"]
    assert result["snapshots"][0]["state"].shape == (8,)


@pytest.mark.parametrize(
    "job_id,mode", [("E4", "random"), ("V00_fixed_zero", "zero"), ("T00", "gaussian")]
)
def test_frozen_explicit_exploration_modes_and_ordered_descriptor(tmp_path, job_id, mode):
    directory = tmp_path / "episode"
    fixture(directory, job_id, mode=mode)
    result = build_episode(directory, tmp_path / "replay.npz")
    assert result["validation"]["mode"] == mode
    assert result["descriptor"]["order_index"] == (0 if job_id == "T00" else -1)
    if mode == "gaussian":
        with np.load(result["descriptor"]["path"]) as replay:
            assert np.max(np.abs(replay["noise"])) > 1


def test_raw_violation_before_contact_stays_strict_failure_and_counted_safe(tmp_path):
    directory = tmp_path / "episode"
    fixture(directory, raw_step=2, contact_step=5)
    validation = build_episode(directory)["validation"]
    labels = validation["labels"]
    assert labels["raw_violation"] and not labels["counted_violation"]
    assert labels["safe_requested"] and not labels["strict_safe_requested"]
    assert validation["transitions"][-1]["base_reward"] == 0


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda a: a["noise"].__setitem__((1, 0), 0.7), "tiled"),
        (lambda a: a["noise"].__setitem__((slice(None), 0), 1.1), "magnitude"),
        (lambda a: a["pixels"].__setitem__((0, 0, 0), 17), "RGB64"),
        (lambda a: a["state"].__setitem__(0, 3), "native float32"),
        (lambda a: a["actions"].__setitem__((0, 0), 3), "executed proposal"),
        (lambda a: a["noise"].__setitem__((0, 0), np.nan), "finite"),
    ],
)
def test_corrupt_query_capture_is_never_admitted(tmp_path, mutation, match):
    directory = tmp_path / "episode"
    saved = fixture(directory)
    rewrite(saved["queries"][0], mutation)
    with pytest.raises(ValueError, match=match):
        build_episode(directory, tmp_path / "replay.npz")
    assert not (tmp_path / "replay.npz").exists()


def test_final_observation_and_cross_query_rng_must_align(tmp_path):
    directory = tmp_path / "episode"
    saved = fixture(directory)
    rewrite(saved["final"], lambda a: a["step"].__setitem__((), 9))
    with pytest.raises(ValueError, match="terminal control"):
        build_episode(directory)
    rewrite(saved["final"], lambda a: a["step"].__setitem__((), 10))
    rewrite(saved["queries"][1], lambda a: a["rng_before"].__setitem__(1, 999))
    with pytest.raises(ValueError, match="RNG continuity"):
        build_episode(directory)


def test_missing_scorer_state_cannot_be_replaced_by_final_labels(tmp_path):
    directory = tmp_path / "episode"
    saved = fixture(directory)
    lines = [json.loads(line) for line in saved["journal"].read_text().splitlines()]
    lines = [e for e in lines if not (e["event"] == "scorer_state" and e["step"] == 4)]
    saved["journal"].write_text("".join(json.dumps(e) + "\n" for e in lines))
    with pytest.raises(ValueError, match="every state"):
        build_episode(directory)


def test_explicit_prefix_remap_preserves_original_journal_identity(tmp_path):
    directory = tmp_path / "episode"
    saved = fixture(directory, "E0")
    old = "/preserved/remote"

    def remap_meta(arrays):
        meta = json.loads(str(arrays["meta"]))
        for q in meta["dsrl_queries"]:
            q["record_path"] = str(Path(old) / Path(q["record_path"]).name)
        meta["final_observation_path"] = str(Path(old) / saved["final"].name)
        arrays["meta"] = np.asarray(json.dumps(meta))

    rewrite(saved["recording"], remap_meta)
    lines = [json.loads(line) for line in saved["journal"].read_text().splitlines()]
    for entry in lines:
        for key in ("record_path", "path"):
            if key in entry:
                entry[key] = str(Path(old) / Path(entry[key]).name)
    saved["journal"].write_text("".join(json.dumps(e) + "\n" for e in lines))
    with pytest.raises(ValueError, match="Missing recorded artifact"):
        build_episode(directory)
    result = build_episode(directory, path_map={old: str(directory)})
    assert Path(result["snapshots"][0]["record_path"]).is_file()


def test_admission_refuses_different_existing_replay(tmp_path):
    directory = tmp_path / "episode"
    fixture(directory)
    output = tmp_path / "replay.npz"
    build_episode(directory, output)
    rewrite(output, lambda a: a["rewards"].__setitem__(0, 99))
    with pytest.raises(ValueError, match="Refusing to replace"):
        build_episode(directory, output)


def test_complete_timeout_uses_actual_terminal_potential_and_zero_base_reward(tmp_path):
    directory = tmp_path / "episode"
    fixture(directory, controls=2000, success=False)
    result = build_episode(directory, tmp_path / "timeout.npz")
    validation = result["validation"]
    assert validation["labels"]["unfinished"]
    assert validation["queries"] == 250
    assert validation["transitions"][-1]["phi_after_raw"] == 0.75
    assert validation["transitions"][-1]["phi_after"] == 0
    assert validation["transitions"][-1]["reward"] == -0.25 * 0.75
    assert validation["return_identity"]["corrected_discounted_shaped_return"] == pytest.approx(0)
    with np.load(result["descriptor"]["path"]) as replay:
        assert replay["pixels"].shape[0] == 251
        assert replay["durations"].sum() == 2000


def test_native_goal_confirmation_cannot_be_replaced_by_instantaneous_held(tmp_path):
    directory = tmp_path / "episode"
    saved = fixture(directory)
    rewrite(saved["recording"], lambda a: a["goal_instantaneous"].__setitem__(1, False))
    lines = [json.loads(line) for line in saved["journal"].read_text().splitlines()]
    for entry in lines:
        if entry["event"] == "scorer_state" and entry["step"] == 1:
            entry["instantaneous_goal"] = False
    saved["journal"].write_text("".join(json.dumps(e) + "\n" for e in lines))
    with pytest.raises(ValueError, match="ten-control hold"):
        build_episode(directory)


def legacy_validity(saved):
    def legacy_meta(arrays):
        meta = json.loads(str(arrays["meta"]))
        meta.pop("dsrl_grasp_validity")
        meta["dsrl_recording_valid"] = bool((arrays["grasping"] >= 0).all())
        arrays["meta"] = np.asarray(json.dumps(meta))

    rewrite(saved["recording"], legacy_meta)
    lines = [json.loads(line) for line in saved["journal"].read_text().splitlines()]
    for entry in lines:
        if entry["event"] == "scorer_state":
            entry["valid"] = bool(saved["data"]["grasping"][entry["step"]] >= 0)
    saved["journal"].write_text("".join(json.dumps(e) + "\n" for e in lines))


def test_legacy_false_enum_validity_is_corrected_with_full_native_proof_and_no_raw_edits(tmp_path):
    directory = tmp_path / "episode"
    saved = fixture(directory)
    legacy_validity(saved)
    before = {
        key: hashlib.sha256(saved[key].read_bytes()).hexdigest() for key in ("recording", "journal")
    }
    result = build_episode(directory, tmp_path / "replay.npz")
    audit = result["validation"]["scorer_audit"]
    assert audit["correction_applied"]
    assert audit["recorded_dsrl_recording_valid"] is False
    assert audit["derived_dsrl_recording_valid"] is True
    assert audit["corrected_scorer_state_valid_steps"] == [0]
    assert audit["held_agreement_states"] == 11
    assert audit["input_sha256"] == before
    assert audit["source_proof"] == GRASP_SOURCE_PROOF
    assert audit["raw_artifacts_modified"] is False
    assert result["validation"]["labels"]["strict_safe_requested"]
    assert before == {
        key: hashlib.sha256(saved[key].read_bytes()).hexdigest() for key in ("recording", "journal")
    }


@pytest.mark.parametrize("enum", [0, -2, 2])
def test_unknown_or_non_native_grasp_enum_is_never_reinterpreted(tmp_path, enum):
    directory = tmp_path / "episode"
    saved = fixture(directory)
    rewrite(saved["recording"], lambda a: a["grasping"].__setitem__(0, enum))
    with pytest.raises(ValueError, match="UNKNOWN=0 is ambiguous"):
        build_episode(directory)


@pytest.mark.parametrize("held", [True, None, 0])
def test_legacy_correction_requires_explicit_native_held_equality_at_every_state(tmp_path, held):
    directory = tmp_path / "episode"
    saved = fixture(directory)
    legacy_validity(saved)
    lines = [json.loads(line) for line in saved["journal"].read_text().splitlines()]
    for entry in lines:
        if entry["event"] == "scorer_state" and entry["step"] == 0:
            entry["goal_detail"]["held"] = held
    saved["journal"].write_text("".join(json.dumps(e) + "\n" for e in lines))
    with pytest.raises(ValueError, match="raw TRUE enum must agree"):
        build_episode(directory)


def test_unexplained_old_validity_flag_is_not_erased_by_repair(tmp_path):
    directory = tmp_path / "episode"
    saved = fixture(directory)
    legacy_validity(saved)
    lines = [json.loads(line) for line in saved["journal"].read_text().splitlines()]
    for entry in lines:
        if entry["event"] == "scorer_state" and entry["step"] == 4:
            entry["valid"] = False
    saved["journal"].write_text("".join(json.dumps(e) + "\n" for e in lines))
    with pytest.raises(ValueError, match="journal disagrees"):
        build_episode(directory)


def test_correct_forward_enum_metadata_needs_no_derived_flag_changes(tmp_path):
    directory = tmp_path / "episode"
    fixture(directory)
    audit = build_episode(directory)["validation"]["scorer_audit"]
    assert not audit["correction_applied"]
    assert audit["corrected_scorer_state_valid_steps"] == []
    assert audit["states_verified"] == 11
