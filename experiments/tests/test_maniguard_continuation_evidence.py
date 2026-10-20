"""Corrupt compound evidence deliberately; no physics, encoder or model calls."""

import copy
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from helpers.maniguard_adaptation import Frame
from helpers.maniguard_continuation_evidence import (
    choice_record,
    decode_bounds,
    encode_bounds,
    replay_stream,
)
from helpers.maniguard_continuation_handoff import HandoffReference
from helpers.maniguard_continuation_limits import DEFAULT_LIMITS
from helpers.maniguard_latent import feature_vector, initial_weights
from helpers.maniguard_latent_policy import array_hash


def response(choice, features, index=0):
    raw = np.arange(128, dtype=np.float32).reshape(16, 8) / 100
    receipt = {
        "episode": "synthetic", "seed": 123, "query_index": index,
        "weights_sha256": choice["controller_sha256"],
        "features_sha256": array_hash(features),
        "identity_native_path": choice["branch"] == "native", "audit": False,
        "status": "returned", "shape": [1, 16, 32],
        "offset_sha256": choice["offset_sha256"], "code": choice["code"],
        "rng_before": [0, 123 + index], "rng_after": [0, 124 + index],
        "actions_sha256": array_hash(raw),
    }
    return {"actions": raw, "latent_receipt": receipt}


def stream(mode="privileged_lid_handoff"):
    weights = initial_weights()[1]
    reference = HandoffReference(weights, mode=mode, clock_switch_step=8)
    rows = [{"type": "continuation_contract", "contract": reference.contract,
             **encode_bounds(np.full(8, -1.), np.ones(8))}]
    previous = previous_eef = None
    for step in range(17):
        monitor = {"valid": True, "observation": {
            "step": step, "ap": {"jar_closed": step > 0, "jar_on_support": True}, "doomed": False,
        }}
        rows.append({"type": "observed", "step": step, "monitor": monitor})
        if reference.privileged:
            reference.observe(step, closed=step > 0, supported=True, violated=False)
        if step == 16:
            break
        if step % 8 == 0:
            features = feature_vector(
                np.zeros(7), np.zeros(2), np.zeros(3), *[np.zeros(8)] * 3,
                previous_executed=previous, previous_eef=previous_eef,
            )
            choice = choice_record(reference.select(step, features))
            result = response(choice, features, step // 8)
            raw = result["actions"]
            rows.extend([
                {"type": "latent_features", "step": step, "features": features,
                 "frame": {"joints": np.zeros(7), "fingers": np.zeros(2), "eef": np.zeros(3)},
                 "code": choice["code"],
                 "selected_controller_sha256": choice["controller_sha256"]},
                {"type": "continuation_choice", "step": step, "choice": choice},
                {"type": "continuation_request", "step": step, "request": {
                    "index": step // 8, "episode": "synthetic", "seed": 123,
                    "controller_sha256": choice["controller_sha256"],
                    "features_sha256": array_hash(features)}},
                {"type": "noise_receipt", "step": step, "receipt": result["latent_receipt"],
                 "raw_actions": raw, "raw_action_dtype": str(raw.dtype)},
                {"type": "chunk", "step": step, "actions": raw},
            ])
            previous_eef = np.zeros(3)
        action = raw[step % 8].copy()
        action[-1] = np.sign(action[-1]) if abs(action[-1]) > .01 else -1.
        previous = np.clip(action, -1, 1)
        rows.extend([
            {"type": "proposal", "step": step, "original": action, "adapted": action},
            {"type": "executed", "step": step, "action": previous},
        ])
    return rows


def replay(rows, mode="privileged_lid_handoff"):
    return replay_stream(rows, initial_weights()[1], mode=mode, debounce=3,
                         clock_switch_step=8, episode="synthetic", seed=123, steps=16)


@pytest.mark.parametrize("mode", ["native_only", "closure_only", "clock_handoff",
                                 "privileged_lid_handoff"])
def test_complete_synthetic_stream_checks_both_components_and_executed_history(mode):
    result = replay(stream(mode), mode)
    assert result["policy_requests_admitted"] == result["policy_returns_verified"] == 2
    assert result["actual_environment_inputs_verified"] == 16
    assert result["complete_native_rng_chain_verified"]
    assert result["state"]["switch_query_step"] == (8 if mode.endswith("handoff") else None)


@pytest.mark.parametrize("corruption", ["branch", "seed", "rng", "offset", "identity", "code"])
def test_wrong_branch_seed_rng_offset_and_code_are_rejected(corruption):
    rows = copy.deepcopy(stream())
    row = next(r for r in rows if r["type"] == "noise_receipt" and r["step"] == 8)
    receipt = row["receipt"]
    if corruption == "branch":
        receipt["weights_sha256"] = "f" * 64
    elif corruption == "seed":
        receipt["seed"] += 1
    elif corruption == "rng":
        receipt["rng_before"] = [0, 123]
    elif corruption == "offset":
        receipt["offset_sha256"] = "f" * 64
    elif corruption == "identity":
        receipt["identity_native_path"] = False
    else:
        receipt["code"] = [.5, 0, 0, 0]
    with pytest.raises(ValueError):
        replay(rows)


@pytest.mark.parametrize("kind", ["observed", "continuation_choice", "continuation_request",
                                 "noise_receipt", "chunk", "executed"])
def test_missing_factual_events_are_rejected(kind):
    rows = stream()
    index = next(i for i, row in enumerate(rows) if row["type"] == kind)
    with pytest.raises((ValueError, AssertionError)):
        replay(rows[:index] + rows[index + 1:])


def test_actual_command_corruption_and_native_code_mislabel_are_rejected():
    for kind in ("executed", "latent_features"):
        rows = copy.deepcopy(stream())
        row = next(r for r in rows if r["type"] == kind and r["step"] == 8)
        if kind == "executed":
            row["action"][0] += .1
        else:
            row["code"] = [.5, 0, 0, 0]
        with pytest.raises(AssertionError):
            replay(rows)


def test_unbounded_native_outer_action_space_has_finite_json_encoding():
    record = encode_bounds(np.full(8, -np.inf), np.full(8, np.inf))
    assert json.loads(json.dumps(record, allow_nan=False)) == record
    low, high = decode_bounds(record)
    np.testing.assert_array_equal(low, np.full(8, -np.inf))
    np.testing.assert_array_equal(high, np.full(8, np.inf))
    with pytest.raises(ValueError):
        encode_bounds(np.full(8, np.nan), np.ones(8))


def job_files(tmp_path, mode="clock_handoff"):
    weights = initial_weights()[1]
    reference = HandoffReference(weights, mode=mode, clock_switch_step=0)
    controller = tmp_path / "controller.npz"
    np.savez(controller, weights=weights)
    job = {"study": "continuation-feasibility-v1", "family": "jar", "random_policy": False,
           "max_steps_override": None, "controller": str(controller), "output": str(tmp_path),
           "controller_sha256": reference.closure.sha256(), "episode_id": "synthetic",
           "runtime_limits": dict(DEFAULT_LIMITS),
           "continuation": {"mode": mode, "debounce": 3, "clock_switch_step": 0},
           "continuation_contract_sha256": reference.contract_sha256}
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job))
    return path


def test_import_under_active_job_constructs_only_the_compound_global(tmp_path):
    path = job_files(tmp_path)
    code = """
import helpers.maniguard_continuation_runtime as compound
import helpers.maniguard_latent_runtime as latent
import helpers.maniguard_runtime as base
assert latent.STUDY is None and base.STUDY is None
assert type(compound.STUDY).__name__ == 'ContinuationStudy'
compound.STUDY.close()
"""
    result = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True,
                            env={**os.environ, "SAFETY_DIAL_MANIGUARD_JOB": str(path)})
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert sum(row["type"] == "job" for row in rows) == 1


@pytest.mark.parametrize("fail", [False, True])
def test_encoder_once_actual_native_code_and_admission_before_client_call(tmp_path, monkeypatch, fail):
    from helpers.maniguard_continuation_runtime import ContinuationStudy

    path = job_files(tmp_path)
    monkeypatch.setenv("SAFETY_DIAL_MANIGUARD_JOB", str(path))
    study = ContinuationStudy()
    calls = []

    class Encoder:
        text_features = np.zeros(8)

        def encode_images(self, *args):
            calls.append("encode")
            return np.zeros(8), np.zeros(8), {}

        def ledger(self):
            return {}

    class Client:
        def __init__(self, **kwargs):
            pass

        def infer(self, observation):
            rows = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
            assert rows[-1]["type"] == "continuation_request"
            assert study.counts["native_policy_requests_admitted"] == 1
            if fail:
                raise RuntimeError("synthetic transport failure")
            return response(study.choice, observation["latent_features"])

    monkeypatch.setitem(sys.modules, "helpers.maniguard_client", SimpleNamespace(StudyClient=Client))
    study.encoder = Encoder()
    study.expected_policy_seed = 123
    image = np.zeros((2, 2, 3), np.uint8)
    obs = {"overview_image": image, "wrist_images": image, "states": np.zeros(8),
           "task_descriptions": "synthetic"}
    frame = Frame(np.zeros(7), np.zeros(2), np.zeros(3))
    try:
        study.before_query(0, obs, frame)
        assert calls == ["encode"] and study.choice["code"] == [0.] * 4
        client, _ = study.connect_policy(SimpleNamespace(use_openpi_client=True, host="", port=0), None)
        request = {**study.policy_payload, "study_episode_id": "synthetic", "episode_seed": 123}
        if fail:
            with pytest.raises(RuntimeError, match="synthetic transport"):
                client.infer(request)
        else:
            client.infer(request)
        assert study.counts["native_policy_requests_admitted"] == 1
        assert study.counts["native_policy_responses_verified"] == int(not fail)
    finally:
        study.close()
