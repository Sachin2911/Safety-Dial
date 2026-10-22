"""Audit real runtime-shaped synthetic traces without simulator interactions."""

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.maniguard_phase2_adapter import Phase2Adapter
from helpers.maniguard_phase2_families import family_spec, load_sources
from helpers.maniguard_phase2_model import genome_hash, identity_genome, save_model
from helpers.maniguard_phase2_results import COST_FIELDS, aggregate, read_episode


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def write_trace(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def synthetic_episode(tmp_path, *, family="jar", short=False, stochastic=True,
                      mode="learned", unused_final=False):
    directory = tmp_path / "episode-fixture"
    directory.mkdir(parents=True)
    steps = 4 if short else 12
    genome = identity_genome()
    save_model(directory / "actor.npz", genome)
    spec = family_spec(family)
    encoder = json.loads((ROOT / "configs/maniguard/phase2/encoder.json").read_text())
    job = {"phase": "phase2", "episode_id": str(directory), "output": str(directory),
           "family": family, "scene": "task_0000/base", "seed": 1, "sampling_seed": 12,
           "mode": mode, "stochastic": stochastic, "random_policy": False, "purpose": "engineering" if short else "train",
           "fixed_choices": [0, 0, 1, 0] if mode == "fixed" else None,
           "model": str(directory / "actor.npz"), "actor_sha256": genome_hash(genome),
           "adapter": {"kind": "phase2", "mode": mode},
           "max_steps_override": steps if short else None,
           "action_horizon": steps if short else spec["max_steps"],
           "policy": load_sources()["policies"][family]}
    job["code_hashes"] = {}
    for name in ("encoder", "sources", "families", "completion"):
        relative = f"configs/maniguard/phase2/{name}.json"
        raw = (ROOT / relative).read_bytes()
        target = directory / "study_source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        job["code_hashes"][relative] = hashlib.sha256(raw).hexdigest()
    runtime = b"# synthetic generated evaluator fixture\n"
    (directory / "benchmark_instrumented.py").write_bytes(runtime)
    write_json(directory / "runtime-provenance.json", {
        "revision": load_sources()["maniguard"]["revision"],
        "original_sha256": hashlib.sha256((ROOT / "third_party/ManiGuard/maniguard/eval/benchmark.py"
                                          ).read_bytes()).hexdigest(),
        "runtime_sha256": hashlib.sha256(runtime).hexdigest(),
    })
    write_json(directory / "job.json", job)
    result = {"scene_name": job["scene"], "status": "completed", "steps": steps,
              "ltl_monitored": True, "ltl_violated": False, "counted_violation": False,
              "success": not short, "ever_contacted": True, "ever_grasped": not short}
    observations = []
    for step in range(steps + 1):
        ap = {"target_dropped": False}
        if family == "jar":
            ap.update(jar_closed=step >= 1, jar_on_support=step < 2)
        observations.append({"step": step, "ap": ap, "state": 0, "accepting": True,
                             "doomed": False})
    monitor = {"error": None, "violated": False, "total_steps_monitored": steps + 1,
               "log": observations}
    write_json(directory / "episode/results.jsonl", result)
    write_json(directory / "episode" / (job["scene"] + "_ltl.json"), monitor)
    trace = [{"type": "job", "job": job},
             {"type": "policy_metadata", "metadata": {
                 "serve_config": job["policy"]["config"],
                 "checkpoint_revision": job["policy"]["revision"],
                 "episode_seed_reset": "study_episode_id"}},
             {"type": "phase2_features", "feature_dim": 64, "actor_sha256": job["actor_sha256"],
              "information_access": "observation_only", "encoder": encoder,
              "robot_features": 40, "vision_features": 16, "text_features": 8},
             {"type": "observed", "step": 0,
              "monitor": {"valid": True, "error": None, "observation": observations[0]}}]
    adapter = Phase2Adapter(genome, stochastic=stochastic, seed=12, mode=mode,
                            fixed_choices=job["fixed_choices"])
    decisions = 0
    for step in range(steps):
        if step % 8 == 0:
            trace.append({"type": "query", "step": step})
            trace.append({"type": "chunk", "step": step, "elapsed_seconds": 0.2,
                          "actions": np.zeros((16, 8)).tolist()})
        if step % 4 == 0:
            record = adapter.decide(np.zeros(64), step)
            record.update(type="decision", observation_step=step, actor_sha256=job["actor_sha256"],
                          context={"image_hashes": ["a" * 64, "b" * 64], "text_sha256": "c" * 64,
                                   "checkpoint_sha256": encoder["checkpoint_sha256"]})
            trace.append(record)
            decisions += 1
        trace.append({"type": "proposal", "step": step, "original": [0] * 8,
                      "adapted": [0] * 8, "frame": {"prerequisite": None, "target_engaged": None}})
        trace.append({"type": "executed", "step": step, "action": [0] * 8})
        adapter.applied(np.zeros(8))
        trace.append({"type": "observed", "step": step + 1,
                      "monitor": {"valid": True, "error": None,
                                  "observation": observations[step + 1]}})
        success = not short and step >= 2
        trace.append({"type": "goal", "step": step + 1, "success": success,
                      "detail": {"mode": "held_intersection", "held": success,
                                 "intersects": success}})
    if unused_final:
        record = adapter.decide(np.zeros(64), steps)
        record.update(type="decision", observation_step=steps, actor_sha256=job["actor_sha256"],
                      context={"image_hashes": ["a" * 64, "b" * 64], "text_sha256": "c" * 64,
                               "checkpoint_sha256": encoder["checkpoint_sha256"]})
        trace.append(record)
        decisions += 1
    trace.extend([{"type": "phase2_actor", "actor_sha256": job["actor_sha256"],
                   "decisions": decisions, "stochastic": stochastic},
                  {"type": "result", "result": result}])
    write_trace(directory / "trace.jsonl", trace)
    ledger = dict.fromkeys(COST_FIELDS, 0)
    ledger.update(phase="completed", physics_accounting_attached=True,
                  physics_steps=(steps + 2) * 6, physics_seconds=(steps + 2) / 20,
                  env_step_attempts=steps + 2, env_steps_returned=steps + 2,
                  policy_queries=(steps + 7) // 8, policy_control_steps=steps,
                  adapter_decisions=decisions, encoder_image_calls=decisions,
                  encoder_images=2 * decisions, encoder_text_calls=1, elapsed_seconds=10.0)
    write_json(directory / "ledger.json", ledger)
    return directory, job, result, trace, ledger


def test_full_episode_reconstructs_measured_ppo_trajectory(tmp_path):
    directory, job, _, _, _ = synthetic_episode(tmp_path)
    row = read_episode(directory)
    assert row["valid"], row["invalid_reasons"]
    assert row["ppo_eligible"] and row["full_episode"] and row["outcome"]["U"] == 1
    assert row["trajectory"]["actor_sha256"] == job["actor_sha256"]
    assert len(row["trajectory"]["features"]) == 3
    assert row["costs"]["policy_control_steps"] == 12
    assert row["costs"]["env_step_attempts"] == 14


def test_short_checks_are_valid_measurements_without_full_training_eligibility(tmp_path):
    directory, *_ = synthetic_episode(tmp_path, short=True)
    row = read_episode(directory)
    assert row["valid"], row["invalid_reasons"]
    assert row["short_check"] and not row["full_episode"] and not row["ppo_eligible"]
    assert aggregate([row])["macro"] is None
    assert aggregate([row])["cost_totals"]["policy_control_steps"] == 4


@pytest.mark.parametrize("mode", ["fixed", "contextual"])
def test_reference_modes_never_fabricate_collection_likelihoods(tmp_path, mode):
    directory, *_ = synthetic_episode(tmp_path, mode=mode, stochastic=False)
    row = read_episode(directory)
    assert row["valid"], row["invalid_reasons"]
    assert row["trajectory"]["old_log_probs"] is None and not row["ppo_eligible"]


def test_terminal_unused_decision_costs_count_but_do_not_enter_ppo(tmp_path):
    directory, *_ = synthetic_episode(tmp_path, unused_final=True)
    row = read_episode(directory)
    assert row["valid"], row["invalid_reasons"]
    assert len(row["decisions"]) == 4 and len(row["trajectory"]["actions"]) == 3
    assert row["costs"]["encoder_images"] == 8


@pytest.mark.parametrize("mutation", ["log_prob", "choices", "hidden", "future", "privileged",
                                      "missing_observation", "missing_cost", "causal_order",
                                      "policy_pin", "encoder_pin", "trace_result"])
def test_audit_rejects_actor_causality_and_accounting_corruption(tmp_path, mutation):
    directory, _, _, trace, ledger = synthetic_episode(tmp_path)
    decision = next(row for row in trace if row["type"] == "decision")
    if mutation == "log_prob":
        decision["log_prob"] += 1
    elif mutation == "choices":
        decision["choices"][0] = 10
    elif mutation == "hidden":
        decision["hidden_before"][0] = 1
    elif mutation == "future":
        decision["observation_step"] = 3
    elif mutation == "privileged":
        next(row for row in trace if row["type"] == "proposal")["frame"]["prerequisite"] = False
    elif mutation == "missing_observation":
        trace[:] = [row for row in trace if not (row["type"] == "observed" and row["step"] == 3)]
    elif mutation == "causal_order":
        trace.remove(decision)
        trace.insert(0, decision)
    elif mutation == "policy_pin":
        next(row for row in trace if row["type"] == "policy_metadata")["metadata"][
            "checkpoint_revision"] = "wrong"
    elif mutation == "encoder_pin":
        next(row for row in trace if row["type"] == "phase2_features")["encoder"][
            "checkpoint_sha256"] = "wrong"
    elif mutation == "trace_result":
        next(row for row in trace if row["type"] == "result")["result"]["success"] = False
    else:
        del ledger["encoder_seconds"]
    write_trace(directory / "trace.jsonl", trace)
    write_json(directory / "ledger.json", ledger)
    row = read_episode(directory)
    assert not row["valid"] and not row["ppo_eligible"] and row["invalid_reasons"]
    assert row["costs"]["policy_control_steps"] == 12


def test_failed_process_without_result_preserves_actual_recorded_work(tmp_path):
    directory, _, _, trace, ledger = synthetic_episode(tmp_path)
    (directory / "episode/results.jsonl").unlink()
    trace[:] = trace[:6]  # attempted query, no executed control action yet
    ledger.update(phase="process_exited_without_result", policy_control_steps=0, policy_queries=1,
                  adapter_decisions=0, encoder_image_calls=0, encoder_images=0,
                  physics_steps=60, physics_seconds=0.5)
    write_trace(directory / "trace.jsonl", trace)
    write_json(directory / "ledger.json", ledger)
    row = read_episode(directory)
    assert not row["valid"] and row["outcome"] is None
    assert row["costs"]["physics_steps"] == 60 and row["costs"]["policy_queries"] == 1
    assert aggregate([row])["macro"] is None


def test_trace_lower_bound_prevents_stale_ledger_from_erasing_failed_work(tmp_path):
    directory, _, _, _, ledger = synthetic_episode(tmp_path)
    ledger["policy_control_steps"] = 2
    write_json(directory / "ledger.json", ledger)
    row = read_episode(directory)
    assert not row["valid"] and row["costs"]["policy_control_steps"] == 12
    assert row["cost_provenance"]["policy_control_steps"] == "trace_lower_bound"


def test_family_macro_is_not_pooled_and_invalid_full_records_block_selection(tmp_path):
    directory, *_ = synthetic_episode(tmp_path)
    jar = read_episode(directory)
    stack = copy.deepcopy(jar)
    stack.update(family="stack", episode_id="stack-one", output="/stack-one")
    stack["outcome"].update(C_req=False, S_req=False, U=0)
    extra_jar = copy.deepcopy(jar)
    extra_jar.update(episode_id="jar-two", output="/jar-two")
    summary = aggregate([jar, extra_jar, stack])
    assert summary["macro"]["U"] == 0.5
    assert summary["cost_totals"]["policy_control_steps"] == 36
    stack["valid"] = False
    summary = aggregate([jar, extra_jar, stack])
    assert summary["macro"] is None and summary["counts"]["invalid"] == 1
    assert summary["cost_totals"]["policy_control_steps"] == 36
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate([jar, jar])


def test_archived_configuration_survives_later_workspace_change(tmp_path, monkeypatch):
    import helpers.maniguard_phase2_results as reader

    directory, *_ = synthetic_episode(tmp_path)
    # Point the former live encoder path at an unrelated workspace. Provenance
    # and outcomes still use the recorded, hashed source archive.
    alternate = tmp_path / "changed-workspace"
    (alternate / "configs/maniguard/phase2").mkdir(parents=True)
    write_json(alternate / "configs/maniguard/phase2/encoder.json", {"changed": True})
    helper = alternate / "experiments/helpers/maniguard_phase2_outcomes.py"
    helper.parent.mkdir(parents=True)
    helper.write_bytes((ROOT / "experiments/helpers/maniguard_phase2_outcomes.py").read_bytes())
    monkeypatch.setattr(reader, "ROOT", alternate)
    row = reader.read_episode(directory)
    assert row["valid"], row["invalid_reasons"]


@pytest.mark.parametrize("artifact", ["encoder", "sources", "families", "completion", "runtime"])
def test_archived_source_tampering_invalidates_episode(tmp_path, artifact):
    directory, *_ = synthetic_episode(tmp_path)
    path = (directory / "benchmark_instrumented.py" if artifact == "runtime" else
            directory / "study_source/configs/maniguard/phase2" / (artifact + ".json"))
    path.write_text(path.read_text() + " ")
    row = read_episode(directory)
    assert not row["valid"]
    assert any("checksum mismatch" in reason for reason in row["invalid_reasons"])


def test_optional_encoder_attempts_preserve_failed_work_without_breaking_old_records(tmp_path):
    directory, _, _, _, ledger = synthetic_episode(tmp_path)
    assert read_episode(directory)["valid"]
    ledger.update(encoder_image_attempts=4, encoder_text_attempts=1)
    write_json(directory / "ledger.json", ledger)
    row = read_episode(directory)
    assert row["valid"] and row["costs"]["encoder_image_attempts"] == 4
    summary = aggregate([row])
    assert summary["cost_record_counts"]["encoder_image_attempts"] == 1
    assert summary["optional_cost_missing_records"]["encoder_image_attempts"] == 0
    ledger["encoder_image_attempts"] = 1
    write_json(directory / "ledger.json", ledger)
    row = read_episode(directory)
    assert not row["valid"] and any("attempts below" in reason for reason in row["invalid_reasons"])
