"""Real Study/adapter/feature hook integration using a deterministic fake encoder.

No simulator, policy server, model download or benchmark outcomes are used.
"""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.maniguard_adaptation import Frame
from helpers.maniguard_phase2_model import (
    genome_hash,
    identity_genome,
    random_genome,
    save_model,
    unpack_genome,
)


class FakeEncoder:
    def __init__(self, device=None):
        self.spec = {"schema": "test-only-encoder", "checkpoint_sha256": "fixture"}
        self.runtime_provenance = {"test_only": "loaded-module-and-owning-distribution"}
        self.text_features = None
        self.text_calls = 0
        self.images = []

    def begin_episode(self, prompt):
        assert isinstance(prompt, str) and prompt
        self.text_features = np.full(8, len(prompt) / 100.0)
        self.text_calls += 1
        return self.text_features.copy()

    def encode_images(self, overview, wrist):
        values = (int(overview[0, 0, 0]), int(wrist[0, 0, 0]))
        self.images.append(values)
        return np.full(8, values[0] / 255), np.full(8, values[1] / 255), {
            "test_observation_pixels": list(values), "checkpoint_sha256": "fixture",
        }

    def ledger(self):
        return {"encoder_image_calls": len(self.images), "encoder_images": 2 * len(self.images),
                "encoder_text_calls": self.text_calls, "encoder_seconds": 0.0}


class FakeMonitor:
    def __init__(self, private_label):
        self.private_label = private_label

    def recording_state(self, step):
        return {"valid": True, "step": step, "observation": {"ap": {
            "jar_closed": self.private_label, "goal": self.private_label,
        }}}


def observation(step):
    return {"states": np.r_[np.full(7, step * .001), .01],
            "overview_image": np.full((4, 4, 3), step, dtype=np.uint8),
            "wrist_images": np.full((4, 4, 3), step + 30, dtype=np.uint8),
            "task_descriptions": "move the object"}


def frame(step):
    return Frame(np.full(7, step * .001), np.array([.01, .01]), np.array([step * .001, 0, 0]))


@pytest.fixture
def study_factory(monkeypatch, tmp_path):
    monkeypatch.delenv("SAFETY_DIAL_MANIGUARD_JOB", raising=False)
    runtime = importlib.import_module("helpers.maniguard_phase2_runtime")
    monkeypatch.setattr(runtime, "FrozenContextEncoder", FakeEncoder)
    monkeypatch.setattr(runtime.Phase2Study, "_evaluation_state",
                        lambda self, step: self.write("evaluation_only", step=step,
                                                      private_goal=self.job.get("private_goal")))
    created = []

    def make(genome=None, private_label=False, actor_digest=None, horizon=20):
        genome = identity_genome() if genome is None else genome
        output = tmp_path / f"episode-{len(created)}"
        output.mkdir()
        model = output / "actor.npz"
        save_model(model, genome)
        job = {"phase": "phase2", "output": str(output), "model": str(model),
               "actor_sha256": genome_hash(genome) if actor_digest is None else actor_digest,
               "stochastic": False, "sampling_seed": 4, "mode": "learned",
               "encoder_device": "cpu", "family": "jar", "action_horizon": horizon,
               "private_goal": private_label}
        job_path = output / "job.json"
        job_path.write_text(json.dumps(job))
        monkeypatch.setenv("SAFETY_DIAL_MANIGUARD_JOB", str(job_path))
        study = runtime.Phase2Study()
        created.append(study)
        study.start({"target_name": "fixture", "private_goal": private_label}, observation(0),
                    monitor=FakeMonitor(private_label))
        return study

    yield make
    for study in created:
        study.close()


def rollout(study, *, actions=20, chunk_size=50, future_offset=0.0, private_label=False):
    """Follow evaluator ordering: observe, decide, possibly query, then next action."""
    needs_query = True
    queries = []
    chunk = None
    chunk_index = 0
    proposed = []
    executed = []
    for step in range(actions):
        if needs_query or chunk_index == len(chunk):
            queries.append(step)
            study.before_query(step, observation(step), frame(step))
            chunk = np.zeros((chunk_size, 8), dtype=np.float32)
            chunk[:, :7] = np.arange(chunk_size)[:, None] * .002
            chunk[:, -1] = -1
            if step >= 8:
                chunk[:, :7] += future_offset
            study.after_query(step, chunk)
            chunk = chunk[:16]
            chunk_index = 0
        original = chunk[chunk_index].copy()
        action = study.action(step, chunk_index, original, frame(step))
        proposed.append(original)
        executed.append(action)
        study.applied(step, action)
        study.observed(step + 1, observation(step + 1), FakeMonitor(private_label))
        needs_query = study.should_replan(step + 1, frame(step + 1))
        chunk_index += 1
    study.finish({"status": "completed"})
    return queries, np.array(proposed), np.array(executed)


def test_identity_runs_actual_hook_sequence_with_baseline_queries_and_fresh_features(study_factory):
    study = study_factory()
    queries, proposed, executed = rollout(study)
    assert queries == [0, 8, 16]
    np.testing.assert_array_equal(executed, proposed)
    assert [row["step"] for row in study.decisions] == [0, 4, 8, 12, 16]
    assert study.encoder.images == [(step, step + 30) for step in (0, 4, 8, 12, 16)]
    assert study.counts["encoder_text_calls"] == 1
    assert study.counts["adapter_decisions"] == 5
    assert study.counts["policy_control_steps"] == 20
    assert study.counts["changed_steps"] == 0
    trace = [json.loads(line) for line in (study.output / "trace.jsonl").read_text().splitlines()]
    at_eight = [(row["type"], row.get("step")) for row in trace if row.get("step") == 8]
    assert at_eight.index(("decision", 8)) < at_eight.index(("query", 8))
    assert at_eight.count(("decision", 8)) == 1
    np.testing.assert_allclose(study.decisions[2]["features"][19:26], (.016 - .008) / .2)


def test_future_query_outputs_cannot_change_prequery_decision_features(study_factory):
    first = study_factory()
    rollout(first, future_offset=0)
    second = study_factory()
    rollout(second, future_offset=1)
    assert first.decisions[:3] == second.decisions[:3]
    assert first.decisions[3]["features"] != second.decisions[3]["features"]


def test_evaluation_monitor_and_goal_labels_do_not_change_actor_inputs_or_decisions(study_factory):
    first = study_factory(private_label=False)
    rollout(first, private_label=False)
    second = study_factory(private_label=True)
    rollout(second, private_label=True)
    assert first.decisions == second.decisions
    assert first.adapter.genome.tolist() == second.adapter.genome.tolist()


def test_new_episode_resets_recurrence_and_encoder_history(study_factory):
    genome = random_genome(np.random.default_rng(5))
    first = study_factory(genome=genome)
    rollout(first)
    second = study_factory(genome=genome)
    rollout(second)
    assert first.decisions == second.decisions
    assert second.decisions[0]["hidden_before"] == [0.0] * 8
    assert second.counts["encoder_image_calls"] == 5
    assert second.counts["encoder_text_calls"] == 1


def test_short_chunks_exhaust_without_fabricated_actions_or_offschedule_decisions(study_factory):
    study = study_factory(horizon=8)
    queries, proposed, executed = rollout(study, actions=8, chunk_size=2)
    assert queries == [0, 2, 4, 6]
    assert [row["step"] for row in study.decisions] == [0, 4]
    np.testing.assert_array_equal(executed, proposed)
    trace = [json.loads(line) for line in (study.output / "trace.jsonl").read_text().splitlines()]
    assert [row["step"] for row in trace if row["type"] == "short_chunk_exhaustion"] == [2, 4, 6]


def test_runtime_rejects_actor_hash_mismatch_before_opening_trace(study_factory):
    with pytest.raises(ValueError, match="declared genome"):
        study_factory(actor_digest="0" * 64)


def test_runtime_rejects_privileged_frames_and_unearned_early_queries(study_factory):
    study = study_factory()
    privileged = Frame(np.zeros(7), np.array([.01, .01]), np.zeros(3), prerequisite=False)
    with pytest.raises(ValueError, match="privileged"):
        study.before_query(0, observation(0), privileged)
    study = study_factory()  # A rejected episode is not silently retried.
    study.before_query(0, observation(0), frame(0))
    study.after_query(0, np.zeros((16, 8)))
    with pytest.raises(ValueError, match="shorter"):
        study.before_query(2, observation(2), frame(2))


def test_short_chunk_exception_requires_actual_chunk_exhaustion(study_factory):
    study = study_factory()
    study.before_query(0, observation(0), frame(0))
    study.after_query(0, np.zeros((2, 8)))
    study.action(0, 0, np.zeros(8), frame(0))
    study.applied(0, np.zeros(8))
    study.observed(1, observation(1), FakeMonitor(False))
    with pytest.raises(ValueError):
        study.before_query(1, observation(1), frame(1))


def test_common_prefix_excludes_policy_proposals_beyond_sixteen_actions(study_factory):
    genome = identity_genome()
    unpack_genome(genome)["b_out"].reshape(4, 3)[0] = [0, 0, 1]
    study = study_factory(genome=genome)
    queries, proposed, executed = rollout(study, chunk_size=50)
    assert queries == [0, 16]
    assert study.chunk.shape == (16, 8)
    np.testing.assert_array_equal(executed, proposed)
    # Before querying at 16, the available prefix is exhausted. The feature
    # uses the last accessible proposal (index15), never hidden index16.
    np.testing.assert_allclose(study.decisions[4]["features"][19:26], (.030 - .016) / .2)


def test_generated_hooks_are_inside_action_loop_after_observation_and_goal_checks(tmp_path):
    import ast

    from helpers.maniguard_phase2_source import prepare_runtime

    generated = prepare_runtime(tmp_path)
    tree = ast.parse(generated.read_text())
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}

    def calls(owner, name):
        return [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == name
                and isinstance(node.func.value, ast.Name) and node.func.value.id == owner]

    def nearest_loop(node):
        while node in parents:
            node = parents[node]
            if isinstance(node, (ast.For, ast.While)):
                return node
        return None

    replans = calls("STUDY", "should_replan")
    assert len(replans) == 1
    replan = replans[0]
    action_loop = nearest_loop(replan)
    assert isinstance(action_loop, ast.For)
    assert isinstance(action_loop.target, ast.Name) and action_loop.target.id == "ci"
    replan_if = parents[replan]
    assert isinstance(replan_if, ast.If) and replan_if in action_loop.body
    assert any(isinstance(node, ast.Break) for node in replan_if.body)
    observed = calls("STUDY", "observed")
    goal_checks = calls("goal_checker", "check")
    assert len(observed) == len(goal_checks) == 1
    assert nearest_loop(observed[0]) is action_loop
    assert nearest_loop(goal_checks[0]) is action_loop
    assert observed[0].lineno < goal_checks[0].lineno < replan.lineno
    success_breaks = [node for node in ast.walk(action_loop) if isinstance(node, ast.Break)
                      and node.lineno < replan.lineno
                      and isinstance(parents[node], ast.If)
                      and "consec_success" in ast.unparse(parents[node].test)]
    assert len(success_breaks) == 1
    start = calls("STUDY", "start")
    assert len(start) == 1
    assert [ast.unparse(argument) for argument in start[0].args] == [
        "scene_info", "obs", "env", "robot", "monitor",
    ]
    assert start[0].lineno < calls("STUDY", "before_query")[0].lineno < action_loop.lineno


def _episode_cli_module():
    path = Path(__file__).resolve().parents[1] / "scripts/maniguard_phase2_episode.py"
    spec = importlib.util.spec_from_file_location("phase2_episode_cli_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("changes", ["train", "stochastic", "fixed", "nonidentity"])
def test_baseline_cli_rejects_nonengineering_nonidentity_modes_before_execution(
    monkeypatch, tmp_path, changes,
):
    module = _episode_cli_module()
    genome = random_genome(np.random.default_rng(6)) if changes == "nonidentity" else identity_genome()
    model = tmp_path / "input.npz"
    save_model(model, genome)
    output = tmp_path / "episode"
    args = ["phase2_episode", "--family", "jar", "--scene", "task_0013/base", "--purpose",
            "train" if changes == "train" else "engineering", "--seed", "0", "--model", str(model),
            "--output", str(output), "--baseline-executor", "--max-steps", "32"]
    if changes == "stochastic":
        args.append("--stochastic")
    if changes == "fixed":
        args.extend(["--mode", "fixed", "--fixed-choices", "0", "0", "0", "0"])
    monkeypatch.setattr(sys, "argv", args)

    def unexpected_execution(*args, **kwargs):
        raise AssertionError("Invalid baseline configuration reached runtime execution")

    monkeypatch.setattr(module.runpy, "run_path", unexpected_execution)
    with pytest.raises(ValueError, match="engineering identity"):
        module.main()
    assert not output.exists()


def test_baseline_cli_routes_identity_to_original_executor_before_physics(monkeypatch, tmp_path):
    from types import SimpleNamespace

    module = _episode_cli_module()
    model = tmp_path / "input.npz"
    save_model(model, identity_genome())
    output = tmp_path / "episode"
    environment = dict(module.os.environ)
    monkeypatch.setattr(module, "os", SimpleNamespace(environ=environment))
    monkeypatch.setattr(sys, "argv", [
        "phase2_episode", "--family", "jar", "--scene", "task_0013/base", "--purpose",
        "engineering", "--seed", "0", "--model", str(model), "--output", str(output),
        "--baseline-executor", "--max-steps", "32",
    ])
    invoked = []
    monkeypatch.setattr(module.runpy, "run_path", lambda path, run_name: invoked.append((path, run_name)))
    module.main()
    assert len(invoked) == 1 and invoked[0][1] == "__main__"
    job = json.loads((output / "job.json").read_text())
    assert job["phase"] == "phase2_baseline_check"
    assert job["adapter"]["kind"] == "identity" and job["adapter"]["horizon"] == 8
    assert sys.argv[sys.argv.index("--execute-horizon") + 1] == "8"
    assert sys.argv[sys.argv.index("--max-steps") + 1] == "32"
    source = Path(invoked[0][0]).read_text()
    assert "from helpers.maniguard_runtime import STUDY" in source
    assert "from helpers.maniguard_phase2_runtime import STUDY" not in source


def test_close_preserves_encoder_work_that_failed_between_decisions(study_factory):
    study = study_factory()
    study.encoder.ledger = lambda: {"encoder_image_calls": 0, "encoder_images": 0,
                                    "encoder_text_calls": 1, "encoder_image_attempts": 1,
                                    "encoder_text_attempts": 1, "encoder_seconds": 1.25}
    study.close()
    ledger = json.loads((study.output / "ledger.json").read_text())
    assert ledger["encoder_image_attempts"] == 1
    assert ledger["encoder_seconds"] == 1.25
    assert ledger["phase"] == "process_exited_without_result"


def test_runtime_records_loaded_encoder_provenance_separately(study_factory):
    study = study_factory()
    study.trace.flush()
    rows = [json.loads(line) for line in (study.output / 'trace.jsonl').read_text().splitlines()]
    row = next(item for item in rows if item['type'] == 'phase2_features')
    assert row['runtime_provenance'] == study.encoder.runtime_provenance
    assert row['encoder'] == study.encoder.spec
    assert 'runtime_provenance' not in row['encoder']
