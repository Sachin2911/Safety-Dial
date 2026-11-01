"""Software checks; synthetic objective results are never ManiGuard evidence."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from maniguard_adapter import AdapterSpec, ResidualAdapter
from maniguard_evolution import Search
from maniguard_evolution_episode import (
    BudgetExhausted,
    ManiGuardEvaluator,
    atomic_json,
    totals,
)
from maniguard_evolution_scoring import candidate_score, ordered_progress
from prepare_maniguard_evolution import instrument
from run_maniguard_evolution import validate_config


ROOT = Path(__file__).resolve().parents[2]


def observation():
    return {"valid": True, "q": np.zeros(7), "dq": np.zeros(7),
            "eef_pos": np.array([0.1, 0.03, 0.2]), "jar_pos": np.zeros(3),
            "eef_quat": [0, 0, 0, 1], "jar_quat": [0, 0, 0, 1], "grasp": -1,
            "ap": {"jar_closed": True, "jar_on_support": True,
                   "jar_upright": True, "jar_dropped": False}, "contact": True}


def command():
    return np.array([0.1, -0.3, 0.2, -1.2, 0.4, 1.2, -0.7, 1], np.float32)


@pytest.mark.parametrize("enabled", [True, False])
def test_initial_controller_preserves_native_commands_exactly(enabled):
    spec = AdapterSpec()
    model = ResidualAdapter(spec, spec.initial_parameters(17), enabled=enabled)
    for step in range(100):
        native = command()
        native[7] = 1 if step % 2 else -1
        output, _ = model.apply(step, native, observation())
        assert np.array_equal(output, native) and output.dtype == native.dtype


def test_disabled_controller_bypasses_large_nonzero_policy():
    spec = AdapterSpec()
    model = ResidualAdapter(spec, np.full(spec.parameter_count, 3), enabled=False)
    for step in range(40):
        output, _ = model.apply(step, command(), observation())
        assert np.array_equal(output, command())


def test_residual_bound_and_discrete_gripper_for_extreme_parameters():
    spec = AdapterSpec()
    model = ResidualAdapter(spec, np.full(spec.parameter_count, 3))
    outputs = [model.apply(i, command(), observation())[0] for i in range(100)]
    assert max(np.max(np.abs(row[:7] - command()[:7])) for row in outputs) <= 0.080001
    assert all(row[7] in (-1, 1) for row in outputs)
    assert any(row[7] == -1 for row in outputs)


def test_controller_memory_and_episode_reset():
    spec = AdapterSpec()
    theta = np.full(spec.parameter_count, 0.1)
    first, second = ResidualAdapter(spec, theta), ResidualAdapter(spec, theta)
    initial, _ = first.apply(0, command(), observation())
    later, _ = first.apply(1, command(), observation())
    reset, _ = second.apply(0, command(), observation())
    assert np.array_equal(initial, reset)
    assert not np.array_equal(initial, later)


def test_world_translation_does_not_change_relative_position_features():
    spec = AdapterSpec()
    a, b = observation(), observation()
    a["jar_pos"] = a["jar_pos"] + [3, 5, 7]
    a["eef_pos"] = a["eef_pos"] + [3, 5, 7]
    _, left = ResidualAdapter(spec, spec.initial_parameters()).apply(0, command(), a)
    _, right = ResidualAdapter(spec, spec.initial_parameters()).apply(0, command(), b)
    np.testing.assert_allclose(left["features"], right["features"], atol=1e-14)


@pytest.mark.parametrize("mutation", ["nan", "invalid", "missing_ap", "bad_grasp"])
def test_invalid_controller_observations_are_explicit_errors(mutation):
    spec, obs = AdapterSpec(), observation()
    if mutation == "nan":
        obs["q"][0] = np.nan
    elif mutation == "invalid":
        obs["valid"] = False
    elif mutation == "missing_ap":
        del obs["ap"]["jar_closed"]
    else:
        obs["grasp"] = 99
    with pytest.raises(ValueError):
        ResidualAdapter(spec, spec.initial_parameters()).apply(0, command(), obs)


def test_ordered_progress_requires_sustained_grasp_and_prior_supported_closure():
    assert ordered_progress([1] * 9, [1] * 9, [1] * 9) == 1
    assert ordered_progress([1] * 10, [1] * 10, [1] * 10) == 2
    assert ordered_progress([1] * 11, [1] * 10 + [0], [1] * 11) == 3
    assert ordered_progress([0] * 11, [1] * 11, [1] * 11) == 0


def outcome(primary=False, progress=0, violation=False):
    return {"valid": True, "strict_safe_completion": primary,
            "benchmark_safe_completion": primary, "requested_completion": primary,
            "raw_violation": violation, "counted_violation": violation,
            "unfinished": not primary, "engaged": True, "safe_stage_progress": progress}


def test_one_completion_outweighs_all_secondary_progress_and_violation_differences():
    complete = [outcome(True)] + [outcome(violation=True) for _ in range(5)]
    partial = [outcome(progress=3) for _ in range(6)]
    assert candidate_score(complete) > candidate_score(partial)
    assert candidate_score(partial, "completion_only") == 0
    assert candidate_score([{"valid": False}]) < candidate_score([outcome()])


def test_failed_starts_and_seed_queries_remain_in_accounting(tmp_path):
    job = tmp_path / "episodes" / "failed"
    job.mkdir(parents=True)
    atomic_json(job / "reservation.json", {})
    atomic_json(job / "preseed-query.json", {"attempted": 1, "completed": 0})
    atomic_json(job / "process.json", {"pid": 1})
    actual = totals(tmp_path)
    assert actual["reserved_starts"] == 1 and actual["model_query_attempts"] == 1
    assert actual["model_query_returns"] == 0 and actual["unknown_bootstrap_starts"] == 1


def test_budget_prevents_an_extra_episode_before_it_is_reserved(tmp_path):
    cfg = json.loads((ROOT / "configs/maniguard_evolution/development-v2.json").read_text())
    cfg["limits"]["max_starts"] = 1
    backend = ManiGuardEvaluator(tmp_path, cfg, tmp_path, ROOT)
    job = tmp_path / "episodes" / "previous"
    job.mkdir()
    atomic_json(job / "reservation.json", {})
    with pytest.raises(BudgetExhausted, match="Episode allowance"):
        backend.admission()
    assert totals(tmp_path)["reserved_starts"] == 1


def toy_config():
    cfg = json.loads((ROOT / "configs/maniguard_evolution/development-v2.json").read_text())
    cfg["adapter"] = asdict(AdapterSpec(hidden_size=1))
    cfg["search"].update(population=4, generations=2, tasks=[0], policy_seeds=[101, 102])
    cfg["validation"].update(tasks=[3], policy_seeds=[201, 202, 203])
    return cfg


class ToyEvaluator:
    def __init__(self, interrupt_at=None):
        self.calls = []
        self.interrupt_at = interrupt_at

    def __call__(self, label, task, seed, artifact):
        if self.interrupt_at is not None and len(self.calls) == self.interrupt_at:
            raise KeyboardInterrupt("Simulated orchestrator interruption")
        self.calls.append((label, task, seed, artifact["parameters"]))
        # Continuous analytic objective, exclusively for testing optimizer orchestration.
        value = float(3 * np.exp(-np.mean((np.asarray(artifact["parameters"]) - 0.2) ** 2)))
        return outcome(progress=value)


@pytest.mark.parametrize("method", ["cmaes", "random"])
def test_search_runs_full_pipeline_and_exports_controller(tmp_path, method):
    cfg = toy_config()
    cfg["search"]["method"] = method
    backend = ToyEvaluator()
    search = Search(cfg, tmp_path, backend)
    result = search.run()
    assert result["status"] == "completed" and result["generation"] == 2
    assert len(backend.calls) == 2 * (4 + 1) + 2 * 3
    assert result["validation"]["selected"]["valid_episodes"] == 3
    exported = json.loads((tmp_path / "controller.json").read_text())
    assert exported["parameters"] == result["selected"]


def test_resume_preserves_candidates_rng_and_validation_without_repeating_finished_jobs(tmp_path):
    cfg = toy_config()
    uninterrupted = Search(cfg, tmp_path / "full", ToyEvaluator()).run()
    interrupted = ToyEvaluator(interrupt_at=7)
    with pytest.raises(KeyboardInterrupt):
        Search(cfg, tmp_path / "resumed", interrupted).run()
    continuation = ToyEvaluator()
    resumed = Search(cfg, tmp_path / "resumed", continuation, resume=True).run()
    np.testing.assert_array_equal(uninterrupted["selected"], resumed["selected"])
    assert len(interrupted.calls) + len(continuation.calls) == 16
    assert [r["scores"] for r in uninterrupted["generations"]] == [
        r["scores"] for r in resumed["generations"]]


def test_resume_rejects_changed_objective(tmp_path):
    cfg = toy_config()
    Search(cfg, tmp_path, ToyEvaluator())
    changed = deepcopy(cfg)
    changed["search"]["objective"] = "completion_only"
    with pytest.raises(ValueError, match="differs"):
        Search(changed, tmp_path, ToyEvaluator(), resume=True)


def test_held_out_tasks_cannot_enter_training_or_development_validation():
    cfg = toy_config()
    panels = json.loads((ROOT / "progress/panels.json").read_text())
    assert validate_config(cfg, panels) > 0
    cfg["search"]["tasks"] = [panels["held_out"][0]]
    with pytest.raises(ValueError, match="development tasks"):
        validate_config(cfg, panels)


def test_missing_simulator_launch_environment_is_detected_before_a_query():
    cfg = toy_config()
    cfg["environment"].pop("OMNI_KIT_ACCEPT_EULA")
    panels = json.loads((ROOT / "progress/panels.json").read_text())
    with pytest.raises(ValueError, match="EULA launch setting"):
        validate_config(cfg, panels)


def test_predecessor_work_is_visible_in_aggregate_accounting(tmp_path):
    old = tmp_path / "old"
    old.mkdir()
    atomic_json(old / "accounting.json", {"reserved_starts": 1, "model_query_attempts": 1})
    new = tmp_path / "new"
    new.mkdir()
    atomic_json(new / "config.json", {"engineering_predecessor": {"run": str(old)}})
    actual = totals(new)
    assert actual["reserved_starts"] == 0
    assert actual["aggregate_with_previous_versions"]["reserved_starts"] == 1
    assert actual["aggregate_with_previous_versions"]["model_query_attempts"] == 1


def test_instrumentation_rejects_wrong_source_without_mutating_it():
    source = "print('unrelated evaluator')\n"
    with pytest.raises(ValueError, match="pinned unmodified"):
        instrument(source, Path("/workspace/checkout"))
    assert source == "print('unrelated evaluator')\n"
