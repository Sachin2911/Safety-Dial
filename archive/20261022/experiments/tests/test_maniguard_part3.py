"""Model/partition/gate checks on synthetic fixtures, never experimental evidence."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.maniguard_part3_data import partition_for
from helpers.maniguard_part3_model import (
    check_gate,
    correlation,
    design_for,
    fit_representation,
    ridge,
    state_for,
    trust_scores,
    windows,
)
from helpers.maniguard_part3_protocol import PARTITIONS, criteria, recording_plan


def synthetic(seed=5, length=40):
    rng = np.random.default_rng(seed)
    return {"episode_id": f"synthetic-{seed}", "scene": "synthetic-fixture",
            "visual": rng.normal(size=(length, 40)).astype(np.float32),
            "robot": rng.normal(size=(length, 9)).astype(np.float32),
            "action": rng.normal(size=(length, 8)).astype(np.float32),
            "query": np.arange(length) % 4 == 0,
            "target": rng.normal(size=(length, 2)).astype(np.float32),
            "correction": np.ones(length, np.float32)}


def test_whole_task_partitions_are_disjoint_and_exclude_exposed_validation():
    scenes = [scene for group in PARTITIONS.values() for scene in group]
    assert len(scenes) == len(set(scenes))
    assert "task_0029/base" not in scenes and "task_0032/base" not in scenes
    for group, entries in PARTITIONS.items():
        assert all(partition_for(scene) == group for scene in entries)
    with pytest.raises(ValueError):
        partition_for("task_0019/base")


def test_recording_option_is_exactly_eight_matched_full_slots():
    jobs = recording_plan()["jobs"]
    assert len(jobs) == 8 and len({j["episode_id"] for j in jobs}) == 8
    for block in range(4):
        selected = [j for j in jobs if j["block"] == block]
        assert len(selected) == 2 and {j["condition"] for j in selected} == {"B", "C"}
        assert len({(j["scene"], j["seed"]) for j in selected}) == 1
        assert all(j["action_horizon"] == 1100 for j in selected)


def test_fit_scalers_and_pca_do_not_consume_holdout():
    fit, holdout = synthetic(), synthetic(7)
    artifacts = fit_representation([fit])
    before = {k: v.copy() for k, v in artifacts.items()}
    holdout["visual"] *= 1e6
    projected = state_for(holdout, artifacts)
    assert projected.shape == (40, 41)
    for key, value in before.items():
        np.testing.assert_array_equal(value, artifacts[key])
    np.testing.assert_allclose(artifacts["visual_mean"], fit["visual"].mean(0))


def test_dynamics_pairs_use_executed_actions_and_causal_previous_state():
    e = synthetic()
    a = fit_representation([e])
    a.update(action_mean=np.zeros(8), action_scale=np.ones(8))
    design, target = design_for(e, a)
    state = state_for(e, a)
    np.testing.assert_array_equal(design[:, -8:], e["action"][1:-1])
    np.testing.assert_array_equal(design[:, -16:-8], e["action"][:-2])
    np.testing.assert_array_equal(design[:, :41], state[1:-1])
    np.testing.assert_array_equal(design[:, 41:82], state[:-2])
    np.testing.assert_allclose(target, state[2:] - state[1:-1])
    w = windows([e], a)
    for i, step in enumerate(w["step"]):
        assert e["query"][step] and step > 0
        np.testing.assert_array_equal(w["actions"][i], e["action"][step:step + 4])
        np.testing.assert_array_equal(w["future"][i], state[step + 1:step + 5])


def test_frozen_criteria_equal_public_predata_receipt():
    root = Path(__file__).resolve().parents[2]
    receipt = json.loads((root / "configs/maniguard/followup/part3-data-decision.json").read_text())
    assert criteria() == receipt["forecast_criteria"]
    assert receipt["common_recording_required"] is True
    assert receipt["part1_comparison_excluded"] is True


def test_sparse_trust_bins_use_explicit_unconditional_fallback():
    calibration = {"bins": [0, 1, 2, 100], "error_quantiles": [0.2, 0.6, 0.6],
                   "disagreement_median_scale": 1}
    m, c = trust_scores(np.ones(3), np.array([0.1, 1.5, 3]), calibration)
    np.testing.assert_array_equal(m, np.ones(3))
    np.testing.assert_allclose(c, [0.3, 0.9, 0.9])


def test_inconclusive_gate_never_turns_missing_failure_support_into_success():
    metrics = {"audit_episodes": 4, "changing_windows": 200,
               "near_calibration_episodes": 0, "near_audit_episodes": 0,
               "crossing_audit_episodes": 0, "varying_progress_windows": 100,
               "persistence_improvement_1": 0.2, "persistence_improvement_4": 0.3,
               "shuffled_improvement_4": 0.2, "relative_ridge_error_4": 0.8,
               "actual_margin_mae_deg": 2, "near_margin_mae_deg": None,
               "progress_mae_m": 0.01, "false_reassurance_fraction": None,
               "progress_ranking_spearman": 0.6, "M_acceptance": 0.4,
               "C_acceptance": 0.4, "screen_seconds_per_16": 1}
    result = check_gate(metrics)
    assert not result["eligible"] and result["status"] == "inconclusive_support"
    assert "near_calibration_episodes" in result["failed_checks"]
    assert not result["accuracy"]["false_reassurance"]
    metrics.update(near_calibration_episodes=2, near_audit_episodes=2,
                   crossing_audit_episodes=1, near_margin_mae_deg=2,
                   false_reassurance_fraction=0)
    assert check_gate(metrics)["eligible"]
    metrics["persistence_improvement_4"] = -0.1
    assert check_gate(metrics)["status"] == "failed_accuracy"


def test_ridge_and_rank_helpers_handle_ties_and_static_windows():
    x = np.arange(20).reshape(-1, 1)
    weights = ridge(x, 2 * x + 3, alpha=0)
    np.testing.assert_allclose(weights[:, 0], [2, 3], atol=1e-10)
    assert correlation([1, 1, 2, 3], [2, 2, 3, 4]) == pytest.approx(1)
    assert correlation([1, 1], [2, 3]) is None


def test_evolution_budget_shared_once_and_full_horizon():
    from helpers.maniguard_part3_search import (
        GRID,
        comparison_jobs,
        initial_population,
        jobs_for,
        mutation_pool,
        select_descendants,
    )
    assert len(GRID) == 160
    initial = initial_population()
    assert initial[0] == [0, 0, 4, 0] and len({tuple(g) for g in initial}) == 4
    shared = mutation_pool(initial, [0, 1, 2, 3])
    assert len(shared["pool"]) <= 16
    for candidate in shared["pool"]:
        assert candidate["genome"][2:] == initial[candidate["parent"]][2:]
        assert candidate["genome"] != initial[candidate["parent"]]
    descendants = select_descendants(shared)
    # Abstention falls back to the common uniform order. It cannot remove any
    # candidate from the exploration pool or fabricate a measured fitness.
    assert descendants["E"]["population"] == descendants["M"]["population"]
    assert descendants["E"]["population"] == descendants["C"]["population"]
    populations = {m: d["population"] for m, d in descendants.items()}
    initial_jobs = jobs_for({"shared": initial}, "initial")
    descendant_jobs = jobs_for(populations, "descendants")
    confirmations = jobs_for({m: p[:2] for m, p in populations.items()}, "confirmation")
    comparison = comparison_jobs({m: p[0] for m, p in populations.items()},
                                 {"free_step_limit": .04, "grasp_step_limit": .02,
                                  "correction_threshold": None, "minimum_interval": 4})
    assert list(map(len, (initial_jobs, descendant_jobs, confirmations, comparison))) == [8, 24, 12, 20]
    all_jobs = initial_jobs + descendant_jobs + confirmations + comparison
    assert len(all_jobs) == 64
    assert all(j["action_horizon"] == 1100 for j in all_jobs)
    assert len({j["episode_id"] for j in all_jobs}) == 64
    assert all(d["slots"][0]["role"] == "reevaluated_elite" for d in descendants.values())
    assert all(d["slots"][1]["role"] == "unassisted_timing_mutant" for d in descendants.values())


def test_model_ranking_does_not_select_timing_gene_or_override_physical_fitness():
    from helpers.maniguard_part3_search import (
        initial_population,
        mutation_pool,
        rank_population,
        select_descendants,
    )
    population = initial_population()
    shared = mutation_pool(population, [0, 1, 2, 3])
    records = [{**p, "ranking": [0, -p["birth"], 0], "M_trusted": True, "C_trusted": False}
               for p in shared["pool"]]
    selected = select_descendants(shared, records)
    assert selected["M"]["selected_pool_indices"] == [len(records) - 1, len(records) - 2]
    assert selected["C"]["population"] == selected["E"]["population"]
    assert selected["M"]["population"][1] == selected["E"]["population"][1]
    with pytest.raises(ValueError):
        rank_population(population, [({"slot": 0}, {"valid": False, "U": None})])


def test_full_coordinator_preserves_64_slots_and_freezes_before_fresh_comparison(tmp_path, monkeypatch):
    import importlib
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    runner = importlib.import_module("maniguard_part3_search_run")
    run = tmp_path / "run"
    (run / "part2").mkdir(parents=True)
    (run / "part3").mkdir()
    model = tmp_path / "model"
    model.mkdir()
    (model / "forecast-audit.json").write_text(json.dumps({"gate": {"eligible": True}}))
    (model / "forecast-freeze.json").write_text("{}")
    (run / "part2/results.json").write_text(json.dumps({
        "records": [{"process": {"elapsed_seconds": 100}} for _ in range(16)]}))
    (run / "part2/protocol.json").write_text(json.dumps({"status": "frozen", "scenes": []}))
    calls = []

    def fake_launch(directory, job, protocol=None):
        if job["purpose"] == "validation":
            assert (run / "part3/search/finalists.json").exists()
        calls.append(job)
        return {"valid": True, "engineering_valid": job["purpose"] == "engineering", "U": 0}

    monkeypatch.setattr(runner, "RUN", run)
    monkeypatch.setattr(runner, "MODEL", model)
    monkeypatch.setattr(runner, "require_guard", lambda: None)
    monkeypatch.setattr(runner, "launch", fake_launch)
    monkeypatch.setattr(runner, "clock", lambda: {"physical_remaining": 100000})
    monkeypatch.setattr(runner, "reference", lambda: {"controller": {
        "free_step_limit": .04, "grasp_step_limit": .02,
        "correction_threshold": None, "minimum_interval": 4}})
    monkeypatch.setattr(runner, "implementation_files", list)
    monkeypatch.setattr(runner, "validate_scene_access", lambda *args: None)
    monkeypatch.setattr(runner, "sha", lambda path: "synthetic-hash")
    monkeypatch.setattr(runner, "validate_frozen", lambda path: json.loads(path.read_text()))
    monkeypatch.setattr(runner, "screen", lambda pool, *args, **kwargs: {
        "records": [{**p, "ranking": [0, 0, 0], "M_trusted": False, "C_trusted": False}
                    for p in pool], "scope": "synthetic fixture"})
    runner.main()
    assert len(calls) == 67
    assert sum(j["action_horizon"] == 32 for j in calls) == 3
    assert sum(j["action_horizon"] == 1100 for j in calls) == 64
    assert len({j["episode_id"] for j in calls}) == 67
    assert json.loads((run / "part3/search/terminal.json").read_text())["status"] == "completed"
