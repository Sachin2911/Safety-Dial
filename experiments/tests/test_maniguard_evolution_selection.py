"""Selection correctness, information separation and restart safety, with no robot calls."""

from copy import deepcopy
import datetime as dt
import json
from pathlib import Path
import warnings

import cma
import numpy as np
import pytest

from build_maniguard_v3_config import build
from analyze_maniguard_evolution import expected_cases
from maniguard_evolution import Search
from maniguard_evolution_episode import BudgetExhausted, ManiGuardEvaluator, atomic_json, totals
from maniguard_evolution_selection import (
    choose_selection, nominate_elite, selection_schedule,
)
from run_maniguard_evolution import on_rented_host, source_manifest, validate_config


ROOT = Path(__file__).resolve().parents[2]


def config():
    return build(ROOT / "configs/maniguard_evolution/development-v3-protocol.json",
                 ROOT / "progress/receipts/20261030-vast-maniguard-v3-preparation.json",
                 "/workspace/maniguard-evolution-v3-20261030", "software-check-only")


def row(success=False, progress=0, contact=False, violation=False):
    return {"valid": True, "strict_safe_completion": success,
            "benchmark_safe_completion": success, "requested_completion": success,
            "raw_violation": violation, "counted_violation": violation,
            "unfinished": not success, "engaged": contact or success,
            "safe_stage_progress": progress, "safe_target_contact": contact}


def test_partial_progress_and_safety_alone_do_not_displace_native():
    rows = {"native": [row(True), row(violation=True)],
            "elite": [row(True, 3, True), row(progress=3, contact=True)],
            "mean": [row(True, 3, True), row(progress=3, contact=True)]}
    result = choose_selection(rows)
    assert result["winner"] == "native" and result["eligible_adapters"] == []


def test_only_strictly_better_completion_is_eligible_and_elite_wins_exact_ties():
    rows = {"native": [row(), row()],
            "elite": [row(True), row()], "mean": [row(True), row()]}
    assert choose_selection(rows)["winner"] == "elite"
    rows["mean"][1] = row(True)
    assert choose_selection(rows)["winner"] == "mean"
    rows["elite"][1] = {"valid": False}
    with pytest.raises(ValueError, match="whole valid panel"):
        choose_selection(rows)


def test_nomination_excludes_anchor_and_breaks_ties_by_earliest_generation_then_index():
    generations = [
        {"generation": 0, "scores": [100, 1, 1], "parameters": [[0], [1], [2]]},
        {"generation": 1, "scores": [1, 1], "parameters": [[3], [4]]}]
    result = nominate_elite(generations, [0])
    assert result["parameters"] == [1] and result["generation"] == 0
    assert result["candidate_index"] == 1


def test_selection_uses_task_major_cases_and_rotates_all_three_candidates():
    actual = selection_schedule({"tasks": [6, 9], "policy_seeds": [201, 202]})
    assert actual == [(6, 201, ["native", "elite", "mean"]),
                      (6, 202, ["elite", "mean", "native"]),
                      (9, 201, ["mean", "native", "elite"]),
                      (9, 202, ["native", "elite", "mean"])]


class SyntheticEvaluator:
    def __init__(self, root, interrupt_at=None, fallback=False):
        self.root, self.interrupt_at, self.fallback = root, interrupt_at, fallback
        self.calls = []

    def __call__(self, label, task, seed, artifact):
        if len(self.calls) == self.interrupt_at:
            raise KeyboardInterrupt("Synthetic interruption, no simulator")
        self.calls.append((label, task, seed, artifact))
        if label.startswith("validation_"):
            assert (self.root / "selection.json").exists()
        if label == "selection_native":
            return row(task == 6 and seed == 303200)
        if label.startswith("selection_"):
            return row(task == 6 and (not self.fallback or seed == 303200))
        return row()


def test_full_108_case_schedule_selects_before_validation_and_keeps_anchor_distinct(tmp_path):
    cfg = config()
    backend = SyntheticEvaluator(tmp_path)
    search = Search(cfg, tmp_path, backend)
    with warnings.catch_warnings():
        warnings.simplefilter("error", cma.evolution_strategy.InjectionWarning)
        state = search.run()
    assert state["status"] == "completed" and len(backend.calls) == 108
    assert state["selected_kind"] == "elite" and search.es.countiter == 6
    assert len(search.es.sent_solutions) == 0
    assert state["selection"]["decision"]["summary"]["native"]["valid_episodes"] == 4
    assert all(len(g["anchor_indices"]) == 1 for g in state["generations"])
    assert len(list((tmp_path / "generation-snapshots").glob("g*/checkpoint.pkl"))) == 6
    validation = [(label, task, seed) for label, task, seed, _ in backend.calls
                  if label.startswith("validation_")]
    assert validation[:4] == [("validation_native", 15, 303300),
                             ("validation_selected", 15, 303300),
                             ("validation_selected", 15, 303301),
                             ("validation_native", 15, 303301)]
    np.testing.assert_array_equal(json.loads((tmp_path / "controller.json").read_text())[
        "parameters"], state["selected"])


@pytest.mark.parametrize("interrupt_at", [89, 97])
def test_resume_during_selection_or_validation_never_repeats_finished_work(tmp_path, interrupt_at):
    cfg = config()
    full = Search(cfg, tmp_path / "full", SyntheticEvaluator(tmp_path / "full")).run()
    first = SyntheticEvaluator(tmp_path / "resumed", interrupt_at=interrupt_at)
    with pytest.raises(KeyboardInterrupt):
        Search(cfg, tmp_path / "resumed", first).run()
    second = SyntheticEvaluator(tmp_path / "resumed")
    resumed = Search(cfg, tmp_path / "resumed", second, resume=True).run()
    assert len(first.calls) + len(second.calls) == 108
    assert len({(a, b, c) for a, b, c, _ in first.calls + second.calls}) == 108
    assert full["selected"] == resumed["selected"]
    assert full["selected_kind"] == resumed["selected_kind"] == "elite"


def test_native_fallback_is_exported_and_validated_as_disabled_native(tmp_path):
    backend = SyntheticEvaluator(tmp_path, fallback=True)
    state = Search(config(), tmp_path, backend).run()
    assert state["selected_kind"] == "native" and state["selected"] == state["initial"]
    artifact = json.loads((tmp_path / "controller.json").read_text())
    assert artifact["enabled"] is False
    assert all(not a[3]["enabled"] for a in backend.calls if a[0].startswith("validation_"))


def test_config_rejects_protocol_changes_and_freezes_new_helpers_and_protocol():
    cfg = config()
    panels = json.loads((ROOT / "progress/panels.json").read_text())
    assert validate_config(cfg, panels) == 205
    changed = deepcopy(cfg)
    changed["search"]["generations"] = 5
    changed["search"]["policy_seeds"].pop()
    with pytest.raises(ValueError, match="prospectively written"):
        validate_config(changed, panels)
    manifest = source_manifest(cfg)
    for name in ("maniguard_evolution_v3", "maniguard_evolution_selection",
                 "maniguard_evolution_diagnostics"):
        assert f"experiments/helpers/{name}.py" in manifest["sources"]
    assert cfg["protocol_binding"]["path"] in manifest["sources"]


def test_rented_host_verification_never_reads_a_process_environment(monkeypatch):
    monkeypatch.delenv("CONTAINER_ID", raising=False)
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    original = Path.read_bytes

    def guarded(path):
        if path.name == "environ":
            raise AssertionError("A process credential environment was accessed")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    assert on_rented_host()


def test_first_episode_deadline_rejects_launch_before_reserving_a_case(tmp_path, monkeypatch):
    cfg = config()
    atomic_json(tmp_path / "config.json", cfg)
    backend = ManiGuardEvaluator(tmp_path, cfg, tmp_path, ROOT)
    deadline = dt.datetime.fromisoformat(cfg["limits"]["deadline_utc"]).timestamp()
    monkeypatch.setattr("maniguard_evolution_episode.time.time", lambda: deadline - 25200 + 1)
    with pytest.raises(BudgetExhausted, match="Seven-hour"):
        backend.admission()
    assert not list((tmp_path / "episodes").glob("*/reservation.json"))


def test_preparation_attempt_journal_overrides_stale_query_summary(tmp_path):
    directory = tmp_path / "primer"
    (directory / "attempt-000").mkdir(parents=True)
    atomic_json(directory / "attempt-000/preseed-query.json",
                {"attempted": 1, "completed": 0, "seconds": 3.5})
    summary = tmp_path / "preparation.json"
    atomic_json(summary, {"attempt_directory": str(directory), "model_query_attempts": 0})
    atomic_json(tmp_path / "config.json", {"server_preparation_accounting": str(summary)})
    actual = totals(tmp_path)
    assert actual["model_query_attempts"] == 1 and actual["model_query_returns"] == 0
    assert actual["model_query_seconds"] == 3.5 and actual["reserved_starts"] == 0


def test_frozen_selection_tampering_blocks_resume_before_any_further_validation(tmp_path):
    first = SyntheticEvaluator(tmp_path, interrupt_at=97)
    with pytest.raises(KeyboardInterrupt):
        Search(config(), tmp_path, first).run()
    path = tmp_path / "selection.json"
    path.write_text(path.read_text() + " ")
    second = SyntheticEvaluator(tmp_path)
    with pytest.raises(ValueError, match="Frozen selection changed"):
        Search(config(), tmp_path, second, resume=True).run()
    assert second.calls == []


def test_final_audit_schedule_includes_all_selection_and_validation_cases():
    actual = expected_cases(config())
    assert len(actual) == 108
    assert sum(label.startswith("selection_") for label, _, _ in actual) == 12
    assert sum(label.startswith("validation_") for label, _, _ in actual) == 12
