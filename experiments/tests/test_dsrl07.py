"""Numerical diagnosis distinguishes correctness, influence and physical outcomes."""

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from helpers.dsrl06_protocol import frozen_jobs as previous_jobs
from helpers.dsrl07_infra import FinancialBook
from helpers.dsrl07_protocol import (
    array_difference,
    candidate_summary,
    decoder_comparison,
    frozen_jobs,
)


def test_same_process_check_does_not_reinterpret_cross_process_difference():
    native = np.zeros((16, 8), dtype=np.float32)
    native[:, 7] = 1
    another_process = native.copy()
    another_process[:, 0] += 0.003
    assert decoder_comparison(native, native)["pass"]
    cross = decoder_comparison(native, another_process)
    assert not cross["pass"] and cross["gripper_prefix_equal"]


def test_threshold_crossing_is_rejected_even_if_numeric_tolerance_passes():
    native = np.zeros((16, 8), dtype=np.float32)
    native[:, 7] = 0.010001
    replay = native.copy()
    replay[:, 7] = 0.009999
    assert array_difference(native, replay, atol=1e-5)["pass"]
    assert not decoder_comparison(native, replay)["pass"]


def test_nonfinite_values_and_shape_changes_fail():
    assert not array_difference([np.nan], [np.nan], atol=1e-6)["pass"]
    assert not array_difference([1], [[1]], atol=1e-6)["pass"]


def test_arm_influence_does_not_become_a_gripper_or_completion_claim():
    values = np.zeros((14, 16, 8), np.float32)
    values[:, :, 7] = 1
    values[1, :, 0] = 0.2
    result = candidate_summary(values)
    assert result["arm_influence"] and not result["gripper_sensitive"]
    assert "safe_completion" not in result


def test_candidate_sensitivity_uses_executed_prefix_and_declared_margins():
    values = np.zeros((14, 16, 8), np.float32)
    values[:, :, 7] = 1
    values[1, 9, 7] = -1
    assert not candidate_summary(values)["gripper_sensitive"]
    values[1, 6, 7] = -0.1
    assert candidate_summary(values)["sensitive_gripper_positions"] == [6]


def test_new_schedule_is_disjoint_and_keeps_comparison_structure():
    old, new = previous_jobs(), frozen_jobs()
    for group in old:
        assert len(old[group]) == len(new[group])
        assert {j["seed"] for j in old[group]}.isdisjoint(j["seed"] for j in new[group])
        for before, after in zip(old[group], new[group], strict=True):
            assert before["task"] == after["task"]
            assert before["actor_rng_seed"] != after["actor_rng_seed"]
    assert {j["task"] for j in new["training_jobs"]} == {0, 3, 6, 9, 15, 17}


def test_followup_budget_imports_spending_once_and_preserves_cap(tmp_path):
    path = tmp_path / "finance.json"
    book = FinancialBook(path)
    before = book.snapshot()
    assert before["active_estimate_usd"] == pytest.approx(2.9268200771)
    assert before["other_reserved_or_actual_usd"] == 1.8
    assert not before["active_instances"]
    again = FinancialBook(path)
    assert len(again.data["allocations"]) == 1
    assert again.limits()["total_usd"] == 15
    assert again.limits()["aggregate_hours"] == 8


def test_native_draw_passthrough_reconstructs_key_without_advancing_service_rng(tmp_path):
    sys.path.insert(0, str(Path(__file__).parent))
    import test_dsrl06_probe as shared

    ledger, policy, server = shared.setup(tmp_path)
    probe = shared.runner(
        tmp_path,
        ledger,
        server.infer,
        freeze={"kind": "authority", "query_ceiling": 2, "test": "native-draw-passthrough"},
    )
    try:
        key = np.array([29, 0], np.uint32)
        native, _ = probe.query(0, shared.observation(), mode="native_replay", rng_before=key)
        explicit, _ = probe.query(
            1,
            shared.observation(),
            mode="explicit",
            noise=native["noise"],
            rng_before=key,
            native_draw_replay=True,
        )
        assert np.array_equal(native["actions"], explicit["actions"])
        assert np.array_equal(explicit["rng_before"], key)
        assert policy._rng == 0 and policy.calls == 2
    finally:
        probe.close()


def test_task7_preservation_includes_reused_source(tmp_path):
    path = Path(__file__).parents[1] / "scripts/preserve_dsrl07.py"
    spec = importlib.util.spec_from_file_location("preserve07_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = Path(__file__).resolve().parents[2]
    selected = {
        str(src.relative_to(root))
        for src, _ in module.local_sources(tmp_path / "empty", tmp_path / "stage", root)
    }
    assert "experiments/helpers/dsrl06_learner.py" in selected
    assert "experiments/scripts/diagnose_dsrl07.py" in selected
    assert "progress/07-dsrl-repair.md" in selected
    assert module.PUBLISHED_MANIFEST.startswith("receipts/dsrl07/")


def test_task7_freeze_rejects_old_jobs_even_with_new_label(tmp_path):
    scripts = Path(__file__).parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    import run_dsrl07

    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "authorization": {"user_launch": True},
                "source_sha256": {},
                "jobs": previous_jobs(),
                "caps": dict(run_dsrl07.CAPS),
                "money": {"total_usd": 15},
            }
        )
    )
    with pytest.raises(ValueError, match="Task-7 frozen jobs"):
        run_dsrl07.verify_plan(tmp_path, path, "diagnostics")
