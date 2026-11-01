"""Synthetic analysis checks on rented hosts only, without final/model/simulator data."""

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_lewm_navigation_campaign_20261031_v2 import (
    ARMS,
    CONDITIONS,
    METRICS,
    bootstrap_summary,
    family_summary,
    plots,
    training_curve,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("RUNPOD_POD_ID") not in ("vukovc3t4o1dui", "wq4ar2qx2ff6r4"),
    reason="Numerical analysis tests belong only on the declared current rented hosts",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def synthetic_panel():
    cases = list(range(5000, 5016))
    profiles = []
    for repeat in range(3):
        rows = []
        for arm in ARMS:
            for case in cases:
                success = int(arm in ("cma", "random") and case < 5008)
                rows.append(
                    {
                        "arm": arm,
                        "source_episode": case,
                        "safe_completion": success,
                        "violation": 0,
                        "unfinished": 1 - success,
                        "engaged": 1,
                    }
                )
        host = {
            "host": "pusht" if repeat == 1 else "navigation",
            "pod_id": "push-host" if repeat == 1 else "nav-host",
        }
        profiles.append({"repeat_id": repeat, "host": host, "rows": rows})
    rng = np.random.Generator(np.random.PCG64(803101))
    indices = np.concatenate(
        [
            rng.choice(
                [i for i, case in enumerate(cases) if case % 4 == condition],
                (10000, 4),
                replace=True,
            )
            for condition in range(len(CONDITIONS))
        ],
        axis=1,
    )
    return cases, profiles, indices


def test_equal_paired_methods_have_exact_zero_estimate_and_interval():
    cases, profiles, indices = synthetic_panel()
    summary, _ = family_summary(np, "primary50", profiles, indices, cases)
    contrast = summary["paired_CMA_minus_baseline_SC"]["random"]
    assert contrast["estimate"] == 0.0
    assert contrast["case_cluster_percentile95"] == [0.0, 0.0]
    assert contrast["repeat_differences"] == [0.0, 0.0, 0.0]
    assert all(
        value["estimate"] == 0 and value["case_cluster_percentile95"] == [0, 0]
        for value in contrast["per_host"].values()
    )


def test_duplicate_repeat_rows_preserve_sixteen_clusters_and_pairing():
    cases, profiles, indices = synthetic_panel()
    summary, matrix = family_summary(np, "primary50", profiles, indices, cases)
    assert len(profiles) * len(cases) == summary["raw_episode_observations_each_arm"] == 48
    assert summary["source_case_clusters"] == 16
    assert summary["physical_host_blocks"]["nav-host"]["raw_episode_denominator_each_arm"] == 32
    assert summary["physical_host_blocks"]["nav-host"]["source_clusters"] == 16
    expected = bootstrap_summary(np, matrix["safe_completion"]["cma"][0], indices)
    assert summary["marginal_fixed_controller_rates"]["safe_completion"]["cma"] == expected
    assert all(row["episodes"] == row["source_clusters"] == 16 for row in summary["repeat_counts"])
    assert all(
        row["zero_violation_rule_of_three_upper_approx"] == 3 / 16
        for row in summary["repeat_counts"]
    )


def test_pg_curve_charges_sample_and_evaluation_then_stops_at_used_budget(tmp_path):
    protocol = {
        "method": "reinforce",
        "training_panel": [{"case_id": "a"}, {"case_id": "b"}],
        "max_environment_actions": 20,
        "learning_curve_reporting": {"action_landmarks": [0, 5, 10, 15, 20]},
    }
    save(tmp_path / "protocol.json", protocol)
    run = tmp_path / "run"
    order = ["anchor-a", "anchor-b", "sample-a", "sample-b", "evaluate-a", "evaluate-b"]
    for index, name in enumerate(order):
        actions = 2 if name.startswith("sample") else 1
        success = index == 4
        folder = run / "jobs" / name
        inner = folder / "episode" / "result.json"
        save(inner, {"episode_seconds": 0.25, "planning_seconds": 0.2})
        row = {
            "safe_completion": success,
            "violation": False,
            "unfinished": not success,
            "progress": 1.0 if success else 0.0,
            "environment_actions": actions,
            "resources": {"environment_actions": actions},
            "extra": {"episode_result": str(inner), "episode_result_sha256": digest(inner)},
        }
        save(folder / "result.json", row)
    anchor = {
        "job_ids": order[:2],
        "fitness": [0, 0, 0.0],
        "provenance": {"kind": "initial_anchor"},
    }
    nominee = {
        "job_ids": order[-2:],
        "fitness": [1, 0, 1.0],
        "provenance": {"kind": "trained_candidate", "completed_updates_before_evaluation": 1},
    }
    save(
        run / "state.json",
        {
            "phase": "complete",
            "active": None,
            "unfinalized_job_ids": [],
            "action_accounting_exact": True,
            "identity": "synthetic",
            "completed_job_ids": order,
            "anchor": anchor,
            "optimizer_updates": 1,
            "actual_environment_actions": 8,
        },
    )
    save(run / "source-freeze.json", {"protocol": protocol, "identity": "synthetic"})
    save(run / "update-000.json", {"nominee": nominee, "update": 0, "sample_job_ids": order[2:4]})
    entry = {
        "family": "primary50",
        "repeat_id": 0,
        "method": "reinforce",
        "protocol": {"path": "protocol.json", "sha256": digest(tmp_path / "protocol.json")},
        "output_namespace": "run",
    }
    curve = training_curve(tmp_path, entry)
    assert curve["actual_environment_actions"] == 8
    assert curve["residual_unspent_allowance"] == 12
    assert [point["actual_actions"] for point in curve["curve"]] == [2, 8]
    assert curve["curve"][-1]["incumbent_SC_rate"] == 0.5
    assert [
        point["actual_action_landmark"]
        for point in curve["landmarks"]
        if point["incumbent"] is not None
    ] == [5]
    assert all(
        point["incumbent"] is None
        for point in curve["landmarks"]
        if point["actual_action_landmark"] > 8
    )
    assert curve["resource_totals_by_field"]["environment_actions"] == 8


def test_publication_figures_export_png_and_svg_without_final_data(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases, profiles, indices = synthetic_panel()
    summary, matrix = family_summary(np, "primary50", profiles, indices, cases)
    assert set(matrix) == set(METRICS)
    plots(plt, np, tmp_path, "synthetic-primary50", summary, matrix, profiles, [])
    expected = {
        "synthetic-primary50-outcomes.png",
        "synthetic-primary50-outcomes.svg",
        "synthetic-primary50-paired-SC-differences.png",
        "synthetic-primary50-paired-SC-differences.svg",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected
    assert all((tmp_path / name).stat().st_size > 1000 for name in expected)
