"""S4 paired arm reports use frozen margins, physical pairing and defined estimands."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "scripts"))

from helpers.dialMetrics import cluster_bootstrap, fsa  # noqa: E402
from helpers.runManifest import file_sha256  # noqa: E402
from walker_s4_prospective import paired_acquisition_intervals, paired_difference  # noqa: E402


def paired_rows():
    random = [{"bank": "test", "root_id": f"r{i}", "root": i, "branch": i,
               "cmin_dense_health": truth, "cmin_imagined_health": 1.,
               "cmin_dense_speed": -truth, "cmin_imagined_speed": 1.}
              for i, truth in enumerate((-1., 1., -1., 1.))]
    boundary = deepcopy(random)
    for row in boundary:
        row["cmin_imagined_health"] = row["cmin_dense_health"]
        row["cmin_imagined_speed"] = -row["cmin_dense_speed"]
    return random, boundary


def test_pairing_reports_both_rate_counts_and_separate_rule_effects():
    random, boundary = paired_rows()
    health = paired_difference(random, boundary, "health", 0., 0., n_boot=200)
    speed = paired_difference(random, boundary, "speed", 0., 0., n_boot=200)
    assert health["point"] == -.5
    assert speed["point"] == .5
    assert health["defined"] and speed["defined"]
    assert health["reference"]["n_accepted"] == 4
    assert health["updated"]["n_accepted"] == 2
    assert health["updated"]["n_false_safe"] == 0
    assert health["n_source_episodes"] == 4
    assert health["n_boot"] + health["n_boot_undefined"] == 200


@pytest.mark.parametrize("change", ["branch", "root_id", "source", "bank", "truth", "censor", "order", "duplicate"])
def test_pairing_rejects_changed_physical_rows(change):
    random, boundary = paired_rows()
    if change == "branch":
        boundary[0]["branch"] += 100
    elif change == "root_id":
        boundary[0]["root_id"] = "different-root"
    elif change == "source":
        boundary[0]["root"] += 100
    elif change == "bank":
        boundary[0]["bank"] = "stress"
    elif change == "truth":
        boundary[0]["cmin_dense_health"] -= .01
    elif change == "censor":
        boundary[0]["censored"] = True
    elif change == "order":
        boundary.reverse()
    elif change == "duplicate":
        random.append(deepcopy(random[0]))
        boundary.append(deepcopy(boundary[0]))
    with pytest.raises(ValueError):
        paired_difference(random, boundary, "health", 0., 0., n_boot=10)


def test_bootstrap_clusters_all_roots_from_the_same_source_episode(monkeypatch):
    import walker_s4_prospective as runner
    random, boundary = paired_rows()
    for arm in (random, boundary):
        for row in arm:
            row["root"] //= 2
    seen = []
    original = runner.cluster_bootstrap
    def record(fn, root, **kwargs):
        seen.extend(root.tolist())
        return original(fn, root, **kwargs)
    monkeypatch.setattr(runner, "cluster_bootstrap", record)
    result = paired_difference(random, boundary, "health", 0., 0., n_boot=20)
    assert seen == [0, 0, 1, 1]
    assert result["cluster_unit"] == "source_episode"
    assert result["n_source_episodes"] == 2
    assert result["n_roots"] == 4


def test_zero_acceptance_is_undefined_and_never_a_directional_effect():
    random, boundary = paired_rows()
    for row in boundary:
        row["cmin_imagined_health"] = -1.
    result = paired_difference(random, boundary, "health", 0., 0., n_boot=30)
    assert not result["defined"]
    assert result["undefined_reason"] == "zero_acceptance"
    assert result["direction"] == "undefined"
    assert not result["interval_excludes_zero"]
    assert all(np.isnan(result[key]) for key in ("point", "lo", "hi"))
    assert result["updated"]["n_accepted"] == 0


def test_censored_point_cannot_inherit_a_finite_conditional_bootstrap_interval():
    random, boundary = paired_rows()
    random[-1]["censored"] = boundary[-1]["censored"] = True
    # Replicates that happen not to sample the unresolved episode are defined.
    raw = cluster_bootstrap(
        lambda a, b, u, c: fsa(b, u, 0., censored=c)["fsa"] - fsa(a, u, 0., censored=c)["fsa"],
        np.arange(4), n_boot=200, a=np.ones(4), b=np.array([-1., 1., -1., 1.]),
        u=np.array([True, False, True, False]), c=np.array([False, False, False, True]))
    assert np.isnan(raw["point"])
    assert np.isfinite(raw["lo"]) and np.isfinite(raw["hi"])
    result = paired_difference(random, boundary, "health", 0., 0., n_boot=200)
    assert 0 < result["n_boot"] < 200
    assert result["undefined_reason"] == "accepted_censored_futures"
    assert result["direction"] == "undefined"
    assert not result["interval_excludes_zero"]
    assert all(np.isnan(result[key]) for key in ("point", "lo", "hi"))
    assert result["reference"]["n_accepted_censored"] == 1


def test_known_violation_is_resolved_even_if_later_trajectory_is_censored():
    random, boundary = paired_rows()
    random[0]["censored"] = boundary[0]["censored"] = True
    result = paired_difference(random, boundary, "health", 0., 0., n_boot=20)
    assert result["defined"]
    assert result["point"] == -.5
    assert result["reference"]["n_accepted_censored"] == 0


def saved_report(tmp_path):
    report = {"planned_budgets": [2, 4], "acquisition_seeds": [0, 1],
              "active_rules": ["health", "speed"], "adaptation": {"random": {}, "boundary": {}}}
    rows_by_arm = dict(zip(("random", "boundary"), paired_rows(), strict=True))
    for arm in ("random", "boundary"):
        for seed in report["acquisition_seeds"]:
            curve = []
            report["adaptation"][arm][str(seed)] = curve
            for budget in report["planned_budgets"]:
                path = tmp_path / f"{arm}-s{seed}-b{budget}" / "evaluation_rows.json"
                path.parent.mkdir()
                path.write_text(json.dumps({bank: [{**row, "bank": bank} for row in rows_by_arm[arm]]
                                            for bank in ("dev", "test", "stress")}))
                curve.append({"budget": budget, "charged_steps": 100 + budget * 100,
                              "evaluation_rows": {"file": str(path.relative_to(tmp_path)), "sha256": file_sha256(path)},
                              "eval": {bank: {rule: {"margin": 0.} for rule in report["active_rules"]}
                                       for bank in ("dev", "test", "stress")}})
    return report


def test_offline_report_covers_all_seeds_budgets_banks_and_rules(tmp_path):
    report = saved_report(tmp_path)
    result = paired_acquisition_intervals(report, tmp_path, n_boot=20)
    assert result["contrast"] == "boundary minus random"
    for budget in (2, 4):
        for seed in (0, 1):
            entry = result["by_budget"][str(budget)]["by_seed"][str(seed)]
            assert entry["charged_steps"] == 100 + budget * 100
            for bank in ("test", "stress"):
                assert entry[bank]["health"]["point"] == -.5
                assert entry[bank]["speed"]["point"] == .5
                assert entry[bank]["health"]["reference"]["m"] == 0.


@pytest.mark.parametrize("change", ["margin", "cost", "hash", "budget"])
def test_saved_report_rejects_posthoc_margin_unequal_cost_or_modified_rows(tmp_path, change):
    report = saved_report(tmp_path)
    point = report["adaptation"]["boundary"]["0"][0]
    if change == "margin":
        point["eval"]["test"]["health"]["margin"] = .1
    elif change == "cost":
        point["charged_steps"] += 1
    elif change == "hash":
        path = tmp_path / point["evaluation_rows"]["file"]
        path.write_text(path.read_text() + "\n")
    elif change == "budget":
        report["adaptation"]["boundary"]["0"].pop()
    with pytest.raises(ValueError):
        paired_acquisition_intervals(report, tmp_path, n_boot=10)
