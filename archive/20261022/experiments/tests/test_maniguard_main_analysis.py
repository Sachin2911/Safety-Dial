"""Synthetic main-evaluation panels; no result files, policies or physics."""

import hashlib
import json
import sys
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.maniguard_main_analysis import (
    REQUIRED_COUNT_COSTS,
    ExpectedCase,
    analyze,
    analyze_protocol,
)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def panel(*, families=("jar", "stack"), scenes=("one", "two"), variants=("base",),
          methods=None, outcome=None):
    """Make a complete crossed panel with one physical identity reference run."""
    methods = methods or {"identity": ("fixed", ("fixed",)), "es": ("trained", ("r0", "r1"))}
    expected, rows = [], []
    for method, (kind, runs) in methods.items():
        for run in runs:
            for family in families:
                for scene in scenes:
                    for variant in variants:
                        case = ExpectedCase(
                            episode_id=f"{method}/{run}/{family}/{scene}/{variant}",
                            method=method, run_id=run, run_kind=kind,
                            actor_sha256=digest(method + run) if kind == "trained" else None,
                            controller_sha256=digest(method + run + "controller"),
                            family=family, base_scene=scene, scene=f"{scene}/{variant}",
                            condition=variant, seed=7, sampling_seed=11)
                        expected.append(case)
                        completion, violation = outcome(case) if outcome else (0, 0)
                        safe = completion * (1 - violation)
                        rows.append({**asdict(case), "valid": True, "invalid_reasons": [],
                                     "output": "/synthetic/" + case.episode_id,
                                     "full_episode": True, "short_check": False,
                                     "outcome": {"valid": True, "C_req": completion,
                                                 "S_req": safe, "V": violation,
                                                 "U": safe - 2 * violation},
                                     "costs": {"policy_queries": 3, "process_elapsed_seconds": 5.0}})
    return expected, rows


def report(expected, rows, **kwargs):
    return analyze(expected, rows, reference_method="identity", bootstrap_replicates=200, **kwargs)


def test_known_scene_balanced_estimand_and_paired_differences():
    expected, rows = panel(outcome=lambda c: (
        int(c.method == "es" and (c.family == "stack" or c.base_scene == "two")), 0))
    # Extra variants of the unsuccessful Jar scene must not outweigh the other base scene.
    for case in list(expected):
        if case.family == "jar" and case.base_scene == "one":
            source = next(row for row in rows if row["episode_id"] == case.episode_id)
            for variant in ("shift1", "shift2"):
                extra = replace(case, episode_id=case.episode_id + variant,
                                scene="one/" + variant, condition=variant)
                expected.append(extra)
                rows.append({**deepcopy(source), **asdict(extra),
                             "output": "/synthetic/" + extra.episode_id})
    result = report(expected, rows)
    es = result["estimates"]["es"]
    assert es["families"]["jar"] == {"S_req": 0.5, "C_req": 0.5, "V": 0, "U": 0.5}
    assert es["families"]["stack"]["S_req"] == 1
    assert es["macro"]["S_req"] == 0.75
    assert result["comparisons"]["es"]["by_run"]["r0"]["by_scene"]["jar"] == {
        "one": {"S_req": 0, "C_req": 0, "V": 0, "U": 0},
        "two": {"S_req": 1, "C_req": 1, "V": 0, "U": 1}}


def test_reference_records_and_costs_are_counted_once_across_training_runs():
    expected, rows = panel(methods={"identity": ("fixed", ("fixed",)),
                                    "es": ("trained", ("r0", "r1", "r2")),
                                    "ppo": ("trained", ("p0", "p1"))})
    result = report(expected, rows)
    assert result["coverage"]["observed"] == 24
    assert result["coverage"]["recorded_cost_totals"]["policy_queries"] == 72
    assert result["estimates"]["identity"]["unique_episode_count"] == 4
    assert len(result["records"]) == len(rows)
    assert len(result["comparisons"]["es"]["by_run"]) == 3


def test_missing_entire_family_and_failed_episode_withhold_analysis_and_keep_costs():
    expected, rows = panel()
    rows = [row for row in rows if row["family"] != "stack"]
    rows[0].update(valid=False, invalid_reasons=["monitor unavailable"])
    rows[0]["costs"]["process_elapsed_seconds"] = 19
    result = report(expected, rows)
    assert result["complete"] is False
    assert result["estimates"] is None and result["comparisons"] is None
    assert result["coverage"]["missing"] == 6
    assert result["coverage"]["invalid"] == 1
    assert result["coverage"]["recorded_cost_totals"]["process_elapsed_seconds"] == 44
    assert result["coverage_by_method_family"]["es"]["stack"]["expected"] == 4
    failed = next(r for r in result["records"] if r["status"] == "invalid")
    assert "monitor unavailable" in failed["invalid_reasons"]
    assert failed["metrics"] is None


@pytest.mark.parametrize("damage", ["actor", "controller", "short", "utility", "nonfinite",
                                   "nonbinary", "seed", "cost"])
def test_invalid_row_contracts_remain_visible_without_performance_metrics(damage):
    expected, rows = panel()
    row = next(row for row in rows if row["method"] == "es")
    if damage == "actor":
        row["actor_sha256"] = digest("different actor")
    elif damage == "controller":
        row.pop("controller_sha256")
    elif damage == "short":
        row.update(short_check=True, full_episode=False)
    elif damage == "utility":
        row["outcome"]["U"] = 1
    elif damage == "nonfinite":
        row["outcome"]["V"] = float("nan")
    elif damage == "nonbinary":
        row["outcome"]["C_req"] = 0.5
    elif damage == "seed":
        row["seed"] = 99
    elif damage == "cost":
        row["costs"]["policy_queries"] = float("inf")
    result = report(expected, rows)
    assert result["coverage"]["invalid_episode_ids"] == [row["episode_id"]]
    assert result["comparisons"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("damage", ["duplicate_row", "extra_row", "reused_path",
                                   "duplicate_expected", "different_panel", "switched_actor",
                                   "copied_reference", "relabelled_cluster"])
def test_grid_and_physical_record_ambiguity_is_rejected(damage):
    expected, rows = panel()
    if damage == "duplicate_row":
        rows.append(deepcopy(rows[0]))
    elif damage == "extra_row":
        rows.append({**rows[0], "episode_id": "undeclared"})
    elif damage == "reused_path":
        rows[1]["output"] = rows[0]["output"]
    elif damage == "duplicate_expected":
        expected.append(expected[0])
    elif damage == "different_panel":
        expected.pop()
    elif damage == "switched_actor":
        index = next(i for i, case in enumerate(expected) if case.method == "es")
        expected[index] = replace(expected[index], actor_sha256=digest("switched"))
    elif damage == "copied_reference":
        expected.extend(replace(case, episode_id=case.episode_id + "copy", run_id="copy")
                        for case in list(expected) if case.method == "identity")
    elif damage == "relabelled_cluster":
        for case in list(expected):
            expected.append(replace(case, episode_id=case.episode_id + "copy", base_scene="fake"))
    with pytest.raises(ValueError):
        report(expected, rows)


def test_training_runs_remain_crossed_with_families_during_resampling():
    expected, rows = panel(families=("jar", "stack", "clutter", "lid"),
                           outcome=lambda case: (int(case.method == "es" and case.run_id == "r1"), 0))
    result = report(expected, rows, seed=432)
    estimate = result["estimates"]["es"]
    assert estimate["macro"]["S_req"] == 0.5
    assert estimate["intervals"]["macro"]["S_req"]["percentile_interval"] == [0, 1]
    # Identical family outcomes cannot manufacture four independent draws of one actor.
    assert estimate["intervals"]["macro"]["S_req"] == estimate["intervals"]["jar"]["S_req"]
    assert set(estimate["by_run"]) == {"r0", "r1"}


def test_shared_scene_draws_cancel_common_case_difficulty_in_paired_differences():
    expected, rows = panel(methods={"identity": ("fixed", ("fixed",)),
                                    "other": ("fixed", ("fixed",))},
                           outcome=lambda case: (int(case.base_scene == "one"), 0))
    result = report(expected, rows)
    assert result["estimates"]["other"]["intervals"]["macro"]["C_req"][
        "resampling_standard_deviation"] > 0
    paired = result["comparisons"]["other"]
    assert paired["macro"]["C_req"] == 0
    assert paired["intervals"]["macro"]["C_req"] == {
        "percentile_interval": None, "unavailable_reason": "no_empirical_resampling_variation",
        "resampling_standard_deviation": 0}


def test_repeating_correlated_variants_does_not_inflate_scene_information():
    outcome = lambda case: (int(case.base_scene == "one"), int(case.base_scene == "two"))
    first = report(*panel(outcome=outcome), seed=71)
    repeated = report(*panel(outcome=outcome, variants=("base", "copy1", "copy2")), seed=71)
    assert first["estimates"]["es"]["intervals"] == repeated["estimates"]["es"]["intervals"]
    assert repeated["estimates"]["es"]["base_scene_counts"] == {"jar": 2, "stack": 2}
    assert repeated["coverage"]["expected"] == 3 * first["coverage"]["expected"]


def test_zero_events_have_counts_but_no_zero_width_risk_bound():
    result = report(*panel(outcome=lambda case: (1, 0)))
    for estimate in result["estimates"].values():
        assert estimate["observed_counts"]["V"] == 0
        assert estimate["unique_episode_count"] > 0
        assert estimate["observed_counts_by_family"]["jar"]["V"] == 0
        assert estimate["unique_episode_counts_by_family"]["jar"] > 0
        assert estimate["intervals"]["macro"]["V"]["percentile_interval"] is None
        assert "boundary" in estimate["intervals"]["macro"]["V"]["unavailable_reason"]
    assert result["comparisons"]["es"]["intervals"]["macro"]["V"]["percentile_interval"] is None
    assert result["risk_inference"]["status"] == "not_assessed"
    assert result["risk_inference"]["margin"] is None


def test_violations_in_one_method_do_not_make_zero_event_reference_risk_resolved():
    result = report(*panel(outcome=lambda case: (
        1, int(case.method == "es" and case.base_scene == "two"))))
    assert result["comparisons"]["es"]["macro"]["V"] == 0.5
    assert result["comparisons"]["es"]["macro"]["U"] == -1.5
    assert result["comparisons"]["es"]["intervals"]["macro"]["V"]["percentile_interval"] is None
    assert result["estimates"]["es"]["intervals"]["macro"]["V"]["percentile_interval"] is not None


@pytest.mark.parametrize("limitation", ["one_scene", "one_run", "disabled"])
def test_insufficient_independent_units_keep_points_but_withhold_intervals(limitation):
    kwargs = {"outcome": lambda case: (int(case.base_scene == "one"), 0)}
    if limitation == "one_scene":
        kwargs["scenes"] = ("one",)
    if limitation == "one_run":
        kwargs["methods"] = {"identity": ("fixed", ("fixed",)), "es": ("trained", ("only",))}
    expected, rows = panel(**kwargs)
    result = analyze(expected, rows, reference_method="identity",
                     bootstrap_replicates=0 if limitation == "disabled" else 200)
    assert result["complete"] is True
    interval = result["estimates"]["es"]["intervals"]["macro"]["C_req"]
    assert interval["percentile_interval"] is None
    assert interval["unavailable_reason"] == {
        "one_scene": "fewer_than_two_base_scenes",
        "one_run": "fewer_than_two_independent_training_runs",
        "disabled": "resampling_disabled"}[limitation]


def test_input_order_and_dictionary_conversion_preserve_reproducible_results():
    expected, rows = panel(outcome=lambda case: (int(case.base_scene == "one"), 0))
    original_expected, original_rows = deepcopy(expected), deepcopy(rows)
    first = report(expected, rows, seed=7)
    second = report([asdict(case) for case in reversed(expected)], reversed(rows), seed=7)
    assert first == second
    assert expected == original_expected and rows == original_rows
    json.dumps(first, allow_nan=False)


@pytest.mark.parametrize("kwargs", [{"bootstrap_replicates": True}, {"bootstrap_replicates": 1},
                                   {"bootstrap_replicates": -1}, {"seed": -1},
                                   {"interval_level": float("nan")}, {"interval_level": True}])
def test_invalid_analysis_controls_are_rejected(kwargs):
    with pytest.raises(ValueError):
        analyze(*panel(), reference_method="identity", **kwargs)


def test_empty_expected_grid_cannot_be_inferred_from_observed_rows():
    with pytest.raises(ValueError, match="nonempty expected"):
        report([], [])


def test_planner_mapping_runtime_extras_are_projected_onto_the_analysis_grid():
    expected, rows = panel()
    mappings = [{**asdict(case), "action_horizon": 2048, "checkpoint_sha256": digest("checkpoint"),
                 "preprocessing_sha256": digest("preprocessing")} for case in expected]
    assert report(mappings, rows) == report(expected, rows)


def test_reserved_summary_name_cannot_overwrite_a_family_interval():
    with pytest.raises(ValueError, match="reserved"):
        report(*panel(families=("macro",)))



@pytest.fixture
def frozen_panel(tmp_path):
    """Use the real planner/freezer over synthetic source and review artifacts."""
    from test_maniguard_main_protocol import freeze_fixture, planned

    import helpers.maniguard_main_analysis as implementation
    from helpers.maniguard_main_protocol import expected_cases, source_reference

    fixture = planned.__wrapped__(tmp_path)
    source = fixture["root"] / "analysis.py"
    source.write_bytes(Path(implementation.__file__).read_bytes())
    reader = fixture["root"] / "synthetic_reader.py"
    reader.write_text("# Synthetic audit row producer; no physics or outcomes.\n")
    fixture["spec"]["implementation_files"] = [
        source_reference(source, root=fixture["root"]),
        source_reference(reader, root=fixture["root"]),
    ]
    draft, _, protocol = freeze_fixture(fixture)
    jobs = expected_cases(protocol)
    rows = []
    for job in jobs:
        costs = dict.fromkeys(REQUIRED_COUNT_COSTS, 0)
        costs.update(policy_control_steps=4, policy_queries=1, physics_steps=24,
                     elapsed_seconds=5.0)
        rows.append({**deepcopy(job), "protocol_sha256": protocol["sha256"],
                     "audit_reader_sha256": fixture["spec"]["implementation_files"][1]["sha256"],
                     "output": "/synthetic/" + job["episode_id"], "valid": True,
                     "invalid_reasons": [], "full_episode": True, "short_check": False,
                     "outcome": {"valid": True, "C_req": 0, "S_req": 0, "V": 0, "U": 0},
                     "costs": costs, "cost_issues": [], "missing_cost_fields": []})
    return fixture, draft, protocol, jobs, rows


def protocol_report(fixture, protocol, rows):
    return analyze_protocol(protocol, rows, config_dir=fixture["config"], root=fixture["root"])


def test_frozen_wrapper_uses_exact_grid_settings_exposure_and_unique_baselines(frozen_panel):
    fixture, _, protocol, jobs, rows = frozen_panel
    original = deepcopy(rows)
    result = protocol_report(fixture, protocol, rows)
    assert result["complete"] is True
    assert result["coverage"]["expected"] == len(jobs) == 36
    assert result["resampling"]["replicates"] == 20
    assert result["resampling"]["seed"] == 11
    assert result["resampling"]["percentile_level"] == protocol["spec"]["analysis"]["confidence_level"]
    assert result["protocol_binding"]["protocol_sha256"] == protocol["sha256"]
    assert result["protocol_binding"]["grid_sha256"] == protocol["grid_sha256"]
    assert result["protocol_binding"]["analysis"] == protocol["spec"]["analysis"]
    assert result["protocol_binding"]["exposure"] == protocol["exposure"]
    for record in result["records"]:
        assert record["case"] == next(job for job in jobs if job["episode_id"] == record["case"]["episode_id"])
    assert result["protocol_binding"]["exposure"]["lid"]["role"] == "design_exposed_transfer"
    assert result["protocol_binding"]["exposure"]["dusty"]["role"] == "reserved_transfer"
    assert result["estimates"]["identity"]["unique_episode_count"] == 12
    assert result["coverage"]["recorded_cost_totals"]["policy_queries"] == 36
    assert result["risk_inference"]["status"] == "not_assessed"
    assert result["risk_inference"]["prespecified_margin"] == 0.02
    assert result["decision_scope"]["safety_margin_rule_evaluated"] is False
    assert rows == original


@pytest.mark.parametrize("omission", ["family", "run"])
def test_frozen_wrapper_cannot_shrink_the_complete_declared_grid(frozen_panel, omission):
    fixture, _, protocol, _, rows = frozen_panel
    key, value = ("family", "dusty") if omission == "family" else ("run_id", "evolution-s29")
    kept = [row for row in rows if row[key] != value]
    result = protocol_report(fixture, protocol, kept)
    assert result["complete"] is False
    assert result["coverage"]["expected"] == 36
    assert result["coverage"]["missing"] == len(rows) - len(kept)
    assert result["estimates"] is None and result["comparisons"] is None


@pytest.mark.parametrize("field,damage", [
    ("action_horizon", 1), ("stochastic", True), ("preprocessing_sha256", "f" * 64),
    ("checkpoint", {}), ("exposure_role", "design_exposed_transfer"),
    ("analysis_sha256", "e" * 64), ("deployment_sha256", "d" * 64),
    ("protocol_sha256", "c" * 64), ("audit_reader_sha256", "b" * 64),
    ("sampling_seed", True), ("purpose", None), ("output", None),
])
def test_frozen_wrapper_rejects_missing_or_changed_runtime_bindings(frozen_panel, field, damage):
    fixture, _, protocol, _, rows = frozen_panel
    if damage is None:
        rows[0].pop(field)
    else:
        rows[0][field] = damage
    result = protocol_report(fixture, protocol, rows)
    assert result["complete"] is False
    assert result["coverage"]["invalid"] == 1
    assert result["estimates"] is None
    assert result["coverage"]["recorded_cost_totals"]["policy_queries"] == 36


@pytest.mark.parametrize("field", REQUIRED_COUNT_COSTS + ("elapsed_seconds",))
def test_frozen_wrapper_requires_explicit_cost_counters_including_zeros(frozen_panel, field):
    fixture, _, protocol, _, rows = frozen_panel
    rows[0]["costs"].pop(field)
    result = protocol_report(fixture, protocol, rows)
    assert result["complete"] is False
    assert result["coverage"]["invalid"] == 1
    assert result["coverage"]["cost_record_counts"][field] == 35


@pytest.mark.parametrize("damage", ["missing_list", "unknown_counts", "reported_issue", "boolean_count"])
def test_frozen_wrapper_rejects_unknown_accounting_and_retains_known_work(frozen_panel, damage):
    fixture, _, protocol, _, rows = frozen_panel
    if damage == "missing_list":
        rows[0].pop("cost_issues")
    elif damage == "unknown_counts":
        rows[0]["missing_cost_fields"] = ["unresolved_failed_encoder_calls"]
    elif damage == "reported_issue":
        rows[0]["cost_issues"] = ["Failed work remains a lower bound"]
    else:
        rows[0]["costs"]["encoder_load_attempts"] = False
    result = protocol_report(fixture, protocol, rows)
    assert result["complete"] is False
    assert result["coverage"]["invalid"] == 1
    assert result["coverage"]["recorded_cost_totals"]["policy_queries"] == 36


def test_frozen_wrapper_rejects_drafts_and_frozen_source_tampering(frozen_panel):
    fixture, draft, protocol, _, rows = frozen_panel
    with pytest.raises(ValueError, match="explicit costed freeze"):
        protocol_report(fixture, draft, rows)
    (fixture["root"] / "analysis.py").write_text("# Modified after freeze\n")
    with pytest.raises(ValueError, match="Bound source changed"):
        protocol_report(fixture, protocol, rows)


def test_frozen_wrapper_verifies_the_running_analysis_source(frozen_panel, monkeypatch, tmp_path):
    import helpers.maniguard_main_analysis as implementation

    fixture, _, protocol, _, rows = frozen_panel
    changed = tmp_path / "different_analysis.py"
    changed.write_text("# Different running implementation\n")
    monkeypatch.setattr(implementation, "__file__", str(changed))
    with pytest.raises(ValueError, match="running analysis implementation"):
        protocol_report(fixture, protocol, rows)
