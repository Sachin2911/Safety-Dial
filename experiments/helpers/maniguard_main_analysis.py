"""Pure, declared-panel analysis for a future ManiGuard main evaluation.

This module never discovers or reads result files. Callers supply a frozen expected
grid and rows from an independently pinned audit reader. It checks the numerical and
identity contract, not the underlying simulator evidence. Incomplete panels retain
their records and costs but produce no performance estimates or comparisons.

The estimand weights families equally, base scenes equally within each family,
declared variants/seeds equally within each scene, and training runs equally. Every
run must have the same case panel. A fixed reference has one physical record per
case, reused in comparisons without duplicating observations or costs.

The optional percentile bootstrap is descriptive: it independently resamples base
scenes within each fixed family and training runs within each method. Scene draws
are shared across methods; each sampled run is retained across all families. All
variants and policy seeds of a scene stay together. This crossed resampling assumes
exchangeable independent base scenes and independent training runs, with the two
sources of variation independent. Equal numeric seeds across optimizers do not
establish paired training runs or identical physical starts. Families, checkpoints and the within-scene case panel are conditioned on, not
resampled. The observed selected actors represent training-run variability; neither
optimization nor validation selection is rerun inside the bootstrap.

Intervals do not establish a safety margin or calibrated coverage with few clusters.
At an observed violation-rate boundary, V intervals (including differences involving
that boundary) are withheld. Zero events are reported with denominators and an
inconclusive-risk statement, never a zero-width risk bound or an independent-episode
3/n calculation for clustered observations. A separate frozen risk analysis is needed.
"""

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from numbers import Real

import numpy as np

METRICS = ("S_req", "C_req", "V", "U")
SCHEMA = "maniguard-main-analysis-v1"
PROTOCOL_SCHEMA = "maniguard-main-protocol-analysis-v1"
REQUIRED_COUNT_COSTS = (
    "policy_control_steps", "policy_queries", "physics_steps", "encoder_image_calls",
    "encoder_images", "encoder_text_calls", "adapter_decisions", "actor_rpc_attempts",
    "encoder_image_attempts", "encoder_text_attempts", "encoder_load_attempts",
)


@dataclass(frozen=True)
class ExpectedCase:
    """One physical evaluation, not one reference copy per optimizer run.

    ``base_scene`` must come from the declared split metadata. ``controller_sha256``
    binds fixed/non-actor methods and may additionally bind trained deployment rules.
    The analyzer projects mapping inputs onto these fields. Extra planner/runtime
    bindings remain the responsibility of the independently pinned audit reader.
    """

    episode_id: str
    method: str
    run_id: str
    actor_sha256: str | None
    family: str
    base_scene: str
    scene: str
    condition: str = "base"
    seed: int = 0
    sampling_seed: int = 0
    controller_sha256: str | None = None
    run_kind: str = "trained"

    @property
    def panel_key(self):
        return (self.family, self.base_scene, self.scene, self.condition,
                self.seed, self.sampling_seed)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _hash(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _validate_grid(expected, reference_method):
    names = {field.name for field in fields(ExpectedCase)}
    cases = sorted((case if isinstance(case, ExpectedCase) else ExpectedCase(
        **{name: case[name] for name in names if name in case})
                    for case in expected), key=lambda case: case.episode_id)
    if not cases:
        raise ValueError("An explicit nonempty expected case grid is required")
    seen, panels, artifacts, kinds = set(), defaultdict(set), {}, {}
    for case in cases:
        for name in ("episode_id", "method", "run_id", "family", "base_scene", "scene", "condition"):
            if not isinstance(getattr(case, name), str) or not getattr(case, name).strip():
                raise ValueError(f"Expected case {name} must be a nonempty string")
        if case.family == "macro":
            raise ValueError("Family name macro is reserved for the equal-family summary")
        for name in ("seed", "sampling_seed"):
            if type(getattr(case, name)) is not int or getattr(case, name) < 0:
                raise ValueError(f"Expected case {name} must be a nonnegative integer")
        if case.run_kind not in ("trained", "fixed"):
            raise ValueError("run_kind must be trained or fixed")
        if case.actor_sha256 is None and case.controller_sha256 is None:
            raise ValueError("Every case needs an actor or controller hash")
        for value in (case.actor_sha256, case.controller_sha256):
            if value is not None and not _hash(value):
                raise ValueError("Actor/controller hashes must be lowercase SHA-256 strings")
        if case.run_kind == "trained" and case.actor_sha256 is None:
            raise ValueError("A trained run requires its selected actor hash")
        if case.episode_id in seen:
            raise ValueError("Duplicate expected episode ID")
        seen.add(case.episode_id)
        key = (case.method, case.run_id)
        if case.panel_key in panels[key]:
            raise ValueError("Duplicate physical case within a method/run")
        panels[key].add(case.panel_key)
        artifact = (case.actor_sha256, case.controller_sha256)
        if key in artifacts and artifacts[key] != artifact:
            raise ValueError("Each run must use one frozen actor/controller across all cases")
        artifacts[key] = artifact
        if case.method in kinds and kinds[case.method] != case.run_kind:
            raise ValueError("A method cannot mix fixed and trained runs")
        kinds[case.method] = case.run_kind
    if reference_method not in kinds or kinds[reference_method] != "fixed":
        raise ValueError("The declared reference must be a fixed method")
    for method, kind in kinds.items():
        if kind == "fixed" and sum(key[0] == method for key in panels) != 1:
            raise ValueError("Fixed methods must store one run; do not copy reused references")
    common = next(iter(panels.values()))
    if any(panel != common for panel in panels.values()):
        raise ValueError("Every method/run must declare exactly the same case panel")
    # One scene name cannot be relabelled into independent clusters.
    scene_groups = {}
    for case in cases:
        key = (case.family, case.scene)
        if key in scene_groups and scene_groups[key] != case.base_scene:
            raise ValueError("A scene cannot belong to multiple base-scene clusters")
        scene_groups[key] = case.base_scene
    return cases, kinds


def _metrics(outcome):
    if not isinstance(outcome, Mapping) or outcome.get("valid") is not True:
        raise ValueError("A valid audited outcome is required")
    result = {}
    for key in METRICS:
        value = outcome.get(key)
        if not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"Outcome {key} must be finite and numerical")
        result[key] = float(value)
    if any(result[key] not in (0.0, 1.0) for key in ("C_req", "S_req", "V")):
        raise ValueError("Completion and violation indicators must be binary")
    if (result["S_req"] != result["C_req"] * (1 - result["V"])
            or result["U"] != result["S_req"] - 2 * result["V"]):
        raise ValueError("Outcome contradicts S_req=C_req*(1-V) or U=S_req-2V")
    return result


def _record(case, row):
    record = {"case": asdict(case), "status": "missing", "invalid_reasons": [],
              "metrics": None, "costs": {}, "cost_issues": [], "audit": {}}
    if row is None:
        record["invalid_reasons"] = ["Expected physical evaluation is missing"]
        return record
    reasons = list(row.get("invalid_reasons") or [])
    if row.get("valid") is not True:
        reasons.append("Source audit did not mark this episode valid")
    if row.get("full_episode") is not True or row.get("short_check") is not False:
        reasons.append("Only full, explicitly non-short evaluations are eligible")
    for name in ("family", "scene", "seed", "sampling_seed", "actor_sha256"):
        if name not in row or row[name] != getattr(case, name):
            reasons.append(f"Audited row differs from expected {name}")
    for name in ("method", "run_id", "base_scene", "condition", "run_kind"):
        if name in row and row[name] != getattr(case, name):
            reasons.append(f"Audited row differs from expected {name}")
    if (case.controller_sha256 is not None
            and row.get("controller_sha256") != case.controller_sha256):
        reasons.append("Audited row differs from expected controller_sha256")
    try:
        metrics = _metrics(row.get("outcome"))
    except ValueError as error:
        reasons.append(str(error))
        metrics = None
    costs = row.get("costs", {})
    if not isinstance(costs, Mapping):
        record["cost_issues"].append("Cost map is malformed")
        costs = {}
    for name, value in costs.items():
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            record["cost_issues"].append(f"Invalid cost: {name}")
        else:
            record["costs"][name] = float(value)
    record["cost_issues"].extend(row.get("cost_issues") or [])
    reasons.extend(record["cost_issues"])
    record.update(status="invalid" if reasons else "valid", invalid_reasons=reasons,
                  metrics=None if reasons else metrics)
    record["audit"] = {name: deepcopy(row[name]) for name in
                       ("output", "audit_reader_sha256", "outcome_reader_sha256",
                        "missing_cost_fields") if name in row}
    return record


def _metric_dict(values):
    return {metric: float(value) for metric, value in zip(METRICS, values, strict=True)}


def _coverage(records):
    counts = Counter(record["status"] for record in records)
    totals, cost_counts = Counter(), Counter()
    for record in records:
        totals.update(record["costs"])
        cost_counts.update(record["costs"].keys())
    return {"expected": len(records), "observed": len(records) - counts["missing"],
            "valid": counts["valid"], "invalid": counts["invalid"], "missing": counts["missing"],
            "invalid_episode_ids": [r["case"]["episode_id"] for r in records
                                    if r["status"] == "invalid"],
            "missing_episode_ids": [r["case"]["episode_id"] for r in records
                                    if r["status"] == "missing"],
            "recorded_cost_totals": dict(totals), "cost_record_counts": dict(cost_counts)}


def _panels(cases, records):
    """Collapse within-scene repeats once, retaining the crossed run/scene axes."""
    lookup = {r["case"]["episode_id"]: r["metrics"] for r in records}
    grouped = defaultdict(list)
    for case in cases:
        grouped[(case.method, case.run_id, case.family, case.base_scene)].append(
            [lookup[case.episode_id][key] for key in METRICS])
    methods = sorted({case.method for case in cases})
    families = sorted({case.family for case in cases})
    runs = {method: sorted({case.run_id for case in cases if case.method == method})
            for method in methods}
    scenes = {family: sorted({case.base_scene for case in cases if case.family == family})
              for family in families}
    panels = {method: {family: np.asarray([
        [np.mean(grouped[(method, run, family, scene)], axis=0) for scene in scenes[family]]
        for run in runs[method]], dtype=np.float64) for family in families} for method in methods}
    return panels, runs, scenes


def _point(panels, runs, scenes):
    result = {}
    for method, families in panels.items():
        family_values = {family: values.mean(axis=(0, 1)) for family, values in families.items()}
        result[method] = {
            "families": {family: _metric_dict(value) for family, value in family_values.items()},
            "macro": _metric_dict(np.mean(list(family_values.values()), axis=0)),
            "by_run": {run: {
                "families": {family: _metric_dict(values[index].mean(axis=0))
                             for family, values in families.items()},
                "macro": _metric_dict(np.mean([values[index].mean(axis=0)
                                               for values in families.values()], axis=0)),
                "by_scene": {family: {scene: _metric_dict(values[index, si])
                                      for si, scene in enumerate(scenes[family])}
                             for family, values in families.items()},
            } for index, run in enumerate(runs[method])},
        }
    return result


def _bootstrap(panels, runs, scenes, kinds, reference, replicates, seed, level):
    rng = np.random.default_rng(seed)
    draws = {method: {family: np.empty((replicates, len(METRICS))) for family in scenes}
             for method in panels}
    for index in range(replicates):
        scene_draw = {family: rng.integers(len(ids), size=len(ids)) for family, ids in scenes.items()}
        for method in panels:
            # The same sampled actor indices apply to every family in this replicate.
            run_draw = (rng.integers(len(runs[method]), size=len(runs[method]))
                        if kinds[method] == "trained" else np.array([0]))
            for family, values in panels[method].items():
                draws[method][family][index] = values[run_draw][:, scene_draw[family]].mean(
                    axis=(0, 1))
    for families in draws.values():
        families["macro"] = np.mean(list(families.values()), axis=0)

    def intervals(method, scope, samples, compared=False):
        families = list(scenes) if scope == "macro" else [scope]
        issue = None
        if replicates == 0:
            issue = "resampling_disabled"
        elif any(len(scenes[family]) < 2 for family in families):
            issue = "fewer_than_two_base_scenes"
        elif kinds[method] == "trained" and len(runs[method]) < 2:
            issue = "fewer_than_two_independent_training_runs"
        boundary = any(np.all(panels[m][family][:, :, 2] == 0)
                       or np.all(panels[m][family][:, :, 2] == 1)
                       for m in ([method, reference] if compared else [method])
                       for family in families)
        result = {}
        for index, metric in enumerate(METRICS):
            reason = issue
            if metric == "V" and boundary:
                reason = reason or "boundary_event_rate_requires_separate_risk_analysis"
            standard_deviation = (float(samples[:, index].std(ddof=1))
                                  if replicates > 1 else None)
            if reason is None and np.ptp(samples[:, index]) == 0:
                reason = "no_empirical_resampling_variation"
            bounds = None if reason else np.quantile(
                samples[:, index], [(1 - level) / 2, (1 + level) / 2]).tolist()
            result[metric] = {"percentile_interval": bounds, "unavailable_reason": reason,
                              "resampling_standard_deviation": standard_deviation}
        return result

    estimates, comparisons = {}, {}
    for method, families in draws.items():
        estimates[method] = {scope: intervals(method, scope, values)
                             for scope, values in families.items()}
        if method != reference:
            comparisons[method] = {scope: intervals(method, scope, values - draws[reference][scope],
                                                    compared=True)
                                    for scope, values in families.items()}
    return estimates, comparisons


def analyze(expected_cases, audited_rows, *, reference_method, bootstrap_replicates=2000,
            seed=0, interval_level=0.95):
    """Return coverage, unique records, scene-balanced estimates and paired deltas.

    Structural grid mistakes, duplicate rows and unexpected episodes raise ValueError.
    Missing/invalid episodes are retained and withhold *all* performance analysis;
    passing a valid-only subset cannot silently alter the declared estimand. The
    caller must pin this module and the upstream outcome/audit readers separately.
    No risk margin, significance decision, or main-study sample size is chosen here.
    """
    if type(bootstrap_replicates) is not int or not 0 <= bootstrap_replicates <= 100000:
        raise ValueError("bootstrap_replicates must be an integer in [0, 100000]")
    if bootstrap_replicates == 1:
        raise ValueError("Use zero or at least two bootstrap replicates")
    if type(seed) is not int or seed < 0:
        raise ValueError("Bootstrap seed must be a nonnegative integer")
    if (isinstance(interval_level, bool) or not isinstance(interval_level, Real)
            or not 0 < interval_level < 1):
        raise ValueError("interval_level must lie strictly between zero and one")
    cases, kinds = _validate_grid(list(expected_cases), reference_method)
    by_id, paths = {}, set()
    declared = {case.episode_id for case in cases}
    for row in audited_rows:
        if not isinstance(row, Mapping) or row.get("episode_id") not in declared:
            raise ValueError("Unexpected or unidentified episode outside the declared grid")
        if row["episode_id"] in by_id:
            raise ValueError("Duplicate physical episode record")
        path = row.get("output")
        if path is not None:
            if not isinstance(path, str) or not path or path in paths:
                raise ValueError("Duplicate or malformed physical episode output path")
            paths.add(path)
        by_id[row["episode_id"]] = row
    records = [_record(case, by_id.get(case.episode_id)) for case in cases]
    coverage = _coverage(records)
    complete = coverage["valid"] == coverage["expected"]
    report = {
        "schema": SCHEMA, "expected_grid_sha256": _digest([asdict(case) for case in cases]),
        "reference_method": reference_method, "complete": complete, "coverage": coverage,
        "coverage_by_method_family": {method: {family: _coverage([
            r for r in records if r["case"]["method"] == method and r["case"]["family"] == family])
            for family in sorted({case.family for case in cases})} for method in sorted(kinds)},
        "records": records, "estimates": None, "comparisons": None,
        "estimand": "equal families / equal base scenes / equal within-scene cases / equal runs",
        "cost_scope": "Unique supplied evaluation records only; not training or total study cost.",
        "resampling": {"replicates": bootstrap_replicates, "seed": seed,
                       "percentile_level": float(interval_level),
                       "method": "crossed_scene_and_training_run_percentile_bootstrap",
                       "assumptions": [
                           "Independent exchangeable base scenes within each fixed family.",
                           "Independent training runs within and between methods; no seed pairing.",
                           "Training-run draws retained across families; scene draws shared by methods.",
                           "All variants and policy seeds of a base scene remain one cluster.",
                           "The observed selected actors represent training runs; optimization/selection is not rerun.",
                           "The family set, checkpoints and within-scene panels are fixed.",
                           "Matched case labels do not assert identical physical starts.",
                           "Descriptive intervals; few clusters do not establish calibrated coverage.",
                       ]},
        "risk_inference": {"status": "not_assessed", "margin": None,
                           "zero_event_rule": "Report counts and scene/run denominators; risk remains "
                           "inconclusive. Withhold V intervals at observed boundaries; no iid 3/n "
                           "bound is applied to correlated evaluations."},
    }
    if not complete:
        return report
    panels, runs, scenes = _panels(cases, records)
    report["estimates"] = _point(panels, runs, scenes)
    differences = {method: {family: values - panels[reference_method][family]
                             for family, values in families.items()}
                   for method, families in panels.items() if method != reference_method}
    report["comparisons"] = _point(differences, runs, scenes)
    intervals, comparison_intervals = _bootstrap(
        panels, runs, scenes, kinds, reference_method, bootstrap_replicates, seed, interval_level)
    for method, estimate in report["estimates"].items():
        own = [r for r in records if r["case"]["method"] == method]
        estimate.update(run_kind=kinds[method], run_ids=runs[method],
                        unique_episode_count=len(own), intervals=intervals[method],
                        observed_counts={metric: int(sum(r["metrics"][metric] for r in own))
                                         for metric in ("C_req", "S_req", "V")},
                        observed_counts_by_family={family: {
                            metric: int(sum(r["metrics"][metric] for r in own
                                            if r["case"]["family"] == family))
                            for metric in ("C_req", "S_req", "V")} for family in scenes},
                        unique_episode_counts_by_family={family: sum(
                            r["case"]["family"] == family for r in own) for family in scenes},
                        base_scene_counts={family: len(ids) for family, ids in scenes.items()})
        if method != reference_method:
            report["comparisons"][method]["intervals"] = comparison_intervals[method]
    return report



def analyze_protocol(protocol, audited_rows, *, config_dir=None, root=None):
    """Analyze the entire verified frozen grid with its exact analysis settings.

    This is the main-study entry point; ``analyze`` remains a lower-level numerical
    API. Each input row must carry every compiled job field at top level, the frozen
    ``protocol_sha256``, and an ``audit_reader_sha256`` present in implementation_files.
    The reader must independently verify those recorded bindings against execution
    evidence. A phase-two row cannot become a main row merely by attaching plan fields.

    All declared attempted-work accounting counters, elapsed_seconds, cost_issues and
    missing_cost_fields must be explicit, including applicable zeros. A missing or
    mismatched contract invalidates that row without discarding its known costs.
    No result files are discovered, and no live services are called. Source validation
    reads only the frozen metadata, models and review/source artifacts. This wrapper
    does not independently replay physical evidence, enforce live budgets, or perform
    the separately preregistered safety-margin/multiple-comparison decision procedure.
    """
    from pathlib import Path

    from helpers.maniguard_main_protocol import expected_cases, validate_main_protocol
    from helpers.maniguard_phase2_families import CONFIG, ROOT

    validate_main_protocol(protocol, config_dir=CONFIG if config_dir is None else config_dir,
                           root=ROOT if root is None else root, require_frozen=True)
    analysis = protocol["spec"]["analysis"]
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if analysis["implementation"]["sha256"] != source_hash:
        raise ValueError("The running analysis implementation differs from the frozen source")
    jobs = expected_cases(protocol)
    by_id = {job["episode_id"]: job for job in jobs}
    reader_hashes = {reference["sha256"]
                     for reference in protocol["spec"]["implementation_files"]}
    checked = []
    for original in audited_rows:
        if not isinstance(original, Mapping) or original.get("episode_id") not in by_id:
            raise ValueError("Unexpected or unidentified episode outside the frozen grid")
        row = deepcopy(dict(original))
        reasons = row.get("invalid_reasons", [])
        if not isinstance(reasons, list) or not all(isinstance(value, str) for value in reasons):
            reasons = ["Invalid source-audit reason list"]
        else:
            reasons = list(reasons)
        for key, expected in by_id[row["episode_id"]].items():
            try:
                matches = key in row and _digest(row[key]) == _digest(expected)
            except (TypeError, ValueError):
                matches = False
            if not matches:
                reasons.append("Audited row differs from frozen job binding: " + key)
        if row.get("protocol_sha256") != protocol["sha256"]:
            reasons.append("Audited row lacks the exact frozen protocol binding")
        if (not _hash(row.get("audit_reader_sha256"))
                or row["audit_reader_sha256"] not in reader_hashes):
            reasons.append("Audited row reader is absent from frozen implementation sources")
        if not isinstance(row.get("output"), str) or not row["output"]:
            reasons.append("A recorded physical episode output path is required")
        costs = row.get("costs")
        if not isinstance(costs, Mapping):
            reasons.append("Explicit attempted-work cost map is required")
            costs = {}
        for name in REQUIRED_COUNT_COSTS:
            if type(costs.get(name)) is not int or costs[name] < 0:
                reasons.append("Missing or invalid attempted-work count: " + name)
        elapsed = costs.get("elapsed_seconds")
        if (type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0):
            reasons.append("Missing or invalid recorded elapsed_seconds")
        for name in ("cost_issues", "missing_cost_fields"):
            value = row.get(name)
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                reasons.append("Explicit accounting audit list is required: " + name)
                row[name] = []
            elif value:
                reasons.append("Unresolved accounting audit: " + name)
        row["invalid_reasons"] = reasons
        checked.append(row)
    result = analyze(jobs, checked, reference_method=analysis["reference_method"],
                     bootstrap_replicates=analysis["bootstrap_replicates"],
                     seed=analysis["bootstrap_seed"], interval_level=analysis["confidence_level"])
    # Keep the whole execution case and exposure history in the final report, rather
    # than only the projection needed by the numerical estimand.
    for record in result["records"]:
        record["case"] = deepcopy(by_id[record["case"]["episode_id"]])
    result.update(schema=PROTOCOL_SCHEMA, protocol_binding={
        "protocol_sha256": protocol["sha256"], "grid_sha256": protocol["grid_sha256"],
        "analysis_sha256": _digest(analysis), "analysis_source_sha256": source_hash,
        "analysis": deepcopy(analysis), "exposure": deepcopy(protocol["exposure"]),
        "required_count_costs": list(REQUIRED_COUNT_COSTS),
        "audit_reader_hashes": sorted(reader_hashes),
    })
    result["risk_inference"]["prespecified_margin"] = analysis["violation_risk_margin"]
    result["decision_scope"] = {
        "status": "descriptive_analysis_only",
        "meaningful_effect_rule_evaluated": False,
        "safety_margin_rule_evaluated": False,
        "multiple_comparison_rule_evaluated": False,
        "physical_evidence_independently_replayed": False,
        "live_budget_enforcement_verified": False,
    }
    return result
