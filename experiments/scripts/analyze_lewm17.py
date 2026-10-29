"""Audit and analyze the complete frozen task-17 campaign from saved evidence."""

import collections
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np

from helpers.lewm17_analysis import crossed_interval, matrix_for, paired_contrast, summarize
from helpers.lewm17_budget import ceilings
from helpers.lewm17_protocol import CONFIG, OUT, boxes, digest, hits, write_json
from helpers.lewm17_selector import MARGINS


def main():
    config = json.loads((CONFIG / "comparison.json").read_text())
    final_manifest = json.loads((ROOT / "progress/receipts/17-final-evaluation.json").read_text())
    paths = sorted((OUT / "attempts").glob("*/result.json"))
    rows = [json.loads(p.read_text()) for p in paths]
    expected_total = 48 + 360 + 30 * config["episodes_per_run"] + len(final_manifest["jobs"])
    assert len(rows) == expected_total
    assert all(r["status"] == "complete" for r in rows)
    assert len(list((OUT / "attempts").iterdir())) == len(rows)
    final = [r for r in rows if r["job"]["stage"] == "H"]
    assert {r["job"]["id"] for r in final} == {j["id"] for j in final_manifest["jobs"]}
    selected = json.loads((OUT / "final-selection.json").read_text())
    access = json.loads((OUT / "final-access.json").read_text())
    assert selected["frozen_epoch"] <= access["utc_epoch"]
    assert digest(OUT / "final-selection.json") == final_manifest["extra"]["selection_sha256"]
    replay_max = collections.defaultdict(float)
    query_calls = 0
    for path, row in zip(paths, rows):
        j, counters = row["job"], row["counters"]
        trace = np.load(path.parent / "states.npz")
        n = len(trace["actions"])
        assert n == row["env_steps"] == counters.get("environment_actions", 0) <= 50
        assert len(trace["fine_states"]) == counters.get("executed_physics_steps", 0)
        assert all(np.isfinite(trace[key]).all() for key in trace.files)
        commands = trace["actions"]
        row["command_norm_sum"] = float(np.linalg.norm(commands, axis=1).sum())
        row["bounded_control_norm_sum"] = float(
            np.linalg.norm(np.clip(commands, -1, 1), axis=1).sum()
        )
        row["action_coordinates_clipped"] = int((np.abs(commands) > 1).sum())
        row["action_coordinates_total"] = int(commands.size)
        fine = trace["fine_states"]
        xy = fine[:, :2] if j["env"] == "tworoom" else fine[:, 4:6]
        original = bool(hits(xy, boxes(j["env"], j["case"])).any())
        margin = bool(hits(xy, boxes(j["env"], j["case"]), j["margin"]).any())
        assert original == row["any_violation"] and margin == row["margin_breach"]
        assert row["requested_safe_completion"] == (row["success"] and not margin)
        assert row["benchmark_safe_completion"] == (row["success"] and not original)
        assert row["unfinished"] == (not row["success"])
        assert row["J"] == (n if row["requested_safe_completion"] else 50)
        limits = ceilings(j["env"])
        if j.get("native_parity"):
            limits["predictor_steps"] += 50000 if j["env"] == "tworoom" else 15000
        assert all(counters.get(k, 0) <= v for k, v in limits.items())
        replans = json.loads((path.parent / "replans.json").read_text())
        history = json.loads((path.parent / "history.json").read_text())
        assert len(replans) <= 10
        calls = len(history) + sum(r.get("executed") in ("mean", "best_checked") for r in replans)
        if row.get("native_parity_max") is not None:
            calls += 10
        query_calls += calls
        row["derived_model_batch_queries"] = calls
        replay_max[j["env"]] = max(replay_max[j["env"]], row["replay_error_max"])
        assert row["replay_error_max"] <= (1e-3 if j["env"] == "tworoom" else 1e-7)
        if j["params"].get("weights") is not None:
            decisions = json.loads((path.parent / "decisions.json").read_text())
            assert len(decisions) == len(replans)
            for d in decisions:
                assert len(d["x"]) == 6 and len(d["probabilities"]) == 6
                assert d["x"][-1] == j["margin"] / max(MARGINS[j["env"]])
    runs = [json.loads(p.read_text()) for p in sorted((OUT / "optimizers").glob("*/selected.json"))]
    assert len(runs) == 30
    for run in runs:
        name = f"{run['env']}-{run['method']}-r{run['replicate']}"
        own = [r for r in rows if r["job"].get("optimizer_run") == name]
        assert len(own) == config["episodes_per_run"]
        measured = summarize(own)["resources"]
        assert measured == run["resources"]
        assert all(
            measured.get(k, 0) <= v
            for k, v in config["per_run_resource_ceilings"][run["env"]].items()
        )
    methods = ["historical", "tuned", "random", "reinforce", "evolution"]
    tables, contrasts, endpoints = {}, {}, {}
    for env in ("tworoom", "reacher"):
        tables[env], contrasts[env], endpoints[env] = {}, {}, {}
        for margin in MARGINS[env]:
            key = f"{margin:g}"
            subset = [r for r in final if r["job"]["env"] == env and r["job"]["margin"] == margin]
            cases = sorted({r["job"]["case"]["id"] for r in subset})
            assert len(cases) == config["final_cases"]
            matrices = {m: matrix_for(subset, m, cases) for m in methods}
            tables[env][key], endpoints[env][key], contrasts[env][key] = {}, {}, {}
            for method in methods:
                own = [
                    r
                    for r in subset
                    if r["job"]["condition"] == method or r["job"].get("optimizer") == method
                ]
                tables[env][key][method] = summarize(own)
                endpoints[env][key][method] = crossed_interval(matrices[method])
            for control in ("random", "reinforce", "tuned", "historical"):
                confidence = (
                    1 - 0.05 / 3
                    if env == "reacher" and margin == 0.015 and control != "historical"
                    else 0.95
                )
                contrasts[env][key][control] = paired_contrast(
                    matrices["evolution"], matrices[control], confidence
                )
    primary = contrasts["reacher"]["0.015"]
    gain_gate = all(
        primary[m]["estimate"] >= 0.1 and primary[m]["interval"][0] > 0
        for m in ("random", "reinforce", "tuned")
    )
    context_tables, tuning_tables = {}, {}
    for env in ("tworoom", "reacher"):
        context_tables[env], tuning_tables[env] = {}, {}
        for margin in MARGINS[env]:
            context_tables[env][f"{margin:g}"] = {
                condition: summarize(
                    [
                        r
                        for r in rows
                        if r["job"]["stage"] == "C"
                        and r["job"]["env"] == env
                        and r["job"]["margin"] == margin
                        and r["job"]["condition"] == condition
                    ]
                )
                for condition in ("native", "canonical")
            }
            tuning_tables[env][f"{margin:g}"] = {
                f"mode{mode}": summarize(
                    [
                        r
                        for r in rows
                        if r["job"]["stage"] == "F"
                        and r["job"]["env"] == env
                        and r["job"]["margin"] == margin
                        and r["job"]["condition"] == f"mode{mode}"
                    ]
                )
                for mode in range(6)
            }
    report = {
        "schema": "lewm17-analysis-v1",
        "episodes": len(rows),
        "final_episodes": len(final),
        "workload": config["workload"],
        "stages": dict(collections.Counter(r["job"]["stage"] for r in rows)),
        "all_attempt_resources": summarize(rows)["resources"],
        "model_batch_query_calls_derived": query_calls,
        "model_query_definition": "One frozen cost-model forward batch per recorded CEM iteration, plus selected-plan predictions and native identity probes; candidate rollouts and predictor steps separately counted",
        "final_tables": tables,
        "context_tables": context_tables,
        "fixed_tuning_tables": tuning_tables,
        "stage_summaries": {
            stage: summarize([r for r in rows if r["job"]["stage"] == stage])
            for stage in ("E", "E2", "C", "F", "A", "T", "V", "H")
        },
        "endpoint_intervals": endpoints,
        "evolution_minus": contrasts,
        "optimizer_runs": runs,
        "practical_completion_gain_gate": gain_gate,
        "replay_max": dict(replay_max),
        "failed_attempts": 0,
        "uncertainty_limits": "Five optimizer runs, twelve independent case blocks, two planner seeds. Bootstrap may degenerate at a ceiling. Zero observed breaches does not establish zero risk; 3/n is only an episode-wise approximation and ignores dependence.",
        "fairness": "Equal enforced resource allowances, unequal actual work. No resource-matched learning-curve checkpoints, hence no search-efficiency claim.",
    }
    write_json(OUT / "analysis/statistics.json", report)
    write_json(ROOT / "progress/receipts/17-final-statistics.json", report)
    write_json(
        ROOT / "progress/receipts/17-final-audit.json",
        {
            "passed": True,
            "all_attempts_complete": len(rows),
            "expected": expected_total,
            "final_ids_exact": True,
            "selection_precedes_final_access": True,
            "dense_monitors_recomputed": True,
            "all_resource_limits_checked": True,
            "optimizer_resource_totals_recomputed": True,
            "replay_max": dict(replay_max),
            "final_manifest_sha256": digest(ROOT / "progress/receipts/17-final-evaluation.json"),
        },
    )
    csv_path = OUT / "analysis/final-episodes.csv"
    fields = [
        "env",
        "condition",
        "replicate",
        "case",
        "margin",
        "seed",
        "safe",
        "violation",
        "margin_breach",
        "unfinished",
        "engaged",
        "censored",
        "J",
        "actions",
        "seconds",
    ]
    with csv_path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in final:
            j = r["job"]
            writer.writerow(
                {
                    "env": j["env"],
                    "condition": j["condition"],
                    "replicate": j.get("replicate", ""),
                    "case": j["case"]["id"],
                    "margin": j["margin"],
                    "seed": j["seed"],
                    "safe": r["requested_safe_completion"],
                    "violation": r["any_violation"],
                    "margin_breach": r["margin_breach"],
                    "unfinished": r["unfinished"],
                    "engaged": r["engaged"],
                    "censored": r["censored"],
                    "J": r["J"],
                    "actions": r["env_steps"],
                    "seconds": r["seconds"],
                }
            )
    print(json.dumps({"episodes": len(rows), "primary": primary, "gain_gate": gain_gate}))


if __name__ == "__main__":
    main()
