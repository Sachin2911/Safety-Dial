"""Independently audit the matched preset/tuning traces and extension gate."""

import argparse
import collections
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.pusht_preset import select, summary
from helpers.safety_model import file_digest, pusht_constraints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    out = args.root
    panel = json.loads((out / "panel.json").read_text())
    design = panel["design"]
    selection = json.loads((out / "selection.json").read_text())
    freeze = json.loads((out / "evaluate-freeze.json").read_text())
    assert freeze["selection_sha256"] == file_digest(out / "selection.json")
    assert freeze["panel_sha256"] == file_digest(out / "panel.json")
    for name, digest in panel["sources"].items():
        assert file_digest(out / "source" / name) == digest
    for name, digest in selection["development_result_hashes"].items():
        assert file_digest(out / name) == digest
    all_rows = []
    audits = []
    by_config = collections.defaultdict(list)
    initials = {}
    for folder in sorted(out.glob("*/*/*/*/result.json")):
        row = json.loads(folder.read_text())
        replans = json.loads((folder.parent / "replans.json").read_text())
        with np.load(folder.parent / "trace.npz", allow_pickle=False) as trace:
            states, actions, dense = trace["states"], trace["actions"], trace["dense"]
        assert len(states) == len(actions) + 1 and len(dense) == 1 + 10 * len(actions)
        case = next(c for c in panel["cases"] if c["id"] == row["case_id"])
        assert np.allclose(states[0], case["settled_initial"], atol=1e-9, rtol=0)
        if case["id"] in initials:
            assert np.array_equal(states[0], initials[case["id"]])
        initials[case["id"]] = states[0]
        constraints = pusht_constraints(row["case"])
        clearance = (
            torch.stack([c.clearance(torch.tensor(dense[:, :2]), 15) for c in constraints], -1)
            .amin(-1)
            .numpy()
        )
        assert clearance[0] >= 10 - 1e-6
        breach = bool((clearance < 6 - 1e-6).any())
        assert breach == bool(row["margin_breach"] or row["arena_margin_breach"])
        goal = np.array(row["goal_state"])
        delta = abs(states[-1, 4] - goal[4])
        angle = min(delta, 2 * np.pi - delta)
        success = bool(np.linalg.norm(states[-1, :4] - goal[:4]) < 20 and angle < np.pi / 9)
        assert success == row["success"]
        p = row["planner"]
        h = p["horizon"]
        r = p["replan_blocks"]
        n = 300 * r // h
        completed = sum(replan["executed"] != "none" for replan in replans)
        search = 30 * n * h * len(replans)
        forecasts = h * completed
        assert row["resources"]["predictor_steps"] == search + forecasts
        assert row["resources"]["model_candidate_rollouts"] == 30 * n * len(replans) + completed
        assert search <= 90000
        assert (
            row["reference_predictors_forbidden"]
            and not row["safety_model"]["reference_controls_execution"]
        )
        assert row["resources"].get("analytic_path_steps", 0) == 0
        assert row["error"] is None
        audits.append(
            {
                "file": str(folder.relative_to(out)),
                "initial_clearance": float(clearance[0]),
                "minimum_clearance": float(clearance.min()),
                "search_predictor_steps": search,
                "forecast_predictor_steps": forecasts,
                "replans": len(replans),
            }
        )
        all_rows.append(row)
        if row["split"] == "development":
            by_config[row["config_id"]].append(row)
    assert select(panel["configurations"], by_config)["selected"] == selection["selected"]
    selected = selection["selected"]["id"]
    arms = ["authors"] if selected == "authors" else ["authors", selected]
    evaluation = {}
    for arm in arms:
        evaluation[arm] = {
            condition: summary(
                [
                    r
                    for r in all_rows
                    if r["split"] == "evaluation"
                    and r["config_id"] == arm
                    and r["condition"] == condition
                ]
            )
            for condition in ("control", "hazard")
        }
        assert all(evaluation[arm][c]["episodes"] == 6 for c in ("control", "hazard"))
    base = evaluation["authors"]
    chosen = evaluation[selected]
    gate = (
        chosen["hazard"]["safe_completions"] > base["hazard"]["safe_completions"]
        and chosen["hazard"]["violations"] == 0
        and chosen["control"]["violations"] == 0
        and chosen["control"]["safe_completions"] >= base["control"]["safe_completions"]
    )
    combined = collections.Counter(summary(all_rows)["resources"])
    combined.update(panel["preparation_resources"])
    training = json.loads(
        (ROOT / "outputs/pusht-preset-motion-h5-20261029/report.json").read_text()
    )
    combined.update(training["resources"])
    combined.update({"environment_resets": 2, "reset_physics_steps": 4})
    assert len(all_rows) <= 120 and sum(r["actions"] for r in all_rows) <= 6000
    assert combined["environment_actions"] <= 23600
    assert combined["executed_physics_steps"] + combined["reset_physics_steps"] <= 238132
    result = {
        "study": design["name"],
        "audited": True,
        "selected": selection["selected"],
        "development": selection["development"],
        "evaluation": evaluation,
        "conditional_all_environment_gate_passed": bool(gate),
        "all_unique_trials": summary(all_rows),
        "training": training,
        "preparation_resources": panel["preparation_resources"],
        "native_test_resources": {
            "environment_resets": 2,
            "reset_physics_steps": 4,
            "environment_actions": 0,
        },
        "combined_resources": dict(combined),
        "combined_physics_steps": combined["executed_physics_steps"]
        + combined["reset_physics_steps"],
        "audit": audits,
        "design_sha256": panel["design_sha256"],
        "limitations": design["limitations"],
        "source_partitions_disjoint": len({c["source_episode"] for c in panel["cases"]}) == 12,
        "all_cases_initially_clear": True,
        "all_frozen_denominators_retained": True,
    }
    (out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "selected",
                    "evaluation",
                    "conditional_all_environment_gate_passed",
                    "combined_resources",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
