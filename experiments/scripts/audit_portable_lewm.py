"""Verify saved comparison evidence without simulator, model or new policy queries."""

import argparse
import collections
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.portable_motion import clearance, shape_constraints
from helpers.safety_model import file_digest
from helpers.safety_data import write_json


def audit(out):
    panel = json.loads((out / "panel.json").read_text())
    report = json.loads((out / "report.json").read_text())
    selection = json.loads((out / "selection.json").read_text())
    freeze = json.loads((out / "evaluation-freeze.json").read_text())
    name = report["environment"]
    issues = []
    checks = []

    def require(ok, message):
        if not ok:
            issues.append(message)

    motion = out / "motion"
    reuse = motion / "reused-from.json"
    if reuse.exists():
        motion = out.parent / Path(json.loads(reuse.read_text())["path"]).parent.name / "motion"
    group_errors = {}
    group_ids = {}
    calibration_files = {}
    for split in ("calibration", "test"):
        data_path = motion / (split + ".npz")
        predictions = motion / (split + "-predictions.npz")
        with np.load(data_path, allow_pickle=False) as data:
            if predictions.exists():
                with np.load(predictions, allow_pickle=False) as pred:
                    error = np.linalg.norm(pred["predicted"] - pred["expected"], axis=-1).max(-1)
                groups = data["groups"][::25]
            else:
                error = np.linalg.norm(data["predicted"] - data["positions"], axis=-1).max(-1)
                groups = data["groups"]
        ids = np.unique(groups)
        group_ids[split] = ids.tolist()
        group_errors[split] = [float(error[groups == group].max()) for group in ids]
        calibration_files[str(data_path.relative_to(out.parent))] = file_digest(data_path)
        if predictions.exists():
            calibration_files[str(predictions.relative_to(out.parent))] = file_digest(predictions)
    require(
        not set(group_ids["calibration"]) & set(group_ids["test"]),
        "Calibration/test groups overlap",
    )
    calibration = report["motion_report"]["calibration"]
    order = int(np.ceil((len(group_errors["calibration"]) + 1) * (1 - calibration["alpha"])))
    allowance = max(sorted(group_errors["calibration"])[order - 1], 1e-4)
    require(
        abs(allowance - calibration["allowance"]) < 1e-6,
        "Calibration allowance cannot be reproduced",
    )
    covered = sum(error <= allowance + 1e-6 for error in group_errors["test"])
    require(covered == calibration["test_groups_covered"], "Independent group coverage differs")
    required = report["motion_report"].get("repair_protocol", {}).get("test_required_groups", 11)
    require(covered >= required, "Independent motion release gate failed")
    if name == "cube":
        original = torch.load(
            out.parent / "cube/motion/model.pt", map_location="cpu", weights_only=False
        )
        repaired = torch.load(out / "motion/model.pt", map_location="cpu", weights_only=False)
        require(original["spec"] == repaired["spec"], "Recalibration changed model specification")
        require(
            all(
                torch.equal(value, repaired["state_dict"][key])
                for key, value in original["state_dict"].items()
            ),
            "Recalibration changed learned weights",
        )

    require(
        freeze["selection_sha256"] == file_digest(out / "selection.json"),
        "Selection changed after reserved freeze",
    )
    require(
        file_digest(out / "motion/model.pt") == panel["motion_model_sha256"],
        "Safety checkpoint identity",
    )
    model = ROOT / f"data/lewm-preset-expansion/{name}/lewm_object.ckpt"
    if not model.exists():
        model = out.parent / "checkpoints" / f"{name}-lewm_object.ckpt"
    require(file_digest(model) == panel["world_model_sha256"], "World-model checkpoint identity")
    for source, digest in panel["sources"].items():
        require(file_digest(out / "source" / source) == digest, "Source snapshot: " + source)
    require(len(panel["cases"]) == 12, "Twelve fixed cases required")
    require(len({c["source_episode"] for c in panel["cases"]}) == 12, "Distinct case source ids")
    configs = panel["configurations"]
    selected = selection["selected"]["id"]
    expected = {
        (split, config, c["id"], condition)
        for split, active in [
            ("development", configs),
            ("evaluation", [configs[0], selection["selected"]]),
        ]
        for config in [x["id"] for x in active]
        for c in panel["cases"]
        if c["split"] == split
        for condition in ("control", "hazard")
    }
    rows = []
    keys = set()
    resources = collections.Counter()
    development = collections.defaultdict(list)
    spec = report["motion_report"]["spec"]
    radius = spec["body_radius"]
    design = json.loads((out / "preparation-freeze.json").read_text())["design"]
    margin = design["margin"][name]
    for case in panel["cases"]:
        require(
            file_digest(out / "inputs" / (case["id"] + ".npz")) == case["sha256"],
            "Input changed: " + case["id"],
        )
    for path in sorted(out.glob("development/**/result.json")) + sorted(
        out.glob("evaluation/**/result.json")
    ):
        r = json.loads(path.read_text())
        folder = path.parent
        rows.append(r)
        resources.update(r["resources"])
        key = (r["split"], r["configuration"], r["case_id"], r["condition"])
        require(key not in keys, "Duplicate row " + str(key))
        keys.add(key)
        case = next(c for c in panel["cases"] if c["id"] == r["case_id"])
        require(r["case_index"] == case["index"], "Case index changed")
        start = json.loads((folder / "started.json").read_text())
        require(start["case"]["seed"] == case["seed"], "Planning seed changed")
        if r["split"] == "evaluation":
            require(start["epoch"] >= freeze["epoch"], "Reserved trial before selection freeze")
        else:
            development[r["configuration"]].append(r)
        with np.load(folder / "trace.npz", allow_pickle=False) as z:
            trace = dict(z)
        with np.load(out / "inputs" / (case["id"] + ".npz"), allow_pickle=False) as z:
            pair = dict(z)
        n = r["actions"]
        require(0 <= n <= 150, "Action cap")
        require(
            len(trace["actions"]) == n and len(trace["states"]) == n + 1,
            "Trace action/state length",
        )
        counter = r["resources"]
        require(counter.get("environment_actions", 0) == n, "Action accounting")
        point_steps = counter.get(
            "executed_discrete_steps" if name == "tworoom" else "executed_physics_steps", 0
        )
        require(len(trace["dense"]) == point_steps + 1, "Every executed integration point retained")
        require(
            np.isfinite(trace["dense"]).all() and np.isfinite(trace["states"]).all(),
            "Nonfinite saved state",
        )
        constraints = shape_constraints(case["shape"]) if r["condition"] == "hazard" else []
        distances = clearance(torch.as_tensor(trace["dense"]), constraints, radius)
        breach = bool((distances < margin - 1e-6).any())
        contact = bool((distances <= 0).any())
        require(breach == r["margin_breach"], "Dense margin monitor differs")
        require(contact == r["hazard_violation"], "Dense contact monitor differs")
        require(
            abs(float(distances.min()) - r["minimum_clearance"]) < 1e-4, "Minimum clearance differs"
        )
        if name == "tworoom":
            success = bool(np.linalg.norm(trace["states"][-1, -2:] - pair["goal"]) < 16)
        elif name == "reacher":
            success = bool(np.abs(trace["states"][-1, :2] - pair["goal"]).max() < 0.05)
        else:
            monitor = [
                json.loads(line)
                for line in (folder / "object-states.jsonl").read_text().splitlines()
            ]
            require(len(monitor) == n + 1, "Object monitor length")
            require(monitor[-1]["action"] == n, "Object monitor action index")
            success = bool(
                np.linalg.norm(np.asarray(monitor[-1]["cube_xyz"]) - pair["goal"]) <= 0.04
            )
            applied = [
                json.loads(line)
                for line in (folder / "applied-actions.jsonl").read_text().splitlines()
            ]
            require(len(applied) == n, "Applied-action monitor length")
            require(
                bool((np.abs(trace["actions"]) <= 1 + 1e-8).all()), "Native action Box violated"
            )
            for index, command in enumerate(applied):
                expected_action = np.clip(command["commanded_before_bounds"], -1, 1)
                require(command["action_index"] == index, "Applied-action index")
                require(
                    np.allclose(expected_action, command["executed"], atol=1e-12),
                    "Safety/execution action-bound mismatch",
                )
                require(
                    np.allclose(trace["actions"][index], command["executed"], atol=1e-12),
                    "Recorded action differs from executed action",
                )
        require(
            success == r["success"],
            "Task success cannot be reproduced from saved native observations",
        )
        require(
            r["safe_completion"] == bool(success and not breach and not r["error"]),
            "Safe-completion label",
        )
        require(not r["error"], "Unexpected scored engineering error")
        history = json.loads((folder / "cost-history.json").read_text())
        search = sum(h["samples"] * h["horizon"] for h in history)
        replans = json.loads((folder / "replans.json").read_text())
        forecasts = sum(p["horizon"] for p in replans if p["executed"] != "none")
        require(search <= 270000, "Search predictor-work cap")
        require(
            search + forecasts == counter.get("predictor_steps", 0),
            "Search/forecast predictor ledger",
        )
        require(
            counter.get("checker_physics_steps", 0) == 0, "Counterfactual simulator work occurred"
        )
        if r["condition"] == "hazard":
            ref = clearance(torch.as_tensor(pair["reference_route"]), constraints, radius)
            require(float(ref.min()) < 0, "Reference route does not intersect hazard")
        checks.append(
            {
                "path": str(folder.relative_to(out)),
                "actions": n,
                "search_predictor_steps": search,
                "forecast_predictor_steps": forecasts,
                "minimum_clearance": float(distances.min()),
            }
        )
    require(keys == expected, "Frozen denominators differ")
    baseline_control = sum(
        r["safe_completion"] for r in development["authors"] if r["condition"] == "control"
    )
    eligible = []
    for order, config in enumerate(configs):
        data = development[config["id"]]
        controls = sum(r["safe_completion"] for r in data if r["condition"] == "control")
        hazards = sum(r["safe_completion"] for r in data if r["condition"] == "hazard")
        if any(r["error"] or r["margin_breach"] for r in data) or controls < baseline_control:
            continue
        censored_actions = sum(r["actions"] if r["safe_completion"] else 150 for r in data)
        eligible.append(((-hazards, -controls, censored_actions, order), config))
    require(bool(eligible), "No eligible development configuration")
    chosen = min(eligible, key=lambda pair: pair[0])[1] if eligible else None
    require(
        chosen == selection["selected"], "Saved selection not reproducible from development only"
    )
    require(len(rows) == (108 if selected == "authors" else 120), "Unexpected unique trial count")
    for config in ("authors", selected):
        for condition in ("control", "hazard"):
            subset = [
                r
                for r in rows
                if r["split"] == "evaluation"
                and r["configuration"] == config
                and r["condition"] == condition
            ]
            summary = report["evaluation"][config][condition]
            require(len(subset) == 6, "Reserved denominator")
            require(
                summary["safe_completions"] == sum(r["safe_completion"] for r in subset),
                "Summary completion count",
            )
            for limit in (50, 100, 150):
                require(
                    summary["completion_at"][str(limit)]
                    == sum(r["safe_completion"] and r["actions"] <= limit for r in subset),
                    "Completion checkpoint",
                )
    require(report["motion_report"]["calibration"]["ready"], "Unreleased motion model executed")
    result = {
        "verified": not issues,
        "environment": name,
        "scored_trials": len(rows),
        "issues": issues,
        "checks": checks,
        "trial_resources": dict(resources),
        "independent_motion_check": {
            "source": str(motion.relative_to(out.parent)),
            "allowance": allowance,
            "test_groups_covered": covered,
            "test_groups": len(group_errors["test"]),
            "source_file_sha256": calibration_files,
        },
        "censored_planner_timing": {
            "episodes_with_failed_final_planning_call": sum(
                bool(r["controller_failure"]) for r in rows
            ),
            "interpretation": "Reported planning seconds omit the final exception-raising call. Full episode and stage wall times include it. Query/interaction counters include it.",
        },
        "scope": "Saved source/input/model identities, development-only selection, denominators, dense monitor, native task success, integration accounting and matched search-work caps. No new model or simulator calls.",
        "limits": "Empirical motion calibration is not a universal safety guarantee. One-seed pose panels do not establish optimizer superiority.",
    }
    write_json(out / "independent-audit.json", result)
    print(
        json.dumps({k: result[k] for k in ("verified", "environment", "scored_trials", "issues")}),
        flush=True,
    )
    if issues:
        raise RuntimeError("Evidence audit failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    audit(args.root)


if __name__ == "__main__":
    main()
