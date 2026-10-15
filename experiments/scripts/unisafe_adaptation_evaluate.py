"""Audit reserved tests, matched observed prefixes and paired outcome contrasts."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.unisafe_source import ROOT, sha256
from helpers.unisafe_statistics import holm, paired_binary


def trace(path):
    return [json.loads(line) for line in (path / "steps.jsonl").read_text().splitlines()]


def matched_prefix(reference, candidate):
    a, b = trace(reference), trace(candidate)
    n = min(len(a), len(b))
    first = next((i for i in range(n) if a[i]["executed_normalized_action"] != b[i]["executed_normalized_action"]), None)
    preceding = n if first is None else first
    compare_count = min(preceding + 1, n)
    x, y = np.load(reference / "initial_state.npz"), np.load(candidate / "initial_state.npz")
    initial_error = max(float(np.abs(x[k] - y[k]).max()) for k in x.files)
    states_match = all(a[i]["before"] == b[i]["before"] for i in range(compare_count))
    cameras_match = all(a[i]["camera_sha256"] == b[i]["camera_sha256"] for i in range(compare_count))
    after_match = all(a[i]["after"] == b[i]["after"] for i in range(preceding))
    equal_lengths_if_no_change = first is not None or len(a) == len(b)
    observed_equal = initial_error == 0 and states_match and cameras_match and after_match and equal_lengths_if_no_change
    if first is None:
        cause, attributable, same_task = "no executed-action difference", equal_lengths_if_no_change, None
    else:
        left, right = a[first], b[first]
        same_task = left["task_proposed_action"] == right["task_proposed_action"]
        proposal_changed = left["proposed_action"] != right["proposed_action"]
        declared_adapter = left.get("adapter_gripper_override", False) or right.get("adapter_gripper_override", False)
        reference_mode = json.loads((reference / "summary.json").read_text())["mode"]
        candidate_mode = json.loads((candidate / "summary.json").read_text())["mode"]
        same_assessment = (
            left["would_intervene"] == right["would_intervene"]
            and left["backup_candidate_action"] == right["backup_candidate_action"]
        )
        if same_task and proposal_changed and declared_adapter:
            cause, attributable = "declared task-adapter proposal", True
        elif (same_task and not proposal_changed and reference_mode != candidate_mode
              and same_assessment and left["intervened"] != right["intervened"]):
            cause, attributable = "declared filter-mode difference", True
        else:
            cause, attributable = "not isolated by recorded controller fields", False
    return {
        "pass": observed_equal and attributable,
        "observed_prefix_equal": observed_equal,
        "first_difference_has_declared_controller_cause": attributable,
        "first_difference_cause": cause,
        "task_proposal_equal_at_first_difference": same_task,
        "first_executed_action_difference_step": first,
        "identical_preceding_actions": preceding, "max_initial_state_error": initial_error,
        "physical_observations_equal_through_first_change": states_match,
        "camera_hashes_equal_through_first_change": cameras_match,
        "preceding_transitions_equal": after_match,
        "equal_lengths_if_actions_never_change": equal_lengths_if_no_change,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    batch = json.loads((args.batch / "summary.json").read_text())
    plan = json.loads((args.batch / "plan.json").read_text())
    assert batch["status"] == "complete", "Inspect and account for incomplete jobs before evaluating"
    assert sha256(plan["validation"]) == plan["validation_sha256"]
    declared = {j["name"]: j["plan"] for j in plan["jobs"]}
    assert len(declared) == len(plan["jobs"]) == len(batch["jobs"])
    assert set(declared) == {j["name"] for j in batch["jobs"]}
    directories = [Path(j["directory"]) for j in batch["jobs"]]
    subprocess.run([
        sys.executable, str(ROOT / "experiments/scripts/unisafe_adaptation_analyze.py"),
        *[str(p.resolve()) for p in directories], "--output", str((args.output / "audit").resolve()),
        "--baseline", "baseline",
    ], cwd=ROOT, check=True)
    groups, paths, artifacts, reference_provenance = {}, {}, {}, None
    for directory in directories:
        run = json.loads((directory / "summary.json").read_text())
        assert len(run["episodes"]) == 1, "This evaluation expects fresh processes per episode"
        assert json.loads((directory / "plan.json").read_text()) == declared[directory.name]
        assert run["episodes"][0]["job"] == declared[directory.name]["episodes"][0]
        if reference_provenance is None:
            reference_provenance = run["model_provenance"]
        assert run["model_provenance"] == reference_provenance
        for relative, expected in plan["frozen_sources_sha256"].items():
            assert sha256(directory / "implementation" / relative) == expected, (directory, relative)
        e = run["episodes"][0]
        method, seed = e["job"]["name"], e["seed"]
        assert seed not in groups.setdefault(method, {})
        groups[method][seed] = e
        paths.setdefault(method, {})[seed] = directory / "episode-0000"
        for relative in ("summary.json", "episode-0000/steps.jsonl", "episode-0000/observations.npz"):
            artifacts[str(directory / relative)] = sha256(directory / relative)
    expected_groups = {c["method"] for c in plan["selected"]} | {"baseline-unfiltered", "evolution-unfiltered"}
    assert set(groups) == expected_groups
    for method, cases in groups.items():
        expected = plan["reserved_seeds"][:32] if method.endswith("-unfiltered") else plan["reserved_seeds"]
        assert set(cases) == set(expected), (method, len(cases), len(expected))
        selected_method = method.removesuffix("-unfiltered")
        selected = next(c for c in plan["selected"] if c["method"] == selected_method)
        for episode in cases.values():
            assert episode["job"]["adapter"] == selected["adapter"]
            assert episode["mode"] == ("unfiltered" if method.endswith("-unfiltered") else "filtered")
    prefixes = []
    for method, cases in groups.items():
        if method == "baseline":
            continue
        reference = "baseline-unfiltered" if method.endswith("-unfiltered") and method != "baseline-unfiltered" else "baseline"
        for seed in cases:
            prefixes.append({
                "method": method, "reference": reference, "seed": seed,
                **matched_prefix(paths[reference][seed], paths[method][seed]),
            })
    contrasts = {}
    for reference in ("baseline", "fixed", "random", "policy_gradient"):
        seeds = sorted(set(groups["evolution"]) & set(groups[reference]))
        name = "evolution-minus-" + reference
        contrasts[name] = {
            "seeds": seeds,
            "safe_success": paired_binary(
                [groups["evolution"][s]["safe_success"] for s in seeds],
                [groups[reference][s]["safe_success"] for s in seeds],
            ),
            "physical_failure": paired_binary(
                [groups["evolution"][s]["physical_failure"] for s in seeds],
                [groups[reference][s]["physical_failure"] for s in seeds],
            ),
            "any_violation": paired_binary(
                [groups["evolution"][s]["physical_failure"] or groups["evolution"][s]["boundary_exit"] for s in seeds],
                [groups[reference][s]["physical_failure"] or groups[reference][s]["boundary_exit"] for s in seeds],
            ),
        }
    adjusted = holm({k: v["safe_success"]["mcnemar_exact_two_sided_p"] for k, v in contrasts.items()})
    for k, value in adjusted.items():
        contrasts[k]["safe_success"]["holm_adjusted_p_four_success_contrasts"] = value
    seeds = sorted(groups["baseline-unfiltered"])
    filtered, unfiltered = [], []
    for s in seeds:
        filtered.append(int(groups["evolution"][s]["safe_success"]) - int(groups["baseline"][s]["safe_success"]))
        unfiltered.append(int(groups["evolution-unfiltered"][s]["safe_success"]) - int(groups["baseline-unfiltered"][s]["safe_success"]))
    interaction = np.asarray(filtered) - np.asarray(unfiltered)
    rng = np.random.default_rng(7753)
    draws = rng.choice(interaction, (10000, len(interaction)), replace=True).mean(axis=1)
    audit = json.loads((args.output / "audit/summary.json").read_text())
    result = {
        "status": "complete", "groups": audit["groups"], "contrasts": contrasts,
        "prefix_audit_all_pass": all(p["pass"] for p in prefixes), "prefix_checks": prefixes,
        "filter_interaction": {
            "seeds": seeds, "filtered_adaptation_gain": float(np.mean(filtered)),
            "unfiltered_adaptation_gain": float(np.mean(unfiltered)),
            "difference_of_gains": float(interaction.mean()),
            "descriptive_paired_bootstrap95": np.quantile(draws, [.025, .975]).tolist(),
            "limits": "Exploratory interaction contrast on 32 paired cases; bootstrap interval is descriptive, especially at boundary outcomes.",
        },
        "selected_candidates": plan["selected"], "model_provenance": reference_provenance,
        "test_environment_steps": audit["environment_steps"],
        "test_physics_steps": audit["physics_steps"], "test_model_queries": audit["model_queries"],
        "batch_elapsed_seconds": batch["elapsed_seconds"],
        "analysis_elapsed_seconds": time.perf_counter() - started,
        "artifact_sha256": artifacts,
        "limits": "Reserved evaluation within this migrated simulator. Original checkpoint training overlap is unknown. Exact paired statistics assume independent sampled initial-condition seeds. No arbitrary simulator snapshot or real-world safety guarantee. Audit failures must be investigated before causal claims.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Matched prefix audit:", result["prefix_audit_all_pass"])
    for name, value in contrasts.items():
        print(name, value["safe_success"])


if __name__ == "__main__":
    main()
