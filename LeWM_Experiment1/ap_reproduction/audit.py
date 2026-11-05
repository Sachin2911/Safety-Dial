#!/usr/bin/env python3
"""Inspect saved AP evidence without importing a model or simulator."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import random

from accounting import validate_boundary_pause


TASKS = ("cube", "pusht", "reacher", "tworoom")
ARMS = ("gaussian_final", "gaussian_observed")
PUBLISHED = {
    "cube": {"gaussian_final": 9, "gaussian_observed": 60},
    "pusht": {"gaussian_final": 3, "gaussian_observed": 89},
    "reacher": {"gaussian_final": 33, "gaussian_observed": 97},
    "tworoom": {"gaussian_final": 2, "gaussian_observed": 60},
}
MAIN_SHA256 = "3243f4ec7c55d349a3f7a73b093664ca7ae695d369f6f718824b54ea47b894b6"
SCALERS_SHA256 = "623d70c2b5e1b4f61ea5f647e0d2c639daa3fbe77f229547e1c52ae6c8e3b02a"


def read_json(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def optimizer_seed(task: str, ordinal: int) -> int:
    base = 26091331 if task == "pusht" else 26091351 + 1000000 * (
        "cube", "reacher", "tworoom"
    ).index(task)
    return base + 1000 * ordinal


def physical_success(task: str, state: dict, goal: list) -> bool:
    if task == "cube":
        position = state["snapshot"]["qpos"][14:17]
        target = [goal[19] / 10 + 0.425, goal[20] / 10, goal[21] / 10]
        return math.dist(position, target) <= 0.04
    if task == "pusht":
        observation = state["observation"]
        if isinstance(observation, dict):
            observation = observation["state"]
        angle = abs(goal[4] - observation[4]) % (2 * math.pi)
        angle = min(angle, 2 * math.pi - angle)
        return math.dist(goal[:4], observation[:4]) < 20 and angle < math.pi / 9
    if task == "reacher":
        positions = state["info"]["qpos"]
        return all(abs(goal[i] - positions[i]) < 0.05 for i in range(2))
    if task == "tworoom":
        physical = state["snapshot"]["state"]
        if isinstance(physical, dict):
            physical = physical["state"]
        return math.dist(goal[:2], physical[:2]) < 16
    raise ValueError(f"Unknown task: {task}")


def wilson(successes: int, count: int) -> list[float]:
    z = 1.959963984540054
    p = successes / count
    divisor = 1 + z * z / count
    center = (p + z * z / (2 * count)) / divisor
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / divisor
    return [center - half, center + half]


def percentile(values: list[float], fraction: float) -> float:
    position = fraction * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def bootstrap(differences: dict[str, list[int]], repeats: int = 10000) -> dict:
    rng = random.Random(20260913)
    samples = {task: [] for task in TASKS}
    macro = []
    for _ in range(repeats):
        estimates = []
        for task in TASKS:
            values = differences[task]
            estimate = sum(values[rng.randrange(len(values))] for _ in values) / len(values)
            samples[task].append(estimate)
            estimates.append(estimate)
        macro.append(sum(estimates) / len(estimates))
    samples["macro"] = macro
    return {
        "resamples": repeats,
        "seed": 20260913,
        "method": "paired query bootstrap independently within task, equal task weights",
        "rng": "Python random.Random; interval algorithm independently implemented",
        "intervals_95": {
            task: [percentile(sorted(values), 0.025), percentile(sorted(values), 0.975)]
            for task, values in samples.items()
        },
    }


def audit(run_roots: list[Path], protocol_path: Path) -> dict:
    protocol = read_json(protocol_path)
    errors = []

    def require(condition, message):
        if not condition:
            errors.append(message)

    require(digest(protocol_path) == MAIN_SHA256, "Original main protocol hash differs")
    indices = {}
    for root in run_roots:
        index_path = root / "artifacts/dataset-index.json"
        if index_path.exists():
            index = read_json(index_path)
            task = index["task"]
            require(task not in indices or indices[task] == index,
                    f"{task}: conflicting dataset episode indices")
            indices[task] = index
    require(set(indices) == set(TASKS), "Include all four preparation run roots for row indices")
    memories = {task: set(protocol["bank_train_episodes"][task]) for task in TASKS}
    expected = {}
    for task in TASKS:
        rows = protocol["studies"][task]["confirm"]
        require(len(rows) == 128, f"{task}: expected 128 frozen queries")
        memory = memories[task]
        require(len(memory) == (16688 if task == "pusht" else 8000),
                f"{task}: original memory coverage differs")
        require(not memory.intersection(row["episode"] for row in rows),
                f"{task}: query episodes overlap memory")
        for ordinal, row in enumerate(rows):
            if task in indices:
                offsets, lengths = indices[task]["ep_offset"], indices[task]["ep_len"]
                start, end = offsets[row["episode"]], (
                    offsets[row["episode"]] + lengths[row["episode"]]
                )
                require(start <= row["global_start"] < row["global_goal"] < end,
                        f"{task}/{row['record_id']}: query rows cross episode boundary")
            for arm in ARMS:
                expected[task, row["record_id"], arm] = ordinal, row

    seen = {}
    starts = defaultdict(dict)
    totals = Counter()
    outcomes = defaultdict(dict)
    group_counts = defaultdict(Counter)
    sources, run_completion, boundary_pauses = [], [], []
    for root in run_roots:
        case_paths = sorted((root / "metrics/cases").glob("*.json"))
        if not case_paths:
            continue
        boundary_pause = None
        try:
            boundary_pause = validate_boundary_pause(root)
            if boundary_pause is not None:
                boundary_pauses.append(boundary_pause)
            run_account = read_json(root / "metrics/accounting.json")
            run_result = read_json(root / "result.json")
            finished_batch = (run_account.get("completed") is True
                              and run_result.get("status") == "completed"
                              and run_result.get("exit_code") == 0)
            require(finished_batch or boundary_pause is not None,
                    f"{root}: incomplete batch has no validated clean boundary pause")
            require(run_account.get("completed_outcomes") == len(case_paths),
                    f"{root}: saved case count differs from batch accounting")
            run_completion.append({"root": str(root), "original_result": run_result,
                                   "batch_completed": run_account.get("completed"),
                                   "saved_cases": len(case_paths),
                                   "validated_boundary_pause": boundary_pause is not None})
        except (OSError, ValueError, KeyError, TypeError) as error:
            require(False, str(error))
        original_path = root / "artifacts/protocol/main.json"
        require(original_path.exists() and digest(original_path) == MAIN_SHA256,
                f"{root}: original protocol is absent or changed")
        scalers = root / "artifacts/protocol/scalers.json"
        require(scalers.exists() and digest(scalers) == SCALERS_SHA256,
                f"{root}: scaler is absent or changed")
        derived_path = root / "artifacts/protocol/derived.json"
        require(derived_path.exists(), f"{root}: derived protocol is absent")
        if derived_path.exists():
            expected_derived = dict(protocol, arms=list(ARMS), conditions=["clean"])
            require(read_json(derived_path) == expected_derived,
                    f"{root}: derived protocol changed more than arm/condition selection")
        provenance_path = root / "artifacts/provenance.json"
        require(provenance_path.exists(), f"{root}: provenance is absent")
        if provenance_path.exists():
            provenance = read_json(provenance_path)
            require(provenance["ap_revision"] == "984327424ca7a2e52c75a8279197750d9d0b7c8e",
                    f"{root}: AP source revision differs")
            require(provenance["native_protocol_sha256"] == MAIN_SHA256,
                    f"{root}: native protocol provenance differs")
            require(provenance["native_scalers_sha256"] == SCALERS_SHA256,
                    f"{root}: native normalization provenance differs")
        for path in case_paths:
            trace = read_json(path)
            compact = trace["compact"]
            task, arm = compact["task"], compact["arm"]
            identity = compact["identity"]
            key = task, identity["record_id"], arm
            label = "/".join(key)
            require(key in expected, f"{label}: unexpected case")
            require(key not in seen, f"{label}: duplicate evidence")
            if key not in expected or key in seen:
                continue
            seen[key] = str(path)
            ordinal, expected_row = expected[key]
            require(identity == expected_row, f"{label}: frozen identity differs")
            require(compact["query_ordinal"] == ordinal, f"{label}: ordinal differs")
            require(compact["optimizer_seed"] == optimizer_seed(task, ordinal),
                    f"{label}: optimizer seed differs")
            require(compact["condition"] == "clean", f"{label}: unexpected condition")
            require(compact["completed"] is True, f"{label}: incomplete evaluator record")
            require(compact["horizon"] == expected_row["horizon"],
                    f"{label}: recorded goal offset differs")
            budget = protocol["studies"][task]["budget"]
            require(compact["budget"] == budget, f"{label}: action allowance differs")
            actions, decisions = trace["actions"], trace["decisions"]
            require(len(actions) == compact["primitive_steps"] <= budget,
                    f"{label}: action accounting differs")
            require(len(decisions) == compact["decisions"] == math.ceil(len(actions) / 5),
                    f"{label}: decision accounting differs")
            require(compact["predictor_calls"] == len(decisions) * 31,
                    f"{label}: predictor call accounting differs")
            require(compact["predictor_candidate_blocks"] == len(decisions) * 9002,
                    f"{label}: predictor block accounting differs")
            executed = 0
            for decision in decisions:
                require(decision["t"] == executed, f"{label}: decision time differs")
                require(decision["remaining"] == max(5, expected_row["horizon"] - executed),
                        f"{label}: adaptive retrieval span differs")
                support = decision["retrieved_sources"]
                require(len(support) == 8, f"{label}: retrieval support count differs")
                if task in indices:
                    offsets = indices[task]["ep_offset"]
                    lengths = indices[task]["ep_len"]
                    memory = memories[task]
                    span = decision["remaining"]
                    for source in support:
                        episode = bisect_right(offsets, source) - 1
                        require(episode in memory, f"{label}: retrieval outside memory episodes")
                        require(0 <= episode < len(offsets) and source >= offsets[episode]
                                and source + span < offsets[episode] + lengths[episode],
                                f"{label}: retrieval target crosses episode boundary")
                target = support[0] + 5 if arm == "gaussian_observed" else identity["global_goal"]
                require(decision["target_index"] == target,
                        f"{label}: target row differs from prescribed target")
                require(decision["target_kind"] == arm, f"{label}: target kind differs")
                require(decision["predictor_calls"] == 31,
                        f"{label}: decision predictor call count differs")
                require(decision["predictor_candidate_blocks"] == 9002,
                        f"{label}: decision predictor block count differs")
                require(1 <= decision["executed_actions"] <= 5,
                        f"{label}: invalid executed block length")
                executed += decision["executed_actions"]
            require(executed == len(actions), f"{label}: decision action total differs")

            trajectory_path = root / "metrics/trajectories" / f"{path.stem}.json.gz"
            require(trajectory_path.exists(), f"{label}: missing physical trajectory")
            if not trajectory_path.exists():
                continue
            trajectory = read_json(trajectory_path)
            initial, steps = trajectory["initial"], trajectory["steps"]
            counts = trajectory["counts"]
            for name in ("resets_attempted", "resets_completed"):
                require(counts.get(name, 0) == 1, f"{label}: {name} differs")
            for name in ("policy_actions_attempted", "policy_actions_completed"):
                require(counts.get(name, 0) == len(actions), f"{label}: {name} differs")
            require(counts.get("predictor_calls", 0) == compact["predictor_calls"],
                    f"{label}: measured predictor calls differ")
            require(counts.get("predictor_candidate_blocks", 0)
                    == compact["predictor_candidate_blocks"],
                    f"{label}: measured predictor block count differs")
            require(counts.get("retrieval_queries", 0) == len(decisions),
                    f"{label}: measured retrieval query count differs")
            goal = initial["true_goal"]
            require(len(steps) == len(actions), f"{label}: physical step count differs")
            scored = [physical_success(task, initial, goal)]
            for index, step in enumerate(steps):
                if index < len(actions):
                    require(step["action"] == actions[index],
                            f"{label}: executed action differs at {index}")
                scored.append(physical_success(task, step, goal))
            require(bool(compact["success"]) == scored[-1],
                    f"{label}: independent physical success disagrees")
            require(not any(scored[:-1]), f"{label}: continued after physical success")
            require(bool(compact["prelude"]["success"]) == scored[0],
                    f"{label}: initial success accounting differs")
            require(bool(compact["prelude"]["initial_success"]) == scored[0],
                    f"{label}: native initial-success flag differs")
            require(compact["prelude"]["primitive_steps"] == 0,
                    f"{label}: clean query contains prefix actions")
            require(not any(step["terminated"] or step["truncated"] for step in steps[:-1]),
                    f"{label}: continued after simulator termination")
            if not scored[-1] and len(actions) < budget:
                require(bool(steps and (steps[-1]["terminated"] or steps[-1]["truncated"]))
                        or bool(compact["prelude"]["terminal_without_success"]),
                        f"{label}: stopped early without terminal event")
            starts[task, identity["record_id"]][arm] = {
                name: initial[name] for name in ("observation", "snapshot", "true_goal")
            }
            render_hashes = trajectory["render_hashes"]
            require(bool(render_hashes) and render_hashes[0]["action_index"] == 0,
                    f"{label}: initial rendered observation missing")
            starts[task, identity["record_id"]][arm]["goal_rgb_sha256"] = initial[
                "goal_rgb_sha256"
            ]
            if render_hashes:
                starts[task, identity["record_id"]][arm]["initial_rgb_sha256"] = (
                    render_hashes[0]["sha256"]
                )
            totals["evaluated_episodes"] += 1
            totals["policy_actions"] += len(actions)
            totals["decisions"] += len(decisions)
            totals["predictor_calls"] += compact["predictor_calls"]
            totals["predictor_candidate_blocks"] += compact["predictor_candidate_blocks"]
            totals["initially_successful"] += scored[0]
            totals["engaged"] += bool(actions)
            totals["successes"] += scored[-1]
            totals["unfinished"] += not scored[-1]
            group = group_counts[task, arm]
            group["initial_successes"] += scored[0]
            group["engaged"] += bool(actions)
            group["policy_actions"] += len(actions)
            group["predictor_calls"] += compact["predictor_calls"]
            group["predictor_candidate_blocks"] += compact["predictor_candidate_blocks"]
            outcomes[task, arm][ordinal] = int(scored[-1])
            sources.append({"trace": str(path), "trace_sha256": digest(path),
                            "trajectory": str(trajectory_path),
                            "trajectory_sha256": digest(trajectory_path)})

    missing = sorted(set(expected) - set(seen))
    require(not missing, f"Missing {len(missing)} of {len(expected)} required cases")
    for key, pair in starts.items():
        require(len(pair) == 2, f"{key}: missing paired trajectory")
        if len(pair) == 2:
            require(pair[ARMS[0]] == pair[ARMS[1]], f"{key}: paired initial states differ")
    require(totals["policy_actions"] <= 238080, "Scientific action limit exceeded")
    require(totals["predictor_candidate_blocks"] <= 428639232,
            "Scientific predictor block limit exceeded")
    summary = {}
    differences = {}
    for task in TASKS:
        summary[task] = {}
        for arm in ARMS:
            values = outcomes[task, arm]
            count, successes = len(values), sum(values.values())
            if count:
                delta = successes / count - PUBLISHED[task][arm] / 128
                summary[task][arm] = {
                    "episodes": count, "successes": successes, "unfinished": count - successes,
                    **dict(group_counts[task, arm]),
                    "success_rate": successes / count, "wilson_95": wilson(successes, count),
                    "paper_successes": PUBLISHED[task][arm], "paper_episodes": 128,
                    "difference_from_paper": delta,
                    "within_10_percentage_points": abs(delta) <= 0.10,
                    "meets_practical_tolerance": delta >= -0.10,
                    "practical_tolerance_applies": arm == "gaussian_observed",
                }
        if all(len(outcomes[task, arm]) == 128 for arm in ARMS):
            differences[task] = [outcomes[task, ARMS[1]][i] - outcomes[task, ARMS[0]][i]
                                 for i in range(128)]
            summary[task]["paired_difference"] = sum(differences[task]) / 128
    result = {
        "passed": not errors, "errors": errors, "missing_cases": missing,
        "scope": "four tasks, original 128 queries, clean starts, two target-matched CEM arms",
        "scientific_totals": dict(totals), "summary": summary,
        "run_completion": run_completion, "validated_boundary_pauses": boundary_pauses,
        "practical_tolerance": {
            "criterion": "AP-CEM success rate no more than 10 percentage points below publication",
            "applies_to": "gaussian_observed; final-goal CEM is reported as a paired control",
            "control_fidelity": "Report the control's signed paper difference separately",
            "validity_gate": False,
        },
        "provenance": {"protocol": str(protocol_path), "sha256": digest(protocol_path)},
        "limitations": [
            "Published rates are external results; agreement is not statistical equivalence.",
            "Original fixed queries are not new independent query or seed replication.",
            "Preparation, engineering, failed attempts and video work need separate ledgers.",
            "This audit checks saved physical states; it does not execute or rerun simulations.",
        ],
        "evidence_files": sources,
    }
    if len(differences) == len(TASKS):
        result["paired_bootstrap"] = bootstrap(differences)
        result["macro_paired_difference"] = sum(map(sum, differences.values())) / 512
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--protocol", type=Path, required=True,
                        help="Original unmodified authors' protocols/main.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run_root, args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": result["passed"], "errors": len(result["errors"]),
                      "scientific_totals": result["scientific_totals"]}))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
