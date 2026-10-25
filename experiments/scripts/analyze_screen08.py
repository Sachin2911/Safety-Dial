"""Describe every task-8 attempt and keep both development stages separate."""

import argparse
import collections
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.screen08_protocol import ARMS, summarize_episode

METRICS = (
    "primary", "benchmark_safe_success", "benchmark_safe_requested", "requested_success",
    "raw_violation", "counted_violation", "unfinished", "engaged", "mechanism_active",
)


def read_events(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            rows.append({"event": "truncated_line"})
    return rows


def wilson(k, n):
    if not n:
        return None
    z = 1.959963984540054
    p, d = k / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0, centre - half), min(1, centre + half)]


def aggregate(episodes):
    counts = {key: sum(bool(e[key]) for e in episodes) for key in METRICS}
    for safe in (True, False):
        for success in (True, False):
            name = ("safe_" if safe else "unsafe_") + ("success" if success else "failure")
            counts[name] = sum((not e["counted_violation"]) == safe
                               and bool(e["row"]["success"]) == success for e in episodes)
    return {"n": len(episodes), "counts": counts,
            "nominal_wilson_95": {key: wilson(counts[key], len(episodes)) for key in METRICS}}


def paired(episodes, arm):
    native = {(e["task"], e["seed"]): e for e in episodes if e["arm"] == "native"}
    pairs = [(e, native[(e["task"], e["seed"])]) for e in episodes
             if e["arm"] == arm and (e["task"], e["seed"]) in native]
    result = {"paired_cells": len(pairs), "metrics": {}}
    for key in METRICS:
        clusters = collections.defaultdict(list)
        for treated, baseline in pairs:
            clusters[treated["task"]].append(int(treated[key]) - int(baseline[key]))
        means = np.asarray([np.mean(clusters[t]) for t in sorted(clusters)])
        if not len(means):
            continue
        rng = np.random.default_rng(90808)
        draws = rng.choice(means, (10000, len(means)), replace=True).mean(axis=1)
        interval = np.quantile(draws, [.025, .975]).tolist()
        result["metrics"][key] = {"difference": float(means.mean()),
                                  "task_clusters": len(means),
                                  "bootstrap_95": interval,
                                  "degenerate_interval": interval[0] == interval[1]}
    return result


def accounting(root):
    attempts = [e for p in root.glob("attempts_*.jsonl") for e in read_events(p)]
    runtime = [e for p in root.glob("episodes/*/screen.jsonl") for e in read_events(p)]
    probes = list(root.glob("episodes/*/reseed.log")) + list(root.glob("logs/*probe*.log"))
    probe_attempts = sum(p.read_text().count("screen08_query_attempt") for p in probes)
    probe_returns = sum(p.read_text().count("screen08_query_returned") for p in probes)
    count = collections.Counter(e["event"] for e in runtime)
    partial_early_controls = 0
    # Initialisation can fail before a screen runtime exists; preserve those rows and
    # report unobserved internal physics rather than claiming zero simulator activity.
    for path in root.glob("episodes/*/eval/**/results.jsonl"):
        for row in read_events(path):
            if row.get("status") != "completed":
                partial_early_controls += int(row.get("steps", 0) or 0)
    return {
        "reserved_starts": sum(e["event"] == "start_reserved" for e in attempts),
        "simulator_launches": sum(e["event"] == "simulator_launch" for e in attempts),
        "validated_episodes": sum(e["event"] == "validated" for e in attempts),
        "policy_controls_returned": count["applied"],
        "policy_controls_attempted": count["command"],
        "warmup_controls_returned": count["warmup_applied"],
        "warmup_controls_attempted": count["warmup_attempt"],
        "scene_reset_calls_attempted": count["scene_reset_attempt"],
        "scene_reset_calls_returned": count["scene_reset_returned"],
        "internal_scene_reset_physics_controls": None,
        "query_attempts": count["query_attempt"] + probe_attempts,
        "query_returns": count["query_returned"] + probe_returns,
        "model_only_query_attempts": probe_attempts,
        "learner_updates": 0,
        "summed_episode_wall_seconds": sum(e["elapsed_s"] for e in attempts
                                           if e["event"] == "simulator_exit"),
        "invalid_row_reported_steps_not_added_twice": partial_early_controls,
        "truncated_journal_lines": count["truncated_line"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/screen08/remote")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/screen08/analysis")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    episodes = [summarize_episode(p.parent)
                for p in sorted(args.root.glob("episodes/*/validation.json"))]
    identities = [e["job"] for e in episodes]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate valid job outcomes")
    stages = {}
    for stage in ("engineering", "stage_a", "stage_b"):
        selected = [e for e in episodes if e["stage"] == stage]
        stages[stage] = {
            "arms": {arm: aggregate([e for e in selected if e["arm"] == arm]) for arm in ARMS},
            "paired": {arm: paired(selected, arm) for arm in ARMS[1:]},
            "per_task": {str(task): {arm: aggregate([e for e in selected
                                                   if e["task"] == task and e["arm"] == arm])
                                     for arm in ARMS}
                         for task in sorted({e["task"] for e in selected})},
            "per_seed": {str(seed): {arm: aggregate([e for e in selected
                                                   if e["seed"] == seed and e["arm"] == arm])
                                     for arm in ARMS}
                         for seed in sorted({e["seed"] for e in selected})},
        }
    result = {"stages": stages, "accounting": accounting(args.root), "episodes": episodes,
              "limits": "Development screen; six task clusters; selected confirmation arms. "
                        "No optimizer advantage, equivalence or held-out generalization claim."}
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"episodes": len(episodes), "accounting": result["accounting"]}))


if __name__ == "__main__":
    main()
