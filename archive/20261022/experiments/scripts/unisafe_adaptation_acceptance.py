"""Check aperture-based adaptation against the recorded matched causal episode."""

import argparse
import json
from pathlib import Path

import numpy as np


def rows(path):
    return [json.loads(line) for line in (path / "steps.jsonl").read_text().splitlines()]


def compare(reference, candidate, count):
    a, b = rows(reference), rows(candidate)
    for x, y in zip(a[:count], b[:count]):
        for key in ("before", "after", "proposed_action", "executed_normalized_action",
                    "uncertainty", "next_q1", "next_q2", "intervened"):
            assert x[key] == y[key], (candidate, x["step"], key)
    x, y = np.load(reference / "observations.npz"), np.load(candidate / "observations.npz")
    for key in x.files:
        assert np.array_equal(x[key][:count + 1], y[key][:count + 1]), (candidate, key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--manual-hold", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.run / "summary.json").read_text())
    assert summary["status"] == "complete"
    for index, mode in enumerate(("unfiltered", "filtered")):
        compare(args.primary / f"seed-000-{mode}", args.run / f"episode-{index:04d}", 300)
    target = args.run / "episode-0002"
    compare(args.primary / "seed-001-filtered", target, 43)
    target_rows = rows(target)
    assert next(i for i, r in enumerate(target_rows) if r["adapter_trigger"]) == 43
    assert summary["episodes"][2]["outcome"] == "success"
    assert summary["episodes"][2]["steps"] == 56
    compare(args.manual_hold, target, 56)
    result = {
        "status": "pass", "warmup_steps_bitwise_equal": 600,
        "target_preceding_steps_bitwise_equal": 43,
        "sensor_trigger_step": 43, "no_hand_timed_trigger": True,
        "matches_entire_manual_hold_trajectory": True,
        "outcome": "success", "completion_time_s": 2.8,
        "environment_steps": summary["environment_steps"],
        "physics_steps": summary["physics_steps"],
    }
    (args.run / "acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
