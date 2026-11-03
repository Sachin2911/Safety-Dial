"""Account for every completed and interrupted episode in the flawed first attempt."""

import collections
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/adapter-pilot-20261101/run-v1"


def main():
    summary = json.loads((OUT / "summary.json").read_text())
    totals = collections.Counter()
    statuses = collections.Counter()
    trace_actions = 0
    complete = []
    for folder in sorted((OUT / "episodes").iterdir()):
        execution = json.loads((folder / "execution.json").read_text())
        statuses[execution["status"]] += 1
        totals.update(execution["resources"])
        with np.load(folder / "trace.npz") as trace:
            actions, states = trace["actions"], trace["states"]
            trace_actions += len(actions)
            assert len(actions) <= execution["resources"].get("environment_actions", 0)
            assert len(states) in [len(actions), len(actions) + 1]
            if execution["status"] == "complete":
                row = json.loads((folder / "result.json").read_text())
                native_hits = np.linalg.norm(states[1:] - trace["goal"], axis=1) < 16
                np.testing.assert_array_equal(native_hits, trace["terminated"])
                assert row["success"] == bool(native_hits[-1])
                assert len(actions) == execution["resources"]["environment_actions"]
                complete.append(row)
    assert dict(totals) == summary["resources"]
    assert totals["environment_resets"] == 459 and statuses["complete"] == 451
    assert statuses["incomplete"] == 8 and totals["environment_actions"] == 58179
    assert not any(r["stage"] == "evaluation" for r in complete)
    report = dict(
        accounted=True,
        optimizer_comparison_valid=False,
        reason="CMA-ES TPA paired-sample order was changed before tell",
        resources=dict(totals),
        execution_statuses=dict(statuses),
        completed_episode_files=len(complete),
        coordinator_collected_results=len(summary["results"]),
        completed_not_collected=len(complete) - len(summary["results"]),
        traced_actions=trace_actions,
        charged_actions_without_trace=totals["environment_actions"] - trace_actions,
        seconds=summary["seconds"],
        final_evaluation_episodes=0,
    )
    (OUT / "interruption-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
