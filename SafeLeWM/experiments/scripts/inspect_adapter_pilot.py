"""Additional saved-state validation and descriptive reporting. No new rollouts."""

import collections
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/adapter-pilot-20261101/run-v1"


def summarize(rows):
    counts = {
        key: sum(r[key] for r in rows)
        for key in ["success", "safe_completion", "violation", "unfinished", "engaged"]
    }
    resources = collections.Counter()
    for row in rows:
        resources.update(row["resources"])
    return dict(
        n=len(rows),
        **counts,
        resources=dict(resources),
        correction_norm_mean=float(np.mean([r["correction_norm_mean"] for r in rows])),
    )


def main():
    global OUT
    if len(sys.argv) > 1:
        OUT = Path(sys.argv[1]).resolve()
    summary = json.loads((OUT / "summary.json").read_text())
    assert summary["status"] == "complete"
    rows = summary["results"]
    for generation in range(4):
        with (OUT / f"cma-state-{generation:02d}.pkl").open("rb") as stream:
            state = pickle.load(stream)
        assert type(state.adapt_sigma).__name__ == "CMAAdaptSigmaCSA"
        assert state.opts["CMA_mirrors"] == 0
        metadata = json.loads((OUT / f"cma-generation-{generation:02d}.json").read_text())
        assert metadata["presentation_order"] == list(range(8))
    for row in rows:
        with np.load(OUT / "episodes" / row["id"] / "trace.npz") as trace:
            hit = np.linalg.norm(trace["states"][1:] - trace["goal"], axis=1) < 16
            np.testing.assert_array_equal(trace["terminated"], hit)
            assert not hit[:-1].any() and not trace["truncated"].any()
            assert bool(hit[-1]) == row["success"]
            assert row["success"] or len(trace["actions"]) == 200
            assert not trace["rewards"].any()
    report = dict(
        native_state_audit_passed=True,
        native_state_audit_episodes=len(rows),
        stage_arm={},
        evaluation_case={},
        training_scores={},
        seconds=summary["seconds"],
        resources=summary["resources"],
        supplemental_audit="Added after launch; reads saved evidence only, changes no selection",
        corrected_optimizer_verified=True,
        cumulative_resources=summary["cumulative_resources"],
    )
    for stage in ["engineering", "witness", "training", "evaluation"]:
        report["stage_arm"][stage] = {}
        for arm in sorted({r["arm"] for r in rows if r["stage"] == stage}):
            report["stage_arm"][stage][arm] = summarize(
                [r for r in rows if r["stage"] == stage and r["arm"] == arm]
            )
    for case in sorted({r["case"] for r in rows if r["stage"] == "evaluation"}):
        report["evaluation_case"][case] = {
            arm: summarize(
                [
                    r
                    for r in rows
                    if r["stage"] == "evaluation" and r["case"] == case and r["arm"] == arm
                ]
            )
            for arm in ["native", "untrained", "cma", "random"]
        }
    for arm in ["cma", "random"]:
        candidates = [r for r in summary["candidates"] if r["optimizer"] == arm]
        report["training_scores"][arm] = dict(
            candidates=len(candidates),
            distinct_scores=len({r["score"] for r in candidates}),
            safe_range=[min(r["safe"] for r in candidates), max(r["safe"] for r in candidates)],
            best=max(candidates, key=lambda r: r["score"]),
        )
    (OUT / "inspection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                evaluation=report["stage_arm"]["evaluation"],
                training=report["training_scores"],
                seconds=report["seconds"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
