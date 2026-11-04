"""Independently reconcile saved episode rows, action traces and model-query counts."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from run_io import publish_artifact

REFERENCES = {"tworoom": 0.87, "reacher": 0.86, "pusht": 0.96, "cube": 0.74}


def read(path):
    return json.loads(path.read_text())


def wilson(successes, count):
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z * z / count
    centre = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [centre - radius, centre + radius]


def audit_panel(run, summary_path):
    summary = read(summary_path)
    if summary["status"] != "completed":
        return {"run": run.name, "status": "incomplete", "summary": summary}
    prefix = summary_path.parent.name
    cases_record = read(run / "artifacts" / prefix / "cases.json")
    episodes = read(summary_path.parent / "episodes.json")
    cases = cases_record["cases"]
    count = summary["episodes"]
    assert len(episodes) == len(cases) == count
    assert cases_record["actual_goal_offset"] == 25
    assert cases_record["evaluator_chunk_length"] == 26
    rows = defaultdict(list)
    for path in sorted(summary_path.parent.glob("steps-*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            rows[row["case"]].append(row)
    for i, (episode, case) in enumerate(zip(episodes, cases)):
        assert all(episode[k] == v for k, v in case.items())
        assert case["goal_step"] - case["start_step"] == 25
        assert [row["step"] for row in rows[i]] == list(range(1, 51))
        assert all(all(math.isfinite(x) for x in row["action"]) for row in rows[i])
        successful_steps = [row["step"] for row in rows[i] if row["terminated"]]
        assert episode["success"] == bool(successful_steps)
        assert episode["first_success_step"] == (successful_steps[0] if successful_steps else None)
    successes = sum(row["success"] for row in episodes)
    assert summary["successes"] == successes
    assert math.isclose(summary["success_rate"], successes / count)
    assert list(map(bool, summary["native_metrics"]["episode_successes"])) == [
        row["success"] for row in episodes]
    counts = summary["interaction_counts"]
    assert counts["evaluation_step_attempts"] == counts["evaluation_step_returns"] == count * 50
    queries = summary["model_queries"]
    expected_calls = count * 2 * summary["iterations"]
    assert queries["cost_calls"] == expected_calls
    assert queries["candidate_rollouts"] == expected_calls * 300
    assert queries["predicted_latent_steps"] == expected_calls * 300 * 5
    assert queries["encoder_frames"] == expected_calls * 2
    receipt = read(run / "artifacts/dataset-provenance.json")
    return {"run": run.name, "panel": prefix, "status": "verified",
            "task": summary["task"], "phase": summary["phase"], "seed": summary["seed"],
            "iterations": summary["iterations"], "successes": successes, "episodes": count,
            "rate": successes / count, "wilson_95": wilson(successes, count),
            "actions": count * 50, "model_queries": queries,
            "elapsed_seconds": summary["elapsed_seconds"],
            "dataset_sha256": receipt["sha256"],
            "restored_image_errors": summary.get("restored_image_mean_absolute_errors", [])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True)
    args = parser.parse_args()
    root = Path("/workspace/LeWM_Experiment1/runs")
    panels = []
    for name in args.run:
        run = root / name
        if run.parent != root or not run.is_dir():
            raise ValueError("Expected an existing run ID")
        for path in sorted((run / "metrics").glob("*/summary.json")):
            panels.append(audit_panel(run, path))
    scientific = [p for p in panels if p.get("phase") == "scientific"]
    expected = {(task, seed, 30 if task == "pusht" else 10)
                for task in REFERENCES for seed in [42, 43, 44]}
    expected |= {(task, 42, 30) for task in REFERENCES if task != "pusht"}
    observed = [(p["task"], p["seed"], p["iterations"]) for p in scientific]
    assert len(observed) == len(set(observed)), "Duplicate scientific panels"
    missing = sorted(expected - set(observed))
    report = {"complete": not missing and all(p["status"] == "verified" for p in panels),
              "missing_panels": missing, "panels": panels,
              "scientific_actions": sum(p["actions"] for p in scientific)}
    text = ["# LeWM frozen-checkpoint reproduction", "",
            "Audit status: " + ("complete" if report["complete"] else "incomplete"), "",
            "Main comparison uses the paper's CEM iterations and a true 25-action goal offset.",
            "The frozen public checkpoints are evaluated without training or safety adaptations.", "",
            "| Environment | Paper | Seed 42 | 95% Wilson interval | Seeds 43 / 44 | Within 10 pp |",
            "| --- | ---: | ---: | --- | --- | --- |"]
    for task, reference in REFERENCES.items():
        primary = [p for p in scientific if p["task"] == task and
                   p["iterations"] == (30 if task == "pusht" else 10)]
        by_seed = {p["seed"]: p for p in primary}
        if 42 not in by_seed:
            text.append(f"| {task} | {reference:.0%} | pending | | | |")
            continue
        p = by_seed[42]
        low, high = p["wilson_95"]
        repeats = " / ".join(f"{by_seed[s]['successes']}/{by_seed[s]['episodes']}"
                             if s in by_seed else "pending" for s in [43, 44])
        consistent = p["rate"] >= reference - 0.10 - 1e-12
        text.append(f"| {task} | {reference:.0%} | {p['successes']}/{p['episodes']} "
                    f"({p['rate']:.0%}) | {low:.1%} to {high:.1%} | {repeats} | "
                    f"{'yes' if consistent else 'no'} |")
    text += ["", "The 10 percentage-point tolerance is an engineering criterion, not an",
             "equivalence test. Intervals are descriptive Bernoulli intervals; repeated",
             "trajectories and a shared checkpoint limit independence. These results concern",
             "nearby-goal competence and do not establish safe or long-horizon completion.", "",
             "## Released shared-config sensitivity", ""]
    for p in scientific:
        if p["task"] != "pusht" and p["iterations"] == 30:
            text.append(f"- {p['task']}, seed 42, CEM 30: {p['successes']}/{p['episodes']} "
                        f"({p['rate']:.0%}).")
    text += ["", "## Evidence", "",
             f"Audited scientific controller actions: {report['scientific_actions']:,}.",
             "Per-panel evidence and model-query counts are in audit.json. Failed and",
             "engineering runs remain separate. See compatibility.md for the API, renderer,",
             "checkpoint-history and exact-goal-index decisions.", ""]
    publish_artifact("artifacts/audit.json", data=json.dumps(report, indent=2) + "\n")
    publish_artifact("artifacts/results.md", data="\n".join(text))
    print(json.dumps({"complete": report["complete"], "missing": missing,
                      "audited_panels": len(panels)}), flush=True)


if __name__ == "__main__":
    main()
