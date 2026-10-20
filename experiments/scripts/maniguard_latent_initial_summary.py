"""Summarize the complete shared initialization without new physics or model calls."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.maniguard_latent_report import initial_diagnostics


def noise_receipts(output):
    return [row["receipt"] for row in
            (json.loads(line) for line in (output / "trace.jsonl").read_text().splitlines())
            if row["type"] == "noise_receipt"]


def summarize(run):
    queue = json.loads((run / "queue.json").read_text())
    report_path = run / "analysis/summary.json"
    report = json.loads(report_path.read_text())
    state = json.loads((run / "search/state.json").read_text())
    attempts = {row["attempt"]: row for row in report["attempts"]}
    outputs, episodes, groups, diagnostic_rows = {}, [], [], []
    labels = ["native", "positive", "negative", "random-initial"]
    for index, label in enumerate(labels):
        members = []
        for case in range(2):
            job = queue["jobs"][f"initial-{index}-case{case}"]
            if job["status"] != "completed":
                raise ValueError("Finish the shared initialization before exporting its summary")
            attempt = job["attempt_ids"][-1]
            row = {**attempts[attempt], "initial_controller": label, "case": case}
            if row["valid_full_episode"] is not True:
                raise ValueError("Refresh the independently verified analysis before this export")
            if (row["scene"], row["seed"]) != (job["definition"]["scene"], job["definition"]["seed"]):
                raise ValueError("Initial case differs from its immutable job")
            members.append(row)
            episodes.append(row)
            outputs[(label, case)] = Path(job["output"])
            diagnostic_rows.append({**row, "block": case, "condition": label})
        scores = [row["score"] for row in members]
        for method in ("ea", "random"):
            candidate = state["optimizers"][method]["candidates"][index]
            if candidate["id"] != f"initial-{index}" or candidate["scores"] != scores:
                raise ValueError("Methods did not receive the same initial fitness data")
        groups.append({"controller": label, "n": 2,
                       **{category: sum(row["requested_category"] == category for row in members)
                          for category in ("safe_completion", "unsafe_completion",
                                           "safe_unfinished", "unsafe_unfinished")},
                       "violations": sum(row["violation"] for row in members),
                       "contacts": sum(row["ever_contacted_task_object"] for row in members),
                       "grasps": sum(row["ever_grasped_task_object"] for row in members),
                       "score_sum": sum(scores),
                       "policy_controls": sum(row["policy_controls"] for row in members),
                       "policy_queries": sum(row["policy_queries"] for row in members)})
    streams = []
    for case in range(2):
        reference = noise_receipts(outputs[("native", case)])
        for label in labels[1:]:
            observed = noise_receipts(outputs[(label, case)])
            pairs = list(zip(reference, observed, strict=False))
            streams.append({"case": case, "controller": label,
                            "reference_queries": len(reference), "condition_queries": len(observed),
                            "paired_queries": len(pairs),
                            "equal_native_noise_hashes": sum(a["native_noise_sha256"] == b[
                                "native_noise_sha256"] for a, b in pairs),
                            "equal_rng_transitions": sum(a["rng_before"] == b["rng_before"]
                                and a["rng_after"] == b["rng_after"] for a, b in pairs)})
    return {"schema": "maniguard-complete-shared-initial-v1",
            "source_parent_commit": subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "analysis_summary_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "observed_epoch": report["accounting"]["observed_epoch"],
            "shared_initial_complete": True, "physical_comparison_complete": False,
            "counts": groups, "episodes": episodes,
            "initial_observation_diagnostics": initial_diagnostics(
                diagnostic_rows, report["accounting"]["attempts"]),
            "stored_native_stream_diagnostics": streams,
            "global_initial_policy_controls": sum(row["policy_controls"] for row in episodes),
            "global_initial_policy_queries": sum(row["policy_queries"] for row in episodes),
            "scope": "Two shared training cases, not independent checking. Common native noise hashes do not imply identical observations or identify a physical counterfactual. The shared acquisition is counted once globally and attributed once to each method's information budget.",
            "new_physical_controls": 0, "new_model_queries": 0, "new_encoder_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.run.resolve())
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"counts": result["counts"], "stored_native_stream_diagnostics":
                      result["stored_native_stream_diagnostics"]}))


if __name__ == "__main__":
    main()
