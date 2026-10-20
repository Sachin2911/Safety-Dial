"""Profile saved training timings without simulation, encoding or policy requests."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGES = {"development", "shared_initial", "search_generation_1", "search_generation_2"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--sensitivity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path to preserve prior review evidence")
    raw = args.sensitivity.read_bytes()
    source = json.loads(raw)
    episodes, suffixes, receipt_hashes = [], [], {}
    for row in source["episodes"]:
        if row["stage"] not in STAGES:
            raise ValueError("Source report contains a nontraining episode")
        name = row["attempt"]
        if Path(name).name != name:
            raise ValueError("Invalid attempt identifier")
        path = args.run / "episodes" / name / "verified-receipt.json"
        receipt_bytes = path.read_bytes()
        receipt = json.loads(receipt_bytes)
        if receipt["valid"] is not True or receipt["full_episode"] is not True:
            raise ValueError("Training receipt is not a valid full episode")
        ledger = receipt["physical"]["ledger"]
        process = float(row["process_elapsed_seconds"])
        internal = float(ledger["elapsed_seconds"])
        query = float(ledger["policy_query_seconds"])
        load = float(ledger["encoder_load_seconds"])
        encoding = float(ledger["encoder_seconds"])
        if not all(math.isfinite(v) and v >= 0
                   for v in (process, internal, query, load, encoding)):
            raise ValueError("Invalid timing value")
        if not process >= internal >= query + load or process == query:
            raise ValueError("Timing regions do not fit inside the process interval")
        for key in ("policy_query_seconds", "encoder_seconds", "physics_seconds"):
            if row[key] != ledger[key]:
                raise ValueError("Source report disagrees with its stored receipt")
        if row["controls"] != receipt["physical"]["steps"]:
            raise ValueError("Source control count disagrees with its stored receipt")
        receipt_hashes[str(path)] = hashlib.sha256(receipt_bytes).hexdigest()
        episodes.append({
            "attempt": name, "stage": row["stage"], "family": row["family"],
            "controls": row["controls"], "process_wall_seconds": process,
            "internal_ledger_wall_seconds": internal,
            "query_region_wall_seconds": query, "encoder_load_wall_seconds": load,
            "encoder_work_wall_seconds_overlaps_query": encoding,
            "outside_internal_ledger_wall_seconds": process - internal,
            "unattributed_internal_wall_seconds": internal - query - load,
            "query_fraction_of_process": query / process,
            "encoder_work_fraction_of_process": encoding / process,
            "encoder_load_fraction_of_process": load / process,
            "hypothetical_speedup_if_query_region_cost_zero": process / (process - query),
            "physics_simulated_seconds_not_wall": ledger["physics_seconds"],
        })
        first = row["first_doomed_monitor_step"]
        if row["stage"] != "development" and first is not None:
            if (type(first) is not int or not 0 <= first <= row["controls"]
                    or row["score"] != -2 or row["outcome"]["V"] is not True):
                raise ValueError("Post-violation suffix does not have a fixed -2 outcome")
            suffixes.append({"attempt": name, "first_doomed_monitor_step": first,
                             "controls_after_first_doomed": row["controls"] - first})

    groups = {}
    for label, selected in {
        "jar_all": [row for row in episodes if row["family"] == "jar"],
        "jar_training": [row for row in episodes
                         if row["family"] == "jar" and row["stage"] != "development"],
        "clutter_development": [row for row in episodes
                                if row["family"] == "clutter" and row["stage"] == "development"],
    }.items():
        if not selected:
            continue
        fields = [key for key, value in selected[0].items()
                  if isinstance(value, (int, float))]
        groups[label] = {"episodes": len(selected), "distribution": {
            key: {"min": min(row[key] for row in selected),
                  "median": statistics.median(row[key] for row in selected),
                  "max": max(row[key] for row in selected)} for key in fields}}
    training_controls = sum(row["controls"] for row in episodes if row["stage"] != "development")
    suffix_controls = sum(row["controls_after_first_doomed"] for row in suffixes)
    source_paths = [Path(__file__), ROOT / "experiments/helpers/maniguard_runtime.py",
                    ROOT / "experiments/helpers/maniguard_latent_runtime.py",
                    ROOT / "experiments/helpers/maniguard_phase2_features.py",
                    ROOT / "experiments/helpers/maniguard_latent.py",
                    ROOT / "third_party/ManiGuard/maniguard/utils/safety_monitor.py",
                    ROOT / "third_party/ManiGuard/maniguard/eval/benchmark.py"]
    result = {
        "schema": "maniguard-review-timing-v1",
        "sensitivity_source": str(args.sensitivity),
        "sensitivity_sha256": hashlib.sha256(raw).hexdigest(),
        "receipt_sha256": receipt_hashes,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in source_paths},
        "episodes": episodes, "groups": groups,
        "retrospective_training_suffix": {
            "episodes": suffixes, "training_controls": training_controls,
            "controls_after_first_doomed": suffix_controls,
            "fraction_of_recorded_training_controls": (
                suffix_controls / training_controls if training_controls else None),
        },
        "interpretation": [
            "physics_seconds integrates simulation dt and is not wall-clock timing.",
            "The query timer includes image encoding, feature work, query NPZ writing and RPC.",
            "Encoder work overlaps the query timer except for episode text encoding.",
            "The unattributed residual does not separate physics, rendering, monitoring or I/O.",
            "Per-process intervals run concurrently and must not be summed as rental wall time.",
            "Hypothetical elimination of a measured region is not a measured optimization.",
            "Post-doom suffix counts are retrospective, not saved controls or saved wall time.",
            "Early stopping would require a new protocol and valid-prefix receipts.",
            "The current frozen full-horizon protocol and evaluation categories are unchanged.",
        ],
        "new_simulator_controls": 0, "new_policy_queries": 0, "new_encoder_calls": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "training_and_development_episodes": len(episodes),
                      "training_controls": training_controls, "post_doom_suffix_controls": suffix_controls,
                      "new_simulator_controls": 0, "new_policy_queries": 0, "new_encoder_calls": 0}))


if __name__ == "__main__":
    main()
