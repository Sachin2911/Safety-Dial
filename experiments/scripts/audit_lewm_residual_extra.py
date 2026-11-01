"""Independent saved-data verification of the completion gate and actual work."""

import argparse
import collections
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.lewm_proposal_pilot import summarize, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    design = json.loads(args.protocol.read_text())
    audit = json.loads((args.out / "audit.json").read_text())
    state = json.loads((args.out / "state.json").read_text())
    if not audit["verified"]:
        raise RuntimeError("Complete trajectory and genotype audit required")
    jobs = state["jobs"]
    zero_checks = []
    for stage in ("diagnostic", "development"):
        if state["phase"] == "influence_gate_failed" and stage == "development":
            continue
        panel = design[stage]
        for case, seed in panel:
            folders = [
                args.out / "episodes" / f"{stage}_{arm}_{case}_s{seed}"
                for arm in ("native", "zero")
            ]
            with (
                np.load(folders[0] / "episode/trace.npz", allow_pickle=False) as native,
                np.load(folders[1] / "episode/trace.npz", allow_pickle=False) as zero,
            ):
                for name in ("states", "actions", "dense", "frames", "goal_image"):
                    if not np.array_equal(native[name], zero[name]):
                        raise RuntimeError("Zero/native saved evidence differs: " + name)
            zero_checks.append({"stage": stage, "case": case, "seed": seed, "exact": True})
    completion_signal = False
    summaries = {}
    if state["phase"] != "influence_gate_failed":
        for arm in ("native", "zero", "untrained", "cma", "random"):
            rows = [jobs[f"selection_{arm}_{case}_s{seed}"] for case, seed in design["selection"]]
            summaries[arm] = summarize(rows)
        if summaries != state["selection"]["summaries"]:
            raise RuntimeError("Saved selection summary differs from episodes")
        cma = summaries["cma"]
        completion_signal = (
            cma["violation"] == 0
            and cma["safe_completion"]
            > max(summaries[k]["safe_completion"] for k in ("native", "zero", "untrained"))
            and cma["engaged"] >= summaries["native"]["engaged"]
        )
        if completion_signal != state["selection"]["development_completion_signal"]:
            raise RuntimeError("Completion gate differs from actual episode outcomes")
        if state["phase"] != (
            "development_signal" if completion_signal else "no_completion_signal"
        ):
            raise RuntimeError("Terminal stage does not follow completion gate")
        if state["selection"]["validation_opened"] or any(
            k.startswith("validation_") for k in jobs
        ):
            raise RuntimeError("Development screen opened validation unexpectedly")
        if len(jobs) != design["maximum_valid_episodes"]:
            raise RuntimeError("Declared development schedule is incomplete")
    work = {}
    for prefix in (
        "cma_g",
        "random_c",
        "development_native_",
        "development_zero_",
        "development_untrained_",
    ):
        rows = [r for k, r in jobs.items() if k.startswith(prefix)]
        totals = collections.Counter()
        for row in rows:
            totals.update(row["resources"])
        work[prefix] = {
            "episodes": len(rows),
            "outcomes": summarize(rows),
            "resources": dict(totals),
        }
    if (
        state["phase"] != "influence_gate_failed"
        and work["cma_g"]["episodes"] != work["random_c"]["episodes"]
    ):
        raise RuntimeError("Unequal candidate/panel evaluation counts")
    value = {
        "verified": True,
        "common_panel_fitness_verified": audit["common_panel_fitness_verified"],
        "completion_gate_independently_verified": True,
        "development_completion_signal": completion_signal,
        "selection_summaries": summaries,
        "zero_equivalence": zero_checks,
        "actual_training_work": work,
        "equal_episode_and_query_caps_not_equal_actual_work": True,
        "fresh_scene_validation_opened": False,
        "ordinary_learning_ran": False,
        "optimizer_advantage_established": False,
        "new_simulator_interactions": 0,
        "new_frozen_model_queries": 0,
    }
    write_json(args.out / "extra-audit.json", value)
    print(json.dumps(value), flush=True)


if __name__ == "__main__":
    main()
