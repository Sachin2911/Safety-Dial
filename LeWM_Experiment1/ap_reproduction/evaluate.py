"""Run the exact AP evaluator on its clean, original long-range paired panels."""

import argparse
from collections import Counter
import copy
import hashlib
import os
from pathlib import Path
import time
from types import SimpleNamespace

from common import (Ledger, TASKS, assets, configuration, configure, provenance, publish,
                    read, sha, write)
from engineering import instrumentation_hashes
from instrumentation import Observer, Recorder, publish_failure
from run_io import publish_artifact, report_progress


def main(task, shards, shard_index):
    config = configuration()
    ap_root, runtime = configure(config, task)
    runtime.check_simulator_versions(task)
    if config["conditions"] != ["clean"]:
        raise ValueError("This prospective study authorizes the clean panel only")
    if config["arms"] != ["gaussian_final", "gaussian_observed"]:
        raise ValueError("Unexpected scientific arms or paired arm order")
    if shards < 1 or not 0 <= shard_index < shards:
        raise ValueError("Invalid deterministic query sharding")
    gate = read(Path(config["asset_root"]) / "gates" / (task + ".json"))
    if not gate["passed"] or gate["source_hashes"] != instrumentation_hashes():
        raise ValueError("Engineering parity gate is absent or instrumentation changed")
    if gate["versions"] != runtime.versions():
        raise ValueError("Runtime changed after the engineering parity gate")
    asset = assets(config, task)
    cache = read(asset["latent_cache"].with_suffix(".receipt.json"))
    if sha(asset["latent_cache"]) != cache["sha256"]:
        raise ValueError("Retrieval cache differs from its published receipt")
    if sha(asset["checkpoint"]) != cache["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the frozen retrieval cache")
    run_root = Path(os.environ["LEWM_RUN_DIR"])
    scratch = run_root / ".control/native"
    derived = scratch / "derived"
    original = read(ap_root / "protocols/main.json")
    protocol = copy.deepcopy(original)
    protocol["conditions"], protocol["arms"] = config["conditions"], config["arms"]
    write(derived / "protocols/main.json", protocol)
    scalers = read(ap_root / "protocols/scalers.json")
    write(derived / "protocols/scalers.json", scalers)
    publish_artifact("artifacts/protocol/main.json", source=ap_root / "protocols/main.json")
    publish("artifacts/protocol/derived.json", protocol)
    publish_artifact("artifacts/protocol/scalers.json", source=ap_root / "protocols/scalers.json")
    publish("artifacts/engineering-gate.json", gate)
    publish("artifacts/cache-receipt.json", cache)
    publish("artifacts/provenance.json", provenance(config, ap_root))
    ledger = Ledger(config)
    observer = Observer(config, task)
    observer.install()
    from anchored_planning import environments
    import h5py
    import hdf5plugin  # noqa: F401
    import numpy as np

    native_make = environments._make_live
    selected = protocol["studies"][task]
    cases = [(ordinal, row, condition, arm)
             for ordinal, row in enumerate(selected["confirm"]) if ordinal % shards == shard_index
             for condition in protocol["conditions"] for arm in protocol["arms"]]
    cursor, outcomes = 0, []
    active = None
    data = h5py.File(asset["dataset"], "r")

    def make(context, row):
        nonlocal cursor, active
        observer.guard()
        ordinal, expected, condition, arm = cases[cursor]
        if row != expected:
            raise AssertionError("Native query order changed")
        case = f"{row['record_id']}_{condition}_{arm}"
        identity = f"scientific:{task}:{case}"
        max_decisions = (selected["budget"] + 4) // 5
        ledger.reserve(identity, "scientific", resets=1, policy_actions=selected["budget"],
                       predictor_calls=max_decisions * 31,
                       predictor_candidate_blocks=max_decisions * 9002)
        cursor += 1
        observer.case, observer.phase = Counter(), "setup"
        observer.case_limits = {"policy_actions_attempted": selected["budget"],
                                "predictor_calls": max_decisions * 31,
                                "predictor_candidate_blocks": max_decisions * 9002}
        observer.add("resets_attempted")
        raw_env = None
        try:
            raw_env, obs, info, goal = native_make(context, row)
            observer.add("resets_completed")
            initial = {"identity": row, "query_ordinal": ordinal, "condition": condition,
                       "arm": arm, "observation": obs, "info": info, "true_goal": goal,
                       "action_budget": selected["budget"]}
            goal_image = np.asarray(data["pixels"][row["global_goal"]])
            initial["goal_rgb_sha256"] = hashlib.sha256(goal_image.tobytes()).hexdigest()

            def finished(trace, video_error):
                nonlocal active
                path = scratch / "traces" / task / (case + ".json")
                complete_case = False
                try:
                    if path.exists():
                        native = read(path)
                        compact = native["compact"]
                        if native["actions"] != [s["action"] for s in trace["steps"]]:
                            raise AssertionError("Recorded actions differ from native actions")
                        if compact["predictor_calls"] != trace["counts"].get("predictor_calls", 0):
                            raise AssertionError("Native and instrumented predictor counts differ")
                        publish_artifact(f"metrics/cases/{case}.json", source=path)
                        outcomes.append(compact)
                        report_progress(len(outcomes), len(cases), unit="paired outcomes")
                        complete_case = video_error is None
                finally:
                    ledger.finish(identity, trace["counts"], complete_case)
                    observer.case = None
                    active = None

            active = Recorder(raw_env, task, case, initial, goal_image, observer, scratch,
                              config, on_close=finished)
            return active, obs, info, goal
        except Exception:
            if raw_env is not None:
                raw_env.close()
            ledger.finish(identity, observer.case, False)
            observer.case = None
            raise

    observer.patch(environments, "_make_live", make)
    completed = False
    try:
        args = SimpleNamespace(task=task, study="main", root=derived, output=scratch,
                               shards=shards, shard_index=shard_index, limit=None, device="cuda:0")
        runtime.evaluate(args, asset)
        if len(outcomes) != len(cases) or cursor != len(cases):
            raise AssertionError("Incomplete original cohort")
        for path in scratch.glob("*.json*"):
            publish_artifact("metrics/native/" + path.name, source=path)
        groups = {arm: [r for r in outcomes if r["arm"] == arm] for arm in config["arms"]}
        summary = {"task": task, "shards": shards, "shard_index": shard_index,
                   "completed": True, "outcomes": outcomes, "by_arm": {
                       arm: {"successes": sum(r["success"] for r in rows), "episodes": len(rows),
                             "policy_actions": sum(r["primitive_steps"] for r in rows)}
                       for arm, rows in groups.items()}}
        publish("metrics/summary.json", summary)
        media = []
        for success in (True, False):
            matches = [r for r in groups["gaussian_observed"] if r["success"] == success]
            row = min(matches, key=lambda r: r["query_ordinal"]) if matches else None
            media.append({"outcome": "success" if success else "failure", "selected": row,
                          "rule": "first AP-CEM outcome in original query order within this shard"})
        publish("artifacts/media-selection.json", media)
        completed = True
    except Exception as error:
        publish_failure(observer, error)
        raise
    finally:
        if active is not None:
            active.close()
        data.close()
        observer.close()
        publish("metrics/accounting.json", {"completed": completed, "counts": observer.total,
                "completed_outcomes": len(outcomes), "attempted_outcomes": cursor,
                "elapsed_seconds": time.monotonic() - observer.started})
        publish("metrics/study-attempt-ledger.json", read(ledger.path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    main(args.task, args.shards, args.shard_index)
