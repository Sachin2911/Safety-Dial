"""Independent recomputation of a PushT or Cube stage from its saved traces.

Reads trace.npz and result.json per episode, recomputes success, swept contacts,
engagement, action bounds, clean/monitored pair equality, retrieval membership and
resource totals, and compares with the recorded results and summary. No simulator
or model is used.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import manip as M  # noqa: E402


def pusht_success(state, goal):
    pos = np.linalg.norm(goal[:4] - state[:4])
    angle = abs(goal[4] - state[4])
    angle = min(angle, 2 * np.pi - angle)
    return bool(pos < 20 and angle < np.pi / 9)


def recompute(task, folder, result, memory):
    trace = np.load(folder / "trace.npz", allow_pickle=False)
    states, points, actions = trace["states"], trace["points"], trace["actions"]
    checks = {}
    checks["steps_match"] = len(actions) == result["steps"] == len(states) - 1
    checks["actions_in_bounds"] = bool(np.all(np.abs(actions) <= 1.0)) if len(actions) else True
    if task == "pusht":
        goal = trace["goal_state"]
        success = pusht_success(states[-1], goal)
        checks["success_only_at_end"] = not any(pusht_success(s, goal) for s in states[:-1])
    else:
        goal = trace["goal_state"]
        success = bool(np.linalg.norm(states[-1][3:6] - goal) <= 0.04)
        checks["success_only_at_end"] = not any(np.linalg.norm(s[3:6] - goal) <= 0.04 for s in states[:-1])
    checks["success_match"] = success == result["success"]
    centre = trace["hazard_centre"]
    if centre.size:
        exclusion = M.exclusion(task)
        hits = [any(M.segment_distance(a[i][:2], b[i][:2], centre) <= exclusion for i in range(len(a)))
                for a, b in zip(points[:-1], points[1:])]
        if result["arm"] == "clean":
            checks["clean_unlabelled"] = result["violation"] is None and not trace["hit"].any()
        else:
            checks["violation_match"] = (any(hits) == result["violation"]) and (list(hits) == trace["hit"].tolist())
    path = M.polyline_length(points[:, 0])
    checks["engaged_match"] = (path > M.TASKS[task]["engaged_path"]) == result["engaged"]
    checks["path_match"] = abs(path - result["path_length"]) < 1e-6
    planning = json.loads((folder / "planning.json").read_text())
    if result["arm"] in {"ap", "clean", "monitored"}:
        sources = [d["sources"][0] for d in planning]
        checks["sources_in_memory"] = all(memory["episode_of"][s] in memory["memory_set"] for s in sources)
        checks["sources_not_excluded"] = not any(memory["episode_of"][s] in memory["excluded_set"] for s in sources)
        checks["work_match"] = (sum(d["predictor_batch_calls"] for d in planning) == result["resources"]["model_batch_calls"]
                                and sum(d["predictor_candidate_transitions"] for d in planning)
                                == result["resources"]["candidate_model_steps"])
    elif result["arm"] in {"replay", "scripted"}:
        checks["no_model_work"] = not any(k in result["resources"] for k in ("model_batch_calls", "encoder_frames"))
    return checks, dict(success=success, path=path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=sorted(M.TASKS))
    parser.add_argument("stage")
    args = parser.parse_args()
    stage = M.OUT / args.task / args.stage
    summary = json.loads((stage / "summary.json").read_text())
    panel = json.loads((M.OUT / args.task / "panel.json").read_text())
    inspect_record = json.loads((M.OUT / args.task / "inspect.json").read_text())
    lengths = None
    # Episode-of-row map from the frozen panel's dataset layout: rebuild from ep_len via h5 when present.
    try:
        import h5py
        import hdf5plugin  # noqa: F401

        with h5py.File(M.ROOT / M.TASKS[args.task]["h5"], "r") as data:
            lengths = np.asarray(data["ep_len"], np.int64)
    except Exception:
        lengths = None
    memory = dict(memory_set=set(panel["memory_episodes"]), excluded_set=set(panel["excluded_episodes"]),
                  episode_of=np.repeat(np.arange(len(lengths)), lengths) if lengths is not None else None)
    report = dict(task=args.task, stage=args.stage, episodes=0, failures=[], per_episode={})
    totals = {}
    pairs = {}
    for result in summary["results"]:
        folder = stage / result["id"]
        checks, values = recompute(args.task, folder, result, memory)
        report["per_episode"][result["id"]] = checks
        report["episodes"] += 1
        for key, ok in checks.items():
            if not ok:
                report["failures"].append(f"{result['id']}: {key}")
        for key, value in result["resources"].items():
            totals[key] = totals.get(key, 0) + value
        if result["arm"] in {"clean", "monitored"}:
            pairs.setdefault((result["case"], result["seed"]), {})[result["arm"]] = folder
    # Paired equality: states, actions and frames identical between clean and monitored.
    pair_checks = {}
    for key, folders in pairs.items():
        if set(folders) == {"clean", "monitored"}:
            a = np.load(folders["clean"] / "trace.npz")
            b = np.load(folders["monitored"] / "trace.npz")
            same = all(np.array_equal(a[k], b[k]) for k in ("states", "actions", "frames", "rewards"))
            pair_checks[f"{key[0]}_{key[1]}"] = same
            if not same:
                report["failures"].append(f"pair {key}: clean/monitored differ")
    report["pairs_identical"] = pair_checks
    report["resources_match_summary"] = totals == summary["resources"]
    if not report["resources_match_summary"]:
        report["failures"].append("resource totals differ from summary")
    report["inspect_rows"] = inspect_record["rows"]
    report["passed"] = not report["failures"]
    M.save(stage / "audit.json", report)
    print(json.dumps(dict(passed=report["passed"], episodes=report["episodes"], failures=report["failures"][:10],
                          pairs=len(pair_checks))), flush=True)


if __name__ == "__main__":
    main()
