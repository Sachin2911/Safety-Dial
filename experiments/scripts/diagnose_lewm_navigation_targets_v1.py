import collections
import json
from pathlib import Path
import numpy as np

base = Path("/workspace/Safety-Dial")
out = base / "results/navigation-target-pilot-small-v1"
s = json.loads((out / "state.json").read_text())
with np.load(base / "results/navigation-memory-cache-v3/memory.npz", allow_pickle=False) as z:
    targets = z["successor_xy"].copy()
records = []
for case_id in ("tworoom-01000-none", "tworoom-01017-circle3", "tworoom-01027-circle7"):
    for row in s["jobs"].values():
        if row["case_id"] != case_id:
            continue
        key = row["job_id"]
        folder = out / "jobs" / key / "episode"
        result = json.loads((folder / "result.json").read_text())
        history = json.loads((folder / "selector-history.json").read_text())
        with np.load(folder / "trace.npz", allow_pickle=False) as z:
            states = z["states"].copy()
            goal = z["original_goal"].copy()
        observed = [r for r in history if r["target_kind"] == "observed_successor"]
        distances = [
            float(np.linalg.norm(targets[r["memory_index"]] - r["current_xy"])) for r in observed
        ]
        ids = [r["memory_index"] for r in observed]
        runs = []
        for i in ids:
            if runs and runs[-1][0] == i:
                runs[-1][1] += 1
            else:
                runs.append([i, 1])
        details = []
        for r in history[:3] + history[-3:]:
            idx = r["memory_index"]
            point = goal if idx is None else targets[idx]
            details.append(
                {
                    "query": r["query"],
                    "kind": r["target_kind"],
                    "index": idx,
                    "current_xy": r["current_xy"],
                    "target_xy": point.tolist(),
                    "target_distance": float(np.linalg.norm(point - r["current_xy"])),
                    "target_original_goal_distance": float(np.linalg.norm(point - goal)),
                    "selected_features": r["selected_features"],
                }
            )
        records.append(
            {
                "case_id": case_id,
                "method": row["method"],
                "initial_xy": states[0].tolist(),
                "original_goal_xy": goal.tolist(),
                "final_xy": states[-1].tolist(),
                "initial_goal_distance": result["initial_original_goal_distance"],
                "final_goal_distance": result["final_original_goal_distance"],
                "physical_progress": result["progress"],
                "actions": result["recorded_successful_actions"],
                "path_length": result["protected_body_path_length"],
                "queries": len(history),
                "observed_targets": len(observed),
                "direct_targets": len(history) - len(observed),
                "observed_target_distance_median": float(np.median(distances))
                if distances
                else None,
                "observed_target_distance_min": min(distances) if distances else None,
                "observed_target_distance_max": max(distances) if distances else None,
                "observed_targets_over30px": sum(d > 30 for d in distances),
                "unique_observed_targets": len(set(ids)),
                "maximum_repeated_target_run": max([n for _, n in runs], default=0),
                "most_common_targets": collections.Counter(ids).most_common(3),
                "controller_failure": result["controller_failure"],
                "planning_seconds": result["planning_seconds"],
                "episode_seconds": result["episode_seconds"],
                "nonplanning_seconds": result["episode_seconds"] - result["planning_seconds"],
                "details": details,
            }
        )
report = {
    "stage": "completed SMALL pilot, selected three fixed case IDs",
    "new_simulator_interactions": 0,
    "new_model_queries": 0,
    "records": records,
}
path = base / "results/navigation-target-diagnosis-small-v1.json"
path.write_text(json.dumps(report, indent=2) + "\n")
for r in records:
    print(json.dumps({k: v for k, v in r.items() if k != "details"}))
print(
    "DETAILS",
    json.dumps(
        [
            {"case": r["case_id"], "method": r["method"], "examples": r["details"][:2]}
            for r in records
            if r["method"] != "native"
        ]
    ),
)
