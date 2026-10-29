"""Stage A acceptance and throughput; sizes the B seed panel and the C ladder rung.

Runs locally on the synced copy of the remote outputs. Writes engineering.json, which the
B and C freezes require, and never looks at scored stage outcomes.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_infra import atomic_json, utc
from helpers.safecem10_protocol import design, schedule_A


def load_results(remote):
    rows = {}
    for path in sorted((remote / "attempts").glob("*/result.json")):
        row = json.loads(path.read_text())
        if row["status"] == "complete":
            row["directory"] = str(path.parent)
            rows[row["job"]["id"]] = row
    return rows


def engineering(remote, spec):
    results = load_results(remote)
    jobs = schedule_A(spec)
    missing = [j["id"] for j in jobs if j["id"] not in results]
    rows = [results[j["id"]] for j in jobs if j["id"] in results]
    checks = {}
    checks["all_jobs_complete"] = not missing
    checks["state_counts_in_range"] = all(1 <= r["n_states"] <= 50 for r in rows)
    baseline = [r for r in rows if r["job"]["mode"] == "penalty"]
    per_layout = {}
    for layout in spec["gate"]["layouts"]:
        group = [r for r in baseline if r["job"]["layout"] == layout["id"]]
        per_layout[layout["id"]] = {
            "episodes": len(group),
            "violating_episodes": sum(r["any_violation"] for r in group),
            "mean_exposure": float(np.mean([r["exposure"] for r in group])) if group else None,
            "successes": sum(r["success"] for r in group),
            "crossings": [r["crossing"] for r in group],
            "px_per_step": float(np.median([r["px_per_step"] for r in group])) if group else None,
        }
    checks["baseline_enters_a_gate"] = any(
        v["violating_episodes"] > 0 for v in per_layout.values()
    )
    safe = {r["job"]["clearances"][0]: r for r in rows if r["job"]["mode"] == "safe"}
    first_seed = spec["stages"]["A"]["seeds"][0]
    base_first = next(
        (
            r
            for r in baseline
            if r["job"]["layout"] == "G0" and r["job"]["seed"] == first_seed
        ),
        None,
    )
    differs = {}
    if base_first is not None:
        base_states = np.load(Path(base_first["directory"]) / "states.npz")["states"]
        for d, row in safe.items():
            states = np.load(Path(row["directory"]) / "states.npz")["states"]
            n = min(len(states), len(base_states))
            differs[str(d)] = bool(not np.allclose(states[:n], base_states[:n]))
    checks["safe_differs_from_unconstrained"] = bool(differs) and all(differs.values())
    imagination_path = remote / "imagination.json"
    imagination = json.loads(imagination_path.read_text()) if imagination_path.exists() else None
    checks["imagination_quantiles"] = bool(
        imagination and imagination["n_snippets"] >= 1000
    )
    seconds = [r["episode_seconds"] for r in rows]
    start = json.loads((remote / "stage-A-start.json").read_text())
    complete = json.loads((remote / "stage-A-complete.json").read_text())
    workers = complete["workers"]
    throughput = len(rows) / complete["elapsed_seconds"] * 3600 if rows else 0.0
    return {
        "utc": utc(),
        "passed": all(checks.values()),
        "checks": checks,
        "missing_jobs": missing,
        "baseline_per_layout": per_layout,
        "safe_differs": differs,
        "episode_seconds_median": float(np.median(seconds)) if seconds else None,
        "episode_seconds_max": float(np.max(seconds)) if seconds else None,
        # Wall seconds per completed episode for the whole worker pool, and per worker.
        "system_seconds_per_episode": complete["elapsed_seconds"] / len(rows) if rows else None,
        "seconds_per_episode_per_worker": (
            complete["elapsed_seconds"] / len(rows) * workers if rows else None
        ),
        "workers": workers,
        "episodes_per_hour_measured": throughput,
        "stage_A_elapsed_seconds": complete["elapsed_seconds"],
        "worker_load_seconds": [w.get("load_seconds") for w in start["workers"]],
        "imagination": imagination,
        "probe_stats": start["workers"][0].get("probe_stats"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", type=Path, default=ROOT / "outputs/safecem10/remote")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/safecem10/engineering.json"
    )
    args = parser.parse_args()
    report = engineering(args.remote, design())
    atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "checks": report["checks"],
                "episodes_per_hour": report["episodes_per_hour_measured"],
                "episode_seconds_median": report["episode_seconds_median"],
            }
        )
    )


if __name__ == "__main__":
    main()
