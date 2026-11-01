"""Read saved campaign state and exact worker identities on the rented host."""

import argparse
import collections
import json
import os
from pathlib import Path


def live(worker):
    try:
        stat = Path(f"/proc/{worker['pid']}/stat").read_text().split()
        return stat[2] != "Z" and stat[21] == str(worker["start_ticks"])
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="results/navigation-budget50-comparison-v1")
    args = parser.parse_args()
    if not os.environ.get("RUNPOD_POD_ID"):
        raise RuntimeError("Inspect experiment state on its rented host")
    root = Path("/workspace/Safety-Dial")
    groups = collections.defaultdict(list)
    for path in sorted((root / "environment/workers").glob("*.json")):
        worker = json.loads(path.read_text())
        out = worker.get("out", "")
        if not out.startswith(args.prefix):
            continue
        groups[out].append({"record": path.name, "pid": worker["pid"], "live": live(worker)})
    for out, workers in groups.items():
        folder = root / out
        state_path = folder / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        rows = [
            json.loads(path.read_text()) for path in sorted((folder / "jobs").glob("*/result.json"))
        ]
        methods = {}
        for method in sorted({row.get("method", "unspecified") for row in rows}):
            subset = [row for row in rows if row.get("method", "unspecified") == method]
            methods[method] = {
                "episodes": len(subset),
                **{
                    key: sum(row[key] for row in subset)
                    for key in (
                        "safe_completion",
                        "violation",
                        "unfinished",
                        "engaged",
                        "environment_actions",
                    )
                },
            }
        summary = {
            "out": out,
            "workers": workers,
            "phase": state.get("phase"),
            "active": state.get("active"),
            "completed_episodes": len(rows),
            "methods": methods,
            "candidate_panels_completed": state.get("candidate_panels_completed"),
            "optimizer_updates": state.get("optimizer_updates"),
            "actual_environment_actions": state.get("actual_environment_actions"),
            "charged_environment_actions": state.get("charged_environment_actions"),
            "best_trained_fitness": (state.get("best_trained") or {}).get("fitness"),
            "anchor_fitness": (state.get("anchor") or {}).get("fitness"),
            "technical_errors": sum(bool(row.get("error")) for row in rows),
        }
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
