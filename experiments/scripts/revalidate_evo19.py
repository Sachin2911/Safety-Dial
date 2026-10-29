"""Re-validate every mirrored task 19 confirmation episode on the workstation.

The counted episode of a job is the first attempt its host's worker log records as
finished ("episode_end", ok), which is the one the controller stored in the search state
and the analysis reads. One job (b-evolved-t6-s193003) was run a second time by the same
worker straight after its counted attempt ended and wrote over the host's result file;
such extra runs are re-validated too and reported as uncounted duplicates.

Each counted episode is checked with `evo19_validation.validate_recording` against both
the stored state record and the host's own validation sidecar. The base-yaw fit gap may
differ by float noise below 1e-6 rad (validation accepts gaps up to 0.01 rad). Writes
outputs/evo19/analysis/revalidation.json.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/helpers"))
sys.path.insert(0, str(ROOT / "experiments"))
from evo19_validation import validate_recording  # noqa: E402

REMOTE = ROOT / "outputs/evo19/remote"
STATE = ROOT / "outputs/evo19/search/state.json"
OUT = ROOT / "outputs/evo19/analysis/revalidation.json"


def finished_runs():
    """Every finished episode per job, in the order the host workers logged them."""
    runs = defaultdict(list)
    for log in sorted(REMOTE.glob("*/worker-events.jsonl")):
        for line in log.read_text().splitlines():
            event = json.loads(line)
            if event.get("event") == "episode_end" and event.get("ok"):
                runs[event["job"]].append((event["utc"], log.parent.name, event["leaf"]))
    return {job: sorted(rows) for job, rows in runs.items()}


def counted_episode(job_id, runs):
    """Episode directory of the counted (first finished) attempt of a job."""
    _, host, leaf = runs[job_id][0]
    return REMOTE / host / "episodes" / leaf


def check(episode, settings, record):
    local = validate_recording(
        next(episode.glob("recordings/*_base.npz")), next(episode.rglob("results.jsonl")),
        next(episode.rglob("*_ltl.json")), episode / "adapter.jsonl", settings)
    sidecar = json.loads((episode / "validation.json").read_text())
    gap = local["adapter"].pop("base_yaw_gap_rad")
    same = {}
    for name, ref in (("record", record), ("sidecar", sidecar)):
        adapter = {k: v for k, v in ref["adapter"].items() if k != "base_yaw_gap_rad"}
        same[name] = (local["outcome"] == ref["outcome"] and local["adapter"] == adapter
                      and abs(gap - ref["adapter"]["base_yaw_gap_rad"]) < 1e-6)
    return local, same


def main():
    state = json.loads(STATE.read_text())
    runs = finished_runs()
    rows, problems, duplicates = [], [], []
    for job_id, job in sorted(state["jobs"].items()):
        if job["status"] != "done" or not job["kind"].startswith("confirm"):
            continue
        settings = {k: float(v) if v in ("inf", "Infinity") else v
                    for k, v in job["settings"].items()}
        try:
            _, same = check(counted_episode(job_id, runs), settings, job)
        except Exception as error:  # report, never hide
            problems.append({"job": job_id, "problem": f"{type(error).__name__}: "
                             f"{str(error)[:200]}"})
            continue
        rows.append({"job": job_id, "match": same["record"] and same["sidecar"]})
        if not rows[-1]["match"]:
            problems.append({"job": job_id, "problem": f"differs: {same}"})
        for utc, host, leaf in runs[job_id][1:]:
            local, same = check(REMOTE / host / "episodes" / leaf, settings, job)
            duplicates.append({"job": job_id, "leaf": leaf, "host": host, "finished_utc": utc,
                               "counted": False, "validates_against_own_sidecar":
                               same["sidecar"], "outcome": local["outcome"]})
    report = {"episodes": len(rows), "matching": sum(r["match"] for r in rows),
              "problems": problems, "uncounted_duplicates": duplicates}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k in ("episodes", "matching")}),
          len(problems), "problems", len(duplicates), "duplicates")


if __name__ == "__main__":
    main()
