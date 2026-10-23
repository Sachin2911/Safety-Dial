"""Freeze the todo 3 gate plan (progress/03-retry-gate.md, item 5) as receipts/03-plan.json.

192 jobs: the six development tasks of progress/panels.json x policy seeds 100 to 115 x two
arms. Order: seed, then task, then the two arms of that (seed, task) pair; the pair's first
arm alternates native / retry from one pair to the next. Jobs are dealt to the two lanes
alternately (job 1 lane 0, job 2 lane 1, ...), so the two arms of a pair run at the same time
on different GPUs and each lane runs 48 native and 48 retry episodes.

Each job is one episode through scripts/todo1/eval_family_lane.sh with TASKS, SEED and PORT
from its entry and output leaf gate03/<arm>/seed<s>/<task> (the task is appended to the
checklist's gate03/<arm>/seed<s> because the lane wrapper refuses a non-empty output leaf and
each job must write its own). Ports: lane 0 native 9000, retry 9100; lane 1 native 9001,
retry 9101; both proxies of a lane forward to that lane's policy server (8000, 8001).

    python3 -I scripts/todo3/make_plan.py [--out progress/receipts/03-plan.json]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEEDS = list(range(100, 116))
ARMS = ("native", "retry")
PORTS = {0: {"native": 9000, "retry": 9100}, 1: {"native": 9001, "retry": 9101}}
SERVER = {0: 8000, 1: 8001}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "progress/receipts/03-plan.json"))
    args = ap.parse_args()
    panels = json.loads((ROOT / "progress/panels.json").read_text())
    tasks = [f"task_{t:04d}" for t in panels["development"]]
    jobs = []
    pair = 0
    for seed in SEEDS:
        for task in tasks:
            order = ARMS if pair % 2 == 0 else ARMS[::-1]
            for arm in order:
                job = len(jobs) + 1
                lane = (job - 1) % 2
                jobs.append({"job": job, "pair": pair + 1, "lane": lane, "seed": seed, "task": task, "arm": arm,
                             "port": PORTS[lane][arm], "server_port": SERVER[lane],
                             "leaf": f"gate03/{arm}/seed{seed}/{task}"})
            pair += 1
    per_lane = {lane: {arm: sum(1 for j in jobs if j["lane"] == lane and j["arm"] == arm) for arm in ARMS} for lane in (0, 1)}
    plan = {
        "schema": "gate03-plan-v1",
        "frozen_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checklist": "progress/03-retry-gate.md item 5",
        "panels": {"file": "progress/panels.json", "frozen_utc": panels["frozen_utc"], "development": panels["development"]},
        "family": "jar_transport", "level": "base",
        "config": "configs/eval/jar_transport_joint_legacy256.yaml (progress/receipts/01-jar-eval.yaml)",
        "runner": "scripts/todo1/eval_family_lane.sh jar_transport <leaf> <config> with TASKS, SEED, PORT per job",
        "horizon_steps": 2000, "execute_horizon": 8,
        "seeds": SEEDS, "tasks": tasks, "arms": list(ARMS),
        "proxy": {"script": "scripts/todo3/supervisor_proxy.py",
                  "native": "--mode off (byte-identical pass-through)",
                  "retry": "--mode retry with the script defaults: patience 10, stall-rad 0.01, warmup 10, "
                           "release-chunks 1, move-chunks 3, max-retreats 3, execute-horizon 8",
                  "ports": PORTS, "server_ports": SERVER},
        "order": "seed, then task, then the pair's two arms; the first arm alternates per pair",
        "dealing": "job i runs on lane (i - 1) mod 2",
        "n_jobs": len(jobs), "per_lane": per_lane,
        "rerun_rule": "a job whose episode does not reach status completed is rerun once into <leaf>_rerun1; both are recorded",
        "jobs": jobs,
    }
    text = json.dumps(plan, indent=1) + "\n"
    Path(args.out).write_text(text)
    print(f"{args.out}: {len(jobs)} jobs, per lane {per_lane}, sha256 {hashlib.sha256(text.encode()).hexdigest()[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
