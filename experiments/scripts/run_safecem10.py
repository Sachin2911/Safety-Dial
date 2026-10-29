"""Remote orchestrator: frozen stage A/B schedules and the stage C search, via the queue.

The orchestrator is the only process that enqueues work. It starts the persistent workers,
tickets every episode once, waits for durable results, and records generation-by-
generation state for stage C so an interrupted run resumes without repeating episodes.
"""

import argparse
import datetime as dt
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_infra import atomic_json, utc
from helpers.safecem10_evo import (
    crowded_subset,
    nsga2_offspring,
    nsga2_survivors,
    pareto_front,
    random_points,
)
from helpers.safecem10_protocol import (
    aggregate,
    candidate_job,
    digest,
    evaluation_seeds,
    grid_for_budget,
    verify_freeze,
)

OUT = ROOT / "outputs/safecem10"
QUEUE = OUT / "queue"
PYTHON = "/venv/main/bin/python"
WORKER = ROOT / "experiments/scripts/worker_safecem10.py"


def inventory(root=None):
    root = OUT if root is None else root
    rows = {}
    for result in sorted((root / "attempts").glob("*/result.json")):
        row = json.loads(result.read_text())
        row["directory"] = str(result.parent)
        rows.setdefault(row["job"]["id"], []).append(row)
    failed = {}
    for ticket in sorted((root / "queue/failed").glob("*.json")):
        data = json.loads(ticket.read_text())
        failed.setdefault(data["job"]["id"], []).append(data)
    return rows, failed


def valid_result(rows, job_id, freeze_sha):
    for row in rows.get(job_id, []):
        if row["status"] == "complete" and row.get("freeze_sha256", freeze_sha) == freeze_sha:
            return row
    return None


class Orchestrator:
    def __init__(self, freeze_path, workers, deadline, caps):
        self.freeze_path = freeze_path.resolve()
        self.freeze = verify_freeze(self.freeze_path)
        self.sha = digest(self.freeze_path)
        self.workers = []
        self.n_workers = workers
        self.deadline = deadline
        self.caps = caps
        self.env_steps = 0
        self.attempts = 0
        for name in ("pending", "claimed", "done", "failed"):
            (QUEUE / name).mkdir(parents=True, exist_ok=True)
        (OUT / "attempts").mkdir(parents=True, exist_ok=True)
        rows, failed = inventory()
        self.attempts = sum(len(v) for v in rows.values()) + sum(len(v) for v in failed.values())
        self.env_steps = sum(r["env_steps"] for v in rows.values() for r in v)

    def start_workers(self):
        (OUT / "logs").mkdir(parents=True, exist_ok=True)
        for i in range(self.n_workers):
            log = (OUT / "logs" / f"worker-{i}.log").open("a", buffering=1)
            process = subprocess.Popen(
                [
                    PYTHON,
                    "-u",
                    str(WORKER),
                    "--freeze",
                    str(self.freeze_path),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=ROOT,
                start_new_session=True,
            )
            self.workers.append(process)
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            ready = [p for p in (OUT / "workers").glob("*.json") if p.exists()]
            states = []
            for path in ready:
                try:
                    data = json.loads(path.read_text())
                except ValueError:
                    continue
                if any(w.pid == data.get("pid") for w in self.workers):
                    states.append(data)
            if len(states) == self.n_workers:
                return states
            if any(w.poll() is not None for w in self.workers):
                raise RuntimeError("A worker exited during model loading")
            time.sleep(1)
        raise RuntimeError("Workers did not report readiness")

    def stop_workers(self):
        for process in self.workers:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in self.workers:
            try:
                process.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()

    def time_left(self):
        return self.deadline - time.time()

    def enqueue(self, jobs):
        """Ticket every job without a valid result; returns the ids awaited."""
        rows, failed = inventory()
        awaited = []
        for job in jobs:
            if valid_result(rows, job["id"], self.sha):
                continue
            prior = len(rows.get(job["id"], [])) + len(failed.get(job["id"], []))
            if prior >= self.caps["job_attempts_per_id"]:
                raise RuntimeError(f"Attempt allowance exhausted for {job['id']}")
            if self.attempts + 1 > self.caps["maximum_episode_attempts"]:
                raise RuntimeError("Episode attempt cap reached")
            budget = self.freeze["design"]["environment"]["eval_budget_steps"]
            if self.env_steps + budget > self.caps["maximum_env_steps"]:
                raise RuntimeError("Simulator step cap reached")
            pending = QUEUE / "pending" / f"{job['id']}.json"
            claimed = QUEUE / "claimed" / f"{job['id']}.json"
            if pending.exists() or claimed.exists():
                awaited.append(job["id"])
                continue
            suffix = "" if prior == 0 else f".retry{prior}"
            folder = OUT / "attempts" / f"{job['id']}{suffix}"
            ticket = {
                "job": job,
                "freeze_sha256": self.sha,
                "attempt_dir": str(folder),
                "attempt": prior,
                "queued_utc": utc(),
            }
            atomic_json(pending, ticket)
            self.attempts += 1
            self.env_steps += budget
            awaited.append(job["id"])
        return awaited

    def wait(self, job_ids):
        """Block until every awaited job has a result or a failure ticket."""
        remaining = set(job_ids)
        while remaining:
            if (OUT / "STOP").exists() or self.time_left() <= 0:
                raise RuntimeError("Preservation margin reached; stop admitted work")
            if all(w.poll() is not None for w in self.workers):
                raise RuntimeError("All workers exited")
            for job_id in list(remaining):
                if (QUEUE / "done" / f"{job_id}.json").exists():
                    remaining.discard(job_id)
                elif (QUEUE / "failed" / f"{job_id}.json").exists():
                    remaining.discard(job_id)
            time.sleep(1)
        rows, failed = inventory()
        results = {}
        for job_id in job_ids:
            row = valid_result(rows, job_id, self.sha)
            if row is None:
                raise RuntimeError(f"Job failed: {job_id}; diagnose before continuing")
            results[job_id] = row
        self.env_steps = sum(r["env_steps"] for v in rows.values() for r in v)
        atomic_json(
            OUT / "attempt-ledger.json",
            {
                "utc": utc(),
                "attempts": self.attempts,
                "env_steps": self.env_steps,
                "complete": sum(1 for v in rows.values() for r in v if r["status"] == "complete"),
                "failed": sum(len(v) for v in failed.values()),
            },
        )
        return results

    def run_fixed(self, jobs):
        awaited = self.enqueue(jobs)
        return self.wait(awaited) if awaited else {}

    def candidate_jobs(self, method, replicate, generation, genotypes, layout, seeds, tag):
        spec = self.freeze["design"]
        stage = self.freeze["stage"]
        jobs = []
        for c, x in enumerate(genotypes):
            for seed in seeds:
                jobs.append(
                    candidate_job(
                        stage,
                        layout,
                        seed,
                        x,
                        spec,
                        f"{method}-r{replicate}-{tag}-c{c:02d}",
                        {"method": method, "replicate": replicate, "generation": generation},
                    )
                )
        return jobs

    def results_for(self, jobs):
        rows, _ = inventory()
        results = {}
        for job in jobs:
            row = valid_result(rows, job["id"], self.sha)
            if row is None:
                raise RuntimeError(f"Missing valid result: {job['id']}")
            results[job["id"]] = row
        return results

    @staticmethod
    def objectives_for(jobs, results, count):
        objectives = []
        for c in range(count):
            rows = [results[j["id"]] for j in jobs if j["id"].endswith(f"-c{c:02d}")]
            objectives.append(aggregate(rows))
        return objectives


def stage_c(orc, args):
    """Generation-synchronous search for every method and replicate, batched per generation.

    All runs' candidates of generation g are ticketed together so the workers never idle on
    one run's tail. Records are durable per run and generation; a resumed invocation replays
    the proposals, verifies them against the records and only tickets missing episodes.
    """
    spec = orc.freeze["design"]
    folder = OUT / f"stage{orc.freeze['stage']}"
    plan_path = folder / "plan.json"
    if not plan_path.exists():
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(plan_path, {**orc.freeze["sizing"]["plan"], "freeze_sha256": orc.sha})
    plan = json.loads(plan_path.read_text())
    if plan.get("freeze_sha256") != orc.sha:
        raise ValueError("Stage C plan belongs to a different freeze")
    stage = spec["stages"]["C"]
    P = stage["population"]
    G = plan["generations"]
    reps = plan["replicates"]
    layout = stage["search_layout"]
    runs = [(m, r) for m in stage["methods"] for r in range(reps)]
    rngs, state = {}, {}
    for method, r in runs:
        key = f"{method}-r{r}"
        (folder / key).mkdir(parents=True, exist_ok=True)
        rngs[key] = np.random.default_rng([2026102510, stage["methods"].index(method), r])
        state[key] = {"population": [], "objectives": [], "all": [], "all_obj": []}
    grid = grid_for_budget(spec, P * G)
    seconds = plan["seconds_per_episode"] / orc.n_workers
    per_generation = P * len(runs) * stage["eval_seeds_per_candidate"]
    finals_episodes = len(runs) * stage["front_cap"] * len(stage["final_seeds"]) * 2
    completed = G
    for g in range(G):
        if orc.time_left() < (per_generation + finals_episodes) * seconds * 1.15 + 1800:
            completed = g
            atomic_json(folder / "truncated.json", {"utc": utc(), "generation": g})
            break
        batch = {}
        for method, r in runs:
            key = f"{method}-r{r}"
            st, rng = state[key], rngs[key]
            if method == "nsga2" and g > 0:
                genotypes = nsga2_offspring(st["population"], st["objectives"], P, rng)
            elif method in ("nsga2", "random"):
                genotypes = random_points(P, 5, rng)
            else:
                genotypes = grid[g * P : (g + 1) * P]
            record = folder / key / f"gen-{g:02d}.json"
            if record.exists():
                saved = json.loads(record.read_text())
                replay = np.asarray([list(map(float, x)) for x in genotypes])
                if not np.allclose(replay, np.asarray(saved["genotypes"])):
                    raise RuntimeError(f"Resumed generation differs from its record: {key} g{g}")
                genotypes = [np.asarray(x) for x in saved["genotypes"]]
            else:
                saved = {"genotypes": [list(map(float, x)) for x in genotypes]}
                atomic_json(record, saved)
            if not genotypes:
                continue
            seeds = evaluation_seeds(spec, method, r, g)
            jobs = orc.candidate_jobs(method, r, g, genotypes, layout, seeds, f"g{g:02d}")
            batch[key] = (genotypes, record, saved, seeds, jobs)
        awaited = orc.enqueue([j for _, _, _, _, jobs in batch.values() for j in jobs])
        if awaited:
            orc.wait(awaited)
        for method, r in runs:
            key = f"{method}-r{r}"
            if key not in batch:
                continue
            genotypes, record, saved, seeds, jobs = batch[key]
            objectives = orc.objectives_for(jobs, orc.results_for(jobs), len(genotypes))
            saved.update(
                {
                    "seeds": seeds,
                    "job_ids": [j["id"] for j in jobs],
                    "objectives": objectives,
                    "utc": utc(),
                }
            )
            atomic_json(record, saved)
            st = state[key]
            st["all"] += genotypes
            st["all_obj"] += objectives
            if method == "nsga2":
                if g == 0:
                    st["population"], st["objectives"] = genotypes, objectives
                else:
                    pop, obj, _ = nsga2_survivors(
                        st["population"] + genotypes, st["objectives"] + objectives, P
                    )
                    st["population"], st["objectives"] = pop, obj
            else:
                st["population"], st["objectives"] = st["all"], st["all_obj"]
            print(
                json.dumps({"event": "generation", "key": key, "g": g, "objectives": objectives}),
                flush=True,
            )
    finals = {}
    for method, r in runs:
        key = f"{method}-r{r}"
        st = state[key]
        if not st["all"]:
            continue
        final_record = folder / key / "final.json"
        if final_record.exists() and json.loads(final_record.read_text()).get("complete"):
            continue
        front_idx = pareto_front(st["objectives"])
        chosen = crowded_subset(st["objectives"], front_idx, stage["front_cap"])
        genotypes = [st["population"][i] for i in chosen]
        record = {
            "complete": False,
            "search_phase_objectives": [st["objectives"][i] for i in chosen],
            "genotypes": [list(map(float, x)) for x in genotypes],
            "generations_completed": completed,
            "candidates_evaluated": len(st["all"]),
            "all_search_objectives": [list(map(float, o)) for o in st["all_obj"]],
        }
        jobs = {}
        for lay_key, lay in (("fresh", layout), ("holdout", stage["holdout_layout"])):
            jobs[lay_key] = orc.candidate_jobs(
                method, r, -1, genotypes, lay, stage["final_seeds"], f"final-{lay_key}"
            )
            record[f"{lay_key}_layout"] = lay
            record[f"{lay_key}_job_ids"] = [j["id"] for j in jobs[lay_key]]
        atomic_json(final_record, record)
        finals[key] = (genotypes, record, final_record, jobs)
    awaited = orc.enqueue(
        [j for _, _, _, jobs in finals.values() for lay in jobs.values() for j in lay]
    )
    if awaited:
        orc.wait(awaited)
    for key, (genotypes, record, final_record, jobs) in finals.items():
        for lay_key, lay_jobs in jobs.items():
            record[f"{lay_key}_objectives"] = orc.objectives_for(
                lay_jobs, orc.results_for(lay_jobs), len(genotypes)
            )
        record["complete"] = True
        record["utc"] = utc()
        atomic_json(final_record, record)
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument(
        "--rental", type=Path, default=ROOT / "progress/rental-safecem10-20261025.json"
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--only", help="Run one frozen job id")
    args = parser.parse_args()
    freeze = verify_freeze(args.freeze)
    rental = json.loads(args.rental.read_text())
    if str(rental["instance_id"]) != os.environ.get("CONTAINER_ID"):
        raise ValueError("This runner may only execute on its recorded rental")
    margin = freeze["design"]["money"]["preservation_margin_minutes"] * 60
    deadline = dt.datetime.fromisoformat(rental["allocation_deadline_utc"]).timestamp() - margin
    stage = freeze["stage"]
    if stage == "preparation":
        raise ValueError("Freeze the engineering stage before execution")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        orc = Orchestrator(args.freeze, args.workers, deadline, freeze["design"]["caps"])
        started = time.time()
        ready = orc.start_workers()
        atomic_json(
            OUT / f"stage-{stage}-start.json",
            {"utc": utc(), "workers": ready, "freeze_sha256": orc.sha, "deadline": deadline},
        )
        try:
            if stage in ("A", "B", "B2"):
                jobs = freeze["jobs"]
                if args.only:
                    jobs = [j for j in jobs if j["id"] == args.only]
                    if not jobs:
                        raise ValueError("No such frozen job")
                results = orc.run_fixed(jobs)
                summary = {"jobs": [j["id"] for j in jobs], "results": len(results)}
            elif stage in ("C", "C2"):
                generations = stage_c(orc, args)
                summary = {"generations_completed": generations}
            else:
                raise ValueError(f"Unknown stage {stage}")
        finally:
            orc.stop_workers()
        atomic_json(
            OUT / f"stage-{stage}-complete.json",
            {
                "utc": utc(),
                "freeze_sha256": orc.sha,
                "elapsed_seconds": time.time() - started,
                "workers": args.workers,
                "partial_request": args.only is not None,
                "attempts": orc.attempts,
                "env_steps": orc.env_steps,
                **summary,
            },
        )
        print(json.dumps({"stage": stage, "elapsed_seconds": time.time() - started, **summary}))


if __name__ == "__main__":
    main()
