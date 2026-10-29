"""Task 19 orchestrator: CMA-ES over the evo19 adapter genes, run on the workstation.

Every candidate is evaluated by one Jar episode on a rented host (worker_evo19.py).
Each host holds at most two jobs (one running, one queued). Search jobs of the current
generation take priority; while hosts would otherwise wait at a generation barrier they
run native confirmation episodes. After the search stops, the final distribution mean is
frozen as the evolved adapter and confirmed on the same task-seed pairs as native.
State persists under outputs/evo19/search/, so the loop can be restarted safely.

    .venv/bin/python experiments/scripts/evolve_evo19.py run
    .venv/bin/python experiments/scripts/evolve_evo19.py status
"""

import argparse
import datetime as dt
import json
import math
import pickle
import shlex
import subprocess
import sys
import time
from pathlib import Path

import cma

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments/scripts"))
import manage_evo19 as infra  # noqa: E402
from helpers.dsrl06_infra import atomic_json, instances, ssh_endpoint, utc  # noqa: E402

OUT = ROOT / "outputs/evo19/search"
PROTOCOL = ROOT / "configs/evo19/protocol.json"
REMOTE = "/workspace/Safety-Dial/outputs/evo19"
TASKS = (0, 3, 6, 9)
POPSIZE = 8
MAX_GENERATIONS = 20
SEARCH_STOP = dt.datetime(2026, 10, 27, 1, 5, tzinfo=dt.timezone.utc).timestamp()  # amendment 3
NO_NEW_EPISODE = dt.datetime(2026, 10, 27, 2, 50, tzinfo=dt.timezone.utc).timestamp()
CONFIRM_SEEDS = range(193000, 193018)  # amendment 1 (22:37 UTC): 72 pairs, was 48
EXTENSION_SEEDS = range(193018, 193030)  # amendment 6 (01:08 UTC): replaces the hand-set arm
HANDSET_SEEDS = range(193000, 193012)
NATIVE = {"mode": "native", "speed_limit_mps": "inf", "near_radius_m": 0.22,
          "retreat_weight": 1.0, "gate": "closure", "stable_controls": 10, "candidates": 0}
HANDSET = {**NATIVE, "mode": "project", "speed_limit_mps": 0.1}


def settings_from(u):
    u = [min(1.0, max(0.0, float(x))) for x in u]
    speed = math.exp(math.log(0.03) + u[0] * (math.log(0.30) - math.log(0.03)))
    return {**NATIVE, "mode": "project", "speed_limit_mps": round(speed, 6),
            "near_radius_m": round(0.10 + 0.25 * u[1], 6), "retreat_weight": round(u[2], 6)}


def score(outcome):
    return (4 * outcome["strict_safe"] + 2 * (not outcome["raw_violation"])
            + 1 * outcome["ever_grasped"])


def settings_from2(u):
    """Stage 2 (amendment 2): isotropic check (retreat weight 1) over speed and radius."""
    u = [min(1.0, max(0.0, float(x))) for x in u]
    speed = math.exp(math.log(0.03) + u[0] * (math.log(0.30) - math.log(0.03)))
    return {**NATIVE, "mode": "project", "speed_limit_mps": round(speed, 6),
            "near_radius_m": round(0.10 + 0.25 * u[1], 6), "retreat_weight": 1.0}


def score2(outcome):
    """Stage 2 rank score: strict safe completion, then grasp; no credit for stalling."""
    return 4 * outcome["strict_safe"] + 1 * outcome["ever_grasped"]


def settings_from3(u):
    """Stage 3 (amendment 3): tilt-triggered retrace shield, open from the first control."""
    u = [min(1.0, max(0.0, float(x))) for x in u]
    return {**NATIVE, "gate": "always", "mode": "shield",
            "tilt_limit_deg": round(3.0 + 12.0 * u[0], 4),
            "retreat_m": round(0.02 + 0.10 * u[1], 5)}


def settings_from4(u):
    """Stage 4 (amendment 4): the stage 3 shield with contact counted over 8 controls."""
    return {**settings_from3(u), "contact_window": 8}


def score3(outcome):
    """Stage 3 rank score: strict safe completion, then no raw violation."""
    return 4 * outcome["strict_safe"] + 1 * (not outcome["raw_violation"])


STAGES = {
    1: {"es": "es.pkl", "gen": "generation", "gens": "generations", "prefix": "a",
        "seed": 191000, "settings": settings_from, "score": score, "max": MAX_GENERATIONS},
    2: {"es": "es2.pkl", "gen": "generation2", "gens": "generations2", "prefix": "a2",
        "seed": 192000, "settings": settings_from2, "score": score2, "max": 30},
    3: {"es": "es3.pkl", "gen": "generation3", "gens": "generations3", "prefix": "a3",
        "seed": 194000, "settings": settings_from3, "score": score3, "max": 30},
    4: {"es": "es4.pkl", "gen": "generation4", "gens": "generations4", "prefix": "a4",
        "seed": 195000, "settings": settings_from4, "score": score3, "max": 30},
}


class Orchestrator:
    def __init__(self):
        OUT.mkdir(parents=True, exist_ok=True)
        self.state_path = OUT / "state.json"
        self.es_path = OUT / "es.pkl"
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
            self.stage = self.state.get("stage", 1)
            self.cfg = STAGES[self.stage]
            self.es_path = OUT / self.cfg["es"]
            self.es = pickle.loads(self.es_path.read_bytes())
            for seed in CONFIRM_SEEDS:  # amendment 1 adds native jobs for the new seeds
                for task in TASKS:
                    if f"b-native-t{task}-s{seed}" not in self.state["jobs"]:
                        self.add_job(f"b-native-t{task}-s{seed}", "confirm_native", task,
                                     seed, NATIVE)
        else:
            self.stage, self.cfg = 1, STAGES[1]
            self.es = cma.CMAEvolutionStrategy([0.5, 0.5, 0.5], 0.25, {
                "popsize": POPSIZE, "seed": 19, "bounds": [[0, 0, 0], [1, 1, 1]],
                "verbose": -9})
            self.state = {"created_utc": utc(), "generation": -1, "jobs": {},
                          "generations": [], "phase": "search", "evolved": None}
            for seed in CONFIRM_SEEDS:
                for task in TASKS:
                    self.add_job(f"b-native-t{task}-s{seed}", "confirm_native", task, seed,
                                 NATIVE)
            self.new_generation()

    def save(self):
        self.es_path.write_bytes(pickle.dumps(self.es))
        atomic_json(self.state_path, self.state)

    def add_job(self, job_id, kind, task, seed, settings, extra=None):
        self.state["jobs"][job_id] = {"id": job_id, "kind": kind, "task": task, "seed": seed,
                                      "settings": settings, "status": "pending",
                                      "host": None, **(extra or {})}

    def new_generation(self):
        cfg = self.cfg
        g = self.state[cfg["gen"]] + 1
        xs = self.es.ask()
        rows = []
        for i, x in enumerate(xs):
            job_id = f"{cfg['prefix']}-g{g:02d}-c{i}"
            self.add_job(job_id, "search", TASKS[i % 4], cfg["seed"] + 10 * g + i,
                         cfg["settings"](x), {"generation": g, "candidate": i,
                                              "stage": self.stage,
                                              "x": [float(v) for v in x]})
            rows.append(job_id)
        self.state[cfg["gen"]] = g
        self.state[cfg["gens"]].append({
            "generation": g, "asked_utc": utc(), "jobs": rows,
            "mean": [float(v) for v in self.es.mean], "sigma": float(self.es.sigma),
            "mean_settings": cfg["settings"](self.es.mean)})
        self.save()

    def start_stage(self, stage):
        """Amendments 2 and 3: close the current stage and start the next two-gene search."""
        if stage != self.stage + 1 or self.state["phase"] != "search":
            raise SystemExit("A new stage starts only from the open previous stage")
        cancelled = []
        for job in self.state["jobs"].values():
            if job["kind"] == "search" and job["status"] == "pending":
                job["status"] = "cancelled"
                cancelled.append(job["id"])
        self.state[f"stage{self.stage}"] = {
            "closed_utc": utc(), "cancelled_unassigned": cancelled,
            "mean": [float(v) for v in self.es.mean], "sigma": float(self.es.sigma),
            "mean_settings": self.cfg["settings"](self.es.mean)}
        self.es_path.write_bytes(pickle.dumps(self.es))
        self.stage, self.cfg = stage, STAGES[stage]
        self.state.update({"stage": stage, self.cfg["gen"]: -1, self.cfg["gens"]: []})
        self.es_path = OUT / self.cfg["es"]
        sigma, seed = {2: (0.2, 1902), 3: (0.25, 1903), 4: (0.25, 1904)}[stage]
        self.es = cma.CMAEvolutionStrategy([0.5, 0.5], sigma, {
            "popsize": 6, "seed": seed, "bounds": [[0, 0], [1, 1]], "verbose": -9})
        self.new_generation()

    def maybe_tell(self):
        if self.state["phase"] != "search":
            return
        cfg = self.cfg
        row = self.state[cfg["gens"]][-1]
        jobs = [self.state["jobs"][j] for j in row["jobs"]]
        if any(j["status"] != "done" for j in jobs):
            return
        xs = [j["x"] for j in jobs]
        scores = [cfg["score"](j["outcome"]) if j.get("outcome") else 0 for j in jobs]
        self.es.tell(xs, [-s for s in scores])
        row.update({"told_utc": utc(), "scores": scores,
                    "mean_after": [float(v) for v in self.es.mean],
                    "sigma_after": float(self.es.sigma),
                    "mean_settings_after": cfg["settings"](self.es.mean)})
        if time.time() >= SEARCH_STOP or self.state[cfg["gen"]] + 1 >= cfg["max"]:
            self.finish_search()
        else:
            self.new_generation()
        self.save()

    def finish_search(self):
        evolved = self.cfg["settings"](self.es.mean)
        self.state["phase"] = "confirm"
        self.state["evolved"] = {"frozen_utc": utc(), "u": [float(v) for v in self.es.mean],
                                 "settings": evolved, "stage": self.stage,
                                 "generations": self.state[self.cfg["gen"]] + 1}
        atomic_json(OUT / "evolved-frozen.json", self.state["evolved"])
        for seed in CONFIRM_SEEDS:
            for task in TASKS:
                self.add_job(f"b-evolved-t{task}-s{seed}", "confirm_evolved", task, seed,
                             evolved)
        # The optional hand-set arm is the stage's starting point (x0): task 15's projection
        # for stages 1 and 2, the designer's shield (9 degrees, 0.07 m) for stage 3.
        handset = HANDSET if self.stage < 3 else self.cfg["settings"]([0.5, 0.5])
        for seed in HANDSET_SEEDS:  # runs only after both primary arms are assigned
            for task in TASKS:
                self.add_job(f"b-handset-t{task}-s{seed}", "confirm_handset", task, seed,
                             handset)

    def pending(self):
        jobs = [j for j in self.state["jobs"].values() if j["status"] == "pending"]
        search = sorted((j for j in jobs if j["kind"] == "search"
                         and j.get("stage", 1) == self.stage), key=lambda j: j["id"])
        native = sorted((j for j in jobs if j["kind"] == "confirm_native"),
                        key=lambda j: (j["seed"], j["task"]))
        evolved = sorted((j for j in jobs if j["kind"] == "confirm_evolved"),
                         key=lambda j: (j["seed"], j["task"]))
        handset = sorted((j for j in jobs if j["kind"] == "confirm_handset"),
                         key=lambda j: (j["seed"], j["task"]))
        if self.state["phase"] == "search":
            return search + native
        # Both primary arms in pair order (seed, task), so a time cut leaves complete pairs.
        # (01:48 UTC scheduling fix: the earlier zip always put an evolved job first, so
        # native jobs waited until every evolved job was assigned.)
        primary = sorted(evolved + native, key=lambda j: (j["seed"], j["task"], j["kind"]))
        return primary + handset

    def remote(self, row, command, payload=None, timeout=60):
        address, port = ssh_endpoint(row)
        return subprocess.run(["ssh", *infra.ssh_options(port), f"root@{address}", command],
                              input=payload, capture_output=True, text=True, timeout=timeout)

    def poll_host(self, name, row):
        script = ("import json,glob,os\n"
                  f"R='{REMOTE}'\n"
                  "res={os.path.basename(p):json.load(open(p)) for p in glob.glob(R+'/results/*.json')}\n"
                  "print(json.dumps({'ready':os.path.exists(R+'/server-ready.txt'),"
                  "'stop':os.path.exists(R+'/STOP'),"
                  "'queue':sorted(os.listdir(R+'/queue')) if os.path.isdir(R+'/queue') else [],"
                  "'claimed':sorted(os.listdir(R+'/claimed')) if os.path.isdir(R+'/claimed') else [],"
                  "'results':res}))\n")
        result = self.remote(row, "python3 -c " + shlex.quote(script), timeout=90)
        if result.returncode:
            raise RuntimeError(f"{name}: poll failed")
        return json.loads(result.stdout)

    def push_job(self, row, job):
        body = json.dumps({k: job[k] for k in ("id", "kind", "task", "seed", "settings")})
        name = f"{job['id']}.json"
        command = (f"mkdir -p {REMOTE}/queue && cat > {REMOTE}/queue/.{name}.tmp && "
                   f"mv {REMOTE}/queue/.{name}.tmp {REMOTE}/queue/{name}")
        result = self.remote(row, command, payload=body)
        if result.returncode:
            raise RuntimeError("job push failed")

    def step(self):
        record = infra.load()
        rows = {r["id"]: r for r in instances()}
        live = {}
        for host in record["hosts"]:
            row = rows.get(host.get("instance_id"))
            if row is not None and row.get("actual_status") == "running":
                live[host["name"]] = row
        # Jobs on hosts that are gone return to the pool.
        for job in self.state["jobs"].values():
            if job["status"] == "assigned" and job["host"] not in live:
                job.update(status="pending", host=None)
        ready = {}
        for name, row in live.items():
            try:
                status = self.poll_host(name, row)
            except (RuntimeError, subprocess.TimeoutExpired, ValueError) as error:
                print(json.dumps({"utc": utc(), "host": name, "poll_error": str(error)}),
                      flush=True)
                continue
            for filename, result in status["results"].items():
                job = self.state["jobs"].get(filename[:-5])
                if job is None or job["status"] == "done":
                    continue
                job.update(status="done", host=name, result_host=result.get("host"),
                           valid=result.get("valid"), outcome=result.get("outcome"),
                           adapter=result.get("adapter"), elapsed_s=result.get("elapsed_s"),
                           attempts=len(result.get("attempts", [])),
                           finished_utc=result.get("finished_utc"))
                print(json.dumps({"utc": utc(), "done": job["id"], "host": name,
                                  "valid": job["valid"],
                                  "score": (STAGES[job.get("stage", 1)]["score"](job["outcome"])
                                            if job["outcome"] and job["kind"] == "search"
                                            else None),
                                  "elapsed_s": job["elapsed_s"]}), flush=True)
            if status["ready"] and not status["stop"]:
                ready[name] = [row, len(status["queue"]) + len(status["claimed"]),
                               status["queue"]]
        self.maybe_tell()
        if self.stage > 1:
            self.cancel_old_stages(ready)
        if self.state["phase"] == "search":
            self.revoke_for_idle(ready)
        # Two passes: every idle host gets a job before any host gets a queued second one,
        # so a generation's jobs spread over all hosts instead of queueing behind others.
        # During the search a queued second slot only ever holds a search job.
        for limit in (1, 2):
            for name, (row, inflight, _) in sorted(ready.items(), key=lambda kv: kv[1][1]):
                if inflight >= limit or time.time() >= NO_NEW_EPISODE:
                    continue
                options = self.pending()
                if limit == 2 and self.state["phase"] == "search":
                    options = [j for j in options if j["kind"] == "search"]
                if not options:
                    break
                job = options[0]
                try:
                    self.push_job(row, job)
                except (RuntimeError, subprocess.TimeoutExpired):
                    continue
                job.update(status="assigned", host=name, assigned_utc=utc())
                ready[name][1] = inflight + 1
                print(json.dumps({"utc": utc(), "assigned": job["id"], "host": name}),
                      flush=True)
        self.save()

    def cancel_old_stages(self, ready):
        """Withdraw earlier-stage search jobs still waiting unclaimed in a host queue."""
        for name, (row, _, queue) in ready.items():
            for filename in queue:
                if not filename.startswith("a") or filename.startswith(f"{self.cfg['prefix']}-"):
                    continue
                result = self.remote(row, f"mkdir -p {REMOTE}/revoked && mv "
                                          f"{REMOTE}/queue/{filename} {REMOTE}/revoked/{filename}")
                if result.returncode == 0:
                    self.state["jobs"][filename[:-5]].update(status="cancelled", host=None)
                    ready[name][1] -= 1
                    ready[name][2] = [f for f in ready[name][2] if f != filename]
                    print(json.dumps({"utc": utc(), "cancelled": filename[:-5], "from": name}),
                          flush=True)

    def revoke_for_idle(self, ready):
        """Move queued, unclaimed search jobs from busy hosts back to the pool for idle ones.

        The worker claims by renaming queue/<id>.json, so the revoking rename fails safely
        if the worker took the job first.
        """
        idle = [n for n, (_, inflight, _) in ready.items() if inflight == 0]
        waiting = sum(j["kind"] == "search" and j["status"] == "pending"
                      for j in self.state["jobs"].values())
        spare = len(idle) - waiting
        for name, (row, _, queue) in ready.items():
            for filename in queue:
                if spare <= 0 or not filename.startswith(f"{self.cfg['prefix']}-"):
                    continue
                job_id = filename[:-5]
                result = self.remote(row, f"mkdir -p {REMOTE}/revoked && mv "
                                          f"{REMOTE}/queue/{filename} {REMOTE}/revoked/{filename}")
                if result.returncode == 0:
                    self.state["jobs"][job_id].update(status="pending", host=None)
                    ready[name][1] -= 1
                    ready[name][2] = [f for f in ready[name][2] if f != filename]
                    spare -= 1
                    print(json.dumps({"utc": utc(), "revoked": job_id, "from": name}),
                          flush=True)

    def summary(self):
        jobs = list(self.state["jobs"].values())
        out = {"utc": utc(), "phase": self.state["phase"], "stage": self.stage,
               "generation": self.state[self.cfg["gen"]],
               "mean_settings": self.cfg["settings"](self.es.mean),
               "sigma": float(self.es.sigma)}
        for kind in ("search", "confirm_native", "confirm_evolved", "confirm_handset"):
            done = [j for j in jobs if j["kind"] == kind and j["status"] == "done"]
            valid = [j for j in done if j.get("outcome")]
            out[kind] = {"done": len(done), "assigned": sum(
                j["kind"] == kind and j["status"] == "assigned" for j in jobs),
                "strict_safe": sum(j["outcome"]["strict_safe"] for j in valid),
                "violations": sum(j["outcome"]["raw_violation"] for j in valid)}
        out["recent_generations"] = [
            {k: g.get(k) for k in ("generation", "scores", "mean_settings_after", "sigma_after")}
            for g in self.state[self.cfg["gens"]][-3:]]
        return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "status", "add-handset", "stage2", "stage3",
                                           "stage4", "extend"])
    parser.add_argument("--poll", type=float, default=20)
    args = parser.parse_args()
    orchestrator = Orchestrator()
    if args.action == "status":
        print(json.dumps(orchestrator.summary(), indent=1))
        return
    if args.action in ("stage2", "stage3", "stage4"):
        orchestrator.start_stage(int(args.action[-1]))
        print(json.dumps(orchestrator.summary(), indent=1))
        return
    if args.action == "extend":
        if orchestrator.state["phase"] != "confirm":
            raise SystemExit("The extension needs the frozen evolved arm")
        evolved = orchestrator.state["evolved"]["settings"]
        cancelled = []
        for job in orchestrator.state["jobs"].values():
            if job["kind"] == "confirm_handset" and job["status"] == "pending":
                job["status"] = "cancelled"
                cancelled.append(job["id"])
        for seed in EXTENSION_SEEDS:
            for task in TASKS:
                for arm, kind, settings in (("native", "confirm_native", NATIVE),
                                            ("evolved", "confirm_evolved", evolved)):
                    job_id = f"b-{arm}-t{task}-s{seed}"
                    if job_id not in orchestrator.state["jobs"]:
                        orchestrator.add_job(job_id, kind, task, seed, settings)
        orchestrator.state["extension"] = {"utc": utc(), "seeds": list(EXTENSION_SEEDS),
                                           "cancelled_handset": cancelled}
        orchestrator.save()
        print(json.dumps(orchestrator.summary(), indent=1))
        return
    if args.action == "add-handset":
        for seed in CONFIRM_SEEDS:
            for task in TASKS:
                job_id = f"b-handset-t{task}-s{seed}"
                if job_id not in orchestrator.state["jobs"]:
                    orchestrator.add_job(job_id, "confirm_handset", task, seed, HANDSET)
        orchestrator.save()
        return
    while True:
        try:
            orchestrator.step()
        except RuntimeError as error:  # provider listing hiccups: retry next poll
            print(json.dumps({"utc": utc(), "step_error": str(error)}), flush=True)
        atomic_json(OUT / "summary.json", orchestrator.summary())
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
