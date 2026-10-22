"""Two serially admitted physical workers under one durable global budget and queue."""

import os
import signal
import subprocess
import time

from helpers.maniguard_latent_queue import ROOT, PhysicalQueue, append, atomic, process_identity
from helpers.maniguard_latent_results import read_episode


class PairedPhysicalQueue(PhysicalQueue):
    def __init__(self, run, protocol):
        super().__init__(run, protocol)
        if self.protocol.get("workers") != 2:
            raise ValueError("Paired physical execution requires its prospective two-worker design")
        self.handles = {}

    def launch(self, row, *, repair_note=None):
        job = row["definition"]
        self._cost_gate(job.get("engineering", False))
        attempt_id = job["id"] + f"-a{len(row['attempt_ids'])}"
        output = self.run / "episodes" / attempt_id
        output.parent.mkdir(exist_ok=True)
        command = self.command(job, output)
        request = output.parent / (attempt_id + "-request.json")
        terminal = output.parent / (attempt_id + "-terminal.json")
        attempt = {"id": attempt_id, "job_id": job["id"], "stage": job["stage"],
                   "engineering": job.get("engineering", False), "worker": job["worker"],
                   "status": "starting", "command": command, "output": str(output),
                   "started_epoch": time.time(), "repair_note": repair_note,
                   "terminal_receipt": str(terminal), "protocol_sha256": self.protocol_hash,
                   "deadline_epoch": min(time.time() + (1200 if job.get("engineering") else 3600),
                                         self.cutoff)}
        row.update(status="running", output=str(output))
        row["attempt_ids"].append(attempt_id)
        self.state["attempts"].append(attempt)
        self.save()
        append(self.run / "attempts.jsonl", attempt)
        atomic(request, {"command": command, "cwd": str(ROOT)})
        wrapper = [str(ROOT / "data/maniguard/envs/tools/bin/python"), "-u",
                   str(ROOT / "experiments/scripts/maniguard_latent_worker.py"),
                   "--request", str(request), "--receipt", str(terminal)]
        try:
            with (output.parent / (attempt_id + ".log")).open("x") as log:
                handle = subprocess.Popen(wrapper, cwd=ROOT, env=self.environment(job), stdout=log,
                                          stderr=subprocess.STDOUT, start_new_session=True)
            self.handles[attempt_id] = handle
            attempt["process_identity"] = process_identity(handle.pid)
            attempt["status"] = "running"
            self.save()
        except Exception as exc:
            attempt.update(status="invalid", error=type(exc).__name__ + ": " + str(exc))
            row["status"] = "invalid"
            self.save()
            append(self.run / "attempts.jsonl", attempt)
            raise
        return attempt

    def evaluate_batch(self, job_ids, *, repair_note=None, episode_seconds=180):
        if not 1 <= len(job_ids) <= 2 or len(set(job_ids)) != len(job_ids):
            raise ValueError("A wave must contain one or two distinct jobs")
        rows = [self.state["jobs"][name] for name in job_ids]
        workers = [row["definition"].get("worker") for row in rows]
        if any(type(worker) is not int or worker not in [0, 1] for worker in workers) \
                or len(set(workers)) != len(workers):
            raise ValueError("At most one physical job may occupy each declared worker")
        if all(row["status"] == "completed" for row in rows):
            return [read_episode(row["output"], require_full=not row["definition"].get("engineering"))
                    for row in rows]
        for attempt in self.state["attempts"]:
            identity = attempt.get("process_identity")
            if (identity is not None and process_identity(identity["pid"]) == identity
                    and attempt["job_id"] not in job_ids):
                raise RuntimeError("Observe the existing physical wave before admitting another job")
        new = [row for row in rows if row["status"] in ["pending", "invalid"]]
        if any(row["status"] == "invalid" for row in new) and not repair_note:
            raise RuntimeError("Diagnose and repair the existing invalid attempt before replacement")
        full_new = sum(not row["definition"].get("engineering", False) for row in new)
        if full_new + sum(not a["engineering"] for a in self.state["attempts"]) > 64:
            raise RuntimeError("The complete wave would exceed the global64-full-attempt ceiling")
        if new and time.time() + episode_seconds > self.cutoff:
            raise RuntimeError("Insufficient prospective time before preservation cutoff")
        attempts = {}
        for name, row in zip(job_ids, rows, strict=True):
            if row["status"] == "completed":
                continue
            if row["status"] == "running":
                attempts[name] = next(a for a in self.state["attempts"]
                                      if a["id"] == row["attempt_ids"][-1])
            else:
                attempts[name] = self.launch(row, repair_note=repair_note)
        # Observe both handles and both absolute timeouts; waiting on one worker
        # must never leave its partner unbounded. A context restart reuses these
        # exact saved identities instead of creating replacement physics.
        receipts, failures, reconciled = {}, [], set()
        while True:
            live = False
            for attempt in attempts.values():
                handle = self.handles.get(attempt["id"])
                if handle is not None:
                    code = handle.poll()
                    if code is not None:
                        attempt["exit_code"] = code
                identity = attempt.get("process_identity")
                if identity is None or process_identity(identity["pid"]) != identity:
                    name = attempt["job_id"]
                    if name not in reconciled:
                        try:
                            receipts[name] = self._reconcile(self.state["jobs"][name], attempt)
                        except RuntimeError as exc:
                            failures.append(str(exc))
                        reconciled.add(name)
                    continue
                live = True
                if time.time() >= attempt["deadline_epoch"]:
                    if "termination_sent_epoch" not in attempt:
                        os.killpg(identity["pid"], signal.SIGTERM)
                        attempt.update(termination_sent_epoch=time.time(), infrastructure_timeout=True)
                        self.save()
                    elif time.time() >= attempt["termination_sent_epoch"] + 5:
                        os.killpg(identity["pid"], signal.SIGKILL)
            if not live:
                break
            time.sleep(1)
        for name, row in zip(job_ids, rows, strict=True):
            if name in reconciled:
                continue
            try:
                if name in attempts:
                    receipts[name] = self._reconcile(row, attempts[name])
                else:
                    receipts[name] = read_episode(row["output"],
                                                   require_full=not row["definition"].get("engineering"))
            except (RuntimeError, OSError, ValueError, AssertionError, KeyError, TypeError) as exc:
                failures.append(str(exc))
        if failures:
            raise RuntimeError("Physical wave returned to recovery: " + "; ".join(failures))
        return [receipts[name] for name in job_ids]
