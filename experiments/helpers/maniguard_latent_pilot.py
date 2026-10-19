"""Matched physical search and sealed independent comparison, with resumable lineage."""

import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from helpers.maniguard_latent import CandidateSearch, LatentController, initial_weights
from helpers.maniguard_latent_queue import ROOT, PhysicalQueue, atomic


class Pilot:
    def __init__(self, run, protocol, *, queue=None, checkpoint=None):
        self.run, self.protocol_path = Path(run).resolve(), Path(protocol).resolve()
        self.protocol = json.loads(self.protocol_path.read_text())
        if (self.protocol["status"] != "capacity_selected_frozen"
                or not self.protocol["capacity"]["fits"]):
            raise ValueError("Physical search requires a prospectively selected complete pilot")
        self.queue = PhysicalQueue(self.run, self.protocol_path) if queue is None else queue
        self.directory = self.run / "search"
        self.directory.mkdir(exist_ok=True)
        self.path = self.directory / "state.json"
        self.hash = hashlib.sha256(self.protocol_path.read_bytes()).hexdigest()
        self.checkpoint = checkpoint
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state["protocol_sha256"] != self.hash:
                raise ValueError("The frozen pilot changed after search began")
        else:
            self.state = {"schema": "maniguard-latent-search-v1", "protocol_sha256": self.hash,
                          "optimizers": {m: CandidateSearch(m).state() for m in ["ea", "random"]},
                          "cohorts": {}, "completed_milestones": [], "finalists": None}
            self.save()

    def save(self):
        atomic(self.path, self.state)

    def weights_file(self, candidate):
        path = self.directory / (candidate["id"] + ".npz")
        controller = LatentController(candidate["weights"])
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if LatentController(saved["weights"]).sha256() != controller.sha256():
                    raise ValueError("An immutable controller checkpoint changed")
        else:
            with path.open("xb") as stream:
                np.savez(stream, weights=controller.weights)
        return str(path)

    def jobs(self, candidate, generation):
        return [{"id": f"{candidate['id']}-case{i}", "stage": "shared_initial" if generation == 0
                 else f"search_generation_{generation}", "family": self.protocol["family"],
                 "scene": case["scene"], "seed": case["seed"],
                 "controller": self.weights_file(candidate)}
                for i, case in enumerate(self.protocol["training_panels"][generation])]

    def evaluate(self, candidate, generation):
        jobs = self.jobs(candidate, generation)
        self.queue.plan(jobs)
        scores = []
        for job in jobs:
            print("Observe or execute", job["id"], flush=True)
            receipt = self.queue.evaluate(job["id"],
                                          episode_seconds=self.protocol["capacity"]["episode_seconds"])
            scores.append(receipt["score"])
        return scores

    def record(self, method, candidate, scores):
        search = CandidateSearch(method, state=self.state["optimizers"][method])
        if not any(c["id"] == candidate["id"] for c in search.candidates):
            search.record(candidate["id"], candidate["weights"], scores,
                          parent=candidate.get("parent"))
            self.state["optimizers"][method] = search.state()
            self.save()
        else:
            saved = next(c for c in search.candidates if c["id"] == candidate["id"])
            if saved["scores"] != scores:
                raise ValueError("A completed candidate's physical scores changed")

    def milestone(self, name):
        if name not in self.state["completed_milestones"]:
            if self.checkpoint is not None:
                self.checkpoint(name)
            self.state["completed_milestones"].append(name)
            self.save()

    def run_search(self):
        if "initial" not in self.state["cohorts"]:
            self.state["cohorts"]["initial"] = [
                {"id": f"initial-{i}", "weights": w.tolist(), "parent": None}
                for i, w in enumerate(initial_weights())]
            self.save()
        for candidate in self.state["cohorts"]["initial"]:
            scores = self.evaluate(candidate, 0)
            for method in ["ea", "random"]:
                self.record(method, candidate, scores)
        self.milestone("shared-initial")
        for generation in [1, 2]:
            for method in ["ea", "random"]:
                key = f"{method}-g{generation}"
                if key not in self.state["cohorts"]:
                    search = CandidateSearch(method, state=self.state["optimizers"][method])
                    if search.generation != generation - 1:
                        raise ValueError("Optimizer and cohort generations are misaligned")
                    self.state["cohorts"][key] = search.propose()
                    self.state["optimizers"][method] = search.state()
                    self.save()
            # The frozen order interleaves methods at every candidate slot.
            for slot in range(4):
                methods = ["ea", "random"] if slot % 2 == 0 else ["random", "ea"]
                for method in methods:
                    candidate = self.state["cohorts"][f"{method}-g{generation}"][slot]
                    self.record(method, candidate, self.evaluate(candidate, generation))
            self.milestone(f"generation-{generation}")
        if self.state["finalists"] is None:
            initial = self.state["optimizers"]["ea"]["candidates"][:4]
            fixed = min(initial[1:3], key=lambda c: (-sum(c["scores"]), c["id"]))
            selected = {"native": initial[0], "fixed": fixed}
            for method in ["ea", "random"]:
                selected[method] = CandidateSearch(
                    method, state=self.state["optimizers"][method]).ranked()[0]
            self.state["finalists"] = {name: {"candidate_id": c["id"],
                                              "sha256": LatentController(c["weights"]).sha256(),
                                              "controller": self.weights_file(c)}
                                       for name, c in selected.items()}
            self.state["sealed_epoch"] = time.time()
            self.state["source_commit"] = subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
            self.save()
        seal = self.directory / "sealed-finalists.json"
        frozen = {"protocol_sha256": self.hash, "sealed_epoch": self.state["sealed_epoch"],
                  "source_commit": self.state["source_commit"],
                  "finalists": self.state["finalists"]}
        if seal.exists() and json.loads(seal.read_text()) != frozen:
            raise ValueError("The independent comparison's finalists changed")
        if not seal.exists():
            atomic(seal, frozen)
        self.milestone("sealed-finalists")
        return frozen

    def run_audit(self):
        seal = self.directory / "sealed-finalists.json"
        if not seal.exists() or self.state["finalists"] is None:
            raise ValueError("Seal all four finalists before any independent comparison")
        jobs = []
        for block, case in enumerate(self.protocol["audit_blocks"]):
            for condition in case["order"]:
                jobs.append({"id": f"audit-b{block}-{condition}", "stage": "independent_audit",
                             "family": self.protocol["family"], "scene": case["scene"],
                             "seed": case["seed"], "condition": condition, "block": block,
                             "controller": self.state["finalists"][condition]["controller"]})
        self.queue.plan(jobs)
        for job in jobs:
            self.queue.evaluate(job["id"],
                                episode_seconds=self.protocol["capacity"]["episode_seconds"])
        self.milestone("independent-audit")
