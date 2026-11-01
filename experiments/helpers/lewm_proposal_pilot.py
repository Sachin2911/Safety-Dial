"""Frozen pilot schedule, completion-dominant fitness and resumable selection."""

import hashlib
import json
from pathlib import Path

import numpy as np

from helpers.lewm_proposal_evolution import ProposalEvolution


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def violation(row):
    return bool(row["margin_breach"] or row["arena_margin_breach"])


def panel_fitness(rows, horizon=150):
    if not rows or any(row.get("error") for row in rows):
        raise ValueError("An engineering failure cannot supply fitness")
    if any(violation(row) for row in rows):
        return -1_000_000.0 - sum(violation(row) for row in rows)
    n = len(rows)
    successes = sum(bool(row["safe_completion"]) for row in rows)
    values = [float(row["safe_progress"]) for row in rows]
    if not all(np.isfinite(value) and 0 <= value <= 1 for value in values):
        raise ValueError("Progress must lie between zero and one per episode")
    progress = sum(values)
    capped = sum(row["actions"] if row["safe_completion"] else horizon for row in rows)
    return (n + 1) * successes + progress - 0.001 * capped / (n * horizon)


def summarize(rows):
    return {
        "episodes": len(rows),
        "safe_completion": sum(bool(r["safe_completion"]) for r in rows),
        "violation": sum(violation(r) for r in rows),
        "unfinished": sum(bool(r["unfinished"]) for r in rows),
        "engaged": sum(r["actions"] > 0 and r["pusher_path_length_px"] > 1 for r in rows),
        "actions": sum(r["actions"] for r in rows),
        "safe_progress_sum": sum(r["safe_progress"] for r in rows),
    }


class ProposalPilot:
    """The evaluator receives key, weights-or-None, case kind and planner seed.

    Episode implementations must recover a completed saved job rather than rerun it
    on resume. The CMA checkpoint is authoritative for completed optimizer updates.
    """

    def __init__(self, design, initial, root, evaluator, *, resume=False):
        self.design, self.initial, self.root, self.evaluate = (
            design,
            np.asarray(initial),
            Path(root),
            evaluator,
        )
        self.identity = hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()
        state_path = self.root / "state.json"
        if resume:
            self.state = json.loads(state_path.read_text())
            if self.state["design_sha256"] != self.identity:
                raise ValueError("Frozen pilot configuration changed")
        else:
            if state_path.exists():
                raise FileExistsError("Use resume for an existing pilot")
            self.state = {
                "design_sha256": self.identity,
                "jobs": {},
                "phase": "initial",
                "selection": None,
                "active": None,
            }
        search = design["search"]
        self.es = ProposalEvolution(
            self.initial,
            seed=search["seed"],
            population=search["population"],
            sigma=search["sigma"],
            checkpoint=self.root / "optimizer.pkl",
            resume=resume,
        )
        self.save()

    def save(self):
        write_json(self.root / "state.json", self.state)

    def job(self, label, parameters, case, seed):
        key = f"{label}_{case}_s{seed}"
        if key not in self.state["jobs"]:
            self.state["active"] = key
            self.save()
            rng = np.random.get_state()
            try:
                row = self.evaluate(key, parameters, case, seed)
            finally:
                # Environment seeding must not change the optimizer's random stream.
                np.random.set_state(rng)
            self.state["jobs"][key] = row
            self.state["active"] = None
            self.es.save()
            self.save()
        row = self.state["jobs"][key]
        if row.get("error"):
            raise RuntimeError("Retain and diagnose invalid episode: " + key)
        return row

    def panel(self, label, parameters, cases):
        return [self.job(label, parameters, case, seed) for case, seed in cases]

    def run(self):
        d = self.design
        first = [(case, d["training"]["seed_start"]) for case in d["training"]["cases"]]
        self.panel("initial", self.initial, first)
        self.state["phase"] = "training"
        self.save()
        while len(self.es.history) < d["search"]["generations"]:
            g = len(self.es.history)
            candidates = self.es.candidates()
            cases = [(case, d["training"]["seed_start"] + g) for case in d["training"]["cases"]]
            order = list(
                np.random.default_rng(d["search"]["order_seed"] + g).permutation(len(candidates))
            )
            if g % 2 == 0:
                self.panel(f"g{g:03}_native", None, cases)
            rows = {}
            for i in order:
                rows[int(i)] = self.panel(f"g{g:03}_c{i:03}", candidates[i], cases)
            if g % 2:
                self.panel(f"g{g:03}_native", None, cases)
            scores = [panel_fitness(rows[i]) for i in range(len(candidates))]
            self.es.submit(scores)
            write_json(
                self.root / f"generation-{g:03}.json",
                {
                    "generation": g,
                    "scores": scores,
                    "parameters": candidates.tolist(),
                    "summaries": [summarize(rows[i]) for i in range(len(candidates))],
                },
            )
        if self.state["selection"] is None:
            nominees = [
                (score, -g, -i, parameters)
                for g, generation in enumerate(self.es.history)
                for i, (score, parameters) in enumerate(
                    zip(generation["fitness"], generation["parameters"], strict=True)
                )
            ]
            score, minus_g, minus_i, elite = max(nominees, key=lambda row: row[:3])
            parameters = {"native": None, "initial": self.initial.tolist(), "elite": elite}
            kinds = list(parameters)
            self.state["phase"] = "selection"
            self.save()
            panels = {kind: [] for kind in kinds}
            for index, (case, seed) in enumerate(d["selection"]["cases"]):
                for kind in kinds[index % 3 :] + kinds[: index % 3]:
                    panels[kind].append(self.job("selection_" + kind, parameters[kind], case, seed))
            scores = {kind: panel_fitness(rows) for kind, rows in panels.items()}
            winner = max(kinds, key=lambda kind: scores[kind])  # Native wins exact ties.
            selection = {
                "winner": winner,
                "parameters": parameters[winner],
                "scores": scores,
                "summaries": {k: summarize(v) for k, v in panels.items()},
                "elite_provenance": {
                    "generation": -minus_g,
                    "candidate_index": -minus_i,
                    "training_fitness": score,
                },
            }
            path = self.root / "selection.json"
            if path.exists() and json.loads(path.read_text()) != selection:
                raise ValueError("Recovered selection differs from frozen receipt")
            write_json(path, selection)
            self.state["selection"] = selection
            self.state["selection_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            self.save()
        self.state["phase"] = "validation"
        self.save()
        selection = self.state["selection"]
        for index, (case, seed) in enumerate(d["validation"]["cases"]):
            if (
                hashlib.sha256((self.root / "selection.json").read_bytes()).hexdigest()
                != self.state["selection_sha256"]
            ):
                raise ValueError("Frozen selection changed")
            kinds = ("native", "selected") if index % 2 == 0 else ("selected", "native")
            for kind in kinds:
                self.job(
                    "validation_" + kind,
                    None if kind == "native" else selection["parameters"],
                    case,
                    seed,
                )
        if len(self.state["jobs"]) != d["maximum_valid_episodes"]:
            raise ValueError("Completed schedule differs from frozen episode count")
        self.state["phase"] = "complete"
        self.save()
        return self.state
