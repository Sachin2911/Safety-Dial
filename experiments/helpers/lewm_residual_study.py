"""Common-panel CMA and parameter-random-search development screen."""

import hashlib
import json
from pathlib import Path

import numpy as np

from helpers.lewm_proposal_adapter import ProposalSpec
from helpers.lewm_proposal_evolution import ProposalEvolution
from helpers.lewm_proposal_pilot import panel_fitness, summarize, write_json
from helpers.lewm_residual import influence_summary


class ResidualStudy:
    """Every candidate is scored on exactly the same predeclared development panel.

    Screens use a different fixed panel; validation remains closed until completion
    improves. Saved jobs are never repeated automatically after interruption.
    """

    def __init__(self, design, initial, root, evaluator, *, resume=False):
        self.design, self.initial, self.root, self.evaluate = (
            design,
            np.asarray(initial),
            Path(root),
            evaluator,
        )
        self.identity = hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()
        path = self.root / "state.json"
        if resume:
            self.state = json.loads(path.read_text())
            if self.state["design_sha256"] != self.identity:
                raise ValueError("Frozen residual protocol changed")
        else:
            if path.exists():
                raise FileExistsError("Use resume for an existing study")
            self.state = {
                "design_sha256": self.identity,
                "jobs": {},
                "phase": "diagnostic",
                "active": None,
            }
        search = design["search"]
        self.es = ProposalEvolution(
            initial,
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
                np.random.set_state(rng)
            self.state["jobs"][key] = row
            self.state["active"] = None
            self.es.save()
            self.save()
        result = self.state["jobs"][key]
        if result.get("error"):
            raise RuntimeError("Technical error retained; never retry an outcome: " + key)
        return result

    def panel(self, label, parameters, panel):
        return [self.job(label, parameters, case, seed) for case, seed in panel]

    def diagnostic(self):
        changed, eligible = 0, 0
        for case, seed in self.design["diagnostic"]:
            folders = [
                self.root / "episodes" / f"diagnostic_{arm}_{case}_s{seed}"
                for arm in ("native", "zero", "untrained")
            ]
            traces = []
            for folder in folders:
                with np.load(folder / "episode/trace.npz", allow_pickle=False) as data:
                    traces.append({name: data[name] for name in ("actions", "states", "dense")})
            if any(not np.array_equal(traces[0][name], traces[1][name]) for name in traces[0]):
                raise RuntimeError("Zero residual does not recover the exact native trajectory")
            rows = [json.loads((folder / "result.json").read_text()) for folder in folders]
            for counter in (
                "environment_actions",
                "executed_physics_steps",
                "predictor_steps",
                "model_candidate_rollouts",
                "model_batch_calls",
            ):
                if rows[0]["resources"].get(counter, 0) != rows[1]["resources"].get(counter, 0):
                    raise RuntimeError("Zero correction changed baseline work")
            changed += not np.array_equal(traces[0]["actions"], traces[2]["actions"])
            plans = json.loads((folders[2] / "episode/replans.json").read_text())
            eligible += influence_summary(plans)["guided_elite_slots"]
        result = {
            "zero_recovers_native": True,
            "same_native_model_work": True,
            "cases_with_changed_actions": int(changed),
            "nonzero_guided_elite_slots": eligible,
            "passed": changed > 0 and eligible > 0,
        }
        write_json(self.root / "influence-gate.json", result)
        return result["passed"]

    def run(self):
        d = self.design
        untrained = ProposalSpec(**d["adapter"]["spec"]).initial_parameters(d["adapter"]["seed"])
        for arm, weights in (("native", None), ("zero", self.initial), ("untrained", untrained)):
            self.panel("diagnostic_" + arm, weights, d["diagnostic"])
        if not self.diagnostic():
            self.state["phase"] = "influence_gate_failed"
            self.save()
            return self.state
        self.state["phase"] = "training"
        self.save()
        for arm, weights in (("native", None), ("zero", self.initial), ("untrained", untrained)):
            self.panel("development_" + arm, weights, d["development"])
        while len(self.es.history) < d["search"]["generations"]:
            generation = len(self.es.history)
            candidates = self.es.candidates()
            order = np.random.default_rng(d["search"]["order_seed"] + generation).permutation(
                len(candidates)
            )
            panels = {}
            for index in order:
                panels[int(index)] = self.panel(
                    f"cma_g{generation:03}_c{index:03}", candidates[index], d["development"]
                )
            scores = [panel_fitness(panels[i]) for i in range(len(candidates))]
            self.es.submit(scores)
            write_json(
                self.root / f"generation-{generation:03}.json",
                {
                    "generation": generation,
                    "scores": scores,
                    "panel": d["development"],
                    "parameters": candidates.tolist(),
                    "summaries": [summarize(panels[i]) for i in range(len(candidates))],
                },
            )
        # Independent draws from the same initial parameter distribution, with no updates.
        count = d["search"]["generations"] * d["search"]["population"]
        rng = np.random.default_rng(d["random_search_seed"])
        random_weights = self.initial + rng.normal(
            0, d["search"]["sigma"], (count, len(self.initial))
        )
        random_scores = []
        self.state["phase"] = "random_search"
        self.save()
        for index, weights in enumerate(random_weights):
            random_scores.append(
                panel_fitness(self.panel(f"random_c{index:03}", weights, d["development"]))
            )
        cma = [
            (score, -g, -i, weights)
            for g, history in enumerate(self.es.history)
            for i, (score, weights) in enumerate(
                zip(history["fitness"], history["parameters"], strict=True)
            )
        ]
        best_score, mg, mi, elite = max(cma, key=lambda item: item[:3])
        best_random = max(range(count), key=lambda i: random_scores[i])
        arms = {
            "native": None,
            "zero": self.initial.tolist(),
            "untrained": untrained.tolist(),
            "cma": elite,
            "random": random_weights[best_random].tolist(),
        }
        self.state["phase"] = "selection"
        self.save()
        panels = {
            arm: self.panel("selection_" + arm, weights, d["selection"])
            for arm, weights in arms.items()
        }
        scores = {arm: panel_fitness(rows) for arm, rows in panels.items()}
        winner = max(arms, key=lambda arm: scores[arm])
        summaries = {arm: summarize(rows) for arm, rows in panels.items()}
        # A development signal only. Progress alone cannot open the next stage.
        cma_summary = summaries["cma"]
        gain = (
            cma_summary["violation"] == 0
            and cma_summary["safe_completion"]
            > max(summaries[k]["safe_completion"] for k in ("native", "zero", "untrained"))
            and cma_summary["engaged"] >= summaries["native"]["engaged"]
        )
        selection = {
            "winner": winner,
            "parameters": arms[winner],
            "scores": scores,
            "summaries": summaries,
            "development_completion_signal": gain,
            "cma_elite": {"generation": -mg, "candidate_index": -mi, "fitness": best_score},
            "random_elite": {"candidate_index": best_random, "fitness": random_scores[best_random]},
            "validation_opened": False,
            "optimizer_advantage_established": False,
        }
        path = self.root / "selection.json"
        if path.exists() and json.loads(path.read_text()) != selection:
            raise RuntimeError("Frozen selection differs on recovery")
        write_json(path, selection)
        self.state["selection"] = selection
        self.state["selection_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.state["phase"] = "development_signal" if gain else "no_completion_signal"
        if len(self.state["jobs"]) != d["maximum_valid_episodes"]:
            raise RuntimeError("Frozen development schedule accounting mismatch")
        self.save()
        write_json(
            self.root / "random-search.json",
            {"parameters": random_weights.tolist(), "fitness": random_scores},
        )
        return self.state
