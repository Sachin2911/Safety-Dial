"""Residual-mixture development study using the unchanged frozen-model evaluator."""

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.lewm_proposal_pilot import panel_fitness, summarize, write_json
from helpers.lewm_proposal_adapter import ProposalSpec
from helpers.lewm_residual import ResidualAdapter, ResidualSpec, influence_summary
from helpers.lewm_residual_study import ResidualStudy
from scripts import run_lewm_proposal_pilot as runtime


def audit(args, design):
    root = args.out
    state = json.loads((root / "state.json").read_text())
    freeze = json.loads((root / "source-freeze.json").read_text())
    if freeze["protocol_sha256"] != runtime.digest(args.protocol):
        raise RuntimeError("Protocol changed")
    runtime.identities(design, freeze)
    if state["phase"] not in (
        "influence_gate_failed",
        "no_completion_signal",
        "development_signal",
    ):
        raise RuntimeError("Incomplete development schedule")
    jobs = state["jobs"]
    total = collections.Counter()
    mechanism = collections.Counter()
    checks = []
    for key, row in jobs.items():
        folder = root / "episodes" / key
        if row != json.loads((folder / "result.json").read_text()) or row.get("error"):
            raise RuntimeError("Invalid saved job: " + key)
        with np.load(folder / "episode/trace.npz", allow_pickle=False) as data:
            states, actions, dense = data["states"], data["actions"], data["dense"]
        shape = design["cases"][row["case_kind"]]
        clearance, arena = runtime.clearances(dense, shape)
        unsafe = bool((clearance < 6 - 1e-6).any() or (arena < 6 - 1e-6).any())
        goal = np.asarray(row["goal_state"])
        angular = abs(goal[4] - states[-1, 4]) % (2 * np.pi)
        angular = min(angular, 2 * np.pi - angular)
        success = bool(np.linalg.norm(goal[:4] - states[-1, :4]) < 20 and angular < np.pi / 9)
        safe = success and not unsafe and not row["controller_failure"]
        if (
            unsafe != bool(row["margin_breach"] or row["arena_margin_breach"])
            or success != row["success"]
            or safe != row["safe_completion"]
        ):
            raise RuntimeError("Independent native-goal/dense monitor mismatch: " + key)
        counts = row["resources"]
        if (
            len(actions) != row["actions"]
            or len(actions) > 150
            or len(dense) != 10 * len(actions) + 1
            or counts.get("executed_physics_steps", 0) != 10 * len(actions)
        ):
            raise RuntimeError("Actual interaction accounting mismatch")
        if (
            counts["predictor_steps"] > design["episode"]["maximum_predictor_steps"]
            or runtime.progress(row) != row["safe_progress"]
        ):
            raise RuntimeError("Frozen budget or progress feedback mismatch")
        plans = json.loads((folder / "episode/replans.json").read_text())
        if any(p.get("executed") != "none" and p["executed_violation"] > 0 for p in plans):
            raise RuntimeError("Unsafe plan execution recorded")
        if row["adapter"]:
            controller = ResidualAdapter.load(folder / "controller.json", shape)
            proposals = json.loads((folder / "proposals.json").read_text())
            if counts["adapter_forward_calls"] != len(proposals):
                raise RuntimeError("Residual call accounting mismatch")
            for proposal in proposals:
                if not np.array_equal(
                    controller.forward(proposal["features"], len(proposal["proposal"])),
                    np.asarray(proposal["proposal"]),
                ):
                    raise RuntimeError(
                        "Residual cannot be replayed from saved features and weights"
                    )
            for plan in plans:
                if plan.get("executed") == "none":
                    continue
                proposal = next(p for p in proposals if p["replan"] == plan["features"]["replan"])
                if not np.allclose(
                    proposal["proposal"], plan["adapter_proposal"], atol=1e-6, rtol=0
                ):
                    raise RuntimeError("Planner residual differs from adapter")
                for item in plan["residual_checks"]:
                    expected = int(
                        (item["samples"] - item["reserved_native_slots"])
                        * design["episode"]["planner"]["proposal_mixture_fraction"]
                    )
                    if (
                        item["new_guided_samples"] != expected
                        or item["guided_elites"] > item["fitted_elites"]
                    ):
                        raise RuntimeError("Mixture allocation or elite provenance mismatch")
            influence = influence_summary(plans)
            if influence["new_guided_samples"] != counts.get(
                "adapter_guided_candidate_evaluations", 0
            ):
                raise RuntimeError("Guided candidate accounting mismatch")
            mechanism.update(influence)
        total.update(counts)
        checks.append(
            {
                "key": key,
                "verified": True,
                "minimum_clearance_px": float(clearance.min()),
                "minimum_arena_clearance_px": float(arena.min()),
            }
        )
    if (
        total["environment_actions"] > design["maximum_total_environment_actions"]
        or total["predictor_steps"] > design["maximum_total_predictor_steps"]
    ):
        raise RuntimeError("Total interaction/model-work ceiling exceeded")
    # Re-run the saved-evidence-only identity diagnostic, never the simulator.
    dummy = object.__new__(ResidualStudy)
    dummy.design, dummy.root = design, root
    if dummy.diagnostic() != (state["phase"] != "influence_gate_failed"):
        raise RuntimeError("Influence gate receipt differs")
    with (root / "optimizer.pkl").open("rb") as stream:
        optimizer = pickle.load(stream)  # Own trusted checkpoint only.
    expected = {}
    initial = (
        ResidualSpec(**design["adapter"]["spec"])
        .initial_parameters(design["adapter"]["seed"])
        .tolist()
    )
    untrained = (
        ProposalSpec(**design["adapter"]["spec"])
        .initial_parameters(design["adapter"]["seed"])
        .tolist()
    )

    def check_panel(label, weights, panel):
        rows = []
        for case, seed in panel:
            key = f"{label}_{case}_s{seed}"
            expected[key] = weights
            row = jobs[key]
            if (
                row["seed"] != seed
                or row["case_kind"] != case
                or row["adapter"] != (weights is not None)
            ):
                raise RuntimeError("Panel identity differs")
            if (
                weights is not None
                and json.loads((root / "episodes" / key / "controller.json").read_text())[
                    "parameters"
                ]
                != weights
            ):
                raise RuntimeError("Candidate genotype differs from the optimizer record")
            rows.append(row)
        return rows

    for arm, weights in (("native", None), ("zero", initial), ("untrained", untrained)):
        check_panel("diagnostic_" + arm, weights, design["diagnostic"])
    selection = None
    if state["phase"] != "influence_gate_failed":
        if (
            len(optimizer["history"]) != design["search"]["generations"]
            or optimizer["es"].countiter != len(optimizer["history"])
            or optimizer["pending"] is not None
        ):
            raise RuntimeError("Optimizer update count differs")
        for arm, weights in (("native", None), ("zero", initial), ("untrained", untrained)):
            check_panel("development_" + arm, weights, design["development"])
        nominees = []
        for generation, history in enumerate(optimizer["history"]):
            for index, weights in enumerate(history["parameters"]):
                rows = check_panel(
                    f"cma_g{generation:03}_c{index:03}", weights, design["development"]
                )
                score = panel_fitness(rows)
                if score != history["fitness"][index]:
                    raise RuntimeError("CMA fitness differs from common-panel outcomes")
                nominees.append((score, -generation, -index, weights))
        score, mg, mi, elite = max(nominees, key=lambda item: item[:3])
        random_search = json.loads((root / "random-search.json").read_text())
        rng = np.random.default_rng(design["random_search_seed"])
        count = design["search"]["population"] * design["search"]["generations"]
        weights = np.asarray(initial) + rng.normal(
            0, design["search"]["sigma"], (count, len(initial))
        )
        if weights.tolist() != random_search["parameters"]:
            raise RuntimeError("Random search distribution changed")
        for index, candidate in enumerate(random_search["parameters"]):
            rows = check_panel(f"random_c{index:03}", candidate, design["development"])
            if panel_fitness(rows) != random_search["fitness"][index]:
                raise RuntimeError("Random-search feedback differs")
        ri = max(range(count), key=lambda i: random_search["fitness"][i])
        arms = {
            "native": None,
            "zero": initial,
            "untrained": untrained,
            "cma": elite,
            "random": weights[ri].tolist(),
        }
        scores = {
            arm: panel_fitness(check_panel("selection_" + arm, theta, design["selection"]))
            for arm, theta in arms.items()
        }
        selection = state["selection"]
        winner = max(arms, key=lambda arm: scores[arm])
        if (
            scores != selection["scores"]
            or winner != selection["winner"]
            or arms[winner] != selection["parameters"]
            or selection["cma_elite"]
            != {"generation": -mg, "candidate_index": -mi, "fitness": score}
        ):
            raise RuntimeError("Frozen common-panel elite or selection rule differs")
        if (
            hashlib.sha256((root / "selection.json").read_bytes()).hexdigest()
            != state["selection_sha256"]
        ):
            raise RuntimeError("Selection receipt changed")
        export = root / "controllers"
        export.mkdir(exist_ok=True)
        for arm in ("zero", "untrained", "cma", "random"):
            artifact = ResidualAdapter(
                ResidualSpec(**design["adapter"]["spec"]), arms[arm], design["cases"]["island"]
            ).artifact()
            write_json(export / (arm + ".json"), artifact)
    if set(expected) != set(jobs) or len(list((root / "episodes").glob("*/started.json"))) != len(
        jobs
    ):
        raise RuntimeError("Missing or unaccounted episode starts")
    summaries = {
        label: summarize([r for key, r in jobs.items() if key.startswith(label + "_")])
        for label in (
            "diagnostic_native",
            "diagnostic_zero",
            "diagnostic_untrained",
            "development_native",
            "development_zero",
            "development_untrained",
            "selection_native",
            "selection_zero",
            "selection_untrained",
            "selection_cma",
            "selection_random",
        )
    }
    report = {
        "verified": True,
        "episodes": len(jobs),
        "phase": state["phase"],
        "optimizer_updates_verified": len(optimizer["history"]),
        "resources": dict(total),
        "influence": dict(mechanism),
        "checks": checks,
        "summaries": summaries,
        "selection": selection,
        "common_panel_fitness_verified": True,
        "optimizer_advantage_established": False,
        "limitations": design["limitations"],
    }
    write_json(root / "audit.json", report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("verified", "episodes", "phase", "summaries", "resources", "influence")
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "audit"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    design = json.loads(args.protocol.read_text())
    if os.environ.get("RUNPOD_POD_ID") != design["execution_pod_id"]:
        raise RuntimeError("Only the declared rented GPU may execute this study")
    if args.mode == "audit":
        audit(args, design)
    else:
        with (
            patch.object(runtime, "ProposalAdapter", ResidualAdapter),
            patch.object(runtime, "ProposalSpec", ResidualSpec),
            patch.object(runtime, "ProposalPilot", ResidualStudy),
            patch.object(runtime, "audit", audit),
        ):
            runtime.train(args, design)


if __name__ == "__main__":
    main()
