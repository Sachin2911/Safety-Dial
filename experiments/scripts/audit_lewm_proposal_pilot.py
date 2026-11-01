"""Independently bind saved optimizer history, candidate fitness and validation weights."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.lewm_proposal_adapter import ProposalSpec
from helpers.lewm_proposal_pilot import panel_fitness, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    d = json.loads(args.protocol.read_text())
    state = json.loads((args.out / "state.json").read_text())
    base = json.loads((args.out / "audit.json").read_text())
    if not base["verified"] or state["phase"] != "complete":
        raise RuntimeError("Saved trajectory audit must finish first")
    # Only load this experiment's own trusted local checkpoint, never external pickle.
    with (args.out / "optimizer.pkl").open("rb") as stream:
        checkpoint = pickle.load(stream)
    history = checkpoint["history"]
    if (
        len(history) != d["search"]["generations"]
        or checkpoint["es"].countiter != len(history)
        or checkpoint["pending"] is not None
    ):
        raise RuntimeError("Optimizer did not finish the frozen number of updates")
    jobs = state["jobs"]
    expected = {}
    initial = ProposalSpec(**d["adapter"]["spec"]).initial_parameters(d["adapter"]["seed"]).tolist()

    def add(label, weights, case, seed):
        key = f"{label}_{case}_s{seed}"
        if key in expected:
            raise RuntimeError("Duplicate frozen job key")
        expected[key] = weights
        row = jobs[key]
        if (
            row["seed"] != seed
            or row["case_kind"] != case
            or row["adapter"] != (weights is not None)
        ):
            raise RuntimeError("Saved job differs from schedule")
        folder = args.out / "episodes" / key
        if weights is not None:
            artifact = json.loads((folder / "controller.json").read_text())
            if artifact["parameters"] != weights or artifact["spec"] != d["adapter"]["spec"]:
                raise RuntimeError("Executed candidate weights differ from genotype")
        elif (folder / "controller.json").exists() or row["resources"].get(
            "adapter_forward_calls", 0
        ):
            raise RuntimeError("Native episode contains a proposal adapter")
        return row

    for case in d["training"]["cases"]:
        add("initial", initial, case, d["training"]["seed_start"])
    nominees = []
    for g, generation in enumerate(history):
        if len(generation["parameters"]) != d["search"]["population"]:
            raise RuntimeError("Population differs")
        seed = d["training"]["seed_start"] + g
        for case in d["training"]["cases"]:
            add(f"g{g:03}_native", None, case, seed)
        for i, weights in enumerate(generation["parameters"]):
            rows = [add(f"g{g:03}_c{i:03}", weights, case, seed) for case in d["training"]["cases"]]
            score = panel_fitness(rows)
            if score != generation["fitness"][i]:
                raise RuntimeError("CMA update fitness differs from saved panel outcomes")
            nominees.append((score, -g, -i, weights))
    elite_score, mg, mi, elite = max(nominees, key=lambda value: value[:3])
    selection = state["selection"]
    if selection["elite_provenance"] != {
        "generation": -mg,
        "candidate_index": -mi,
        "training_fitness": elite_score,
    }:
        raise RuntimeError("Elite nomination rule differs")
    candidates = {"native": None, "initial": initial, "elite": elite}
    scores = {}
    for label, weights in candidates.items():
        rows = [
            add("selection_" + label, weights, case, seed) for case, seed in d["selection"]["cases"]
        ]
        scores[label] = panel_fitness(rows)
    winner = max(candidates, key=lambda label: scores[label])
    if (
        scores != selection["scores"]
        or winner != selection["winner"]
        or candidates[winner] != selection["parameters"]
    ):
        raise RuntimeError("Selection did not follow the prospective rule")
    for case, seed in d["validation"]["cases"]:
        add("validation_native", None, case, seed)
        add("validation_selected", candidates[winner], case, seed)
    if set(expected) != set(jobs) or len(expected) != d["maximum_valid_episodes"]:
        raise RuntimeError("Actual jobs differ from frozen schedule")
    starts = list((args.out / "episodes").glob("*/started.json"))
    if len(starts) != len(jobs):
        raise RuntimeError("Unaccounted episode starts")
    pairs = []
    for case, seed in d["validation"]["cases"]:
        native, selected = [
            jobs[f"validation_{arm}_{case}_s{seed}"] for arm in ("native", "selected")
        ]
        pairs.append(
            {
                "case": case,
                "seed": seed,
                "native_safe_completion": native["safe_completion"],
                "selected_safe_completion": selected["safe_completion"],
                "native_actions": native["actions"],
                "selected_actions": selected["actions"],
                "native_predictor_steps": native["resources"]["predictor_steps"],
                "selected_predictor_steps": selected["resources"]["predictor_steps"],
            }
        )
    value = {
        "verified": True,
        "optimizer_updates_verified": len(history),
        "all_76_job_identities_and_genotypes_verified": len(expected) == 76,
        "selection_winner": winner,
        "selection_fitness": scores,
        "selection_sha256": hashlib.sha256((args.out / "selection.json").read_bytes()).hexdigest(),
        "validation_pairs": pairs,
        "matched_caps_not_equal_actual_work": True,
        "optimizer_advantage_established": False,
        "new_simulator_interactions": 0,
        "new_frozen_model_queries": 0,
    }
    write_json(args.out / "optimizer-audit.json", value)
    print(json.dumps(value), flush=True)


if __name__ == "__main__":
    main()
