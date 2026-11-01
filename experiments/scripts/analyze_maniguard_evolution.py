"""Audit saved evolution work, source snapshots, optimizer state and final outcomes.

Runs on the rented host. Uses saved data only: zero simulator interactions and
zero model queries. Decision replay uses the same frozen validator as the runner,
not an independently implemented physical-safety validator.
"""

import argparse
from collections import Counter, defaultdict
import datetime as dt
import hashlib
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/helpers"))
from maniguard_evolution import digest
from maniguard_evolution_episode import atomic_json, totals, utc
from maniguard_evolution_scoring import analyze_episode, candidate_score, summarize
from maniguard_evolution_selection import choose_selection, nominate_elite, task_major_cases


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def expected_cases(config):
    search, validation = config["search"], config["validation"]
    expected = set()
    for g in range(search["generations"]):
        seed = search["policy_seeds"][g]
        for task in search["tasks"]:
            expected.add((f"g{g:03d}_native", task, seed))
            for i in range(search["population"]):
                expected.add((f"g{g:03d}_c{i:03d}", task, seed))
    for task in validation["tasks"]:
        for seed in validation["policy_seeds"]:
            for arm in ("native", "selected"):
                expected.add(("validation_" + arm, task, seed))
    if "selection" in config:
        for task, seed in task_major_cases(config["selection"]):
            for kind in ("native", "elite", "mean"):
                expected.add(("selection_" + kind, task, seed))
    return expected


def checkpoint_state_matches(checkpoint_state, json_state):
    # JSON serializes the scheduler's (task, seed) tuples as lists.
    normalized = json.loads(json.dumps(checkpoint_state, allow_nan=False))
    return normalized == json_state


def audit(run, *, allow_incomplete=False):
    run = Path(run).resolve()
    state = json.loads((run / "status.json").read_text())
    config = json.loads((run / "config.json").read_text())
    manifest = json.loads((run / "source-manifest.json").read_text())
    if manifest["configuration"] != config:
        raise ValueError("Configuration differs from the pre-run source manifest")
    terminal = state["status"] not in ("ready", "running")
    if not allow_incomplete and state["status"] != "completed":
        raise ValueError("Final audit requires completed training and validation")
    for name, expected in manifest["sources"].items():
        if name == "initial_controller":
            continue
        if sha(run / "source" / name) != expected:
            raise ValueError("Frozen source snapshot changed: " + name)
    with (run / "checkpoint.pkl").open("rb") as stream:
        checkpoint = pickle.load(stream)
    if checkpoint["config_hash"] != digest(config):
        raise ValueError("Checkpoint configuration binding differs")
    if terminal and not checkpoint_state_matches(checkpoint["state"], state):
        raise ValueError("Canonical checkpoint and terminal JSON state differ")
    if checkpoint["es"] is not None:
        if checkpoint["es"].countiter != len(checkpoint["state"]["generations"]):
            raise ValueError("Optimizer iteration count differs from completed generations")
    expected = expected_cases(config)
    actual, rows, groups, invalid = set(), {}, defaultdict(list), []
    valid_reservations = []
    for directory in sorted((run / "episodes").iterdir()):
        if not directory.is_dir():
            continue
        specification = json.loads((directory / "reservation.json").read_text())["specification"]
        key = (specification["label"], specification["task"], specification["policy_seed"])
        if key not in expected or key in actual:
            raise ValueError("Unexpected or duplicate reserved episode")
        actual.add(key)
        path = directory / "outcome.json"
        if not path.exists():
            if allow_incomplete:
                continue
            raise ValueError("Reserved episode has no terminal outcome")
        saved = json.loads(path.read_text())
        if not saved.get("valid"):
            if not allow_incomplete or state["status"] == "completed":
                raise ValueError("Invalid episode retained in completed run: " + directory.name)
            invalid.append({"job": directory.name, "error": saved.get("error"),
                            "error_type": saved.get("error_type")})
            continue
        controller = json.loads((directory / "controller.json").read_text())
        if controller != specification["controller"]:
            raise ValueError("Episode controller differs from its durable reservation")
        replay = analyze_episode(directory)
        for field, value in replay.items():
            if saved[field] != value:
                raise ValueError("Saved outcome differs from full replay: " + directory.name)
        state_key = f"{key[0]}_t{key[1]}_s{key[2]}"
        if state_key not in state["jobs"]:
            if allow_incomplete:
                continue
            raise ValueError("Episode is absent from the final optimizer job journal")
        if state["jobs"][state_key] != saved:
            raise ValueError("Optimizer journal differs from saved episode outcome")
        rows[key] = saved
        valid_reservations.append(json.loads((directory / "reservation.json").read_text())[
            "reserved_utc"])
        label = key[0]
        if "selection" in config:
            if label.startswith(("selection_", "validation_")):
                group = label
            elif label.endswith("_native"):
                group = "training_native_references"
            elif controller["parameters"] == state["initial"]:
                group = "training_native_anchors"
            else:
                group = "training_mutations"
        else:
            group = label if label.startswith("validation_") else (
                "training_native" if label.endswith("_native") else "training_candidates_all")
        groups[group].append(saved)
        if "selection" in config:
            if label.startswith("g"):
                g = int(label[1:4])
                parameters = (state["initial"] if label.endswith("_native") else
                              (state["generations"][g]["parameters"]
                               if g < len(state["generations"]) else
                               state["current"]["candidates"])[int(label[-3:])])
                enabled = not label.endswith("_native")
            elif label.startswith("selection_"):
                kind = label.removeprefix("selection_")
                parameters = state["selection"]["nominations"][kind]
                enabled = kind != "native"
            else:
                native = label == "validation_native"
                parameters = state["initial"] if native else state["selected"]
                enabled = not native and state["selected_kind"] != "native"
            if controller["parameters"] != parameters or controller["enabled"] != enabled:
                raise ValueError("Episode used a different controller or modification mode")
    if (not allow_incomplete or state["status"] == "completed") and actual != expected:
        raise ValueError("Final run does not contain the whole declared schedule")
    if "selection" in config and valid_reservations:
        first = dt.datetime.fromisoformat(min(valid_reservations))
        deadline = dt.datetime.fromisoformat(config["limits"]["deadline_utc"])
        if (deadline - first).total_seconds() < config["limits"][
                "minimum_remaining_seconds_before_first_episode"]:
            raise ValueError("First scored episode breached the seven-hour launch requirement")
        freeze = json.loads((run / "freeze-receipt.json").read_text())
        if first < dt.datetime.fromisoformat(freeze["frozen_utc"]):
            raise ValueError("An episode preceded the source freeze")
        if freeze["source_manifest_sha256"] != sha(run / "source-manifest.json"):
            raise ValueError("Source freeze receipt differs from the archived source manifest")
    for generation in state["generations"]:
        g = generation["generation"]
        seed = config["search"]["policy_seeds"][g]
        scores = [candidate_score([
            rows[(f"g{g:03d}_c{i:03d}", task, seed)] for task in config["search"]["tasks"]
        ], config["search"]["objective"]) for i in range(config["search"]["population"])]
        if scores != generation["scores"]:
            raise ValueError("Recorded generation fitness differs from complete episode panels")
    if state["selected"] is not None:
        controller = json.loads((run / "controller.json").read_text())
        if controller["parameters"] != state["selected"]:
            raise ValueError("Exported controller differs from selected checkpoint parameters")
        if checkpoint["es"] is not None and not allow_incomplete and "selection" not in config:
            if checkpoint["es"].result.xfavorite.tolist() != controller["parameters"]:
                raise ValueError("Selected controller is not the final CMA-ES phenotype mean")
    selection_result = None
    if "selection" in config and state.get("selection") is not None:
        selection = state["selection"]
        elite = nominate_elite(state["generations"], state["initial"])
        if (selection["nominations"]["native"] != state["initial"]
                or selection["nominations"]["elite"] != elite["parameters"]
                or selection["nominations"]["mean"] != state["final_mean"]
                or checkpoint["es"].result.xfavorite.tolist() != state["final_mean"]):
            raise ValueError("Selection nominations differ from training or optimizer state")
        provenance = {k: v for k, v in elite.items() if k != "parameters"}
        if selection["elite_provenance"] != provenance:
            raise ValueError("Elite nomination violates its deterministic tie rule")
        if selection["decision"] is not None:
            panels = {kind: [rows[("selection_" + kind, task, seed)]
                             for task, seed in task_major_cases(config["selection"])]
                      for kind in ("native", "elite", "mean")}
            selection_result = choose_selection(panels, config["search"]["objective"])
            selection_result["parameters"] = selection["nominations"][selection_result["winner"]]
            selection_result["elite_provenance"] = provenance
            saved_decision = json.loads((run / "selection.json").read_text())
            if ({k: v for k, v in saved_decision.items() if k != "frozen_utc"}
                    != selection_result or saved_decision != selection["decision"]
                    or sha(run / "selection.json") != selection["receipt_sha256"]
                    or state["selected_kind"] != selection_result["winner"]
                    or state["selected"] != selection_result["parameters"]
                    or controller["enabled"] != (selection_result["winner"] != "native")):
                raise ValueError("Frozen final selection differs from its declared rule")
            for path in (run / "episodes").glob("validation_*/reservation.json"):
                if json.loads(path.read_text())["reserved_utc"] < saved_decision["frozen_utc"]:
                    raise ValueError("Validation started before selection was frozen")
    arms = {name: summarize(items) for name, items in groups.items()}
    if state["status"] == "completed":
        for arm in ("native", "selected"):
            if state["validation"][arm] != arms["validation_" + arm]:
                raise ValueError("Terminal validation summary differs from episode outcomes")
    return {
        "schema": "maniguard-evolution-final-audit-v1", "audited_utc": utc(),
        "status": state["status"], "final": terminal,
        "complete_schedule": state["status"] == "completed" and len(rows) == len(expected),
        "declared_episodes": len(expected), "reserved_episodes": len(actual),
        "valid_replayed_episodes": len(rows),
        "replayed_controls": sum(row["controls"] for row in rows.values()),
        "completed_generations": state["generation"], "arms": arms,
        "invalid_attempts": invalid,
        "missing_cases": [list(case) for case in sorted(expected - set(rows))],
        "selection_result": selection_result,
        "selected_kind": state.get("selected_kind"),
        "validation_selected_is_native_repeat": state.get("selected_kind") == "native",
        "phase_summary": {name: dict(Counter(
            row["phase_diagnostics"]["farthest_valid_phase"] for row in items
            if "phase_diagnostics" in row)) for name, items in groups.items()},
        "training_candidates_are_not_final_policy_evaluation": True,
        "accounting": totals(run), "source_manifest_sha256": sha(run / "source-manifest.json"),
        "checkpoint_sha256": sha(run / "checkpoint.pkl"),
        "controller_sha256": sha(run / "controller.json") if (run / "controller.json").exists()
        else None,
        "source_snapshots_verified": True, "checkpoint_loadable": True,
        "new_simulator_interactions": 0, "new_model_queries": 0,
        "validator": ("Same frozen action/outcome validator as the runner, "
                      "not independent validation"),
        "held_out_tasks_used": [], "evolutionary_advantage_established": False,
        "limitations": ["One optimizer seed and a small development validation panel",
                        "No matched fixed/random/ordinary-learning comparison in this run",
                        "Privileged simulator inputs; no held-out or generalization claim"] + (
            ["V3 conservative exploration and measured selection are development evidence only",
             "Native fallback is a repeated control, not an evolved improvement"]
            if "selection" in config else [
                "Frozen v2 auxiliary carry-stage ordering discrepancy is documented; "
                "strict safe-completion scoring is unaffected"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    result = audit(args.run, allow_incomplete=args.allow_incomplete)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out, result)
    print(json.dumps({"status": result["status"], "final": result["final"],
                      "valid_replayed_episodes": result["valid_replayed_episodes"],
                      "replayed_controls": result["replayed_controls"]}), flush=True)
