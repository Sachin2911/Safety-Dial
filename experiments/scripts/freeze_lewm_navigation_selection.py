"""Read-only nominee inspection and stdlib-only NAV selection/final-lock freezing.

Subcommands write new protocol/lock JSON only. They never import a model, load
numeric arrays, open final images, launch workers or execute controller episodes.
"""

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("native", "strong_anchor", "tuned_fixed", "cma", "random", "reinforce")
OPTIMIZERS = ("cma", "random", "reinforce")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    require(not path.exists(), "Never overwrite a frozen protocol/lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return {"path": str(path), "sha256": digest(path)}


def genes(value):
    require(len(value) == 10, "The shared controller requires ten genes")
    result = [float(v) for v in value]
    require(
        all(math.isfinite(v) and -6 <= v <= 6 for v in result),
        "Genotype outside frozen shared bounds",
    )
    return result


def effective_genes(value):
    return tuple(struct.unpack("f", struct.pack("f", v))[0] for v in genes(value))


def fitness(rows):
    require(bool(rows), "A nominee requires a full nonempty panel")
    for row in rows:
        require(not row.get("error"), "Technical episodes cannot supply selection fitness")
        require(not row.get("extra", {}).get("frame_flags"), "Flagged input requires review")
        require(
            math.isfinite(row["progress"]) and 0 <= row["progress"] <= 1,
            "Invalid original-task progress",
        )
        require(
            not (row["safe_completion"] and (row["violation"] or row["unfinished"])),
            "Inconsistent episode outcome",
        )
    return (
        sum(bool(r["safe_completion"]) for r in rows),
        -sum(bool(r["violation"]) for r in rows),
        sum(r["progress"] for r in rows),
    )


def completed_training(run_root, protocol_path, reservation, method):
    protocol = load(protocol_path)
    state = load(run_root / "state.json")
    freeze = load(run_root / "source-freeze.json")
    require(freeze["protocol"] == protocol, "Producer protocol differs from prospective file")
    require(state["identity"] == freeze["identity"], "Producer identity differs")
    require(
        state["phase"] == "complete" and state["active"] is None,
        "Training is not complete/budget-stopped",
    )
    require(
        not state["unfinalized_job_ids"] and state["action_accounting_exact"],
        "Unresolved failed/partial training work",
    )
    require(
        protocol["method"] == method and protocol["repeat_id"] == reservation["repeat_id"],
        "Wrong optimizer/repeat producer",
    )
    require(
        protocol["evaluation"]["execution_pod_id"] == reservation["host"]["pod_id"],
        "Nominee trained on another host",
    )
    require(
        protocol["source_sha256"] == reservation["source_sha256"], "Producer source family changed"
    )
    require(
        protocol["evaluation"]["maximum_episode_actions"]
        == reservation["evaluation"]["maximum_episode_actions"],
        "Producer horizon differs",
    )
    nominees = load(run_root / "nominees.json")
    require(
        nominees["anchor"] == state["anchor"] and nominees["trained"] == state["best_trained"],
        "Nominees/state differ",
    )
    return (
        protocol,
        state,
        nominees,
        {
            "protocol_sha256": digest(protocol_path),
            "state_sha256": digest(run_root / "state.json"),
            "source_freeze_sha256": digest(run_root / "source-freeze.json"),
            "nominees_sha256": digest(run_root / "nominees.json"),
            "identity": state["identity"],
            "actual_environment_actions": state["actual_environment_actions"],
            "residual_unspent_allowance": state["remaining_action_allowance"],
            "optimizer_updates": state["optimizer_updates"],
        },
    )


def verify_training_panel(run_root, nominee, protocol, identity):
    rows, references = [], []
    expected_cases = {c["case_id"]: c for c in protocol["training_panel"]}
    require(len(nominee["job_ids"]) == len(expected_cases), "Nominee panel is partial")
    for job_id in nominee["job_ids"]:
        folder = run_root / "jobs" / job_id
        row, started = load(folder / "result.json"), load(folder / "started.json")
        require(
            row["identity"] == started["identity"] == identity, "Candidate episode identity differs"
        )
        spec = started["job"]
        require(
            row["input_sha256"] == started["input_sha256"] == canonical_hash(spec),
            "Candidate full job input hash does not reconstruct",
        )
        case = expected_cases.get(spec["case_id"])
        require(
            case is not None and spec["case"] == case, "Candidate complete case metadata differs"
        )
        require(
            spec["environment_seed"] == case["seed"]
            and spec["max_actions"] == protocol["max_episode_actions"],
            "Candidate environment seed/action cap differs",
        )
        require(
            spec["method"] == protocol["method"] and spec["repeat_id"] == protocol["repeat_id"],
            "Candidate method/repeat differs",
        )
        expected_policy_seed = int.from_bytes(
            hashlib.sha256(f"{protocol['optimizer_seed']}/{job_id}".encode()).digest()[:4], "big"
        ) % (2**31 - 1)
        require(spec["policy_seed"] == expected_policy_seed, "Candidate categorical seed differs")
        require(
            row["status"] == "complete" and row["accounting_finalized"],
            "Candidate is failed/partial",
        )
        require(
            genes(spec["parameters"]) == genes(nominee["parameters"]),
            "Candidate genotype provenance differs",
        )
        rows.append(row)
        references.append(
            {
                "job_id": job_id,
                "result_sha256": digest(folder / "result.json"),
                "started_sha256": digest(folder / "started.json"),
            }
        )
    require(
        {r["case_id"] for r in rows} == set(expected_cases),
        "Candidate cases differ from frozen TRAIN panel",
    )
    require(
        list(fitness(rows)) == nominee["fitness"], "Saved candidate fitness does not reconstruct"
    )
    return references


def fixed_nominee(run_root, protocol_path, reservation):
    protocol, state, freeze = (
        load(protocol_path),
        load(run_root / "state.json"),
        load(run_root / "source-freeze.json"),
    )
    require(
        freeze["protocol_sha256"] == digest(protocol_path),
        "Fixed-grid prospective protocol differs",
    )
    require(freeze["sources"] == reservation["source_sha256"], "Fixed-grid source family differs")
    require(
        state["phase"] == "complete" and state["active"] is None, "Fixed tuning is not complete"
    )
    require(
        protocol["evaluation"]["execution_pod_id"] == reservation["host"]["pod_id"],
        "Fixed-grid host differs",
    )
    planned = protocol["jobs"]
    require(set(state["jobs"]) == {j["job_id"] for j in planned}, "Fixed grid is incomplete")
    panel_size = len({j["case"]["case_id"] for j in planned})
    require(len(planned) % panel_size == 0, "Fixed grid has a partial panel")
    candidates = []
    for offset in range(0, len(planned), panel_size):
        block = planned[offset : offset + panel_size]
        parameters = genes(block[0]["parameters"])
        require(
            all(genes(j["parameters"]) == parameters for j in block),
            "Mixed fixed genotypes in panel",
        )
        rows = []
        for job in block:
            folder = run_root / "jobs" / job["job_id"]
            row, started = load(folder / "result.json"), load(folder / "started.json")
            require(
                row == state["jobs"][job["job_id"]] and started == job,
                "Fixed-grid state/result/started prospective job differ",
            )
            require(
                row["case_id"] == job["case"]["case_id"] and row["method"] == job["method"],
                "Fixed-grid complete case/method identity differs",
            )
            require(
                genes(row["parameters"]) == genes(job["parameters"]),
                "Executed fixed-grid weights differ",
            )
            require(
                row["environment_seed"] == job["environment_seed"]
                and row["policy_seed"] == job["policy_seed"],
                "Fixed-grid environment/policy seeds differ",
            )
            require(
                row["environment_actions"]
                <= job.get("max_actions", protocol["evaluation"]["maximum_episode_actions"]),
                "Fixed-grid episode exceeded its cap",
            )
            rows.append(row)
        score = fitness(rows)
        require(
            all(
                row["environment_seed"] == job["environment_seed"]
                for row, job in zip(rows, block, strict=True)
            ),
            "Fixed-grid seed differs",
        )
        candidates.append((score, parameters, [j["job_id"] for j in block]))
    # Strict comparison keeps the first candidate, the separately evaluated anchor,
    # on ties; subsequent grid ties keep ascending prospective index.
    selected = candidates[0]
    for candidate in candidates[1:]:
        if candidate[0] > selected[0]:
            selected = candidate
    return selected[1], {
        "kind": "prospective_fixed_grid",
        "selected_candidate_job_ids": selected[2],
        "training_fitness": list(selected[0]),
        "protocol_sha256": digest(protocol_path),
        "source_freeze_sha256": digest(run_root / "source-freeze.json"),
        "state_sha256": digest(run_root / "state.json"),
        "is_trained": False,
        "note": "Fixed configuration selected by a prospective grid; no evolutionary credit",
    }


def materialize(args):
    reservation = load(args.reservation)
    require(
        reservation["schema"] == "lewm-navigation-selection-reservation-v1", "Unknown reservation"
    )
    repeat = reservation["repeat_id"]
    artifact_root = args.artifact_root.resolve()
    peer_dir = args.reservation.parent
    anchor = genes(reservation["candidate_sources"]["strong_anchor"])
    parameters = {"native": anchor, "strong_anchor": anchor}
    provenance = {
        "native": {"kind": "native", "is_trained": False},
        "strong_anchor": {"kind": "initial_anchor", "is_trained": False},
    }
    for method in OPTIMIZERS:
        source = reservation["candidate_sources"][method]
        run_root = artifact_root / source["output_namespace"]
        protocol_path = peer_dir / f"training-{method}-r{repeat}.json"
        protocol, state, nominees, refs = completed_training(
            run_root, protocol_path, reservation, method
        )
        require(genes(nominees["anchor"]["parameters"]) == anchor, "Common anchor drifted")
        trained = nominees["trained"]
        nominee = nominees["anchor"] if trained is None else trained
        rows = verify_training_panel(run_root, nominee, protocol, state["identity"])
        parameters[method] = genes(nominee["parameters"])
        trained_flag = trained is not None and effective_genes(
            parameters[method]
        ) != effective_genes(anchor)
        numerical_only_change = parameters[method] != anchor and not trained_flag
        require(
            not trained_flag or nominee["provenance"]["kind"] == "trained_candidate",
            "Missing trained-candidate provenance",
        )
        provenance[method] = {
            **refs,
            "kind": "trained_candidate" if trained_flag else "anchor_fallback",
            "is_trained": trained_flag,
            "float32_equivalent_to_anchor": effective_genes(parameters[method])
            == effective_genes(anchor),
            "parameter_difference_only_float32_conversion": numerical_only_change,
            "producer": nominee["provenance"],
            "completed_updates_before_nominee": nominee["provenance"].get(
                "completed_updates_before_proposal",
                nominee["provenance"].get("completed_updates_before_evaluation", 0),
            ),
            "do_not_attribute_later_updates_to_this_genotype": True,
            "panel_results": rows,
            "training_fitness": nominee["fitness"],
        }
        if method == "reinforce" and trained is not None:
            update = nominee["provenance"]["update"]
            update_path = run_root / f"update-{update:03}.json"
            update_record = load(update_path)
            require(update_record["nominee"] == nominee, "PG nominee/update provenance differs")
            provenance[method]["selected_update_statistics"] = update_record["statistics"]
            provenance[method]["selected_update_sha256"] = digest(update_path)
    fixed_root = artifact_root / reservation["candidate_sources"]["tuned_fixed"]["output"]
    parameters["tuned_fixed"], provenance["tuned_fixed"] = fixed_nominee(
        fixed_root, peer_dir / f"fixed-tuning-r{repeat}.json", reservation
    )
    sources = dict(reservation["source_sha256"])
    for name, expected in sources.items():
        require(digest(ROOT / name) == expected, "Frozen selection source changed: " + name)
    own_name = str(Path(__file__).relative_to(ROOT))
    sources[own_name] = digest(Path(__file__))
    cases = reservation["selection_panel"]
    require(
        len(cases) == 8 and len({c["source_episode"] for c in cases}) == 8,
        "Exactly eight fresh cases",
    )
    cap = reservation["evaluation"]["maximum_episode_actions"]
    jobs = []
    for case in cases:
        for arm in ARMS:
            jobs.append(
                {
                    "job_id": arm + "_" + case["case_id"] + "_s" + str(case["seed"]),
                    "arm": arm,
                    "method": "native" if arm == "native" else arm,
                    "parameters": parameters[arm],
                    "case": copy.deepcopy(case),
                    "environment_seed": case["seed"],
                    "policy_seed": case["seed"],
                    "max_actions": cap,
                    "repeat_id": repeat,
                }
            )
    require(len(jobs) == reservation["maximum_episode_starts"], "Reserved episode cap differs")
    require(
        len(jobs) * cap == reservation["maximum_environment_actions"], "Reserved action cap differs"
    )
    protocol = {
        "schema": "lewm-navigation-comparison-v1",
        "name": f"navigation-selection-r{repeat}",
        "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "output_namespace": args.namespace,
        "evaluation": reservation["evaluation"],
        "case_preparation": reservation["case_preparation"],
        "source_files": sorted(sources),
        "source_sha256": sources,
        "jobs": jobs,
        "maximum_episode_starts": len(jobs),
        "maximum_native_resets": len(jobs),
        "maximum_environment_actions": len(jobs) * cap,
        "optimizer_updates": 0,
        "repeat_id": repeat,
        "host": reservation["host"],
        "reservation": {"sha256": digest(args.reservation), "path": str(args.reservation)},
        "nominee_provenance": provenance,
        "controller_static_parameters": {"strong_anchor": anchor, **parameters},
        "selection_rule": reservation["selection"],
        "final_lock_rule": reservation["final_lock"],
        "duplicate_parameters_are_still_charged": True,
        "final_pixels_read": 0,
        "new_model_simulator_calls_by_materializer": 0,
    }
    return write(args.out, protocol)


def lock(args):
    protocol, state = load(args.protocol), load(args.state)
    require(state["phase"] == "complete" and state["active"] is None, "Selection not complete")
    freeze = load(args.state.parent / "source-freeze.json")
    require(freeze["protocol_sha256"] == digest(args.protocol), "Selection source freeze differs")
    require(freeze["sources"] == protocol["source_sha256"], "Selection source family differs")
    require(
        set(state["jobs"]) == {j["job_id"] for j in protocol["jobs"]}, "Selection panel incomplete"
    )
    rows, parameters = {arm: [] for arm in ARMS}, {}
    for job in protocol["jobs"]:
        row = state["jobs"][job["job_id"]]
        require(
            row["case_id"] == job["case"]["case_id"] and row["method"] == job["method"],
            "Selection job identity differs",
        )
        require(
            row["environment_seed"] == job["environment_seed"]
            and row["policy_seed"] == job["policy_seed"],
            "Selection seed differs",
        )
        require(
            genes(row["parameters"]) == genes(job["parameters"]),
            "Executed selection weights differ",
        )
        rows[job["arm"]].append(row)
        parameters[job["arm"]] = genes(job["parameters"])
    scores = {arm: fitness(values) for arm, values in rows.items()}
    require(all(len(values) == 8 for values in rows.values()), "Every arm requires eight cases")
    anchor_score = scores["strong_anchor"]
    selected = {}
    for arm in ARMS:
        candidate_wins = arm not in ("native", "strong_anchor") and scores[arm] > anchor_score
        retain = arm in ("native", "strong_anchor") or candidate_wins
        source_arm = arm if retain else "strong_anchor"
        producer = protocol["nominee_provenance"][source_arm]
        selected[arm] = {
            "parameters": parameters[source_arm],
            "selection_fitness": list(scores[source_arm]),
            "candidate_fitness": list(scores[arm]),
            "anchor_fitness": list(anchor_score),
            "selected_source_arm": source_arm,
            "strict_candidate_improvement": candidate_wins,
            "selected_is_trained": candidate_wins and bool(producer["is_trained"]),
            "provenance": producer,
            "ties_keep_labeled_anchor": True,
        }
    value = {
        "schema": "lewm-navigation-selection-lock-v1",
        "locked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repeat_id": protocol["repeat_id"],
        "host": protocol["host"],
        "protocol_sha256": digest(args.protocol),
        "state_sha256": digest(args.state),
        "source_freeze_sha256": digest(args.state.parent / "source-freeze.json"),
        "source_sha256": protocol["source_sha256"],
        "evaluation": protocol["evaluation"],
        "scores": {arm: list(score) for arm, score in scores.items()},
        "selected": selected,
        "denominator_each": 8,
        "actual_environment_actions": sum(
            row["environment_actions"] for row in state["jobs"].values()
        ),
        "residual_unspent_action_allowance": protocol["maximum_environment_actions"]
        - sum(row["environment_actions"] for row in state["jobs"].values()),
        "final_pixels_read": 0,
        "new_model_simulator_calls": 0,
        "no_best_repeat_selection": True,
        "interpretation": "Lock all six arms for this repeat. Anchor selection is untrained and receives no evolutionary credit. Preserve all3 learner repeats for final evaluation.",
    }
    return write(args.out, value)


def seal(args):
    locks = [load(path) for path in args.locks]
    require(
        sorted(value["repeat_id"] for value in locks) == [0, 1, 2],
        "All three repeats must be retained",
    )
    require(
        all(
            value["schema"] == "lewm-navigation-selection-lock-v1"
            and value["final_pixels_read"] == 0
            for value in locks
        ),
        "Invalid final-opening lock",
    )
    require(
        len({value["host"]["pod_id"] for value in locks}) == 2, "Both physical-host blocks required"
    )
    require(
        len({value["evaluation"]["maximum_episode_actions"] for value in locks}) == 1,
        "Final endpoints differ",
    )
    require(
        all(value["source_sha256"] == locks[0]["source_sha256"] for value in locks),
        "Source families differ across repeat locks",
    )
    common_fields = (
        "world_model",
        "motion_model",
        "scalers",
        "memory_path",
        "memory_sha256",
        "controller_backend",
        "controller_config",
        "goal_readout",
        "planner",
        "precision",
        "body_radius",
        "margin",
        "motion_allowance",
        "maximum_episode_actions",
    )
    require(
        all(
            all(
                value["evaluation"].get(key) == locks[0]["evaluation"].get(key)
                for key in common_fields
            )
            for value in locks
        ),
        "Shared model/controller/information/planner contract differs",
    )
    reference_assets = locks[0]["evaluation"]["asset_sha256"]
    common_asset_names = {
        locks[0]["evaluation"][key] for key in ("world_model", "motion_model", "scalers")
    }
    common_asset_names.add(locks[0]["evaluation"]["memory_path"])
    if locks[0]["evaluation"].get("goal_readout"):
        common_asset_names.add(locks[0]["evaluation"]["goal_readout"]["path"])
    require(
        all(
            all(
                name in value["evaluation"]["asset_sha256"]
                and value["evaluation"]["asset_sha256"][name] == reference_assets[name]
                for name in common_asset_names
            )
            for value in locks
        ),
        "Frozen checkpoint/head/scaler/memory/readout asset SHA identities differ",
    )
    value = {
        "schema": "lewm-navigation-final-opening-lock-v1",
        "locked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selection_locks": [
            {
                "path": str(path),
                "sha256": digest(path),
                "repeat_id": value["repeat_id"],
                "host": value["host"],
            }
            for path, value in zip(args.locks, locks, strict=True)
        ],
        "all_three_repeat_selections_fixed": True,
        "best_repeat_selection_forbidden": True,
        "final_pixels_read": 0,
        "new_model_simulator_calls": 0,
        "next_stage": "Freeze metadata-only final admission, exact selected parameter hashes, per-host native/strong/tuned-fixed controls, all case/source/seed/job/action caps and reuse rules before loading any final pixel/model input. This file starts no preparation or experiment.",
    }
    return write(args.out, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    material = commands.add_parser("materialize")
    material.add_argument("--reservation", type=Path, required=True)
    material.add_argument("--artifact-root", type=Path, default=ROOT)
    material.add_argument("--out", type=Path, required=True)
    material.add_argument("--namespace", required=True)
    material.set_defaults(function=materialize)
    locker = commands.add_parser("lock")
    locker.add_argument("--protocol", type=Path, required=True)
    locker.add_argument("--state", type=Path, required=True)
    locker.add_argument("--out", type=Path, required=True)
    locker.set_defaults(function=lock)
    sealer = commands.add_parser("seal")
    sealer.add_argument("--locks", type=Path, nargs=3, required=True)
    sealer.add_argument("--out", type=Path, required=True)
    sealer.set_defaults(function=seal)
    args = parser.parse_args()
    print(json.dumps(args.function(args), indent=2))


if __name__ == "__main__":
    main()
