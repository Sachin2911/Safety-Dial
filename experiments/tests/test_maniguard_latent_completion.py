"""A completion flag cannot conceal changed cases, finalists, budgets or execution."""

import copy
import hashlib
import json
from pathlib import Path

from helpers.maniguard_latent import CandidateSearch, LatentController, initial_weights
from helpers.maniguard_latent_completion import (
    execution_errors,
    runtime_errors,
    source_errors,
    verify_design,
)
from helpers.maniguard_latent_hardware import GPU_UUIDS

ROOT = Path(__file__).resolve().parents[2]


def complete_record():
    protocol = json.loads((ROOT / "configs/maniguard/latent-evolution/capacity-20261020-v2.json").read_text())
    protocol_hash = "frozen-protocol"
    protocol["selected_utc"] = "1970-01-01T00:00:00Z"
    state = {"protocol_sha256": protocol_hash, "cohorts": {}, "optimizers": {},
             "completed_milestones": ["shared-initial", "generation-1", "generation-2",
                                      "sealed-finalists", "independent-audit"]}
    queue = {"jobs": {}, "attempts": []}
    report = {"attempts": [], "accounting": {"engineering_starts": 6,
              "engineering_controls": 96, "decoder_calls": {"jar": 160, "clutter": 160}}}
    report["authority"] = {family: {"authority_pass": True, "same_noise_identity_pass": True,
                                    "summary_sha256": protocol["capacity"]["families"][family][
                                        "authority_summary_sha256"]} for family in ("jar", "clutter")}
    jobs = {}

    def physical(name, definition, weights, score):
        attempt_id = name + "-a0"
        queue["jobs"][name] = {"definition": definition, "status": "completed",
                               "attempt_ids": [attempt_id], "score": score}
        audit = definition["stage"] == "independent_audit"
        queue["attempts"].append({"id": attempt_id, "job_id": name, "status": "completed",
                                  "stage": definition["stage"], "engineering": False,
                                  "started_epoch": (1001 if audit else 100) + len(queue["attempts"]),
                                  "finished_epoch": (1001 if audit else 100) + len(queue["attempts"]) + .5})
        report["attempts"].append({"attempt": attempt_id, "valid_full_episode": True,
                                   "score": score, "stage": definition["stage"],
                                   "worker": definition.get("worker"),
                                   "condition": definition.get("condition")})
        worker = definition.get("worker", 0)
        jobs[attempt_id] = {key: definition[key] for key in ("family", "scene", "seed")}
        jobs[attempt_id].update(controller_sha256=LatentController(weights).sha256(),
                                max_steps_override=None, random_policy=False,
                                source_profile="high_fidelity", protocol_sha256=protocol_hash,
                                action_horizon={"jar": 2000, "clutter": 1100}[definition["family"]],
                                adapter={"kind": "identity", "horizon": 8},
                                simulation_gpu_uuid=GPU_UUIDS[worker], policy_gpu_uuid=GPU_UUIDS[worker])

    def evaluate(candidate, generation, scores):
        for i, case in enumerate(protocol["training_panels"][generation]):
            name = f"{candidate['id']}-case{i}"
            physical(name, {"id": name, "family": protocol["family"], **case,
                            "stage": "shared_initial" if generation == 0 else
                            f"search_generation_{generation}", "worker": (i + generation) % 2},
                     candidate["weights"], scores[i])

    initial = [{"id": f"initial-{i}", "weights": weights.tolist(), "parent": None}
               for i, weights in enumerate(initial_weights())]
    state["cohorts"]["initial"] = initial
    searches = {method: CandidateSearch(method) for method in ("ea", "random")}
    for i, candidate in enumerate(initial):
        scores = [[0, -2], [1, 0], [1, 1], [0, 0]][i]
        evaluate(candidate, 0, scores)
        for search in searches.values():
            search.record(candidate["id"], candidate["weights"], scores)
    for generation in (1, 2):
        for method, search in searches.items():
            cohort = search.propose()
            state["cohorts"][f"{method}-g{generation}"] = cohort
        for i in range(4):
            for method in (["ea", "random"] if i % 2 == 0 else ["random", "ea"]):
                search = searches[method]
                candidate = state["cohorts"][f"{method}-g{generation}"][i]
                scores = ([[0, 0], [1, 0], [-2, 1], [1, 1]] if generation == 1 else
                          [[-2, -2], [0, 0], [1, 1], [0, 1]])[i]
                evaluate(candidate, generation, scores)
                search.record(candidate["id"], candidate["weights"], scores, candidate["parent"])
    state["optimizers"] = {method: search.state() for method, search in searches.items()}
    chosen = {"native": initial[0], "fixed": initial[2],
              **{method: search.ranked()[0] for method, search in searches.items()}}
    state["finalists"] = {condition: {"candidate_id": candidate["id"],
                                     "sha256": LatentController(candidate["weights"]).sha256()}
                           for condition, candidate in chosen.items()}
    seal = {"protocol_sha256": protocol_hash, "sealed_epoch": 1000,
            "finalists": copy.deepcopy(state["finalists"])}
    for item in protocol["development_episodes"]:
        definition = {key: item[key] for key in ("id", "family", "scene", "seed")}
        definition["stage"] = "development"
        physical(item["id"], definition, initial[item["initial_controller"]]["weights"], 0)
    for block, panel in enumerate(protocol["audit_blocks"]):
        for position, condition in enumerate(panel["order"]):
            name = f"audit-b{block}-{condition}"
            physical(name, {"id": name, "family": protocol["family"], "scene": panel["scene"],
                            "seed": panel["seed"], "stage": "independent_audit", "block": block,
                            "condition": condition, "worker": position % 2},
                     chosen[condition]["weights"], 1)
    return protocol, protocol_hash, state, seal, queue, report, jobs


def test_complete_record_reconstructs_selection_and_budgets():
    result = verify_design(*complete_record())
    assert result["physical_design_complete"], result
    assert result["verified_full_jobs"] == result["full_attempts"] == 60


def test_wrong_finalist_rejected_even_when_state_and_seal_agree():
    args = complete_record()
    wrong = {"candidate_id": "ea-g2-0", "sha256": hashlib.sha256(b"wrong").hexdigest()}
    args[2]["finalists"]["ea"] = wrong
    args[3]["finalists"]["ea"] = wrong
    result = verify_design(*args)
    assert not result["physical_design_complete"]
    assert any("Finalist not selected" in error for error in result["errors"])


def test_wrong_case_gpu_and_unpaid_replacement_are_visible():
    args = complete_record()
    args[-1]["audit-b0-ea-a0"]["seed"] += 1
    args[-1]["audit-b1-random-a0"]["policy_gpu_uuid"] = "another-device"
    args[4]["attempts"].extend([{**args[4]["attempts"][0], "id": f"extra-{i}",
                                "job_id": "undeclared", "status": "invalid"} for i in range(5)])
    result = verify_design(*args)
    assert not result["physical_design_complete"]
    assert any("Executed case differs" in error for error in result["errors"])
    assert any("GPU allocation" in error for error in result["errors"])
    assert any("ceiling exceeded" in error for error in result["errors"])


def test_mutated_offspring_and_early_audit_are_rejected():
    args = complete_record()
    args[2]["cohorts"]["ea-g1"][1]["weights"][0][0] += .001
    next(a for a in args[4]["attempts"] if a["stage"] == "independent_audit")["started_epoch"] = 999
    result = verify_design(*args)
    assert any("offspring/ancestry" in error for error in result["errors"])
    assert any("precedes seal" in error for error in result["errors"])


def test_missing_audit_cannot_be_declared_complete():
    args = complete_record()
    state, queue, report = args[2], args[4], args[5]
    audit_ids = {name for name in queue["jobs"] if name.startswith("audit-")}
    queue["jobs"] = {name: item for name, item in queue["jobs"].items() if name not in audit_ids}
    queue["attempts"] = [a for a in queue["attempts"] if a["job_id"] not in audit_ids]
    report["attempts"] = [a for a in report["attempts"] if a["stage"] != "independent_audit"]
    state["completed_milestones"].remove("independent-audit")
    result = verify_design(*args)
    assert not result["errors"]
    assert result["pending"] and not result["physical_design_complete"]
    assert result["verified_full_jobs"] == 44


def test_runtime_receipts_require_native_schedule_camera_and_actual_device():
    job = {"policy_gpu_uuid": GPU_UUIDS[0], "simulation_gpu_uuid": GPU_UUIDS[0],
           "policy": {"revision": "pinned"}}
    trace = [{"type": kind, "step": step} for kind in ("query", "latent_features", "noise_receipt")
             for step in (0, 8)]
    trace += [{"type": "policy_metadata", "metadata": {
        "audit_service": False, "model_shape": [16, 32], "physical_gpu_uuid": GPU_UUIDS[0],
        "checkpoint_revision": "pinned"}},
        {"type": "cuda_device_verified", "actual_gpu_uuid": GPU_UUIDS[0]},
        {"type": "runtime_contract", "overview_shape": [640, 640, 3], "wrist_shape": [640, 640, 3],
         "reference_asset_paths": ["franka_panda_longfinger.usda"]}]
    assert not runtime_errors(job, trace, 9)
    trace[1]["step"] = 16
    trace[-1]["wrist_shape"] = [256, 256, 3]
    trace[-2]["actual_gpu_uuid"] = GPU_UUIDS[1]
    errors = runtime_errors(job, trace, 9)
    assert len(errors) == 3


def test_source_checks_reject_changed_weights_interface_and_checkpoint():
    job = {"policy": {"revision": "pinned"}, "code_hashes": {"actor.py": "declared"}}
    assert not source_errors(job, {"actor.py": "declared"}, {"revision": "pinned"})
    errors = source_errors(job, {"actor.py": "changed"}, {"revision": "another-checkpoint"})
    assert len(errors) == 2


def test_wave_order_and_overlapping_replacements_are_rejected():
    protocol, _, _, _, queue, _, _ = complete_record()
    assert not execution_errors(protocol, queue)
    first = queue["attempts"][0]
    first["status"] = "invalid"
    duplicate = {**first, "id": "overlapping-replacement", "status": "completed",
                 "started_epoch": first["started_epoch"] + .25}
    queue["attempts"].append(duplicate)
    assert any("attempts overlap" in error for error in execution_errors(protocol, queue))
    queue["attempts"].pop()
    later = next(a for a in queue["attempts"] if a["job_id"] == "initial-1-case0")
    later["started_epoch"] = first["started_epoch"]
    assert any("wave order" in error for error in execution_errors(protocol, queue))
