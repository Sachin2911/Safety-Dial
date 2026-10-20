"""Check the recorded study against its frozen design without launching work.

The physical reader supplies verified outcomes. This separate check reconstructs
optimizer proposals, case allocation and finalist selection rather than trusting
a completion flag. Remote preservation and scientific interpretation remain
separate completion requirements.
"""

import calendar
import hashlib
import json
import subprocess
import time
from collections import Counter
from itertools import pairwise
from pathlib import Path

import numpy as np

from helpers.maniguard_latent import LatentController
from helpers.maniguard_latent_hardware import GPU_UUIDS


def initial_controllers(seed):
    """Reconstruct the four prospectively declared matrices independently."""
    values = [np.zeros((4, 44), np.float32) for _ in range(3)]
    values[1][0, 0], values[2][0, 0] = np.arctanh(.5), -np.arctanh(.5)
    values.append(np.random.default_rng(seed).uniform(-1, 1, (4, 44)).astype(np.float32))
    return [{"id": f"initial-{i}", "weights": w.tolist(), "parent": None}
            for i, w in enumerate(values)]


def proposals(method, generation, parents, rng):
    """Reconstruct the declared retained-incumbent and offspring distribution."""
    cohort = []
    for slot in range(4):
        parent = parents[0 if slot == 0 else (slot - 1) % 2]
        if slot == 0:
            weights = np.asarray(parent["weights"], np.float32)
        elif method == "ea":
            weights = np.clip(np.asarray(parent["weights"])
                              + rng.normal(0, .25, (4, 44)), -1, 1).astype(np.float32)
        else:
            weights = rng.uniform(-1, 1, (4, 44)).astype(np.float32)
        cohort.append({"id": f"{method}-g{generation}-{slot}", "weights": weights.tolist(),
                       "parent": parent["id"] if slot == 0 or method == "ea" else None})
    return cohort


def execution_errors(protocol, queue):
    """Check declared wave order and prevent overlapping same-job replacements."""
    waves = [[f"initial-{i}-case{case}" for case in (0, 1)] for i in range(4)]
    for generation in (1, 2):
        for slot in range(4):
            methods = ("ea", "random") if slot % 2 == 0 else ("random", "ea")
            waves.extend([[f"{method}-g{generation}-{slot}-case{case}" for case in (0, 1)]
                          for method in methods])
    for block, panel in enumerate(protocol["audit_blocks"]):
        names = [f"audit-b{block}-{condition}" for condition in panel["order"]]
        waves.extend([names[:2], names[2:]])
    errors = []
    previous_finished = calendar.timegm(time.strptime(protocol["selected_utc"], "%Y-%m-%dT%H:%M:%SZ"))
    previous_complete = True
    for wave in waves:
        attempts = [a for a in queue["attempts"] if a["job_id"] in wave]
        if not attempts:
            previous_complete = False
            continue
        first_start = min(a["started_epoch"] for a in attempts)
        if not previous_complete or first_start < previous_finished:
            errors.append("Frozen wave order or prior-wave completion differs: " + ", ".join(wave))
        for name in wave:
            retries = sorted((a for a in attempts if a["job_id"] == name),
                             key=lambda a: a["started_epoch"])
            for before, after in pairwise(retries):
                if after["started_epoch"] < before.get("finished_epoch", float("inf")):
                    errors.append("Same-job physical attempts overlap: " + name)
        previous_complete = all(queue["jobs"].get(name, {}).get("status") == "completed"
                                for name in wave)
        previous_finished = max(a.get("finished_epoch", float("inf")) for a in attempts)
    return errors


def verify_design(protocol, protocol_hash, state, seal, queue, report, episode_jobs):
    """Return explicit errors/pending evidence; partial work cannot pass completion."""
    errors, pending = [], []

    def check(condition, message):
        if not condition:
            errors.append(message)

    check(protocol["status"] == "capacity_selected_frozen" and protocol["capacity"]["fits"],
          "No prospectively selected complete comparison")
    check(state["protocol_sha256"] == protocol_hash, "Search protocol checksum differs")
    rows = {row["attempt"]: row for row in report["attempts"]}
    check(len(rows) == len(report["attempts"]), "Attempt identifiers repeat")
    completed = {}
    for job_id, planned in queue["jobs"].items():
        if planned["status"] == "completed":
            name = planned["attempt_ids"][-1]
            row = rows.get(name)
            check(row is not None, "Completed job missing from physical accounting: " + job_id)
            if row is not None and row["valid_full_episode"]:
                completed[job_id] = row
                check(planned.get("score") == row["score"], "Queue fitness differs: " + job_id)

    initial = initial_controllers(protocol["comparison_template"]["initial_seed"])
    expected = {}
    for item in protocol["development_episodes"]:
        expected[item["id"]] = {key: item[key] for key in ("family", "scene", "seed")}
        expected[item["id"]]["stage"] = "development"
        expected[item["id"]]["controller_sha256"] = LatentController(
            initial[item["initial_controller"]]["weights"]).sha256()

    def register(cohort, generation):
        for candidate in cohort:
            for case, panel in enumerate(protocol["training_panels"][generation]):
                expected[f"{candidate['id']}-case{case}"] = {
                    "family": protocol["family"], **panel,
                    "stage": "shared_initial" if generation == 0
                    else f"search_generation_{generation}",
                    "worker": (case + generation) % 2,
                    "controller_sha256": LatentController(candidate["weights"]).sha256()}

    def ranked(cohort):
        scores = {}
        for candidate in cohort:
            cases = [completed.get(f"{candidate['id']}-case{i}") for i in range(2)]
            if any(row is None for row in cases):
                return None
            scores[candidate["id"]] = sum(row["score"] for row in cases)
        return sorted(cohort, key=lambda c: (-scores[c["id"]], c["id"]))

    register(initial, 0)
    check(state["cohorts"].get("initial") == initial, "Shared initialization matrices changed")
    expected_candidates = {c["id"]: (c, 0) for c in initial}
    finalists = {"native": initial[0]}
    initial_ranking = ranked(initial)
    if initial_ranking is None:
        pending.append("Complete all eight shared initialization episodes")
    else:
        finalists["fixed"] = next(c for c in initial_ranking
                                  if c["id"] in ("initial-1", "initial-2"))

    for method in ("ea", "random"):
        previous = initial
        rng = np.random.default_rng(protocol["comparison_template"]["optimizer_seed"])
        generation_proposed = 0
        for generation in (1, 2):
            key = f"{method}-g{generation}"
            parents = ranked(previous)
            if key not in state["cohorts"]:
                pending.append("Complete cohort " + key)
                break
            check(parents is not None, "Cohort proposed before complete parents: " + key)
            if parents is None:
                break
            cohort = proposals(method, generation, parents, rng)
            check(state["cohorts"][key] == cohort, "Frozen offspring/ancestry differ: " + key)
            register(cohort, generation)
            expected_candidates.update({c["id"]: (c, generation) for c in cohort})
            generation_proposed = generation
            previous = cohort
        optimizer = state["optimizers"][method]
        check(optimizer["generation"] == generation_proposed, "Optimizer generation differs: " + method)
        check(optimizer["rng"] == rng.bit_generator.state, "Optimizer RNG differs: " + method)
        seen = set()
        for candidate in optimizer["candidates"]:
            name = candidate["id"]
            check(name not in seen, "Duplicate candidate: " + name)
            seen.add(name)
            expected_candidate = expected_candidates.get(name)
            check(expected_candidate is not None, "Undeclared candidate: " + name)
            check(name.startswith(("initial-", method + "-g")),
                  "Candidate belongs to the other optimizer: " + name)
            if expected_candidate is None:
                continue
            target, generation = expected_candidate
            for key in ("weights", "parent"):
                check(candidate.get(key) == target[key], "Recorded candidate differs: " + name + "/" + key)
            check(candidate["generation"] == generation, "Candidate generation differs: " + name)
            check(candidate["sha256"] == LatentController(target["weights"]).sha256(),
                  "Candidate checksum differs: " + name)
            case_rows = [completed.get(f"{name}-case{i}") for i in range(2)]
            check(all(row is not None for row in case_rows), "Fitness precedes physical cases: " + name)
            if all(row is not None for row in case_rows):
                check(candidate["scores"] == [row["score"] for row in case_rows],
                      "Fitness differs from independently read outcomes: " + name)
        if len(optimizer["candidates"]) != 12:
            pending.append("Record twelve complete candidate evaluations for " + method)
        if generation_proposed == 2 and ranked(previous) is not None:
            finalists[method] = ranked(previous)[0]

    audit_attempts = [a for a in queue["attempts"] if a["stage"] == "independent_audit"]
    if seal is None:
        pending.append("Seal all four finalists before independent checking")
        check(not audit_attempts, "Independent physics started without a finalist seal")
    else:
        check(seal["protocol_sha256"] == protocol_hash, "Finalist seal protocol differs")
        check(state["finalists"] == seal["finalists"], "Search state differs from immutable seal")
        check(set(finalists) == {"native", "fixed", "ea", "random"},
              "Finalists sealed before all selection evidence")
        for condition, candidate in finalists.items():
            sealed = seal["finalists"][condition]
            check(sealed["candidate_id"] == candidate["id"]
                  and sealed["sha256"] == LatentController(candidate["weights"]).sha256(),
                  "Finalist not selected by frozen rule: " + condition)
        for attempt in audit_attempts:
            check(attempt["started_epoch"] >= seal["sealed_epoch"],
                  "Independent physics precedes seal: " + attempt["id"])
        for attempt in queue["attempts"]:
            if attempt["stage"] in ("shared_initial", "search_generation_1", "search_generation_2"):
                check(attempt.get("finished_epoch", float("inf")) <= seal["sealed_epoch"],
                      "Finalists sealed before training physics finished: " + attempt["id"])

    for block, panel in enumerate(protocol["audit_blocks"]):
        for position, condition in enumerate(panel["order"]):
            definition = {"family": protocol["family"], "scene": panel["scene"],
                          "seed": panel["seed"], "stage": "independent_audit",
                          "condition": condition, "block": block, "worker": position % 2}
            if condition in finalists:
                definition["controller_sha256"] = LatentController(
                    finalists[condition]["weights"]).sha256()
            expected[f"audit-b{block}-{condition}"] = definition

    for name, definition in expected.items():
        if name not in completed:
            pending.append("Complete physical job " + name)
        planned = queue["jobs"].get(name)
        if planned is None:
            continue
        for key, value in definition.items():
            if key != "controller_sha256":
                check(planned["definition"].get(key) == value, "Planned case differs: " + name + "/" + key)
        for attempt in (a for a in queue["attempts"] if a["job_id"] == name):
            job = episode_jobs.get(attempt["id"])
            if job is None:
                check(attempt["status"] != "completed", "Completed episode job missing: " + attempt["id"])
                continue
            for key in ("family", "scene", "seed", "controller_sha256"):
                if key in definition:
                    check(job.get(key) == definition[key], "Executed case differs: " + name + "/" + key)
            check(job["max_steps_override"] is None and not job["random_policy"]
                  and job["source_profile"] == "high_fidelity"
                  and job["action_horizon"] == {"jar": 2000, "clutter": 1100}[job["family"]]
                  and job["adapter"]["kind"] == "identity" and job["adapter"]["horizon"] == 8,
                  "Native execution contract differs: " + name)
            if definition["stage"] != "development":
                check(job["protocol_sha256"] == protocol_hash, "Executed protocol differs: " + name)
            if "worker" in definition:
                device = definition["worker"]
                check(job["simulation_gpu_uuid"] == job["policy_gpu_uuid"] == GPU_UUIDS[device],
                      "Executed GPU allocation differs: " + name)

    full = [a for a in queue["attempts"] if not a["engineering"]]
    timing_errors = execution_errors(protocol, queue)
    errors.extend(timing_errors)
    if any(a["status"] not in ("completed", "invalid") for a in full):
        pending.append("Classify every full attempt and finish all physical workers")
    check(all(a["job_id"] in expected for a in full), "Undeclared full physical attempt")
    check(len(full) <= 64 and sum(a["status"] == "invalid" for a in full) <= 4,
          "Full-attempt or replacement ceiling exceeded")
    cost = report["accounting"]
    check(cost["engineering_starts"] <= 6 and cost["engineering_controls"] <= 96,
          "Engineering allowance exceeded")
    check(cost["decoder_calls"] == {"jar": 160, "clutter": 160}, "Bounded decoder audit incomplete/exceeded")
    for family in ("jar", "clutter"):
        authority = report["authority"].get(family, {})
        check(authority.get("authority_pass") is True
              and authority.get("same_noise_identity_pass") is True
              and authority.get("summary_sha256") == protocol["capacity"]["families"][family][
                  "authority_summary_sha256"], "Frozen decoder authority evidence differs: " + family)
    check(len([a for a in full if a["status"] == "completed"]) == len(completed),
          "Completed full-attempt accounting differs")
    for condition in ("native", "fixed", "ea", "random"):
        actual = Counter(row["worker"] for row in completed.values()
                         if row["stage"] == "independent_audit" and row["condition"] == condition)
        if actual != {0: 2, 1: 2}:
            pending.append("Complete balanced four-block checking for " + condition)
    milestones = {"shared-initial", "generation-1", "generation-2", "sealed-finalists", "independent-audit"}
    if not milestones.issubset(state["completed_milestones"]):
        pending.append("Complete all five preservation milestones")
    return {"schema": "maniguard-latent-design-verification-v1",
            "protocol_sha256": protocol_hash, "report_observed_epoch": cost.get("observed_epoch"),
            "physical_design_complete": not errors and not pending,
            "started_attempts_follow_frozen_wave_order": not timing_errors,
            "errors": errors, "pending": pending, "verified_full_jobs": len(completed),
            "expected_full_jobs": 60, "full_attempts": len(full),
            "scope": "Frozen physical design and optimizer-selection reconstruction. Independently verify remote preservation and final scientific claims before declaring the goal complete."}


def runtime_errors(job, trace, steps, *, require_device_receipts=True):
    """Inspect actual runtime receipts, independent of planned device/horizon fields."""
    errors = []
    expected_steps = list(range(0, steps, 8))
    for kind in ("query", "latent_features", "noise_receipt"):
        if [row["step"] for row in trace if row["type"] == kind] != expected_steps:
            errors.append("Native eight-action query schedule differs: " + kind)
    metadata = [row["metadata"] for row in trace if row["type"] == "policy_metadata"]
    if (len(metadata) != 1 or metadata[0].get("audit_service") is not False
            or metadata[0].get("model_shape") != [16, 32]
            or metadata[0].get("checkpoint_revision") != job["policy"]["revision"]):
        errors.append("Actual policy checkpoint, model shape or GPU differs")
    if require_device_receipts:
        if len(metadata) == 1 and metadata[0].get("physical_gpu_uuid") != job.get("policy_gpu_uuid"):
            errors.append("Actual policy GPU differs")
        devices = [row.get("actual_gpu_uuid") for row in trace if row["type"] == "cuda_device_verified"]
        if not job.get("simulation_gpu_uuid") or devices != [job["simulation_gpu_uuid"]]:
            errors.append("Actual simulator GPU differs")
    contracts = [row for row in trace if row["type"] == "runtime_contract"]
    if (len(contracts) != 1 or contracts[0].get("overview_shape") != [640, 640, 3]
            or contracts[0].get("wrist_shape") != [640, 640, 3]
            or not any("franka_panda_longfinger" in path
                       for path in contracts[0].get("reference_asset_paths", []))):
        errors.append("Actual native cameras or long-finger asset differ")
    return errors


def source_errors(job, hashes, policy):
    errors = []
    if job.get("policy") != policy:
        errors.append("Family checkpoint/config differs from frozen public pins")
    for name, expected in hashes.items():
        if job.get("code_hashes", {}).get(name) != expected:
            errors.append("Actor, encoder, execution or label source changed: " + name)
    return errors


def verify(run, protocol, report):
    run, protocol = Path(run), Path(protocol)
    specification = json.loads(protocol.read_text())
    state_bytes = (run / "search/state.json").read_bytes()
    state = json.loads(state_bytes)
    seal_path = run / "search/sealed-finalists.json"
    seal = json.loads(seal_path.read_text()) if seal_path.exists() else None
    queue_bytes = (run / "queue.json").read_bytes()
    queue = json.loads(queue_bytes)
    jobs = {}
    for attempt in queue["attempts"]:
        path = Path(attempt["output"]) / "job.json"
        if path.exists():
            jobs[attempt["id"]] = json.loads(path.read_text())
    result = verify_design(specification,
                           hashlib.sha256(protocol.read_bytes()).hexdigest(),
                           state, seal, queue, report, jobs)
    result["search_state_sha256"] = hashlib.sha256(state_bytes).hexdigest()
    result["queue_sha256"] = hashlib.sha256(queue_bytes).hexdigest()
    root = Path(__file__).resolve().parents[2]
    critical = ["experiments/helpers/" + name + ".py" for name in [
        "maniguard_latent", "maniguard_latent_policy", "maniguard_latent_runtime",
        "maniguard_latent_source", "maniguard_runtime", "maniguard_source",
        "maniguard_adaptation", "maniguard_phase2_features", "maniguard_phase2_outcomes"]]
    critical += ["experiments/scripts/maniguard_latent_episode.py",
                 "configs/maniguard/phase2/encoder.json", "configs/maniguard/phase2/completion.json",
                 specification["source_pins"]]
    frozen = {name: subprocess.check_output(
        ["git", "-C", str(root), "show", specification["source_commit"] + ":" + name])
        for name in critical}
    policies = json.loads(frozen[specification["source_pins"]])["policies"]
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in frozen.items()}
    result["frozen_critical_source_commit"] = specification["source_commit"]
    result["frozen_critical_source_hashes"] = hashes
    completed = {row["attempt"]: row for row in report["attempts"] if row["valid_full_episode"]}
    result["limitations"] = []
    for attempt in queue["attempts"]:
        if attempt["id"] not in completed:
            continue
        trace = [json.loads(line) for line in
                 (Path(attempt["output"]) / "trace.jsonl").read_text().splitlines()]
        steps = completed[attempt["id"]]["policy_controls"]
        job = jobs[attempt["id"]]
        early_development = (completed[attempt["id"]]["stage"] == "development"
                             and job["family"] == "jar" and "policy_gpu_uuid" not in job)
        if early_development:
            result["limitations"].append(attempt["id"] + ": early development records planned GPU mapping; per-process UUID receipts were introduced before the paired comparison")
        result["errors"].extend(attempt["id"] + ": " + error
                                for error in runtime_errors(job, trace, steps,
                                                            require_device_receipts=not early_development))
        result["errors"].extend(attempt["id"] + ": " + error for error in source_errors(
            job, {} if completed[attempt["id"]]["stage"] == "development" else hashes,
            policies[job["family"]]))
    result["physical_design_complete"] = not result["errors"] and not result["pending"]
    return result
