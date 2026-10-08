"""Pure identities and provenance for paired Push-T acquisition and oracle reports.

Call ``candidate_pool_identity`` after proposal/seed selection and before executing
any candidate. Validate the reference inputs once, then each seed identity before
its first query. These helpers never read files, construct models or contact HF.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import re

import numpy as np


SHA256 = re.compile(r"[0-9a-f]{64}\Z")
BANK_FILES = {"roots.json", "branches.h5", "layouts.json"}
MATCH_FIELDS = ("e2_report_sha256", "recipe", "input_sha256", "evaluation_bank_identity",
                "acquisition_root_identity", "acquisition_seeds", "seed_branches",
                "rounds", "budgets_added")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _integer(value, label, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _candidate(candidate):
    root = _integer(candidate.root_index, "Candidate root")
    if not isinstance(candidate.key, (tuple, list)) or len(candidate.key) != 2:
        raise ValueError("Candidate keys must contain root and pool position")
    key = [_integer(x, "Candidate key") for x in candidate.key]
    if key[0] != root:
        raise ValueError("Candidate key and root_index disagree")
    kind = candidate.proposal.kind
    if not isinstance(kind, str) or not kind:
        raise ValueError("Candidate kind must be a nonempty string")
    tape = np.asarray(candidate.proposal.tape)
    if tape.dtype.kind not in "fiu" or not tape.size or not np.isfinite(tape).all():
        raise ValueError("Candidate tape must contain finite numeric actions")
    return {"key": key, "root_index": root, "kind": kind,
            "tape_dtype": tape.dtype.str, "tape_shape": list(tape.shape),
            "tape_sha256": hashlib.sha256(tape.tobytes(order="C")).hexdigest()}


def candidate_pool_identity(pool, schedule, seed_candidates) -> dict:
    """Fingerprint ordered keys, root/kind, exact tape dtype/shape/bytes and seed plan.

    Tape strides are irrelevant: C-order action values define execution. Numeric
    precision, endian order, signed zero, shape and candidate order remain distinct.
    Model feature dictionaries are intentionally excluded from prequery identity.
    """
    candidates = list(pool.candidates)
    if not candidates or any(c.executed for c in candidates):
        raise ValueError("Fingerprint a nonempty, entirely unqueried candidate pool")
    records = [_candidate(c) for c in candidates]
    by_key = {tuple(record["key"]): record for record in records}
    if len(by_key) != len(records):
        raise ValueError("Candidate pool has duplicate keys")
    schedule = [_integer(root, "Scheduled root") for root in schedule]
    capacity = Counter(record["root_index"] for record in records)
    if not schedule or any(n > capacity[root] for root, n in Counter(schedule).items()):
        raise ValueError("Root schedule exceeds the candidate pool")
    seeds = [_candidate(c) for c in seed_candidates]
    keys = [record["key"] for record in seeds]
    if not seeds or len({tuple(k) for k in keys}) != len(keys):
        raise ValueError("Common seed candidates must be nonempty and unique")
    if any(by_key.get(tuple(record["key"])) != record for record in seeds):
        raise ValueError("Common seed candidates differ from the frozen pool")
    if [record["root_index"] for record in seeds] != schedule[:len(seeds)]:
        raise ValueError("Common seed roots must match the schedule prefix")
    identity = {"protocol_version": 1, "candidate_count": len(records),
                "root_count": len(capacity), "schedule_count": len(schedule),
                "seed_count": len(seeds), "pool_sha256": _sha(records),
                "root_schedule_sha256": _sha(schedule), "common_seed_sha256": _sha(keys),
                "root_schedule": schedule, "common_seed_keys": keys}
    return {**identity, "sha256": _sha(identity)}


def _hashes(mapping, label, required=()):
    if not isinstance(mapping, dict) or not mapping or not set(required).issubset(mapping):
        raise ValueError(f"Missing {label} hashes")
    if any(not isinstance(k, str) or not k or not isinstance(v, str) or not SHA256.fullmatch(v)
           for k, v in mapping.items()):
        raise ValueError(f"Invalid {label} SHA256 map")


def _protocol(report):
    for key in MATCH_FIELDS:
        if key not in report:
            raise ValueError(f"Missing reference input field: {key}")
    if not isinstance(report["e2_report_sha256"], str) or not SHA256.fullmatch(report["e2_report_sha256"]):
        raise ValueError("Invalid E2 report SHA256")
    if not isinstance(report["recipe"], dict) or not report["recipe"]:
        raise ValueError("Missing fixed adaptation recipe")
    _hashes(report["input_sha256"], "input")
    banks = report["evaluation_bank_identity"]
    if not isinstance(banks, dict) or set(banks) != {"dev", "test", "stress"}:
        raise ValueError("Expected development, test and stress bank identities")
    for name, identity in banks.items():
        _hashes(identity, f"{name} bank", BANK_FILES)
    _hashes(report["acquisition_root_identity"], "acquisition root", BANK_FILES | {"contexts.npz"})
    seeds, budgets, rounds = (report[k] for k in ("acquisition_seeds", "budgets_added", "rounds"))
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Acquisition seeds must be a nonempty unique ordered list")
    for seed in seeds:
        _integer(seed, "Acquisition seed")
    if not isinstance(rounds, list) or not rounds:
        raise ValueError("Missing acquisition rounds")
    total, expected = 0, []
    for n in rounds:
        total += _integer(n, "Round budget", 1)
        expected.append(total)
    if budgets != expected:
        raise ValueError("Additional budgets differ from cumulative acquisition rounds")
    _integer(report["seed_branches"], "Common seed count", 1)
    _json({key: report[key] for key in MATCH_FIELDS})


def _pool_identity(identity, report):
    if not isinstance(identity, dict) or identity.get("protocol_version") != 1:
        raise ValueError("Missing versioned candidate pool identity")
    payload = {key: value for key, value in identity.items() if key != "sha256"}
    if identity.get("sha256") != _sha(payload):
        raise ValueError("Candidate pool identity digest does not match its contents")
    for field in ("pool_sha256", "root_schedule_sha256", "common_seed_sha256"):
        if not isinstance(identity.get(field), str) or not SHA256.fullmatch(identity[field]):
            raise ValueError(f"Invalid candidate {field}")
    schedule, keys = identity.get("root_schedule"), identity.get("common_seed_keys")
    if not isinstance(schedule, list) or not isinstance(keys, list):
        raise ValueError("Missing exact root schedule or common seed keys")
    if identity["root_schedule_sha256"] != _sha(schedule) or identity["common_seed_sha256"] != _sha(keys):
        raise ValueError("Candidate pool schedule/common seed digest differs")
    if (identity.get("schedule_count") != len(schedule)
            or len(schedule) != report["seed_branches"] + report["budgets_added"][-1]
            or identity.get("seed_count") != len(keys) or len(keys) != report["seed_branches"]):
        raise ValueError("Candidate pool schedule/common seed counts differ from declared budgets")
    if (_integer(identity.get("candidate_count"), "Candidate count", 1) < len(schedule)
            or _integer(identity.get("root_count"), "Root count", 1) < len(set(schedule))):
        raise ValueError("Candidate pool cannot supply the declared root schedule")
    if (any(not isinstance(key, list) or len(key) != 2 for key in keys)
            or len({tuple(key) for key in keys}) != len(keys)
            or [key[0] for key in keys] != schedule[:len(keys)]):
        raise ValueError("Invalid common seed candidate keys")
    for root in schedule:
        _integer(root, "Scheduled root")
    for key in keys:
        for value in key:
            _integer(value, "Common seed key")


def validate_reference(reference, current_report) -> None:
    """Reject incomplete/diagnostic or mismatched normal references before queries.

    The oracle run must declare the same ordered seeds and all budgets. The current
    report may have no per-seed pool identities yet; validate_reference_pool checks
    each as it is constructed, before that seed's first simulator action.
    """
    if (reference.get("status") != "complete"
            or reference.get("gate", {}).get("diagnostic") is not False
            or reference.get("gate", {}).get("e2_passed") is not True):
        raise ValueError("Oracle reference must be a complete qualified prospective study")
    _protocol(reference)
    _protocol(current_report)
    for key in MATCH_FIELDS:
        if _json(reference[key]) != _json(current_report[key]):
            raise ValueError(f"Oracle reference differs in {key}")
    arms = reference.get("arms", {})
    if not isinstance(arms, dict) or not {"random", "boundary"}.issubset(arms) or "oracle" in arms:
        raise ValueError("Oracle reference must contain normal random and boundary arms")
    seeds = {str(seed) for seed in reference["acquisition_seeds"]}
    identities = reference.get("candidate_pool_identities", {})
    if set(identities) != seeds:
        raise ValueError("Reference must identify every declared seed's candidate pool")
    for seed in seeds:
        _pool_identity(identities[seed], reference)
    for arm in ("random", "boundary"):
        if set(arms[arm]) != seeds:
            raise ValueError(f"Reference {arm} does not contain all declared acquisition seeds")
        for seed in seeds:
            entry = arms[arm][seed]
            if [point.get("budget_added") for point in entry.get("curve", [])] != reference["budgets_added"]:
                raise ValueError(f"Reference {arm}/{seed} lacks complete budgets")
            if entry.get("root_schedule") != identities[seed]["root_schedule"]:
                raise ValueError(f"Reference {arm}/{seed} used a different root schedule")
    for seed, identity in current_report.get("candidate_pool_identities", {}).items():
        validate_reference_pool(reference, seed, identity)


def validate_reference_pool(reference, seed, identity) -> None:
    """Require identical ordered proposals, common seed keys and complete root plan."""
    _pool_identity(identity, reference)
    recorded = reference.get("candidate_pool_identities", {}).get(str(seed))
    if recorded is None or _json(recorded) != _json(identity):
        raise ValueError(f"Oracle candidate pool differs from reference seed {seed}")


def result_provenance(report) -> dict:
    """Return detached JSON manifest data with supplied HF receipts and point costs.

    Saved references are supplied by the caller; this function never resolves a tag
    or uploads data. Missing upload fields remain explicit nulls for local-only runs.
    """
    data = {key: report[key] for key in MATCH_FIELDS if key in report}
    for key in ("run_id", "status", "e2_report", "candidate_pool_identities", "upstream_revisions",
                "root_bank_hf_revision", "oracle_reference", "oracle_pool_results", "root_collection_ledger",
                "evaluation_ledger", "wall_clock_s", "no_update_rows"):
        if key in report:
            data[key] = report[key]
    data.setdefault("upstream_revisions", {})
    checkpoints = []
    for arm, by_seed in report.get("arms", {}).items():
        for seed, entry in by_seed.items():
            for point in entry.get("curve", []):
                checkpoints.append({"arm": arm, "acquisition_seed": int(seed),
                    "budget_added": point["budget_added"], "weights_sha256": point.get("weights_sha256"),
                    "hf_repo": point.get("hf_repo"), "hf_revision": point.get("hf_revision"),
                    "bank_hf_revision": entry.get("bank_hf_revision"),
                    "evaluation_rows": point.get("evaluation_rows"),
                    "costs": {key: point[key] for key in (
                        "charged_steps", "ledger", "n_queried_branches", "n_retained_branches",
                        "n_trainable_branches", "n_censored_round", "n_invalid_training_round",
                        "discarded_padding_steps", "train_time_s", "training_time_cumulative_s",
                        "scoring_time_cumulative_s") if key in point},
                    "seed_training_time_s": entry.get("seed_training_time_s"),
                    "oracle_pool_steps": entry.get("oracle_pool_steps")})
    data["checkpoints"] = checkpoints
    return json.loads(_json(data))
