"""JSON-only guards for availability-driven final-profile continuation.

Original profile/selection/training metadata stays authoritative. This module
only binds explicit actual execution hosts and deadlines for unstarted profiles.
"""

import copy
import datetime as dt
import hashlib
import json
from pathlib import Path


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def path_in(root, name):
    path = Path(name)
    if path.is_absolute() and "Safety-Dial" in path.parts:
        path = Path(*path.parts[path.parts.index("Safety-Dial") + 1 :])
    result = (root / path).resolve()
    require(result.is_relative_to(root.resolve()), "Relocation reference escaped evidence root")
    return result


def load_ref(root, ref):
    path = path_in(root, ref["path"])
    require(digest(path) == ref["sha256"], "Relocation reference SHA differs")
    return json.loads(path.read_text())


def authority(root, ref):
    value = load_ref(root, ref)
    require(
        value["schema"] == "lewm-frozen-final-relocation-authority-v1"
        and value["execution_enabled"] is True
        and value["maximum_duration_seconds"] == 7200
        and value["total_campaign_cap_usd"] == 13,
        "Wrong capacity-driven continuation authority",
    )
    for key in (
        "no_new_training",
        "no_outcome_based_changes",
        "no_profile_selection_from_outcomes",
    ):
        require(value[key] is True, "Outcome-driven continuation is prohibited")
    require(
        value["reason"] == "original_host_gpu_capacity_unavailable",
        "Relocation needs the declared availability reason",
    )
    load_ref(root, value["capacity_failure_evidence"])
    original_inputs = load_ref(root, value["original_analysis_inputs"])
    require(
        original_inputs["schema"] == "lewm-navigation-analysis-inputs-20261031-v3"
        and original_inputs["all_final_output_namespaces_absent_at_binding"] is True,
        "Exact original prospective analysis binding is required",
    )
    first = dt.datetime.fromisoformat(value["first_restart_utc"])
    deadline = dt.datetime.fromisoformat(value["new_deadline_utc"])
    require(
        first.tzinfo is not None
        and deadline.tzinfo is not None
        and 0 < (deadline - first).total_seconds() <= 7200,
        "New time authority exceeds the strict bound",
    )
    rows = value["profiles"]
    require(len(rows) == 10, "All eight remaining main/aux plus two mean profiles are required")
    require(len({r["protocol"]["sha256"] for r in rows}) == 10, "Duplicate relocation profile")
    require(len({r["output_namespace"] for r in rows}) == 10, "Duplicate output namespace")
    target = value["actual_final_host"]["pod_id"]
    require(all(r["pod_id"] == target for r in rows), "All remaining profiles move together")
    for row in rows:
        require(row["original_pod_id"] != target, "New final host must be explicit")
        protocol = load_ref(root, row["protocol"])
        require(
            protocol["evaluation"]["execution_pod_id"] == row["original_pod_id"]
            and protocol["evaluation"]["deadline_utc"] == value["original_deadline_utc"]
            and protocol["output_namespace"] == row["output_namespace"],
            "Original frozen identity or output differs",
        )
        require(
            value["source_sha256"].get(row["entrypoint"]["path"]) == row["entrypoint"]["sha256"],
            "Original evaluator is not pinned by the continuation authority",
        )
    return value


def relocated_row(value, entry):
    matches = [r for r in value["profiles"] if r["protocol"] == entry["protocol"]]
    require(len(matches) <= 1, "Ambiguous profile relocation")
    return matches[0] if matches else None


def actual_host(value, entry):
    return copy.deepcopy(
        value["actual_final_host"] if relocated_row(value, entry) else entry["host"]
    )


def verify_execution(root, entry, protocol, source, plan):
    value = authority(root, plan["final_execution_relocation"])
    row = relocated_row(value, entry)
    expected_host = actual_host(value, entry)
    require(source["pod_id"] == expected_host["pod_id"], "Actual final execution pod differs")
    if row is None:
        require("continuation" not in source, "Preserved completed profile was replayed")
        return expected_host
    record = load_ref(root, source["continuation"])
    require(
        record["schema"] == "lewm-frozen-final-relocation-execution-v1"
        and record["authority"] == plan["final_execution_relocation"]
        and record["original_protocol"] == entry["protocol"]
        and record["original_entrypoint"] == row["entrypoint"]
        and record["actual_final_host"] == expected_host
        and record["original_logical_host"] == entry["host"]
        and record["output_namespace"] == row["output_namespace"]
        and record["pod_id"] == expected_host["pod_id"],
        "Explicit continuation execution lineage differs",
    )
    expected = {
        "evaluation.deadline_utc": {
            "original": protocol["evaluation"]["deadline_utc"],
            "effective": value["new_deadline_utc"],
        },
        "evaluation.execution_pod_id": {
            "original": row["original_pod_id"],
            "effective": row["pod_id"],
        },
    }
    require(record["effective_override"] == expected, "Only host and deadline may change")
    require(
        record["all_other_protocol_fields_identical"] is True
        and record["source_assets_cases_seeds_parameters_budgets_unchanged"] is True
        and record["new_training_or_selection"] is False
        and record["capacity_failure_evidence"] == value["capacity_failure_evidence"],
        "Scientific settings changed during relocation",
    )
    wrapper = record["wrapper"]
    require(
        value["source_sha256"].get(wrapper["path"]) == wrapper["sha256"]
        and plan["source_sha256"].get(wrapper["path"]) == wrapper["sha256"],
        "Unpinned relocation wrapper",
    )
    recorded = dt.datetime.fromisoformat(record["recorded_utc"])
    require(
        dt.datetime.fromisoformat(value["first_restart_utc"])
        <= recorded
        <= dt.datetime.fromisoformat(source["started_utc"])
        < dt.datetime.fromisoformat(value["new_deadline_utc"]),
        "Continuation override was not recorded before execution",
    )
    return expected_host


def validate_binding(root, inputs, value):
    entries = inputs["profiles"]
    original_refs = {e["protocol"]["sha256"] for e in entries}
    remaining = [r for r in value["profiles"] if r["protocol"]["sha256"] in original_refs]
    require(len(entries) == 11 and len(remaining) == 8, "Original eleven-profile coverage changed")
    means = load_ref(root, value["original_terminal_mean_manifest"])
    require(
        means["schema"] == "lewm-terminal-mean-final-sidecar-v1"
        and means["all_main_outputs_absent_at_binding"] is True
        and {row["repeat_id"] for row in means["profiles"]} == {0, 1, 2},
        "Original all-three fixed terminal-mean sidecar required",
    )
    extra = {row["protocol"]["sha256"] for row in value["profiles"]} - original_refs
    require(
        extra
        == {row["protocol"]["sha256"] for row in means["profiles"] if row["repeat_id"] in (1, 2)},
        "Only the two unstarted frozen means may accompany the eight main/aux profiles",
    )
    preserved = [e for e in entries if relocated_row(value, e) is None]
    require(
        {(e["family"], e["repeat_id"]) for e in preserved}
        == {("primary50", 0), ("primary50", 2), ("secondary30", 0)},
        "Exactly the three completed Nav profiles must remain untouched",
    )
    for entry in entries:
        if not entry["family"].startswith("aux-"):
            continue
        cap = 50 if entry["family"].endswith("50") else 30
        family = "primary50" if cap == 50 else "secondary30"
        host = actual_host(value, entry)["pod_id"]
        candidates = sorted(
            [
                e
                for e in entries
                if e["family"] == family and actual_host(value, e)["pod_id"] == host
            ],
            key=lambda e: e["repeat_id"],
        )
        require(bool(candidates), "Auxiliary lacks an actual-host native control")
        auxiliary = load_ref(root, entry["protocol"])
        native = load_ref(root, candidates[0]["protocol"])
        native_rows = {j["case"]["case_id"]: j for j in native["jobs"] if j["arm"] == "native"}
        for job in auxiliary["jobs"]:
            baseline = native_rows[job["case"]["case_id"]]
            auxiliary_case = dict(job["case"])
            require(
                auxiliary_case.pop("task_cost_backend", None) == "decoded-current-target-v1",
                "Auxiliary must retain its exact declared decoded-cost treatment label",
            )
            require(
                auxiliary_case == baseline["case"]
                and job["environment_seed"] == baseline["environment_seed"]
                and job["policy_seed"] == baseline["policy_seed"]
                and job["max_actions"] == baseline["max_actions"],
                "Actual-host auxiliary/native case or common seed differs",
            )
    return {
        "main_profile_count": 11,
        "relocated_main_aux_profile_count": 8,
        "preserved_completed_main_profiles": 3,
        "auxiliary_pairing": "Lowest final repeat ID on ACTUAL final host at the same endpoint",
        "all_case_seed_pairs_checked": True,
        "duplicate_auxiliary_host_blocks_are_not_independent_hosts": True,
    }
