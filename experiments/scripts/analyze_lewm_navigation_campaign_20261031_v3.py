"""V3 prospectively relocated-host frozen rented-host final analysis and publication exports for20261031 NAV.

bind-inputs reads prospective protocol metadata only and refuses existing final
outcomes. analyze reads completed saved evidence, never models/images/simulators.
All numerical work and plotting are restricted to the declared rented host.
"""

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("native", "strong_anchor", "tuned_fixed", "cma", "random", "reinforce")
LABELS = ("Native", "Strong fixed", "Tuned fixed", "CMA", "Random search", "REINFORCE")
METRICS = ("safe_completion", "violation", "unfinished", "engaged")
CONDITIONS = ("none", "circle3", "circle5", "circle7")
BASELINES = ("strong_anchor", "tuned_fixed", "random", "reinforce")


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def resolve(root, value):
    path = Path(value)
    if path.is_absolute() and path.is_relative_to(root.resolve()):
        result = path.resolve()
        require(result.is_relative_to(root.resolve()), "Evidence symlink escaped artifact root")
        return result
    if path.is_absolute() and "Safety-Dial" in path.parts:
        path = Path(*path.parts[path.parts.index("Safety-Dial") + 1 :])
    result = (root / path).resolve()
    require(
        result.is_relative_to(root.resolve()),
        "Evidence path must stay inside the retained project tree",
    )
    return result


def reference(path):
    path = path.resolve()
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}


def load_reference(root, ref):
    path = resolve(root, ref["path"])
    require(digest(path) == ref["sha256"], "Frozen metadata reference changed: " + ref["path"])
    return path, load(path)


def fp32(values):
    require(
        len(values) == 10 and all(math.isfinite(v) for v in values), "Ten finite genes required"
    )
    return tuple(struct.unpack("!f", struct.pack("!f", v))[0] for v in values)


def verify_plan(path):
    plan = load(path)
    require(plan["schema"] == "lewm-navigation-analysis-plan-20261031-v1", "Unknown analysis plan")
    require(
        plan["analysis_script_sha256"] == digest(Path(__file__)),
        "Analysis source differs from prospective freeze",
    )
    require(
        plan["bootstrap"]["draws"] == 10000 and plan["bootstrap"]["seed"] == 803101,
        "Bootstrap configuration changed",
    )
    for name, expected in plan["source_sha256"].items():
        require(digest(ROOT / name) == expected, "Frozen analysis dependency changed: " + name)
    return plan


def joint_readiness(root, ref):
    # This helper is JSON/hash only. It must run before the first final-state read.
    sys.path.insert(0, str(ROOT / "experiments"))
    from helpers.lewm_navigation_joint_final_v2 import validate_joint

    joint, primary, secondary = validate_joint(ref, root)
    require(
        joint["schema"] == "lewm-navigation-joint-final-opening-v2",
        "Three-host joint opening required",
    )
    require(
        joint["all_six_repeat_selections_fixed"] is True,
        "Both families' six selections must be locked before any final outcome read",
    )
    return {
        "joint_opening": ref,
        "secondary_activation": joint["secondary_activation"],
        "secondary_relocation": joint["secondary_relocation"],
        "primary_repeat_ids": sorted(primary),
        "secondary_repeat_ids": sorted(secondary),
        "all_six_selections_verified_before_final_outcome_reads": True,
    }


def relocation_contract(root, readiness, plan):
    require(
        readiness["secondary_relocation"]["sha256"] == plan["secondary_relocation"]["sha256"],
        "Joint guard relocation differs from the prospective analysis plan",
    )
    _, relocation = load_reference(root, readiness["secondary_relocation"])
    require(
        relocation["schema"] == "lewm-navigation-secondary-relocation-v1"
        and relocation["study_family"] == "secondary30"
        and relocation["repeat_id"] == 2,
        "Only the prospective whole secondaryR2 relocation is supported",
    )
    for key in (
        "whole_unstarted_repeat_block",
        "scientific_settings_unchanged",
        "all_cases_seeds_budgets_preserved",
        "all_local_methods_and_controls_moved",
    ):
        require(
            relocation[key] is True,
            "Relocation must preserve the entire unstarted scientific block",
        )
    require(relocation["final_pixels_read"] == 0, "Relocation must precede final access")
    target = relocation["new_host"]["pod_id"]
    require(
        target == plan["profiles"]["secondary30"]["host_assignment"]["2"]
        and target in plan["execution"]["allowed_pod_ids"],
        "Relocated host differs from frozen analysis assignments",
    )
    primary_hosts = set(plan["profiles"]["primary50"]["host_assignment"].values())
    secondary_hosts = set(plan["profiles"]["secondary30"]["host_assignment"].values())
    require(
        len(primary_hosts) == 2 and len(secondary_hosts) == 3 and primary_hosts < secondary_hosts,
        "Primary remains two-host; secondary now retains all three distinct host blocks",
    )
    require(
        relocation["unstarted_proof"] == plan["secondary_unstarted_proof"],
        "Relocation qualification proof differs from the analysis plan",
    )
    _, proof = load_reference(root, relocation["unstarted_proof"])
    require(
        proof["schema"] == "lewm-secondary-r2-unstarted-proof-v1"
        and proof["pod_id"] == relocation["old_host"]["pod_id"]
        and proof["old_secondary_r2_output_exists"] is False
        and proof["old_secondary_r2_output_children"] == []
        and proof["original_secondary_manifest"]["sha256"]
        == relocation["original_secondary_manifest"]["sha256"],
        "Relocation needs the exact old-host whole-repeat unstarted proof",
    )
    return relocation["unstarted_proof"]


def auxiliary_contract(plan, manifest, profiles):
    enabled = set(manifest["auxiliary_endpoints_frozen_before_outcomes"])
    require(enabled <= {30, 50}, "Unknown auxiliary endpoint")
    for cap in (30, 50):
        family = f"aux-decoded-native{cap}"
        selected = [entry for entry in profiles if entry["family"] == family]
        expected = (
            set(plan["auxiliary_decoded_native"]["eligible_pod_ids_by_endpoint"][str(cap)])
            if cap in enabled
            else set()
        )
        require(
            len(selected) == len(expected)
            and {entry["host"]["pod_id"] for entry in selected} == expected,
            "Auxiliary profiles must match endpoint-specific participating host blocks",
        )


def profile_contract(root, entry, protocol, plan):
    family = entry["family"]
    require(
        family in ("primary50", "secondary30", "aux-decoded-native30", "aux-decoded-native50"),
        "Unfrozen profile family",
    )
    cap = 50 if family.endswith("50") else 30
    require(
        protocol["evaluation"]["maximum_episode_actions"] == cap
        and all(job["max_actions"] == cap for job in protocol["jobs"]),
        "Profile endpoint differs from its family",
    )
    require(
        entry["host"] == protocol["host"]
        and protocol["evaluation"]["execution_pod_id"] == entry["host"]["pod_id"],
        "Protocol host identity differs",
    )
    if family.startswith("aux-"):
        require(
            entry["host"]["pod_id"]
            in plan["auxiliary_decoded_native"]["eligible_pod_ids_by_endpoint"][str(cap)],
            "Auxiliary host has no native reference at this endpoint",
        )
        require(
            len(protocol["jobs"]) == 16
            and {j["arm"] for j in protocol["jobs"]} == {"decoded_native"}
            and all(j["method"] == "decoded_native" for j in protocol["jobs"]),
            "Auxiliary must retain exactly16 decoded-native jobs",
        )
        require(
            len({j["case"]["source_episode"] for j in protocol["jobs"]}) == 16,
            "Auxiliary source case duplicated",
        )
        return
    repeat = entry["repeat_id"]
    require(repeat in (0, 1, 2) and protocol["repeat_id"] == repeat, "Repeat identity differs")
    require(
        entry["host"]["pod_id"] == plan["profiles"][family]["host_assignment"][str(repeat)],
        "Host assignment differs from the prospective plan",
    )
    _, lock = load_reference(root, protocol["selection_lock"])
    require(
        lock["repeat_id"] == repeat
        and lock["host"] == entry["host"]
        and lock["evaluation"]["maximum_episode_actions"] == cap,
        "Selected lock host/repeat/horizon differs",
    )
    anchor = fp32(lock["selected"]["strong_anchor"]["parameters"])
    require(len(protocol["jobs"]) == 96, "Six full16-case optimizer arms required")
    for job in protocol["jobs"]:
        arm = job["arm"]
        require(arm in ARMS, "Unknown selected arm")
        chosen = lock["selected"][arm]
        require(
            job["parameters"] == chosen["parameters"]
            and job["selected_source_arm"] == chosen["selected_source_arm"],
            "Final planned genes/source differ from selected lock",
        )
        require(
            job["selection_lock_sha256"] == protocol["selection_lock"]["sha256"]
            and job["producer_provenance"] == chosen["provenance"],
            "Planned genotype provenance differs",
        )
        duplicate = arm != "native" and fp32(job["parameters"]) == anchor
        require(
            job["selected_is_trained"] == chosen["selected_is_trained"]
            and job["effective_selected_is_trained"]
            == bool(chosen["selected_is_trained"] and not duplicate),
            "Effective trained/anchor classification differs",
        )
        expected_method = (
            "native"
            if arm == "native"
            else "fixed_geometry_progress"
            if arm in ("strong_anchor", "tuned_fixed")
            else arm
        )
        require(job["method"] == expected_method, "Wrong method for selected arm")
    for arm in ARMS:
        require(
            len({j["case"]["source_episode"] for j in protocol["jobs"] if j["arm"] == arm}) == 16,
            "An arm omitted or duplicated a source case",
        )


def bind_inputs(args):
    plan = verify_plan(args.plan)
    require(not args.out.exists(), "Never overwrite a bound analysis input freeze")
    manifest = load(args.manifest)
    require(
        manifest["schema"] == plan["joint_final_manifest_schema"],
        "Activated secondary requires the joint final manifest",
    )
    readiness = joint_readiness(ROOT, manifest["joint_opening"])
    readiness["secondary_unstarted_proof"] = relocation_contract(ROOT, readiness, plan)
    require(
        readiness["secondary_activation"]["sha256"] == plan["secondary_activation"]["sha256"],
        "Joint opening used a different secondary activation",
    )
    expected_training = {
        value["training_manifest"]["sha256"] for value in plan["profiles"].values()
    }
    require(
        len(args.training_manifest) == 2
        and {digest(path) for path in args.training_manifest} == expected_training,
        "Supplied training manifests differ from the exact two frozen plan references",
    )
    manifest_files = {str(resolve(ROOT, ref["path"])): ref["sha256"] for ref in manifest["files"]}
    profiles = []
    for planned in manifest["schedule"]:
        path = resolve(ROOT, planned["protocol"])
        require(
            str(path) in manifest_files and digest(path) == manifest_files[str(path)],
            "Final protocol bytes differ from joint manifest.files freeze",
        )
        protocol = load(path)
        namespace = protocol["output_namespace"]
        require(
            not (ROOT / namespace).exists(),
            "Final outputs already exist; input binding must precede them",
        )
        family = planned.get("family") or (
            "primary50" if planned.get("kind") == "primary" else "aux-decoded-native50"
        )
        require(re.fullmatch(r"[A-Za-z0-9_-]+", family), "Unsafe profile family")
        profiles.append(
            {
                "family": family,
                "repeat_id": protocol.get("repeat_id"),
                "host": protocol["host"],
                "protocol": reference(path),
                "state_path": namespace + "/state.json",
            }
        )
    auxiliary_contract(plan, manifest, profiles)
    for entry in profiles:
        _, protocol = load_reference(ROOT, entry["protocol"])
        profile_contract(ROOT, entry, protocol, plan)
    training = []
    for path in args.training_manifest:
        value = load(path)
        require("schedule" in value, "Training manifest requires a frozen schedule")
        for repeat in value["schedule"]:
            for method, run in repeat["runs"].items():
                protocol_path = resolve(ROOT, run["protocol"])
                protocol = load(protocol_path)
                family = (
                    f"primary{protocol['max_episode_actions']}"
                    if protocol["max_episode_actions"] == 50
                    else f"secondary{protocol['max_episode_actions']}"
                )
                training.append(
                    {
                        "family": family,
                        "repeat_id": repeat["repeat_id"],
                        "method": method,
                        "protocol": reference(protocol_path),
                        "output_namespace": run["output_namespace"],
                    }
                )
            fixed_refs = [
                ref
                for ref in value["files"]
                if Path(ref["path"]).name == f"fixed-tuning-r{repeat['repeat_id']}.json"
            ]
            require(
                len(fixed_refs) == 1,
                "Authoritative manifest must bind one fixed-tuning protocol per repeat",
            )
            fixed_path, fixed = load_reference(ROOT, fixed_refs[0])
            require(
                fixed["output_namespace"] == repeat["fixed_tuning"],
                "Fixed-tuning source ref differs from the scheduled original/relocated namespace",
            )
            cap = fixed["evaluation"]["maximum_episode_actions"]
            training.append(
                {
                    "family": f"primary{cap}" if cap == 50 else f"secondary{cap}",
                    "repeat_id": repeat["repeat_id"],
                    "method": "tuned_fixed",
                    "protocol": reference(fixed_path),
                    "output_namespace": fixed["output_namespace"],
                }
            )
    require(
        {
            (p["family"], p["repeat_id"])
            for p in profiles
            if p["family"] in ("primary50", "secondary30")
        }
        == {(family, repeat) for family in ("primary50", "secondary30") for repeat in range(3)},
        "Both complete three-repeat final profiles required",
    )
    require(
        {(t["family"], t["repeat_id"], t["method"]) for t in training}
        == {
            (family, repeat, method)
            for family in ("primary50", "secondary30")
            for repeat in range(3)
            for method in ("cma", "random", "reinforce", "tuned_fixed")
        },
        "Both frozen training manifests are required for actual-action curves/budgets",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write(
        args.out,
        {
            "schema": "lewm-navigation-analysis-inputs-20261031-v3",
            "bound_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "plan": reference(args.plan),
            "final_manifest": reference(args.manifest),
            "joint_readiness": readiness,
            "profiles": profiles,
            "training_runs": training,
            "all_final_output_namespaces_absent_at_binding": True,
            "final_outcomes_read": 0,
            "new_model_simulator_calls": 0,
        },
    )
    return {"inputs": str(args.out), "sha256": digest(args.out), "profiles": len(profiles)}


def saved_profile(root, entry, plan):
    _, protocol = load_reference(root, entry["protocol"])
    profile_contract(root, entry, protocol, plan)
    path = resolve(root, entry["state_path"])
    state = load(path)
    source_path = path.parent / "source-freeze.json"
    source = load(source_path)
    require(
        source["protocol_sha256"] == entry["protocol"]["sha256"],
        "Final state belongs to another protocol",
    )
    require(source["sources"] == protocol["source_sha256"], "Final source freeze differs")
    require(
        source["pod_id"] == entry["host"]["pod_id"],
        "Actual execution pod differs from declared host",
    )
    require(state["phase"] == "complete" and state["active"] is None, "Final profile is incomplete")
    require(
        set(state["jobs"]) == {j["job_id"] for j in protocol["jobs"]}, "Final job panel differs"
    )
    rows, resources = [], collections.Counter()
    for planned in protocol["jobs"]:
        row = state["jobs"][planned["job_id"]]
        folder = path.parent / "jobs" / planned["job_id"]
        require(
            load(folder / "result.json") == row and load(folder / "started.json") == planned,
            "State/retained result/started prospective job differ",
        )
        require(row["method"] == planned["method"], "Executed method differs")
        require(
            not row["error"] and not row["extra"]["frame_flags"],
            "Technical/input failure prevents efficacy analysis",
        )
        for key in ("case_id", "environment_seed", "policy_seed", "parameters"):
            expected = planned["case"]["case_id"] if key == "case_id" else planned[key]
            require(row[key] == expected, "Executed final job differs: " + key)
        require(all(isinstance(row[m], bool) for m in METRICS), "Outcome metric must be boolean")
        require(
            not (row["safe_completion"] and (row["violation"] or row["unfinished"])),
            "Inconsistent outcome",
        )
        require(row["environment_actions"] <= planned["max_actions"], "Final action cap exceeded")
        inner_path = resolve(root, row["extra"]["episode_result"])
        require(
            digest(inner_path) == row["extra"]["episode_result_sha256"],
            "Episode provenance bytes changed",
        )
        inner = load(inner_path)
        for key, value in row["resources"].items():
            require(
                isinstance(value, (int, float)) and math.isfinite(value) and value >= 0,
                "Invalid resource counter",
            )
            resources[key] += value
        rows.append(
            {
                "arm": planned["arm"],
                "source_episode": planned["case"]["source_episode"],
                "case_id": planned["case"]["case_id"],
                "condition": planned["case"]["condition"],
                **{m: int(row[m]) for m in METRICS},
                "environment_actions": row["environment_actions"],
                "resources": row["resources"],
                "episode_seconds": inner["episode_seconds"],
                "planning_seconds": inner["planning_seconds"],
                "job_id": planned["job_id"],
                "provenance": {
                    key: planned.get(key)
                    for key in (
                        "selected_is_trained",
                        "effective_selected_is_trained",
                        "selected_source_arm",
                        "producer_provenance",
                        "fp32_effective_policy_duplicate_of_anchor",
                        "fp32_duplicate_controller_arms",
                        "parameters_ignored_by_native",
                    )
                },
            }
        )
    actions = sum(row["environment_actions"] for row in rows)
    require(actions <= protocol["maximum_environment_actions"], "Final profile exceeded allowance")
    by_arm = {}
    for arm in sorted({row["arm"] for row in rows}):
        selected = [row for row in rows if row["arm"] == arm]
        totals = collections.Counter()
        for row in selected:
            totals.update(row["resources"])
        allowance = sum(job["max_actions"] for job in protocol["jobs"] if job["arm"] == arm)
        used = sum(row["environment_actions"] for row in selected)
        by_arm[arm] = {
            "episodes": len(selected),
            "actual_environment_actions": used,
            "hard_action_allowance": allowance,
            "residual_unspent_allowance": allowance - used,
            "episode_seconds_sum": sum(row["episode_seconds"] for row in selected),
            "planning_seconds_sum": sum(row["planning_seconds"] for row in selected),
            "resource_counter_totals_by_field": dict(totals),
        }
    result = {
        **entry,
        "resource_budget_by_arm": by_arm,
        "state_sha256": digest(path),
        "source_freeze_sha256": digest(source_path),
        "rows": rows,
        "resources": dict(resources),
        "actual_environment_actions": actions,
        "environment_action_allowance": protocol["maximum_environment_actions"],
        "residual_unspent_allowance": protocol["maximum_environment_actions"] - actions,
        "episode_seconds_sum": sum(row["episode_seconds"] for row in rows),
        "planning_seconds_sum": sum(row["planning_seconds"] for row in rows),
    }
    if "selection_lock" in protocol:
        _, lock = load_reference(root, protocol["selection_lock"])
        result["selection_lock"] = protocol["selection_lock"]
        anchor = fp32(lock["selected"]["strong_anchor"]["parameters"])
        result["selected_controller_provenance"] = {}
        for arm, chosen in lock["selected"].items():
            provenance = chosen["provenance"]
            producer = provenance.get("producer", {})
            updates = provenance.get(
                "completed_updates_before_nominee",
                producer.get(
                    "completed_updates_before_proposal",
                    producer.get("completed_updates_before_evaluation", 0),
                ),
            )
            duplicate = arm != "native" and fp32(chosen["parameters"]) == anchor
            result["selected_controller_provenance"][arm] = {
                "selected_source_arm": chosen["selected_source_arm"],
                "selected_native": chosen["selected_source_arm"] == "native",
                "anchor_equivalent_fp32": duplicate,
                "selected_is_trained_declared": chosen["selected_is_trained"],
                "effective_selected_is_trained": bool(
                    chosen["selected_is_trained"] and not duplicate
                ),
                "actual_updates_before_genotype_nomination": updates,
                "genotype_proposed_after_an_actual_update": updates > 0,
                "producer": provenance,
                "native_ignores_selector_genes": arm == "native",
            }
    return result


def bootstrap_summary(np, values, indices):
    draws = values[indices].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "case_cluster_percentile95": [
            float(v) for v in np.quantile(draws, [0.025, 0.975], method="linear")
        ],
        "source_clusters": 16,
        "bootstrap_draws": len(indices),
        "interval_scope": "Nominal descriptive95percent case-cluster interval; not family-wise adjusted; no family-wise optimizer-superiority claim",
    }


def family_summary(np, family, profiles, indices, expected_cases):
    require(sorted(p["repeat_id"] for p in profiles) == [0, 1, 2], "All three repeats required")
    matrix = {metric: {} for metric in METRICS}
    counts, by_host = [], {}
    for arm in ARMS:
        for metric in METRICS:
            matrix[metric][arm] = np.asarray(
                [
                    [
                        next(
                            row[metric]
                            for row in p["rows"]
                            if row["arm"] == arm and row["source_episode"] == episode
                        )
                        for episode in expected_cases
                    ]
                    for p in profiles
                ],
                dtype=float,
            )
    for p in profiles:
        for arm in ARMS:
            selected = [r for r in p["rows"] if r["arm"] == arm]
            require(len(selected) == 16, "Every arm requires exactly16 source cases")
            counts.append(
                {
                    "family": family,
                    "repeat_id": p["repeat_id"],
                    "host": p["host"],
                    "arm": arm,
                    "episodes": 16,
                    "source_clusters": 16,
                    "zero_violation_rule_of_three_upper_approx": 3 / 16
                    if not any(r["violation"] for r in selected)
                    else None,
                    **{m: sum(r[m] for r in selected) for m in METRICS},
                }
            )
    for host in sorted({p["host"]["pod_id"] for p in profiles}):
        selected_repeats = [i for i, p in enumerate(profiles) if p["host"]["pod_id"] == host]
        by_host[host] = {
            "fixed_repeat_ids": [profiles[i]["repeat_id"] for i in selected_repeats],
            "raw_episode_denominator_each_arm": 16 * len(selected_repeats),
            "source_clusters": 16,
            "arms": {
                arm: {
                    m: {
                        "raw_count": int(matrix[m][arm][selected_repeats].sum()),
                        "mean_fixed_repeat_rate": float(matrix[m][arm][selected_repeats].mean()),
                    }
                    for m in METRICS
                }
                for arm in ARMS
            },
        }
    marginal = {
        m: {arm: bootstrap_summary(np, matrix[m][arm].mean(axis=0), indices) for arm in ARMS}
        for m in METRICS
    }
    contrasts = {}
    for arm in BASELINES:
        difference = matrix["safe_completion"]["cma"] - matrix["safe_completion"][arm]
        contrasts[arm] = {
            **bootstrap_summary(np, difference.mean(axis=0), indices),
            "repeat_differences": [float(v) for v in difference.mean(axis=1)],
            "per_host": {
                host: bootstrap_summary(
                    np,
                    difference[
                        [i for i, p in enumerate(profiles) if p["host"]["pod_id"] == host]
                    ].mean(axis=0),
                    indices,
                )
                for host in by_host
            },
        }
    all_contrasts = {}
    for metric in METRICS:
        all_contrasts[metric] = {}
        for arm in BASELINES:
            difference = matrix[metric]["cma"] - matrix[metric][arm]
            all_contrasts[metric][arm] = {
                **bootstrap_summary(np, difference.mean(axis=0), indices),
                "repeat_differences": [float(v) for v in difference.mean(axis=1)],
            }
    return {
        "family": family,
        "repeat_counts": counts,
        "physical_host_blocks": by_host,
        "marginal_fixed_controller_rates": marginal,
        "paired_CMA_minus_baseline_SC": contrasts,
        "paired_CMA_minus_baseline_all_metrics": all_contrasts,
        "raw_episode_observations_each_arm": 48,
        "source_case_clusters": 16,
        "fixed_learner_repeats": 3,
        "uncertainty": "Conditional on these learned controllers and assigned hosts; source-case sampling only",
    }, matrix


def training_curve(root, entry):
    _, protocol = load_reference(root, entry["protocol"])
    folder = resolve(root, entry["output_namespace"])
    state = load(folder / "state.json")
    require(
        state["phase"] == "complete" and state["active"] is None, "Training/tuning not complete"
    )
    fixed = entry["method"] == "tuned_fixed"
    if fixed:
        freeze = load(folder / "source-freeze.json")
        require(
            freeze["protocol_sha256"] == entry["protocol"]["sha256"],
            "Tuned-fixed source protocol differs",
        )
        order = [j["job_id"] for j in protocol["jobs"]]
        rows = state["jobs"]
        panel_size = len({j["case"]["case_id"] for j in protocol["jobs"]})
        candidates = [
            {
                "job_ids": order[i : i + panel_size],
                "provenance": {
                    "kind": "anchor" if i == 0 else "fixed_grid",
                    "index": i // panel_size,
                },
            }
            for i in range(0, len(order), panel_size)
        ]
        allowance = protocol["maximum_environment_actions"]
    else:
        require(
            not state["unfinalized_job_ids"] and state["action_accounting_exact"],
            "Training has unresolved failed work",
        )
        freeze = load(folder / "source-freeze.json")
        require(
            freeze["protocol"] == protocol and freeze["identity"] == state["identity"],
            "Training source identity differs",
        )
        order = state["completed_job_ids"]
        rows = {job: load(folder / "jobs" / job / "result.json") for job in order}
        candidates = [state["anchor"]]
        candidates += [load(path) for path in sorted(folder.glob("candidate-*.json"))]
        candidates += [load(path)["nominee"] for path in sorted(folder.glob("update-*.json"))]
        panel_size = len(protocol["training_panel"])
        allowance = protocol["max_environment_actions"]
    require(len(order) == len(set(order)), "Chronological job ledger duplicated")
    cumulative, positions, consumed = {}, {}, 0
    for position, job in enumerate(order):
        consumed += rows[job]["environment_actions"]
        cumulative[job], positions[job] = consumed, position
    require(0 <= consumed <= allowance, "Used training actions exceed the frozen allowance")
    if not fixed:
        require(
            consumed == state["actual_environment_actions"],
            "Chronological training actions differ from actual ledger",
        )
    else:
        require(
            consumed == state["resource_totals"].get("environment_actions", 0),
            "Fixed tuning actions differ from its resource ledger",
        )
    events = []
    for candidate in candidates:
        require(
            candidate is not None and len(candidate["job_ids"]) == panel_size,
            "Incomplete nominee panel",
        )
        values = [rows[job] for job in candidate["job_ids"]]
        score = (
            sum(v["safe_completion"] for v in values),
            -sum(v["violation"] for v in values),
            sum(v["progress"] for v in values),
        )
        if "fitness" in candidate:
            require(list(score) == candidate["fitness"], "Training fitness does not reconstruct")
        last = max(candidate["job_ids"], key=positions.get)
        events.append(
            {
                "actual_actions": cumulative[last],
                "last_job_position": positions[last],
                "fitness": list(score),
                "safe_completion_rate": score[0] / panel_size,
                "provenance": candidate["provenance"],
            }
        )
    exploratory = []
    if entry["method"] in ("cma", "random"):
        for generation in sorted(
            {
                e["provenance"].get("generation")
                for e in events
                if e["provenance"].get("kind") == "trained_candidate"
            }
        ):
            population = [
                e
                for e in events
                if e["provenance"].get("kind") == "trained_candidate"
                and e["provenance"].get("generation") == generation
            ]
            exploratory.append(
                {
                    "generation_or_fixed_random_batch": generation,
                    "candidate_panels": len(population),
                    "actual_actions_at_batch_end": max(e["actual_actions"] for e in population),
                    "population_mean_TRAIN_SC_rate": sum(
                        e["safe_completion_rate"] for e in population
                    )
                    / len(population),
                }
            )
    elif entry["method"] == "reinforce":
        for update_path in sorted(folder.glob("update-*.json")):
            update = load(update_path)
            sample = [rows[job] for job in update["sample_job_ids"]]
            evaluation = [rows[job] for job in update["nominee"]["job_ids"]]
            exploratory.append(
                {
                    "update": update["update"],
                    "stochastic_sample_panel_SC_rate": sum(r["safe_completion"] for r in sample)
                    / len(sample),
                    "deterministic_postupdate_panel_SC_rate": sum(
                        r["safe_completion"] for r in evaluation
                    )
                    / len(evaluation),
                    "sample_panel_actual_actions_at_end": max(
                        cumulative[job] for job in update["sample_job_ids"]
                    ),
                    "evaluation_panel_actual_actions_at_end": max(
                        cumulative[job] for job in update["nominee"]["job_ids"]
                    ),
                }
            )
    resource_totals = collections.Counter()
    episode_seconds = planning_seconds = 0.0
    for row in rows.values():
        resource_totals.update(row["resources"])
        inner_path = resolve(root, row["extra"]["episode_result"])
        require(
            digest(inner_path) == row["extra"]["episode_result_sha256"],
            "Training episode artifact changed",
        )
        inner = load(inner_path)
        episode_seconds += inner["episode_seconds"]
        planning_seconds += inner["planning_seconds"]
    incumbent, curve = None, []
    for event in sorted(events, key=lambda e: e["last_job_position"]):
        if incumbent is None or tuple(event["fitness"]) > tuple(incumbent["fitness"]):
            incumbent = event
        curve.append(
            {
                "actual_actions": event["actual_actions"],
                "incumbent_SC_rate": incumbent["safe_completion_rate"],
                "incumbent_fitness": incumbent["fitness"],
                "incumbent_provenance": incumbent["provenance"],
            }
        )
    landmarks = protocol.get("learning_curve_reporting", {}).get(
        "action_landmarks", [0, allowance // 4, allowance // 2, 3 * allowance // 4, allowance]
    )
    sampled = []
    for point in landmarks:
        available = [value for value in curve if value["actual_actions"] <= point]
        sampled.append(
            {
                "actual_action_landmark": point,
                "incumbent": available[-1] if available and point <= consumed else None,
                "reason_if_unavailable": "No complete anchor/panel yet, or landmark lies in unspent allowance",
            }
        )
    return {
        **entry,
        "training_panel_size": panel_size,
        "actual_environment_actions": consumed,
        "hard_action_allowance": allowance,
        "residual_unspent_allowance": allowance - consumed,
        "actual_optimizer_updates": state.get("optimizer_updates", 0),
        "episode_starts": len(order),
        "resource_totals_by_field": dict(resource_totals),
        "episode_seconds_sum": episode_seconds,
        "planning_seconds_sum": planning_seconds,
        "model_query_alias_fields_are_not_added_together": True,
        "curve": curve,
        "exploratory_TRAIN_population_or_PG_panels": exploratory,
        "exploratory_scope": "Prompted by observed R0 TRAIN totals, not final outcomes; population behavior is not validated selected-policy gain. PG stochastic and deterministic panels remain separate.",
        "landmarks": sampled,
        "no_extrapolation_into_unspent_allowance": True,
        "state_sha256": digest(folder / "state.json"),
        "source_freeze_sha256": digest(folder / "source-freeze.json"),
    }


def save_figure(fig, out, name):
    fig.savefig(out / (name + ".png"), dpi=240, bbox_inches="tight")
    fig.savefig(out / (name + ".svg"), bbox_inches="tight")


def plots(plt, np, out, family, summary, matrices, profiles, curves):
    colors = ("#8a8a8a", "#377eb8", "#4daf4a", "#e67e22", "#984ea3", "#e41a1c")
    fig, grid = plt.subplots(2, 2, figsize=(7.2, 6.2), sharey=True)
    axes = grid.ravel()
    for axis, metric in zip(axes, METRICS, strict=True):
        x = np.arange(len(ARMS))
        mean = np.asarray(
            [summary["marginal_fixed_controller_rates"][metric][a]["estimate"] * 100 for a in ARMS]
        )
        intervals = (
            np.asarray(
                [
                    summary["marginal_fixed_controller_rates"][metric][a][
                        "case_cluster_percentile95"
                    ]
                    for a in ARMS
                ]
            )
            * 100
        )
        axis.bar(x, mean, color=colors, alpha=0.55)
        axis.errorbar(
            x,
            mean,
            yerr=np.asarray([mean - intervals[:, 0], intervals[:, 1] - mean]),
            fmt="none",
            color="black",
            capsize=2,
        )
        for r, profile in enumerate(profiles):
            axis.scatter(
                x + (r - 1) * 0.14,
                [matrices[metric][a][r].mean() * 100 for a in ARMS],
                marker=("o", "s", "^")[r],
                color="black",
                s=16,
                label=f"r{profile['repeat_id']} / {profile['host']['host']}",
            )
        axis.set_xticks(x, LABELS, rotation=45, ha="right", fontsize=8)
        axis.set_title(
            {
                "safe_completion": "Safe completion",
                "violation": "Violation",
                "unfinished": "Unfinished",
                "engaged": "Engaged",
            }[metric]
        )
        axis.set_ylim(0, 105)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Percent of observed cases")
    axes[2].set_ylabel("Percent of observed cases")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles, labels, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=3
    )
    fig.suptitle(
        f"{family}:16 source-case clusters,3 fixed repeat/host scores", fontsize=10, y=0.995
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=1.6)
    save_figure(fig, out, family + "-outcomes")
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(6.5, 3.8))
    for x, arm in enumerate(BASELINES):
        value = summary["paired_CMA_minus_baseline_SC"][arm]
        interval = np.asarray(value["case_cluster_percentile95"]) * 100
        estimate = value["estimate"] * 100
        axis.errorbar(
            x,
            estimate,
            yerr=[[estimate - interval[0]], [interval[1] - estimate]],
            fmt="D",
            color="#e67e22",
            capsize=4,
        )
        axis.scatter(
            [x - 0.13, x, x + 0.13],
            np.asarray(value["repeat_differences"]) * 100,
            color="black",
            s=20,
        )
    axis.axhline(0, color="gray", linestyle=":")
    axis.set_xticks(range(4), ["Strong fixed", "Tuned fixed", "Random search", "REINFORCE"])
    axis.set_ylabel("CMA-selected SC difference (percentage points)")
    axis.set_title(f"{family}: paired conditional source-case bootstrap")
    save_figure(fig, out, family + "-paired-SC-differences")
    plt.close(fig)
    selected_curves = [entry for entry in curves if entry["family"] == family]
    if selected_curves:
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.2), sharey=True)
        for repeat, axis in enumerate(axes):
            for entry in selected_curves:
                if entry["repeat_id"] != repeat:
                    continue
                points = entry["curve"]
                axis.step(
                    [p["actual_actions"] for p in points],
                    [100 * p["incumbent_SC_rate"] for p in points],
                    where="post",
                    label=entry["method"],
                )
                if points:
                    axis.scatter(
                        points[-1]["actual_actions"], 100 * points[-1]["incumbent_SC_rate"], s=12
                    )
                axis.set_xlim(0, entry["hard_action_allowance"])
            axis.set_title(f"Repeat{repeat}", fontsize=9)
            axis.set_xlabel("Actual training actions", fontsize=8)
            axis.tick_params(labelsize=8)
            axis.grid(alpha=0.2)
        axes[0].set_ylabel("Incumbent TRAIN SC (%)", fontsize=9)
        axes[-1].legend(fontsize=7)
        fig.tight_layout()
        save_figure(fig, out, family + "-training-incumbents")
        plt.close(fig)


def latex(out, summaries):
    lines = [
        "% Frozen conditional analysis. Denominators are repeated observations, not independent samples.",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Family / repeat / host & Controller & SC & V & Unfinished & Engaged \\",
        r"\midrule",
    ]
    for family, value in summaries.items():
        for row in value["repeat_counts"]:
            label = f"{family}/r{row['repeat_id']}/{row['host']['host']}".replace("_", r"\_")
            counts = " & ".join(f"{row[m]}/16" for m in METRICS)
            lines.append(f"{label} & {LABELS[ARMS.index(row['arm'])]} & {counts} " + r"\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "% Each family has16 shared source-case clusters and3 fixed learner-repeat/host assignments.",
    ]
    (out / "final-counts.tex").write_text("\n".join(lines) + "\n")
    lines = [
        r"\begin{tabular}{llr}",
        r"\toprule",
        r"Family & Paired contrast & SC difference (nominal95\% case-cluster interval) \\",
        r"\midrule",
    ]
    for family, value in summaries.items():
        for arm, contrast in value["paired_CMA_minus_baseline_SC"].items():
            lower, upper = contrast["case_cluster_percentile95"]
            lines.append(
                f"{family} & CMA minus {LABELS[ARMS.index(arm)]} & {100 * contrast['estimate']:.1f} [{100 * lower:.1f}, {100 * upper:.1f}] "
                + r"\\"
            )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "% Conditional on the fixed learned controllers and hosts; no future-learner/host inference or multiplicity-adjusted significance claim.",
    ]
    (out / "paired-SC-differences.tex").write_text("\n".join(lines) + "\n")


def analyze(args):
    require(
        os.environ.get("RUNPOD_POD_ID") == args.pod_id,
        "Numerical analysis belongs only on the declared rented host",
    )
    started = time.perf_counter()
    plan = verify_plan(args.plan)
    require(
        args.pod_id in plan["execution"]["allowed_pod_ids"],
        "Analysis pod is not in the frozen plan whitelist",
    )
    inputs = load(args.inputs)
    require(
        inputs["schema"] == "lewm-navigation-analysis-inputs-20261031-v3",
        "V3 bound inputs required",
    )
    require(
        inputs["plan"]["sha256"] == digest(args.plan)
        and inputs["all_final_output_namespaces_absent_at_binding"] is True,
        "Inputs were not bound to the prospective plan",
    )
    require(not args.out.exists(), "Use a fresh immutable analysis namespace")
    root = args.artifact_root.resolve()
    readiness = joint_readiness(root, inputs["joint_readiness"]["joint_opening"])
    readiness["secondary_unstarted_proof"] = relocation_contract(root, readiness, plan)
    require(
        readiness["secondary_activation"]["sha256"] == plan["secondary_activation"]["sha256"],
        "Secondary activation differs from plan",
    )
    _, manifest = load_reference(root, inputs["final_manifest"])
    require(
        manifest["schema"] == plan["joint_final_manifest_schema"],
        "Wrong relocated-host final manifest",
    )
    auxiliary_contract(plan, manifest, inputs["profiles"])
    manifest_files = {str(resolve(root, ref["path"])): ref["sha256"] for ref in manifest["files"]}
    require(
        all(
            manifest_files.get(str(resolve(root, entry["protocol"]["path"])))
            == entry["protocol"]["sha256"]
            for entry in inputs["profiles"]
        ),
        "Bound protocols differ from frozen final manifest",
    )
    profiles = [saved_profile(root, entry, plan) for entry in inputs["profiles"]]
    primary = [p for p in profiles if p["family"] == "primary50"]
    require(
        len(primary) == 3 and len([p for p in profiles if p["family"] == "secondary30"]) == 3,
        "Both activated families require all3 fixed repeats",
    )
    cases = sorted({row["source_episode"] for row in primary[0]["rows"]})
    require(
        len(cases) == 16 and all(5000 <= case <= 5255 for case in cases),
        "Exactly16 frozen final source clusters",
    )
    strata = [
        [i for i, case in enumerate(cases) if case % 4 == condition] for condition in range(4)
    ]
    require(all(len(indices) == 4 for indices in strata), "Exactly4 source cases per condition")
    for profile in profiles:
        require(
            {row["source_episode"] for row in profile["rows"]} == set(cases),
            "Final source-case clusters differ across profiles",
        )
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.Generator(np.random.PCG64(plan["bootstrap"]["seed"]))
    indices = np.concatenate(
        [rng.choice(value, (10000, 4), replace=True) for value in strata], axis=1
    )
    curves = [training_curve(root, entry) for entry in inputs["training_runs"]]
    summaries, matrices = {}, {}
    args.out.mkdir(parents=True)
    for family in ("primary50", "secondary30"):
        chosen = sorted(
            (p for p in profiles if p["family"] == family), key=lambda p: p["repeat_id"]
        )
        if not chosen:
            continue
        summaries[family], matrices[family] = family_summary(np, family, chosen, indices, cases)
        plots(plt, np, args.out, family, summaries[family], matrices[family], chosen, curves)
    auxiliary = []
    for profile in profiles:
        if not profile["family"].startswith("aux-decoded-native"):
            continue
        cap = profile["family"].removeprefix("aux-decoded-native")
        family = "primary50" if cap == "50" else "secondary30"
        candidates = sorted(
            (
                p
                for p in profiles
                if p["family"] == family and p["host"]["pod_id"] == profile["host"]["pod_id"]
            ),
            key=lambda p: p["repeat_id"],
        )
        require(bool(candidates), "Auxiliary lacks same-host/endpoint native reference")
        baseline = candidates[0]
        difference = np.asarray(
            [
                next(r["safe_completion"] for r in profile["rows"] if r["source_episode"] == case)
                - next(
                    r["safe_completion"]
                    for r in baseline["rows"]
                    if r["arm"] == "native" and r["source_episode"] == case
                )
                for case in cases
            ],
            dtype=float,
        )
        auxiliary.append(
            {
                "family": profile["family"],
                "host": profile["host"],
                "native_reference_repeat": baseline["repeat_id"],
                "decoded_minus_native_SC": bootstrap_summary(np, difference, indices),
                "denominator_each": 16,
                "counts": {
                    "decoded_native": {
                        metric: sum(row[metric] for row in profile["rows"]) for metric in METRICS
                    },
                    "native": {
                        metric: sum(
                            row[metric] for row in baseline["rows"] if row["arm"] == "native"
                        )
                        for metric in METRICS
                    },
                },
                "interpretation": "Separate shared-architecture control, no evolutionary credit; never pooled with primary/secondary optimizer contrasts",
            }
        )
    if auxiliary:
        for aux_family in sorted({item["family"] for item in auxiliary}):
            blocks = [item for item in auxiliary if item["family"] == aux_family]
            fig, axes = plt.subplots(1, len(blocks), figsize=(7.2, 3.5), squeeze=False)
            for axis, block in zip(axes[0], blocks, strict=True):
                x = np.arange(4)
                axis.bar(
                    x - 0.18,
                    [100 * block["counts"]["native"][metric] / 16 for metric in METRICS],
                    width=0.36,
                    label="Native",
                )
                axis.bar(
                    x + 0.18,
                    [100 * block["counts"]["decoded_native"][metric] / 16 for metric in METRICS],
                    width=0.36,
                    label="Decoded native",
                )
                axis.set_xticks(x, ["SC", "V", "Unfinished", "Engaged"], rotation=30)
                axis.set_ylim(0, 105)
                axis.set_title(block["host"]["host"] + ":16 paired cases")
                axis.set_ylabel("Percent of cases")
                axis.legend(fontsize=8)
            fig.suptitle(
                aux_family + ": separate architecture control, no evolutionary credit", fontsize=10
            )
            fig.tight_layout(rect=(0, 0, 1, 0.9))
            save_figure(fig, args.out, aux_family + "-separate-controls")
            plt.close(fig)
        aux_lines = [
            r"\begin{tabular}{lllrrrr}",
            r"\toprule",
            "Profile & Host & Controller & SC & V & Unfinished & Engaged " + chr(92) * 2,
            r"\midrule",
        ]
        for block in auxiliary:
            for arm in ("native", "decoded_native"):
                values = " & ".join(f"{block['counts'][arm][metric]}/16" for metric in METRICS)
                aux_lines.append(
                    f"{block['family']} & {block['host']['host']} & {arm.replace('_', ' ')} & {values} "
                    + r"\\"
                )
        aux_lines += [
            r"\bottomrule",
            r"\end{tabular}",
            "% Separate auxiliary controls; no evolutionary optimizer inference.",
        ]
        (args.out / "auxiliary-separate-counts.tex").write_text("\n".join(aux_lines) + "\n")
    latex(args.out, summaries)
    caption = (
        "Outcome points show all3 fixed learner-repeat/host scores; bars show their equal-weight mean. "
        "Intervals resample16 source-case clusters within four condition strata,10000 PCG64 draws seed803101, "
        "conditional on these selected controllers and hosts. Repeated controllers and host duplicates are not independent cases. "
        "Zero observed violations is not zero risk; a rough per-controller3/16 upper bound requires independent cases. "
        "Training curves charge every actual action, including anchor, rejected candidates and PG sampling, "
        "and stop at used budget without extrapolation into unspent allowance. Auxiliary and30-action results are separate."
    )
    caption += " Intervals are nominal descriptive95percent intervals, not family-wise adjusted; no family-wise optimizer-superiority claim is licensed across four contrasts or two profiles."
    (args.out / "figure-captions.txt").write_text(caption + "\n")
    costs = [
        r"\begin{tabular}{lllrrrrr}",
        r"\toprule",
        "Profile & Repeat/host & Arm & Actions used/cap & Unspent & Episode s & Cost NN attempts & Safety NN attempts "
        + chr(92) * 2,
        r"\midrule",
    ]
    for profile in profiles:
        for arm, values in profile["resource_budget_by_arm"].items():
            resource = values["resource_counter_totals_by_field"]
            repeat = str(profile.get("repeat_id")) + "/" + profile["host"]["host"]
            costs.append(
                f"{profile['family']} & {repeat} & {arm.replace('_', ' ')} & {values['actual_environment_actions']}/{values['hard_action_allowance']} & {values['residual_unspent_allowance']} & {values['episode_seconds_sum']:.2f} & {resource.get('actual_cost_call_attempts', 'NA')} & {resource.get('actual_safety_call_attempts', 'NA')} "
                + chr(92) * 2
            )
    costs += [
        r"\bottomrule",
        r"\end{tabular}",
        "% Query counters are explicit attempts; do not add aliases. Other encoder/readout/predictor counters remain in analysis.json.",
    ]
    (args.out / "final-resource-budgets.tex").write_text("\n".join(costs) + "\n")
    report = {
        "schema": "lewm-navigation-analysis-20261031-v3",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "plan_sha256": digest(args.plan),
        "inputs_sha256": digest(args.inputs),
        "script_sha256": digest(Path(__file__)),
        "execution_pod_id": args.pod_id,
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
        "bootstrap": plan["bootstrap"],
        "source_episode_ids": cases,
        "families": summaries,
        "auxiliary_separate": auxiliary,
        "profiles_and_resource_budgets": profiles,
        "training_incumbent_curves": curves,
        "inference_scope": plan["inference_scope"],
        "zero_violation_caution": "A zero empirical bootstrap interval is not zero risk. For each fixed controller,0/16 independent source cases gives a rough3/16 upper bound; do not use48 repeated observations as independent trials.",
        "joint_readiness_before_final_reads": readiness,
        "method_selection_from_final_results": False,
        "multiplicity": plan["multiplicity"],
        "family_wise_optimizer_superiority_claim": False,
        "secondary_relocation": plan["secondary_relocation"],
        "elapsed_hardware_scope": "Report elapsed/query/action costs within each physical-host block; the added RTX4000Ada and original A40 hosts are heterogeneous, so pooled elapsed times do not establish optimizer speed superiority.",
        "resources": {
            "environment_resets": 0,
            "environment_actions": 0,
            "new_model_queries": 0,
            "new_dataset_pixel_rows_read": 0,
            "bootstrap_draws": 10000,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    write(
        args.out / "exploratory-training-populations.json",
        {
            "scope": plan["exploratory_training_diagnostic"],
            "runs": [
                {
                    "family": item["family"],
                    "repeat_id": item["repeat_id"],
                    "method": item["method"],
                    "summary": item["exploratory_TRAIN_population_or_PG_panels"],
                }
                for item in curves
            ],
        },
    )
    write(args.out / "analysis.json", report)
    write(
        args.out / "artifact-manifest.json",
        {
            "files": {
                str(path.name): digest(path)
                for path in sorted(args.out.iterdir())
                if path.is_file()
            },
            "plan_sha256": digest(args.plan),
            "script_sha256": digest(Path(__file__)),
        },
    )
    return {
        "analysis": str(args.out / "analysis.json"),
        "sha256": digest(args.out / "analysis.json"),
        "families": list(summaries),
        "final_data_used_for_tuning": False,
        "new_model_simulator_calls": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    binder = commands.add_parser("bind-inputs")
    binder.add_argument("--plan", type=Path, required=True)
    binder.add_argument("--manifest", type=Path, required=True)
    binder.add_argument("--training-manifest", type=Path, action="append", default=[])
    binder.add_argument("--out", type=Path, required=True)
    binder.set_defaults(function=bind_inputs)
    worker = commands.add_parser("analyze")
    worker.add_argument("--plan", type=Path, required=True)
    worker.add_argument("--inputs", type=Path, required=True)
    worker.add_argument("--artifact-root", type=Path, default=ROOT)
    worker.add_argument("--pod-id", required=True)
    worker.add_argument("--out", type=Path, required=True)
    worker.set_defaults(function=analyze)
    args = parser.parse_args()
    print(json.dumps(args.function(args), indent=2))


if __name__ == "__main__":
    main()
