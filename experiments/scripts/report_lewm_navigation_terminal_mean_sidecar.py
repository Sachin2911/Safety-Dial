"""JSON-only, owning-host terminal-mean reports and a three-host descriptive merge.

No arrays, checkpoints, models, simulator or fitting library are imported. Own-host
reports follow complete physical accounting; merge reads only those report JSONs.
"""

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/scripts"))
import analyze_lewm_navigation_campaign_20261031_v3 as analysis

BASELINES = ("strong_anchor", "tuned_fixed", "random", "reinforce", "cma")
METRICS = ("safe_completion", "violation", "unfinished", "engaged")
MEAN = "cma_terminal_mean"
ACCOUNTING = "lewm-campaign-accounting-20261031-v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def ref(root, path):
    return {"path": str(Path(path).resolve().relative_to(root)), "sha256": digest(path)}


def genes(values):
    require(len(values) == 10, "Exactly ten terminal genes required")
    require(all(math.isfinite(v) and -6 <= v <= 6 for v in values), "Invalid bounded genes")
    return tuple(struct.unpack("!f", struct.pack("!f", v))[0] for v in values)


def context(root, manifest_path):
    manifest = load(manifest_path)
    require(manifest["schema"] == "lewm-terminal-mean-final-sidecar-v1", "Unknown sidecar")
    _, plan = analysis.load_reference(root, manifest["plan"])
    require(plan["schema"] == "lewm-terminal-mean-exploratory-plan-v1", "Unknown mean plan")
    pod = os.environ.get("RUNPOD_POD_ID")
    require(pod in plan["allowed_pod_ids"], "Reporting belongs on a declared rented host")
    own_source = str(Path(__file__).resolve().relative_to(ROOT))
    require(plan["source_sha256"][own_source] == digest(__file__), "Report source changed")
    for name, expected in plan["source_sha256"].items():
        require(digest(ROOT / name) == expected, "Frozen report dependency changed: " + name)
    require(
        tuple(plan["baseline_arms"]) == BASELINES
        and plan["mean_arm"] == MEAN
        and plan["denominator_each_repeat"] == 16,
        "Prospective contrasts or denominator changed",
    )
    require(manifest["means"] == plan["means"], "Sidecar uses another mean freeze")
    _, means = analysis.load_reference(root, manifest["means"])
    require(
        means["schema"] == "lewm-secondary-terminal-means-v1"
        and means["all_three_means_fixed_before_final_metadata"] is True,
        "All three terminal means must be fixed prospectively",
    )
    _, main = analysis.load_reference(root, manifest["main_manifest"])
    require(main["schema"] == "lewm-navigation-joint-final-comparison-v2", "Wrong main manifest")
    analysis.joint_readiness(root, main["joint_opening"])
    require(
        manifest["all_main_outputs_absent_at_binding"] is True,
        "Sidecar must be bound before any main final outcome",
    )
    analysis_path, _ = analysis.load_reference(root, plan["main_analysis_plan"])
    main_plan = analysis.verify_plan(analysis_path)
    profiles = manifest["profiles"]
    require(sorted(p["repeat_id"] for p in profiles) == [0, 1, 2], "Retain all three means")
    locks = {}
    for item in means["locks"]:
        _, lock = analysis.load_reference(root, item)
        repeat = item["repeat_id"]
        require(repeat not in locks, "Mean repeat duplicated")
        require(
            lock["schema"] == "lewm-secondary-terminal-mean-lock-v1"
            and lock["repeat_id"] == repeat
            and lock["host"] == item["host"]
            and lock["optimizer_updates"] == 6,
            "Wrong terminal mean/update/host lock",
        )
        genes(lock["parameters"])
        require(
            len(lock["raw_mean"]) == 10 and all(math.isfinite(v) for v in lock["raw_mean"]),
            "Nonfinite raw terminal mean",
        )
        require(tuple(lock["parameters_float32"]) == genes(lock["parameters"]), "FP32 genes differ")
        locks[repeat] = (item, lock)
    require(set(locks) == {0, 1, 2}, "Exactly three terminal mean locks required")
    main_files = {(item["path"], item["sha256"]) for item in main["files"]}
    for entry in profiles:
        repeat = entry["repeat_id"]
        require(
            all(entry["host"][key] == locks[repeat][1]["host"][key] for key in ("host", "pod_id"))
            and entry["host"]["pod_id"]
            == main_plan["profiles"]["secondary30"]["host_assignment"][str(repeat)],
            "Mean profile changed physical host",
        )
        ordinary = entry["ordinary_protocol"]
        require((ordinary["path"], ordinary["sha256"]) in main_files, "Ordinary profile not frozen")
    return manifest, plan, main_plan, locks


def clone_contract(root, entry, lock_ref, lock):
    _, protocol = analysis.load_reference(root, entry["protocol"])
    _, ordinary = analysis.load_reference(root, entry["ordinary_protocol"])
    for key in ("evaluation", "source_files", "source_sha256", "case_preparation", "host"):
        require(protocol[key] == ordinary[key], "Mean changed common runtime contract: " + key)
    require(
        protocol["host"] == entry["host"]
        and protocol["evaluation"]["execution_pod_id"] == entry["host"]["pod_id"]
        and protocol["evaluation"]["maximum_episode_actions"] == 30
        and protocol["output_namespace"] == entry["output_namespace"]
        and protocol["maximum_episode_starts"] == 16
        and protocol["maximum_environment_actions"] == 480,
        "Mean host, output or hard allowance differs",
    )
    declared = protocol["terminal_mean"]
    require(
        declared["lock"] == {k: lock_ref[k] for k in ("path", "sha256")}
        and declared["mean_parameters"] == lock["parameters"]
        and declared["completed_updates"] == 6
        and declared["ordinary_protocol"] == entry["ordinary_protocol"],
        "Mean provenance differs from frozen lock",
    )
    reference = {j["case"]["source_episode"]: j for j in ordinary["jobs"] if j["arm"] == "cma"}
    require(len(reference) == 16 and len(protocol["jobs"]) == 16, "Exactly sixteen source cases")
    require(len({j["job_id"] for j in protocol["jobs"]}) == 16, "Mean job duplicated")
    for job in protocol["jobs"]:
        original = reference[job["case"]["source_episode"]]
        for key in ("case", "environment_seed", "policy_seed", "repeat_id", "max_actions"):
            require(job[key] == original[key], "Mean changed original input/seed/cap: " + key)
        require(
            job["arm"] == MEAN
            and job["method"] == "cma"
            and job["parameters"] == lock["parameters"]
            and job["max_actions"] == 30
            and job.get("mode", "deterministic") == original.get("mode", "deterministic"),
            "Wrong mean policy, method or mode",
        )
    require(
        collections.Counter(j["case"]["condition"] for j in protocol["jobs"])
        == {c: 4 for c in analysis.CONDITIONS},
        "Mean condition quota changed",
    )
    return protocol, ordinary


def completed(root, protocol_ref, protocol, accounting, host):
    folder = analysis.resolve(root, protocol["output_namespace"])
    state_path, source_path = folder / "state.json", folder / "source-freeze.json"
    state, source = load(state_path), load(source_path)
    require(
        source["protocol_sha256"] == protocol_ref["sha256"]
        and source["sources"] == protocol["source_sha256"]
        and source["pod_id"] == host,
        "Completed source/protocol/execution host changed",
    )
    require(state["phase"] == "complete" and state["active"] is None, "Profile incomplete")
    require(set(state["jobs"]) == {j["job_id"] for j in protocol["jobs"]}, "Job set differs")
    physical = {}
    for item in accounting["components"]:
        if item["kind"] == "physical_episode" and item["owning_pod_id"] == host:
            path = analysis.resolve(root, item["source"]["path"])
            require(path not in physical, "Physical episode accounted twice")
            physical[path] = item
    rows, references = [], [ref(root, state_path), ref(root, source_path)]
    for planned in protocol["jobs"]:
        job_folder = folder / "jobs" / planned["job_id"]
        row = state["jobs"][planned["job_id"]]
        require(load(job_folder / "started.json") == planned, "Started job differs")
        require(load(job_folder / "result.json") == row, "Retained outer result differs")
        require(not row["error"] and not row["extra"]["frame_flags"], "Technical/input failure")
        for key in ("method", "environment_seed", "policy_seed", "parameters"):
            require(row[key] == planned[key], "Actual job differs: " + key)
        require(row["case_id"] == planned["case"]["case_id"], "Actual case differs")
        inner_path = analysis.resolve(root, row["extra"]["episode_result"])
        require(inner_path == job_folder / "episode/result.json", "Foreign physical import refused")
        inner_sha = digest(inner_path)
        require(inner_sha == row["extra"]["episode_result_sha256"], "Inner bytes changed")
        inner = load(inner_path)
        for key in (
            *METRICS,
            "environment_actions",
            "progress",
            "resources",
            "method",
            "case_id",
            "environment_seed",
            "policy_seed",
        ):
            require(inner[key] == row[key], "Outer/inner mismatch: " + key)
        require(
            all(isinstance(row[k], bool) for k in METRICS)
            and not (row["safe_completion"] and (row["violation"] or row["unfinished"]))
            and isinstance(row["environment_actions"], int)
            and 0 <= row["environment_actions"] <= planned["max_actions"]
            and math.isfinite(row["progress"])
            and 0 <= row["progress"] <= 1,
            "Invalid outcomes, progress or actions",
        )
        require(
            inner["maximum_actions"] == planned["max_actions"]
            and inner["original_goal"] == planned["case"]["goal_xy"]
            and inner["shape"] == planned["case"]["shape"],
            "Original goal/geometry/cap changed",
        )
        component = physical.get(inner_path)
        require(
            component and component["source"]["sha256"] == inner_sha,
            "Physical episode missing from complete own-host accounting",
        )
        for key, value in row["resources"].items():
            require(
                isinstance(value, (int, float)) and math.isfinite(value) and value >= 0,
                "Invalid resource counter",
            )
            require(component["resources"].get(key) == value, "Accounting counter differs")
        rows.append(
            {
                "arm": planned["arm"],
                "case_id": row["case_id"],
                "source_episode": planned["case"]["source_episode"],
                "condition": planned["case"]["condition"],
                **{k: int(row[k]) for k in METRICS},
                "progress": row["progress"],
                "environment_actions": row["environment_actions"],
                "resources": row["resources"],
                "episode_seconds": inner["episode_seconds"],
                "planning_seconds": inner["planning_seconds"],
            }
        )
        references.extend(
            ref(root, p)
            for p in (job_folder / "started.json", job_folder / "result.json", inner_path)
        )
    require(
        sum(r["environment_actions"] for r in rows) <= protocol["maximum_environment_actions"],
        "Profile action allowance exceeded",
    )
    return rows, references


def totals(rows):
    resources = collections.Counter()
    for row in rows:
        resources.update(row["resources"])
    used = sum(r["environment_actions"] for r in rows)
    allowance = 30 * len(rows)
    return {
        "episodes": len(rows),
        **{k: sum(r[k] for r in rows) for k in METRICS},
        "progress_sum": sum(r["progress"] for r in rows),
        "actual_environment_actions": used,
        "hard_environment_action_allowance": allowance,
        "residual_unspent_allowance": allowance - used,
        "episode_seconds_sum": sum(r.get("episode_seconds", 0) for r in rows),
        "planning_seconds_sum": sum(r.get("planning_seconds", 0) for r in rows),
        "elapsed_scope": "Observed owning-host elapsed only; no cross-host speed claim.",
        "resource_counter_totals_by_field": dict(resources),
    }


def contrasts(rows):
    mean = {r["source_episode"]: r for r in rows if r["arm"] == MEAN}
    result = {}
    for arm in BASELINES:
        other = {r["source_episode"]: r for r in rows if r["arm"] == arm}
        require(
            len(mean) == len(other) == 16 and set(mean) == set(other), "Paired case set differs"
        )
        pairs = []
        for episode in sorted(mean):
            a, b = mean[episode], other[episode]
            require(
                a["case_id"] == b["case_id"] and a["condition"] == b["condition"],
                "Paired original case differs",
            )
            pairs.append(
                {
                    "source_episode": episode,
                    "case_id": a["case_id"],
                    "condition": a["condition"],
                    **{k: a[k] - b[k] for k in (*METRICS, "progress", "environment_actions")},
                    "resource_counter_difference_by_field": {
                        k: a["resources"].get(k, 0) - b["resources"].get(k, 0)
                        for k in sorted(set(a["resources"]) | set(b["resources"]))
                    },
                }
            )
        result[arm] = {
            "direction": MEAN + " minus " + arm,
            "source_clusters": 16,
            "case_pairs": pairs,
            "count_differences": {k: sum(p[k] for p in pairs) for k in METRICS},
            "safe_completion_rate_difference": sum(p["safe_completion"] for p in pairs) / 16,
            "progress_sum_difference": sum(p["progress"] for p in pairs),
            "actual_action_difference": sum(p["environment_actions"] for p in pairs),
        }
    return result


def own_host(args):
    root = args.project_root.resolve()
    manifest, plan, main_plan, locks = context(root, args.manifest)
    pod = os.environ["RUNPOD_POD_ID"]
    entries = [p for p in manifest["profiles"] if p["host"]["pod_id"] == pod]
    require(len(entries) == 1, "Exactly one own-host terminal mean profile required")
    entry = entries[0]
    require(digest(args.accounting_report) == args.accounting_report_sha256, "Accounting changed")
    accounting = load(args.accounting_report)
    require(
        accounting["schema"] == ACCOUNTING
        and accounting["owning_pod_id"] == pod
        and accounting["complete_final_accounting"] is True
        and accounting["unresolved"] == [],
        "Complete own-host final accounting must precede reporting",
    )
    lock_ref, lock = locks[entry["repeat_id"]]
    protocol, ordinary = clone_contract(root, entry, lock_ref, lock)
    ordinary_entry = {
        "family": "secondary30",
        "repeat_id": entry["repeat_id"],
        "host": entry["host"],
        "protocol": entry["ordinary_protocol"],
        "state_path": ordinary["output_namespace"] + "/state.json",
    }
    analysis.saved_profile(root, ordinary_entry, main_plan)
    old_rows, old_refs = completed(root, entry["ordinary_protocol"], ordinary, accounting, pod)
    mean_rows, mean_refs = completed(root, entry["protocol"], protocol, accounting, pod)
    rows = old_rows + mean_rows
    duplicates = {
        arm: genes(lock["parameters"])
        == genes(next(j["parameters"] for j in ordinary["jobs"] if j["arm"] == arm))
        for arm in BASELINES
    }
    return {
        "schema": "lewm-terminal-mean-own-host-report-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "owning_pod_id": pod,
        "repeat_id": entry["repeat_id"],
        "host": entry["host"],
        "manifest": ref(root, args.manifest),
        "plan": manifest["plan"],
        "means": manifest["means"],
        "mean_lock": lock_ref,
        "accounting_report": ref(root, args.accounting_report),
        "complete_own_host_final_accounting": True,
        "mean_parameters": lock["parameters"],
        "optimizer_updates": 6,
        "fp32_parameter_equal_to_baseline": duplicates,
        "all_ordinary96_and_mean16_physical_episodes_hash_bound_in_accounting": True,
        "totals_by_arm": {
            arm: totals([r for r in rows if r["arm"] == arm]) for arm in (MEAN, *analysis.ARMS)
        },
        "paired_descriptive_contrasts": contrasts(rows),
        "rows": rows,
        "owned_json_references": old_refs + mean_refs,
        "source_sha256": plan["source_sha256"],
        "new_model_simulator_calls": 0,
        "scope": "Exploratory unselected terminal means after six updates, all three retained. Descriptive paired counts only; no CIs, p-values, family-wise or optimizer-superiority inference.",
    }


def merge(args):
    require(os.environ.get("RUNPOD_POD_ID") == "vukovc3t4o1dui", "Merge belongs on Nav only")
    root = args.project_root.resolve()
    manifest, _, _, _ = context(root, args.manifest)
    require(len(args.reports) == 3, "Exactly three SHA-bound own-host reports required")
    require(
        all(
            digest(path) == expected
            for path, expected in zip(args.reports, args.report_sha256, strict=True)
        ),
        "Transferred own-host report bytes differ from their recorded source hashes",
    )
    reports = [load(path) for path in args.reports]
    expected = {p["host"]["pod_id"] for p in manifest["profiles"]}
    require(
        {r["owning_pod_id"] for r in reports} == expected
        and sorted(r["repeat_id"] for r in reports) == [0, 1, 2],
        "All three hosts/repeats required",
    )
    for report in reports:
        require(
            report["schema"] == "lewm-terminal-mean-own-host-report-v1"
            and report["manifest"] == ref(root, args.manifest)
            and report["all_ordinary96_and_mean16_physical_episodes_hash_bound_in_accounting"]
            and report["complete_own_host_final_accounting"]
            and report["optimizer_updates"] == 6,
            "Incomplete, foreign or differently bound own-host report",
        )
    accounting = load(args.accounting_merge)
    require(
        accounting["schema"] == ACCOUNTING + "-aggregate"
        and accounting["complete_final_accounting"] is True
        and accounting["unresolved"] == []
        and {r["sha256"] for r in accounting["source_reports"]}
        == {r["accounting_report"]["sha256"] for r in reports},
        "Merge needs the exact complete three-host accounting aggregate",
    )
    cases = [set(r["source_episode"] for r in report["rows"]) for report in reports]
    require(all(c == cases[0] and len(c) == 16 for c in cases), "Repeat case clusters differ")
    return {
        "schema": "lewm-terminal-mean-aggregate-report-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manifest": ref(root, args.manifest),
        "accounting_merge": ref(root, args.accounting_merge),
        "own_host_reports": [ref(root, path) for path in args.reports],
        "all_three_fixed_terminal_means_retained": True,
        "source_case_clusters": 16,
        "fixed_controller_repeats": 3,
        "raw_observations_per_arm": 48,
        "raw_observations_are_not_independent_source_cases": True,
        "per_host": reports,
        "equal_fixed_repeat_safe_completion_rate_difference": {
            arm: sum(
                r["paired_descriptive_contrasts"][arm]["safe_completion_rate_difference"]
                for r in reports
            )
            / 3
            for arm in BASELINES
        },
        "all_five_paired_descriptive_contrasts": {
            arm: {
                "direction": MEAN + " minus " + arm,
                "raw_count_differences_across_fixed_repeats": {
                    metric: sum(
                        r["paired_descriptive_contrasts"][arm]["count_differences"][metric]
                        for r in reports
                    )
                    for metric in METRICS
                },
                "equally_averaged_fixed_repeat_rate_differences": {
                    metric: sum(
                        r["paired_descriptive_contrasts"][arm]["count_differences"][metric]
                        for r in reports
                    )
                    / 48
                    for metric in METRICS
                },
                "mean_progress_difference_per_case_and_fixed_repeat": sum(
                    r["paired_descriptive_contrasts"][arm]["progress_sum_difference"]
                    for r in reports
                )
                / 48,
                "actual_action_difference_across_fixed_repeats": sum(
                    r["paired_descriptive_contrasts"][arm]["actual_action_difference"]
                    for r in reports
                ),
                "resource_counter_difference_by_field": {
                    key: sum(
                        r["totals_by_arm"][MEAN]["resource_counter_totals_by_field"].get(key, 0)
                        - r["totals_by_arm"][arm]["resource_counter_totals_by_field"].get(key, 0)
                        for r in reports
                    )
                    for key in sorted(
                        set().union(
                            *(
                                set(r["totals_by_arm"][MEAN]["resource_counter_totals_by_field"])
                                | set(r["totals_by_arm"][arm]["resource_counter_totals_by_field"])
                                for r in reports
                            )
                        )
                    )
                },
                "source_case_clusters": 16,
                "fixed_controller_repeats": 3,
            }
            for arm in BASELINES
        },
        "foreign_physical_arrays_or_episode_json_imported_by_merge": False,
        "new_model_simulator_calls": 0,
        "scope": "Exploratory descriptive mean-minus-five-baseline contrasts conditional on the three frozen controllers/hosts and sixteen source cases. No CIs, p-values, multiplicity correction or optimizer-superiority claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    own = commands.add_parser("own-host")
    own.add_argument("--accounting-report", type=Path, required=True)
    own.add_argument("--accounting-report-sha256", required=True)
    combined = commands.add_parser("merge")
    combined.add_argument("--reports", nargs=3, type=Path, required=True)
    combined.add_argument("--report-sha256", nargs=3, required=True)
    combined.add_argument("--accounting-merge", type=Path, required=True)
    for child in (own, combined):
        child.add_argument("--manifest", type=Path, required=True)
        child.add_argument("--project-root", type=Path, default=ROOT)
        child.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "Never overwrite a frozen report")
    report = own_host(args) if args.command == "own-host" else merge(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"path": str(args.out), "sha256": digest(args.out)}))


if __name__ == "__main__":
    main()
