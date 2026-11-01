"""Relocation-aware metadata guards around the unchanged terminal-mean summaries.

Reports remain per fixed repeat, even when two repeats share one actual host and
one accounting report. The merge retains the original Nav repeat0 report exactly.
No arrays, models, simulator, fitting or new statistical inference are used.
"""

import argparse
import datetime as dt
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/scripts"))
sys.path.insert(0, str(ROOT / "experiments"))
import analyze_lewm_navigation_campaign_20261031_v4 as analysis
import report_lewm_navigation_terminal_mean_sidecar as core
from helpers.lewm_navigation_final_relocation import actual_host, authority, verify_execution

RETAINED_NAV_REPORT = "49ce479fc75dff2f06d8ae3aa2b684102eeb158511c45ccee6bc3d7eba379184"
RETAINED_NAV_ACCOUNTING = "312d4ba2e8f68ca10fa1898b0c127f0a4975d5f7f8c0235b7376feb9990eda2d"
BASELINES, METRICS, MEAN = core.BASELINES, core.METRICS, core.MEAN
require, digest, load, ref = core.require, core.digest, core.load, core.ref


def context(args):
    root = args.project_root.resolve()
    plan = analysis.verify_plan(args.plan)
    own_name = str(Path(__file__).relative_to(ROOT))
    require(plan["source_sha256"][own_name] == digest(__file__), "New report source changed")
    relocation = authority(root, plan["final_execution_relocation"])
    require(
        relocation["original_terminal_mean_manifest"] == ref(root, args.manifest),
        "Relocation authority binds another original mean sidecar",
    )
    manifest = load(args.manifest)
    require(manifest["schema"] == "lewm-terminal-mean-final-sidecar-v1", "Unknown mean sidecar")
    _, old_plan = analysis.load_reference(root, manifest["plan"])
    require(
        old_plan["schema"] == "lewm-terminal-mean-exploratory-plan-v1"
        and tuple(old_plan["baseline_arms"]) == BASELINES
        and old_plan["mean_arm"] == MEAN
        and old_plan["denominator_each_repeat"] == 16,
        "Original descriptive contract changed",
    )
    for name, expected in old_plan["source_sha256"].items():
        require(digest(ROOT / name) == expected, "Original descriptive dependency changed")
    require(manifest["means"] == old_plan["means"], "Original mean seal changed")
    _, means = analysis.load_reference(root, manifest["means"])
    require(
        means["schema"] == "lewm-secondary-terminal-means-v1"
        and means["all_three_means_fixed_before_final_metadata"] is True
        and manifest["all_main_outputs_absent_at_binding"] is True,
        "Means must remain fixed before original final access",
    )
    _, main = analysis.load_reference(root, manifest["main_manifest"])
    analysis.joint_readiness(root, main["joint_opening"])
    main_files = {(item["path"], item["sha256"]) for item in main["files"]}
    require(sorted(p["repeat_id"] for p in manifest["profiles"]) == [0, 1, 2], "Retain3 repeats")
    locks = {}
    for item in means["locks"]:
        _, lock = analysis.load_reference(root, item)
        repeat = item["repeat_id"]
        require(repeat not in locks and lock["repeat_id"] == repeat, "Duplicate/wrong mean repeat")
        require(
            lock["schema"] == "lewm-secondary-terminal-mean-lock-v1"
            and lock["optimizer_updates"] == 6
            and lock["host"] == item["host"]
            and tuple(lock["parameters_float32"]) == core.genes(lock["parameters"]),
            "Original terminal genotype provenance changed",
        )
        locks[repeat] = (item, lock)
    require(set(locks) == {0, 1, 2}, "Three original mean locks required")
    for entry in manifest["profiles"]:
        lock = locks[entry["repeat_id"]][1]
        require(
            all(entry["host"][k] == lock["host"][k] for k in ("host", "pod_id")),
            "Original logical host changed",
        )
        original = entry["ordinary_protocol"]
        require(
            (original["path"], original["sha256"]) in main_files,
            "Ordinary comparator is not in original main manifest",
        )
    return root, plan, relocation, manifest, locks


def repeat_report(args):
    root, plan, relocation, manifest, locks = context(args)
    require(args.repeat_id in (1, 2), "Preserve existing Nav repeat0 instead of rerunning it")
    entry = next(p for p in manifest["profiles"] if p["repeat_id"] == args.repeat_id)
    actual = actual_host(relocation, entry)
    pod = os.environ.get("RUNPOD_POD_ID")
    require(pod == actual["pod_id"], "Per-repeat report belongs on its actual owning pod")
    require(digest(args.accounting_report) == args.accounting_report_sha256, "Accounting changed")
    accounting = load(args.accounting_report)
    require(
        accounting["schema"] == core.ACCOUNTING
        and accounting["owning_pod_id"] == pod
        and accounting["complete_final_accounting"] is True
        and accounting["unresolved"] == [],
        "Complete own-host final accounting must precede physical imports/reporting",
    )
    lock_ref, lock = locks[args.repeat_id]
    protocol, ordinary = core.clone_contract(root, entry, lock_ref, lock)
    ordinary_entry = {
        "family": "secondary30",
        "repeat_id": args.repeat_id,
        "host": entry["host"],
        "protocol": entry["ordinary_protocol"],
        "state_path": ordinary["output_namespace"] + "/state.json",
    }
    analysis.saved_profile(root, ordinary_entry, plan)
    mean_source = load(analysis.resolve(root, protocol["output_namespace"]) / "source-freeze.json")
    verified = verify_execution(root, entry, protocol, mean_source, plan)
    require(verified["pod_id"] == pod, "Mean execution does not match authorized actual host")
    ordinary_rows, ordinary_refs = core.completed(
        root, entry["ordinary_protocol"], ordinary, accounting, pod
    )
    mean_rows, mean_refs = core.completed(root, entry["protocol"], protocol, accounting, pod)
    rows = ordinary_rows + mean_rows
    return {
        "schema": "lewm-terminal-mean-repeat-report-v2",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repeat_id": args.repeat_id,
        "owning_pod_id": pod,
        "host": actual,
        "original_logical_host": entry["host"],
        "manifest": ref(root, args.manifest),
        "plan": manifest["plan"],
        "relocation_analysis_plan": ref(root, args.plan),
        "final_execution_relocation": plan["final_execution_relocation"],
        "means": manifest["means"],
        "mean_lock": lock_ref,
        "accounting_report": ref(root, args.accounting_report),
        "complete_own_host_final_accounting": True,
        "all_ordinary96_and_mean16_physical_episodes_hash_bound_in_accounting": True,
        "mean_parameters": lock["parameters"],
        "optimizer_updates": 6,
        "fp32_parameter_equal_to_baseline": {
            arm: core.genes(lock["parameters"])
            == core.genes(next(j["parameters"] for j in ordinary["jobs"] if j["arm"] == arm))
            for arm in BASELINES
        },
        "totals_by_arm": {
            arm: core.totals([r for r in rows if r["arm"] == arm])
            for arm in (MEAN, *core.analysis.ARMS)
        },
        "paired_descriptive_contrasts": core.contrasts(rows),
        "rows": rows,
        "owned_json_references": ordinary_refs + mean_refs,
        "source_sha256": plan["source_sha256"],
        "new_model_simulator_calls": 0,
        "scope": "Same frozen sixteen source cases, six-update means and descriptive arithmetic. Per fixed repeat, not per unique actual host; no intervals, p-values or optimizer-superiority inference.",
    }


def merge(args):
    root, plan, relocation, manifest, _ = context(args)
    require(
        os.environ.get("RUNPOD_POD_ID") in plan["execution"]["allowed_pod_ids"],
        "Merge belongs on a declared rented analysis host",
    )
    require(
        all(digest(p) == s for p, s in zip(args.reports, args.report_sha256, strict=True)),
        "Transferred per-repeat report bytes changed",
    )
    reports = [load(p) for p in args.reports]
    require(sorted(r["repeat_id"] for r in reports) == [0, 1, 2], "Retain exactly3 fixed repeats")
    accounting = load(args.accounting_merge)
    require(
        accounting["schema"] == core.ACCOUNTING + "-aggregate"
        and accounting["complete_final_accounting"] is True
        and accounting["unresolved"] == [],
        "Complete global source-host accounting is required",
    )
    raw_reports = [load(p) for p in args.accounting_reports]
    require(
        len(raw_reports) == 4 and len({r["owning_pod_id"] for r in raw_reports}) == 4,
        "Exactly4 authoritative source-pod accounting reports required",
    )
    require(
        {digest(p) for p in args.accounting_reports}
        == {r["sha256"] for r in accounting["source_reports"]},
        "Global accounting refs changed",
    )
    accounts = {}
    for path, value in zip(args.accounting_reports, raw_reports, strict=True):
        require(
            value["schema"] == core.ACCOUNTING
            and value["complete_final_accounting"] is True
            and value["unresolved"] == [],
            "Incomplete authoritative source accounting",
        )
        accounts[value["owning_pod_id"]] = digest(path)
    require(
        set(plan["profiles"]["secondary30"]["host_assignment"].values()) <= set(accounts),
        "Original training/selection source host missing from accounting",
    )
    for path, report in zip(args.reports, reports, strict=True):
        repeat = report["repeat_id"]
        entry = next(p for p in manifest["profiles"] if p["repeat_id"] == repeat)
        actual = actual_host(relocation, entry)
        require(
            report["manifest"] == ref(root, args.manifest)
            and report["owning_pod_id"] == actual["pod_id"]
            and report["accounting_report"]["sha256"] == accounts[actual["pod_id"]]
            and report["complete_own_host_final_accounting"] is True
            and report["all_ordinary96_and_mean16_physical_episodes_hash_bound_in_accounting"]
            and report["optimizer_updates"] == 6,
            "Per-repeat ownership/accounting differs",
        )
        if repeat == 0:
            require(
                digest(path) == RETAINED_NAV_REPORT
                and report["schema"] == "lewm-terminal-mean-own-host-report-v1"
                and report["accounting_report"]["sha256"] == RETAINED_NAV_ACCOUNTING,
                "Use the exact retained original Nav mean0 report; no rerun/substitution",
            )
        else:
            require(
                report["schema"] == "lewm-terminal-mean-repeat-report-v2"
                and report["source_sha256"] == plan["source_sha256"]
                and report["final_execution_relocation"] == plan["final_execution_relocation"],
                "Relocated repeat report source/authority differs",
            )
    cases = [set(r["source_episode"] for r in report["rows"]) for report in reports]
    require(
        all(c == cases[0] and len(c) == 16 for c in cases), "Repeat source-case clusters differ"
    )
    return {
        "schema": "lewm-terminal-mean-aggregate-report-v2",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manifest": ref(root, args.manifest),
        "analysis_plan": ref(root, args.plan),
        "final_execution_relocation": plan["final_execution_relocation"],
        "accounting_merge": ref(root, args.accounting_merge),
        "accounting_source_reports": [ref(root, p) for p in args.accounting_reports],
        "per_repeat_reports": [ref(root, p) for p in args.reports],
        "all_three_fixed_terminal_means_retained": True,
        "source_case_clusters": 16,
        "fixed_controller_repeats": 3,
        "raw_observations_per_arm": 48,
        "raw_observations_are_not_independent_source_cases": True,
        "actual_final_host_count": len({r["owning_pod_id"] for r in reports}),
        "reports_may_share_an_authoritative_own_host_accounting_sha": True,
        "per_repeat": reports,
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
        "new_model_simulator_calls": 0,
        "foreign_physical_arrays_or_episode_json_imported_by_merge": False,
        "scope": "Original conditional descriptive estimand and arithmetic retained. All3 fixed means, sixteen source cases, explicit actual-host blocks and4-source accounting. No CIs, p-values, family-wise or optimizer-superiority claim.",
    }


def main():
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    own = commands.add_parser("repeat")
    own.add_argument("--repeat-id", type=int, choices=(1, 2), required=True)
    own.add_argument("--accounting-report", type=Path, required=True)
    own.add_argument("--accounting-report-sha256", required=True)
    combined = commands.add_parser("merge")
    combined.add_argument("--reports", nargs=3, type=Path, required=True)
    combined.add_argument("--report-sha256", nargs=3, required=True)
    combined.add_argument("--accounting-merge", type=Path, required=True)
    combined.add_argument("--accounting-reports", nargs=4, type=Path, required=True)
    for child in (own, combined):
        child.add_argument("--manifest", type=Path, required=True)
        child.add_argument("--plan", type=Path, required=True)
        child.add_argument("--project-root", type=Path, default=ROOT)
        child.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "Never overwrite an original or relocated report")
    report = repeat_report(args) if args.command == "repeat" else merge(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"path": str(args.out), "sha256": digest(args.out)}))


if __name__ == "__main__":
    main()
