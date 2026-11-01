"""Explicit final-relocation metadata guards around the unchanged V8 auditor.

Only audit execution authority changes. All saved physics, target, geometry,
counter, refusal and journal replay is delegated to the exact immutable V8 code.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/scripts"))
sys.path.insert(0, str(ROOT / "experiments"))
import analyze_lewm_navigation_campaign_20261031_v4 as analysis
from helpers import lewm_navigation_final_relocation as relocation

PARENT = ROOT / "experiments/scripts/audit_lewm_campaign_evidence_v8.py"
PARENT_SHA = "7f581e5602b6251aad20570ed9df3394706429f25eb5cd9bd6a1626c89118cdc"
PLAN_SHA = "85fd2c076245e063a63ec80b5b2d996e3e5f240839c7b7cb0b2e768f08ab0393"


def parent():
    analysis.require(analysis.digest(PARENT) == PARENT_SHA, "Immutable V8 source changed")
    spec = importlib.util.spec_from_file_location("lewm_exact_v8_auditor", PARENT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def guard(root, stage, protocol_path, plan_path):
    analysis.require(analysis.digest(plan_path) == PLAN_SHA, "Unqualified continuation plan")
    plan = analysis.verify_plan(plan_path)
    value = relocation.authority(root, plan["final_execution_relocation"])
    pod = os.environ.get("RUNPOD_POD_ID")
    analysis.require(pod == value["actual_final_host"]["pod_id"], "Wrong actual audit pod")
    declaration = {
        "path": str(protocol_path.relative_to(root)),
        "sha256": analysis.digest(protocol_path),
    }
    rows = [row for row in value["profiles"] if row["protocol"] == declaration]
    analysis.require(len(rows) == 1, "Audit only an exact authorized relocated final profile")
    protocol = analysis.load(protocol_path)
    analysis.require(
        stage == analysis.resolve(root, protocol["output_namespace"]), "Foreign stage namespace"
    )
    source = analysis.load(stage / "source-freeze.json")
    analysis.require(
        source["protocol_sha256"] == declaration["sha256"]
        and source["sources"] == protocol["source_sha256"],
        "Original protocol/source pins changed",
    )
    entry = {"protocol": declaration, "host": protocol["host"]}
    actual = relocation.verify_execution(root, entry, protocol, source, plan)
    analysis.require(actual["pod_id"] == pod, "Audited execution belongs to another actual pod")
    return {
        "kind": "exact_capacity_driven_final_relocation_v3",
        "pod_id": pod,
        "analysis_plan": {"path": str(plan_path.relative_to(root)), "sha256": PLAN_SHA},
        "authority": plan["final_execution_relocation"],
        "continuation": source["continuation"],
        "original_protocol": declaration,
        "original_logical_host": entry["host"],
        "actual_final_host": actual,
        "unchanged_parent_auditor_sha256": PARENT_SHA,
        "physics_geometry_goal_counter_refusal_journal_replay_unchanged": True,
    }, (entry, protocol, source, plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--analysis-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--metadata-qualification", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    stage, protocol_path, plan_path, out = [
        analysis.resolve(root, path)
        for path in (args.stage, args.protocol, args.analysis_plan, args.out)
    ]
    analysis.require(not out.exists() and not out.is_relative_to(stage), "Fresh audit output")
    host, context = guard(root, stage, protocol_path, plan_path)
    auditor = parent()
    out.parent.mkdir(parents=True, exist_ok=True)
    source_path = out.with_suffix(".source.py")
    analysis.require(not source_path.exists(), "Fresh retained audit source required")
    source_path.write_bytes(Path(__file__).read_bytes())
    try:
        if args.metadata_qualification:
            entry, protocol, source, plan = context
            negatives = {}
            for label, changed in (
                ("wrong_actual_pod", {**source, "pod_id": "not-the-authorized-pod"}),
                (
                    "changed_sidecar_digest",
                    {**source, "continuation": {**source["continuation"], "sha256": "0" * 64}},
                ),
            ):
                try:
                    relocation.verify_execution(root, entry, protocol, changed, plan)
                except (ValueError, AssertionError):
                    negatives[label] = True
                else:
                    raise AssertionError("Metadata negative accepted: " + label)
            report = {
                "schema": "lewm-final-relocation-auditor-v9-metadata-qualification-v1",
                "positive_exact_sidecar": True,
                "negative_rejections": negatives,
                "checks_passed": 3,
                "real_trace_arrays_read": 0,
            }
        else:
            report = auditor.audit(root, stage, protocol_path, False)
    except BaseException:
        out.with_suffix(".failure.json").write_text(
            json.dumps({"error": traceback.format_exc(), "auditor_sha256": analysis.digest(__file__)})
            + "\n"
        )
        raise
    report.update(
        execution_host_authority=host,
        auditor_source_path=str(source_path.relative_to(root)),
        auditor_source_sha256=analysis.digest(__file__),
        new_simulator_interactions=0,
        new_frozen_model_queries=0,
    )
    temporary = out.with_suffix(".part")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(out)
    print(json.dumps({"path": str(out), "sha256": analysis.digest(out), "passed": True}))


if __name__ == "__main__":
    main()
