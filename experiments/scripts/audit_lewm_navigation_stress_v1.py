"""Audit terminal stress geometry, original goals, selectors and attempt journals."""

import argparse
import collections
import json
from pathlib import Path

from audit_lewm_navigation_progress_v2 import ROOT, audit, digest, require


def run(protocol_path, output_path):
    from helpers.lewm_progress_journal import replay_progress_journal

    core_path = output_path.with_name(output_path.stem + "-core.json")
    report = audit(protocol_path, core_path)
    protocol = json.loads(protocol_path.read_text())
    metadata = protocol["metadata_panel"]
    panel_path = ROOT / metadata["path"]
    require(digest(panel_path) == metadata["sha256"], "Derived geometry panel changed")
    panel = json.loads(panel_path.read_text())
    parent_path = ROOT / panel["parent_panel"]["path"]
    require(
        digest(parent_path) == panel["parent_panel"]["sha256"],
        "Original admission metadata changed",
    )
    parents = {row["case_id"]: row for row in json.loads(parent_path.read_text())["cases"]}
    require(
        panel["final_pixels_read"] == 0 and not panel["selection_geometry_derived"],
        "Holdout changed",
    )
    rule = panel["geometry_transform"]
    require(
        rule["maximum_radius"] == 12 and rule["admissible_fraction"] == 0.95, "Frozen stress rule"
    )
    require(rule["body_radius"] == 7 and rule["margin"] == 3, "Body or safety margin changed")
    require(rule["motion_allowance"] == 12.405431747436523, "Motion allowance changed")
    cases = {row["case_id"]: row for row in panel["cases"]}
    geometry = []
    for case in panel["cases"]:
        original = parents[case["parent_case_id"]]
        for field in (
            "input_path",
            "input_sha256",
            "initial_xy",
            "goal_xy",
            "source_episode",
            "source_step",
            "goal_step",
        ):
            require(case[field] == original[field], "Stress changed original input/goal: " + field)
        require(case["seed"] == 321000 + case["source_episode"], "Frozen fresh seed changed")
        if original["shape"]["kind"] == "none":
            require(case["shape"] == original["shape"], "No-hazard control changed")
            continue
        old_radius = original["shape"]["radius"]
        required = 3 + 12.405431747436523 + 2
        maximum = old_radius + min(original["endpoint_clearance"]) - required
        expected_radius = min(12, max(old_radius, 0.95 * maximum))
        require(case["shape"]["radius"] == expected_radius, "Wrong stress radius")
        require(case["shape"]["centre"] == original["shape"]["centre"], "Hazard centre changed")
        require(case["shape"]["axis"] == original["shape"]["axis"], "Hazard axis changed")
        expected_clearance = [
            value - (expected_radius - old_radius) for value in original["endpoint_clearance"]
        ]
        require(
            case["endpoint_clearance"] == expected_clearance, "Derived admission clearance changed"
        )
        require(min(expected_clearance) >= required, "Buffered endpoint admission failed")
        geometry.append(
            {
                "case_id": case["case_id"],
                "radius": expected_radius,
                "minimum_endpoint_clearance": min(expected_clearance),
                "required_endpoint_clearance": required,
            }
        )
    journal_records, per_condition = 0, collections.defaultdict(lambda: collections.Counter())
    run_path = ROOT / protocol["output_namespace"]
    for planned, row in zip(protocol["jobs"], report["records"], strict=True):
        require(
            planned["case"] == cases[planned["case"]["case_id"]],
            "Pilot case differs from derived panel",
        )
        job = run_path / "jobs" / row["job_id"]
        replay = replay_progress_journal(job / "progress-journal.jsonl")
        require(
            replay["last"]["event"] == "final" and replay["truncated_tail_bytes"] == 0,
            "Nonterminal journal",
        )
        saved = json.loads((job / "episode/result.json").read_text())
        require(
            replay["last"]["resources"] == saved["resources"], "Attempt journal counters differ"
        )
        require(
            replay["last"]["environment_actions"] == saved["environment_actions"],
            "Action accounting differs",
        )
        journal_records += replay["records"]
        row["condition"] = planned["case"]["condition"]
        row["journal_sha256"] = replay["sha256"]
        group = per_condition[(row["method"], row["condition"])]
        group["episodes"] += 1
        for field in ("safe_completion", "violation", "unfinished", "engaged", "actions"):
            group[field] += row[field]
    report.update(
        stress_geometry_verified=True,
        metadata_admission=geometry,
        final_journals_verified=16,
        journal_records_checked=journal_records,
        condition_counts={
            method + "/" + condition: dict(value)
            for (method, condition), value in per_condition.items()
        },
        stress_audit_source_sha256=digest(Path(__file__)),
        core_report_sha256=digest(core_path),
    )
    if output_path.exists():
        raise FileExistsError("Use a fresh audit namespace")
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.protocol, args.out)
    print(
        json.dumps(
            {
                "summary": report["summary"],
                "condition_counts": report["condition_counts"],
                "journal_records": report["journal_records_checked"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
