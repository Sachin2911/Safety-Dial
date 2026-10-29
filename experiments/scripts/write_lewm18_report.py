"""Write a readable result record from the independently audited task-18 summary."""

import argparse
import json
from pathlib import Path

from package_lewm18_evidence import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = json.loads((args.root / "analysis/summary.json").read_text())
    assert report["unique_attempts"] == 1352 and not report["missing_result_attempts"]
    names = {
        "native": "Native",
        "canonical": "Original SafeCEM",
        "fixed": "Tuned fixed",
        "feedback": "Feedback",
    }
    lines = [
        "# Task 18: helping SafeCEM finish object-moving tasks",
        "",
        "27 October 2026. Final experiment results; preservation and rental closeout are recorded separately.",
        "",
        "We kept the learned world model and requested safety margin unchanged and changed how the planner searches for moves. The final panel has twelve cases and three planner seeds per task. Development and final cases come from different demonstration episodes. All settings were frozen before the final panel ran.",
        "",
        "## The result at the largest safety margin",
        "",
        "Safe completion means reaching the task goal without entering the forbidden area or its requested buffer. A breach and an unfinished task can happen in the same attempt. Engagement means movement, not useful progress.",
        "",
        "| Task | Planner | Safe / attempts | Breaches | Unfinished | Engaged | Capped actions |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    strict = {}
    for kind, margin in (("pusht", 12), ("cube", 0.01)):
        strict[kind] = {}
        for method, name in names.items():
            r = next(
                r
                for r in report["tables"]
                if (r["stage"], r["env"], r["margin"], r["method"]) == ("H", kind, margin, method)
            )
            assert r["n"] == 36 and not r["error"]
            strict[kind][method] = r
            label = "PushT, 12 px" if kind == "pusht" else "Cube, 1 cm"
            lines.append(
                f"| {label} | {name} | {r['safe']}/36 | {r['margin_breach']} | {r['unfinished']} | {r['engaged']}/36 | {r['mean_J_safe']:.2f} |"
            )
    lines += [
        "",
        "Capped actions count the actions used by a safe completion. Every unsafe or unfinished attempt costs 50 actions. This includes early controller failures, so stopping early cannot look fast. Conditional successful-action means have their own denominators in the full tables.",
        "",
        "## Paired comparisons",
        "",
    ]
    for kind, margin in (("pusht", 12), ("cube", 0.01)):
        for first, second in (("fixed", "canonical"), ("feedback", "fixed")):
            for metric in ("safe", "J_safe"):
                c = next(
                    c
                    for c in report["contrasts"]
                    if (c["env"], c["margin"], c["first"], c["second"], c["metric"])
                    == (kind, margin, first, second, metric)
                )
                scale = 100 if metric == "safe" else 1
                unit = (
                    "percentage points of safe completion" if metric == "safe" else "capped actions"
                )
                lo, hi = (x * scale for x in c["interval"])
                lines.append(
                    f"- {kind}: {names[first]} minus {names[second]}: {c['estimate'] * scale:+.2f} {unit}, descriptive 95% case-bootstrap interval [{lo:.2f}, {hi:.2f}]."
                )
    lines += [
        "",
        "Intervals resample twelve whole cases, preserving all paired seeds. They are conditional on the selected settings, not uncertainty over repeated tuning campaigns. No multiple-comparison claim gate is applied. A collapsed empirical interval means the observed case effects were identical; it does not prove population equivalence or zero uncertainty.",
        "",
        "## Actual planning time",
        "",
        "The serial timing pass uses the first two final cases, a separate common seed and the largest clearance. It runs without competing experiment workers. These are descriptive timings for two cases, not a population speed claim. Video playback is eight actions per second and is unrelated to compute latency.",
        "",
        "| Task | Planner | Seconds per solve | Safe / attempts |",
        "|---|---|---:|---:|",
    ]
    for kind in ("pusht", "cube"):
        for method, name in names.items():
            r = next(
                r
                for r in report["tables"]
                if (r["stage"], r["env"], r["method"]) == ("T", kind, method)
            )
            lines.append(
                f"| {kind} | {name} | {r['planning_seconds_per_replan']:.3f} | {r['safe']}/{r['n']} |"
            )
    lines += [
        "",
        "## What the comparison covers",
        "",
        "PushT keeps the pinned environment's joint pusher/block position and block-angle goal. Cube measures the object centre reaching within 4 cm of its target, including height. This is the original object-moving objective with LeWM dataset-pair goals, not the earlier gripper-reaching probe and not the full official five-goal OGBench evaluation.",
        "",
        "The safety rule covers the pusher centre or gripper pinch point against a synthetic forbidden region. It does not certify the whole robot or moved object against collision. PushT checks every physics step using its known pusher dynamics. Cube uses approximate population screening followed by true-simulator prefix checks, with full state restoration and no unchecked fallback.",
        "",
        "Each task uses its own frozen checkpoint. Six fixed configurations are compared on six development cases, two seeds and three margins (216 attempts). Four feedback presets then use six cases, one seed and three margins (72 more). Feedback inherits fixed tuning and has additional development cost. No evolutionary search ran in task 18. The previous task-17 optimizer comparison remains a separate, negative result.",
        "",
        "The separate native context panel uses twelve unfiltered dataset pairs and 25 actions per replan. Cube completes 10/12, but seven pairs already meet the object goal at reset. Among the five not initially solved, three complete. This cannot substitute for the nontrivial final moving-object panel.",
        "",
        "## Repair and complete accounting",
        "",
        "Publication review found that PushT reset advances the simulation slightly, while the initial cohort anchored its first safety prediction to the pre-reset dataset state. We corrected the anchor to the returned observation and repeated the entire frozen protocol, including development. All 676 original attempts remain preserved and charged, but excluded from efficacy estimates. Repaired engineering verified the post-reset state and dense pusher predictions. The cases, seeds, candidate library and selection rule did not change.",
        "",
        "Each valid task cohort has 676 attempts: 8 engineering, 216 fixed development, 72 feedback development, 360 unique final episodes, 12 context and 8 serial timing. The two valid cohorts total 1,352; the excluded PushT cohort brings all scored attempts to 2,028. Native final episodes are rescored at three margins and are not counted as three separate experiments. Pre-data failed checks and synthetic model checks are counted separately.",
        "",
        "| Charged work across all cohorts and preliminary checks | Count |",
        "|---|---:|",
    ]
    for key, value in report["all_charged_work"].items():
        lines.append(f"| {key.replace('_', ' ')} | {value:,} |")
    lines += [
        "",
        "Reset stabilization and counterfactual previews are included explicitly. The original PushT reset work and pre-data reset work were reconstructed from pinned reset code; the repair measures reset work directly. Cube's fifty reset physics steps per episode are already in its recorded physics counter and are not counted twice. Actual actions, simulator checks and learned-model predictions are different costs.",
        "",
        "Every scored-stage source archive and every selected result was checked against the frozen job lists. Independent scoring recomputed completion, original-hazard violations, margin breaches, unfinished outcomes, engagement, action counts and capped time from saved traces. Paired initial states matched within 1e-10. The pre-data engineering checks retain scripts, results and source hashes, but their complete intermediate runtime sources were not recovered.",
        "",
        "## Evidence and examples",
        "",
        "- [Frozen protocol](../../configs/lewm18/protocol.json), [repair admission](18-reset-repair-admission.json), [engineering repair check](18-pusht-repair-engineering.json).",
        "- [Final local preservation](18-final-local-preservation.json), [dataset upload verification](18-datasets-preservation.json), [final closeout](18-closeout.json), [final publication](18-final-publication.json).",
        "- [PushT data, videos and pictures](https://huggingface.co/datasets/Sachioster/SafetyDial-PushT/tree/main/campaigns/lewm18).",
        "- [Cube data, videos and pictures](https://huggingface.co/datasets/Sachioster/SafetyDial-Cube/tree/main/campaigns/lewm18).",
        "",
        "The two videos per task show the first two cases and first seed selected before outcomes. Galleries include all twelve final cases at that seed, including failures. Aggregate tables support the claims; examples illustrate behavior. Hugging Face datasets remain private.",
        "",
        "The result supports evaluating a strong fixed configuration before attributing improvement to feedback or evolution. Configuration changes are bundled; there is no single-knob causal claim. Without a tuned native ablation, the experiment also cannot separate a generic planning improvement from an improvement specific to planning under the safety constraint.",
        "",
    ]
    (ROOT / "progress/receipts/18-lewm-results.md").write_text("\n".join(lines))
    print(json.dumps({"report_written": True, "selected_attempts": report["unique_attempts"]}))


if __name__ == "__main__":
    main()
