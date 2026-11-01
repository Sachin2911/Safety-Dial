"""Account for each physical study component once, retaining interrupted-work bounds."""

import argparse
import collections
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.safety_data import write_json
from helpers.safety_model import file_digest

BASE = ROOT / "outputs/lewm-preset-expansion-20261030"


def read(path):
    return json.loads(Path(path).read_text())


def collect(require_complete=False):
    components = []
    lower = collections.Counter()
    extra = collections.Counter()

    def add(label, path, resources, **fields):
        lower.update(resources)
        components.append(
            {
                "component": label,
                "source": str(Path(path).relative_to(ROOT)),
                "source_sha256": file_digest(path),
                "recorded_resources": dict(resources),
                **fields,
            }
        )

    primary = ROOT / "outputs/pusht-preset-20261029-v1/report.json"
    add(
        "pusht-primary-shared-training-preparation-tests-and120-trials",
        primary,
        read(primary)["combined_resources"],
    )
    sensitivity = ROOT / "outputs/pusht-budget150-20261030-v1/report.json"
    add(
        "pusht-fixed-setting-budget-sensitivity24-trials",
        sensitivity,
        read(sensitivity)["resources"],
    )
    for name in ("tworoom", "reacher", "cube"):
        p = BASE / (name + "-engineering.json")
        if p.exists():
            add(name + "-native-engineering", p, read(p)["resources"])
    for name in ("tworoom", "reacher", "cube"):
        p = BASE / name / "motion/resources.json"
        add(name + "-motion-data", p, read(p))
    repair = BASE / "cube-v2/motion/repair-resources.json"
    add("cube-fresh-calibration-only", repair, read(repair))
    for name in ("tworoom", "reacher", "cube", "cube-v2", "cube-v3", "cube-v4"):
        p = BASE / name / "preparation-resources.json"
        if p.exists():
            add(name + "-preparation", p, read(p))
    incomplete = []
    trial_counts = {}
    for name in (
        "tworoom",
        "reacher",
        "cube-v2",
        "cube-v3",
        "cube-v4",
        "cube-renderer-probes-20261030",
    ):
        folder = BASE / name
        results = [p for p in folder.glob("**/result.json") if "/source/" not in str(p)]
        trial_counts[name] = len(results)
        for p in sorted(results):
            r = read(p)
            add(
                name + "-episode",
                p,
                r["resources"],
                scored=name in ("tworoom", "reacher", "cube-v4"),
                episode_error=bool(r["error"]),
                condition=r["condition"],
                split=r["split"],
            )
        for p in sorted(folder.glob("**/started.json")):
            if "/source/" in str(p) or (p.parent / "result.json").exists():
                continue
            records = []
            journal = p.parent / "interactions.jsonl"
            if journal.exists():
                for line in journal.read_text().splitlines():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            last = records[-1] if records else {"status": "no_journal", "resources": {}}
            counters = last["resources"]
            add(name + "-incomplete-episode", p, counters, scored=False)
            config = read(p)["config"]["params"]
            h = config["horizon"]
            r = config["replan_blocks"]
            n = int(300 * r / h)
            bound = collections.Counter(
                {
                    "environment_actions": 1,
                    "executed_physics_steps": 25,
                    "model_batch_calls": 31,
                    "model_candidate_rollouts": 30 * n + 1,
                    "predictor_steps": 30 * n * h + h,
                    "encoder_frames": 2,
                    "learned_candidate_checks": 30 * n + 2,
                    "learned_transition_rows": (30 * n + 2) * h * 5,
                    "render_frames": 2,
                }
            )
            if last["status"] == "reset_started":
                bound.update({"reset_physics_steps": 50, "reset_control_actions": 2})
            extra.update(bound)
            incomplete.append(
                {
                    "folder": str(p.parent.relative_to(ROOT)),
                    "last_journal_status": last["status"],
                    "last_recorded_resources": counters,
                    "unrecorded_work_upper_bound": dict(bound),
                    "interpretation": "Journal snapshots bound completed work. An interrupted call may add at most one physical action and one CEM replan beyond its last snapshot; bounds are not claimed as measured work.",
                }
            )
    failed = BASE / "cube/failed-starts/accounting-v1.json"
    first = read(failed)["conservative_charged_upper_bound"]
    extra.update(first)
    unknown = {
        "source": str(failed.relative_to(ROOT)),
        "source_sha256": file_digest(failed),
        "upper_bound": first,
        "interpretation": "First native-import failure has no recovered interaction journal. Preserve its conservative full training-split bound; do not report it as exact or as zero.",
        "bound_basis": "No train.npz was written. The frozen producer saves that array after the first 80 groups, before calibration/test collection. Thus this failed start cannot have passed 320 resets and 8000 explicit actions.",
    }
    current = {}
    complete = True
    for name in ("tworoom", "reacher", "cube-v4"):
        out = BASE / name
        report = out / "report.json"
        audit = out / "independent-audit.json"
        ready = report.exists() and audit.exists() and read(audit)["verified"]
        current[name] = {"report_exists": report.exists(), "verified": ready}
        if ready:
            data = read(report)
            current[name].update(
                selected=data["selection"]["selected"],
                evaluation=data["evaluation"],
                report_sha256=file_digest(report),
                audit_sha256=file_digest(audit),
            )
        complete = complete and ready
    upper = lower + extra
    result = {
        "recorded_epoch": time.time(),
        "all_three_extension_components_verified": complete,
        "scope": "Current author-preset PushT goal and conditional three-environment extension only. Prior tuning pilots remain separately recorded. Reused Cube model resources are counted at their original training and fresh-calibration components, never again at consumers.",
        "studies": current,
        "completed_episode_counts": trial_counts,
        "recorded_resource_totals": dict(lower),
        "additional_unmeasured_work_upper_bound": dict(extra),
        "conservative_resource_upper_totals": dict(upper),
        "incomplete_episodes": incomplete,
        "first_failed_start": unknown,
        "components": components,
        "simulator_action_accounting": {
            "recorded_explicit_environment_actions": lower["environment_actions"],
            "recorded_native_reset_control_actions": lower["reset_control_actions"],
            "recorded_total_control_calls": lower["environment_actions"]
            + lower["reset_control_actions"],
            "conservative_total_control_call_upper_bound": upper["environment_actions"]
            + upper["reset_control_actions"],
            "recorded_physics_integrations": lower["executed_physics_steps"]
            + lower["reset_physics_steps"],
            "conservative_physics_integration_upper_bound": upper["executed_physics_steps"]
            + upper["reset_physics_steps"],
            "recorded_discrete_navigation_steps": lower["executed_discrete_steps"],
        },
        "limits": "Incomplete calls are reported as intervals, not invented exact counts. Model queries and elapsed stage times are separate from simulator calls. Posthoc audits and media add no environment interactions. Any running episode is an interim snapshot, not a terminal failure.",
    }
    write_json(BASE / "campaign-accounting.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "all_three_extension_components_verified",
                    "completed_episode_counts",
                    "simulator_action_accounting",
                )
            }
        ),
        flush=True,
    )
    if require_complete and (
        not complete or any("cube-v4/" in row["folder"] for row in incomplete)
    ):
        raise RuntimeError("Campaign evidence is not complete")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    collect(parser.parse_args().require_complete)


if __name__ == "__main__":
    main()
