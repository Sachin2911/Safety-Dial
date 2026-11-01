"""Frozen old/new NAV persistence parity, with four charged controller episodes."""

import argparse
import collections
from dataclasses import fields
import datetime as dt
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.lewm_campaign_training import EpisodeOutcome, JobSpec
from helpers.lewm_progress_journal import BACKEND, replay_progress_journal
import run_lewm_navigation_campaign as old
import run_lewm_navigation_campaign_v2 as new


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def compare_pair(output, legacy_id, journal_id):
    """Inspect saved evidence only; exact arrays/choices/counters are required."""
    left, right = [output / "jobs" / name / "episode" for name in (legacy_id, journal_id)]
    rows = [json.loads((folder / "result.json").read_text()) for folder in (left, right)]
    require(not any(row["error"] for row in rows), "A technical episode cannot qualify parity")
    require(not any(row["frame_flags"] for row in rows), "Input-image review required")
    with (
        np.load(left / "trace.npz", allow_pickle=False) as a,
        np.load(right / "trace.npz", allow_pickle=False) as b,
    ):
        require(set(a.files) == set(b.files), "Trace schema differs")
        arrays = {}
        for name in a.files:
            arrays[name] = bool(a[name].dtype == b[name].dtype and np.array_equal(a[name], b[name]))
            require(arrays[name], "Exact trace parity failed: " + name)
    histories = [
        json.loads((folder / "selector-history.json").read_text()) for folder in (left, right)
    ]
    require(histories[0] == histories[1], "Exact target/features/provenance history differs")
    exclude = {"retrieval_microseconds"}
    for key, value in rows[0]["resources"].items():
        if key not in exclude:
            require(rows[1]["resources"].get(key, 0) == value, "Original counter differs: " + key)
    for key in (
        "safe_completion",
        "success",
        "violation",
        "hazard_violation",
        "unfinished",
        "engaged",
        "progress",
        "environment_actions",
        "recorded_successful_actions",
        "minimum_clearance",
        "initial_original_goal_distance",
        "final_original_goal_distance",
        "controller_failure",
    ):
        require(rows[0][key] == rows[1][key], "Original task outcome differs: " + key)
    resources = rows[1]["resources"]
    require(
        resources.get("actual_safety_candidate_rollouts_attempted", 0)
        == resources.get("actual_safety_candidate_rollouts_successful", 0)
        == resources.get("learned_candidate_checks", 0),
        "Safety-head attempted/successful candidates do not reconstruct original successful work",
    )
    require(
        resources.get("actual_safety_transition_rows_attempted", 0)
        == resources.get("actual_safety_transition_rows_successful", 0)
        == resources.get("learned_transition_rows", 0),
        "Safety-head attempted/successful rows do not reconstruct original successful work",
    )
    replay = replay_progress_journal(right.parent / "progress-journal.jsonl")
    require(replay["last"]["event"] == "final", "New journal is not finalized")
    for key, value in replay["last"]["resources"].items():
        require(resources.get(key, 0) == value, "Journal and saved original outcome differ: " + key)
    return {
        "legacy_job": legacy_id,
        "journal_job": journal_id,
        "exact_arrays": arrays,
        "exact_selector_history": True,
        "selector_decisions": len(histories[0]),
        "original_count_counters_exact": True,
        "excluded_timing_counter": sorted(exclude),
        "legacy_episode_seconds": rows[0]["episode_seconds"],
        "journal_episode_seconds": rows[1]["episode_seconds"],
        "legacy_planning_seconds": rows[0]["planning_seconds"],
        "journal_planning_seconds": rows[1]["planning_seconds"],
        "environment_actions_each": rows[0]["environment_actions"],
        "journal_records": replay["records"],
        "journal_sha256": replay["sha256"],
    }


def compare(protocol, output):
    pairs = [compare_pair(output, *pair) for pair in protocol["pairs"]]
    return {
        "qualified": True,
        "pairs": pairs,
        "new_nn_calls_in_comparison": 0,
        "new_simulator_actions_in_comparison": 0,
        "interpretation": "Persistence and head accounting parity only, no controller efficacy claim",
    }


def run(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    require(protocol["schema"] == "lewm-navigation-journal-parity-v1", "Unknown parity protocol")
    design = protocol["evaluation"]
    require(os.environ.get("RUNPOD_POD_ID") == design["execution_pod_id"], "Wrong rented pod")
    require(design["controller_backend"] == "observed-target-v1", "Architecture must remain v1")
    require(design["progress_backend"] == BACKEND, "New journal backend is not frozen")
    require(not design.get("goal_readout"), "No readout may enter persistence qualification")
    require(
        output.resolve().is_relative_to(ROOT), "All actual progress journals belong on project NFS"
    )
    require(not output.exists(), "Use a new run namespace; no episode replay")
    require(len(protocol["jobs"]) == protocol["maximum_episode_starts"] == 4, "Exactly four jobs")
    require(
        {name: old.digest(ROOT / name) for name in protocol["source_files"]}
        == protocol["source_sha256"],
        "Prospectively frozen source/test/comparator changed",
    )
    new.verify_case_preparation(protocol)
    require(
        dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=15)
        < dt.datetime.fromisoformat(design["deadline_utc"]),
        "Campaign preservation reserve reached",
    )
    output.mkdir(parents=True)
    (output / "jobs").mkdir()
    old.write_json(
        output / "source-freeze.json",
        {
            "protocol": protocol,
            "protocol_sha256": old.digest(protocol_path),
            "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "pod_id": os.environ["RUNPOD_POD_ID"],
        },
    )
    state = {
        "phase": "running",
        "active": None,
        "jobs": {},
        "environment_actions": 0,
        "resources": {},
        "episode_starts": 0,
    }
    old.write_json(output / "state.json", state)
    runtime = new.load_navigation_runtime(design)
    evaluators = {
        "legacy": old.NavigationEvaluator(design, output, runtime),
        "journal": new.NavigationEvaluator(design, output, runtime),
    }
    try:
        for planned in protocol["jobs"]:
            require(
                state["environment_actions"] + planned["max_actions"]
                <= protocol["maximum_environment_actions"],
                "Hard action allowance cannot fund another complete episode",
            )
            folder = output / "jobs" / planned["job_id"]
            folder.mkdir()
            old.write_json(folder / "started.json", planned)
            state["episode_starts"] += 1
            state["active"] = planned["job_id"]
            old.write_json(output / "state.json", state)
            spec = JobSpec(
                job_id=planned["job_id"],
                directory=folder,
                case_id=planned["case"]["case_id"],
                case=planned["case"],
                environment_seed=planned["environment_seed"],
                policy_seed=planned["policy_seed"],
                parameters=tuple(planned["parameters"]),
                mode="deterministic",
                max_actions=planned["max_actions"],
                temperature=1.0,
                method=planned["method"],
                repeat_id=0,
                stage="persistence_parity",
                generation=-1,
                candidate_index=-1,
            )

            def progress(actions, resources=None):
                old.write_json(
                    folder / "live.json",
                    {
                        "environment_actions": actions,
                        "resources": resources or {},
                    },
                )

            outcome = evaluators[planned["backend"]](spec, progress)
            row = {
                f.name: getattr(outcome, f.name)
                for f in fields(EpisodeOutcome)
                if f.name != "decisions"
            }
            old.write_json(folder / "result.json", row)
            state["jobs"][spec.job_id] = row
            state["environment_actions"] += outcome.environment_actions
            totals = collections.Counter(state["resources"])
            totals.update(outcome.resources)
            state["resources"] = dict(totals)
            state["active"] = None
            old.write_json(output / "state.json", state)
            require(not outcome.error, "Technical episode retained; parity qualification stopped")
            require(not outcome.extra["frame_flags"], "Input-image review required")
        qualification = compare(protocol, output)
        old.write_json(output / "qualification.json", qualification)
        state.update(phase="qualified", completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        old.write_json(output / "state.json", state)
        return qualification
    except BaseException:
        state["phase"] = "failed_or_mismatched"
        if state["active"]:
            folder = output / "jobs" / state["active"]
            if (folder / "live.json").exists():
                state["active_live_counters"] = json.loads((folder / "live.json").read_text())
            if (folder / "progress-journal.jsonl").exists():
                state["active_journal_prefix"] = replay_progress_journal(
                    folder / "progress-journal.jsonl"
                )
            active = state.get("active_journal_prefix", {}).get("last")
            if active is None:
                active = state.get("active_live_counters", {})
            observed = int(active.get("environment_actions", 0))
            state["observed_environment_actions_including_active"] = (
                state["environment_actions"] + observed
            )
            state["conservative_action_charge_including_active"] = state[
                "environment_actions"
            ] + max(observed, spec.max_actions)
            totals = collections.Counter(state["resources"])
            totals.update(active.get("resources", {}))
            state["resources_including_active_prefix"] = dict(totals)
            state["active_job_is_not_adopted_or_replayed"] = True
        state["failure"] = traceback.format_exc()
        old.write_json(output / "state.json", state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    output = args.out if args.out else ROOT / protocol["output_namespace"]
    if args.compare_only:
        result = compare(protocol, output)
        old.write_json(output / "qualification.json", result)
    else:
        result = run(args.protocol, output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
