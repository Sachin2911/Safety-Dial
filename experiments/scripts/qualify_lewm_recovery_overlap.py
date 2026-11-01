"""Compare one frozen recovery overlap episode with its first recorded strong control.

Metadata uses the standard library. NumPy is imported only after the exact rented
pod guard. No simulator, model, training or final-case evidence is accessed.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
POD = "nj2fqkp2prgk8x"
REFERENCE_PROTOCOL = (
    "configs/lewm_campaign_20261031/navigation-budget30-relocated-v1/training-controls-r2.json"
)
REFERENCE_SHA = "ee96d8fd6071987a2acd486bd47d694a6259d60c2408348cb6562dd33a744009"
REFERENCE_STAGE = "results/navigation-budget30-relocated-v1/r2/training-controls"
REFERENCE_JOB = "strong_tworoom-01000-none_s355000"
ENGINEERING_STAGES = {
    "results/navigation-recovery-overlap-qualification-v1",
    "results/navigation-recovery-overlap-qualification-v2",
}
TIMING_RESULT_FIELDS = {"episode_seconds", "planning_seconds"}
TIMING_COUNTERS = {"retrieval_microseconds"}
JOURNAL_ARTIFACT_FIELDS = {"path", "sha256", "last_record_digest"}
TARGET_FIELDS = ("query", "target_kind", "memory_index", "memory_sha256", "source_provenance")


def require(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolve(root, name):
    path = (root / name).resolve()
    require(path.is_relative_to(root), "Evidence path escaped the project root")
    return path


def reference(root, path):
    return {"path": str(path.relative_to(root)), "sha256": digest(path)}


def peer_proof(root, config, protocol_path, engineering_stage):
    path = resolve(root, config["peer_observations_path"])
    value = load(path)
    require(
        value["schema"] == "lewm-recovery-overlap-peer-observations-v1"
        and value["pod_id"] == POD
        and value["returncode"] == 0
        and value["all_peer_observations_live"] is True
        and value["new_model_simulator_calls_by_monitor"] == 0,
        "Successful monitored overlap on the exact recovery host is required",
    )
    expected = config["expected_peer_identity"]
    authorization_meta = value["authorization"]
    authorization_path = resolve(root, authorization_meta["authorization"])
    require(
        digest(authorization_path) == authorization_meta["authorization_sha256"],
        "Recorded overlap authorization changed",
    )
    authorization = load(authorization_path)
    stage = authorization["stage"]
    require(
        authorization["schema"] == "lewm-recovery-two-worker-authorization-v1"
        and authorization["purpose"] == "engineering_qualification"
        and authorization["execution_pod_id"] == POD
        and authorization["max_concurrent_experiment_workers"] == 2
        and authorization["scientific_settings_unchanged"] is True
        and authorization["all_actual_work_charged"] is True
        and authorization["peer"]["pid"] == expected["pid"]
        and str(authorization["peer"]["start_ticks"]) == str(expected["start_ticks"])
        and stage["protocol"] == str(protocol_path.relative_to(root))
        and stage["protocol_sha256"] == digest(protocol_path)
        and stage["out"] == engineering_stage
        and stage["entrypoint"] == "experiments/scripts/run_lewm_navigation_campaign_v2.py"
        and stage["entry_args"] == [],
        "The monitored command must match its exact prospective engineering authorization",
    )
    launcher = "experiments/scripts/launch_lewm_campaign_frozen_overlap.py"
    require(
        authorization_meta["launcher_sha256"]
        == authorization["launcher_sha256"]
        == config["engineering_source_sha256"][launcher]
        and value["scientific_command"]
        == [
            "python", "-u", stage["entrypoint"], "--protocol", stage["protocol"],
            "--out", stage["out"],
        ],
        "Peer monitor source or scientific command differs from the freeze",
    )
    require(
        isinstance(expected["pid"], int)
        and expected["pid"] > 0
        and str(expected["start_ticks"]).isdigit(),
        "Exact peer PID and start ticks are required",
    )
    samples = value["samples"]
    require(samples, "At least one peer observation during the scientific child is required")
    observations = [value["before"], *samples, value["after"]]
    times = []
    for item in observations:
        actual, declared = item["observed_identity"], item["expected_identity"]
        require(
            item["live"] is True
            and declared["pid"] == expected["pid"]
            and str(declared["start_ticks"]) == str(expected["start_ticks"])
            and actual["pid"] == expected["pid"]
            and str(actual["start_ticks"]) == str(expected["start_ticks"])
            and actual["state"] not in ("Z", "X"),
            "The exact authorized peer must remain live for every overlap observation",
        )
        times.append(float(item["monotonic_seconds"]))
    require(all(b > a for a, b in zip(times, times[1:])), "Peer observations are not ordered")
    child = value["child_identity"]
    require(
        child["pid"] > 0
        and child["pid"] != expected["pid"]
        and str(child["start_ticks"]).isdigit(),
        "Distinct scientific child process identity is required",
    )
    return {
        "sidecar": reference(root, path),
        "authorization": reference(root, authorization_path),
        "peer_identity": expected,
        "child_identity": child,
        "observations": len(observations),
        "monitored_seconds": times[-1] - times[0],
        "interval_scope": (
            "The exact peer was observed before, during and after the scientific subprocess. "
            "V2 records episode durations rather than UTC episode endpoints, so no absent "
            "episode timestamp is inferred."
        ),
    }


def deterministic_result(row):
    result = {k: v for k, v in row.items() if k not in TIMING_RESULT_FIELDS | {"resources"}}
    journal = result["progress_journal"]
    require(JOURNAL_ARTIFACT_FIELDS <= set(journal), "Journal artifact identities are missing")
    result["progress_journal"] = {
        key: value for key, value in journal.items() if key not in JOURNAL_ARTIFACT_FIELDS
    }
    return result


def qualify(root, protocol_path):
    require(os.environ.get("RUNPOD_POD_ID") == POD, "Only the exact recovery rented pod may compare")
    protocol = load(protocol_path)
    config = protocol["overlap_qualification"]
    for name, expected in config["engineering_source_sha256"].items():
        require(digest(resolve(root, name)) == expected, "Frozen engineering source changed: " + name)
    require(
        config["reference_protocol"] == {"path": REFERENCE_PROTOCOL, "sha256": REFERENCE_SHA}
        and config["reference_stage"] == REFERENCE_STAGE
        and config["reference_job_id"] == REFERENCE_JOB,
        "Only the prospectively fixed first strong control is eligible",
    )
    original_path = resolve(root, REFERENCE_PROTOCOL)
    require(digest(original_path) == REFERENCE_SHA, "Original controls protocol changed")
    original = load(original_path)
    first = next(j for j in original["jobs"] if j["method"] == "fixed_geometry_progress")
    require(first["job_id"] == REFERENCE_JOB, "Reference is not the first frozen strong job")
    require(
        protocol["jobs"] == [first]
        and protocol["evaluation"] == original["evaluation"]
        and protocol["case_preparation"] == original["case_preparation"]
        and protocol["source_files"] == original["source_files"]
        and protocol["source_sha256"] == original["source_sha256"]
        and protocol["maximum_episode_starts"] == 1
        and protocol["maximum_environment_actions"] == first["max_actions"] == 30
        and protocol["output_namespace"] in ENGINEERING_STAGES,
        "Engineering qualification must repeat the exact original scientific contract once",
    )
    for name, expected in original["source_sha256"].items():
        require(digest(resolve(root, name)) == expected, "Frozen scientific source changed: " + name)
    old = resolve(root, REFERENCE_STAGE)
    new = resolve(root, protocol["output_namespace"])
    required_files = {
        str((old / name).relative_to(root))
        for name in (
            "source-freeze.json",
            "state.json",
            f"jobs/{REFERENCE_JOB}/result.json",
            f"jobs/{REFERENCE_JOB}/episode/result.json",
            f"jobs/{REFERENCE_JOB}/episode/trace.npz",
            f"jobs/{REFERENCE_JOB}/episode/selector-history.json",
        )
    }
    require(
        required_files <= set(config["reference_artifacts"]),
        "Freeze the complete original source/state/result/trace/selector evidence before rerun",
    )
    for name, expected in config["reference_artifacts"].items():
        require(digest(resolve(root, name)) == expected, "Frozen original evidence changed: " + name)
    for stage, expected_protocol in ((old, REFERENCE_SHA), (new, digest(protocol_path))):
        state, source = load(stage / "state.json"), load(stage / "source-freeze.json")
        require(state["phase"] == "complete" and state["active"] is None, "Stage is incomplete")
        require(
            source["protocol_sha256"] == expected_protocol
            and source["pod_id"] == POD
            and source["sources"] == original["source_sha256"],
            "Scientific source or actual execution host differs",
        )
        folder = stage / "jobs" / REFERENCE_JOB
        require(load(folder / "started.json") == first, "Recorded job settings differ")
        outer = load(folder / "result.json")
        require(outer == state["jobs"][REFERENCE_JOB], "Outer result differs")
        inner = load(folder / "episode/result.json")
        require(
            digest(folder / "episode/result.json") == outer["extra"]["episode_result_sha256"]
            and digest(folder / "episode/selector-history.json")
            == outer["extra"]["decision_reconstruction_sha256"]
            and outer["resources"] == inner["resources"],
            "Recorded outer/inner/selector provenance differs",
        )
        for key in ("safe_completion", "violation", "unfinished", "engaged", "environment_actions"):
            require(outer[key] == inner[key], "Recorded outcome fields differ: " + key)
    require(set(load(new / "state.json")["jobs"]) == {REFERENCE_JOB}, "Exactly one engineering job")
    overlap = peer_proof(root, config, protocol_path, protocol["output_namespace"])
    folders = [s / "jobs" / REFERENCE_JOB / "episode" for s in (old, new)]
    rows = [load(f / "result.json") for f in folders]
    require(
        all(not r["error"] and not r["controller_failure"] and not r["frame_flags"] for r in rows),
        "Technical failure cannot qualify overlap",
    )
    deterministic = [deterministic_result(row) for row in rows]
    resources = [
        {k: v for k, v in row["resources"].items() if k not in TIMING_COUNTERS} for row in rows
    ]
    histories = [load(f / "selector-history.json") for f in folders]
    targets = [[{k: h.get(k) for k in TARGET_FIELDS} for h in history] for history in histories]

    import numpy as np

    arrays = {}
    with np.load(folders[0] / "trace.npz", allow_pickle=False) as a:
        with np.load(folders[1] / "trace.npz", allow_pickle=False) as b:
            same_keys = set(a.files) == set(b.files)
            for key in sorted(set(a.files) | set(b.files)):
                if key not in a.files or key not in b.files:
                    arrays[key] = {"equal": False, "reason": "missing array"}
                    continue
                x, y = a[key], b[key]
                arrays[key] = {
                    "equal": bool(
                        x.dtype == y.dtype and x.shape == y.shape
                        and np.array_equal(x, y) and x.tobytes() == y.tobytes()
                    ),
                    "reference_shape": list(x.shape),
                    "engineering_shape": list(y.shape),
                    "reference_dtype": str(x.dtype),
                    "engineering_dtype": str(y.dtype),
                    "reference_array_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                    "engineering_array_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
                }
    checks = {
        "all_trace_keys_equal": same_keys,
        "all_trace_arrays_exact": all(item["equal"] for item in arrays.values()),
        "all_deterministic_result_fields_exact": deterministic[0] == deterministic[1],
        "all_deterministic_resource_counters_exact": resources[0] == resources[1],
        "selected_target_identity_sequence_exact": targets[0] == targets[1],
    }
    return {
        "schema": "lewm-recovery-overlap-qualification-v1",
        "qualified": all(checks.values()),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pod_id": POD,
        "comparator_source": reference(root, Path(__file__).resolve()),
        "engineering_protocol": reference(root, protocol_path),
        "reference_protocol": reference(root, original_path),
        "reference_job_id": REFERENCE_JOB,
        "reference_artifacts": config["reference_artifacts"],
        "overlap": overlap,
        "checks": checks,
        "trace_arrays": arrays,
        "selected_targets": targets,
        "full_selector_history_equal_descriptive": histories[0] == histories[1],
        "reference_resources": rows[0]["resources"],
        "engineering_resources": rows[1]["resources"],
        "timing_exclusions": {
            "episode_result_fields": sorted(TIMING_RESULT_FIELDS),
            "resource_fields": sorted(TIMING_COUNTERS),
        },
        "artifact_identity_exclusions": {
            "progress_journal_fields": sorted(JOURNAL_ARTIFACT_FIELDS),
            "reference": {key: rows[0]["progress_journal"][key] for key in JOURNAL_ARTIFACT_FIELDS},
            "engineering": {key: rows[1]["progress_journal"][key] for key in JOURNAL_ARTIFACT_FIELDS},
            "all_other_journal_fields_compared_exactly": True,
        },
        "reference_episode_result": reference(root, folders[0] / "result.json"),
        "engineering_episode_result": reference(root, folders[1] / "result.json"),
        "new_model_simulator_calls": 0,
        "scope": "One same-host engineering parity episode only; no efficacy or general speed claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.project_root.resolve()
    require(os.environ.get("RUNPOD_POD_ID") == POD, "Rented recovery pod required")
    protocol = resolve(root, args.protocol)
    out = resolve(root, args.out)
    require(not out.exists(), "Never overwrite a qualification result")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = qualify(root, protocol)
    except BaseException:
        report = {
            "schema": "lewm-recovery-overlap-qualification-v1",
            "qualified": False,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "comparator_source": reference(root, Path(__file__).resolve()),
            "error": traceback.format_exc(),
            "new_model_simulator_calls": 0,
        }
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"path": str(out.relative_to(root)), "sha256": digest(out), **report["checks"]})
          if "checks" in report else json.dumps({"path": str(out), "qualified": False}))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    sys.exit(main())
