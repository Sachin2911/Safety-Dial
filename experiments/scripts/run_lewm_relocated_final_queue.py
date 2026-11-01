"""Local serial orchestration of exactly ten frozen, explicitly authorized profiles."""

import argparse
import datetime as dt
import hashlib
import inspect
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "experiments/scripts/launch_lewm_campaign_relocated_final.py"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def remote_snapshot(profile, worker, authority_ref):
    """Inspect saved job/identity evidence on the rented host, never open arrays."""
    import collections
    import hashlib
    import json
    from pathlib import Path

    root = Path("/workspace/Safety-Dial")

    def load(path):
        return json.loads(path.read_text())

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    proc = Path("/proc") / str(worker["pid"]) / "stat"
    live = False
    try:
        process_stat = proc.read_text()
    except FileNotFoundError:
        process_stat = None
    if process_stat is not None:
        fields = process_stat.rsplit(")", 1)[1].split()
        live = fields[0] != "Z" and fields[19] == str(worker["start_ticks"])
    output = root / profile["output_namespace"]
    state_path = output / "state.json"
    state = load(state_path) if state_path.exists() else {}
    result = {
        "worker_live": live,
        "phase": state.get("phase", "not_created"),
        "completed_jobs": len(state.get("jobs", {})),
        "active": state.get("active"),
    }
    if live:
        print(json.dumps(result))
        return
    if state.get("phase") != "complete" or state.get("active") is not None:
        raise RuntimeError("Worker retired without a complete, inactive frozen profile")
    protocol_path = root / profile["protocol"]["path"]
    if sha(protocol_path) != profile["protocol"]["sha256"]:
        raise RuntimeError("Original protocol changed after launch")
    protocol = load(protocol_path)
    jobs = {row["job_id"]: row for row in protocol["jobs"]}
    if len(jobs) != len(protocol["jobs"]) or set(jobs) != set(state["jobs"]):
        raise RuntimeError("Completed profile differs from its entire frozen job schedule")
    if {p.parent.name for p in (output / "jobs").glob("*/started.json")} != set(jobs):
        raise RuntimeError("A started job was dropped or added")
    freeze_path = output / "source-freeze.json"
    freeze = load(freeze_path)
    if (
        freeze["protocol_sha256"] != profile["protocol"]["sha256"]
        or freeze["sources"] != protocol["source_sha256"]
        or freeze["pod_id"] != profile["pod_id"]
    ):
        raise RuntimeError("Actual execution/source/protocol freeze differs")
    continuation = freeze["continuation"]
    continuation_path = root / continuation["path"]
    if sha(continuation_path) != continuation["sha256"]:
        raise RuntimeError("Continuation sidecar changed")
    record = load(continuation_path)
    authority_path = root / authority_ref["path"]
    if sha(authority_path) != authority_ref["sha256"]:
        raise RuntimeError("Bound continuation authority changed")
    authority = load(authority_path)
    expected_override = {
        "evaluation.deadline_utc": {
            "original": protocol["evaluation"]["deadline_utc"],
            "effective": authority["new_deadline_utc"],
        },
        "evaluation.execution_pod_id": {
            "original": profile["original_pod_id"],
            "effective": profile["pod_id"],
        },
    }
    if (
        record["authority"] != authority_ref
        or record["original_protocol"] != profile["protocol"]
        or record["original_entrypoint"] != profile["entrypoint"]
        or record["pod_id"] != profile["pod_id"]
        or not record["all_other_protocol_fields_identical"]
        or record["effective_override"] != expected_override
        or record["actual_final_host"] != authority["actual_final_host"]
        or record["original_logical_host"] != protocol["host"]
        or record["new_training_or_selection"] is not False
    ):
        raise RuntimeError("Actual continuation lineage differs")
    totals = collections.Counter()
    arms = collections.defaultdict(collections.Counter)
    for key, planned in jobs.items():
        folder = output / "jobs" / key
        row = load(folder / "result.json")
        if load(folder / "started.json") != planned or state["jobs"][key] != row:
            raise RuntimeError("Started/result/state row differs from the frozen job")
        if (
            row["parameters"] != planned["parameters"]
            or row["environment_seed"] != planned["environment_seed"]
            or row["policy_seed"] != planned["policy_seed"]
            or row["method"] != planned["method"]
            or row.get("error")
            or row["extra"]["frame_flags"]
            or row["environment_actions"] > planned["max_actions"]
        ):
            raise RuntimeError(
                "Completed job changed inputs, exceeded cap, or retained a technical error"
            )
        inner = Path(row["extra"]["episode_result"])
        inner = inner if inner.is_absolute() else root / inner
        if not inner.resolve().is_relative_to(output.resolve()):
            raise RuntimeError("Episode reference escaped its immutable stage")
        if sha(inner) != row["extra"]["episode_result_sha256"]:
            raise RuntimeError("Saved inner episode hash differs")
        totals.update(row["resources"])
        arm = arms[planned["arm"]]
        arm["episodes"] += 1
        arm["environment_actions"] += row["environment_actions"]
        for metric in ("safe_completion", "violation", "unfinished", "engaged"):
            arm[metric] += int(row[metric])
    if dict(totals) != state["resource_totals"]:
        raise RuntimeError("Saved cumulative resource ledger differs")
    if totals["environment_actions"] > protocol["maximum_environment_actions"]:
        raise RuntimeError("Frozen profile action allowance exceeded")
    if totals["environment_resets"] > protocol["maximum_native_resets"]:
        raise RuntimeError("Frozen profile reset allowance exceeded")
    result.update(
        verified_full_frozen_job_schedule=True,
        state_sha256=sha(state_path),
        source_freeze_sha256=sha(freeze_path),
        per_arm=dict(arms),
        environment_actions=totals["environment_actions"],
        new_model_simulator_calls_by_monitor=0,
    )
    print(json.dumps(result))


def prepare(args):
    authority_path = args.authority.resolve()
    require(authority_path.is_relative_to(ROOT), "Use canonical-project authority metadata")
    require(digest(authority_path) == args.authority_sha256, "Authority SHA differs")
    authority = json.loads(authority_path.read_text())
    require(
        authority["schema"] == "lewm-frozen-final-relocation-authority-v1"
        and authority["execution_enabled"] is True
        and len(authority["profiles"]) == 10,
        "Exact active all-ten-profile authority required",
    )
    launcher_name = str(LAUNCHER.relative_to(ROOT))
    require(authority["source_sha256"][launcher_name] == digest(LAUNCHER), "Launcher differs")
    connection = json.loads(args.connection_json.read_text())
    require(
        connection["pod_id"] == authority["actual_final_host"]["pod_id"]
        and connection["provider_verified"] is True,
        "Provider-verified replacement connection required",
    )
    plan = {
        "schema": "lewm-ten-profile-serial-queue-v1",
        "authority": {
            "path": str(authority_path.relative_to(ROOT)),
            "sha256": args.authority_sha256,
        },
        "connection_json": str(args.connection_json.resolve()),
        "connection_sha256": digest(args.connection_json),
        "known_hosts": str(args.known_hosts.resolve()),
        "known_hosts_sha256": digest(args.known_hosts),
        "launcher": {"path": launcher_name, "sha256": digest(LAUNCHER)},
        "orchestrator_sha256": digest(__file__),
        "gate": args.gate,
        "tests": args.tests,
        "name_prefix": args.name_prefix,
        "profiles_in_original_authority_order": authority["profiles"],
        "no_new_training_or_profile_selection": True,
        "new_model_simulator_calls_by_orchestrator": 0,
    }
    write_new(args.out, plan)
    return {"plan": str(args.out), "sha256": digest(args.out), "profiles": 10}


def run(args):
    require(args.root_go, "Explicit root GO after V4 binding and new runtime gate is required")
    require(digest(args.plan) == args.plan_sha256, "Reviewed queue plan differs")
    plan = json.loads(args.plan.read_text())
    require(plan["orchestrator_sha256"] == digest(__file__), "Queue source changed")
    require(digest(LAUNCHER) == plan["launcher"]["sha256"], "Launcher changed")
    require(not args.out.exists(), "Never resume or overwrite a partial queue execution")
    args.out.mkdir(parents=True)
    write_new(args.out / "plan.json", plan)
    events = args.out / "events.jsonl"
    previous = None

    def event(kind, **fields):
        nonlocal previous
        value = {
            "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "event": kind,
            "previous_digest": previous,
            **fields,
        }
        previous = hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        value["record_digest"] = previous
        with events.open("a") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(value), flush=True)

    def remote(profile, worker):
        metadata = (profile, worker, plan["authority"])
        code = inspect.getsource(remote_snapshot) + "\nremote_snapshot(*" + repr(metadata) + ")\n"
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "experiments/scripts/lewm_completion_replacement_ssh.py"),
                "exec",
                "python3 -c " + shlex.quote(code),
            ],
            capture_output=True,
            text=True,
            timeout=45,
        )
        if result.returncode:
            raise RuntimeError("Remote saved-state/identity check failed: " + result.stderr[-1500:])
        return json.loads(result.stdout.strip().splitlines()[-1])

    event("queue_started", profiles=10, authority=plan["authority"])
    try:
        for index, profile in enumerate(plan["profiles_in_original_authority_order"]):
            require(
                digest(ROOT / plan["authority"]["path"]) == plan["authority"]["sha256"],
                "Authority changed during serial execution",
            )
            require(
                digest(plan["connection_json"]) == plan["connection_sha256"],
                "Verified connection changed",
            )
            require(digest(plan["known_hosts"]) == plan["known_hosts_sha256"], "Host keys changed")
            name = plan["name_prefix"] + f"-{index:02d}-" + Path(profile["protocol"]["path"]).stem
            command = [
                sys.executable,
                str(LAUNCHER),
                "--connection-json",
                plan["connection_json"],
                "--known-hosts",
                plan["known_hosts"],
                "--name",
                name,
                "--gate",
                plan["gate"],
                "--tests",
                str(plan["tests"]),
                "--authority",
                plan["authority"]["path"],
                "--authority-sha256",
                plan["authority"]["sha256"],
                "--protocol",
                profile["protocol"]["path"],
                "--protocol-sha256",
                profile["protocol"]["sha256"],
                "--entrypoint",
                profile["entrypoint"]["path"],
                "--out",
                profile["output_namespace"],
            ]
            event("launch_requested", index=index, name=name, profile=profile)
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
            (args.out / f"launch-{index:02d}.stdout").write_text(result.stdout)
            (args.out / f"launch-{index:02d}.stderr").write_text(result.stderr)
            require(
                result.returncode == 0, "Frozen launcher rejected profile; inspect retained output"
            )
            worker = json.loads(result.stdout.strip().splitlines()[-1])
            event("worker_launched", index=index, worker=worker)
            last = None
            while True:
                snapshot = remote(profile, worker)
                marker = (snapshot["phase"], snapshot["completed_jobs"])
                if marker != last:
                    event("progress", index=index, snapshot=snapshot)
                    last = marker
                if not snapshot["worker_live"]:
                    require(snapshot.get("verified_full_frozen_job_schedule"), "Incomplete profile")
                    event("profile_completed_and_worker_retired", index=index, snapshot=snapshot)
                    break
                time.sleep(5)
        event("queue_complete", profiles=10)
    except BaseException as exc:
        event("queue_stopped_on_error", error_type=type(exc).__name__, error=str(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    planner = commands.add_parser("prepare")
    planner.add_argument("--authority", type=Path, required=True)
    planner.add_argument("--authority-sha256", required=True)
    planner.add_argument("--connection-json", type=Path, required=True)
    planner.add_argument("--known-hosts", type=Path, required=True)
    planner.add_argument("--gate", required=True)
    planner.add_argument("--tests", type=int, required=True)
    planner.add_argument("--name-prefix", required=True)
    planner.add_argument("--out", type=Path, required=True)
    executor = commands.add_parser("run")
    executor.add_argument("--plan", type=Path, required=True)
    executor.add_argument("--plan-sha256", required=True)
    executor.add_argument("--out", type=Path, required=True)
    executor.add_argument("--root-go", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args)))
    else:
        run(args)


if __name__ == "__main__":
    main()
