"""Complete explicitly amended engineering after preserving the failed combined check."""

import argparse
import json
from pathlib import Path
import time

from run_dsrl06 import (
    AdmissionClosed,
    Coordinator,
    PendingOperation,
    atomic_json,
    sha256,
    timestamp,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repair-receipt", type=Path, required=True)
    parser.add_argument("--rental", type=Path, default=Path("progress/rental-dsrl06-20261025.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    queue = Coordinator(
        root, args.plan, args.rental, "engineering", repair_receipt=args.repair_receipt
    )
    stop = False
    try:
        amendment = queue.plan["amendment"]["measurement_association"]
        if amendment.get("original_profile_success") is not False:
            raise ValueError("The original failed combined profile must remain failed")
        original = queue.processes.spec("http.engineering-profile")
        original_exit = queue.processes.status("http.engineering-profile")
        begin, end = timestamp(original["created_utc"]), timestamp(original_exit["finished_utc"])
        oldest_exit = min(
            timestamp(queue.state["episodes"][j]["finished_utc"]) for j in ("E0", "E1", "E2")
        )
        overlap = max(0.0, min(end, oldest_exit) - begin)
        topology = end <= oldest_exit and len(queue.state["engineering_overlap"]) == 3
        calls = [
            json.loads(line)
            for line in (queue.learner_dir / "learner-ledger.jsonl").read_text().splitlines()
        ]
        calls = [
            row
            for row in calls
            if row["event"] == "update_reserved" and begin <= timestamp(row["utc"]) <= end
        ]
        if [row.get("label") for row in calls] != ["author_reference", "bridge_parity"]:
            raise ValueError("Original coexistence evidence lacks the two charged source calls")
        if "v4_restore_old_process" not in queue.state:
            queue.state["v4_restore_old_process"] = queue.state.get("learner_process", "learner")
            queue.save()
        queue.http("shutdown-pre-v4", "/shutdown", {})
        queue.wait(queue.state["v4_restore_old_process"])
        queue.learner("engineering")
        request = json.loads((queue.output / "http/engineering-profile.request.json").read_text())[
            "body"
        ]
        request.update(operation_id="engineering-profile-repair-01", updates=128, parity=True)
        profile = queue.http("engineering-profile-repair-01", "/profile", request)
        atomic_json(queue.output / "engineering-profile-repaired.json", profile)
        checkpoint = queue.checkpoint("engineering-profile-repaired")
        parity_path = queue.output / "engineering-cpu-parity.json"
        cpu = (
            json.loads(parity_path.read_text())
            if parity_path.exists()
            else queue.cpu_actor_parity()
        )
        try:
            authority = queue.authority()
        except ValueError as error:
            authority = {"gate": {"valid": False, "admit": False, "reason": str(error)}}
            atomic_json(queue.output / "authority-result.json", authority)
        authority_admit = bool(
            authority.get("gate", {}).get("valid") and authority["gate"].get("admit")
        )
        source_valid = profile["source_parity"]["maximum_state_difference"] <= 1e-6
        calibrated = profile["timed_updates"] == 128 and profile["median_update_seconds"] > 0
        if authority_admit and source_valid and calibrated and cpu.get("valid") is True:
            for job in queue.plan["jobs"]["engineering_jobs"][4:]:
                queue.start_episode(job, actor_version=0)
            queue.wait_episodes(["E4", "E5"])
        durations = [
            timestamp(e["finished_utc"]) - timestamp(e["started_utc"])
            for key, e in queue.state["episodes"].items()
            if key in ("E4", "E5") and e.get("valid")
        ]
        cached = max(durations) if durations else None
        projected = (
            (
                20 * cached
                + 90000 * profile["median_update_seconds"]
                + 3000 * cpu["median_cpu_seconds"]
                + 1500
            )
            if cached
            else None
        )
        remaining = timestamp(queue.rental["allocation_deadline_utc"]) - time.time()
        admit = bool(
            authority_admit
            and topology
            and source_valid
            and calibrated
            and cpu.get("valid") is True
            and projected is not None
            and projected <= remaining
        )
        summary = {
            "schema": "dsrl06-stage-a-summary-v1",
            "plan_sha256": sha256(args.plan),
            "instance_id": queue.rental["instance_id"],
            "admit": admit,
            "reason": "ready_for_review" if admit else "engineering_did_not_admit_learning",
            "authority_admit": authority_admit,
            "authority": authority.get("gate"),
            "original_profile_success": False,
            "original_profile_http_exit": original_exit,
            "topology_three_rollouts": topology,
            "topology_gradient_calls_observed": len(calls),
            "three_rollout_profile_overlap_seconds": overlap,
            "measurement_association_amendment": amendment,
            "profile": profile,
            "source_parity_valid": source_valid,
            "engineering_checkpoint": checkpoint,
            "cpu_actor_parity": cpu,
            "resources": queue.sync_counters(),
            "cached_episode_seconds": cached,
            "projected_stage_b_seconds_including_preservation": projected,
            "remaining_allocation_seconds": remaining,
        }
        atomic_json(queue.output / "stage-a-summary.json", summary)
        print(json.dumps(summary), flush=True)
        stop = not admit
    except PendingOperation:
        raise
    except AdmissionClosed:
        stop = True
        raise
    finally:
        queue.close(stop=stop)


if __name__ == "__main__":
    main()
