"""Task-7 repair, live integration and fresh development queue using tested primitives."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl07_infra import stage_b_projection
from helpers.dsrl07_protocol import CAPS, decoder_comparison, frozen_jobs
from helpers.dsrl07_reallocation import (
    ORIGINAL_LIMITS,
    REALLOCATED_LIMITS,
    verify_reallocation_receipt,
)
from run_dsrl06 import (
    Coordinator as PreviousCoordinator,
    PendingOperation,
    atomic_json,
    sha256,
    timestamp,
    utc,
    validate_amendment as validate_source_amendment,
)


def verify_plan(root, path, stage):
    plan = json.loads(Path(path).read_text())
    if plan.get("authorization", {}).get("user_launch") is not True:
        raise ValueError("Task-7 launch authorization missing")
    for name, digest in plan["source_sha256"].items():
        if sha256(Path(root) / name) != digest:
            raise ValueError(f"Frozen task-7 source changed: {name}")
    if plan["jobs"] != frozen_jobs() or plan["caps"] != dict(CAPS):
        raise ValueError("Task-7 frozen jobs or caps differ")
    if plan["money"]["total_usd"] != 15:
        raise ValueError("Cumulative DSRL cap changed")
    if stage == "pilot" and (plan.get("stage") != "B" or not plan.get("stage_a_summary_sha256")):
        raise ValueError("Development needs a stage-B freeze")
    return plan


def validate_amendment(old, new, receipt, old_hash, new_hash):
    if new.get("amendment", {}).get("type") != "financial_reserve_reallocation":
        return validate_source_amendment(old, new, receipt, old_hash, new_hash)
    verify_reallocation_receipt(receipt, old_hash, new_hash)
    if (
        new.get("supersedes_sha256") != old_hash
        or old.get("money") != ORIGINAL_LIMITS
        or new.get("money") != REALLOCATED_LIMITS
    ):
        raise ValueError("Reserve reallocation must retain the authorized financial envelope")
    permitted = {"source_sha256", "created_utc", "supersedes_sha256", "amendment", "money"}
    if {k: v for k, v in old.items() if k not in permitted} != {
        k: v for k, v in new.items() if k not in permitted
    }:
        raise ValueError("Reserve reallocation cannot change the scientific freeze")


class Coordinator(PreviousCoordinator):
    output_relative = "outputs/dsrl07"
    learner_service_relative = "experiments/scripts/dsrl07_learner_service.py"
    verify_plan = staticmethod(verify_plan)
    validate_amendment = staticmethod(validate_amendment)

    def deadline(self):
        return timestamp(self.rental["allocation_deadline_utc"]) - 1200

    def diagnostics(self):
        destination = self.output / "numerical-summary.json"
        if destination.exists():
            result = json.loads(destination.read_text())
            if result.get("admit_live_integration"):
                return result
            raise PendingOperation("Recorded numerical failure needs a diagnosis and repair")
        for lane in range(4):
            self.server(lane)
        for lane in range(4):
            self.processes.launch(
                f"decoder07.lane{lane}",
                [
                    self.openpi,
                    str(self.root / "experiments/scripts/diagnose_dsrl07.py"),
                    "decoder",
                    "--lane",
                    str(lane),
                    "--operation",
                    "initial",
                ],
                cwd=self.root,
                kind="probe",
            )
        decoders = []
        for lane in range(4):
            self.wait(f"decoder07.lane{lane}")
            decoders.append(
                json.loads((self.output / f"decoder-initial-lane{lane}.json").read_text())
            )
        self.processes.stop(self.state["server_generations"]["3"])
        attempts, selected = [], None
        for mode in ("default", "deterministic", "deterministic-no-autotune"):
            operation = f"learner-{mode}"
            self.processes.launch(
                operation,
                [
                    self.learner_python,
                    "-u",
                    str(self.root / "experiments/scripts/diagnose_dsrl07.py"),
                    "learner",
                    "--mode",
                    mode,
                    "--operation",
                    operation,
                ],
                cwd=self.root,
                environment=self.gpu_environment(3),
                kind="learner_diagnostic",
            )
            status = self.wait(operation, allow_failure=True)
            path = self.output / "diagnostics" / operation / "result.json"
            result = (
                json.loads(path.read_text())
                if path.exists()
                else {
                    "mode": mode,
                    "process_exit": status,
                    "source_parity_pass": False,
                }
            )
            attempts.append(result)
            if (
                mode != "default"
                and result.get("source_parity_pass")
                and result.get("reference_repeatability_pass")
            ):
                selected = mode
                break
        cross = []
        for lane in range(1, 4):
            for base, other in zip(decoders[0]["cases"], decoders[lane]["cases"], strict=True):
                if base["snapshot_id"] != other["snapshot_id"]:
                    raise ValueError("Decoder cases are misaligned")
                cross.append(
                    {
                        "lane": lane,
                        "snapshot_id": base["snapshot_id"],
                        **decoder_comparison(base["native_replays"][0], other["native_replays"][0]),
                    }
                )
        result = {
            "schema": "dsrl07-numerical-summary-v1",
            "created_utc": utc(),
            "plan_sha256": sha256(self.plan_path),
            "learner_mode": selected,
            "same_process_pass": all(d["same_process_pass"] for d in decoders),
            "native_draw_passthrough_pass": decoders[0]["native_draw_passthrough_pass"],
            "influence_present": decoders[0]["influence_present"],
            "descriptive_gripper_sensitive_inputs": decoders[0][
                "descriptive_gripper_sensitive_inputs"
            ],
            "descriptive_gripper_sensitive_tasks": decoders[0][
                "descriptive_gripper_sensitive_tasks"
            ],
            "cross_process_comparisons": cross,
            "learner_attempts": attempts,
            "admit_live_integration": bool(
                selected
                and all(d["same_process_pass"] for d in decoders)
                and decoders[0]["native_draw_passthrough_pass"]
                and decoders[0]["influence_present"]
            ),
            "resources": self.sync_counters(),
        }
        atomic_json(destination, result)
        if not result["admit_live_integration"]:
            raise PendingOperation("Numerical diagnosis identifies a failure requiring repair")
        return result

    def engineering(self):
        summary_path = self.output / "stage-a-summary.json"
        if summary_path.exists():
            return json.loads(summary_path.read_text())
        numerical = self.diagnostics()
        jobs = {job["job_id"]: job for job in self.plan["jobs"]["engineering_jobs"]}
        self.server(3, "live-native")
        self.start_episode(jobs["E3"])
        while not self.first_response("E3"):
            self.tick()
        for lane in range(3):
            self.server(lane)
            self.start_episode(jobs[f"E{lane}"])
        self.wait_episodes(["E3"])
        self.processes.stop(self.state["server_generations"]["3"])
        while not all(self.first_response(job) for job in ("E0", "E1", "E2")):
            self.tick()
        self.learner("engineering")
        self.profile()
        self.wait_episodes(["E0", "E1", "E2"])
        profile = self.finish_http("engineering-profile")
        atomic_json(self.output / "engineering-profile.json", profile)
        checkpoint = self.checkpoint("engineering-profile")
        cpu_parity = self.cpu_actor_parity()
        if not cpu_parity.get("valid"):
            raise PendingOperation("CPU actor parity requires diagnosis")
        self.start_episode(jobs["E4"], actor_version=0)
        self.start_episode(jobs["E5"], actor_version=0)
        self.wait_episodes(["E4", "E5"])
        duration = max(
            timestamp(self.state["episodes"][job]["finished_utc"])
            - timestamp(self.state["episodes"][job]["started_utc"])
            for job in ("E4", "E5")
        )
        projection = stage_b_projection(
            duration, profile["median_update_seconds"], cpu_parity["median_cpu_seconds"]
        )
        projected = projection["total_seconds"]
        remaining = timestamp(self.rental["allocation_deadline_utc"]) - time.time()
        source_difference = profile.get("source_parity", {}).get("maximum_state_difference")
        source_pass = (
            isinstance(source_difference, (int, float))
            and np.isfinite(source_difference)
            and source_difference <= 1e-6
        )
        summary = {
            "schema": "dsrl07-stage-a-summary-v1",
            "created_utc": utc(),
            "plan_sha256": sha256(self.plan_path),
            "instance_id": self.rental["instance_id"],
            "admit": bool(source_pass and projected <= remaining),
            "numerical_summary_sha256": sha256(self.output / "numerical-summary.json"),
            "learner_mode": numerical["learner_mode"],
            "profile": profile,
            "source_parity_valid": source_pass,
            "cpu_actor_parity": cpu_parity,
            "engineering_checkpoint": checkpoint,
            "cached_episode_seconds": duration,
            "projected_stage_b_seconds_including_preservation": projected,
            "stage_b_projection_components": projection,
            "remaining_allocation_seconds": remaining,
            "resources": self.sync_counters(),
            "reason": "ready_for_stage_b"
            if source_pass and projected <= remaining
            else "projection_requires_review",
            "topology_note": "Collectors and learner physically coexisted; updates in training run after each three-episode collection barrier",
        }
        atomic_json(summary_path, summary)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("diagnostics", "engineering", "pilot"), required=True)
    parser.add_argument(
        "--plan", type=Path, default=ROOT / "progress/receipts/07-dsrl-stage-a.json"
    )
    parser.add_argument(
        "--rental", type=Path, default=ROOT / "progress/rental-dsrl07-20261025.json"
    )
    parser.add_argument("--repair-receipt", type=Path)
    parser.add_argument("--stop-workers", action="store_true")
    args = parser.parse_args()
    coordinator = Coordinator(
        ROOT, args.plan, args.rental, args.stage, repair_receipt=args.repair_receipt
    )
    complete = False
    try:
        result = getattr(coordinator, args.stage)()
        complete = args.stage == "pilot"
        print(
            json.dumps({k: result[k] for k in ("schema", "admit", "reason") if k in result}),
            flush=True,
        )
    except PendingOperation as error:
        coordinator.event("pending", reason=str(error))
        print(json.dumps({"pending": str(error)}), flush=True)
        return 2
    finally:
        coordinator.close(stop=complete or args.stop_workers)


if __name__ == "__main__":
    raise SystemExit(main())
