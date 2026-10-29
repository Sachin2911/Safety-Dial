"""Freeze added-host jobs without changing the original C scientific schedule."""

import argparse
import copy
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_infra import utc
from helpers.unisafe09_protocol import digest, verify_freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["A_parallel", "C_parallel"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--engineering", type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.bundle.exists():
        raise ValueError("Use new immutable freeze and bundle paths")
    receipt = ROOT / "progress/receipts"
    original_path = receipt / "09-unisafe-C.json"
    original = verify_freeze(original_path)
    allocation_path = receipt / "09-unisafe-parallel-allocation.json"
    allocation = json.loads(allocation_path.read_text())
    preparation = json.loads((receipt / "09-unisafe-parallel-preparation.json").read_text())
    source = dict(preparation["source_sha256"])
    extra = [
        "configs/unisafe09p/vulkan-icd.json", "experiments/helpers/unisafe09p_runtime.py",
        "experiments/scripts/worker_unisafe09p.py", "experiments/scripts/run_unisafe09p.py",
        "experiments/scripts/freeze_unisafe09p.py", "experiments/scripts/validate_unisafe09p.py",
        "experiments/tests/test_unisafe09p_queue.py",
    ]
    for name, expected in source.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Preparation source changed: {name}")
    source.update({name: digest(ROOT / name) for name in extra})
    lanes = {}
    jobs = []
    engineering = None
    if args.stage == "A_parallel":
        old = json.loads((receipt / "09-unisafe-A-v7.json").read_text())
        by_id = {j["id"]: j for j in old["jobs"]}
        for lane in range(4):
            for index in (0, 2, 8):
                job = copy.deepcopy(by_id[f"engineering-{index:02d}"])
                job.update(id=f"A_parallel-g{lane}-{index:02d}", stage="A_parallel")
                lanes[job["id"]] = lane
                jobs.append(job)
    else:
        if args.engineering is None:
            raise ValueError("New-host confirmation requires engineering acceptance")
        engineering = json.loads(args.engineering.read_text())
        if not engineering["passed"]:
            raise ValueError("Engineering gates did not pass")
        by_id = {j["id"]: j for j in original["jobs"]}
        for row in allocation["assignments"]:
            if row["host"] == "parallel":
                jobs.append(copy.deepcopy(by_id[row["job_id"]]))
                lanes[row["job_id"]] = row["gpu_lane"]
        if len(jobs) != 192 or [list(lanes.values()).count(i) for i in range(4)] != [48]*4:
            raise ValueError("The complete 32-block assignment must be preserved")
    rental_path = ROOT / "progress/rental-unisafe09p-20261025.json"
    record = {
        "schema": "unisafe09p-freeze-v1", "utc": utc(), "stage": args.stage,
        "authorization": {"user_launch": True, "persistent_engineering_repairs": True,
                          "combined_cap_usd": 15},
        "source_sha256": source, "jobs": jobs, "gpu_lanes": lanes,
        "gpu_uuids": {
            "0": "GPU-53f3c349-054c-1c99-fa92-e69a8dfd0722",
            "1": "GPU-667146b1-4765-f84d-d642-f545b63a18f0",
            "2": "GPU-b11a996e-8030-30fa-4cc2-49b514b2e805",
            "3": "GPU-d2c8906d-28ef-62dd-993a-e00320a6d614",
        },
        "local_caps": {"attempts": 208, "controls": 70720, "offline_queries": 4360,
                       "replacements": 4},
        "global_caps": {"attempts": 606, "controls": 198360,
                        "decimated_physics_steps": 991800, "offline_queries": 5000,
                        "total_query_bundles": 203360},
        "original_C_freeze_sha256": digest(original_path),
        "allocation_sha256": digest(allocation_path),
        "plan_sha256": digest(ROOT / "progress/09-unisafe-dial-plan.md"),
        "rental_sha256": digest(rental_path),
        "engineering_acceptance": engineering,
        "engineering_acceptance_sha256": digest(args.engineering) if args.engineering else None,
        "selection": original["selection"], "design": original["design"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2)+"\n")
    include = list(source) + [str(args.output.relative_to(ROOT)) if args.output.is_absolute()
                            else str(args.output)]
    include += [str(p.relative_to(ROOT)) for p in (
        rental_path, original_path, allocation_path, ROOT / "progress/09-unisafe-dial-plan.md",
        receipt / "09-unisafe-parallel-authorization.json",
        receipt / "09-unisafe-parallel-model-authorization.json",
    )]
    if args.engineering:
        include.append(str(args.engineering.resolve().relative_to(ROOT)))
    args.bundle.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.bundle, "w:gz") as archive:
        for name in dict.fromkeys(include):
            archive.add(ROOT / name, arcname=name)
    print(json.dumps({"freeze": str(args.output), "sha256": digest(args.output),
                      "jobs": len(jobs), "source_files": len(source),
                      "bundle_sha256": digest(args.bundle)}))


if __name__ == "__main__":
    main()
