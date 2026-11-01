"""Rented-only saved JSON timing and bounded engineering evidence packaging.

No model, simulator, final-case or array load. PG job-file modification times give
completion observations, never an exact action-throughput denominator.
"""

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile


ROOT = Path("/workspace/Safety-Dial")
NAME = "navigation-recovery-overlap-qualification-v2"
POD = "nj2fqkp2prgk8x"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(name):
    return json.loads((ROOT / name).read_text())


def main():
    if os.environ.get("RUNPOD_POD_ID") != POD:
        raise RuntimeError("Saved timing inspection belongs on the actual recovery pod")
    protocol = read("configs/lewm_campaign_20261031/" + NAME + ".json")
    config = protocol["overlap_qualification"]
    observations = read(config["peer_observations_path"])
    first = config["reference_stage"] + "/jobs/" + config["reference_job_id"] + "/episode"
    second = protocol["output_namespace"] + "/jobs/" + config["reference_job_id"] + "/episode"
    original, engineering = read(first + "/result.json"), read(second + "/result.json")
    preflight = read("environment/" + NAME + "-preflight.json")
    pg_root = "results/navigation-budget30-relocated-v1/r2/reinforce"
    state = read(pg_root + "/state.json")
    pg_after = {key: state.get(key) for key in preflight["pg_before"] if key != "completed_jobs"}
    pg_after["completed_jobs"] = len(state["completed_job_ids"])
    before = dt.datetime.fromisoformat(observations["before"]["utc"]).timestamp()
    after = dt.datetime.fromisoformat(observations["after"]["utc"]).timestamp()
    committed = []
    for path in sorted((ROOT / pg_root / "jobs").glob("*/result.json")):
        stamp = path.stat().st_mtime
        if before <= stamp <= after:
            row = json.loads(path.read_text())
            committed.append(
                {
                    "job_id": row["job_id"],
                    "result_mtime_utc": dt.datetime.fromtimestamp(
                        stamp, dt.timezone.utc
                    ).isoformat(),
                    "environment_actions": row["environment_actions"],
                }
            )
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    report = {
        "schema": "lewm-recovery-overlap-timing-v2",
        "pod_id": POD,
        "saved_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "summary_source_sha256": sha(Path(__file__)),
        "reference_episode_seconds": original["episode_seconds"],
        "engineering_episode_seconds": engineering["episode_seconds"],
        "reference_planning_seconds": original["planning_seconds"],
        "engineering_planning_seconds": engineering["planning_seconds"],
        "engineering_subprocess_seconds": observations["after"]["monotonic_seconds"]
        - observations["before"]["monotonic_seconds"],
        "engineering_resources": engineering["resources"],
        "pg_preflight_utc": preflight["checked_utc"],
        "pg_before": preflight["pg_before"],
        "pg_after": pg_after,
        "pg_completed_result_files_during_engineering_subprocess": committed,
        "pg_completed_file_observation_caveat": "Filesystem result-commit times identify completed PG jobs in the subprocess interval; their full charged actions can cross its boundaries. These are not exact in-interval actions or a causal throughput estimate.",
        "gpu_snapshot_after_qualification_during_following_PG_CMA_overlap": gpu.stdout.strip(),
        "gpu_snapshot_caveat": "A current sample is not the maximum memory used during qualification.",
        "scope": "One same-host engineering replay; timings are descriptive and include filesystem contention. No optimizer or efficacy claim.",
        "new_model_simulator_calls_by_summary": 0,
    }
    timing = ROOT / "results/evidence-audits/navigation-recovery-overlap-timing-v2.json"
    if timing.exists():
        raise FileExistsError("Never overwrite timing evidence")
    timing.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    selected = {
        "experiments/scripts/launch_lewm_campaign_frozen_overlap.py",
        "experiments/scripts/freeze_lewm_recovery_overlap_qualification_v2.py",
        "experiments/scripts/qualify_lewm_recovery_overlap.py",
        "experiments/scripts/summarize_lewm_recovery_overlap_timing.py",
        "configs/lewm_campaign_20261031/" + NAME + ".json",
        "configs/lewm_campaign_20261031/" + NAME + "-authorization.json",
        "environment/" + NAME + "-preflight.json",
        "environment/" + NAME + "-peer-observations.json",
        "environment/" + NAME + ".log",
        "environment/" + NAME + ".sh",
        "environment/workers/" + NAME + ".json",
        "results/evidence-audits/" + NAME + ".json",
        str(timing.relative_to(ROOT)),
    }
    for path in (ROOT / protocol["output_namespace"]).rglob("*"):
        if path.is_file() and not path.is_symlink():
            selected.add(str(path.relative_to(ROOT)))
    manifest = {
        name: {"sha256": sha(ROOT / name), "bytes": (ROOT / name).stat().st_size}
        for name in sorted(selected)
    }
    destination = Path("/workspace/recovery-overlap-engineering-v2.tar.gz")
    if destination.exists():
        raise FileExistsError("Never overwrite the engineering evidence archive")
    with tarfile.open(destination, "w:gz") as archive:
        for name in sorted(manifest):
            archive.add(ROOT / name, arcname=name, recursive=False)
    for name, record in manifest.items():
        if sha(ROOT / name) != record["sha256"]:
            raise RuntimeError("Selected retired engineering evidence changed")
    receipt = {
        "archive": str(destination),
        "archive_sha256": sha(destination),
        "members": manifest,
        "new_model_simulator_calls": 0,
    }
    Path("/workspace/recovery-overlap-engineering-v2-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in receipt.items() if key != "members"}))


if __name__ == "__main__":
    main()
