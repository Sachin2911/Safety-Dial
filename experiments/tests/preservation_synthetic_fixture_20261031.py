"""Rented-host-only dummy filesystem smoke test of the frozen 20261031 preserver.

This does not open real campaign results, models, datasets, images or simulators.
It invokes one actual build CLI and its independent verify CLI in an isolated tree.
"""

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time


EXPECTED_SOURCE = "d8f393d9adea59e721e7ce2a63a66654a82f99bd4fc7c6db8318af424d482568"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--pod-id", required=True)
    args = parser.parse_args()
    if os.environ.get("RUNPOD_POD_ID") != args.pod_id:
        raise RuntimeError("This fixture is authorized only on its activated rented pod")
    if sha(args.builder) != EXPECTED_SOURCE:
        raise RuntimeError("Frozen builder source differs")
    home = args.fixture_dir.resolve()
    if home.exists():
        raise RuntimeError("Use one fresh isolated fixture namespace")
    home.mkdir(parents=True)
    root, out = home / "project", home / "package"
    root.mkdir()
    builder = root / "experiments/scripts/preserve_lewm_campaign_20261031.py"
    builder.parent.mkdir(parents=True)
    shutil.copyfile(args.builder, builder)
    spec = importlib.util.spec_from_file_location("frozen_preserver", builder)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Real PID plus start tick of a harmless process which has already retired.
    worker = subprocess.Popen([sys.executable, "-c", "pass"], cwd=home)
    retired = module.owner(worker.pid)
    worker.wait(timeout=10)
    assert retired is not None and not module.live(retired)
    write(root / "environment/workers/retired-fixture.json", retired)
    write(root / "environment/source-freeze.json", {"synthetic": True})
    write(root / "results/failed-partial-attempt/live.json", {"attempts": 3, "partial": True})
    (root / "results/failed-partial-attempt/progress-journal.jsonl").write_text(
        '{"environment_actions":0,"encoder_attempts":3}\n'
    )
    for name in module.NAVIGATION_REQUIRED:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("synthetic bytes only: " + name + "\n").encode())
    dataset = {
        "repo_id": "synthetic/metadata-only",
        "revision": "synthetic-frozen-revision",
        "archive_sha256": "0" * 64,
        "member": "not-downloaded.h5",
    }
    reference = root / "configs/lewm_campaign_20261031/dataset.json"
    write(reference, {"dataset": dataset})
    write(
        root / "results/navigation-memory-cache-v3/complete.json",
        {
            "acquisition": {
                "archive_sha256": dataset["archive_sha256"],
                "dataset_revision": dataset["revision"],
                "h5_sha256": "1" * 64,
                "h5_bytes": 17,
                "synthetic": True,
            }
        },
    )
    (root / "assets").mkdir()
    (root / "assets/synthetic-checkpoint.bin").write_bytes(b"dummy asset\n" * 400000)
    (root / "assets/raw.h5").write_bytes(b"excluded raw dummy")
    (root / "environment/.env").write_text("SYNTHETIC_ONLY=do-not-archive\n")
    outside = home / "outside-sentinel.txt"
    outside.write_text("must not be opened through selected symlink\n")
    (root / "assets/outside-link").symlink_to(outside)
    for name in ("reference/dummy.py", "references/dummy.txt", "ruff.toml", "AGENTS.md"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic preserved fixture\n")

    # Keep the exact volatile files actively changing throughout the real CLI.
    stopped = threading.Event()
    heartbeats = []

    def heartbeat():
        while not stopped.is_set():
            payload = {
                "pod_id": args.pod_id,
                "heartbeat_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "status": "armed",
                "synthetic": True,
                "sequence": len(heartbeats),
            }
            temporary = root / "environment/deadline-guard.part"
            write(temporary, payload)
            temporary.replace(root / "environment/deadline-guard.json")
            heartbeats.append(payload)
            stopped.wait(0.005)

    updater = threading.Thread(target=heartbeat, daemon=True)
    updater.start()
    while not heartbeats:
        time.sleep(0.001)
    own = module.owner(os.getpid())
    command = [
        sys.executable,
        str(builder),
        "build",
        "--project-root",
        str(root),
        "--out-dir",
        str(out),
        "--pod-id",
        args.pod_id,
        "--raw-data-reference",
        str(reference),
        "--require-navigation-bank",
        "--allow-analysis-owner",
        str(own["pid"]) + ":" + own["start_ticks"],
    ]
    start = time.monotonic()
    try:
        built = subprocess.run(command, text=True, capture_output=True, cwd=home, timeout=60)
        (home / "build.stdout.log").write_text(built.stdout)
        (home / "build.stderr.log").write_text(built.stderr)
        if built.returncode:
            raise RuntimeError("Synthetic build failed; see retained fixture logs")
        receipt = json.loads((out / "receipt.json").read_text())
        verify_command = [
            sys.executable,
            str(builder),
            "verify",
            str(out / "lewm-campaign-20261031.tar.gz"),
            "--expected-archive-sha256",
            receipt["archive_sha256"],
        ]
        checked = subprocess.run(
            verify_command, text=True, capture_output=True, cwd=home, timeout=60
        )
        (home / "verify.stdout.log").write_text(checked.stdout)
        (home / "verify.stderr.log").write_text(checked.stderr)
        if checked.returncode:
            raise RuntimeError("Synthetic verify failed; see retained fixture logs")
    finally:
        stopped.set()
        updater.join(timeout=5)
    manifest = json.loads((out / "manifest.json").read_text())
    verification = json.loads(checked.stdout)
    names = set(manifest["files"])
    mandatory = set(module.NAVIGATION_REQUIRED) | {
        "experiments/scripts/preserve_lewm_campaign_20261031.py",
        "reference/dummy.py",
        "references/dummy.txt",
        "ruff.toml",
        "environment/workers/retired-fixture.json",
        "results/failed-partial-attempt/live.json",
        "results/failed-partial-attempt/progress-journal.jsonl",
        "assets/synthetic-checkpoint.bin",
    }
    assert mandatory <= names
    assert not {
        "assets/raw.h5",
        "environment/.env",
        "assets/outside-link",
        *module.VOLATILE,
    } & names
    skipped = {row["path"]: row for row in manifest["skipped"]}
    assert skipped["assets/outside-link"]["reason"] == "symlink not followed"
    assert all(name in skipped for name in module.VOLATILE)
    assert manifest["control_snapshot"]["heartbeat"]["synthetic"]
    assert manifest["quiescence_before"]["all_declared_workers_retired"]
    assert receipt["final_quiescence"]["all_declared_workers_retired"]
    assert manifest["raw_public_dataset"]["raw_h5_included"] is False
    assert "verified_h5_acquisition" in manifest["raw_public_dataset"]
    assert verification["every_member_streamed_and_hashed"]
    assert verification["archive_sha256"] == receipt["archive_sha256"]
    assert sha(builder) == EXPECTED_SOURCE
    assert len(heartbeats) >= 2
    report = {
        "schema": "lewm-preservation-synthetic-smoke-20261031-v1",
        "synthetic": True,
        "qualified": True,
        "pod_id": args.pod_id,
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "builder_sha256": EXPECTED_SOURCE,
        "fixture_source_sha256": sha(Path(__file__)),
        "build_cli": command,
        "verify_cli": verify_command,
        "receipt": receipt,
        "verification": verification,
        "checks": [
            "actual subcommand-first build and verify CLIs",
            "live atomic guard heartbeat excluded from source inventory and captured",
            "retired exact PID/starttick worker and explicit fixture analysis owner",
            "source, references, ruff, all required navigation-bank dummy files included",
            "failed/partial evidence retained",
            "raw H5, env and external symlink excluded without following",
            "all archive members independently streamed and hash checked",
            "frozen builder source unchanged",
        ],
        "synthetic_guard_heartbeat_writes": len(heartbeats),
        "elapsed_seconds": time.monotonic() - start,
        "real_campaign_results_read": 0,
        "new_model_queries": 0,
        "new_simulator_calls": 0,
    }
    write(home / "qualification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
