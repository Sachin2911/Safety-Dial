"""Freeze task 15 sources for one stage and build the bundle shipped to the rental.

The freeze records the authorization, the $10 money limits and a SHA256 for every
shipped source file. The bundle contains exactly those files plus the freeze itself
and, when it exists, the rental record. Later stages write new freeze versions; no
earlier freeze is edited.
"""

import argparse
import datetime as dt
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RENTAL = "progress/rental-regrasp15-20261025.json"

PREPARATION = [
    "experiments/helpers/dsrl06_infra.py",
    "experiments/helpers/regrasp15_infra.py",
    "experiments/scripts/bootstrap_regrasp15.sh",
    "experiments/scripts/manage_regrasp15.py",
    "experiments/scripts/prepare_regrasp15.sh",
    "experiments/scripts/regrasp15_watchdog.py",
    "experiments/scripts/make_regrasp15_freeze.py",
    "progress/15-regrasp-dial-plan.md",
    "progress/panels.json",
    "progress/receipts/01-jar-eval.yaml",
    "progress/receipts/01-sources.json",
    "progress/receipts/01b-env-behavior-freeze.txt",
    "scripts/stage2/prepare_vast_host.sh",
    "scripts/todo1/download_pinned.py",
    "scripts/todo1/eval_family_lane.sh",
    "scripts/todo1/probe_server.py",
    "scripts/todo1/serve_jar.sh",
    "scripts/todo1b/install_behavior_replay.sh",
    "scripts/todo1b/install_maniguard_deps.sh",
    "scripts/todo1b/nvidia-egl-vendor.json",
    "scripts/todo1b/serve_jar_lane.sh",
    "scripts/todo2/record_rollout.py",
    "scripts/todo2/check_recording.py",
]
STAGE0 = PREPARATION + [
    "experiments/helpers/regrasp15_kinematics.py",
    "experiments/helpers/regrasp15_check.py",
    "experiments/helpers/regrasp15_sampler.py",
    "experiments/scripts/pack_regrasp15_stage0.py",
    "experiments/scripts/stage0_regrasp15.sh",
    "experiments/scripts/steer_regrasp15.py",
]
LIVE = STAGE0 + [
    "configs/regrasp15/design.json",
    "configs/regrasp15/jar_eval.yaml",
    "experiments/helpers/accounted_recorder.py",
    "experiments/helpers/regrasp15_runtime.py",
    "experiments/helpers/regrasp15_validation.py",
    "experiments/helpers/tracking.py",
    "experiments/scripts/serve_regrasp15.py",
    "experiments/scripts/patch_regrasp15.py",
    "experiments/scripts/plan_regrasp15.py",
    "experiments/scripts/run_regrasp15.py",
    "experiments/scripts/launch_regrasp15.sh",
    "experiments/scripts/probe_regrasp15.py",
    "experiments/scripts/analyze_regrasp15.py",
    "experiments/scripts/resume_behavior_regrasp15.sh",
    "experiments/tests/test_regrasp15.py",
]
STAGE_D = LIVE + [
    "progress/receipts/15-regrasp-stage-d.json",
    "experiments/scripts/stage_d_regrasp15.py",
    "experiments/scripts/launch_regrasp15_d.sh",
    "experiments/scripts/analyze_regrasp15_d.py",
]
STAGES = {"preparation": PREPARATION, "stage0": STAGE0, "live": LIVE, "stage_d": STAGE_D}
AUTHORIZATION = {
    "user_launch": True,
    "scope": "Task 15 regrasp adapter: one rented RTX 3090 or 4090 under a $10 total cap, "
    "stage 0 then live engineering, dial screen and confirmation",
    "quotes": [
        "Yes that is exactly it, please go ahead",
        "Actually don't run it locally rent your own GPU with a $10 budget",
        "So then after you do the initial tests you can run the rest too",
        "Also make sure you don't stop until you get it working",
        "Yes please go ahead",
    ],
}
MONEY = {"total_usd": 10.0, "active_usd": 8.0, "active_shutdown_usd": 7.9,
         "other_reserve_usd": 2.0, "aggregate_hours": 20.0, "rate_usd_hour": 0.65}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    files = STAGES[args.stage]
    freeze = (ROOT / args.freeze) if not args.freeze.is_absolute() else args.freeze
    if freeze.exists():
        raise SystemExit(f"Refusing to overwrite an existing freeze: {freeze}")
    record = {
        "schema": "regrasp15-freeze-v1",
        "stage": args.stage,
        "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "authorization": AUTHORIZATION,
        "money": MONEY,
        "plan_sha256": sha256(ROOT / "progress/15-regrasp-dial-plan.md"),
        "source_sha256": {name: sha256(ROOT / name) for name in files},
        "note": args.note,
    }
    freeze.parent.mkdir(parents=True, exist_ok=True)
    freeze.write_text(json.dumps(record, indent=2) + "\n")
    print("froze", freeze.relative_to(ROOT), len(files), "files")
    if args.bundle:
        args.bundle.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.bundle, "w:gz") as tar:
            for name in [*files, str(freeze.relative_to(ROOT))]:
                tar.add(ROOT / name, arcname=name)
            if (ROOT / RENTAL).exists():
                tar.add(ROOT / RENTAL, arcname=RENTAL)
        print("bundle", args.bundle, sha256(args.bundle))


if __name__ == "__main__":
    main()
