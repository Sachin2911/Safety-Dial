"""Freeze task 15 recovery sources and build the bundle shipped to the recovery rental.

Two stages, mirroring the original task 15 flow: recovery_prep (host preparation, frozen
before the create request) and recovery_live (everything the recovery episodes execute,
frozen before any simulator start). Each freeze records the authorization, the unchanged
$10 money limits and a SHA256 for every source. Earlier task 15 freezes are never edited;
the recovery lists extend the stage D list, whose files match the stage D freeze.
--bundle-only rebuilds a bundle for an existing freeze after re-verifying every hash, so
the bundle can carry the rental record written by the create request.
"""

import argparse
import datetime as dt
import hashlib
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/scripts"))
import make_regrasp15_freeze as original  # noqa: E402

RENTAL = "progress/rental-regrasp15r1-20261026.json"
RECOVERY_PREP = original.PREPARATION + [
    "experiments/scripts/bootstrap_regrasp15r1.sh",
    "experiments/scripts/prepare_regrasp15r1.sh",
    "experiments/scripts/regrasp15r1_watchdog.py",
    "experiments/scripts/manage_regrasp15r1.py",
    "experiments/scripts/make_regrasp15r1_freeze.py",
    "experiments/scripts/admit_regrasp15r1.py",
    "progress/15-regrasp-recovery-amendment.md",
    "progress/receipts/15-regrasp-recovery-evidence-20261026.json",
    "progress/receipts/15-regrasp-recovery-admission.json",
]
RECOVERY_LIVE = original.STAGE_D + [
    name for name in RECOVERY_PREP if name not in original.STAGE_D
] + [
    "configs/regrasp15r1/recovery.json",
    "configs/regrasp15r1/historical-stage-c-decision.json",
    "experiments/scripts/plan_regrasp15r1.py",
    "experiments/scripts/run_regrasp15r1.py",
    "experiments/scripts/launch_regrasp15r1.sh",
    "experiments/scripts/analyze_regrasp15r1.py",
    "experiments/tests/test_regrasp15r1.py",
]
STAGES = {"recovery_prep": RECOVERY_PREP, "recovery_live": RECOVERY_LIVE}
AUTHORIZATION = {
    **original.AUTHORIZATION,
    "scope": original.AUTHORIZATION["scope"]
    + "; recovery on a replacement host after the original rental became provider-absent, "
    "within the same cumulative $10 cap and 20 aggregate hours",
    "quotes": original.AUTHORIZATION["quotes"] + [
        "Complete task 15: the ManiGuard Jar regrasp-adapter experiment.",
        "If further experiments require a replacement GPU, complete financial admission "
        "before renting. Retain the original cumulative $10 cap, including all previous "
        "charges.",
        "Follow the handoff's replacement-host rules: collect appropriate native controls "
        "on the same host, preserve historical episodes separately, and freeze recovery "
        "changes before execution. Keep the declared conditions and endpoints unchanged.",
    ],
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_bundle(bundle, files, freeze):
    bundle.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle, "w:gz") as tar:
        for name in [*files, str(freeze.relative_to(ROOT))]:
            tar.add(ROOT / name, arcname=name)
        if (ROOT / RENTAL).exists():
            tar.add(ROOT / RENTAL, arcname=RENTAL)
    print("bundle", bundle, sha256(bundle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--bundle-only", action="store_true")
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    files = STAGES[args.stage]
    freeze = (ROOT / args.freeze) if not args.freeze.is_absolute() else args.freeze
    if args.bundle_only:
        record = json.loads(freeze.read_text())
        if record["stage"] != args.stage or sorted(record["source_sha256"]) != sorted(files):
            raise SystemExit("Existing freeze does not match this stage's file list")
        for name, expected in record["source_sha256"].items():
            if sha256(ROOT / name) != expected:
                raise SystemExit(f"Frozen source changed: {name}")
        if not args.bundle:
            raise SystemExit("--bundle-only needs --bundle")
        write_bundle(args.bundle, files, freeze)
        return
    if freeze.exists():
        raise SystemExit(f"Refusing to overwrite an existing freeze: {freeze}")
    record = {
        "schema": "regrasp15-freeze-v1",
        "stage": args.stage,
        "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "authorization": AUTHORIZATION,
        "money": original.MONEY,
        "plan_sha256": sha256(ROOT / "progress/15-regrasp-dial-plan.md"),
        "amendment_sha256": sha256(ROOT / "progress/15-regrasp-recovery-amendment.md"),
        "source_sha256": {name: sha256(ROOT / name) for name in files},
        "note": args.note,
    }
    freeze.parent.mkdir(parents=True, exist_ok=True)
    freeze.write_text(json.dumps(record, indent=2) + "\n")
    print("froze", freeze.relative_to(ROOT), len(files), "files")
    if args.bundle:
        write_bundle(args.bundle, files, freeze)


if __name__ == "__main__":
    main()
