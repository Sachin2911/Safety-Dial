"""Six-lock access wrapper around the unchanged primary final-case preparer."""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.lewm_navigation_joint_final_v2 import validate_joint_preparation


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    joint, _, _, primary = validate_joint_preparation(protocol, ROOT)
    if os.environ.get("RUNPOD_POD_ID") not in primary["allowed_pod_ids"]:
        raise RuntimeError("Only the declared rented preparation host may open final data")
    if output.exists():
        raise FileExistsError("Use a fresh shared final-case output")
    receipt = {
        "schema": "lewm-navigation-joint-final-access-v2",
        "authorized_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol_sha256": digest(protocol_path),
        "joint_opening": protocol["joint_opening"],
        "all_six_repeat_selections_fixed": True,
        "primary_preparation_protocol": protocol["primary_preparation_protocol"],
        "source_sha256": protocol["source_sha256"],
        "final_pixels_read_before_delegation": 0,
    }
    gate_path = output.with_name(output.name + "-joint-access.json")
    if gate_path.exists():
        raise FileExistsError("Retain earlier joint access receipts")
    gate_path.write_text(json.dumps(receipt, indent=2) + "\n")
    # Heavy imports occur only AFTER all six locks and their sources are validated.
    from prepare_lewm_navigation_final_cases import prepare as primary_prepare

    primary_path = ROOT / protocol["primary_preparation_protocol"]["path"]
    try:
        primary_prepare(primary, output, primary_path)
    finally:
        if output.exists():
            receipt["joint_primary_opening_sha256"] = joint["primary_opening"]["sha256"]
            (output / "joint-preparation.json").write_text(json.dumps(receipt, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "results/navigation-final-cases-v1")
    args = parser.parse_args()
    prepare(args.protocol, args.out)


if __name__ == "__main__":
    main()
