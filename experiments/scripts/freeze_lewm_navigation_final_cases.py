"""Materialize a final preparation freeze after the three-repeat opening lock exists.

Reads JSON and source hashes only. This script never opens the public dataset,
loads numerical arrays, constructs an environment or starts an experiment.
"""

import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.lewm_navigation_final_admission import validate_final_opening


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def freeze(template_path, opening_path, output_path):
    template = json.loads(template_path.read_text())
    if template.get("execution_enabled") is not False:
        raise ValueError("Expected the disabled prospective design template")
    for name, expected in template["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Final preparation design source changed: " + name)
    opening = json.loads(opening_path.read_text())
    result = copy.deepcopy(template)
    result.update(
        frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        execution_enabled=True,
        final_opening={"path": str(opening_path), "sha256": digest(opening_path)},
        selection_locks=opening.get("selection_locks", []),
        design_template={"path": str(template_path), "sha256": digest(template_path)},
    )
    validate_final_opening(result, ROOT)
    if output_path.exists():
        raise FileExistsError("Never overwrite a final preparation freeze")
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return {
        "protocol": str(output_path),
        "sha256": digest(output_path),
        "workers_started": 0,
        "final_pixels_read": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template",
        type=Path,
        default=ROOT / "configs/lewm_campaign_20261031/navigation-final-cases-template-v1.json",
    )
    parser.add_argument("--opening-lock", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "configs/lewm_campaign_20261031/navigation-final-cases-v1.json",
    )
    args = parser.parse_args()
    print(json.dumps(freeze(args.template, args.opening_lock, args.out), indent=2))


if __name__ == "__main__":
    main()
