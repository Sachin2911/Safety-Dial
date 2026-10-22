"""Explicit-model-environment workers; no simulation or VLA calls are made here."""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.maniguard_part2_protocol import RUN
from helpers.maniguard_part3_acquisition import context_bank
from helpers.maniguard_part3_data import extract
from helpers.maniguard_part3_model import audit, calibrate, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("extract", "fit", "calibrate", "audit"))
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = parser.parse_args()
    features = RUN / "part3/features"
    directory = ROOT / "data/maniguard/models/phase3-parts2-3-20261018"
    if args.stage == "extract":
        result = extract(features, device=args.device)
    elif args.stage == "fit":
        result = train(features, directory, device=args.device)
    elif args.stage == "calibrate":
        result = calibrate(features, directory, device=args.device)
        context_bank(features, directory)
    else:
        result = audit(features, directory, device=args.device)
    print(json.dumps(result if args.stage == "audit" else
                     {"stage": args.stage, "status": "completed"}), flush=True)


if __name__ == "__main__":
    main()
