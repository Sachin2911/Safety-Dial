"""Export independent offline diagnostics without loading a VLA or simulator."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.maniguard_review_sensitivity import analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path to preserve prior review evidence")
    result = analyze(args.run)
    source_paths = [Path(__file__), ROOT / "experiments/helpers/maniguard_review_sensitivity.py"]
    result["review_source_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "completed_training_episodes": len(
        result["episodes"]), "families": list(result["candidate_responses_by_family"]),
        "new_simulator_controls": 0, "new_policy_queries": 0, "new_encoder_calls": 0}))


if __name__ == "__main__":
    main()
