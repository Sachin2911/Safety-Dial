"""Revalidate every original full episode from separately restored private evidence."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.maniguard_latent_restoration import build_inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--source-proof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Do not overwrite a restoration proof")
    proof = build_inventory(args.run, args.protocol, args.paths, args.source_proof,
                            require_complete=not args.allow_incomplete)
    with args.output.open("x") as stream:
        json.dump(proof, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"restored_full_episodes": len(proof["episodes"]),
                      "all_60_covered": proof["all_60_original_valid_episodes_covered"],
                      "new_physics_or_inference": False}))


if __name__ == "__main__":
    main()
