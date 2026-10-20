"""Summarize an already completed and independently verified sealed comparison."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.maniguard_review_final import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "design", "protocol", "seal", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve existing results; select a new output file")
    raw = {name: getattr(args, name).read_bytes() for name in
           ("report", "design", "protocol", "seal")}
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()}
    result = summarize(**{name: json.loads(value) for name, value in raw.items()},
                       protocol_sha256=hashes["protocol"])
    result["input_sha256"] = hashes
    result["analysis_source_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__), ROOT / "experiments/helpers/maniguard_review_final.py")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "episodes": 16, "paired_contrasts": 6}))


if __name__ == "__main__":
    main()
