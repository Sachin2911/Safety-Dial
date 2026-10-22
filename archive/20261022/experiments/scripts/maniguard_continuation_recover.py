"""Recover the two frozen screen waves, retaining every original archive tree."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from maniguard_latent_recover import recover

from helpers.maniguard_continuation_restoration import merge_recovered_waves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, nargs=4, required=True,
                        help="Wave0 model/dataset, then wave1 model/dataset receipts")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for index, path in enumerate(args.receipts):
        receipt = json.loads(path.read_text())
        expected = f"20261020/continuation-screen-wave{index // 2}-20261020/"
        if receipt["prefix"] != expected or receipt["repo_type"] != ("model", "dataset")[index % 2]:
            raise ValueError("Use the exact ordered two-wave screen receipts")
    args.output.mkdir(parents=True, exist_ok=False)
    waves = [recover(args.receipts[2 * wave:2 * wave + 2], args.output / "waves" / str(wave))
             for wave in (0, 1)]
    merged = args.output / "merged"
    replacements = merge_recovered_waves(waves, merged)
    sources = [Path(__file__), ROOT / "experiments/helpers/maniguard_continuation_restoration.py"]
    result = {
        "schema": "maniguard-continuation-recovery-v1",
        "archives": [row for wave in waves for row in wave["archives"]],
        "merged": str(merged.resolve()),
        "all_archives_redownloaded_and_sha256_checked": True,
        "mutable_metadata_replacements": replacements,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources},
        "scope": "Each wave independently uses the unchanged strict original recovery tool. "
                 "Only the screen ledger and coordinator log may change between waves. "
                 "All per-episode evidence and other metadata remain immutable. Every "
                 "original wave tree is retained. No process or policy service is started.",
    }
    (args.output / "recovery-receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"archives": len(result["archives"]),
                      "mutable_metadata_replacements": len(replacements), "merged": str(merged)}))


if __name__ == "__main__":
    main()
