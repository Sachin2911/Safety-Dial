"""Resume predeclared one- or two-job waves under the global physical queue."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.maniguard_latent_parallel import PairedPhysicalQueue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--repair-note")
    args = parser.parse_args()
    waves = json.loads(args.jobs.read_text())
    queue = PairedPhysicalQueue(args.run, args.protocol)
    for wave in waves:
        queue.plan(wave)
        print("Observe or execute wave", [job["id"] for job in wave], flush=True)
        receipts = queue.evaluate_batch([job["id"] for job in wave], repair_note=args.repair_note)
        print(json.dumps([{"id": job["id"], "score": receipt["score"],
                           "steps": receipt["physical"]["steps"]}
                          for job, receipt in zip(wave, receipts, strict=True)]), flush=True)


if __name__ == "__main__":
    main()
