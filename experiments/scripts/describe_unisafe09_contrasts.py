"""Post hoc descriptive paired contrasts for the plan's mechanism questions.

These are not primary contrasts. They use unadjusted 95% intervals, the same
paired exact binomial and seed-bootstrap helpers, and carry no confirmation weight.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_infra import atomic_json, utc
from helpers.unisafe09_protocol import digest
from helpers.unisafe09p_statistics import blocked_binary, blocked_mean, matched_groups

# Each pair answers a question listed in the task-9 plan, section 1.
CONTRASTS = (
    ("d15_g100", "d00_g100", "Sensed grasp-release delay versus native"),
    ("d40_g100", "d15_g100", "Longer bounded delay"),
    ("d15_g075", "d15_g100", "Slower extraction translation at delay 15"),
    ("d15_g075", "reference_random", "Primary pair, continuation outcome only"),
    ("history_only", "d15_g100", "Primary pair, continuation outcome only"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validation = json.loads(args.validation.read_text())
    if not validation["passed"] or len(validation["episodes"]) != 384:
        raise ValueError("A validated complete C panel is required")
    rows = validation["episodes"]
    hosts = sorted({r["physical_host"] for r in rows})
    rng = np.random.default_rng(2026102609)
    draws = {}
    for host in hosts:
        n = len({r["seed"] for r in rows if r["physical_host"] == host})
        draws[host] = rng.integers(0, n, size=(10000, n))
    results = []
    for candidate, reference, question in CONTRASTS:
        groups = matched_groups(rows, candidate, reference)
        results.append({
            "candidate": candidate, "reference": reference, "question": question,
            "C": blocked_binary(groups, "safe_success"),
            "V": blocked_binary(groups, "violation"),
            "J": blocked_mean(groups, draws),
            "sustained": blocked_binary(groups, "sustained_safe_success"),
        })
    atomic_json(args.output, {
        "utc": utc(), "status": "post hoc descriptive; not primary; unadjusted 95%",
        "selected_after_condition_aggregates_were_seen": True,
        "selection_rule": "Pairs answering the plan's section-1 questions, plus the "
                          "continuation outcome for both frozen primary pairs",
        "validation_sha256": digest(args.validation), "contrasts": results,
        "source_sha256": digest(Path(__file__)),
    })
    for row in results:
        print(row["candidate"], "-", row["reference"], {
            k: (round(row[k]["difference"], 3), [round(x, 3) for x in row[k]["interval"]])
            for k in ("C", "V", "J", "sustained")})


if __name__ == "__main__":
    main()
