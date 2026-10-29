"""Summarize a validated panel and apply the frozen selection/contrast rules."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_infra import atomic_json, utc
from helpers.unisafe09_protocol import select_confirmation
from helpers.unisafe09_statistics import paired_binary, paired_mean, wilson


def summarize(rows, attempted):
    result = {}
    for cid in sorted({r["condition"] for r in rows}):
        group = sorted([r for r in rows if r["condition"] == cid], key=lambda r: r["seed"])
        n = len(group)
        outcomes = Counter(r["outcome"] for r in group)
        successes, violations = (
            sum(r["safe_success"] for r in group),
            sum(r["violation"] for r in group),
        )
        result[cid] = {
            "valid": n,
            "attempted": attempted[cid],
            "invalid": attempted[cid] - n,
            "outcomes": dict(outcomes),
            "C_count": successes,
            "C": successes / n,
            "C_wilson95": wilson(successes, n),
            "V_count": violations,
            "V": violations / n,
            "V_wilson95": wilson(violations, n),
            "J": float(np.mean([r["J"] for r in group])),
            "C_by_deadline": {
                str(t): sum(r["safe_success"] and r["completion_time_s"] <= t for r in group) / n
                for t in (2, 5, 10, 15)
            },
            "mean_success_time_secondary_s": float(
                np.mean([r["completion_time_s"] for r in group if r["safe_success"]])
            )
            if successes
            else None,
            "sustained_count": sum(r["sustained_safe_success"] is True for r in group),
            "post_success_violations": sum(r["post_success_violation"] is True for r in group),
            "post_success_predicate_losses": sum(
                r["post_success_predicate_loss"] is True for r in group
            ),
            "continuation_enabled": all(r["continuation_enabled"] for r in group),
            "engagement_means": {
                key: float(np.mean([r[key] for r in group]))
                for key in (
                    "native_closing_proposals",
                    "grasp_evidence_steps",
                    "gripper_override_steps",
                    "translation_active_steps",
                    "interventions",
                    "eef_path_length_m",
                    "max_middle_displacement_m",
                )
            },
        }
    return result


def contrast(rows, candidate, reference, draws, alpha):
    a = sorted([r for r in rows if r["condition"] == candidate], key=lambda r: r["seed"])
    b = sorted([r for r in rows if r["condition"] == reference], key=lambda r: r["seed"])
    if [r["seed"] for r in a] != [r["seed"] for r in b]:
        raise ValueError("Contrast panels are not matched")
    return {
        "candidate": candidate,
        "reference": reference,
        "C": paired_binary([r["safe_success"] for r in a], [r["safe_success"] for r in b], alpha),
        "V": paired_binary([r["violation"] for r in a], [r["violation"] for r in b], alpha),
        "J": paired_mean([r["J"] for r in a], [r["J"] for r in b], draws, alpha),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--raw", type=Path, default=ROOT / "outputs/unisafe09/remote")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path)
    args = parser.parse_args()
    validation = json.loads(args.validation.read_text())
    freeze = json.loads(args.freeze.read_text())
    if not validation["passed"]:
        raise ValueError("Only a validated complete panel can be analyzed")
    from helpers.unisafe09_protocol import digest

    if digest(args.freeze) != validation["freeze_sha256"]:
        raise ValueError("Validation belongs to another freeze")
    rows = validation["episodes"]
    attempted = Counter()
    for intent in (args.raw / "attempts").glob("*/intent.json"):
        job = json.loads(intent.read_text())["job"]
        if job["stage"] == freeze["stage"]:
            attempted[job["condition"]] += 1
    result = {
        "utc": utc(),
        "stage": freeze["stage"],
        "conditions": summarize(rows, attempted),
        "selection_estimates": freeze["stage"] == "B",
        "optimizer_updates": 0,
    }
    if freeze["stage"] == "B":
        selection = select_confirmation(rows)
        result["selection"] = selection
        if args.selection_output:
            atomic_json(args.selection_output, selection)
    elif freeze["stage"] == "C":
        selection = freeze["selection"]
        contrasts = selection["primary_contrasts"]
        n = len({r["seed"] for r in rows})
        draws = np.random.default_rng(2026102609).integers(0, n, size=(10000, n))
        result["primary_descriptive95"] = [contrast(rows, a, b, draws, 0.05) for a, b in contrasts]
        alpha = 0.05 / (3 * len(contrasts)) if contrasts else 0.05
        result["primary_simultaneous"] = [contrast(rows, a, b, draws, alpha) for a, b in contrasts]
        result["familywise_alpha"] = 0.05
        result["adjusted_endpoint_alpha"] = alpha
        supported_tradeoff = False
        speed, completion = selection["speed_endpoint"], selection["completion_endpoint"]
        for comparison in result["primary_simultaneous"]:
            if comparison["candidate"] == completion and comparison["reference"] == speed:
                supported_tradeoff = (
                    comparison["C"]["interval"][0] > 0 and comparison["J"]["interval"][0] > 0
                )
            elif comparison["candidate"] == speed and comparison["reference"] == completion:
                supported_tradeoff = (
                    comparison["C"]["interval"][1] < 0 and comparison["J"]["interval"][1] < 0
                )
        result["supported_speed_completion_tradeoff"] = supported_tradeoff
        result["risk_nonincrease_supported"] = [
            {
                "candidate": x["candidate"],
                "reference": x["reference"],
                "supported": x["V"]["interval"][1] <= 0,
            }
            for x in result["primary_simultaneous"]
        ]
    else:
        raise ValueError("Engineering is not a performance estimate")
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "stage": result["stage"],
                "conditions": result["conditions"],
                "selection": result.get("selection"),
                "supported_tradeoff": result.get("supported_speed_completion_tradeoff"),
            }
        )
    )


if __name__ == "__main__":
    main()
