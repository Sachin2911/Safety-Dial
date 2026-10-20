"""Descriptive paired outcomes after the complete, sealed physical comparison.

This consumes the existing physical/design verification. It never reads episode
directories, selects a finalist, changes fitness or infers optimizer superiority.
"""

import itertools
import math
import re

from helpers.maniguard_latent import outcome_score

CONDITIONS = ("native", "fixed", "ea", "random")
CATEGORIES = ("safe_completion", "unsafe_completion", "safe_unfinished", "unsafe_unfinished")
BOOLEANS = ("requested_completion", "benchmark_completion", "violation",
            "ever_contacted_task_object", "ever_grasped_task_object")


def _require(value, message):
    if not value:
        raise ValueError(message)


def _category(completed, violation):
    return ("unsafe_" if violation else "safe_") + ("completion" if completed else "unfinished")


def _safe(row):
    return row["requested_completion"] and not row["violation"]


def summarize(report, design, protocol, seal, *, protocol_sha256):
    """Require the complete four-block export and bind every row to its frozen plan."""
    _require(report.get("physical_comparison_complete") is True,
             "Finish the physical comparison before reviewing independent outcomes")
    _require(design.get("physical_design_complete") is True
             and design.get("errors") == [] and design.get("pending") == []
             and design.get("verified_full_jobs") == design.get("expected_full_jobs") == 60,
             "Successful complete 60-job design verification is required")
    _require(re.fullmatch(r"[0-9a-f]{64}", protocol_sha256) is not None
             and design.get("protocol_sha256") == seal.get("protocol_sha256") == protocol_sha256,
             "Design, seal and actual frozen protocol bytes must agree")
    _require(protocol.get("status") == "capacity_selected_frozen"
             and protocol.get("workers") == 2 and len(protocol["audit_blocks"]) == 4,
             "Require the original frozen two-worker four-block protocol")
    observed = report["accounting"]["observed_epoch"]
    _require(type(observed) in (int, float) and math.isfinite(observed)
             and observed == design.get("report_observed_epoch"),
             "Design verification must describe this exact dated export")
    _require(set(seal["finalists"]) == set(CONDITIONS), "Seal all four original conditions")
    for finalist in seal["finalists"].values():
        _require(isinstance(finalist.get("candidate_id"), str)
                 and re.fullmatch(r"[0-9a-f]{64}", finalist.get("sha256", "")) is not None,
                 "Require explicit sealed candidate and genotype identities")

    attempts = {row["attempt"]: row for row in report["attempts"]}
    _require(len(attempts) == len(report["attempts"])
             and sum(row.get("valid_full_episode") is True for row in attempts.values()) == 60,
             "The dated export must retain all sixty distinct valid full episodes")
    indexed = {}
    for row in report["audit"]:
        _require(attempts.get(row["attempt"]) == row,
                 "Audit row differs from the export's verified physical attempt")
        block, condition = row["block"], row["condition"]
        _require(type(block) is int and 0 <= block < 4 and condition in CONDITIONS,
                 "Unexpected independent block or condition")
        key = (block, condition)
        _require(key not in indexed, "Duplicate independent block-condition")
        case = protocol["audit_blocks"][block]
        _require(set(case["order"]) == set(CONDITIONS) and len(case["order"]) == 4,
                 "Every frozen block needs its four declared conditions")
        _require(row["stage"] == "independent_audit" and row["status"] == "completed"
                 and row["valid_full_episode"] is True
                 and row["family"] == protocol["family"]
                 and (row["scene"], row["seed"]) == (case["scene"], case["seed"])
                 and row["worker"] == case["order"].index(condition) % 2
                 and row["controller_sha256"] == seal["finalists"][condition]["sha256"],
                 "Independent outcome differs from its sealed case/controller/worker")
        _require(all(type(row[name]) is bool for name in BOOLEANS),
                 "Complete factual outcome and engagement Booleans are required")
        completed, violation = row["requested_completion"], row["violation"]
        _require(type(row["score"]) is int and row["score"] == outcome_score(completed, violation)
                 and row["requested_category"] == _category(completed, violation)
                 and row["benchmark_category"] == _category(row["benchmark_completion"], violation)
                 and (not completed or row["benchmark_completion"]),
                 "Saved categories or utility contradict the primary outcomes")
        indexed[key] = row
    _require(set(indexed) == set(itertools.product(range(4), CONDITIONS)),
             "Retain all sixteen independent block-condition outcomes")

    counts = {}
    for condition in CONDITIONS:
        rows = [indexed[block, condition] for block in range(4)]
        counts[condition] = {
            **seal["finalists"][condition], "episodes": len(rows),
            **{category: sum(row["requested_category"] == category for row in rows)
               for category in CATEGORIES},
            "violations": sum(row["violation"] for row in rows),
            "raw_completions": sum(row["benchmark_completion"] for row in rows),
            "contacted_episodes": sum(row["ever_contacted_task_object"] for row in rows),
            "grasped_episodes": sum(row["ever_grasped_task_object"] for row in rows),
            "mean_utility": sum(row["score"] for row in rows) / len(rows),
        }
    pairs = []
    for reference, candidate in itertools.combinations(CONDITIONS, 2):
        differences = []
        for block in range(4):
            first, second = indexed[block, reference], indexed[block, candidate]
            differences.append({
                "block": block, "scene": second["scene"], "seed": second["seed"],
                "utility": second["score"] - first["score"],
                "safe_completion": int(_safe(second)) - int(_safe(first)),
                "violation": int(second["violation"]) - int(first["violation"]),
            })
        pairs.append({
            "reference": reference, "candidate": candidate,
            "same_genotype": counts[reference]["sha256"] == counts[candidate]["sha256"],
            "direction": "candidate minus reference", "blocks": differences,
            "mean_difference": {name: sum(row[name] for row in differences) / 4
                                for name in ("utility", "safe_completion", "violation")},
        })
    return {
        "schema": "maniguard-review-final-paired-description-v1",
        "protocol_sha256": protocol_sha256, "report_observed_epoch": observed,
        "conditions": counts, "paired_contrasts": pairs,
        "episodes": [{key: indexed[block, condition][key] for key in (
            "attempt", "block", "condition", "scene", "seed", "worker", "controller_sha256",
            *BOOLEANS, "requested_category", "benchmark_category", "score",
        )} for block, condition in itertools.product(range(4), CONDITIONS)],
        "scope": "Descriptive outcomes on four frozen case/seed blocks with one optimizer run. "
                 "Consumes complete physical/design verification; does not repeat it. All six "
                 "contrasts are reported, without significance tests or a selected winner. "
                 "Zero observed violations is not a zero underlying rate or a safety guarantee. "
                 "Global interactions, failed attempts and allocation costs remain in the input "
                 "export. These checking outcomes cannot tune the follow-up method.",
    }
