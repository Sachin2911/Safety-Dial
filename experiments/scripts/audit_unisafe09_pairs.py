"""Describe paired-start agreement without excluding or selecting any episode."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "experiments/scripts")]
from helpers.unisafe09_infra import atomic_json, utc
from helpers.unisafe09_protocol import digest
from validate_unisafe09 import paired_prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "validation", "root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text())
    validation = json.loads(args.validation.read_text())
    if not validation["passed"] or validation["freeze_sha256"] != digest(args.freeze):
        raise ValueError("A passed validation of this exact freeze is required")
    jobs = {job["id"] for job in freeze["jobs"]}
    rows = {row["job"]["id"]: row for row in validation["episodes"]}
    if len(rows) != len(validation["episodes"]) or set(rows) != jobs:
        raise ValueError("The complete validated panel is required")
    if freeze["stage"] not in {"B", "C"}:
        raise ValueError("This audit describes performance-stage pairs only")
    conditions = sorted({row["condition"] for row in rows.values()})
    # Native is a descriptive anchor; this creates no new primary outcome contrast.
    contrasts = [(condition, "d00_g100") for condition in conditions if condition != "d00_g100"]
    for pair in (freeze.get("selection") or {}).get("primary_contrasts", []):
        if tuple(pair) not in contrasts:
            contrasts.append(tuple(pair))
    observed = {}
    inputs = {}
    for path in sorted((args.root / "attempts").glob("*/intent.json")):
        intent = json.loads(path.read_text())
        if intent["freeze_sha256"] != digest(args.freeze):
            continue
        folder = path.parent
        if (folder / "invalidation.json").exists():
            continue
        job_id = intent["job"]["id"]
        if job_id not in rows:
            raise ValueError("Unscheduled attempt shares the source freeze")
        summary = json.loads((folder / "summary.json").read_text())
        if summary != rows[job_id]:
            raise ValueError("Current summary differs from validated evidence")
        steps = [json.loads(line) for line in (folder / "steps.jsonl").read_text().splitlines()]
        if len(steps) != summary["controls"]:
            raise ValueError("The current control trace is incomplete")
        key = summary["condition"], summary["seed"]
        if key in observed:
            raise ValueError("Duplicate condition/seed")
        observed[key] = folder, summary, steps
        for name in ("intent.json", "summary.json", "steps.jsonl", "initial_state.npz"):
            inputs[str((folder / name).relative_to(args.root))] = digest(folder / name)
    if len(observed) != len(rows):
        raise ValueError("A validated attempt is missing locally")
    seeds = sorted({row["seed"] for row in rows.values()})
    comparisons = []
    for candidate, reference in contrasts:
        pairs = []
        for seed in seeds:
            result = paired_prefix(observed[candidate, seed], observed[reference, seed])
            changed = result["first_changed_executed_action"]
            result.update(
                seed=seed,
                all_initial_physical_arrays_equal=all(
                    result["initial_physical_arrays_equal"].values()
                ),
                matched_observed_prefix_through_first_changed_action=(
                    result["observed_prefix_controls"] > changed
                    if changed is not None else None
                ),
            )
            pairs.append(result)
        comparisons.append({
            "candidate": candidate, "reference": reference, "pairs": pairs,
            "n": len(pairs),
            "all_initial_arrays_equal_count": sum(
                row["all_initial_physical_arrays_equal"] for row in pairs
            ),
            "observed_first_input_equal_count": sum(
                row["observed_prefix_controls"] > 0 for row in pairs
            ),
            "matched_prefix_through_first_change_count": sum(
                row["matched_observed_prefix_through_first_changed_action"] is True
                for row in pairs
            ),
            "no_changed_action_in_common_trace_count": sum(
                row["first_changed_executed_action"] is None for row in pairs
            ),
        })
    result = {
        "utc": utc(), "stage": freeze["stage"], "comparisons": comparisons,
        "scope": (
            "Descriptive reproducibility audit only. Exact stored physical arrays and "
            "pre-action observations/camera hashes are compared. Matching does not prove "
            "equality of unrecorded simulator state. No pair is excluded, reweighted or "
            "used to change the frozen selection or inferential contrasts."
        ),
        "freeze_sha256": digest(args.freeze),
        "validation_sha256": digest(args.validation),
        "audit_source_sha256": digest(Path(__file__)),
        "paired_prefix_source_sha256": digest(ROOT / "experiments/scripts/validate_unisafe09.py"),
        "input_sha256": inputs,
    }
    atomic_json(args.output, result)
    print(json.dumps({
        "stage": result["stage"],
        "comparisons": [{k: v for k, v in row.items() if k != "pairs"} for row in comparisons],
    }))


if __name__ == "__main__":
    main()
