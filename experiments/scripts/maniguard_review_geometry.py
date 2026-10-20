"""Reproduce the frozen R10 feature sample without reading independent checking.

The reference report fixes membership and input hashes. A later queue can confirm
roles/completion, but cannot silently add newer episodes or change that sample.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

from helpers.maniguard_review_sensitivity import ROOT, matrix

ALLOWED = {"development", "shared_initial"}


def geometry(features, prompts, episodes):
    features = np.asarray(features, dtype=np.float64)
    centered = features - features.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    std = features.std(axis=0)
    return {
        "feature_rows": len(features), "episodes": episodes,
        "distinct_prompt_hashes": len(set(prompts)),
        "distinct_projected_instruction_vectors": len(np.unique(features[:, 36:44], axis=0)),
        "instruction_feature_std_max": float(std[36:44].max()),
        "near_constant_columns_std_below_1e_6": np.flatnonzero(std < 1e-6).tolist(),
        "centered_singular_values": singular.tolist(),
        "centered_relative_rank": {str(t): int(np.sum(singular > singular[0] * t))
                                   for t in (.01, .001, .0001)},
    }


def _same(expected, actual):
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and expected.keys() == actual.keys()
                and all(_same(v, actual[k]) for k, v in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(expected) == len(actual)
                and all(_same(x, y) for x, y in zip(expected, actual, strict=True)))
    if isinstance(expected, float):
        return bool(np.isclose(expected, actual, rtol=1e-10, atol=1e-12))
    return expected == actual


def reproduce(run, reference):
    run, reference = Path(run).resolve(), Path(reference).resolve()
    reference_bytes = reference.read_bytes()
    frozen = json.loads(reference_bytes)
    if frozen.get("schema") != "maniguard-review-feature-geometry-v1":
        raise ValueError("Require the original frozen R10 report")
    queue_bytes = (run / "queue.json").read_bytes()
    attempts = {row["id"]: row for row in json.loads(queue_bytes)["attempts"]}
    inputs = frozen["inputs"]
    if not inputs or len({row["attempt"] for row in inputs}) != len(inputs):
        raise ValueError("Invalid frozen input membership")
    # Validate every role before resolving or opening any episode output path.
    for item in inputs:
        attempt = attempts.get(item["attempt"])
        if (attempt is None or attempt["stage"] not in ALLOWED
                or attempt["status"] != "completed"):
            raise ValueError("Reference includes excluded or incomplete episode")
    groups, initial, provenance = {}, {}, []
    for item in inputs:
        attempt = attempts[item["attempt"]]
        output = Path(attempt["output"])
        output = output if output.is_absolute() else ROOT / output
        raw = (output / "trace.jsonl").read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["trace_sha256"]:
            raise ValueError("Frozen trace hash changed")
        job = json.loads((output / "job.json").read_text())
        receipt_bytes = (output / "verified-receipt.json").read_bytes()
        receipt = json.loads(receipt_bytes)
        if (receipt["valid"] is not True or receipt["full_episode"] is not True
                or job["family"] != item["family"]):
            raise ValueError("Frozen development input lacks matching full verification")
        rows, prompts = [], set()
        for line in raw.splitlines():
            row = json.loads(line)
            if row["type"] == "latent_features":
                rows.append(matrix(row["features"], (44,), "factual features"))
                prompts.add(row["context"]["text_sha256"])
        if len(rows) != item["rows"] or prompts != {item["prompt_sha256"]}:
            raise ValueError("Frozen feature rows or prompt hash changed")
        values = np.asarray(rows, dtype=np.float64)
        family = item["family"]
        group = groups.setdefault(family, {"features": [], "prompts": []})
        group["features"].append(values)
        group["prompts"].extend(prompts)
        provenance.append({"attempt": item["attempt"], "stage": attempt["stage"],
                           "scene": job["scene"],
                           "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest()})
        if attempt["stage"] == "shared_initial":
            group = initial.setdefault(family, {"features": [], "prompts": [], "tasks": set()})
            group["features"].append(values)
            group["prompts"].extend(prompts)
            group["tasks"].add(job["scene"].split("/")[0])
    families = {name: geometry(np.concatenate(g["features"]), g["prompts"], len(g["features"]))
                for name, g in groups.items()}
    if not _same(frozen["families"], families):
        raise ValueError("Original feature geometry was not reproduced")
    instruction = {}
    for name, group in initial.items():
        text = np.concatenate(group["features"])[:, 36:44]
        singular = np.linalg.svd(text - text.mean(axis=0), compute_uv=False)
        instruction[name] = {
            "episodes": len(group["features"]), "base_tasks": sorted(group["tasks"]),
            "distinct_prompt_hashes": len(set(group["prompts"])),
            "distinct_projected_instruction_vectors": len(np.unique(text, axis=0)),
            "centered_instruction_singular_values": singular.tolist(),
            "centered_instruction_rank_relative_1e_6": int(np.sum(singular > singular[0] * 1e-6)),
        }
    return {
        "schema": "maniguard-review-feature-geometry-reproduction-v1",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "loader_source_sha256": hashlib.sha256(
            (ROOT / "experiments/helpers/maniguard_review_sensitivity.py").read_bytes()
        ).hexdigest(),
        "reference_report_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "original_queue_sha256": frozen["queue_sha256"],
        "role_check_queue_sha256": hashlib.sha256(queue_bytes).hexdigest(),
        "inputs": inputs, "input_provenance": provenance, "families": families,
        "shared_initial_instruction_geometry": instruction,
        "original_geometry_reproduced": True, "numeric_rtol": 1e-10, "numeric_atol": 1e-12,
        "new_simulator_controls": 0, "new_policy_queries": 0, "new_encoder_calls": 0,
        "scope": "Original development/initialization membership and exact traces only. "
                 "Queue hash changes identify role readback, not a changed sample. "
                 "Singular values weight correlated query rows and describe factual input "
                 "geometry, not physical controllability, semantic competence or transfer.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing reports; select a new output path")
    result = reproduce(args.run, args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "original_geometry_reproduced": True}))


if __name__ == "__main__":
    main()
