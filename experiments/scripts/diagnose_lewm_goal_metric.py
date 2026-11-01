"""Inspect saved development latents on a rented host, without new model queries."""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    protocol_path = Path(args.protocol)
    protocol = json.loads(protocol_path.read_text())
    if os.environ.get("RUNPOD_POD_ID") != protocol["execution_pod_id"]:
        raise RuntimeError("This diagnostic may run only on its frozen rented host")
    for name, expected in protocol["source_and_asset_sha256"].items():
        if digest(name) != expected:
            raise RuntimeError("A frozen input changed: " + name)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    readout = json.loads(Path(protocol["readout_coefficients"]).read_text())
    coefficient = np.asarray(readout["raw_feature_coefficient"], dtype=np.float64)
    intercept = np.asarray(readout["raw_feature_intercept"], dtype=np.float64)
    records, inputs = [], {}
    for stage in protocol["saved_stages"]:
        for folder in sorted((Path(stage) / "jobs").glob("*/episode")):
            history_path = folder / "selector-history.json"
            result_path = folder / "result.json"
            history = json.loads(history_path.read_text())
            if not history:
                continue
            result = json.loads(result_path.read_text())
            inputs[str(history_path)] = digest(history_path)
            inputs[str(result_path)] = digest(result_path)
            current = np.asarray([row["current_latent"] for row in history])
            final = np.asarray([row["final_latent"] for row in history])
            actual_xy = np.asarray([row["current_xy"] for row in history])
            original_goal = np.asarray(result["original_goal"])
            goal_estimates = final @ coefficient + intercept
            current_estimates = current @ coefficient + intercept
            actual_distance = np.linalg.norm(actual_xy - original_goal, axis=-1)
            latent_distance = np.mean((current - final) ** 2, axis=-1)
            learned_distance = np.linalg.norm(actual_xy - goal_estimates, axis=-1)

            def change_agreement(score):
                a, b = np.diff(actual_distance), np.diff(score)
                valid = (np.abs(a) > 0.1) & (np.abs(b) > 1e-9)
                return {
                    "eligible_changes": int(valid.sum()),
                    "same_direction_changes": int(((a * b > 0) & valid).sum()),
                }

            records.append(
                {
                    "stage": stage,
                    "job_id": folder.parent.name,
                    "case_id": result["case_id"],
                    "method": result["method"],
                    "queries": len(history),
                    "original_goal": original_goal.tolist(),
                    "estimated_goal_first": goal_estimates[0].tolist(),
                    "goal_estimate_error_px": np.linalg.norm(
                        goal_estimates - original_goal, axis=-1
                    ).tolist(),
                    "current_readout_error_px": np.linalg.norm(
                        current_estimates - actual_xy, axis=-1
                    ).tolist(),
                    "latent_distance_change_agreement": change_agreement(latent_distance),
                    "readout_distance_change_agreement": change_agreement(learned_distance),
                    "true_goal_distances": actual_distance.tolist(),
                    "latent_goal_distances": latent_distance.tolist(),
                    "learned_goal_distances": learned_distance.tolist(),
                }
            )
    report = {
        "protocol_sha256": digest(protocol_path),
        "source_sha256": digest(Path(__file__)),
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "Previously executed development queries only; no final-pool pixels read",
        "new_simulator_interactions": 0,
        "new_encoder_or_predictor_queries": 0,
        "new_cached_affine_readout_rows": sum(row["queries"] * 2 for row in records),
        "saved_input_sha256": inputs,
        "records": records,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for row in records:
        print(
            json.dumps(
                {
                    k: v
                    for k, v in row.items()
                    if k
                    in (
                        "stage",
                        "case_id",
                        "method",
                        "queries",
                        "estimated_goal_first",
                        "latent_distance_change_agreement",
                        "readout_distance_change_agreement",
                    )
                }
                | {"goal_error_px": row["goal_estimate_error_px"][0]}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
