"""Freeze a JSON-only prospective native/strong-fixed observed-target-v2 pilot.

This orchestration script reads JSON and hashes source bytes. It never imports
experiment helpers, reads numerical arrays, or performs simulator/model work.
"""

import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
READOUT_PATH = "results/navigation-readout-v1/readout.npz"
READOUT_SHA256 = "a9a779c8f20e457cf52758c66423e2bace9a3fe980c61f44e86d25ead8e018c0"
STRONG_FIXED = [0.25, 0.25, 0.1, 0.25, 0.1, 2.0, 0.25, 2.0, 4.0, 1.3862943611198906]
CONTROLLER_CONFIG = {
    "min_target_distance": 6.0,
    "max_target_distance": 30.0,
    "recent_window": 6,
    "recent_scale": 6.0,
    "stall_distance": 3.0,
    "readout_progress": True,
    "direct_goal_locality": True,
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def freeze(base_path, parity_report, output_path, *, parity_remote_path, namespace):
    base = json.loads(base_path.read_text())
    qualification = json.loads(parity_report.read_text())
    if qualification.get("qualified") is not True or len(qualification.get("pairs", [])) != 2:
        raise ValueError("Exact native/fixed old-new persistence parity must qualify first")
    if base["schema"] != "lewm-navigation-comparison-v1" or len(base["pilot_case_ids"]) != 8:
        raise ValueError(
            "The original prospectively selected balanced eight-case panel is required"
        )
    if base["evaluation"]["planner"]["iterations"] != 8:
        raise ValueError("This architecture pilot uses the frozen SMALL planner")
    native_jobs = [row for row in base["jobs"] if row["method"] == "native"]
    if [row["case"]["case_id"] for row in native_jobs] != base["pilot_case_ids"]:
        raise ValueError("Native case ordering must reproduce the original eight-case panel")
    protocol = copy.deepcopy(base)
    protocol.update(
        name="lewm-navigation-progress-target-pilot-20261031-v2",
        frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        output_namespace=namespace,
        purpose="Development-only local-target/persistence/visual-progress architecture pilot; no optimizer updates or evolutionary efficacy claim",
        parent_protocol={"path": str(base_path.relative_to(ROOT)), "sha256": digest(base_path)},
        persistence_parity={
            "path": parity_remote_path,
            "sha256": digest(parity_report),
            "qualified": True,
            "pairs": 2,
            "meaning": "Native and observed-target-v1 exact action/state/target parity under append-fsync persistence, qualified before freezing this pilot",
        },
    )
    evaluation = protocol["evaluation"]
    evaluation.update(
        controller_backend="observed-target-v2",
        progress_backend="navigation-append-fsync-v1",
        controller_config=CONTROLLER_CONFIG.copy(),
        goal_readout={"path": READOUT_PATH, "sha256": READOUT_SHA256},
    )
    evaluation["asset_sha256"].update(
        {
            READOUT_PATH: READOUT_SHA256,
            parity_remote_path: digest(parity_report),
        }
    )
    protocol["jobs"] = []
    for native in native_jobs:
        for method in ("native", "fixed_geometry_progress"):
            job = copy.deepcopy(native)
            job.update(
                job_id=method + "_" + native["case"]["case_id"] + "_s" + str(native["policy_seed"]),
                method=method,
                parameters=STRONG_FIXED.copy(),
            )
            job["case"]["information"] = (
                "True goal XY is only for reset/admission/original task monitoring and common fitness. "
                "All arms have access to the same frozen memory-trained visual readout; "
                "the target selector receives final visual embedding, not true query-goal XY."
            )
            protocol["jobs"].append(job)
    protocol.update(
        maximum_episode_starts=16,
        maximum_native_resets=16,
        maximum_environment_actions=2400,
        optimizer_updates=0,
        parameter_count=10,
        pairing="The same original eight training cases, environment/planner seeds and SMALL planner are paired across projected native and strong fixed geometry/progress retrieval, interleaved in that arm order.",
        controller_static_parameters={
            "fixed_geometry_progress": STRONG_FIXED,
            "origin": "Literal strong fixed helper parameters, all ten nonzero; no outcome tuning or local numerical experiment computation",
        },
        information="Frozen LeWM embeddings, real memory successor targets, causal current body XY/history, hazard geometry and a frozen memory-only Ridge goal-position estimate are available to both arms. The native reference continues to target the original observation. No true query-goal coordinates enter the target selector; original physical distance<16px remains the only success rule.",
        architecture={
            "observed_target_band_px": [6, 30],
            "direct_original_goal": "Available only when frozen readout-estimated current-to-goal distance<=30px; no true goal XY is used for admission",
            "persistent_features": "Last six selected target positions plus previous causal body XY; Gaussian spatial repetition score radius6px and stall multiplier based on displacement<3px",
            "learned_progress": "Signed change in distance to frozen visual-goal estimate, divided by30px and clipped[-1,1]",
            "weights": "One fixed source-match weight, nine signed linear genes clamped[-6,6], exp(clamped tenth) direct-goal weight; all methods share exactly this family",
            "safety": "Unchanged body7px/margin3px/learned motion allowance12.405431747436523; successor clearance admission and learned action checks remain authoritative",
            "empty_support": "NoSafeAction; no unsafe or nonlocal fallback action",
        },
        readout_qualification={
            "source_memory_episodes": list(range(128, 176)),
            "train": [128, 159],
            "calibration": [160, 167],
            "test": [168, 175],
            "selected_ridge_lambda": 0.001,
            "test_median_px": 2.70,
            "test_p90_px": 6.64,
            "test_p95_px": 8.39,
            "test_rmse_px": 8.30,
            "test_max_px": 66.56,
            "arithmetic": "Exact raw coefficient/intercept formula evaluated in float64; input embeddings remain frozen FP32 encoder outputs",
            "limitation": "Qualified on disjoint memory episodes, with large individual outliers; shared supervised representation benefit is architectural and cannot be attributed to evolution",
        },
        comparison_limits="This is a paired architecture pilot. Any gain from locality, causal persistence or the supervised readout belongs to the shared architecture. Evolutionary advantage still requires matched native/strong-fixed/random-search/ordinary-learning comparisons with the same information and actual-work accounting. No extra ablation is authorized by this freeze.",
        accounting="Count every native reset/action and failed attempt, actual encoder/predictor/safety/readout calls before execution, final resources and elapsed time. Durable append-fsync journals preserve attempted work. Memory/readout preparation costs remain separately declared shared overhead.",
    )
    extra_sources = [
        "experiments/helpers/lewm_navigation_retrieval_v2.py",
        "experiments/helpers/lewm_progress_journal.py",
        "experiments/scripts/run_lewm_navigation_campaign_v2.py",
        "experiments/scripts/qualify_lewm_goal_readout.py",
        "experiments/scripts/freeze_lewm_navigation_progress_pilot.py",
    ]
    protocol["source_files"] = sorted(set(protocol["source_files"] + extra_sources))
    protocol["source_sha256"] = {name: digest(ROOT / name) for name in protocol["source_files"]}
    if output_path.exists():
        raise FileExistsError("Frozen protocols are not overwritten")
    output_path.write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    return {"protocol": str(output_path), "sha256": digest(output_path), "jobs": 16}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        default=ROOT / "configs/lewm_campaign_20261031/navigation-target-pilot-small-v1.json",
    )
    parser.add_argument("--parity-report", type=Path, required=True)
    parser.add_argument(
        "--parity-remote-path",
        default="results/navigation-journal-parity-v1/qualification.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "configs/lewm_campaign_20261031/navigation-progress-target-pilot-v2.json",
    )
    parser.add_argument("--namespace", default="results/navigation-progress-target-pilot-v2")
    args = parser.parse_args()
    print(
        json.dumps(
            freeze(
                args.base.resolve(),
                args.parity_report,
                args.out,
                parity_remote_path=args.parity_remote_path,
                namespace=args.namespace,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
