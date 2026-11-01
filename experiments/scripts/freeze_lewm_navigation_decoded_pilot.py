"""Prepare the prospective 2x2 task-metric/controller pilot after v2 is terminal.

JSON and source hashes only. No numerical helpers, caches, models or simulators
are imported. This builder does not launch anything and never overwrites a freeze.
"""

import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARMS = (
    ("native", "original-latent-v1"),
    ("fixed_geometry_progress", "original-latent-v1"),
    ("decoded_native", "decoded-current-target-v1"),
    ("decoded_fixed_geometry_progress", "decoded-current-target-v1"),
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def freeze(base_path, terminal_state_path, out, *, state_remote_path, namespace, seed_base):
    base = json.loads(base_path.read_text())
    state = json.loads(terminal_state_path.read_text())
    if state.get("phase") != "complete" or state.get("active") is not None:
        raise ValueError("The existing frozen v2 architecture pilot must be terminal first")
    expected = {job["job_id"] for job in base["jobs"]}
    if set(state.get("jobs", {})) != expected or len(expected) != 16:
        raise ValueError("All sixteen original v2 starts/outcomes must be retained")
    design = base["evaluation"]
    if (
        design.get("controller_backend") != "observed-target-v2"
        or base.get("parameter_count") != 10
        or design["planner"]["iterations"] != 8
    ):
        raise ValueError("Use the frozen ten-gene v2 controller and SMALL planner")
    native = [job for job in base["jobs"] if job["method"] == "native"]
    if len(native) != 8 or [job["case"]["case_id"] for job in native] != base["pilot_case_ids"]:
        raise ValueError("Preserve the already selected balanced eight-case development panel")
    protocol = copy.deepcopy(base)
    protocol.update(
        schema="lewm-navigation-comparison-v2",
        name="lewm-navigation-decoded-current-target-pilot-v3",
        frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        output_namespace=namespace,
        purpose="Development-only paired 2x2 metric/controller pilot, without optimizer updates",
        parent_protocol={"path": str(base_path.relative_to(ROOT)), "sha256": digest(base_path)},
        parent_terminal_state={
            "path": state_remote_path,
            "sha256": digest(terminal_state_path),
            "phase": "complete",
            "retained_episode_starts": 16,
            "purpose": "Eligibility proof, not a final comparison or hidden discarded allocation",
        },
        maximum_episode_starts=32,
        maximum_native_resets=32,
        maximum_environment_actions=4800,
        optimizer_updates=0,
    )
    design = protocol["evaluation"]
    design.update(
        runtime_backend="navigation-campaign-v3",
        task_cost_backend="per-case-explicit",
        task_cost_backends=["original-latent-v1", "decoded-current-target-v1"],
    )
    design["asset_sha256"][state_remote_path] = digest(terminal_state_path)
    fixed = base["controller_static_parameters"]["fixed_geometry_progress"]
    protocol["jobs"] = []
    for index, original in enumerate(native):
        seed = seed_base + index
        for method, backend in ARMS:
            job = copy.deepcopy(original)
            job.update(
                job_id=method + "_" + original["case"]["case_id"] + "_s" + str(seed),
                method=method,
                parameters=copy.deepcopy(fixed),
                environment_seed=seed,
                policy_seed=seed,
                max_actions=150,
            )
            job["case"].update(seed=seed, task_cost_backend=backend)
            protocol["jobs"].append(job)
    protocol["controller_static_parameters"].update(
        decoded_fixed_geometry_progress=copy.deepcopy(fixed)
    )
    protocol.update(
        pairing="Each already selected development case receives all four arms in fixed original-native/original-fixed/decoded-native/decoded-fixed order, with one prospectively fresh environment/planner seed shared across its arms. Original source pixels, all safety constants, clipping and SMALL CEM caps are identical.",
        seed_rule={"base": seed_base, "formula": "base + frozen zero-based case index"},
        task_cost={
            "original-latent-v1": "Unchanged frozen LeWM last-predicted-embedding squared distance to current planning target",
            "decoded-current-target-v1": "One exact original frozen get_cost call, reuse its predicted_emb endpoint, then squared distance between FP64 frozen linear-readout XY of endpoint and CURRENT planning-target embedding. Return original score dtype. No blending, horizon reward, controller coefficient change or additional encoder/predictor query.",
            "selected_target_authority": "Native targets the original observation; fixed v2 targets its selected actual memory successor or admitted original goal. Never replace an evolved/fixed selected target with an always-original target.",
            "current_native_xy": "Unchanged observed body XY enters the causal v2 selector and learned safety head; true query-goal XY stays confined to reset/admission/original-task monitor/common fitness.",
            "readout_domain_caveat": "The readout was qualified on real observed embeddings, not autoregressive predictions. This pilot tests the full resulting behavior and cannot assume decoded prediction calibration.",
            "legacy_planner_diagnostics": "Current-cost/improvement diagnostic features remain the old latent metric; adaptive planning is disabled in this frozen SMALL recipe.",
        },
        interpretation="A useful decoded task metric would establish a shared architectural lever only. The exact v2 ten-gene family remains common to fixed/CMA/RS/ordinary-learning work; this no-update pilot establishes no evolutionary efficacy or optimizer advantage.",
        comparison_limits="Development cases have already informed architecture. No final/selection query episode is introduced. Report every arm's safe completion, violations, unfinished episodes, engagement, actual action/model/safety/readout work and elapsed time; equal caps do not imply equal consumption.",
        final_evidence="V3 preserves initial controller.json and writes controller-final.json, refusal-history.json and decoded-task-cost.json in finalization, with hashes in outer evidence. Counts preserve attempted/successful projection and candidate rows, including errors.",
    )
    additions = (
        "experiments/helpers/lewm_navigation_decoded_cost.py",
        "experiments/scripts/run_lewm_navigation_campaign_v3.py",
        "experiments/scripts/freeze_lewm_navigation_decoded_pilot.py",
        "experiments/tests/test_lewm_navigation_decoded_cost.py",
    )
    protocol["source_files"] = sorted(set(protocol["source_files"]) | set(additions))
    protocol["source_sha256"] = {name: digest(ROOT / name) for name in protocol["source_files"]}
    if out.exists():
        raise FileExistsError("Never overwrite a prospective protocol")
    out.write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    return {"path": str(out), "sha256": digest(out), "episode_starts": 32, "action_cap": 4800}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--v2-terminal-state", type=Path, required=True)
    parser.add_argument(
        "--state-remote-path", default="results/navigation-progress-target-pilot-v2/state.json"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "configs/lewm_campaign_20261031/navigation-decoded-target-pilot-v3.json",
    )
    parser.add_argument("--namespace", default="results/navigation-decoded-target-pilot-v3")
    parser.add_argument("--seed-base", type=int, default=319000)
    args = parser.parse_args()
    print(
        json.dumps(
            freeze(
                args.base.resolve(),
                args.v2_terminal_state,
                args.out,
                state_remote_path=args.state_remote_path,
                namespace=args.namespace,
                seed_base=args.seed_base,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
