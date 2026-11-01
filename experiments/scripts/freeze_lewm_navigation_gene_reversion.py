"""Freeze one exploratory development-only ten-gene reversion study as JSON.

This metadata builder never imports numerical libraries, opens arrays, starts a
controller, reads selection outcomes or accesses final-case evidence.
"""

import argparse
import collections
import copy
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GROUP = "configs/lewm_campaign_20261031/navigation-budget30-comparison-v1"
STAGE = "results/navigation-budget30-comparison-v1/r0/cma"
NAME = "navigation-budget30-cma-r0-gene-reversion-v1"
GENE_FEATURES = (
    "endpoint_goal_latent",
    "transition_goal_latent",
    "successor_current_latent",
    "successor_goal_latent",
    "successor_offset_div_10",
    "straight_line_margin_deficit",
    "successor_current_xy_div_max_distance",
    "recent_target_similarity_and_stall",
    "estimated_goal_distance_change_div_max_distance",
    "log_weight_of_direct_goal_latent",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ref(path, root):
    return {"path": str(path.relative_to(root)), "sha256": sha(path)}


def freeze(args):
    evidence = args.artifact_root.resolve()
    stage = evidence / STAGE
    original_path = ROOT / GROUP / "training-cma-r0.json"
    control_path = ROOT / GROUP / "training-controls-r0.json"
    original, controls = load(original_path), load(control_path)
    state_path, source_path, nominees_path = (
        stage / "state.json", stage / "source-freeze.json", stage / "nominees.json"
    )
    state, source, nominees = load(state_path), load(source_path), load(nominees_path)
    require(
        state["phase"] == "complete" and state["active"] is None
        and state["optimizer_updates"] == 6 and state["candidate_panels_completed"] == 48
        and not state["unfinalized_job_ids"] and state["action_accounting_exact"],
        "Require the full already-completed repeat0 CMA producer",
    )
    require(
        source["protocol"] == original and source["protocol_sha256"] == hashlib.sha256(
            json.dumps(original, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        and source["identity"] == state["identity"],
        "Producer source/protocol/identity differs",
    )
    trained, anchor = nominees["trained"], nominees["anchor"]
    require(
        trained == state["best_trained"] and anchor == state["anchor"]
        and nominees["selected_on_training"] == trained
        and nominees["selected_is_trained"] is True,
        "Bind the actual completed TRAIN nominee, without development selection",
    )
    require(
        trained["provenance"] == {
            "kind": "trained_candidate", "method": "cma", "repeat_id": 0,
            "generation": 5, "candidate_index": 1, "completed_updates_before_proposal": 5,
        },
        "The fixed first-repeat CMA nominee identity changed",
    )
    require(
        len(trained["parameters"]) == len(anchor["parameters"]) == 10
        and all(
            math.isfinite(value) and -6 <= value <= 6
            for vector in (trained["parameters"], anchor["parameters"]) for value in vector
        ),
        "Require the bounded shared ten-gene family",
    )
    require(
        anchor["parameters"] == original["initial_parameters"]
        and original["repeat_id"] == 0 and original["method"] == "cma"
        and original["allowed_pod_ids"] == ["vukovc3t4o1dui"]
        and original["evaluation"]["maximum_episode_actions"] == 30
        and controls["evaluation"] == original["evaluation"]
        and controls["source_sha256"] == original["source_sha256"],
        "Shared anchor, runtime, endpoint or host contract changed",
    )
    panel = original["training_panel"]
    require(
        len(panel) == 8 and len({case["case_id"] for case in panel}) == 8
        and collections.Counter(case["condition"] for case in panel)
        == {"none": 2, "circle3": 2, "circle5": 2, "circle7": 2},
        "Keep every original balanced training case",
    )
    producer_panel = []
    require(len(trained["job_ids"]) == len(panel), "Nominee lacks its full original panel")
    for index, (job_id, case) in enumerate(zip(trained["job_ids"], panel, strict=True)):
        require(job_id == f"cma_g005_c001_p{index:03}", "Nominee panel job identity changed")
        folder = stage / "jobs" / job_id
        result, started = load(folder / "result.json"), load(folder / "started.json")
        spec = started["job"]
        episode_path = folder / "episode/result.json"
        episode = load(episode_path)
        policy_seed = int.from_bytes(
            hashlib.sha256(f"{original['optimizer_seed']}/{job_id}".encode()).digest()[:4], "big"
        ) % (2**31 - 1)
        require(
            spec["parameters"] == trained["parameters"] and not result["error"]
            and started["identity"] == result["identity"] == state["identity"]
            and started["input_sha256"] == result["input_sha256"]
            and spec["case"] == case and result["case_id"] == case["case_id"]
            and spec["environment_seed"] == episode["environment_seed"] == case["seed"]
            and spec["policy_seed"] == episode["policy_seed"] == policy_seed
            and spec["max_actions"] == episode["maximum_actions"] == 30
            and spec["mode"] == "deterministic" and spec["method"] == "cma"
            and job_id in state["completed_job_ids"] and result["accounting_finalized"]
            and sha(episode_path) == result["extra"]["episode_result_sha256"],
            "Completed nominee panel provenance differs",
        )
        producer_panel.append({
            "job_id": job_id, "result": ref(folder / "result.json", evidence),
            "started": ref(folder / "started.json", evidence),
            "episode_result": ref(episode_path, evidence),
        })
    selection = evidence / "results/navigation-budget30-comparison-v1/r0/selection/state.json"
    require(not selection.exists(), "Freeze this exploratory study before selection outcomes")
    arms = [
        {"arm": "cma_full", "parameters": trained["parameters"], "reverted_gene": None},
        {"arm": "strong_anchor", "parameters": anchor["parameters"], "reverted_gene": None},
    ]
    for gene in range(10):
        vector = list(trained["parameters"])
        vector[gene] = anchor["parameters"][gene]
        arms.append({
            "arm": f"revert_gene_{gene:02}", "parameters": vector,
            "reverted_gene": gene, "gene_feature": GENE_FEATURES[gene],
        })
    jobs = []
    for index, source_case in enumerate(panel):
        case = copy.deepcopy(source_case)
        seed = 623000 + case["source_episode"]
        case["seed"] = seed
        order = arms[index:] + arms[:index]
        for arm in order:
            jobs.append({
                "job_id": f"{arm['arm']}_{case['case_id']}_s{seed}",
                "arm": arm["arm"], "method": "fixed_geometry_progress",
                "parameters": list(arm["parameters"]), "case": copy.deepcopy(case),
                "environment_seed": seed, "policy_seed": seed,
                "repeat_id": 0, "max_actions": 30,
                "ablation_reverted_gene": arm["reverted_gene"],
            })
    require(len(jobs) == 96 and len({job["job_id"] for job in jobs}) == 96, "Complete12x8 study")
    output = ROOT / "configs/lewm_campaign_20261031" / (NAME + ".json")
    require(not output.exists(), "Never overwrite a prospective ablation freeze")
    protocol = {
        "schema": controls["schema"], "name": NAME,
        "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "output_namespace": "results/" + NAME,
        "evaluation": copy.deepcopy(original["evaluation"]),
        "case_preparation": copy.deepcopy(original["case_preparation"]),
        "source_files": list(original["source_files"]),
        "source_sha256": copy.deepcopy(original["source_sha256"]),
        "jobs": jobs, "maximum_episode_starts": 96, "maximum_native_resets": 96,
        "maximum_environment_actions": 2880, "optimizer_updates": 0,
        "host": {"host": "navigation", "pod_id": "vukovc3t4o1dui"}, "repeat_id": 0,
        "exploratory_gene_reversion": {
            "scope": (
                "Development TRAIN mechanism study motivated by the completed repeat0 CMA "
                "training result. It is not a final efficacy endpoint or optimizer-superiority "
                "test and cannot change any nominee, selection, final panel or primary analysis."
            ),
            "producer_protocol": ref(original_path, ROOT),
            "producer_state": ref(state_path, evidence),
            "producer_source_freeze": ref(source_path, evidence),
            "producer_nominees": ref(nominees_path, evidence),
            "producer_identity": state["identity"], "producer_panel": producer_panel,
            "producer_protocol_canonical_json_sha256": source["protocol_sha256"],
            "producer_nominee": trained, "common_anchor": anchor,
            "updates_before_nominee_proposal": 5,
            "selection_state_absent_at_freeze": True,
            "arms": arms,
            "gene_mapping": {
                "fixed_coefficient": "source_current_latent coefficient remains one",
                "genes_0_to_8": "Signed linear coefficients for feature columns1 through9",
                "gene_9": "Log coefficient for direct_goal_latent; runtime exponentiates it",
            },
            "fresh_seed_rule": "623000 + original source_episode, common to all12 arms",
            "order": "Original8-case order; all12 arms per case, cyclic offset by case index",
            "duplicate_vectors_still_executed_and_charged": True,
            "all_cases_arms_and_negative_outcomes_retained": True,
            "new_work_is_separate_from_optimizer_comparison": True,
            "execution_condition": (
                "Root may launch only after main secondaryR0 selection if a safe pre-final "
                "waiting window leaves at least8minutes before the joint-final critical path. "
                "Otherwise retain this protocol clearly unexecuted. This builder launches nothing."
            ),
            "builder": ref(Path(__file__).resolve(), ROOT),
            "selection_outcomes_read": 0, "final_data_read": 0,
            "new_model_simulator_calls_by_builder": 0,
        },
    }
    output.write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    return {"protocol": str(output.relative_to(ROOT)), "sha256": sha(output),
            "jobs": 96, "actions_ceiling": 2880, "launched": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root", type=Path,
        default=ROOT / "outputs/lewm-campaign-20261031/mirror/navigation",
    )
    args = parser.parse_args()
    print(json.dumps(freeze(args), indent=2))


if __name__ == "__main__":
    main()
