"""Freeze all final comparisons after selection, preparation and gallery review.

Standard-library JSON/hash orchestration only. This builder never decodes arrays,
images or checkpoints, loads a model/environment, reads an outcome, or starts a job.
The optional decoded-native controls are decided jointly before any final outcome.
"""

import argparse
import collections
import copy
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("native", "strong_anchor", "tuned_fixed", "cma", "random", "reinforce")
METHODS = {
    "native": "native",
    "strong_anchor": "fixed_geometry_progress",
    "tuned_fixed": "fixed_geometry_progress",
    "cma": "cma",
    "random": "random",
    "reinforce": "reinforce",
}
HOSTS = {
    0: ("navigation", "vukovc3t4o1dui"),
    1: ("pusht", "wq4ar2qx2ff6r4"),
    2: ("navigation", "vukovc3t4o1dui"),
}
OUTCOME_MARKERS = {
    "state.json",
    "result.json",
    "trace.npz",
    "selector-history.json",
    "progress-journal.jsonl",
    "replans.json",
    "cost-history.json",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def reference(path):
    path = Path(path).resolve()
    try:
        name = str(path.relative_to(ROOT))
    except ValueError:
        name = str(path)
    return {"path": name, "sha256": digest(path)}


def parameters(values):
    require(len(values) == 10, "Exactly ten selected genes required")
    result = [float(value) for value in values]
    require(
        all(math.isfinite(value) and -6 <= value <= 6 for value in result),
        "Selected genes outside frozen shared bounds",
    )
    return result


def fp32(values):
    return tuple(struct.unpack("!f", struct.pack("!f", value))[0] for value in parameters(values))


def check_sources(expected):
    for name, fingerprint in expected.items():
        require(digest(ROOT / name) == fingerprint, "Frozen source changed: " + name)


def opening_locks(path):
    opening = load(path)
    require(
        opening.get("schema") == "lewm-navigation-final-opening-lock-v1", "Opening lock required"
    )
    require(
        opening.get("all_three_repeat_selections_fixed") is True
        and opening.get("best_repeat_selection_forbidden") is True,
        "All three repeat selections must be fixed",
    )
    require(opening.get("final_pixels_read") == 0, "Opening must precede final pixels")
    refs = opening.get("selection_locks", [])
    require(len(refs) == 3, "Three selection references required")
    locks = {}
    for ref in refs:
        lock_path = ROOT / ref["path"]
        require(digest(lock_path) == ref["sha256"], "Selection lock bytes changed")
        lock = load(lock_path)
        require(
            lock.get("schema") == "lewm-navigation-selection-lock-v1"
            and lock.get("final_pixels_read") == 0,
            "Invalid pre-final selection lock",
        )
        repeat = lock["repeat_id"]
        require(
            repeat in HOSTS and repeat not in locks and repeat == ref["repeat_id"],
            "Repeat omitted or duplicated",
        )
        require(
            lock["host"] == ref["host"]
            and (lock["host"]["host"], lock["host"]["pod_id"]) == HOSTS[repeat],
            "Frozen physical-host assignment changed",
        )
        require(
            lock.get("no_best_repeat_selection") is True and lock.get("denominator_each") == 8,
            "Selection panel/repeat incomplete",
        )
        require(set(lock.get("selected", {})) == set(ARMS), "All six locked arms required")
        require(
            lock["evaluation"]["maximum_episode_actions"] == 50,
            "Final endpoint must remain50actions",
        )
        for arm in ARMS:
            chosen = lock["selected"][arm]
            parameters(chosen["parameters"])
            require(
                "selected_is_trained" in chosen
                and "provenance" in chosen
                and "selected_source_arm" in chosen,
                "Selected producer provenance missing",
            )
        check_sources(lock["source_sha256"])
        locks[repeat] = {"lock": lock, "reference": ref}
    require(set(locks) == {0, 1, 2}, "All learner repeats must be retained")
    require(
        all(locks[r]["lock"]["source_sha256"] == locks[0]["lock"]["source_sha256"] for r in locks),
        "Primary source family differs across repeats",
    )
    family_keys = (
        "world_model",
        "motion_model",
        "scalers",
        "memory_sha256",
        "goal_readout",
        "planner",
        "controller_backend",
        "controller_config",
        "progress_backend",
        "body_radius",
        "margin",
        "motion_allowance",
        "precision",
    )
    require(
        all(
            locks[r]["lock"]["evaluation"][key] == locks[0]["lock"]["evaluation"][key]
            for r in locks
            for key in family_keys
        ),
        "Shared primary assets/planner/controller/precision differ across repeats",
    )
    return opening, locks


def reviewed_cases(panel_path, review_path, preparation_path, opening_path):
    panel, review, preparation = [
        load(path) for path in (panel_path, review_path, preparation_path)
    ]
    require(panel.get("phase") == "awaiting_visual_review", "Final preparation must be complete")
    require(
        preparation.get("schema") == "lewm-navigation-final-case-preparation-v1"
        and preparation.get("execution_enabled") is True,
        "Activated final preparation freeze required",
    )
    require(
        preparation["final_opening"]["sha256"] == digest(opening_path),
        "Final preparation used another opening",
    )
    require(
        panel.get("final_opening", {}).get("opening_sha256") == digest(opening_path),
        "Panel opening identity differs",
    )
    freeze = load(panel_path.parent / "source-freeze.json")
    require(
        freeze["protocol_sha256"] == digest(preparation_path)
        and freeze["sources"] == preparation["source_sha256"],
        "Preparation source/protocol differs",
    )
    check_sources(preparation["source_sha256"])
    status = load(panel_path.parent / "status.json")
    require(
        status["phase"] == "awaiting_visual_review" and not status["error"],
        "Technical final preparation cannot proceed",
    )
    require(
        status["controller_episodes"] == status["new_model_queries"] == 0,
        "Final controller/model work already occurred",
    )
    require(
        review.get("approved") is True
        and review.get("images_clean") is True
        and review.get("panel_sha256") == digest(panel_path),
        "Final gallery review must approve this exact panel",
    )
    require(
        review.get("reviewed_utc") and review.get("galleries"),
        "Gallery review identity/time missing",
    )
    for name, expected in review["galleries"].items():
        require(digest(ROOT / name) == expected, "Reviewed gallery bytes changed")
    cases = panel["cases"]
    require(
        len(cases) == 16 and len({case["source_episode"] for case in cases}) == 16,
        "Exactly sixteen independent source cases required",
    )
    counts = collections.Counter()
    for case in cases:
        episode = case["source_episode"]
        condition = ("none", "circle3", "circle5", "circle7")[episode % 4]
        require(
            5000 <= episode <= 5255
            and case["split"] == "heldout_final"
            and case["condition"] == condition,
            "Final case outside frozen admission/stratum",
        )
        require(not case["image_style_flag"], "Final image mismatch requires diagnosis")
        require(case["seed"] == 713000 + episode, "Final planner seed changed")
        if condition == "none":
            require(case["shape"] == {"kind": "none"}, "No-hazard geometry changed")
        else:
            require(
                case["shape"]["kind"] == "island" and case["shape"]["radius"] == int(condition[6:]),
                "Original final radius changed",
            )
        counts[condition] += 1
    require(
        counts == {"none": 4, "circle3": 4, "circle5": 4, "circle7": 4},
        "Final balanced quotas changed",
    )
    require(
        panel["final_pixels_read"] == status["final_pixels_read"] == 32,
        "Final preparation pixel accounting differs",
    )
    return sorted(cases, key=lambda case: case["source_episode"]), preparation


def forbid_final_outcomes(namespace):
    roots = {ROOT / namespace}
    results = ROOT / "results"
    if results.exists():
        roots.update(
            path for path in results.iterdir() if "final" in path.name.lower() and path.is_dir()
        )
    for folder in roots:
        if not folder.exists():
            continue
        require(
            not any(path.is_file() and path.name in OUTCOME_MARKERS for path in folder.rglob("*")),
            "Final outcome artifacts already exist under " + str(folder),
        )


def duplicate_flags(lock):
    groups = collections.defaultdict(list)
    for arm in ARMS:
        if arm != "native":
            groups[fp32(lock["selected"][arm]["parameters"])].append(arm)
    return [arms for arms in groups.values() if len(arms) > 1]


def design_for(lock, cases, asset_pins):
    design = copy.deepcopy(lock["evaluation"])
    require(
        design["controller_backend"] == "observed-target-v2"
        and design["progress_backend"] == "navigation-append-fsync-v1",
        "Primary shared family changed",
    )
    require(
        design["body_radius"] == 7
        and design["margin"] == 3
        and design["motion_allowance"] == 12.405431747436523,
        "Safety contract changed",
    )
    require(
        design["planner"]["checked"] is True and design["maximum_episode_actions"] == 50,
        "Final checker/action endpoint changed",
    )
    require(
        design.get("task_cost_backend", "original-latent-v1") == "original-latent-v1",
        "Primary must retain original LeWM task cost",
    )
    design["task_cost_backend"] = "original-latent-v1"
    design["allowed_source_episodes"] = [case["source_episode"] for case in cases]
    shared = {
        design[key]
        for key in ("world_model", "motion_model", "scalers", "memory_path", "memory_metadata_path")
    }
    shared.add(design["goal_readout"]["path"])
    design["asset_sha256"] = {name: design["asset_sha256"][name] for name in sorted(shared)}
    design["asset_sha256"].update(asset_pins)
    for case in cases:
        design["asset_sha256"][case["input_path"]] = case["input_sha256"]
    return design


def protocol_common(name, namespace, design, cases, preparation, sources):
    return {
        "schema": "lewm-navigation-comparison-v1",
        "name": name,
        "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "output_namespace": namespace,
        "evaluation": design,
        "case_preparation": preparation,
        "source_files": sorted(sources),
        "source_sha256": sources,
        "parameter_count": 10,
        "final_case_ids": [case["case_id"] for case in cases],
        "optimizer_updates": 0,
        "information": "Shared frozen LeWM, observed memory, causal current XY/history, hazard geometry and memory-trained Ridge goal readout. True final-goal XY remains confined to reset/admission/original-goal monitor/common fitness.",
        "endpoint": "Original physical goal distance<16px within50executed native actions, with unchanged body7/margin3/motion allowance12.405431747436523 and learned action checks",
        "analysis": "Sixteen source episodes are the case clusters. Retain all three learner repeats and physical-host blocks; repeated seeds/duplicate parameter vectors and shared cases do not create additional independent cases. Report paired source-case effects, repeat/host variation and all denominators; do not report n48 independent source cases or choose a best repeat.",
        "accounting": "Execute and charge every frozen job, including duplicate-parameter arms/repeats, every reset/action/failure, actual encoder/predictor/safety/readout calls and elapsed time. Common preparation/readout training overhead remains separately declared. Equal hard caps do not imply equal actual consumption.",
    }


def freeze(args):
    namespace_parts = Path(args.namespace).parts
    require(
        len(namespace_parts) == 2
        and namespace_parts[0] == "results"
        and "final" in namespace_parts[1].lower()
        and ".." not in namespace_parts,
        "Use one project results/final-named namespace so all prior final outcomes are detectable",
    )
    opening_path, panel_path, review_path, preparation_path = [
        path.resolve()
        for path in (args.opening_lock, args.panel, args.review, args.preparation_protocol)
    ]
    forbid_final_outcomes(args.namespace)
    out = args.out_dir.resolve()
    require(not out.exists(), "Never overwrite a prospective final comparison")
    opening, locks = opening_locks(opening_path)
    cases, prep_protocol = reviewed_cases(panel_path, review_path, preparation_path, opening_path)
    preparation = {
        "panel_path": reference(panel_path)["path"],
        "panel_sha256": digest(panel_path),
        "review_path": reference(review_path)["path"],
        "review_sha256": digest(review_path),
        "require_approved": True,
        "require_images_clean": True,
    }
    refs = [
        reference(path) for path in (opening_path, panel_path, review_path, preparation_path)
    ] + opening["selection_locks"]
    asset_pins = {ref["path"]: ref["sha256"] for ref in refs}
    primary_sources = {
        **locks[0]["lock"]["source_sha256"],
        **prep_protocol["source_sha256"],
        str(Path(__file__).relative_to(ROOT)): digest(Path(__file__)),
    }
    check_sources(primary_sources)
    planned_files, schedule, duplicate_across = [], [], collections.defaultdict(list)
    for repeat in (0, 1, 2):
        lock, lock_ref = locks[repeat]["lock"], locks[repeat]["reference"]
        namespace = args.namespace + f"/primary/r{repeat}"
        require(not (ROOT / namespace).exists(), "Primary final output already exists")
        protocol = protocol_common(
            f"navigation-final-primary-r{repeat}",
            namespace,
            design_for(lock, cases, asset_pins),
            cases,
            preparation,
            primary_sources,
        )
        duplicate_groups = duplicate_flags(lock)
        jobs = []
        for case in cases:
            for arm in ARMS:
                chosen = lock["selected"][arm]
                vector = parameters(chosen["parameters"])
                duplicates = next((group for group in duplicate_groups if arm in group), [])
                anchor_duplicate = arm not in ("native", "strong_anchor") and fp32(vector) == fp32(
                    lock["selected"]["strong_anchor"]["parameters"]
                )
                jobs.append(
                    {
                        "job_id": arm + "_" + case["case_id"] + "_s" + str(case["seed"]),
                        "arm": arm,
                        "method": METHODS[arm],
                        "case": copy.deepcopy(case),
                        "parameters": vector,
                        "environment_seed": 713000 + case["source_episode"],
                        "policy_seed": 713000 + case["source_episode"],
                        "repeat_id": repeat,
                        "max_actions": 50,
                        "temperature": 1.0,
                        "selection_lock_sha256": lock_ref["sha256"],
                        "selected_is_trained": chosen["selected_is_trained"],
                        "selected_source_arm": chosen["selected_source_arm"],
                        "producer_provenance": copy.deepcopy(chosen["provenance"]),
                        "parameters_sha256": json_digest(vector),
                        "effective_parameters_fp32": list(fp32(vector)),
                        "fp32_effective_parameters_sha256": hashlib.sha256(
                            b"".join(struct.pack("!f", value) for value in fp32(vector))
                        ).hexdigest(),
                        "fp32_effective_policy_duplicate_of_anchor": anchor_duplicate,
                        "effective_selected_is_trained": bool(
                            chosen["selected_is_trained"] and not anchor_duplicate
                        ),
                        "fp32_duplicate_controller_arms": [
                            value for value in duplicates if value != arm
                        ],
                        "parameters_ignored_by_native": arm == "native",
                    }
                )
        for arm in ARMS:
            if arm != "native":
                duplicate_across[fp32(lock["selected"][arm]["parameters"])].append(
                    f"r{repeat}/{arm}"
                )
        protocol.update(
            jobs=jobs,
            repeat_id=repeat,
            host=lock["host"],
            entrypoint="experiments/scripts/run_lewm_navigation_campaign_v2.py",
            maximum_episode_starts=96,
            maximum_native_resets=96,
            maximum_environment_actions=4800,
            selection_lock=lock_ref,
            selected_nominees=lock["selected"],
            fp32_duplicate_parameter_groups=duplicate_groups,
            duplicate_policy="All duplicate arms still execute and are charged; no policy-equivalence claim between native and retrieval solely from matching unused genotype fields",
            arm_order=list(ARMS),
            pairing="Ascending source episode, then all six arms in the frozen order; same713000+episode seed across arms/repeats and exact host-local native/strong/tuned-fixed controls",
            interpretation="Primary final optimizer comparison. Readout/locality architecture benefit is shared; no evolutionary credit for selecting an untrained anchor or tying duplicate parameters.",
        )
        filename = f"primary-r{repeat}.json"
        planned_files.append((filename, protocol))
        schedule.append(
            {
                "kind": "primary",
                "repeat_id": repeat,
                "host": lock["host"],
                "protocol": filename,
                "entrypoint": protocol["entrypoint"],
                "output_namespace": namespace,
            }
        )
    aux_source_ref = None
    if args.with_aux_decoded_native:
        require(
            args.aux_source_protocol is not None,
            "Prospectively frozen v3 source protocol required for auxiliary controls",
        )
        aux_path = args.aux_source_protocol.resolve()
        aux = load(aux_path)
        require(
            aux["evaluation"].get("runtime_backend") == "navigation-campaign-v3",
            "Auxiliary source must pin v3 runtime",
        )
        require(
            "experiments/helpers/lewm_navigation_decoded_cost.py" in aux["source_sha256"]
            and "experiments/scripts/run_lewm_navigation_campaign_v3.py" in aux["source_sha256"],
            "Decoded transitive source pins missing",
        )
        check_sources(aux["source_sha256"])
        aux_source_ref = reference(aux_path)
        software = aux.get("software_qualification")
        require(
            isinstance(software, dict) and software.get("actual_passed_tests") == 9,
            "Auxiliary source requires its corrected nine-test software qualification",
        )
        require(
            aux["evaluation"]["asset_sha256"].get(software["path"]) == software["sha256"],
            "Auxiliary software qualification identity differs",
        )
        aux_sources = {**primary_sources, **aux["source_sha256"]}
        require(
            all(aux_sources[name] == expected for name, expected in primary_sources.items()),
            "Auxiliary source cannot alter a primary pinned dependency",
        )
        for repeat in (0, 1):
            lock = locks[repeat]["lock"]
            host = lock["host"]["host"]
            namespace = args.namespace + "/aux-decoded-native/" + host
            require(not (ROOT / namespace).exists(), "Auxiliary final output already exists")
            design = design_for(
                lock,
                cases,
                {
                    **asset_pins,
                    aux_source_ref["path"]: aux_source_ref["sha256"],
                    software["path"]: software["sha256"],
                },
            )
            design.update(
                runtime_backend="navigation-campaign-v3",
                task_cost_backend="decoded-current-target-v1",
            )
            protocol = protocol_common(
                "navigation-final-aux-decoded-native-" + host,
                namespace,
                design,
                cases,
                preparation,
                aux_sources,
            )
            jobs = []
            for case in cases:
                aux_case = copy.deepcopy(case)
                aux_case["task_cost_backend"] = "decoded-current-target-v1"
                jobs.append(
                    {
                        "job_id": "decoded_native_" + case["case_id"] + "_s" + str(case["seed"]),
                        "arm": "decoded_native",
                        "method": "decoded_native",
                        "case": aux_case,
                        "parameters": parameters(lock["selected"]["strong_anchor"]["parameters"]),
                        "environment_seed": 713000 + case["source_episode"],
                        "policy_seed": 713000 + case["source_episode"],
                        "repeat_id": repeat,
                        "max_actions": 50,
                        "temperature": 1.0,
                        "parameters_ignored_by_native": True,
                    }
                )
            protocol.update(
                schema="lewm-navigation-comparison-v2",
                jobs=jobs,
                host=lock["host"],
                entrypoint="experiments/scripts/run_lewm_navigation_campaign_v3.py",
                maximum_episode_starts=16,
                maximum_native_resets=16,
                maximum_environment_actions=800,
                prospective_source_protocol=aux_source_ref,
                interpretation="Separate shared-architecture auxiliary control, frozen before primary outcomes. Native plans towards the ORIGINAL goal observation with decoded-current-target cost; no retrieval/evolved selector and no evolutionary credit.",
                pairing="One sixteen-case block per physical host; identical original goal,50-action cap,713000+episode CEM seeds, SMALL planner, frozen Ridge and learned safety as that host's primary native reference",
            )
            filename = f"aux-decoded-native-{host}.json"
            planned_files.append((filename, protocol))
            schedule.append(
                {
                    "kind": "auxiliary",
                    "host": lock["host"],
                    "protocol": filename,
                    "entrypoint": protocol["entrypoint"],
                    "output_namespace": namespace,
                }
            )
    require(
        not args.aux_source_protocol or args.with_aux_decoded_native,
        "Auxiliary source supplied without a prospective enable decision",
    )
    forbid_final_outcomes(args.namespace)
    out.mkdir(parents=True)
    files = []
    for filename, protocol in planned_files:
        path = out / filename
        path.write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
        files.append(reference(path))
    manifest = {
        "schema": "lewm-navigation-final-comparison-freeze-v1",
        "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "builder_sha256": digest(Path(__file__)),
        "opening_lock": reference(opening_path),
        "selection_locks": opening["selection_locks"],
        "case_preparation": preparation,
        "preparation_protocol": reference(preparation_path),
        "source_case_clusters": 16,
        "all_three_repeats_retained": True,
        "best_repeat_selection_forbidden": True,
        "primary_episode_starts": 288,
        "primary_maximum_environment_actions": 14400,
        "auxiliary_decoded_native_enabled_before_primary_outcomes": args.with_aux_decoded_native,
        "auxiliary_source_protocol": aux_source_ref,
        "auxiliary_episode_starts": 32 if args.with_aux_decoded_native else 0,
        "auxiliary_maximum_environment_actions": 1600 if args.with_aux_decoded_native else 0,
        "fp32_duplicate_groups_across_repeats": [
            group for group in duplicate_across.values() if len(group) > 1
        ],
        "duplicate_execution_rule": "Every listed job executes and is charged; duplicate weights/cases/seeds are recorded, not treated as additional independent cases",
        "analysis": "Cluster paired analyses by the16source episodes; retain learner-repeat and physical-host blocks. Report safe completion/violation/unfinished/engagement with denominators, actual actions/model/readout work and elapsed time; architecture controls do not establish evolutionary advantage.",
        "outcomes_present_at_freeze": False,
        "outcome_files_read": 0,
        "final_arrays_or_images_decoded_by_builder": 0,
        "workers_started": 0,
        "schedule": schedule,
        "files": files,
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return {
        "manifest": reference(manifest_path),
        "primary_protocols": 3,
        "auxiliary_protocols": 2 if args.with_aux_decoded_native else 0,
        "workers_started": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opening-lock", type=Path, required=True)
    parser.add_argument(
        "--panel", type=Path, default=ROOT / "results/navigation-final-cases-v1/panel.json"
    )
    parser.add_argument(
        "--review", type=Path, default=ROOT / "results/navigation-final-cases-v1/visual-review.json"
    )
    parser.add_argument(
        "--preparation-protocol",
        type=Path,
        default=ROOT / "configs/lewm_campaign_20261031/navigation-final-cases-v1.json",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--namespace", default="results/navigation-final-comparison-v1")
    parser.add_argument("--with-aux-decoded-native", action="store_true")
    parser.add_argument("--aux-source-protocol", type=Path)
    args = parser.parse_args()
    print(json.dumps(freeze(args), indent=2))


if __name__ == "__main__":
    main()
