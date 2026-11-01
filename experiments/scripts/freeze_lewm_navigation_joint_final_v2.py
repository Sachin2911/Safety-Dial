"""Inactive joint-final orchestration, reusing the immutable primary builders.

Seal both three-repeat families, bind the unchanged shared-image preparation, then
derive joint final profiles from primary JSON templates. No subcommand executes
models, environments, array/image decoding or controller episodes.
"""

import argparse
import copy
import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.lewm_navigation_joint_final_v2 import (
    digest,
    family_opening,
    load_reference,
    require,
    validate_joint,
    validate_joint_preparation,
    validate_relocation,
)
import freeze_lewm_navigation_final_comparison as primary

NEW_SOURCES = (
    "experiments/helpers/lewm_navigation_joint_final_v2.py",
    "experiments/scripts/prepare_lewm_navigation_joint_final_cases_v2.py",
    "experiments/scripts/freeze_lewm_navigation_joint_final_v2.py",
    "experiments/scripts/freeze_lewm_navigation_selection_opening_v2.py",
)


def write(path, value):
    require(not path.exists(), "Never overwrite a joint-final freeze")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return primary.reference(path)


def seal(args):
    primary.forbid_final_outcomes(args.namespace)
    require(
        not (ROOT / "results/navigation-final-cases-v1").exists(),
        "Shared final preparation already began; six-lock readiness cannot be backdated",
    )
    p_ref, s_ref, activation_ref = [
        primary.reference(path)
        for path in (args.primary_opening, args.secondary_opening, args.activation_record)
    ]
    p_open, p_locks = family_opening(p_ref, ROOT, 50)
    s_open, s_locks = family_opening(s_ref, ROOT, 30)
    relocation_ref = primary.reference(args.relocation_record)
    relocation = validate_relocation(relocation_ref, ROOT)
    require(
        s_open["secondary_relocation"] == relocation_ref,
        "Secondary opening uses another relocation record",
    )
    activation = load_reference(activation_ref, ROOT)
    require(
        activation.get("schema") == "lewm-navigation-secondary-study-choice-v1"
        and activation.get("activate_secondary") is True
        and activation.get("all_six_selection_locks_required_before_final") is True
        and activation.get("final_pixels_read") == 0,
        "Secondary root choice must require all six locks",
    )
    require(
        {row["lock"]["host"]["pod_id"] for row in s_locks.values()}
        == {row["lock"]["host"]["pod_id"] for row in p_locks.values()}
        | {relocation["new_host"]["pod_id"]},
        "Secondary must retain both original hosts and the recovery host",
    )
    require(
        all(
            p_locks[r]["lock"]["host"]["pod_id"] == pod
            for r, pod in {0: "vukovc3t4o1dui", 1: "wq4ar2qx2ff6r4", 2: "vukovc3t4o1dui"}.items()
        ),
        "Primary host assignment is immutable",
    )
    require(
        len({row["reference"]["sha256"] for locks in (p_locks, s_locks) for row in locks.values()})
        == 6,
        "Six distinct family/repeat selection locks required",
    )
    value = {
        "schema": "lewm-navigation-joint-final-opening-v2",
        "locked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "primary_opening": p_ref,
        "secondary_opening": s_ref,
        "secondary_activation": activation_ref,
        "secondary_relocation": relocation_ref,
        "primary_selection_locks": p_open["selection_locks"],
        "secondary_selection_locks": s_open["selection_locks"],
        "all_six_repeat_selections_fixed": True,
        "primary_episode_actions": 50,
        "secondary_episode_actions": 30,
        "best_repeat_selection_forbidden": True,
        "final_pixels_read": 0,
        "workers_started": 0,
        "builder_sha256": digest(Path(__file__)),
        "next_stage": "Use the NEW six-lock preparation wrapper, preserving the unchanged primary geometry/template. Both groups are locked before ANY final pixels or primary/secondary final outcome. Never call the old primary-only preparer directly after activating this secondary study.",
    }
    result = write(args.out, value)
    validate_joint(result, ROOT)
    return result


def preparation(args):
    joint_ref = primary.reference(args.joint_opening)
    joint, _, _ = validate_joint(joint_ref, ROOT)
    primary_ref = primary.reference(args.primary_preparation_protocol)
    original = load_reference(primary_ref, ROOT)
    require(
        original["final_opening"]["sha256"] == joint["primary_opening"]["sha256"],
        "Original primary opening differs",
    )
    sources = {**original["source_sha256"], **{name: digest(ROOT / name) for name in NEW_SOURCES}}
    value = {
        "schema": "lewm-navigation-joint-final-preparation-v2",
        "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "execution_enabled": True,
        "joint_opening": joint_ref,
        "primary_preparation_protocol": primary_ref,
        "source_sha256": sources,
        "source_files": sorted(sources),
        "shared_images": "Exact unchanged primary16case admission, radii3/5/7, body7/margin3/allowance12.405431747436523, same source/goal pixels and visual review. Secondary endpoint30 does not change geometry/images or the physical goal.",
        "access": "All six locks and source bytes validated before importing the old preparer, dataset/renderer/model access or output creation. New receipt binds the joint lock before delegation.",
        "final_pixels_read": 0,
        "workers_started": 0,
    }
    validate_joint_preparation(value, ROOT)
    return write(args.out, value)


def selected_jobs(profile, lock, lock_ref, family, cap):
    groups = primary.duplicate_flags(lock)
    for job in profile["jobs"]:
        arm = job["arm"]
        chosen = lock["selected"][arm]
        vector = primary.parameters(chosen["parameters"])
        duplicate = arm not in ("native", "strong_anchor") and primary.fp32(vector) == primary.fp32(
            lock["selected"]["strong_anchor"]["parameters"]
        )
        group = next((value for value in groups if arm in value), [])
        job.update(
            parameters=vector,
            repeat_id=lock["repeat_id"],
            max_actions=cap,
            selection_lock_sha256=lock_ref["sha256"],
            selected_is_trained=chosen["selected_is_trained"],
            selected_source_arm=chosen["selected_source_arm"],
            producer_provenance=copy.deepcopy(chosen["provenance"]),
            parameters_sha256=primary.json_digest(vector),
            effective_parameters_fp32=list(primary.fp32(vector)),
            fp32_effective_parameters_sha256=primary.hashlib.sha256(
                b"".join(primary.struct.pack("!f", v) for v in primary.fp32(vector))
            ).hexdigest(),
            fp32_effective_policy_duplicate_of_anchor=duplicate,
            effective_selected_is_trained=bool(chosen["selected_is_trained"] and not duplicate),
            fp32_duplicate_controller_arms=[value for value in group if value != arm],
        )
    profile.update(
        study_family=family,
        repeat_id=lock["repeat_id"],
        host=lock["host"],
        selection_lock=lock_ref,
        selected_nominees=lock["selected"],
        fp32_duplicate_parameter_groups=groups,
        maximum_environment_actions=96 * cap,
    )


def comparison(args):
    joint_ref = primary.reference(args.joint_opening)
    joint, p_locks, s_locks = validate_joint(joint_ref, ROOT)
    joint_prep_ref = primary.reference(args.joint_preparation_protocol)
    joint_prep = load_reference(joint_prep_ref, ROOT)
    validate_joint_preparation(joint_prep, ROOT)
    require(joint_prep["joint_opening"] == joint_ref, "Preparation used another joint lock")
    receipt_path = args.panel.parent / "joint-preparation.json"
    receipt = primary.load(receipt_path)
    require(
        receipt["protocol_sha256"] == joint_prep_ref["sha256"]
        and receipt["joint_opening"] == joint_ref
        and receipt["all_six_repeat_selections_fixed"] is True,
        "Final images were not prepared under the six-lock gate",
    )
    primary.forbid_final_outcomes(args.namespace)
    out = args.out_dir.resolve()
    require(not out.exists(), "Use a new joint comparison freeze namespace")
    aux_caps = sorted(set(args.aux_endpoints))
    require(all(cap in (30, 50) for cap in aux_caps), "Auxiliary endpoints must be30or50")
    require(
        not aux_caps or args.aux_source_protocol is not None,
        "Frozen source-only decoded protocol required",
    )
    # Immutable primary orchestration generates JSON design templates only. No
    # produced template is executed or edited; final profiles are fresh copies.
    primary.freeze(
        SimpleNamespace(
            opening_lock=ROOT / joint["primary_opening"]["path"],
            panel=args.panel,
            review=args.review,
            preparation_protocol=ROOT / joint_prep["primary_preparation_protocol"]["path"],
            out_dir=out / "primary-design-templates",
            namespace=args.namespace + "-templates",
            with_aux_decoded_native=bool(aux_caps),
            aux_source_protocol=args.aux_source_protocol,
        )
    )
    templates = out / "primary-design-templates"
    common_refs = [
        joint_ref,
        joint_prep_ref,
        primary.reference(receipt_path),
        joint["secondary_relocation"],
        *joint["primary_selection_locks"],
        *joint["secondary_selection_locks"],
    ]
    relocation = validate_relocation(joint["secondary_relocation"], ROOT)
    if "unstarted_proof" in relocation:
        common_refs.append(relocation["unstarted_proof"])
    extra_assets = {ref["path"]: ref["sha256"] for ref in common_refs}
    sources = {name: digest(ROOT / name) for name in NEW_SOURCES}
    files, schedule = [], []
    representatives = {row["lock"]["host"]["pod_id"]: repeat for repeat, row in p_locks.items()}
    for family, cap, locks in (("primary50", 50, p_locks), ("secondary30", 30, s_locks)):
        for repeat in (0, 1, 2):
            lock, lock_ref = locks[repeat]["lock"], locks[repeat]["reference"]
            template_repeat = (
                repeat if cap == 50 else representatives.get(lock["host"]["pod_id"], 0)
            )
            profile = primary.load(templates / f"primary-r{template_repeat}.json")
            for key in (
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
            ):
                require(
                    profile["evaluation"][key] == lock["evaluation"][key],
                    "Joint study shared information/model/planner differs: " + key,
                )
            profile["evaluation"].update(
                execution_pod_id=lock["host"]["pod_id"], maximum_episode_actions=cap
            )
            profile["evaluation"]["asset_sha256"].update(extra_assets)
            profile["source_sha256"].update({**lock["source_sha256"], **sources})
            primary.check_sources(profile["source_sha256"])
            profile["source_files"] = sorted(profile["source_sha256"])
            selected_jobs(profile, lock, lock_ref, family, cap)
            profile.update(
                name=f"navigation-joint-final-{family}-r{repeat}",
                output_namespace=args.namespace + f"/{family}/r{repeat}",
                joint_opening=joint_ref,
                endpoint=f"Unchanged original physical goal<16px within{cap}native actions; same final images/geometry and713000+episode seeds",
                analysis="Sixteen source-case clusters shared across both study families and all six learner repeats. Preserve family-specific30/50 endpoints, host blocks and duplicate policy flags; never pool cases/repeats into independent n96 or select a best family/repeat from final outcomes.",
                interpretation=f"Locked {family} final optimizer comparison at{cap}actions. Preserve this study endpoint and all repeats; no post-final family choice or evolutionary credit for shared architecture/anchor duplicates.",
            )
            path = out / f"{family}-r{repeat}.json"
            files.append(write(path, profile))
            schedule.append(
                {
                    "family": family,
                    "repeat_id": repeat,
                    "host": lock["host"],
                    "protocol": str(path),
                    "entrypoint": profile["entrypoint"],
                    "output_namespace": profile["output_namespace"],
                }
            )
    auxiliary_starts, auxiliary_actions = 0, 0
    for cap in aux_caps:
        host_representatives = {}
        for repeat, row in (p_locks if cap == 50 else s_locks).items():
            host_representatives.setdefault(row["lock"]["host"]["host"], repeat)
        for host, repeat in sorted(host_representatives.items()):
            lock = (p_locks if cap == 50 else s_locks)[repeat]["lock"]
            template_host = host if host in ("navigation", "pusht") else "navigation"
            profile = primary.load(templates / f"aux-decoded-native-{template_host}.json")
            profile["host"] = lock["host"]
            profile["evaluation"].update(
                maximum_episode_actions=cap, execution_pod_id=lock["host"]["pod_id"]
            )
            profile["evaluation"]["asset_sha256"].update(extra_assets)
            profile["source_sha256"].update({**lock["source_sha256"], **sources})
            primary.check_sources(profile["source_sha256"])
            profile["source_files"] = sorted(profile["source_sha256"])
            for job in profile["jobs"]:
                job["max_actions"] = cap
                job["repeat_id"] = repeat
            profile.update(
                name=f"navigation-joint-aux-decoded-native-{cap}-{host}",
                output_namespace=args.namespace + f"/aux-decoded-native{cap}/{host}",
                joint_opening=joint_ref,
                maximum_environment_actions=16 * cap,
                endpoint=f"Original goal within{cap}actions, shared geometry/images/CEM seeds/safety/Ridge; decoded-native shared-architecture control only",
                pairing=f"One16case block per physical host at{cap}actions, same713000+episode CEM seeds and original goal as that endpoint's primary/secondary native controls",
            )
            path = out / f"aux-decoded-native{cap}-{host}.json"
            files.append(write(path, profile))
            auxiliary_starts += 16
            auxiliary_actions += 16 * cap
            schedule.append(
                {
                    "family": f"aux-decoded-native{cap}",
                    "host": profile["host"],
                    "protocol": str(path),
                    "entrypoint": profile["entrypoint"],
                    "output_namespace": profile["output_namespace"],
                }
            )
    manifest = {
        "schema": "lewm-navigation-joint-final-comparison-v2",
        "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "joint_opening": joint_ref,
        "joint_preparation_protocol": joint_prep_ref,
        "all_six_repeat_selections_fixed_before_final_pixels": True,
        "shared_source_case_clusters": 16,
        "primary_action_cap_each_repeat": 4800,
        "secondary_action_cap_each_repeat": 2880,
        "primary_and_secondary_episode_starts": 576,
        "auxiliary_endpoints_frozen_before_outcomes": aux_caps,
        "auxiliary_episode_starts": auxiliary_starts,
        "auxiliary_action_cap": auxiliary_actions,
        "secondary_relocation": joint["secondary_relocation"],
        "primary_physical_host_blocks": 2,
        "secondary_physical_host_blocks": 3,
        "outcome_files_read": 0,
        "workers_started": 0,
        "files": files,
        "schedule": schedule,
        "builder_sha256": digest(Path(__file__)),
        "primary_design_templates": primary.reference(templates / "manifest.json"),
        "primary_design_templates_are_not_an_execution_schedule": True,
        "rule": "All final profiles are jointly frozen before any primary/secondary/auxiliary outcome; retain every selected repeat and its actual lock-derived host. No post-final family/baseline selection.",
    }
    return write(out / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sealer = commands.add_parser("seal")
    sealer.add_argument("--primary-opening", type=Path, required=True)
    sealer.add_argument("--secondary-opening", type=Path, required=True)
    sealer.add_argument("--activation-record", type=Path, required=True)
    sealer.add_argument("--relocation-record", type=Path, required=True)
    sealer.add_argument("--namespace", default="results/navigation-joint-final-comparison-v2")
    sealer.add_argument("--out", type=Path, required=True)
    sealer.set_defaults(function=seal)
    prep = commands.add_parser("preparation")
    prep.add_argument("--joint-opening", type=Path, required=True)
    prep.add_argument("--primary-preparation-protocol", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.set_defaults(function=preparation)
    compare = commands.add_parser("comparison")
    compare.add_argument("--joint-opening", type=Path, required=True)
    compare.add_argument("--joint-preparation-protocol", type=Path, required=True)
    compare.add_argument("--panel", type=Path, required=True)
    compare.add_argument("--review", type=Path, required=True)
    compare.add_argument("--out-dir", type=Path, required=True)
    compare.add_argument("--namespace", default="results/navigation-joint-final-comparison-v2")
    compare.add_argument("--aux-source-protocol", type=Path)
    compare.add_argument("--aux-endpoints", nargs="*", type=int, choices=(30, 50), default=[])
    compare.set_defaults(function=comparison)
    args = parser.parse_args()
    print(json.dumps(args.function(args), indent=2))


if __name__ == "__main__":
    main()
