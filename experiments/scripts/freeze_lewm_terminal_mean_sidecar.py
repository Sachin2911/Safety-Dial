"""Metadata-only seal and independent final sidecar for all three CMA terminal means.

No outcome, array, checkpoint, model, environment or final-case data is opened by
seal. Build later copies existing reviewed ordinary final protocol metadata only.
The original eleven-profile manifest, V3 analysis and six selection gates stay intact.
"""

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NAV = "vukovc3t4o1dui"
HOSTS = {
    0: {"host": "navigation", "pod_id": NAV},
    1: {"host": "pusht", "pod_id": "wq4ar2qx2ff6r4"},
    2: {"host": "recovery", "pod_id": "nj2fqkp2prgk8x"},
}
ABSENT = (
    "results/navigation-final-cases-v1",
    "results/navigation-final-cases-v1-joint-access.json",
    "configs/lewm_campaign_20261031/navigation-final-cases-v1.json",
    "configs/lewm_campaign_20261031/navigation-joint-final-cases-v2.json",
    "configs/lewm_campaign_20261031/navigation-joint-final-comparison-v2",
)
CONVERSION = "es.gp.pheno(es.mean, into_bounds=es.boundary_handler.repair)"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text())


def path(value):
    result = Path(value)
    if result.is_absolute():
        require(result.resolve().is_relative_to(ROOT), "Metadata escaped canonical root")
    else:
        require(".." not in result.parts, "Metadata traversal")
        result = ROOT / result
    require(result.suffix == ".json" and not result.is_symlink(), "JSON metadata only")
    return result.resolve()


def ref(value):
    value = value.resolve()
    require(value.is_relative_to(ROOT), "References stay in canonical project")
    return {"path": str(value.relative_to(ROOT)), "sha256": sha(value)}


def referenced(reference):
    target = path(reference["path"])
    require(sha(target) == reference["sha256"], "Frozen metadata bytes differ")
    return load(target)


def write_new(target, value):
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return ref(target)


def validate_lock(lock, repeat):
    require(
        lock["schema"] == "lewm-secondary-terminal-mean-lock-v1"
        and lock["repeat_id"] == repeat
        and lock["host"] == HOSTS[repeat]
        and lock["optimizer_updates"] == 6
        and lock["conversion"] == CONVERSION,
        "Require the exact sixth-update mean of every fixed secondary repeat",
    )
    require(
        len(lock["parameters"]) == len(lock["raw_mean"]) == 10
        and all(math.isfinite(v) and -6 <= v <= 6 for v in lock["parameters"])
        and all(math.isfinite(v) for v in lock["raw_mean"]),
        "Uniform bounded phenotype conversion requires ten finite parameters",
    )


def seal(args):
    require(os.environ.get("RUNPOD_POD_ID") == NAV, "Seal on actual rented navigation")
    require(
        len(args.locks) == 3 and all(not (ROOT / name).exists() for name in ABSENT),
        "All three means must be sealed BEFORE any final metadata or access",
    )
    records = {}
    for target in args.locks:
        lock = load(target)
        repeat = lock["repeat_id"]
        require(repeat in HOSTS and repeat not in records, "Every fixed repeat exactly once")
        validate_lock(lock, repeat)
        proof = referenced(lock["prefinal_absence"]["navigation_proof"])
        require(
            proof["schema"] == "lewm-terminal-mean-prefinal-absence-v1" and proof["pod_id"] == NAV,
            "Require the authoritative pre-final Nav absence proof",
        )
        records[repeat] = {**ref(target), "repeat_id": repeat, "host": HOSTS[repeat]}
    require(set(records) == {0, 1, 2}, "Never select or omit a learner repeat")
    output = {
        "schema": "lewm-secondary-terminal-means-v1",
        "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "locks": [records[r] for r in range(3)],
        "all_three_means_fixed_before_final_metadata": True,
        "canonical_nav_absence_rechecked": {name: False for name in ABSENT},
        "conversion": CONVERSION,
        "source_builder": ref(Path(__file__)),
        "all_means_retained_regardless_of_duplicates_or_performance": True,
        "no_selection_between_mean_and_archive_from_final_outcomes": True,
        "new_model_simulator_calls": 0,
    }
    print(json.dumps(write_new(args.out, output), indent=2))


def build(args):
    require(os.environ.get("RUNPOD_POD_ID") == NAV, "Freeze final metadata on actual rented Nav")
    require(not args.out_dir.exists(), "Never overwrite a mean-sidecar metadata freeze")
    plan = load(args.plan)
    require(plan["schema"] == "lewm-terminal-mean-exploratory-plan-v1", "Frozen mean plan required")
    for name, expected in plan["source_sha256"].items():
        require(sha(ROOT / name) == expected, "Mean metadata/report source changed")
    means = referenced(plan["means"])
    require(
        means["all_three_means_fixed_before_final_metadata"] is True,
        "All three fixed terminal means must predate final metadata",
    )
    lock_refs = {item["repeat_id"]: item for item in means["locks"]}
    require(set(lock_refs) == {0, 1, 2}, "Three mean locks required")
    locks = {repeat: referenced(lock_refs[repeat]) for repeat in range(3)}
    for repeat, lock in locks.items():
        validate_lock(lock, repeat)
    manifest = load(args.main_manifest)
    require(
        manifest["schema"] == "lewm-navigation-joint-final-comparison-v2"
        and len(manifest["schedule"]) == 11,
        "Keep the original eleven-profile main manifest",
    )
    main_sha = sha(args.main_manifest)
    file_refs = {str(path(item["path"])): item["sha256"] for item in manifest["files"]}
    ordinary = {}
    for scheduled in manifest["schedule"]:
        target = path(scheduled["protocol"])
        require(
            file_refs.get(str(target)) == sha(target), "Main protocol differs from its manifest"
        )
        protocol = load(target)
        require(
            not (ROOT / protocol["output_namespace"]).exists(),
            "Freeze every mean profile BEFORE any main final episode/output",
        )
        if scheduled.get("family") == "secondary30":
            repeat = protocol["repeat_id"]
            require(
                repeat not in ordinary
                and repeat in HOSTS
                and all(protocol["host"][key] == HOSTS[repeat][key] for key in ("host", "pod_id")),
                "Original secondary repeat/host mapping changed",
            )
            ordinary[repeat] = (target, protocol)
    require(set(ordinary) == {0, 1, 2}, "All three original secondary profiles required")
    namespace = Path(args.namespace)
    require(
        not namespace.is_absolute()
        and namespace.parts[0] == "results"
        and len(namespace.parts) == 2
        and ".." not in namespace.parts
        and "mean" in namespace.name
        and "final" in namespace.name,
        "Use a fresh distinct results/final-mean namespace",
    )
    require(not (ROOT / namespace).exists(), "Mean outcomes must not already exist")
    args.out_dir.mkdir(parents=True)
    profiles = []
    for repeat in range(3):
        target, base = ordinary[repeat]
        lock, lock_ref = locks[repeat], lock_refs[repeat]
        selected = [job for job in base["jobs"] if job["arm"] == "cma"]
        require(
            len(selected) == 16 and len({job["case"]["source_episode"] for job in selected}) == 16,
            "Exactly the same sixteen ordinary CMA source cases required",
        )
        profile = copy.deepcopy(base)
        for field in ("selected_nominees", "selection_lock", "fp32_duplicate_parameter_groups"):
            profile.pop(field, None)
        profile.update(
            name=f"navigation-exploratory-terminal-mean-r{repeat}",
            frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            output_namespace=str(namespace / f"r{repeat}"),
            maximum_episode_starts=16,
            maximum_native_resets=16,
            maximum_environment_actions=480,
            optimizer_updates=0,
        )
        profile["arm_order"] = ["cma_terminal_mean"]
        profile["duplicate_policy"] = (
            "Execute the fixed terminal mean on all sixteen cases even if FP32-equivalent "
            "to an ordinary selected controller. Retain lock duplicate flags; do not give "
            "evolutionary credit for duplicate policies."
        )
        profile["pairing"] = (
            "Exact ordinary secondary30 cases, environment/policy seeds and action caps; "
            "one terminal-mean arm only, no mean/archive choice from final outcomes."
        )
        profile["jobs"] = [
            {
                "job_id": "cma_terminal_mean_"
                + job["case"]["case_id"]
                + "_s"
                + str(job["environment_seed"]),
                "arm": "cma_terminal_mean",
                "method": "cma",
                "parameters": lock["parameters"],
                **{
                    key: copy.deepcopy(job[key])
                    for key in (
                        "case",
                        "environment_seed",
                        "policy_seed",
                        "repeat_id",
                        "max_actions",
                    )
                },
                "ordinary_cma_job_id": job["job_id"],
            }
            for job in selected
        ]
        profile["terminal_mean"] = {
            "lock": {key: lock_ref[key] for key in ("path", "sha256")},
            "mean_parameters": lock["parameters"],
            "completed_updates": 6,
            "ordinary_protocol": ref(target),
            "selected_on_development": False,
            "duplicate_flags": lock["duplicate_flags"],
        }
        profile["mean_sidecar"] = {
            "plan": ref(args.plan),
            "means": plan["means"],
            "main_manifest": ref(args.main_manifest),
        }
        profile["interpretation"] = plan["scope"]
        profile["analysis"] = plan["scope"]
        require(
            profile["evaluation"] == base["evaluation"]
            and profile["source_sha256"] == base["source_sha256"]
            and profile["source_files"] == base["source_files"]
            and profile["case_preparation"] == base["case_preparation"]
            and profile["evaluation"]["maximum_episode_actions"] == 30,
            "No runtime, assets, sources, geometry, case admission or budget change",
        )
        profile_ref = write_new(args.out_dir / f"terminal-mean-r{repeat}.json", profile)
        profiles.append(
            {
                "repeat_id": repeat,
                "host": copy.deepcopy(base["host"]),
                "protocol": profile_ref,
                "ordinary_protocol": ref(target),
                "output_namespace": profile["output_namespace"],
            }
        )
    output = {
        "schema": "lewm-terminal-mean-final-sidecar-v1",
        "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "plan": ref(args.plan),
        "means": plan["means"],
        "main_manifest": ref(args.main_manifest),
        "profiles": profiles,
        "maximum_episode_starts": 48,
        "maximum_environment_actions": 1440,
        "all_main_outputs_absent_at_binding": True,
        "original_eleven_profile_manifest_unchanged": True,
        "ordinary_profiles_first_on_each_host": True,
        "separate_from_optimizer_training_and_selection_budgets": True,
        "scope": plan["scope"],
        "new_model_simulator_calls": 0,
    }
    require(sha(args.main_manifest) == main_sha, "Original main manifest changed")
    print(json.dumps(write_new(args.out_dir / "manifest.json", output), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    seal_parser = commands.add_parser("seal")
    seal_parser.add_argument("--locks", nargs=3, type=Path, required=True)
    seal_parser.add_argument("--out", type=Path, required=True)
    builder = commands.add_parser("build")
    builder.add_argument("--plan", type=Path, required=True)
    builder.add_argument("--main-manifest", type=Path, required=True)
    builder.add_argument("--out-dir", type=Path, required=True)
    builder.add_argument("--namespace", required=True)
    args = parser.parse_args()
    (seal if args.command == "seal" else build)(args)


if __name__ == "__main__":
    main()
