"""Copy only the frozen48 NAV-R0 showcase trace/history pairs after accounting.

Stdlib only: select from frozen case metadata, copy exact existing bytes, hash and
read back every source/destination. No arrays, images, models or simulators load.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat


ROOT = Path(__file__).resolve().parents[2]
NAV = "vukovc3t4o1dui"
DESTINATION = Path("/workspace/lewm-analysis-inputs-20261031")
MEDIA_PLAN = "configs/lewm_campaign_20261031/navigation-scientific-media-plan-v2.json"
MEDIA_PLAN_SHA = "e43271d8aff95350779e90e6ffff525060d584a29ba333e19de56c6171286e53"
ARMS = ("native", "strong_anchor", "tuned_fixed", "cma", "random", "reinforce")
FAMILIES = ("primary50", "secondary30")
CONDITIONS = ("none", "circle3", "circle5", "circle7")


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text())


def safe_path(root, relative):
    path = PurePosixPath(relative)
    require(
        relative
        and "\\" not in relative
        and not path.is_absolute()
        and ".." not in path.parts
        and str(path) == relative,
        "A safe unchanged relative evidence path is required",
    )
    result = root / relative
    require(
        result.resolve().is_relative_to(root.resolve()) and not result.is_symlink(),
        "Evidence path escapes its root or is a symlink",
    )
    return result


def frozen_json(root, reference):
    path = safe_path(root, reference["path"])
    require(
        path.suffix == ".json" and sha(path) == reference["sha256"], "Frozen JSON reference changed"
    )
    return load(path)


def file_identity(info):
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--assembly-validation", type=Path, required=True)
    parser.add_argument("--accounting-merge", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    parser.add_argument("--manifest-out", type=Path, required=True)
    args = parser.parse_args()
    require(
        os.environ.get("RUNPOD_POD_ID") == NAV and ROOT.resolve() == Path("/workspace/Safety-Dial"),
        "Showcase staging belongs only on the actual canonical navigation host",
    )
    require(not args.manifest_out.exists(), "Never overwrite a media-transfer manifest")
    require(sha(args.request) == args.request_sha256, "Assembly request changed")
    request = load(args.request)
    require(
        request["schema"] == "lewm-analysis-json-assembly-20261031-v1"
        and request["never_collect_from_import_destination"] is True,
        "Require the owner-bound JSON-only assembly request",
    )
    destination = args.destination.resolve()
    require(
        destination == DESTINATION
        and str(destination) == request["destination"]
        and not destination.is_relative_to(ROOT),
        "Only the sibling analysis root is allowed",
    )
    marker = load(destination / ".analysis-only-input-root.json")
    validation = load(args.assembly_validation)
    require(
        marker["analysis_only"] is True
        and marker["request_sha256"] == args.request_sha256
        and validation["complete"] is True
        and validation["request_sha256"] == args.request_sha256
        and validation["namespace_count"] == 35,
        "All three owner archives and35 namespaces must validate before media copying",
    )
    account = load(args.accounting_merge)
    require(
        account["complete_final_accounting"] is True
        and account["unresolved"] == []
        and sha(args.accounting_merge) == request["accounting"]["aggregate_sha256"],
        "Exact complete accounting merge must precede supplemental copying",
    )
    require(sha(args.inputs) == request["inputs"]["sha256"], "Original bound inputs changed")
    inputs = load(args.inputs)
    require(
        inputs["schema"] == "lewm-navigation-analysis-inputs-20261031-v3"
        and inputs["all_final_output_namespaces_absent_at_binding"] is True,
        "Keep the original prospective input freeze",
    )
    require(sha(ROOT / MEDIA_PLAN) == MEDIA_PLAN_SHA, "Frozen media plan changed")
    plan = load(ROOT / MEDIA_PLAN)
    for name, expected in plan["source_sha256"].items():
        require(sha(ROOT / name) == expected, "Frozen media/analysis code changed")
    require(
        plan["showcase"]["repeat_id"] == 0 and plan["showcase"]["arms"] == list(ARMS),
        "Fixed showcase arm/repeat selection changed",
    )
    profiles, chosen = {}, {}
    for family in FAMILIES:
        entries = [
            entry
            for entry in inputs["profiles"]
            if entry["family"] == family and entry["repeat_id"] == 0
        ]
        require(
            len(entries) == 1 and entries[0]["host"]["pod_id"] == NAV,
            "Only the fixed NAV-R0 profile from each endpoint may supply media",
        )
        entry = entries[0]
        protocol = frozen_json(ROOT, entry["protocol"])
        require(
            protocol["host"]["pod_id"] == NAV and protocol["evaluation"]["execution_pod_id"] == NAV,
            "Actual showcase protocol belongs to another physical host",
        )
        cases = {job["case"]["case_id"]: job["case"] for job in protocol["jobs"]}
        chosen[family] = {}
        for condition in CONDITIONS:
            eligible = sorted(
                (case for case in cases.values() if case["condition"] == condition),
                key=lambda case: (case["source_episode"], case["source_step"], case["case_id"]),
            )
            require(len(eligible) == 4, "Each condition needs all four frozen admitted cases")
            chosen[family][condition] = eligible[0]["case_id"]
        profiles[family] = (entry, protocol)
    require(chosen[FAMILIES[0]] == chosen[FAMILIES[1]], "Both endpoints must use the same cases")
    selected = []
    for family, (entry, protocol) in profiles.items():
        state_path = safe_path(ROOT, entry["state_path"])
        require(
            sha(state_path) == sha(safe_path(destination, entry["state_path"])),
            "Original and assembled NAV state bytes differ",
        )
        state = load(state_path)
        require(
            state["phase"] == "complete" and state["active"] is None,
            "Showcase physical execution must be complete",
        )
        source = load(state_path.parent / "source-freeze.json")
        require(
            source["pod_id"] == NAV and source["protocol_sha256"] == entry["protocol"]["sha256"],
            "Original physical source freeze is not NAV-owned",
        )
        namespace = protocol["output_namespace"]
        owners = [item for item in request["namespaces"] if item["output_namespace"] == namespace]
        require(
            len(owners) == 1
            and owners[0]["owning_pod_id"] == NAV
            and owners[0]["kind"] == "final_profile",
            "Physical namespace ownership differs",
        )
        for condition in CONDITIONS:
            for arm in ARMS:
                jobs = [
                    job
                    for job in protocol["jobs"]
                    if job["arm"] == arm and job["case"]["case_id"] == chosen[family][condition]
                ]
                require(len(jobs) == 1, "Exactly one frozen job per showcase slot is required")
                job = jobs[0]
                row = state["jobs"][job["job_id"]]
                prefix = namespace + "/jobs/" + job["job_id"] + "/episode/"
                inner = safe_path(ROOT, prefix + "result.json")
                require(
                    sha(inner)
                    == row["extra"]["episode_result_sha256"]
                    == sha(safe_path(destination, prefix + "result.json")),
                    "Original and assembled inner-result provenance differs",
                )
                for basename in ("trace.npz", "selector-history.json"):
                    relative = prefix + basename
                    source_path = safe_path(ROOT, relative)
                    target_path = safe_path(destination, relative)
                    require(
                        source_path.is_file()
                        and stat.S_ISREG(source_path.stat().st_mode)
                        and not target_path.exists()
                        and not target_path.is_symlink(),
                        "Copy only existing regular source bytes into a fresh absent target",
                    )
                    expected = sha(source_path)
                    if basename == "selector-history.json":
                        require(
                            expected == row["extra"]["decision_reconstruction_sha256"],
                            "Frozen target-history source hash differs",
                        )
                    selected.append(
                        {
                            "relative_path": relative,
                            "source_sha256": expected,
                            "owning_pod_id": NAV,
                            "family": family,
                            "condition": condition,
                            "arm": arm,
                            "case_id": job["case"]["case_id"],
                        }
                    )
    require(
        len(selected) == 96 and len({item["relative_path"] for item in selected}) == 96,
        "Copy exactly48trace/history pairs, with no input dataset/model/checkpoint",
    )
    for record in selected:
        source_path = safe_path(ROOT, record["relative_path"])
        target_path = safe_path(destination, record["relative_path"])
        require(target_path.parent.is_dir(), "Validated JSON assembly must supply the parent")
        fd = os.open(source_path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as source:
            before = file_identity(os.fstat(source.fileno()))
            digest = hashlib.sha256()
            with target_path.open("xb") as target:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
                    target.write(block)
            require(
                file_identity(os.fstat(source.fileno())) == before,
                "Original showcase source changed during copy",
            )
        source_readback, target_readback = sha(source_path), sha(target_path)
        require(
            digest.hexdigest() == record["source_sha256"] == source_readback == target_readback,
            "Showcase source/destination readback hashes differ",
        )
        record.update(destination_sha256=target_readback, bytes=target_path.stat().st_size)
    manifest = {
        "schema": "lewm-navigation-showcase-supplement-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_helper_sha256": sha(Path(__file__)),
        "pod_id": NAV,
        "physical_source_root": str(ROOT),
        "analysis_destination": str(destination),
        "request_sha256": args.request_sha256,
        "accounting_aggregate_sha256": sha(args.accounting_merge),
        "assembly_validation_sha256": sha(args.assembly_validation),
        "original_bound_inputs_sha256": sha(args.inputs),
        "media_plan_sha256": MEDIA_PLAN_SHA,
        "chosen_cases_by_frozen_metadata_only": chosen,
        "file_count": 96,
        "trace_history_pairs": 48,
        "files": selected,
        "raw_video_enabled": False,
        "no_array_image_or_model_decoding": True,
        "new_model_simulator_calls": 0,
        "never_collect_from_import_destination": True,
    }
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest_out.open("x") as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "manifest": str(args.manifest_out),
                "sha256": sha(args.manifest_out),
                "pairs": 48,
                "all_sources_and_destinations_hash_verified": True,
                "new_model_simulator_calls": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
