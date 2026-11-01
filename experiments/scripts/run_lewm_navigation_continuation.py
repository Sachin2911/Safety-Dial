"""Run an exactly frozen, unstarted final profile under explicit new time authority.

The original protocol file, cases, seeds, source pins and analysis bindings remain
unchanged. Only evaluation.deadline_utc is replaced in a private in-memory copy;
the original runtime writes its original protocol SHA plus an explicit sidecar ref.
"""

import argparse
import copy
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_DEADLINE = "2026-10-31T07:23:14+00:00"
ENTRYPOINTS = {
    "experiments/scripts/run_lewm_navigation_campaign_v2.py",
    "experiments/scripts/run_lewm_navigation_campaign_v3.py",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def project_path(value):
    path = Path(value)
    result = (ROOT / path).resolve()
    require(result.is_relative_to(ROOT), "Path leaves canonical project")
    return result


def timestamp(value):
    result = dt.datetime.fromisoformat(value)
    require(result.tzinfo is not None, "Explicit timezone required")
    return result


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def prepare(args):
    """Stdlib-only authorization and exact one-field-delta validation, no imports."""
    authority_path = project_path(args.authority)
    require(digest(authority_path) == args.authority_sha256, "Authority SHA differs")
    authority = json.loads(authority_path.read_text())
    require(
        authority["schema"] == "lewm-frozen-final-continuation-authority-v1",
        "Unknown continuation authority",
    )
    require(authority["original_deadline_utc"] == ORIGINAL_DEADLINE, "Wrong original campaign")
    require(
        authority["maximum_duration_seconds"] == 5400
        and authority["total_campaign_cap_usd"] == 13
        and authority["no_new_training"] is True
        and authority["no_outcome_based_changes"] is True,
        "Continuation scope/budget differs from explicit authority",
    )
    deadline = timestamp(authority["new_deadline_utc"])
    first = timestamp(authority["first_restart_utc"])
    require(0 < (deadline - first).total_seconds() <= 5400, "Continuation exceeds90minutes")
    now = dt.datetime.now(dt.timezone.utc)
    require(first <= now and now + dt.timedelta(minutes=20) < deadline, "Preservation reserve")
    protocol_path, output = project_path(args.protocol), project_path(args.out)
    require(not output.exists(), "No continuation replay, resume or output overwrite")
    entry = project_path(args.entrypoint)
    entry_name = str(entry.relative_to(ROOT))
    require(entry_name in ENTRYPOINTS, "Only original frozen final evaluators are allowed")
    protocol = json.loads(protocol_path.read_text())
    require(
        protocol["schema"] in ("lewm-navigation-comparison-v1", "lewm-navigation-comparison-v2"),
        "Only comparison profiles, never training",
    )
    protocol_name = str(protocol_path.relative_to(ROOT))
    matches = [
        item
        for item in authority["profiles"]
        if item["protocol"]["path"] == protocol_name
    ]
    require(len(matches) == 1, "One exact prospective profile authorization required")
    profile = matches[0]
    require(digest(protocol_path) == profile["protocol"]["sha256"], "Original protocol differs")
    require(
        profile["entrypoint"]["path"] == entry_name
        and digest(entry) == profile["entrypoint"]["sha256"],
        "Original evaluator differs",
    )
    require(
        profile["output_namespace"] == protocol["output_namespace"]
        and output == project_path(profile["output_namespace"]),
        "Original output namespace differs",
    )
    design = protocol["evaluation"]
    pod = os.environ.get("RUNPOD_POD_ID")
    require(
        pod == profile["pod_id"] == design["execution_pod_id"], "Wrong physical execution pod"
    )
    require(design["deadline_utc"] == ORIGINAL_DEADLINE, "Original deadline differs")
    source_pins = authority["source_sha256"]
    wrapper_name = str(Path(__file__).resolve().relative_to(ROOT))
    require(
        source_pins.get(wrapper_name) == digest(Path(__file__))
        and source_pins.get(entry_name) == digest(entry),
        "Authority must pin wrapper and unchanged evaluator",
    )
    for name, expected in source_pins.items():
        require(digest(project_path(name)) == expected, "Continuation source differs: " + name)
    for declaration in (protocol, design):
        for field in ("source_sha256", "asset_sha256"):
            for name, expected in declaration.get(field, {}).items():
                require(digest(project_path(name)) == expected, "Frozen dependency differs: " + name)
    effective = copy.deepcopy(protocol)
    effective["evaluation"]["deadline_utc"] = authority["new_deadline_utc"]
    reverted = copy.deepcopy(effective)
    reverted["evaluation"]["deadline_utc"] = ORIGINAL_DEADLINE
    require(reverted == protocol, "Only the execution deadline may change")
    sidecar = ROOT / "environment/continuation" / (profile["protocol"]["sha256"] + ".json")
    record = {
        "schema": "lewm-frozen-final-continuation-execution-v1",
        "recorded_utc": now.isoformat(),
        "pod_id": pod,
        "authority": {
            "path": str(authority_path.relative_to(ROOT)),
            "sha256": args.authority_sha256,
        },
        "wrapper": {"path": wrapper_name, "sha256": digest(Path(__file__))},
        "original_protocol": profile["protocol"],
        "original_entrypoint": profile["entrypoint"],
        "output_namespace": profile["output_namespace"],
        "effective_override": {
            "field": "evaluation.deadline_utc",
            "original": ORIGINAL_DEADLINE,
            "effective": authority["new_deadline_utc"],
        },
        "all_other_protocol_fields_identical": True,
        "source_assets_cases_seeds_parameters_budgets_unchanged": True,
        "new_training_or_selection": False,
        "original_protocol_bytes_unchanged": True,
    }
    return protocol_path, output, entry, effective, sidecar, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", required=True)
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    protocol_path, output, entry, effective, sidecar, record = prepare(args)
    write_new(sidecar, record)
    reference = {"path": str(sidecar.relative_to(ROOT)), "sha256": digest(sidecar)}
    sys.path.insert(0, str(ROOT / "experiments"))
    spec = importlib.util.spec_from_file_location("lewm_unchanged_final_runtime", entry)
    runtime = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runtime
    spec.loader.exec_module(runtime)
    original_writer = runtime.write_json

    def explicit_writer(path, value):
        if Path(path) == output / "source-freeze.json":
            require(
                value["protocol_sha256"] == record["original_protocol"]["sha256"],
                "Runtime did not bind the unchanged original protocol",
            )
            value = {**value, "continuation": reference}
        return original_writer(path, value)

    runtime.write_json = explicit_writer
    runtime.run_comparison(effective, output, protocol_path)


if __name__ == "__main__":
    main()
