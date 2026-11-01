"""Freeze a single, order-selected recovery development replay and exact live peer.

Metadata only. This builder never reads arrays, starts workers or accesses final cases.
"""

import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = "configs/lewm_campaign_20261031/navigation-budget30-relocated-v1/training-controls-r2.json"
PREFIX = "results/navigation-budget30-relocated-v1/r2/training-controls"
JOB = "strong_tworoom-01000-none_s355000"
NEW_NAME = "navigation-recovery-overlap-qualification-v1"
NEW_PROTOCOL = "configs/lewm_campaign_20261031/" + NEW_NAME + ".json"
AUTHORIZATION = "configs/lewm_campaign_20261031/" + NEW_NAME + "-authorization.json"
OBSERVATIONS = "environment/" + NEW_NAME + "-peer-observations.json"
PEER = {
    "name": "navigation-budget30-recovery-reinforce-r2",
    "record_sha256": "f1322d249da5a961f3a89bf44840a60b986b682aba4b405d2d9a4d5e54022400",
    "pid": 1607,
    "start_ticks": "444626392",
}
REFERENCE_FILES = {
    "/jobs/"
    + JOB
    + "/started.json": "9f55831d84d7235c418fccb8d99477c62486bba24482256b726c88078fa30fd7",
    "/jobs/"
    + JOB
    + "/result.json": "2e77cbcbebf7c80921b6d31643511fc234281be04792d188d0a0323619194ee7",
    "/jobs/"
    + JOB
    + "/episode/result.json": "48f66e837a40ec756e55182caec514da09454fb4818d1383dae39726795ce374",
    "/jobs/"
    + JOB
    + "/episode/trace.npz": "c7a2922c72e2c6576f67c99a98e9560205e35120ae146759e0e140c9c492f880",
    "/jobs/"
    + JOB
    + "/episode/selector-history.json": "d01326c53f5eeb0699d8f41e4f138cf370ade4d507035a6b0777de54684551f8",
}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, payload):
    if path.exists():
        raise FileExistsError("Never overwrite a frozen qualification")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-state-sha256", required=True)
    parser.add_argument("--reference-source-freeze-sha256", required=True)
    args = parser.parse_args()
    original = json.loads((ROOT / BASE).read_text())
    first = next(job for job in original["jobs"] if job["method"] == "fixed_geometry_progress")
    if first["job_id"] != JOB:
        raise RuntimeError("The first order-selected strong control changed")
    if first["max_actions"] != 30 or first["environment_seed"] != 355000:
        raise RuntimeError("Reference qualification identity changed")
    launcher = "experiments/scripts/launch_lewm_campaign_frozen_overlap.py"
    comparator = "experiments/scripts/qualify_lewm_recovery_overlap.py"
    source_files = (launcher, comparator, str(Path(__file__).relative_to(ROOT)))
    engineering_sources = {name: sha(ROOT / name) for name in source_files}
    protocol = copy.deepcopy(original)
    protocol["name"] = NEW_NAME
    protocol["output_namespace"] = "results/" + NEW_NAME
    protocol["jobs"] = [copy.deepcopy(first)]
    protocol["maximum_episode_starts"] = 1
    protocol["maximum_native_resets"] = 1
    protocol["maximum_environment_actions"] = 30
    protocol["engineering_overlap_qualification"] = {"no_final_access": True}
    protocol["overlap_qualification"] = {
        "reference_protocol": {"path": BASE, "sha256": sha(ROOT / BASE)},
        "reference_job_id": JOB,
        "reference_stage": PREFIX,
        "reference_artifacts": {
            **{PREFIX + name: digest for name, digest in REFERENCE_FILES.items()},
            PREFIX + "/state.json": args.reference_state_sha256,
            PREFIX + "/source-freeze.json": args.reference_source_freeze_sha256,
        },
        "expected_peer_identity": {key: PEER[key] for key in ("pid", "start_ticks")},
        "peer_observations_path": OBSERVATIONS,
        "selection_rule": "Exactly the FIRST fixed_geometry_progress job in original protocol order; no outcome selection",
        "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "engineering_source_sha256": engineering_sources,
        "new_work_is_additional": True,
        "new_work_budget": {"episode_starts": 1, "environment_actions_maximum": 30},
        "strict_parity": "All saved trace arrays, selected target identities and original deterministic counters identical; elapsed timings and retrieval_microseconds reported separately",
        "approval_rule": "No optimizer overlap before exact parity, actual memory headroom, timing review and a NEW prospective overlap authorization",
    }
    assert protocol["evaluation"] == original["evaluation"]
    assert protocol["source_files"] == original["source_files"]
    assert protocol["source_sha256"] == original["source_sha256"]
    write(ROOT / NEW_PROTOCOL, protocol)
    authorization = {
        "schema": "lewm-recovery-two-worker-authorization-v1",
        "purpose": "engineering_qualification",
        "execution_pod_id": "nj2fqkp2prgk8x",
        "deadline_utc": "2026-10-31T07:23:14+00:00",
        "max_concurrent_experiment_workers": 2,
        "scientific_settings_unchanged": True,
        "all_actual_work_charged": True,
        "launcher_sha256": engineering_sources[launcher],
        "peer": PEER,
        "peer_observations_path": OBSERVATIONS,
        "stage": {
            "name": NEW_NAME,
            "gate": "recovery-foundation-source-v1",
            "tests": 55,
            "protocol": NEW_PROTOCOL,
            "protocol_sha256": sha(ROOT / NEW_PROTOCOL),
            "entrypoint": "experiments/scripts/run_lewm_navigation_campaign_v2.py",
            "entry_args": [],
            "out": "results/" + NEW_NAME,
        },
    }
    write(ROOT / AUTHORIZATION, authorization)
    print(
        json.dumps(
            {
                "protocol": NEW_PROTOCOL,
                "protocol_sha256": sha(ROOT / NEW_PROTOCOL),
                "authorization": AUTHORIZATION,
                "authorization_sha256": sha(ROOT / AUTHORIZATION),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
