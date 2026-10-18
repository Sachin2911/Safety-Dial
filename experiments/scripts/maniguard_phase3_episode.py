"""One serial physical Part 1 episode, with exact scene and controller provenance."""

import argparse
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "third_party/ManiGuard"))

from helpers.maniguard_phase2_families import family_spec, validate_scene_access
from helpers.maniguard_phase3_adapter import Controller, SafetyDial, genome_hash
from helpers.maniguard_phase3_budget import atomic, clock
from helpers.maniguard_phase3_protocol import (
    check_sources,
    implementation_files,
    sha,
    validate_frozen,
)
from helpers.maniguard_phase3_source import prepare_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--port", type=int, default=18000)
    args = parser.parse_args()
    if clock()["physical_remaining"] <= 30:
        raise RuntimeError("Physical allocation deadline reached")
    job = json.loads(args.job.read_text())
    purpose = job["purpose"]
    if purpose not in ("engineering", "train", "validation"):
        raise ValueError("Part 1 excludes the reserved test partition")
    protocol = None if args.protocol is None else validate_frozen(args.protocol)
    if purpose != "engineering" and protocol is None:
        raise ValueError("Freeze before search or comparison")
    if purpose != "engineering":
        run_root = args.job.resolve().parent.parent
        plan_path = run_root / (
            f"population-{job['generation']}-plan.json"
            if purpose == "train"
            else "comparison-plan.json"
        )
        plan = json.loads(plan_path.read_text())
        if [j for j in plan["jobs"] if j["episode_id"] == job["episode_id"]] != [job]:
            raise ValueError("Episode differs from the prospectively sealed exact job")
        if purpose == "train" and plan["protocol_sha256"] != sha(args.protocol):
            raise ValueError("Population plan belongs to a different frozen protocol")
        if purpose == "validation":
            finalist_path = run_root / "finalist.json"
            finalist = json.loads(finalist_path.read_text())
            if sha(finalist_path) != plan["finalist_sha256"] or job["genome"] != finalist["genome"]:
                raise ValueError("Comparison must use the same pre-frozen finalist")
    horizon = job["action_horizon"]
    if horizon != 1100 and not (purpose == "engineering" and 1 <= horizon <= 32):
        raise ValueError("Native 1100-action horizon is required")
    validate_scene_access("clutter", job["scene"], purpose)
    if purpose == "train" and job["dial"] != 1:
        raise ValueError("Search at dial 1 only")
    adapter = SafetyDial(Controller.from_genome(job["genome"]), job["dial"])
    sources = check_sources()
    spec = family_spec("clutter")
    scene_root = ROOT / "data/maniguard/benchmark/clutter_pickup"
    scene = scene_root / job["scene"] / "scene_ep1.json"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    job.update(
        phase="phase3_part1",
        family="clutter",
        output=str(output),
        random_policy=False,
        genome_sha256=genome_hash(job["genome"]),
        effective_horizon=adapter.horizon,
        scene_sha256=sha(scene),
        policy=sources["policies"]["clutter"],
        max_steps_override=horizon if purpose == "engineering" else None,
        protocol_sha256=sha(args.protocol) if args.protocol else None,
        checkpoint_receipt_sha256=sha(
            ROOT / "data/maniguard/policies/clutter/download-provenance.json"
        ),
        adapter={"kind": "phase3_part1", "genome": job["genome"], "dial": job["dial"]},
        repository_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        code_hashes={},
    )
    for path in implementation_files():
        relative = path.relative_to(ROOT)
        destination = output / "study_source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
        job["code_hashes"][str(relative)] = sha(path)
    atomic(output / "job.json", job)
    os.environ["SAFETY_DIAL_MANIGUARD_JOB"] = str(output / "job.json")
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ["OMNIGIBSON_DATA_PATH"] = str(ROOT / "data/maniguard/datasets")
    runtime = prepare_runtime(output)
    # Execute the archived source copy, avoiding live edits between engineering jobs.
    sys.path.insert(0, str(output / "study_source/experiments"))
    sys.argv = [
        str(runtime),
        "--config",
        str(ROOT / "third_party/ManiGuard" / spec["eval_config"]),
        "--benchmark-root",
        str(scene_root),
        "--scenes",
        job["scene"],
        "--seed",
        str(job["seed"]),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--source-profile",
        "legacy",
        "--output-dir",
        str(output),
        "--run-name",
        "episode",
        "--execute-horizon",
        str(adapter.horizon),
    ]
    if purpose == "engineering":
        sys.argv += ["--max-steps", str(horizon)]
    runpy.run_path(str(runtime), run_name="__main__")


if __name__ == "__main__":
    main()
