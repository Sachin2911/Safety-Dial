"""Run only the predeclared eight common recordings, after Part 2 completes."""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments/scripts"))

from maniguard_part2_run import launch, require_guard, save_new

from helpers.maniguard_followup_budget import atomic, clock, run_owned
from helpers.maniguard_part2_protocol import RUN, validate_frozen
from helpers.maniguard_part3_protocol import COLLECT, recording_plan
from helpers.maniguard_phase2_families import validate_scene_access
from helpers.maniguard_phase3_protocol import sha


def main():
    require_guard()
    part2 = RUN / "part2"
    while not (part2 / "results.json").exists():
        if clock()["physical_remaining"] < 8 * 240 + 3600:
            raise RuntimeError("Part2/model/cleanup work no longer fits; preserve unstarted recording")
        progress = part2 / "progress.json"
        if progress.exists() and any(not r["valid"] for r in json.loads(progress.read_text())["records"]):
            raise RuntimeError("Part 2 is incomplete; stop dependent recording")
        time.sleep(5)
    results = json.loads((part2 / "results.json").read_text())["records"]
    if len(results) != 16 or any(not r["valid"] for r in results):
        raise ValueError("Complete Part 2 before Part 3")
    source = validate_frozen(part2 / "protocol.json")
    decision_path = RUN / "part3/data-decision.json"
    decision = json.loads(decision_path.read_text())
    if decision["recording_schedule"] != recording_plan() or not decision["common_recording_required"]:
        raise ValueError("Recording decision differs from the predata seal")
    directory = RUN / "part3/recording"
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / "protocol.json").exists():
        protocol = dict(source)
        protocol.update(schema="maniguard-part3-common-recording-v1", part=3,
                        data_decision_sha256=sha(decision_path),
                        schedule=recording_plan(), full_episode_allowance=8,
                        code_scope="Part2 frozen physical evaluator; model tools frozen separately before fitting")
        for scene in COLLECT:
            validate_scene_access("clutter", scene, "train")
            scene_file = ROOT / "data/maniguard/benchmark/clutter_pickup" / scene / "scene_ep1.json"
            protocol["scenes"] = protocol["scenes"] + [
                {"scene": scene, "purpose": "train", "sha256": sha(scene_file)}]
        save_new(directory / "protocol.json", protocol)
        schedule = recording_plan()
        schedule["protocol_sha256"] = sha(directory / "protocol.json")
        save_new(directory / "jobs-plan.json", schedule)
    records = []
    for job in recording_plan()["jobs"]:
        audit_path = directory / "attempts" / (job["episode_id"] + "-a1") / "audit.json"
        record = json.loads(audit_path.read_text()) if audit_path.exists() else launch(
            directory, job, directory / "protocol.json",
        )
        records.append(record)
        atomic(directory / "progress.json", {"records": records, "clock": clock()})
        if not record["valid"]:
            raise RuntimeError("Invalid full recording slot stops model fitting; no retries")
    save_new(directory / "results.json", {"records": records, "clock": clock()})
    # Every model worker has an owned, separately supervised process group and a
    # bounded allowance. No worker is launched if its whole allowance cannot fit.
    env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
               CUBLAS_WORKSPACE_CONFIG=":4096:8")
    freeze_path = RUN / "part3/model-source-freeze.json"
    source_freeze = json.loads(freeze_path.read_text())
    for relative, digest in source_freeze["code_hashes"].items():
        if sha(ROOT / relative) != digest:
            raise ValueError("Model source changed after freeze: " + relative)
    for stage, timeout in (("extract", 3600), ("fit", 2400), ("calibrate", 600), ("audit", 600)):
        if clock()["physical_remaining"] < timeout + 60:
            raise RuntimeError("Model stage cannot fit its remaining physical allowance")
        receipt = run_owned([
            "uv", "run", "--no-project", "--python",
            str(ROOT / "data/maniguard/envs/model/bin/python"), "python",
            str(ROOT / "experiments/scripts/maniguard_part3_model_run.py"), stage,
        ], RUN / "part3/model-processes" / (stage + "-a1"), "model", timeout=timeout,
            registry=RUN, env=env)
        if receipt["status"] != "completed":
            raise RuntimeError("Model worker incomplete: " + stage)
    directory = ROOT / "data/maniguard/models/phase3-parts2-3-20261018"
    audit = json.loads((directory / "forecast-audit.json").read_text())
    atomic(RUN / "part3/forecast-audit.json", audit)
    print(json.dumps({"model_gate": audit["gate"], "clock": clock()}), flush=True)
    if audit["gate"]["eligible"]:
        timeout = min(18000, clock()["physical_remaining"] - 45)
        receipt = run_owned([
            "uv", "run", "--no-project", "--python",
            str(ROOT / "data/maniguard/envs/model/bin/python"), "python",
            str(ROOT / "experiments/scripts/maniguard_part3_search_run.py"),
        ], RUN / "part3/search-process-a1", "search", timeout=timeout, registry=RUN, env=env)
        if receipt["status"] != "completed":
            raise RuntimeError("Eligible physical search is incomplete; preserve evidence")
    else:
        atomic(RUN / "part3/terminal.json", {
            "status": "scientifically_gated", "gate": audit["gate"], "clock": clock(),
            "recording_episodes": 8, "optimizer_episodes": 0,
            "reason": "The predeclared forecast gate does not support physical model-assisted search; criteria remain unchanged",
        })


if __name__ == "__main__":
    main()
