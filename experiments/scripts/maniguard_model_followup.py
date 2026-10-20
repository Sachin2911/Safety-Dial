"""Freeze and audit a deferred CPU model prototype; physical execution is disabled."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.maniguard_model_budget import WorkLedger
from helpers.maniguard_model_data import digest, freeze_manifest, load_manifest
from helpers.maniguard_model_transition import TransitionModel


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def require_offline(config):
    if (config.get("physical_execution_enabled") is not False
            or config["cpu_threads"] != 1
            or any(config["work_limits"][name] != 0 for name in (
                "simulator_controls", "physical_episode_starts", "decoder_requests",
                "encoder_images"))):
        raise ValueError("This entry point supports CPU recorded-data preparation only")


def audit(args):
    raw_manifest = json.loads(args.manifest.read_text())
    require_offline(raw_manifest["configuration"])
    for name, expected in raw_manifest["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Prototype source changed after corpus freeze: " + name)
    manifest, datasets = load_manifest(args.manifest)
    config = manifest["configuration"]
    ledger = WorkLedger(manifest["canonical_ledger"], config["work_limits"], digest(args.manifest))
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write_new(args.output / "started.json", {"attempt_id": args.attempt_id,
              "manifest": str(args.manifest.resolve()), "manifest_sha256": digest(args.manifest),
              "physical_execution_enabled": False})
    try:
        model = TransitionModel.fit(datasets["fit"], config["model"], ledger,
                                    args.attempt_id + "/fit")
        calibration = model.calibrate(datasets["calibration"], config["gate"], ledger,
                                       args.attempt_id + "/calibration")
        completions = sum(entry["safe_requested_completion"] for entry in manifest["entries"]
                          if entry["split"] == "diagnostic")
        validation = model.audit(datasets["diagnostic"], config["gate"], ledger,
                                  args.attempt_id + "/diagnostic", safe_completions=completions)
        artifact = args.output / "transition-model.npz"
        model.save(artifact, corpus_sha256=digest(args.manifest))
        report = {"schema": "maniguard-model-offline-audit-v1", "attempt_id": args.attempt_id,
                  "configuration": config, "manifest": str(args.manifest.resolve()),
                  "manifest_sha256": digest(args.manifest), "model_sha256": digest(artifact),
                  "rows_by_split": {split: len(data) for split, data in datasets.items()},
                  "episodes_by_split": {split: sum(e["split"] == split for e in manifest["entries"])
                                        for split in datasets},
                  "reused_acquisition_controls": manifest["reused_acquisition_controls"],
                  "new_simulator_controls": 0, "new_decoder_requests": 0, "new_encoder_images": 0,
                  "calibration": calibration, "prediction_audit": validation,
                  "physical_execution_enabled": False,
                  "elapsed_cpu_process_seconds": time.monotonic() - started,
                  "work_ledger": ledger.summary(),
                  "scope": ("Previously exposed task-disjoint preparation audit. "
                            "A failed gate is retained; no physical job is dispatched.")}
        write_new(args.output / "report.json", report)
        print(json.dumps({"report": str(args.output / "report.json"),
                          "prediction_gate_pass": validation["prediction_gate_pass"],
                          "gate_failures": validation["failures"],
                          "rows_by_split": report["rows_by_split"],
                          "elapsed_seconds": report["elapsed_cpu_process_seconds"],
                          "physical_execution_enabled": False}))
    except Exception as exc:
        write_new(args.output / "failed.json", {"error": type(exc).__name__, "message": str(exc),
                  "work_ledger": ledger.summary(), "elapsed_seconds": time.monotonic() - started})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze",
                                 help="Freeze exact recorded files and the canonical ledger")
    freeze.add_argument("--run", type=Path, required=True)
    freeze.add_argument("--config", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("audit", help="Fit and audit the CPU prototype once")
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--attempt-id", default="offline-v1")
    check = commands.add_parser("check-schedule",
                                help="Validate the deferred plan without dispatch")
    check.add_argument("--schedule", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        config = json.loads(args.config.read_text())
        require_offline(config)
        result = freeze_manifest(args.run, config, args.output)
        print(json.dumps({"manifest": str(args.output), "episodes": len(result["entries"]),
                          "canonical_ledger": result["canonical_ledger"],
                          "new_simulator_controls": 0, "new_decoder_requests": 0}))
    elif args.command == "audit":
        audit(args)
    else:
        schedule = json.loads(args.schedule.read_text())
        physical = schedule["next_physical_development"]
        total = (physical["screen"]["valid_full_episodes"]
                 + physical["confirmation"]["valid_full_episodes"])
        if (schedule["automatic_physical_dispatch"] is not False
                or physical["start_time_utc"] is not None
                or total != physical["valid_full_episodes_proposed"]
                or total + physical["invalid_replacement_slots_proposed"]
                != physical["maximum_full_attempts_proposed"]):
            raise ValueError("Deferred schedule is inconsistent or attempts automatic "
                             "physical dispatch")
        print(json.dumps({"schedule": str(args.schedule), "physical_state": physical["state"],
                          "planned_valid_episodes": total,
                          "maximum_attempts_proposed": physical["maximum_full_attempts_proposed"],
                          "physical_dispatch": False}))


if __name__ == "__main__":
    main()
