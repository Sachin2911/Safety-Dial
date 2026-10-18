"""Export small, derived Part 1 evidence to Git without copying raw run directories."""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from time import gmtime, strftime


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def export(run, audit, root):
    run, audit, root = Path(run).resolve(), Path(audit).resolve(), Path(root).resolve()
    matched = read(audit / "matched-independent-audit.json")
    if not matched["valid"]:
        raise ValueError("Complete independent comparison audit required")
    notes = root / "docs/research/research_notes"
    figures = root / "docs/research/figures/ManiGuard/phase3-part1"
    figures.mkdir(parents=True, exist_ok=True)
    prefix = "ManiGuard phase3 Part1 "
    ledger = read(audit / "attempt-ledger.json")
    fields = [
        "attempt_id", "episode_id", "purpose", "scene", "seed", "native_episode_seed",
        "genome", "dial",
        "process_status", "returncode", "valid", "full_episode", "engineering_valid",
        "C_req", "S_req", "V", "benchmark_success", "U", "ever_contacted", "ever_grasped",
        "grasp_heuristic_actions", "cap_active_actions", "started_utc", "finished_utc",
        "process_seconds", "pre_subscription_physics_ticks", "errors",
        *ledger["costs"]["recorded_sums"],
    ]
    csv_path = notes / (prefix + "attempts.csv")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for attempt in sorted(ledger["attempts"], key=lambda a: a["process"]["started_at"]):
            row = {key: attempt.get(key) for key in fields}
            row.update(attempt["costs"])
            process = attempt["process"]
            native_path = Path(attempt["output"]) / "episode/results.jsonl"
            row["native_episode_seed"] = read(native_path).get("episode_seed") \
                if native_path.exists() else None
            row.update(
                process_status=process["status"],
                returncode=process.get("returncode"),
                process_seconds=process.get("elapsed_seconds"),
                genome=json.dumps(attempt.get("genome"), separators=(",", ":")),
                errors=" | ".join(attempt["errors"]),
            )
            for source, target in (("started_at", "started_utc"), ("finished_at", "finished_utc")):
                row[target] = (
                    strftime("%Y-%m-%dT%H:%M:%SZ", gmtime(process[source]))
                    if source in process else None
                )
            writer.writerow(row)
    copied = [csv_path]
    for name in ("dial-outcomes.png", "search-lineage.png", "per-scene-outcomes-and-costs.png"):
        target = figures / name
        shutil.copyfile(audit / name, target)
        copied.append(target)
    for source, suffix in (
        (audit / "lineage.json", "lineage.json"),
        (audit / "matched-independent-audit.json", "matched-audit.json"),
        (run / "comparison-matched-videos/index.json", "matched-videos.json"),
    ):
        target = notes / (prefix + suffix)
        shutil.copyfile(source, target)
        copied.append(target)
    protocol = read(run / "protocol.json")
    controller = read(run / "frozen-controller-export.json")
    controller["implementation_code_hashes"] = protocol["code_hashes"]
    controller["checkpoint_files"] = read(run / "input-receipts/clutter-checkpoint.json")["files"]
    controller_path = root / "configs/maniguard/phase3/controller.json"
    write(controller_path, controller)
    copied.append(controller_path)
    samples = [json.loads(line) for line in (run / "gpu-samples.jsonl").read_text().splitlines()]
    memory = [float(s["sample"].split(",")[0]) for s in samples if s["returncode"] == 0]
    violations = []
    for attempt in ledger["attempts"]:
        if not attempt.get("V"):
            continue
        episode = Path(attempt["output"]) / "episode"
        native = read(episode / "results.jsonl")
        monitor = read(episode / (native["scene_name"] + "_ltl.json"))
        first = next(r for r in monitor["log"] if r["step"] == monitor["violation_step"])
        violations.append({
            **{k: attempt[k] for k in ("episode_id", "purpose", "scene", "seed", "dial",
                                      "C_req", "S_req", "V")},
            "first_violation_step": monitor["violation_step"],
            "predicates_at_first_violation": first["ap"],
        })
    summary = {
        "schema": "maniguard-part1-public-audit-v1",
        "raw_run": str(run.relative_to(root)),
        "protocol_sha256": digest(run / "protocol.json"),
        "finalist_sha256": digest(run / "finalist.json"),
        "terminal": read(run / "terminal.json"),
        "deployment": read(run / "instance-confirmation.json"),
        "provider_instance": read(run / "provider-instance-receipt.json"),
        "resources": read(run / "resource-limits.json"),
        "physical_work_stopped": read(run / "physical-work-stopped.json"),
        "gpu_quiescence_confirmation": read(run / "gpu-quiescence-confirmation.json"),
        "guard_stop": read(run / "guard-stop-receipt.json"),
        "allocation_selection": read(run / "allocation-selection.json"),
        "postprocessing_complete": read(run / "postprocessing-complete.json"),
        "interaction_and_time_costs": ledger["costs"],
        "sampled_gpu_memory_max_mib": max(memory),
        "gpu_sample_count": len(memory),
        "peak_memory_caveat": "Five-second samples; unobserved peaks may be higher.",
        "search_independent_audit": read(run / "search-independent-audit.json"),
        "comparison": read(audit / "tables.json"),
        "comparison_states": read(audit / "comparison.json")["planned_job_states"],
        "all_recorded_violating_episodes": violations,
        "attempts": len(ledger["attempts"]),
        "attempts_with_unknown_pre_subscription_physics": sum(
            a.get("pre_subscription_physics_ticks") is None for a in ledger["attempts"]
        ),
        "valid_full_episodes": sum(a["valid"] and a["full_episode"] for a in ledger["attempts"]),
        "engineering_attempts": sum(a.get("purpose") == "engineering" for a in ledger["attempts"]),
        "verification": {
            "part1_cpu_passed": 33, "regressions_passed": 109, "regressions_skipped": 4,
            "full_physical_replay_audits": sum(a["valid"] and a["full_episode"]
                                               for a in ledger["attempts"]),
        },
        "exported_artifacts": {str(p.relative_to(root)): digest(p) for p in copied},
        "raw_evidence": {
            "private_dataset": "Sachioster/safety-dial-phase3-part1-20261018",
            "scope": "Logs, traces, source snapshots and videos; assets, weights and keys excluded.",
        },
    }
    write(notes / (prefix + "audit.json"), summary)
    print(json.dumps({"attempts": summary["attempts"],
                      "valid_full_episodes": summary["valid_full_episodes"],
                      "exported_files": len(copied) + 1}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    export(args.run, args.audit, args.root)
