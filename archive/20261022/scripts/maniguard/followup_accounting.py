"""Combine immutable Part1 costs with followup attempts once, preserving unknown counters."""

import csv
import json
import time
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from followup_evidence import FIGURES, ROOT, RUN, atomic, savefig, sha

NOTES = ROOT / "docs/research/research_notes"


def stamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() if value else None


def main():
    old = json.loads((NOTES / "ManiGuard phase3 Part1 audit.json").read_text())
    new = json.loads((RUN / "evidence/audit.json").read_text())
    old_cost = old["interaction_and_time_costs"]["recorded_sums"]
    costs = {}
    for key, value in new["costs"].items():
        costs[key] = {
            "followup_known_total": value["known_total"],
            "followup_unknown_attempts": value["unknown_attempts"],
            "Part1_known_total": old_cost.get(key),
            "cumulative_known_total": value["known_total"] + old_cost.get(key, 0),
            "scope": "new passive-capture measure only"
            if key not in old_cost
            else "Part1 plus followups once",
        }
    with (NOTES / "ManiGuard phase3 Part1 attempts.csv").open() as f:
        old_rows = list(csv.DictReader(f))
    rows = []
    for row in old_rows:
        rows.append(
            {
                "part": 1,
                "episode_id": row["episode_id"],
                "purpose": row["purpose"],
                "scene": row["scene"],
                "seed": row["seed"],
                "condition": row["dial"],
                "valid": row["valid"],
                "full_episode": row["full_episode"],
                "C_req": row.get("C_req"),
                "S_req": row.get("S_req"),
                "V": row.get("V"),
                "policy_control_steps": int(row["policy_control_steps"])
                if row["policy_control_steps"]
                else None,
                "policy_queries": int(row["policy_queries"]) if row["policy_queries"] else None,
                "physics_steps": int(row["physics_steps"]) if row["physics_steps"] else None,
                "process_seconds": float(row["process_seconds"]),
                "started_unix": stamp(row["started_utc"]),
                "finished_unix": stamp(row["finished_utc"]),
            }
        )
    for row in new["records"]:
        process = row["process"]
        rows.append(
            {
                "part": row["part"],
                "episode_id": row["episode_id"],
                "purpose": row["purpose"],
                "scene": row["scene"],
                "seed": row["seed"],
                "condition": row["condition"],
                "valid": row["valid"],
                "full_episode": row["full_episode"],
                "C_req": row.get("C_req"),
                "S_req": row.get("S_req"),
                "V": row.get("V"),
                **{
                    k: row["costs"].get(k)
                    for k in ("policy_control_steps", "policy_queries", "physics_steps")
                },
                "process_seconds": process.get("elapsed_seconds"),
                "started_unix": process.get("started_at"),
                "finished_unix": process.get("finished_at"),
            }
        )
    assert len(rows) == 80 and len({(r["part"], r["episode_id"]) for r in rows}) == 80
    target = NOTES / "ManiGuard phase3 cumulative attempts.csv"
    with target.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    extraction = json.loads((RUN / "part3/features/extraction-ledger.json").read_text())
    model = ROOT / "data/maniguard/models/phase3-parts2-3-20261018"
    training = json.loads((model / "training-ledger.json").read_text())
    calibration = json.loads((model / "calibration.json").read_text())
    forecast = json.loads((model / "forecast-audit.json").read_text())
    cpu = [
        json.loads((RUN / "part3" / name).read_text())
        for name in (
            "cpu-torch-check.json",
            "cpu-acquisition-check.json",
            "cpu-audit-pipeline-check.json",
        )
    ]
    model_costs = {
        "scientific_encoder_calls": extraction["forward_calls"],
        "scientific_encoder_images": extraction["images"],
        "encoder_seconds": extraction["seconds"],
        "scientific_optimizer_steps": training["optimizer_steps"],
        "scientific_training_member_examples": training["training_member_examples"],
        "training_seconds": training["seconds"],
        "scientific_inference_member_calls": calibration["member_calls"]
        + forecast["metrics"]["member_calls"],
        "scientific_inference_member_examples": calibration["member_examples"]
        + forecast["metrics"]["member_examples"],
        "CPU_engineering_member_calls": cpu[0]["toy_member_calls"]
        + cpu[1]["member_calls"]
        + cpu[2]["member_calls"],
        "CPU_engineering_member_examples": cpu[0]["toy_member_examples"]
        + cpu[1]["member_examples"]
        + cpu[2]["member_examples"],
        "CPU_engineering_training_examples": cpu[0]["toy_training_examples"],
        "CPU_engineering_optimizer_steps": cpu[0]["toy_training_steps"],
        "CPU_engineering_encoder_calls": 1,
        "CPU_engineering_encoder_images": 2,
        "posthoc_matched_readout_matrix_calls": 4,
        "posthoc_matched_readout_examples": 2208,
        "scientific_linear_dynamics_matrix_calls_derived": 8,
        "scientific_linear_dynamics_examples_derived": 8 * forecast["metrics"]["audit_windows"],
        "scientific_readout_matrix_calls_derived": 39,
        "scientific_readout_examples_derived": 4 * forecast["metrics"]["audit_windows"]
        + 3 * 3067
        + 16 * (5 * 32 * 4 + 32),
        "linear_count_basis": "Derived from frozen audit source and retained array dimensions: two four-step linear dynamics references; seven audit readouts, plus two readouts for each of sixteen 32-context candidate screens. CPU engineering linear operations were not separately instrumented.",
    }
    memory = []
    for line in (RUN / "gpu-samples.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d.get("returncode") == 0:
            try:
                memory.append(float(d["sample"].split(",")[0]))
            except ValueError:
                pass
    from helpers.maniguard_followup_budget import clock

    result = {
        "schema": "maniguard-phase3-cumulative-accounting-v1",
        "at": time.time(),
        "clock_at_export": clock(),
        "allocation_start_utc": "2026-10-18T16:00:37Z",
        "original_physical_target_utc": "2026-10-18T23:00:37Z",
        "original_hard_end_utc": "2026-10-19T00:00:37Z",
        "authorized_physical_target_utc": "2026-10-19T03:00:37Z",
        "authorized_hard_end_utc": "2026-10-19T04:00:37Z",
        "attempts": {"Part1": 52, "Part2": 20, "Part3": 8, "cumulative": 80},
        "full_episodes": {"Part1": 48, "Part2": 16, "Part3_recording": 8, "Part3_optimizer": 0},
        "physical_costs": costs,
        "model_costs": model_costs,
        "outer_episode_process_seconds": sum(r["process_seconds"] or 0 for r in rows),
        "followup_outer_episode_seconds": sum(
            r["process_seconds"] or 0 for r in rows if r["part"] != 1
        ),
        "physical_work_stopped": json.loads((RUN / "gpu-work-stopped.json").read_text()),
        "sampled_followup_gpu_memory_max_mib": max(memory),
        "gpu_memory_sample_count": len(memory),
        "memory_scope": "Five-second whole-GPU samples, not a bound on unobserved peaks; torch workers separately record allocated-memory peaks.",
        "pre_subscription_physics": "Unknown for every simulator application launch in all parts; never imputed as zero.",
        "deduplication": "Part1 common prior experience once; Part2 fitting corpus once; eight common Part3 recordings once. Model replay is not new physical experience.",
        "tooling_failures": [
            "Pre-app Part2 import failure is one of the twenty attempts, with unknown native counters.",
            "Coordinator startup failed from resolving a venv interpreter symlink; no simulation job.",
            "Policy handshake probe used an interpreter without websockets; corrected to the simulation interpreter before full runs.",
            "An initial asset inventory used the wrong directory prefix; preserved diagnostic, corrected inventory finds assets.",
            "The immediate GPU query after supervisor stop still listed a context; following independent polls verified empty.",
        ],
        "corrections": [
            "The interim Part2 prose stated twenty engineering queries. Its audited engineering records are 4 + 8 + 4 = sixteen. Comparative outcomes and raw ledgers are unchanged."
        ],
        "native_followup_videos": new["native_video_files"],
        "matched_followup_videos": 16,
        "source_hashes": {
            "Part1_audit": sha(NOTES / "ManiGuard phase3 Part1 audit.json"),
            "followup_audit": sha(RUN / "evidence/audit.json"),
            "attempt_csv": sha(target),
            "script": sha(__file__),
        },
    }
    for name in ("provider-stop-response.json", "provider-stop-verification.json"):
        if (RUN / name).exists():
            result[name.replace(".json", "")] = json.loads((RUN / name).read_text())
    atomic(RUN / "evidence/cumulative-accounting.json", result)
    atomic(NOTES / "ManiGuard phase3 cumulative accounting.json", result)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    start = result["clock_at_export"]["started_utc"]
    ordered = sorted(
        (r for r in rows if r["finished_unix"] is not None), key=lambda r: r["finished_unix"]
    )
    x, y, total = [0], [0], 0
    for row in ordered:
        if row["policy_control_steps"] is None:
            continue
        total += row["policy_control_steps"]
        x.append((row["finished_unix"] - start) / 3600)
        y.append(total / 1000)
    end = result["clock_at_export"]["elapsed_seconds"] / 3600
    axes[0].step(x + [end], y + [total / 1000], where="post", color="#416d83")
    axes[0].set(
        xlabel="Continuous allocated hours from original start",
        ylabel="Known cumulative executed controls (thousands)",
        xlim=(0, end),
        ylim=(0, None),
    )
    colors = {1: "#597d90", 2: "#b48047", 3: "#9376a1"}
    for row in rows:
        if row["started_unix"] is not None and row["finished_unix"] is not None:
            axes[1].broken_barh(
                [
                    (
                        (row["started_unix"] - start) / 3600,
                        (row["finished_unix"] - row["started_unix"]) / 3600,
                    )
                ],
                (4 - row["part"] - 0.16, 0.32),
                facecolors=colors[row["part"]],
            )
    for path in (RUN / "part3/model-processes").glob("*/process.json"):
        p = json.loads(path.read_text())
        axes[1].broken_barh(
            [((p["started_at"] - start) / 3600, (p["finished_at"] - p["started_at"]) / 3600)],
            (-0.16, 0.32),
            facecolors="#488b75",
        )
    axes[1].set(
        xlabel="Continuous allocated hours from original start",
        yticks=[0, 1, 2, 3],
        yticklabels=["Model workers", "Part3 recording", "Part2 episodes", "Part1 episodes"],
        xlim=(0, end),
        title="All gaps remain charged allocation time",
    )
    provenance = []
    savefig(fig, "cumulative-costs", provenance)
    atomic(
        FIGURES / "accounting-provenance.json",
        {
            "files": provenance,
            "accounting_sha256": sha(RUN / "evidence/cumulative-accounting.json"),
            "script_sha256": sha(__file__),
        },
    )
    print(
        json.dumps(
            {
                "attempts": len(rows),
                "controls": costs["policy_control_steps"],
                "queries": costs["policy_queries"],
                "clock": result["clock_at_export"],
            }
        )
    )


if __name__ == "__main__":
    main()
