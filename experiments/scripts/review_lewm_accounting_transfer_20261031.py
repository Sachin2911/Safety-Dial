"""Rented read-only TRAIN accounting closeout review, plus a bounded zero-work proof.

Reads the named provisional report, its TRAIN jobs and one known argparse failure.
No final data, models, simulators, foreign imports or collector modifications.
"""

import argparse
import ast
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

EXPECTED_SOURCE = "08596c23ce476978251a5aaf6e1c14919216c30843df74fca6c00bb884a088b4"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text())


def retired(worker):
    path = Path("/proc") / str(worker["pid"]) / "stat"
    if not path.exists():
        return True
    text = path.read_text()
    fields = text[text.rfind(")") + 2 :].split()
    return fields[0] == "Z" or fields[19] != str(worker["start_ticks"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/Safety-Dial"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--proof-out", type=Path, required=True)
    args = parser.parse_args()
    require(
        os.environ.get("RUNPOD_POD_ID") == "vukovc3t4o1dui",
        "Review belongs only on the rented Nav host",
    )
    root = args.root.resolve()
    require(
        not args.out.exists() and not args.proof_out.exists(),
        "Never overwrite accounting/proof evidence",
    )
    report = load(args.report)
    require(report["owning_pod_id"] == "vukovc3t4o1dui", "Wrong provisional host")
    reviews = []
    for unresolved in report["unresolved"]:
        require(
            unresolved["kind"] == "incomplete_attempt_requires_closeout",
            "Unexpected true unresolved record needs separate review",
        )
        name = unresolved["path"]
        require(
            name.startswith("results/navigation-budget50-comparison-v1/r0/reinforce/jobs/")
            or name.startswith("results/navigation-budget30-comparison-v1/r0/reinforce/jobs/"),
            "Only prespecified provisional TRAIN PG jobs may be inspected",
        )
        folder = root / name
        wrapper, inner = folder / "result.json", folder / "episode/result.json"
        require(
            wrapper.is_file() and inner.is_file() and (folder / "episode/trace.npz").is_file(),
            "A provisional TRAIN job is still missing terminal artifacts",
        )
        row = load(wrapper)
        require(
            row["status"] == "complete" and row["accounting_finalized"] and not row["error"],
            "A provisional TRAIN job is not successfully finalized",
        )
        require(
            digest(inner) == row["extra"]["episode_result_sha256"],
            "TRAIN inner bytes differ from finalized wrapper",
        )
        reviews.append(
            {
                "job_path": name,
                "classification_now": "complete_physical_episode",
                "environment_actions": row["environment_actions"],
                "wrapper_sha256": digest(wrapper),
                "inner_sha256": digest(inner),
                "would_be_counted_once_by_final_collector": True,
                "explanation": "Live provisional scan saw a moving prefix; final collection must replace, never merge with, this provisional host report",
            }
        )
    require(len(reviews) == 4, "Expected exactly the four reported moving TRAIN jobs")
    worker_path = root / "environment/workers/navigation-progress-target-pilot-v2.json"
    repair_path = (
        root / "environment/workers/navigation-progress-target-pilot-v2-invocation-repair.json"
    )
    log_path = root / "environment/navigation-progress-target-pilot-v2.log"
    source_path = root / "experiments/scripts/run_lewm_navigation_campaign_v2.py"
    worker, repair = load(worker_path), load(repair_path)
    require(retired(worker), "The failed exact worker identity is not retired")
    require(
        worker["entry_args"] == "--mode compare" and repair["entry_args"] == "",
        "Invocation history differs",
    )
    require(
        worker["out"] == repair["out"]
        and worker["expected_protocol_sha256"] == repair["expected_protocol_sha256"],
        "Repair used another namespace/protocol",
    )
    require(
        digest(source_path) == EXPECTED_SOURCE,
        "Argparse source differs from immutable executed source",
    )
    log = log_path.read_text()
    require("unrecognized arguments: --mode compare" in log, "Missing explicit argparse rejection")
    require("Traceback" not in log, "Unexpected error path in failed-invocation log")
    tree = ast.parse(source_path.read_text())
    main_function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    parse_lines = [
        node.lineno
        for node in ast.walk(main_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "parse_args"
    ]
    load_lines = [
        node.lineno
        for node in ast.walk(main_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("load_navigation_runtime", "run_comparison")
    ]
    require(
        len(parse_lines) == 1 and load_lines and parse_lines[0] < min(load_lines),
        "Argparse rejection no longer precedes runtime/comparison",
    )
    actual_freeze = load(root / worker["out"] / "source-freeze.json")
    require(
        actual_freeze["protocol_sha256"] == worker["expected_protocol_sha256"],
        "Repaired output protocol differs",
    )
    require(
        dt.datetime.fromisoformat(actual_freeze["started_utc"])
        >= dt.datetime.fromisoformat(repair["launched_utc"]),
        "Output existed before the distinct repaired worker",
    )
    proof = {
        "worker_name": worker["name"],
        "evidence_path": str(log_path.relative_to(root)),
        "evidence_sha256": digest(log_path),
        "reviewed_by": "Codex rented-host source/log/PID identity audit",
        "reason": "The unsupported --mode compare invocation was rejected by argparse before parse_args returned; immutable source parses before runtime loading/comparison/output creation. A later distinct worker without those args created the unused namespace. This proves zero NEW campaign model/simulator calls for this failed invocation; dependency hashing/import and pod elapsed costs are not asserted zero.",
        "worker_record": str(worker_path.relative_to(root)),
        "worker_record_sha256": digest(worker_path),
        "exact_retired_identity": {"pid": worker["pid"], "start_ticks": worker["start_ticks"]},
        "entrypoint_source": str(source_path.relative_to(root)),
        "entrypoint_sha256": digest(source_path),
        "protocol_sha256": worker["expected_protocol_sha256"],
        "repair_worker_record": str(repair_path.relative_to(root)),
        "repair_worker_record_sha256": digest(repair_path),
        "later_output_freeze_sha256": digest(root / worker["out"] / "source-freeze.json"),
        "resources": {"environment_actions": 0, "environment_resets": 0, "new_model_queries": 0},
        "parse_args_line": parse_lines[0],
        "first_runtime_or_comparison_line": min(load_lines),
    }
    proof_doc = {
        "schema": "lewm-campaign-zero-work-proofs-20261031-v1",
        "reviewed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "owning_pod_id": "vukovc3t4o1dui",
        "proofs": [proof],
        "final_outcomes_read": 0,
        "new_model_simulator_calls": 0,
    }
    args.proof_out.parent.mkdir(parents=True, exist_ok=True)
    args.proof_out.write_text(json.dumps(proof_doc, indent=2) + "\n")
    result = {
        "schema": "lewm-accounting-transfer-review-20261031-v1",
        "reviewed_utc": proof_doc["reviewed_utc"],
        "provisional_report_sha256": digest(args.report),
        "collector_v1_sha256": digest(
            root / "experiments/scripts/collect_lewm_campaign_20261031.py"
        ),
        "former_unresolved_jobs": reviews,
        "true_remaining_unresolved_among_these_four": 0,
        "zero_work_proof": {
            "path": str(args.proof_out.relative_to(root)),
            "sha256": digest(args.proof_out),
        },
        "final_data_read": 0,
        "new_model_simulator_calls": 0,
        "collector_correction_needed_under_ordered_workflow": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
