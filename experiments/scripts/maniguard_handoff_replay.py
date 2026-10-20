"""Replay handoff timing on completed training prefixes, without new inference."""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

from helpers.maniguard_continuation_handoff import (
    MODES,
    HandoffReference,
    replay_factual_prefix,
)
from helpers.maniguard_latent import LatentController
from helpers.maniguard_review_sensitivity import ROOT, TRAINING_STAGES


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze(run, candidate, *, debounce=3, clock_switch_step=488, queue_snapshot=None):
    if re.fullmatch(r"(?:ea|random)-g[12]-[0-3]", candidate) is None:
        raise ValueError("Declare an existing development search candidate")
    run = Path(run)
    queue_path = run / "queue.json" if queue_snapshot is None else Path(queue_snapshot)
    queue_bytes = queue_path.read_bytes()
    queue = json.loads(queue_bytes)
    selected = []
    for attempt in queue["attempts"]:
        if attempt["stage"] not in TRAINING_STAGES or attempt["status"] != "completed":
            continue  # Exclude checking before opening any output.
        if re.fullmatch(re.escape(candidate) + r"-case[01]-a\d+", attempt["id"]):
            selected.append(attempt)
    if len(selected) != 2 or sorted(a["id"].split("-case")[1][0] for a in selected) != ["0", "1"]:
        raise ValueError("Require one completed, verified full training pair")
    controller_path = run / "search" / f"{candidate}.npz"
    with np.load(controller_path, allow_pickle=False) as data:
        weights = data["weights"].copy()
    controller = LatentController(weights)
    episodes = []
    for attempt in sorted(selected, key=lambda item: item["id"]):
        output = Path(attempt["output"])
        output = output if output.is_absolute() else ROOT / output
        job_path, receipt_path = output / "job.json", output / "verified-receipt.json"
        job, receipt = json.loads(job_path.read_bytes()), json.loads(receipt_path.read_bytes())
        if (
            job["family"] != "jar"
            or receipt["valid"] is not True
            or receipt["full_episode"] is not True
            or job["controller_sha256"] != controller.sha256()
            or receipt["controller_sha256"] != controller.sha256()
        ):
            raise ValueError("Require the exact factual Jar controller and full verification")
        trace_path, modes = output / "trace.jsonl", []
        for mode in sorted(MODES):
            reference = HandoffReference(
                weights, mode=mode, debounce=debounce, clock_switch_step=clock_switch_step
            )
            with trace_path.open() as stream:
                modes.append(
                    replay_factual_prefix(
                        (json.loads(raw) for raw in stream),
                        reference,
                        controls=receipt["physical"]["steps"],
                    )
                )
        episodes.append(
            {
                "attempt": attempt["id"],
                "stage": attempt["stage"],
                "scene": job["scene"],
                "seed": job["seed"],
                "job_sha256": file_hash(job_path),
                "receipt_sha256": file_hash(receipt_path),
                "trace_sha256": file_hash(trace_path),
                "original_repository_commit": job["repository_commit"],
                "original_code_hashes": job["code_hashes"],
                "references": modes,
            }
        )
    sources = [
        Path(__file__).resolve(),
        ROOT / "experiments/helpers/maniguard_latent.py",
        ROOT / "experiments/helpers/maniguard_continuation_handoff.py",
    ]
    return {
        "schema": "maniguard-factual-handoff-prefix-replay-v1",
        "candidate": candidate,
        "queue_sha256": hashlib.sha256(queue_bytes).hexdigest(),
        "controller_file_sha256": file_hash(controller_path),
        "controller_sha256": controller.sha256(),
        "source_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in sources},
        "episodes": episodes,
        "new_physical_controls": 0,
        "new_vla_requests": 0,
        "new_encoder_images": 0,
        "new_learned_model_training_or_queries": 0,
        "scope": "Numerical controller and switch timing on existing training prefixes. "
        "Factual parsing stops at each reference's first changed controller choice. "
        "Later original frames are not hybrid outcomes. Existing full verification "
        "is consumed, not repeated. No decoder or RNG continuity is verified. "
        "No independent checking output is opened; no physical fitness is assigned.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--debounce", type=int, default=3)
    parser.add_argument("--clock-switch-step", type=int, default=488)
    parser.add_argument("--queue-snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Select a fresh output path to preserve earlier evidence")
    result = analyze(
        args.run,
        args.candidate,
        debounce=args.debounce,
        clock_switch_step=args.clock_switch_step,
        queue_snapshot=args.queue_snapshot,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": file_hash(args.output),
                "episodes": [
                    {
                        "attempt": e["attempt"],
                        "switches": {
                            r["mode"]: r["first_changed_query_step"] for r in e["references"]
                        },
                    }
                    for e in result["episodes"]
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
