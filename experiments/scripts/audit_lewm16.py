"""Audit task-16 provenance, episode payloads, dense monitors and all-attempt costs."""

import argparse
import collections
import hashlib
import json
import sys
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.lewm16_protocol import boxes, hits, write_json


def audit(root, manifests):
    failures, missing, checked_archives = [], [], []
    archived_manifests = {}
    for archive_path in sorted((root / "source").glob("*.tar.gz")):
        with tarfile.open(archive_path) as archive:
            manifest_name = (
                "progress/receipts/" + archive_path.name.removesuffix(".tar.gz") + ".json"
            )
            manifest_bytes = archive.extractfile(manifest_name).read()
            manifest = json.loads(manifest_bytes)
            for name, expected in manifest["source_sha256"].items():
                actual = hashlib.sha256(archive.extractfile(name).read()).hexdigest()
                if actual != expected:
                    failures.append(f"Source archive differs: {archive_path.name}:{name}")
            archived_manifests[hashlib.sha256(manifest_bytes).hexdigest()] = manifest
            checked_archives.append(archive_path.name)
    counters, stages = collections.Counter(), collections.defaultdict(collections.Counter)
    attempts = {}
    for folder in sorted((root / "attempts").iterdir()):
        if not folder.is_dir():
            continue
        job = json.loads((folder / "ticket.json").read_text())
        worker = json.loads((folder / "worker.json").read_text())
        manifest = archived_manifests.get(worker["manifest_sha256"])
        if manifest is None or job not in manifest["jobs"]:
            failures.append(f"Attempt not in archived frozen source: {folder.name}")
        if not (folder / "result.json").exists():
            missing.append(folder.name)
            progress = folder / "progress.json"
            if progress.exists():
                counters.update(json.loads(progress.read_text()).get("counters", {}))
            continue
        row = json.loads((folder / "result.json").read_text())
        if row["job"] != job:
            failures.append(f"Result/ticket mismatch: {folder.name}")
        attempts[job["id"]] = row
        counters.update(row.get("counters", {}))
        stages[job["stage"]][row["status"]] += 1
        if row["status"] != "complete":
            continue
        data = np.load(folder / "states.npz")
        states, actions, fine = (data[k] for k in ("states", "actions", "fine_states"))
        if len(states) != len(actions) or len(states) != row["env_steps"]:
            failures.append(f"Episode lengths differ: {folder.name}")
        if row["counters"].get("environment_actions", 0) != len(actions):
            failures.append(f"Action counter differs: {folder.name}")
        if row["counters"].get("executed_physics_steps", 0) != len(fine):
            failures.append(f"Physics counter differs: {folder.name}")
        xy = fine[:, :2] if job["env"] == "tworoom" else fine[:, 4:6]
        hazard = boxes(job["env"], job["case"])
        violation = bool(hits(xy, hazard).any())
        breach = bool(hits(xy, hazard, job["margin"]).any())
        if (violation, breach) != (row["any_violation"], row["margin_breach"]):
            failures.append(f"Independent monitor differs: {folder.name}")
        if row["requested_safe_completion"] != (row["success"] and not breach):
            failures.append(f"Safe completion label differs: {folder.name}")
        if not all(np.isfinite(x).all() for x in (states, actions, fine)):
            failures.append(f"Nonfinite execution payload: {folder.name}")
        tolerance = 0.001 if job["env"] == "tworoom" else 1e-6
        if row["replay_error_max"] > tolerance:
            failures.append(f"Checker replay failed: {folder.name}")
    expected = {}
    for path in manifests:
        manifest = json.loads(path.read_text())
        for job in manifest["jobs"]:
            if job["id"] in expected:
                failures.append(f"Duplicate planned job: {job['id']}")
            expected[job["id"]] = job
            if job["id"] not in attempts:
                missing.append(job["id"])
            elif attempts[job["id"]]["status"] != "complete":
                failures.append(f"Required comparison invalid: {job['id']}")
    correction = ROOT / "progress/receipts/16-engineering-invalidations.json"
    adjustment = 0
    if correction.exists():
        for item in json.loads(correction.read_text())["invalid"]:
            if all(name in attempts for name in item["ids"]):
                adjustment += item.get("additional_executed_physics_steps", 0)
    counters["reconstructed_missing_executed_physics_steps"] = adjustment
    counters["total_executed_physics_steps"] = counters["executed_physics_steps"] + adjustment
    if counters["environment_actions"] > 160000:
        failures.append("Campaign action cap exceeded")
    return dict(
        valid=not failures and not missing,
        failures=failures,
        missing=sorted(set(missing)),
        stages={k: dict(v) for k, v in stages.items()},
        attempts=len(attempts),
        required_comparison_episodes=len(expected),
        resources=dict(counters),
        verified_source_archives=checked_archives,
        accounting_scope="All attempted controlled actions, executed repeat steps, counterfactual integrations and model work. Reset/forward initialization is not counted as controlled physics integration.",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/lewm16/remote")
    parser.add_argument("--manifest", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.manifest)
    write_json(args.out, result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "valid",
                    "attempts",
                    "required_comparison_episodes",
                    "failures",
                    "missing",
                    "stages",
                )
            }
        )
    )
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
