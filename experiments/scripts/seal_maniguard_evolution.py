"""Seal a completed rented-host project for byte-verified transfer and teardown."""

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

from preserve_maniguard_evolution import eligible


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def process_matches(pid, command, ticks=None):
    proc = Path("/proc") / str(pid)
    try:
        stat = (proc / "stat").read_text().split()
        return (stat[2] != "Z" and (ticks is None or stat[21] == ticks)
                and command.encode() in (proc / "cmdline").read_bytes())
    except FileNotFoundError:
        return False


def seal(project, run_name, output, *, allow_incomplete=False):
    project, output = Path(project).resolve(), Path(output).resolve()
    run = project / run_name
    state = json.loads((run / "status.json").read_text())
    audit = json.loads((project / "supervision/final-audit.json").read_text())
    if not audit["final"] or audit["status"] != state["status"]:
        raise ValueError("Completed training or an explicit incomplete terminal audit is required")
    if state["status"] in ("ready", "running"):
        raise ValueError("Trainer state is not terminal")
    if not allow_incomplete and state["status"] != "completed":
        raise ValueError("Completed training and a successful final audit are required")
    if (not allow_incomplete and
            audit["valid_replayed_episodes"] != audit["declared_episodes"]):
        raise ValueError("Audit does not cover the whole declared episode schedule")
    service = json.loads((project / ("service-" + run_name.removeprefix("run-") + ".json"))
                         .read_text())
    if process_matches(service["pid"], "run_maniguard_evolution.py"):
        raise ValueError("Trainer has not exited yet")
    for path in (run / "episodes").glob("*/process.json"):
        process = json.loads(path.read_text())
        if process_matches(process["pid"], "benchmark_adapter.py", process["start_ticks"]):
            raise ValueError("An episode worker is still live")
    generated = json.loads((run / "generated/evaluator-source.json").read_text())
    upstream = Path(generated["checkout"]) / "maniguard/eval/benchmark.py"
    if digest(upstream) != generated["upstream_sha256"]:
        raise ValueError("The original public evaluator was changed")
    entries = {}
    for path in sorted(project.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlink in final project")
        if not path.is_file() or path == output:
            continue
        relative = path.relative_to(project)
        if not eligible(relative):
            continue
        entries[relative.as_posix()] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    manifest = {
        "schema": "maniguard-evolution-remote-seal-v1",
        "sealed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project": str(project), "run": run_name, "status": state["status"],
        "complete_schedule": audit.get("complete_schedule", state["status"] == "completed"),
        "trainer_exited": True, "all_episode_workers_retired": True,
        "original_public_evaluator_unchanged": True,
        "audit_sha256": digest(project / "supervision/final-audit.json"),
        "files": entries, "file_count_excluding_seal": len(entries),
        "bytes_excluding_seal": sum(value["bytes"] for value in entries.values()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(output)
    return {"remote_seal_sha256": digest(output), "files_excluding_seal": len(entries),
            "bytes_excluding_seal": manifest["bytes_excluding_seal"],
            "trainer_exited": True, "all_episode_workers_retired": True}


def verify_local(project, seal_path, expected_seal_sha256):
    project, seal_path = Path(project).resolve(), Path(seal_path).resolve()
    if digest(seal_path) != expected_seal_sha256:
        raise ValueError("Transferred seal differs from the exact remote seal")
    manifest = json.loads(seal_path.read_text())
    for relative, metadata in manifest["files"].items():
        path = project / relative
        if project not in path.resolve().parents or path.is_symlink():
            raise ValueError("Invalid path in transferred artifact tree")
        if path.stat().st_size != metadata["bytes"] or digest(path) != metadata["sha256"]:
            raise ValueError("Local artifact differs from remote seal: " + relative)
    actual = {path.relative_to(project).as_posix() for path in project.rglob("*")
              if path.is_file() and path != seal_path and eligible(path.relative_to(project))}
    if actual != set(manifest["files"]):
        raise ValueError("Local artifact inventory differs from remote seal")
    return {"local_verified": True, "files_including_seal": len(actual) + 1,
            "bytes_excluding_seal": manifest["bytes_excluding_seal"],
            "remote_seal_sha256": expected_seal_sha256}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--run", default="run-v2")
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--verify-local-sha256")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.verify_local_sha256:
        result = verify_local(args.project, args.seal, args.verify_local_sha256)
    else:
        result = seal(args.project, args.run, args.seal, allow_incomplete=args.allow_incomplete)
    print(json.dumps(result), flush=True)
