"""Inventory retained failed-launch records on their rented host, without rollouts."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import socket
import tarfile
import time


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, choices=[0, 1, 2], required=True)
    args = parser.parse_args()
    repo = Path("/workspace/Safety-Dial")
    assert repo.is_dir(), "Run on the existing rented study host"
    roots = sorted((repo / "outputs").glob(f"discarded-5lanes-r{args.repeat}-*"))
    assert roots
    out = repo / "SafeLeWM/outputs/block-paper-audit-20261102"
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"discarded-r{args.repeat}-v1.tar.gz"
    assert not target.exists(), "Preserve earlier collections"
    records, files = [], {}
    for root in roots:
        for start in sorted(root.rglob("worker-start.json")):
            attempt = start.parent
            account_path = attempt / "accounting.json"
            account = json.loads(account_path.read_text()) if account_path.exists() else None
            summary_path = attempt / "summary.json"
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
            steps_path = attempt / "steps.jsonl"
            rows, malformed = [], []
            if steps_path.exists():
                for number, line in enumerate(steps_path.read_text().splitlines(), 1):
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        malformed.append(number)
            assert [r["step"] for r in rows] == list(range(len(rows)))
            if summary:
                assert summary["status"] == "complete"
                assert len(rows) == summary["controls"]
                assert account and account["completed_controls"] == summary["controls"]
                assert not malformed
            lower = max(len(rows), account["completed_controls"] if account else 0)
            for name in ("worker-start.json", "accounting.json", "summary.json",
                         "steps.jsonl", "worker-error.json"):
                path = attempt / name
                if path.exists():
                    files[str(path.relative_to(repo))] = path
            records.append(dict(
                attempt=str(attempt.relative_to(repo)), completed=summary is not None,
                saved_steps=len(rows), malformed_step_lines=malformed,
                recorded_completed_controls_lower_bound=lower, accounting=account,
                measured_rollout_physics_steps=summary["rollout_physics_steps"] if summary else None,
                exact_completed_episode_controls=summary["controls"] if summary else None,
            ))
    result = dict(
        checked_at=time.time(), host=socket.gethostname(), repeat=args.repeat,
        attempts=len(records), completed_episodes=sum(r["completed"] for r in records),
        recorded_controls_lower_bound=sum(
            r["recorded_completed_controls_lower_bound"] for r in records
        ), records=records,
        scope="Retained five-lane discarded launches only; excluded from efficacy outcomes",
        limitations=[
            "Incomplete attempts provide a lower bound, not proof of zero unrecorded work",
            "No simulator replay, full camera backup or reset/initialization physics audit",
            "Pre-study pilot-host smoke/sanity and withdrawn R3 are outside this collection",
        ],
    )
    report = out / f"discarded-r{args.repeat}-v1.json"
    assert not report.exists()
    report.write_text(json.dumps(result, indent=2) + "\n")
    files["audit/report.json"] = report
    files["audit/collector.py"] = Path(__file__).resolve()
    manifest = {name: dict(bytes=p.stat().st_size, sha256=sha(p)) for name, p in files.items()}
    payload = json.dumps(dict(files=manifest), indent=2).encode() + b"\n"
    with tarfile.open(target, "w:gz", compresslevel=1) as tar:
        for name, path in files.items():
            tar.add(path, arcname=name, recursive=False)
        info = tarfile.TarInfo("manifest.json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    receipt = dict(
        archive=str(target), bytes=target.stat().st_size, sha256=sha(target),
        files=len(files), scope=result["scope"], limitations=result["limitations"],
    )
    (out / f"discarded-r{args.repeat}-v1-packaging.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps({k: result[k] for k in (
        "repeat", "attempts", "completed_episodes", "recorded_controls_lower_bound"
    )} | {"archive": receipt}), flush=True)


if __name__ == "__main__":
    main()
