"""Preserve failed attempts omitted from the successful-episode Block record bundle.

Run once after the original study and its independent audit are complete. This
reads existing records only and never launches or retries an experiment.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, choices=[0, 1, 2], required=True)
    args = parser.parse_args()
    repo = Path("/workspace/Safety-Dial")
    assert repo.is_dir(), "Use the original rented study host"
    root = repo / "SafeLeWM/outputs/block-paper-audit-20261102"
    run = repo / f"outputs/search2-20261102-r{args.repeat}"
    audit_path = root / f"r{args.repeat}-v1/audit.json"
    audit = read(audit_path)
    assert audit["passed"] and read(run / "status.json")["phase"] == "complete"
    out = root / f"r{args.repeat}-failures-v1"
    out.mkdir(exist_ok=False)
    failures = {f["path"] for f in audit["failed_attempts"]}
    actual = {
        str(p.relative_to(repo)) for phase in ("smoke", "cma", "select", "final", "random")
        for p in (run / phase).glob("*/attempt-*") if not (p / "summary.json").exists()
    }
    assert actual == failures, "Terminal failure inventory differs from the audit"
    ledgers = {}
    for path in run.glob("*/ledger.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            assert row["attempt"] not in ledgers
            ledgers[row["attempt"]] = row
    records, files, sources = [], {}, {}
    for relative in sorted(failures):
        attempt = repo / relative
        account_path = attempt / "accounting.json"
        account = read(account_path) if account_path.exists() else {}
        audited = next(f for f in audit["failed_attempts"] if f["path"] == relative)
        assert account == audited["accounting"]
        assert relative in ledgers and ledgers[relative].get("result") is None
        trace = attempt / "steps.jsonl"
        rows, malformed = [], []
        if trace.exists():
            for number, line in enumerate(trace.read_text().splitlines(), 1):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    malformed.append(number)
        assert [r.get("step") for r in rows] == list(range(len(rows)))
        for name in ("worker-start.json", "accounting.json", "worker-error.json",
                     "renderer.json", "steps.jsonl"):
            path = attempt / name
            if path.exists():
                files[str(path.relative_to(repo))] = path
        source_map = {}
        for path in sorted((attempt / "implementation").rglob("*")):
            if not path.is_file():
                continue
            digest = sha(path)
            source_map[str(path.relative_to(attempt / "implementation"))] = digest
            files.setdefault("sources/" + digest, path)
        sources[relative] = source_map
        log = attempt / "worker.log"
        completed = [p for p in attempt.parent.glob("attempt-*/summary.json")]
        assert len(completed) == 1, "Expected one audited successful attempt for this job"
        records.append(dict(
            attempt=relative, ledger=ledgers.get(relative), accounting=account,
            saved_step_rows=len(rows), malformed_step_lines=malformed,
            step_indices=[r.get("step") for r in rows],
            recorded_controls_lower_bound=max(len(rows), account.get("completed_controls", 0)),
            reset_evidence_present=(attempt / "initial_state.npz").exists(),
            successful_attempt=str(completed[0].parent.relative_to(repo)),
            worker_log_sha256=sha(log) if log.exists() else None,
        ))
    report = dict(
        checked_at=time.time(), repeat=args.repeat, audit_sha256=sha(audit_path),
        failed_attempts=len(records), records=records,
        recorded_controls_lower_bound=sum(r["recorded_controls_lower_bound"] for r in records),
        attempts_with_recorded_controls=sum(r["recorded_controls_lower_bound"] > 0 for r in records),
        source_snapshots=sources,
        scope="Failed-attempt records and source copies supplement the successful-episode bundle",
        limitations=[
            "Incomplete traces provide recorded lower bounds, not exact initialization work",
            "Worker logs are hashed, not included; camera/model/reset binary data excluded",
            "Original discarded five-lane launches and withdrawn R3 are separate",
        ],
    )
    report_path = out / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    files.update({"audit/report.json": report_path, "audit/source.py": Path(__file__),
                  "audit/main-audit.json": audit_path})
    manifest = {name: dict(bytes=p.stat().st_size, sha256=sha(p)) for name, p in files.items()}
    payload = json.dumps(dict(files=manifest), indent=2).encode() + b"\n"
    archive = out.with_suffix(".tar.gz")
    assert not archive.exists()
    with tarfile.open(archive, "w:gz", compresslevel=1) as tar:
        for name, path in files.items():
            tar.add(path, arcname=name, recursive=False)
        info = tarfile.TarInfo("manifest.json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    receipt = dict(completed=time.time(), archive=str(archive), bytes=archive.stat().st_size,
                   sha256=sha(archive), files=len(files), failed_attempts=len(records))
    (out / "packaging.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
