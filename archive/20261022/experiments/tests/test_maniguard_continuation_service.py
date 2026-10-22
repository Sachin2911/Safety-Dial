"""Service-tail accounting from small saved JSON records, without a live server."""

import hashlib
import json

import pytest

from helpers.maniguard_continuation_service import (
    capture_service_records,
    service_baseline,
    verify_service_records,
)


def setup(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    path = original / "server-jar-worker0-physical.jsonl"
    path.write_text('{"episode":"original"}\n')
    protocol = {"original_run": str(original), "workers": {"0": {"service_ledger": str(path)}}}
    job = {"worker": 0, "episode_id": "new", "service_ledger_baseline": service_baseline(protocol, 0)}
    output = tmp_path / "episode"
    output.mkdir()
    (output / "protocol.json").write_text(json.dumps(protocol))
    (output / "job.json").write_text(json.dumps(job))
    return path, output, job


def test_service_tail_is_exact_restorable_and_preserves_the_original_prefix(tmp_path):
    path, output, job = setup(tmp_path)
    prefix = path.read_bytes()
    records = [{"episode": "new", "query_index": i, "status": "returned"} for i in range(2)]
    with path.open("a") as stream:
        for row in records:
            stream.write(json.dumps(row) + "\n")
    original = path.read_bytes()
    assert capture_service_records(output) == records
    assert path.read_bytes() == original and original.startswith(prefix)
    path.unlink()  # Recovery validation needs only the copied evidence.
    rows = [{"type": "noise_receipt", "receipt": row} for row in records]
    assert verify_service_records(output, job, rows) == 2
    rows[0]["receipt"] = {**records[0], "query_index": 9}
    with pytest.raises(ValueError, match="actual returned"):
        verify_service_records(output, job, rows)


@pytest.mark.parametrize("change", ["prefix", "foreign", "partial", "truncate", "snapshot"])
def test_changed_original_history_or_foreign_work_is_rejected(tmp_path, change):
    path, output, job = setup(tmp_path)
    prefix = path.read_bytes()
    own = b'{"episode":"new","query_index":0}\n'
    path.write_bytes(prefix + own)
    if change == "prefix":
        path.write_bytes(b"x" + (prefix + own)[1:])
    elif change == "foreign":
        path.write_bytes(prefix + b'{"episode":"foreign"}\n')
    elif change == "partial":
        path.write_bytes(prefix + own[:-1])
    elif change == "truncate":
        path.write_bytes(b"")
    elif change == "snapshot":
        (output / "server-receipts.jsonl").write_bytes(b"different")
    with pytest.raises(ValueError):
        capture_service_records(output)
    assert job["service_ledger_baseline"]["sha256"] == hashlib.sha256(prefix).hexdigest()
