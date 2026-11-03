"""Ordering assertions across filesystem timestamp ticks; run on rental only."""

import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "experiments/scripts")]
import audit_mechanism_v2 as audit  # noqa: E402


@pytest.mark.parametrize(
    "spec_time,result_time,passes", [(10, 12, True), (9.99, 12, False), (10, 10.002, False)]
)
def test_ordering_ticks_and_true_reversals(tmp_path, monkeypatch, spec_time, result_time, passes):
    monkeypatch.setattr(audit, "OUT", tmp_path)
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps(dict(locked_at=10.003)))
    os.utime(lock, (10, 10))
    folder = tmp_path / "episodes/e"
    folder.mkdir(parents=True)
    for name, stamp in [("spec.json", spec_time), ("result.json", result_time)]:
        path = folder / name
        path.write_text("{}")
        os.utime(path, (stamp, stamp))
    if passes:
        audit.verify_order("lock.json", [dict(id="e")])
    else:
        with pytest.raises(AssertionError):
            audit.verify_order("lock.json", [dict(id="e")])
