"""Run the task-6 preservation suite against the task-11 preservation tool."""

import importlib.util
from pathlib import Path

import pytest
import test_dsrl06_preserve as base

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "preserve_dsrl11", ROOT / "experiments/scripts/preserve_dsrl11.py"
)
preserve = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preserve)


@pytest.fixture(autouse=True)
def task11_tool(monkeypatch):
    monkeypatch.setattr(base, "preserve", preserve)


def test_fresh_task11_destinations():
    assert preserve.DATASET == "Sachioster/maniguard-jar-replication-20261022"
    assert preserve.MODEL == "Sachioster/safety-dial-dsrl11-20261025"
    assert preserve.PUBLISHED_MANIFEST == "receipts/dsrl11/preservation-manifest.json"


for _name in dir(base):
    if _name.startswith("test_"):
        globals()[_name] = getattr(base, _name)


def test_late_source_or_receipt_and_new_remote_evidence_require_restage(tmp_path):
    # The task-6 version plants a 06-* receipt; task 11 watches its own 11-* receipts.
    repository, local, remote, stage, live = base.prepared_stage(tmp_path)
    late = base.write(repository / "progress/receipts/11-late.md", "late receipt")
    with pytest.raises(ValueError, match="Late source or receipt"):
        preserve.verify_local(stage)
    late.unlink()
    base.write(remote / "episodes/E0/late-frame.npz", "new scientific evidence")
    with pytest.raises(ValueError, match="changed after"):
        preserve.verify_local(stage)
    assert not (local / "preservation-verification.json").exists()
    live.close()
