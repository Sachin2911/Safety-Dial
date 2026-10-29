"""Task 15 recovery preservation: executed bundles, separate cohort prefixes, verification."""

import importlib.util
import json
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"experiments/scripts/{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def episode(remote, leaf, job, task, seed):
    folder = remote / "episodes" / leaf
    (folder / "eval" / f"task_{task:04d}").mkdir(parents=True)
    for suffix in ("main.mp4", "wrist.mp4", "ltl.json"):
        (folder / "eval" / f"task_{task:04d}" / f"base_{suffix}").write_text(suffix)
    (folder / "job.json").write_text(json.dumps(job))
    row = {"scene_name": f"task_{task:04d}/base", "seed": seed, "success": False,
           "ltl_violated": False, "outcome": "manipulated", "steps": 2000}
    (folder / "validation.json").write_text(json.dumps({"row": row}))


def test_stage_keeps_cohorts_apart_and_ships_executed_bundles(tmp_path, monkeypatch):
    preserve = module("preserve_regrasp15r1")
    remote = tmp_path / "remote"
    episode(remote, "job300_native_task00_seed152000_a1",
            {"job": 300, "stage": "stage_c", "arm": "native"}, 0, 152000)
    (remote / "episodes/job304_partial").mkdir(parents=True)
    (remote / "attempts_stage_c.jsonl").write_text("{}\n")
    bundles = tmp_path / "source"
    bundles.mkdir()
    with tarfile.open(bundles / "executed.tar.gz", "w:gz") as tar:
        tar.add(remote / "attempts_stage_c.jsonl", arcname="frozen.txt")
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    (receipts / "15-regrasp-results.md").write_text("results")
    (receipts / "15-regrasp-recovery-evidence-20261026.json").write_text("{}")
    monkeypatch.setattr(preserve, "RECEIPTS", receipts)
    spec = {**preserve.COHORTS["historical"], "raw": remote, "local": {}, "bundles": bundles,
            "documents": [], "stage": tmp_path / "stage"}
    monkeypatch.setitem(preserve.COHORTS, "historical", spec)
    staged = preserve.stage_cohort("historical")
    stage = tmp_path / "stage"
    assert staged["canonical_episodes"] == 1
    assert (stage / "receipts/regrasp15/source-bundles/executed.tar.gz").is_file()
    assert not (stage / "receipts/regrasp15/source").exists()
    assert (stage / "receipts/regrasp15/15-regrasp-results.md").is_file()
    assert not (stage / "receipts/regrasp15/15-regrasp-recovery-evidence-20261026.json").exists()
    assert (stage / "recordings/regrasp15/raw/episodes/job304_partial").is_dir()
    leaves = list(stage.glob("jar/regrasp15_stage_c_native/seed152000/*/row.json"))
    assert len(leaves) == 1
    assert not list(stage.glob("**/regrasp15r1*"))
    preserve.verify_local(stage)
    (stage / "receipts/regrasp15/15-regrasp-results.md").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        preserve.verify_local(stage)
