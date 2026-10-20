"""Explicit nested evidence selection; no network, study or policy execution."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/maniguard_latent_checkpoint.py"


@pytest.fixture
def checkpoint(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("checkpoint_analysis_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def indexed(tmp_path, path="cpu-proof/bundle.tar.gz"):
    analysis = tmp_path / "analysis"
    source = analysis / path
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"synthetic evidence bundle")
    item = {"path": path, "bytes": source.stat().st_size,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    (analysis / "preservation-artifacts.json").write_text(json.dumps({
        "schema": "maniguard-explicit-analysis-artifacts-v1", "files": [item]}))
    return analysis, source


def test_explicit_nested_artifact_and_flat_report_are_frozen_without_sweeping_folders(tmp_path, checkpoint):
    analysis, source = indexed(tmp_path)
    (analysis / "flat.json").write_text("{}")
    (source.parent / "not-selected.bin").write_bytes(b"not selected")
    output = tmp_path / "snapshot"
    entries = checkpoint.snapshot_analysis(analysis, output)
    assert set(entries) == {"analysis/flat.json", "analysis/preservation-artifacts.json",
                            "analysis/cpu-proof/bundle.tar.gz"}
    assert Path(entries["analysis/cpu-proof/bundle.tar.gz"]).read_bytes() == source.read_bytes()
    source.write_bytes(b"later change")
    assert Path(entries["analysis/cpu-proof/bundle.tar.gz"]).read_bytes() != source.read_bytes()


def test_changed_explicit_artifact_is_rejected_before_upload(tmp_path, checkpoint):
    analysis, source = indexed(tmp_path)
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed before snapshot"):
        checkpoint.snapshot_analysis(analysis, tmp_path / "snapshot")


@pytest.mark.parametrize("kind", ["traversal", "symlink"])
def test_manifest_cannot_select_paths_outside_analysis(tmp_path, checkpoint, kind):
    analysis, source = indexed(tmp_path)
    index = analysis / "preservation-artifacts.json"
    manifest = json.loads(index.read_text())
    if kind == "traversal":
        manifest["files"][0]["path"] = "../outside"
        index.write_text(json.dumps(manifest))
    else:
        outside = tmp_path / "outside"
        outside.write_bytes(source.read_bytes())
        source.unlink()
        source.symlink_to(outside)
    with pytest.raises(ValueError):
        checkpoint.snapshot_analysis(analysis, tmp_path / "snapshot")
