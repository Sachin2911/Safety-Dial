"""Preservation rejects invalid source provenance before remote access or writes."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "latent_preserve", ROOT / "experiments/scripts/maniguard_latent_preserve.py")
preservation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preservation)


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def test_existing_full_commit_is_accepted():
    preservation.validate_source_commit(git("rev-parse", "HEAD"))


@pytest.mark.parametrize("reference", ["HEAD", "0000000000000000000000000000000000000000"])
def test_invalid_source_is_rejected_before_remote_access_or_output(
        reference, monkeypatch, tmp_path):
    def forbidden_remote():
        raise AssertionError("Invalid provenance must fail before any remote access")

    monkeypatch.setattr(preservation, "HfApi", forbidden_remote)
    output = tmp_path / "receipt.json"
    with pytest.raises(ValueError, match="existing full Git commit"):
        preservation.preserve("unused", "model", "unused", {}, reference, output)
    assert not output.exists()


def test_existing_tree_is_not_a_source_commit():
    with pytest.raises(ValueError, match="existing full Git commit"):
        preservation.validate_source_commit(git("rev-parse", "HEAD^{tree}"))
