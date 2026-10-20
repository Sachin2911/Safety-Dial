"""Synthetic restoration bindings; real physical replay remains the existing reader's job."""

import copy
import json
from pathlib import Path

import pytest

from helpers import maniguard_latent_restoration as restoration


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def restored(tmp_path, monkeypatch):
    original = tmp_path / "original/episodes/example-a0"
    output = tmp_path / "restored/merged/episodes/example-a0"
    attempt = {"id": "example-a0", "output": str(original), "stage": "search_generation_1"}
    physical = {"output": str(original), "steps": 1, "ledger": {"policy_queries": 1}}
    saved = {"valid": True, "full_episode": True, "score": 0, "source_commit": "synthetic",
             "controller_sha256": "c" * 64, "physical": physical,
             "outcome": {"C_req": False, "V": False}}
    write(original / "verified-receipt.json", saved)
    write(output / "verified-receipt.json", saved)
    job = {"episode_id": str(original), "family": "jar", "scene": "synthetic-scene",
           "seed": 1, "policy": {}, "code_hashes": {}, "policy_gpu_uuid": "synthetic"}
    write(output / "job.json", job)
    monitor = {"step": 0, "ap": {"jar_closed": False}, "doomed": False}
    rows = [{"type": "observed", "monitor": {"valid": True, "observation": monitor}},
            {"type": "result", "result": {"scene_name": "synthetic-scene"}}]
    (output / "trace.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    write(output / "episode/synthetic-scene_ltl.json", {"log": [monitor]})
    recovery_path = tmp_path / "restored/recovery-receipt.json"
    write(recovery_path, {"merged": str(tmp_path / "restored/merged"),
          "all_archives_redownloaded_and_sha256_checked": True,
          "archives": [{"repo_type": kind, "revision": "a" * 40, "sha256": "b" * 64}
                       for kind in ("model", "dataset")]})
    receipt = copy.deepcopy(saved)
    receipt["physical"]["output"] = str(output)
    monkeypatch.setattr(restoration, "read_episode", lambda path: receipt)
    monkeypatch.setattr(restoration, "runtime_errors", lambda *args, **kwargs: [])
    return attempt, {"output": str(output), "recovery_receipt": str(recovery_path)}, receipt


def test_restoration_uses_separate_paths_exact_receipts_and_original_native_monitor(restored):
    attempt, location, _ = restored
    row = restoration.verify_restored_attempt(attempt, location, {}, {"jar": {}})
    assert row["attempt"] == "example-a0" and row["policy_queries"] == 1
    assert row["exact_native_monitor_entries_joined"] == 1
    assert row["restored_output"] != attempt["output"]


@pytest.mark.parametrize("change", ["live_path", "other_path", "no_readback", "revision",
                                   "counter", "original_receipt", "monitor"])
def test_misbound_restoration_or_changed_physical_evidence_is_refused(restored, change):
    attempt, location, receipt = restored
    if change in ("live_path", "other_path"):
        location["output"] = (attempt["output"] if change == "live_path"
                              else str(Path(attempt["output"]).parent))
    elif change in ("no_readback", "revision"):
        path = Path(location["recovery_receipt"])
        value = json.loads(path.read_text())
        if change == "no_readback":
            value["all_archives_redownloaded_and_sha256_checked"] = False
        else:
            value["archives"][0]["revision"] = "main"
        write(path, value)
    elif change == "counter":
        receipt["physical"]["ledger"]["policy_queries"] = 2
    elif change == "original_receipt":
        path = Path(attempt["output"]) / "verified-receipt.json"
        value = json.loads(path.read_text())
        value["controller_sha256"] = "d" * 64
        write(path, value)
    else:
        path = Path(location["output"]) / "episode/synthetic-scene_ltl.json"
        value = json.loads(path.read_text())
        value["log"][0]["ap"]["jar_closed"] = True
        write(path, value)
    with pytest.raises(ValueError):
        restoration.verify_restored_attempt(attempt, location, {}, {"jar": {}})


def test_incomplete_original_cannot_create_a_complete_restoration_proof(tmp_path):
    run = tmp_path / "original"
    attempts = [{"id": f"example-{i}", "status": "completed", "engineering": False}
                for i in range(42)]
    write(run / "queue.json", {"attempts": attempts})
    protocol, paths = tmp_path / "protocol.json", tmp_path / "paths.json"
    write(protocol, {})
    write(paths, {"original_queue_sha256": restoration.digest(run / "queue.json"),
                  "episodes": {a["id"]: [] for a in attempts}})
    with pytest.raises(ValueError, match="exact current original queue"):
        restoration.build_inventory(run, protocol, paths, tmp_path / "unused-source-proof.json")
