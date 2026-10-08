"""Recovery guards and final upload completion never optimize or accept mixed exports."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from helpers.walkerLewm import save_checkpoint
from helpers.walkerProtocol import file_sha256
from helpers.walkerTrainingFinalize import finalize_completed_run
from helpers.walkerTrainingState import save_training_state, upload_checkpoint, verify_resume_cadence


class FakeStore:
    def __init__(self):
        self.calls = []
    def repo_id(self, environment):
        return "private/walker-models"
    def upload_run(self, environment, name, run_dir, *, run_id, tag):
        self.calls.append((environment, name, run_id, tag))
        return "b" * 40


def completed_fixture(tmp_path):
    run = tmp_path / "completed-run"
    run.mkdir()
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones(4, 3)).square().mean().backward()
        optimizer.step()
        scheduler.step()
    roles = {"training": [0], "validation": [1]}
    scaler = (np.zeros(6), np.ones(6))
    architecture = {"fixture": "linear"}
    identity = {"total_steps": 2, "data_sha256": "source", "model": architecture,
        "episode_splits": roles, "scaler": [a.tolist() for a in scaler],
        "recipe": {"batch": 4, "lr": .01, "seed": 42}}
    history = [{"step": 2, "loss": .5}]
    save_training_state(run / "optimizer.pt", model=model, optimizer=optimizer, scheduler=scheduler,
                        step=2, identity=identity, history=history)
    save_checkpoint(model, run, cfg=architecture, scaler=scaler, step=2,
        extra={"val_pred_loss": .1, "history": history, "recovery_bundle": "optimizer.pt",
               "recovery_sha256": file_sha256(run / "optimizer.pt")})
    OmegaConf.save(OmegaConf.create({"name": "lewm-a", "total_steps": 2, "model": architecture,
        "episode_splits": roles, "source_hashes": {}, "push_every": 4000, **identity["recipe"]}), run / "config.yaml")
    (run / "splits.json").write_text(json.dumps(roles))
    manifest = {"run_id": run.name, "kind": "lewm-a", "metrics": {"step": 2, "val_pred_loss": .1, "train_loss": .5},
        "data": {"sha256": "source", "episode_splits": roles, "source_hashes": {}, "n_clips": 10}}
    (run / "manifest.json").write_text(json.dumps(manifest))
    return run, tmp_path / "final-report.json"


def test_legacy_resume_interval_uses_sibling_immutable_recipe(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("push_every: 4000\n")
    assert verify_resume_cadence(tmp_path / "optimizer.pt", 4000) == 4000
    with pytest.raises(ValueError, match="original --push-every 4000"):
        verify_resume_cadence(tmp_path / "optimizer.pt", 2000)
    config.unlink()
    with pytest.raises(ValueError, match="sibling config"):
        verify_resume_cadence(tmp_path / "optimizer.pt", 4000)


def test_completed_upload_recovery_performs_zero_optimizer_steps(tmp_path, monkeypatch):
    run, report_path = completed_fixture(tmp_path)
    def no_step(*args, **kwargs):
        raise AssertionError("Finalization must not optimize")
    monkeypatch.setattr(torch.optim.AdamW, "step", no_step)
    store = FakeStore()
    result = finalize_completed_run(run, report_path, store)
    assert store.calls == [("walker2d", "lewm-a", run.name, True)]
    assert result["steps"] == 2
    assert result["recovery_completion"]["optimizer_steps_executed"] == 0
    assert result["render_validation"]["passed"] is None
    assert result["wall_clock_s"] is None
    assert json.loads((run / "upload_receipt.json").read_text())["step"] == 2
    assert json.loads(report_path.read_text())["checkpoint"]["revision"] == "b" * 40
    with pytest.raises(FileExistsError, match="already exists"):
        finalize_completed_run(run, report_path, store)
    assert len(store.calls) == 1


@pytest.mark.parametrize("changed", ["weights.pt", "lewm_object.ckpt", "train_state.json", "manifest.json"])
def test_finalization_rejects_mixed_exports_before_upload(tmp_path, changed):
    run, report_path = completed_fixture(tmp_path)
    path = run / changed
    if changed == "weights.pt":
        state = torch.load(path, weights_only=True)
        state["weight"] += 1
        torch.save(state, path)
    elif changed == "lewm_object.ckpt":
        model = torch.load(path, weights_only=False)
        with torch.no_grad():
            model.weight.add_(1)
        torch.save(model, path)
    else:
        state = json.loads(path.read_text())
        if changed == "manifest.json":
            state["metrics"]["step"] = 1
        else:
            state["step"] = 1
        path.write_text(json.dumps(state))
    store = FakeStore()
    with pytest.raises(ValueError, match="differ|disagree"):
        finalize_completed_run(run, report_path, store)
    assert not store.calls and not report_path.exists()


def test_upload_retry_is_bounded_and_does_not_change_run_identity(monkeypatch):
    import helpers.walkerTrainingState as recovery
    monkeypatch.setattr(recovery.time, "sleep", lambda duration: None)
    class Flaky(FakeStore):
        def upload_run(self, *args, **kwargs):
            result = super().upload_run(*args, **kwargs)
            if len(self.calls) < 3:
                raise ConnectionError("temporary")
            return result
    store = Flaky()
    assert upload_checkpoint(store, "walker2d", "lewm-a", Path("run"), run_id="run", final=True) == "b" * 40
    assert len(store.calls) == 3 and len(set(store.calls)) == 1


def test_S2_rejects_changed_interval_before_loading_data(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/scripts"))
    import walker_s2_train_lewm as entry
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.yaml").write_text("push_every: 4000\n")
    def no_dataset(*args, **kwargs):
        raise AssertionError("Cadence mismatch must fail before loading data or optimizing")
    monkeypatch.setattr(entry, "make_dataset", no_dataset)
    monkeypatch.setattr(entry, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["walker_s2_train_lewm.py", "--run-id", "fresh",
        "--resume-from", str(source / "optimizer.pt"), "--push-every", "2000"])
    with pytest.raises(ValueError, match="original --push-every 4000"):
        entry.main()
    assert not (tmp_path / "runs/fresh").exists()
