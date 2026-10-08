"""Scientific reporting regressions: horizon attribution and controlled S5 inputs."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "scripts"))

from helpers.walkerComparison import completed_s4, load_predictor_variant, pinned_input, probe_sample_plan, verify_matched_training  # noqa: E402
from helpers.walkerProtocol import file_sha256  # noqa: E402
from helpers.walkerReporting import cumulative_clearance, decomposition_report, merge_frozen_sources  # noqa: E402
from walker_s5_compare import paired_fsa, write_summary  # noqa: E402


def attribution_rows():
    rows = []
    trajectories = [
        {"dense": [-1., -1.], "endpoint": [1., 1.], "real_readout": [1., 1.], "imagined": [1., 1.]},
        {"dense": [1., -1.], "endpoint": [1., -1.], "real_readout": [1., 1.], "imagined": [1., 1.]},
        {"dense": [1., -1.], "endpoint": [1., -1.], "real_readout": [1., -1.], "imagined": [1., 1.]},
    ]
    for i, sources in enumerate(trajectories):
        row = {"branch": i, "root": i // 2, "root_id": f"r{i // 2}",
               "kind": "policy" if i == 0 else "random", "speed_regime": "near_limit",
               "health_regime": "interior" if i == 0 else "near_band"}
        for source, values in sources.items():
            row[f"cmin_by_block_{source}_health"] = values
            row[f"cmin_{source}_health"] = values[-1]
        rows.append(row)
    return rows


def test_horizon_attribution_uses_cumulative_dense_truth():
    result = decomposition_report(attribution_rows(), "health", n_boot=20)
    first = result["by_horizon"]["1"]
    second = result["by_horizon"]["2"]
    assert first["n_unsafe"] == 1
    assert first["attribution"] == {"n_false_safe_imagined": 1, "temporal": 1, "readout": 0, "imagination": 0}
    assert second["attribution"] == {"n_false_safe_imagined": 3, "temporal": 1, "readout": 1, "imagination": 1}
    assert result["overall"]["n_source_episodes"] == 2
    for strata in result["by_regime"].values():
        assert sum(group["overall"]["n"] for group in strata.values()) == 3
    assert result["by_regime"]["kind"]["policy"]["by_horizon"]["1"]["attribution"]["temporal"] == 1


def test_early_violation_is_not_lost_at_later_endpoint():
    dense = np.ones(20)
    dense[3] = -2.
    assert cumulative_clearance(dense) == [-2., -2.]
    with pytest.raises(ValueError, match="Nonfinite"):
        cumulative_clearance([np.nan] * 10)


def test_frozen_readout_merge_requires_identical_tapes():
    reference = attribution_rows()
    rows = [{k: v for k, v in row.items() if "real_readout" not in k} for row in reference]
    merged = merge_frozen_sources(rows, reference)
    assert merged == reference
    rows[0]["root_id"] = "a different source"
    with pytest.raises(ValueError, match="paired"):
        merge_frozen_sources(rows, reference)


def completed_report():
    report = {"status": "complete", "acquisition_mode": "prospective",
              "development_gate": {"go": True}, "planned_budgets": [2, 4],
              "acquisition_seeds": [0, 1], "adaptation": {}}
    for arm in ("random", "boundary"):
        report["adaptation"][arm] = {str(seed): [
            {"budget": b, "charged_steps": 1000 + 100 * b,
             "selected_ids": [f"{arm}-{seed}-{i}" for i in range(b)]} for b in [2, 4]] for seed in (0, 1)}
    return report


def test_comparison_requires_all_s4_seeds_and_equal_charged_budgets():
    report = completed_report()
    assert completed_s4(report, 0) == 4
    del report["adaptation"]["boundary"]["1"]
    with pytest.raises(ValueError, match="Incomplete"):
        completed_s4(report, 0)
    report = completed_report()
    report["adaptation"]["random"]["0"][-1]["charged_steps"] += 1
    with pytest.raises(ValueError, match="matched charged"):
        completed_s4(report, 0)
    report = completed_report()
    report["status"] = "running"
    with pytest.raises(ValueError, match="completed"):
        completed_s4(report, 0)


def test_fresh_probe_plan_matches_frames_and_keeps_episodes_whole():
    plan_a = probe_sample_plan([30, 40, 50, 60], 60, seed=7)
    plan_b = probe_sample_plan([30, 40, 50, 60], 60, seed=7)
    assert plan_a == plan_b
    assert not set(plan_a["training_episodes"]) & set(plan_a["validation_episodes"])
    assert plan_a["n_frames"] == sum(len(p["indices"]) for p in plan_a["parts"])
    assert len({p["episode"] for p in plan_a["parts"]}) == len(plan_a["parts"])
    with pytest.raises(ValueError, match="two source"):
        probe_sample_plan([10000], 10)


def test_predictor_variant_cannot_change_the_frozen_representation(tmp_path):
    base = torch.nn.Module()
    for name in ("encoder", "obs_proj", "action_encoder", "predictor", "pred_proj"):
        setattr(base, name, torch.nn.Linear(2, 2))
    allowed = {k: v.clone() + 1 for k, v in base.state_dict().items()
               if k.split(".")[0] in {"action_encoder", "predictor", "pred_proj"}}
    torch.save(allowed, tmp_path / "weights.pt")
    (tmp_path / "manifest.json").write_text(json.dumps({"data": {"model_sha256": "base"}}))
    variant = load_predictor_variant(base, tmp_path, "base")
    assert torch.equal(base.encoder.weight, variant.encoder.weight)
    assert torch.equal(base.obs_proj.weight, variant.obs_proj.weight)
    assert torch.equal(base.predictor.weight + 1, variant.predictor.weight)
    allowed["encoder.weight"] = torch.zeros_like(base.encoder.weight)
    torch.save(allowed, tmp_path / "weights.pt")
    with pytest.raises(ValueError, match="no encoder changes"):
        load_predictor_variant(base, tmp_path, "base")
    with pytest.raises(ValueError, match="different base"):
        load_predictor_variant(base, tmp_path, "other")


def test_comparison_pairs_source_tapes_before_bootstrap():
    rows_a = attribution_rows()
    rows_b = copy.deepcopy(rows_a)
    for row in rows_b:
        row["cmin_imagined_health"] = -1.
    ci = paired_fsa(rows_a, rows_b, "health", 0., 0., 20)
    assert np.isnan(ci["point"])  # no accepted plans is undefined, never perfect safety
    rows_b[0]["branch"] = 19
    with pytest.raises(ValueError, match="identical evaluation"):
        paired_fsa(rows_a, rows_b, "health", 0., 0., 20)


def test_inputs_need_pinned_receipts_unless_explicitly_local(tmp_path):
    (tmp_path / "weights.pt").write_bytes(b"weights")
    with pytest.raises(ValueError, match="receipt"):
        pinned_input(tmp_path, ["weights.pt"])
    assert pinned_input(tmp_path, ["weights.pt"], allow_local=True)["hf"] is None
    (tmp_path / "hf_upload.json").write_text(json.dumps({"repo_id": "private/model", "revision": "abc123", "path": "models/example"}))
    assert pinned_input(tmp_path, ["weights.pt"])["hf"]["revision"] == "abc123"


def test_matched_pretraining_rejects_optimizer_budget_drift(tmp_path):
    a, b, data = [tmp_path / n for n in ("a", "b", "data")]
    for path in (a, b, data):
        path.mkdir()
    (data / "setB.h5").write_bytes(b"data")
    sha_b = file_sha256(data / "setB.h5")
    (a / "manifest.json").write_text(json.dumps({"data": {"sha256": "data-a"}}))
    (b / "manifest.json").write_text(json.dumps({"data": {"sha256": sha_b}}))
    (data / "manifest.json").write_text(json.dumps({"data": {"selection_sha256": "selection", "setA_sha256": "data-a", "setB_sha256": sha_b},
        "metrics": {"ordinary_steps_replaced": 400, "branch_steps_added": 400, "total_A_steps": 1000, "total_B_steps": 1000}}))
    config = {"batch": 128, "lr": 1e-3, "wd": 1e-4, "warmup": 10, "epochs": 10, "seed": 0, "clip_stride": 1, "model": {"width": 192}}
    for path in (a, b):
        (path / "config.yaml").write_text(json.dumps(config))
        (path / "config.json").write_text(json.dumps(config["model"]))
        (path / "train_state.json").write_text(json.dumps({"step": 100}))
        np.savez(path / "scalers.npz", action_mean=np.zeros(6), action_std=np.ones(6))
    assert verify_matched_training(a, b, data, 4, "selection")["optimizer_steps_each"] == 100
    (b / "train_state.json").write_text(json.dumps({"step": 101}))
    with pytest.raises(ValueError, match="optimizer budgets"):
        verify_matched_training(a, b, data, 4, "selection")


def test_report_renders_undefined_acceptance_without_safety_claim(tmp_path):
    report = {"variants": {"A": {name: {"health": {"at_matched": {"n_false_safe": 0, "n_accepted": 0, "fsa": float("nan"), "acceptance_rate": 0.}}} for name in ("test", "stress")}},
              "contrasts": {}, "training_control": {"pretraining_steps_each": 1000, "optimizer_steps_each": 100, "additional_experience_steps": 400}}
    write_summary(tmp_path / "README.md", report)
    text = (tmp_path / "README.md").read_text()
    assert "undefined" in text
    assert "0 / 0" in text


def test_analysis_records_a_dense_transient_that_endpoints_miss():
    from types import SimpleNamespace
    from helpers.walkerRules import WalkerRoot
    from walker_s4_study import analyse

    qpos = np.zeros((101, 9))
    qpos[:, 1] = 1.2
    qpos[4, 1] = 0.5
    root = WalkerRoot("dev-e0-t20", 0, 20, qpos[0], np.zeros(9),
        np.repeat(qpos[0:1], 3, axis=0), np.zeros((3, 9)), np.zeros((2, 10, 6)),
        np.zeros((100, 6)), {"x_velocity": 0.})
    bank = SimpleNamespace(roots=[root], dir=Path("dev"), indices_for_root=lambda _: np.array([0]),
        h5={"tape": np.zeros((1, 100, 6)), "qpos": qpos[None], "x_velocity": np.zeros((1, 100)),
            "kind": np.array(["policy"]), "frames": np.zeros((1, 11, 1, 1, 3), dtype=np.uint8)})
    class Imaginer:
        def encode(self, frames):
            return torch.tensor([[1.2, 0., 0.]] * len(frames))
        def rollout(self, history, actions, tapes):
            return torch.tensor([[[1.2, 0., 0.]] * 10] * len(tapes))
    class Probe:
        def predict(self, z):
            return np.asarray(z)
    ctx = SimpleNamespace(render_many=lambda qpos, qvel: np.zeros((len(qpos), 1, 1, 3), dtype=np.uint8))
    rows = analyse(bank, Imaginer(), Probe(), ctx, rules=("health",))
    assert len(rows[0]["cmin_by_block_dense_health"]) == 10
    assert rows[0]["cmin_by_block_dense_health"][0] < 0
    assert rows[0]["cmin_by_block_endpoint_health"][0] > 0
    report = decomposition_report(rows, "health", n_boot=5)
    assert report["by_horizon"]["1"]["attribution"]["temporal"] == 1
    assert report["by_horizon"]["10"]["attribution"]["temporal"] == 1


def test_evaluation_rejects_render_stack_drift(tmp_path, monkeypatch):
    import h5py
    from helpers import locoData
    from helpers.walkerValidation import verify_render_fingerprint

    with h5py.File(tmp_path / "source.h5", "w") as source:
        source.attrs["render_fingerprint"] = "original"
    monkeypatch.setattr(locoData, "render_fingerprint", lambda _: "original")
    assert verify_render_fingerprint(object(), tmp_path / "source.h5") == "original"
    monkeypatch.setattr(locoData, "render_fingerprint", lambda _: "changed")
    with pytest.raises(ValueError, match="fingerprint"):
        verify_render_fingerprint(object(), tmp_path / "source.h5")


def test_fresh_probe_fitting_shares_targets_roles_and_capacity(tmp_path, monkeypatch):
    import h5py
    from types import SimpleNamespace
    import walker_s5_compare as runner
    from helpers.poseProbes import ProbeSpec

    source_path = tmp_path / "probe.h5"
    with h5py.File(source_path, "w") as f:
        f["ep_len"] = [4, 4, 4]
        f["ep_offset"] = [0, 4, 8]
        qpos = np.zeros((12, 9))
        qpos[:, 1] = np.repeat([1., 2., 3.], 4)
        f["qpos"], f["qvel"], f["x_velocity"] = qpos, np.zeros((12, 9)), np.arange(12)
    recorded = []
    class Encoder:
        def __init__(self, model, scaler, device):
            self.offset = model
        def encode(self, frames):
            return torch.tensor(np.repeat(frames[:, None], 192, axis=1) + self.offset)
    class Probe:
        def parameters(self):
            return iter([])
        def eval(self):
            return self
    def train(z_train, y_train, z_val, y_val, spec, **kwargs):
        recorded.append((z_train.copy(), y_train.copy(), z_val.copy(), y_val.copy(), copy.deepcopy(spec)))
        return Probe(), {"val": {"rmse": 0.}}
    monkeypatch.setattr(runner, "WalkerImaginer", Encoder)
    monkeypatch.setattr(runner, "fit_probe", train)
    monkeypatch.setattr(runner, "save_probe", lambda probe, stats, path: path.write_bytes(b"probe"))
    monkeypatch.setattr(runner, "build_manifest", lambda **kwargs: {"run_id": kwargs["run_id"]})
    plan = probe_sample_plan([4, 4, 4], 6)
    ctx = SimpleNamespace(render_many=lambda qp, qv: qp[:, 1])
    runner.fit_fresh_probes({"A": 0., "B": 10.}, {"A": None, "B": None}, source_path,
        plan, ProbeSpec(target="walker"), ctx, tmp_path, {"A": {"hf": None}, "B": {"hf": None}}, None, "cpu")
    assert len(recorded) == 2
    np.testing.assert_array_equal(recorded[0][1], recorded[1][1])
    np.testing.assert_array_equal(recorded[0][3], recorded[1][3])
    np.testing.assert_array_equal(recorded[0][0] + 10., recorded[1][0])
    assert recorded[0][4] == recorded[1][4]
    assert (tmp_path / "probe-A/sample_plan.json").read_text() == (tmp_path / "probe-B/sample_plan.json").read_text()
