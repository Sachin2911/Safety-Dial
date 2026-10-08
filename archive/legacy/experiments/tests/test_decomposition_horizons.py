"""Supplementary prefix reporting preserves observed support and physical sample counts."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import decomposition as decomp


def fixture_branch():
    states = np.zeros((26, 7))
    states[:, :2] = 256
    states[:, 2:4] = [30, 40]
    return {"states": states, "observed": np.ones(26, bool), "observation_valid": np.ones(26, bool),
        "terminated": np.zeros(26, bool), "truncated": np.zeros(26, bool), "kind": "random",
        "n_contacts": np.zeros(26, int), "pusher_block_contacts": np.zeros(26, int),
        "block_wall_contacts": np.zeros(26, int), "contact_kind": "pusher_block", "contact_counter": "typed-test"}


def diagnostics(branch, *, offsets=(0,), clearances=None, angle_offset=0.):
    ends = branch["states"][::5, 2:5]
    poses = {source: ends.copy() for source in ("endpoint", "real_readout", "imagined")}
    poses["real_readout"][:, 0] += 1
    poses["imagined"][:, 0] += 2
    poses["imagined"][:, 2] += angle_offset
    if clearances is None:
        clearances = {source: np.full(26, 5. + delta) for source, delta in
            (("dense", 0), ("endpoint", 0), ("real_readout", 1), ("imagined", 2), ("stationary", 0))}
    layouts = {str(offset): {source: values - offset for source, values in clearances.items()} for offset in offsets}
    root = SimpleNamespace(meta={"episode": 42})
    return decomp.cumulative_branch_diagnostics("test", 0, 0, root, branch, {}, poses, layouts)


def test_late_censoring_preserves_shorter_prefix_and_never_scores_padded_truth():
    branch = fixture_branch()
    branch["observed"][8:] = False
    branch["truncated"][7] = True
    branch["pusher_block_contacts"][12:] = 9  # Unobserved counter padding is irrelevant.
    clearances = {source: np.full(26, 5.) for source in ("dense", "endpoint", "real_readout", "imagined", "stationary")}
    clearances["dense"][8:] = -1000  # Cannot manufacture an observed hazard.
    rows, poses = diagnostics(branch, clearances=clearances)
    short, longer = rows[:2]
    assert not short["censored"] and not short["truncated"]
    assert short["observed_steps"] == 5 and short["source_available"]["real_readout"]
    assert longer["censored"] and longer["truncated"] and longer["observed_steps"] == 7
    assert not longer["unsafe_composite"] and longer["cmin_dense"] == 5
    assert longer["contact"] is None and short["contact"] is False
    assert longer["cmin_real_readout"] is None and not longer["source_available"]["endpoint"]
    assert longer["source_available"]["imagined"]
    report = decomp.cumulative_horizon_report(rows, poses)["by_horizon"]
    assert report["1"]["pose_error"]["imagined"]["centre_px"]["n"] == 1
    assert report["2"]["pose_error"]["imagined"]["centre_px"]["n"] == 0
    table = report["2"]["decisions"]["at_m0"]
    assert table["sources"]["real_readout"]["n_available"] == 0
    assert np.isnan(table["sources"]["imagined"]["fsa"])
    assert table["sources"]["imagined"]["fsa_lower"] == 0
    assert table["sources"]["imagined"]["fsa_upper"] == 1
    assert report["2"]["clearance_error"]["imagined"]["n"] == 0
    assert "contact_unknown" in report["2"]["by_regime"]


def test_observed_failure_resolves_censoring_but_does_not_invent_attribution_chain():
    branch = fixture_branch()
    branch["observed"][8:] = False
    clearance = {source: np.ones(26) for source in ("dense", "endpoint", "real_readout", "imagined")}
    clearance["dense"][6] = 0.
    rows, poses = diagnostics(branch, clearances=clearance)
    row = rows[1]
    assert row["censored"] and row["unsafe_composite"]
    table = decomp.cumulative_horizon_report(rows, poses)["by_horizon"]["2"]["decisions"]["at_m0"]
    assert table["sources"]["imagined"]["fsa"] == 1
    assert table["sources"]["imagined"]["n_accepted_censored"] == 0
    assert table["attribution"]["censored"] == 1
    assert table["attribution"]["n_attributed"] == 0


def test_domain_exit_then_reentry_invalidates_later_targets_but_not_earlier_prefix():
    branch = fixture_branch()
    branch["states"][12, 0] = 513
    # Deliberately emulate old per-state validity: re-entry would otherwise look valid.
    rows, poses = diagnostics(branch)
    report = decomp.cumulative_horizon_report(rows, poses)["by_horizon"]
    assert not rows[1]["observation_domain_exit"] and rows[1]["observation_valid"]
    assert rows[2]["observation_domain_exit"] and not rows[2]["observation_valid"]
    assert rows[2]["source_available"]["endpoint"] and rows[2]["source_available"]["imagined"]
    assert not rows[2]["source_available"]["real_readout"]
    assert report["2"]["pose_error"]["imagined"]["n_branches"] == 1
    assert report["3"]["pose_error"]["imagined"]["n_branches"] == 0
    assert report["3"]["decisions"]["at_m0"]["attribution"]["domain_exit"] == 1
    assert report["3"]["clearance_error"]["imagined"]["n_rows_excluded"] == 1


def test_pose_counts_do_not_multiply_with_layouts_and_angle_error_wraps():
    branch = fixture_branch()
    branch["states"][:, 4] = np.deg2rad(179)
    rows, poses = diagnostics(branch, offsets=(0, 40), angle_offset=np.deg2rad(-358))
    report = decomp.cumulative_horizon_report(rows, poses + poses)["by_horizon"]["1"]
    assert report["n_rows"] == 2 and report["n_branches"] == report["n_source_episodes"] == 1
    assert report["pose_error"]["imagined"]["centre_px"]["n"] == 1
    assert report["pose_error"]["imagined"]["angle_deg"]["mean"] == pytest.approx(2)
    assert report["clearance_error"]["imagined"]["n"] == 2
    assert report["clearance_error"]["imagined"]["mean"] == pytest.approx(2)
    for group in ("free", "near_boundary(|c|<20)", "far(|c|>=20)"):
        assert report["by_regime"][group]["pose_error"]["imagined"]["n_branches"] == 1


def test_regime_contact_and_rotation_are_prefix_specific_and_unknown_stays_unknown():
    branch = fixture_branch()
    branch["states"][5:10, 4] = np.deg2rad(5)
    branch["states"][10:, 4] = np.deg2rad(20)
    branch["pusher_block_contacts"][10] = 1
    branch["block_wall_contacts"][2] = 5
    branch["n_contacts"][2] = 5
    rows, poses = diagnostics(branch)
    report = decomp.cumulative_horizon_report(rows, poses)["by_horizon"]
    assert "free" in report["1"]["by_regime"] and "contact" not in report["1"]["by_regime"]
    assert "rotation<10deg" in report["1"]["by_regime"]
    assert "contact" in report["2"]["by_regime"] and "rotation>=10deg" in report["2"]["by_regime"]
    branch["contact_kind"] = "any_collision"
    branch["pusher_block_contacts"] = None
    rows, poses = diagnostics(branch)
    groups = decomp.cumulative_horizon_report(rows, poses)["by_horizon"]["2"]["by_regime"]
    assert "contact_unknown" in groups and "free" not in groups and "contact" not in groups


def test_exact_zero_truth_is_unsafe_and_zero_margin_acceptance_is_inclusive():
    branch = fixture_branch()
    clearance = {source: np.zeros(26) for source in ("dense", "endpoint", "real_readout", "imagined")}
    rows, poses = diagnostics(branch, clearances=clearance)
    report = decomp.cumulative_horizon_report(rows, poses, matched_margin=1)["by_horizon"]["1"]
    assert rows[0]["unsafe_composite"]
    for source in clearance:
        assert report["decisions"]["at_m0"]["sources"][source]["fsa"] == 1
        assert report["decisions"]["at_matched"]["sources"][source]["n_accepted"] == 0
        assert report["clearance_error"][source]["mean"] == 0
        assert report["clearance_error"][source]["frac_optimistic"] == 0
    assert report["decisions"]["at_m0"]["n_exact_zero_dense_clearance"] == 1


@pytest.mark.parametrize("endpoint,real,link", [(1, 1, "temporal"), (-1, 1, "readout"), (-1, -1, "imagination")])
def test_each_prefix_attributes_to_the_first_available_failing_link(endpoint, real, link):
    branch = fixture_branch()
    clearance = {source: np.full(26, value) for source, value in
                 (("dense", -1.), ("endpoint", endpoint), ("real_readout", real), ("imagined", 1.))}
    rows, poses = diagnostics(branch, clearances=clearance)
    report = decomp.cumulative_horizon_report(rows, poses)["by_horizon"]
    for entry in report.values():
        attribution = entry["decisions"]["at_m0"]["attribution"]
        assert attribution[link] == 1
        assert sum(attribution[key] for key in ("temporal", "readout", "imagination")) == 1


def test_analyse_bank_preserves_legacy_rows_metrics_and_model_call_counts(monkeypatch):
    branch = fixture_branch()
    branch["states"][:, 2:5] = [2, 0, 0]
    branch["states"][2, 2] = -1
    branch.update(root_index=0, tape=np.zeros((5, 5, 2), np.float32), frames=np.tile([3., 0, 0], (6, 1)))
    root = SimpleNamespace(root_id="root", meta={"episode": 42})
    bank = SimpleNamespace(branch=lambda *args, **kwargs: branch, root_of=lambda j: root,
        h5={key: branch[key][None] for key in ("observed", "observation_valid", "terminated", "truncated")})
    bank_class = type("SyntheticBank", (), {"__len__": lambda self: 1})
    wrapped = bank_class()
    wrapped.__dict__.update(vars(bank))
    class Imaginer:
        encodes = rollouts = 0
        def encode(self, frames):
            self.encodes += 1
            return frames
        def rollout(self, *args):
            self.rollouts += 1
            return np.tile([4., 0, 0], (1, 5, 1))
    imaginer = Imaginer()
    probe = SimpleNamespace(predict_pose=lambda z: z)
    monkeypatch.setattr(decomp, "RootLatentCache", lambda *args: SimpleNamespace(get=lambda ri: (np.zeros((3, 3)), None, None), env=SimpleNamespace(close=lambda: None)))
    monkeypatch.setattr(decomp, "clearance_trace", lambda poses, hazard: poses[:, 0] - hazard)
    layouts = [SimpleNamespace(root_id="root", family=family, shape=offset) for family, offset in (("familiar", 0), ("heldout", 10))]
    result = decomp.analyse_bank("dev", wrapped, layouts, imaginer, probe, verbose=False)
    expected = []
    for family, offset in (("familiar", 0), ("heldout", 10)):
        expected.append({"bank": "dev", "branch": 0, "root": 0, "layout": family, "kind": "random",
            "contact": False, "contact_kind": "pusher_block", "contact_mechanism_identifiable": True,
            "any_collision": False, "block_wall_contact": False, "contact_counter": "typed-test",
            "rotation_deg": 0., "displacement_px": 0., "imag_displacement_px": 1., "imag_rotation_deg": 0.,
            "dense_argmin_step": 2, "cmin_dense": -1. - offset, "cmin_endpoint": 2. - offset,
            "cmin_real_readout": 3. - offset, "cmin_imagined": 3. - offset, "cmin_stationary": 2. - offset,
            "hazard_unsafe_observed": True, "hazard_boundary_convention": "clearance <= 0 includes contact",
            "hazard_only_unsafe": True, "hazard_censored": False, "observation_domain_exit": False,
            "arena_exit": False, "censored": False, "unsafe_composite": True, "observed_steps": 25,
            "horizon_steps": 25, "observation_valid": True, "terminated": False, "truncated": False,
            "termination_flags_recorded": True})
    assert result["rows"] == expected
    assert result["pose_err"]["imagined"]["centre_by_block"] == [1., 2., 2., 2., 2., 2.]
    assert result["pose_err"]["imagined"]["angle_by_block"] == [0.] * 6
    table = decomp.decision_table(result["rows"], 0., boot=0)
    assert table["imagined"]["fsa"] == 1 and table["attribution"]["temporal"] == 1
    assert len(result["horizon_rows"]) == 10 and len(result["diagnostic_pose_rows"]) == 5
    assert imaginer.encodes == imaginer.rollouts == 1


def test_diagnostic_figures_are_written_with_missing_prefix_support(tmp_path, monkeypatch):
    branch = fixture_branch()
    branch["observed"][8:] = False
    rows, poses = diagnostics(branch)
    report = {"banks": {"test": {"cumulative_horizon": decomp.cumulative_horizon_report(rows, poses)}}}
    path = Path(__file__).resolve().parents[1] / "scripts/pusht_e1_decompose.py"
    spec = importlib.util.spec_from_file_location("decompose_figure_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "RESULTS", tmp_path)
    module.make_diagnostic_figures(report)
    for name in ("attribution_by_cumulative_horizon.png", "regime_error_by_horizon.png"):
        assert (tmp_path / name).read_bytes().startswith(b"\x89PNG")
