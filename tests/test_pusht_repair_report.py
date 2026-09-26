"""Saved E2 evidence remains paired, deterministic and explicit about censored futures."""
import copy
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from helpers.dialMetrics import fsa
from helpers.pushtRepairReport import (
    attach_observed_trace, build_paired_report, paired_rows, select_examples, write_repair_report,
)


def saved_rows():
    # One row per category; last two demonstrate unresolved vs already-known unsafe.
    truth = [(True, False), (True, False), (True, False), (False, False),
             (False, False), (False, True), (True, True)]
    before_clearance = [1., -1., 1., -1., 1., -1., 1.]
    after_clearance = [-1., 1., 1., 1., -1., 1., -1.]
    before, after = [], []
    for i, ((unsafe, censored), ca, cb) in enumerate(zip(truth, before_clearance, after_clearance, strict=True)):
        row = {"bank": "test", "root": i, "branch": i, "layout": "familiar", "kind": "random",
            "cmin_dense": -1. if unsafe else 1., "unsafe_composite": unsafe, "censored": censored,
            "observation_domain_exit": i == 6, "observed_steps": 7 if censored else 25,
            "horizon_steps": 25, "cmin_imagined": ca}
        before.append(row)
        after.append({**row, "cmin_imagined": cb})
    return before, after


def repair_fixture():
    before, after = saved_rows()
    rows = {"no_update": {"test": before}, "adapted": {"test": after},
            "readout_correction": {"test": copy.deepcopy(before)}}
    arms = {}
    for arm, entries in [*rows.items(), ("fixed_margin", rows["no_update"])]:
        rr = entries["test"]
        m = .5 if arm == "fixed_margin" else 0.
        metric = fsa([r["cmin_imagined"] for r in rr], [r["unsafe_composite"] for r in rr], m,
                     censored=[r["censored"] for r in rr])
        arms[arm] = {("margin" if arm == "fixed_margin" else "margin_matched_dev"): m,
            "test": {"at_matched": metric, "ordinary_motion": {}, "auc_dial": float("nan")}}
    arms["adapted"]["test"]["paired_fsa_diff_ci"] = {"point": float("nan"), "lo": -.5, "hi": -.1, "n_boot": 44}
    return {"run_id": "unit-e2", "gate": {"passes": False}, "arms": arms}, rows


def test_pairing_uses_complete_identity_not_saved_order():
    before, after = saved_rows()
    pairs = paired_rows(before[::-1], after[2:] + after[:2])
    assert [p["key"][2] for p in pairs] == list(range(7))
    assert pairs[0]["before_index"] == 6
    assert pairs[0]["after_index"] == 5
    for bad in (after[:-1], after + after[:1]):
        with pytest.raises(ValueError):
            paired_rows(before, bad)
    changed = copy.deepcopy(after)
    changed[0]["layout"] = "heldout"
    with pytest.raises(ValueError, match="identical paired"):
        paired_rows(before, changed)
    changed = copy.deepcopy(after)
    changed[0]["censored"] = True
    with pytest.raises(ValueError, match="observed truth"):
        paired_rows(before, changed)


def test_selection_is_sorted_and_never_treats_unseen_suffix_as_safe():
    before, after = saved_rows()
    selections = select_examples(paired_rows(before[::-1], after[::-1]), 0., 0.)
    assert selections["corrected_false_safe"]["n_qualifying"] == 2
    assert selections["corrected_false_safe"]["example"]["row_id"]["branch_index"] == 0
    assert selections["unresolved_accepted_future"]["example"]["row_id"]["branch_index"] == 5
    assert selections["restored_safe_acceptance"]["n_qualifying"] == 1
    assert selections["new_false_rejection"]["n_qualifying"] == 1
    # A known arena violation remains unsafe even when the later suffix is unobserved.
    last = select_examples(paired_rows(before[6:], after[6:]), 0., 0.)
    assert last["corrected_false_safe"]["n_qualifying"] == 1
    assert last["unresolved_accepted_future"]["example"] is None
    assert last["restored_safe_acceptance"]["example"] is None


def test_summary_preserves_failed_gate_all_counts_and_recorded_margins():
    repair, rows = repair_fixture()
    report = build_paired_report(repair, rows)
    assert report["gate"] == {"passes": False}
    metrics = report["banks"]["test"]["arms"]["adapted"]["at_matched"]
    assert metrics["n"] == 7 and metrics["n_accepted_censored"] == 1
    assert np.isnan(metrics["fsa"]) and metrics["fsa_upper"] > metrics["fsa_lower"]
    assert report["banks"]["test"]["arms"]["fixed_margin"]["margin"] == .5
    repair["arms"]["adapted"]["test"]["at_matched"]["n"] = 99
    with pytest.raises(ValueError, match="metrics disagree"):
        build_paired_report(repair, rows)


def make_bank(tmp_path, rows):
    bank = tmp_path / "banks" / "test"
    bank.mkdir(parents=True)
    roots = [{"root_id": f"r{i}", "meta": {"episode": 10 + i}} for i in range(len(rows))]
    layouts = [{"root_id": r["root_id"], "family": "familiar",
                "hazard": {"kind": "disc", "cx": 250., "cy": 250., "r": 20.}} for r in roots]
    (bank / "roots.json").write_text(json.dumps({"roots": roots}))
    (bank / "layouts.json").write_text(json.dumps({"layouts": layouts}))
    with h5py.File(bank / "branches.h5", "w") as h5:
        h5["root_index"] = np.arange(len(rows))
        states = np.tile([100., 100., 200., 180., 0., 0., 0.], (len(rows), 26, 1))
        h5["states"] = states
        observed = np.ones((len(rows), 26), bool)
        observed[5:, 8:] = False
        h5["observed"] = observed
        h5["observation_valid"] = observed
        h5["frames"] = np.zeros((len(rows), 6, 8, 8, 3), np.uint8)
    return bank, roots, {(lay["root_id"], lay["family"]): lay for lay in layouts}


def test_only_observed_valid_frames_are_illustrated(tmp_path):
    before, after = saved_rows()
    bank, roots, layouts = make_bank(tmp_path, before)
    example = select_examples(paired_rows(before, after), 0., 0.)["unresolved_accepted_future"]["example"]
    with h5py.File(bank / "branches.h5", "r") as h5:
        frames = attach_observed_trace(example, bank, roots, layouts, h5)
    assert example["trace"]["observed_state_steps"] == list(range(8))
    assert len(example["trace"]["states"]) == 8
    assert [step for step, _ in frames] == [0, 5]
    assert example["trace"]["unobserved_steps"] == list(range(8, 26))


def test_failed_gate_still_writes_complete_cpu_report(tmp_path):
    repair, rows = repair_fixture()
    bank, _, _ = make_bank(tmp_path, rows["no_update"]["test"])
    repair["banks_dir"] = str(bank.parent)
    report_file, rows_file = tmp_path / "repair.json", tmp_path / "repair_rows.npz"
    report_file.write_text(json.dumps(repair))
    np.savez_compressed(rows_file, rows=json.dumps(rows))
    original = report_file.read_bytes(), rows_file.read_bytes()
    output = tmp_path / "paired"
    result = write_repair_report(report_file, rows_file, bank.parent, output)
    saved = json.loads(result.read_text())
    assert saved["gate"]["passes"] is False
    assert saved["banks"]["test"]["arms"]["adapted"]["at_matched"]["fsa"] is None
    assert len(list(output.glob('*.png'))) == 6
    text = (output / "paired_report.md").read_text()
    assert "contrast is undefined" in text and "No qualifying example" not in text
    assert (output / "manifest.json").exists()
    assert original == (report_file.read_bytes(), rows_file.read_bytes())
    with pytest.raises(FileExistsError):
        write_repair_report(report_file, rows_file, bank.parent, output)
