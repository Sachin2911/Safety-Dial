"""E4 reports preserve geometry, censoring, pair identity and the H4 contrast sign."""
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from helpers.pushtSourceFamilies import SOURCE_FAMILY_PROTOCOL
from helpers.pushtTransferReport import (
    CELLS, cell_statistics, check_acquisition, load_rows, paired_statistics,
    source_metadata, validate_layouts, validate_rows,
)


def roots():
    result = []
    for i, family in enumerate(("familiar", "familiar", "heldout", "heldout")):
        xy = [64., 64.] if family == "familiar" else [192., 64.]
        state = [100., 100., *xy, 0., 0., 0.]
        result.append({"root_id": f"r{i}", "goal_state": state,
                       "meta": {"state_at_root": state, "source_family": family,
                                "source_family_protocol": SOURCE_FAMILY_PROTOCOL["version"], "episode": i}})
    return result


def rows(arm="random"):
    result = []
    for root in range(4):
        for tape, unsafe in enumerate((False, True)):
            for layout in ("familiar", "heldout"):
                familiar = root < 2 and layout == "familiar"
                predicted = -1. if arm == "boundary" and familiar and unsafe else 1.
                result.append({"root": root, "branch": 2 * root + tape, "layout": layout,
                    "cmin_imagined": predicted, "cmin_dense": -1. if unsafe else 1.,
                    "unsafe_composite": unsafe, "censored": False, "arena_exit": False})
    return result


def test_geometry_labels_are_verified_not_episode_parity_assumptions():
    data = roots()
    assert source_metadata(data)[2]["family"] == "heldout"
    data[2]["meta"]["source_family"] = "familiar"
    with pytest.raises(ValueError, match="actual geometric"):
        source_metadata(data)


def test_frozen_hazard_geometry_and_two_by_two_layout_coverage():
    sources = source_metadata(roots())
    layouts = [{"root_id": source["root_id"], "family": family,
                "hazard": {"kind": "disc", "cx": 64. if family == "familiar" else 192.,
                           "cy": 64., "r": 20. if family == "familiar" else 30.}}
               for source in sources.values() for family in ("familiar", "heldout")]
    validate_layouts(layouts, sources)
    layouts[-1]["hazard"]["cx"] = 64.
    with pytest.raises(ValueError, match="Hazard geometry"):
        validate_layouts(layouts, sources)


def test_counts_are_per_cell_and_bootstrap_clusters_sources():
    source = source_metadata(roots())
    result = cell_statistics(rows(), source, 0., n_boot=40)
    assert set(result) == set(CELLS)
    for value in result.values():
        assert value["n"] == value["n_accepted"] == 4
        assert value["n_roots"] == value["n_source_episodes"] == 2
        assert value["n_false_safe"] == 2
        assert value["fsa"] == .5
    data = rows()
    data[0]["censored"] = True
    result = cell_statistics(data, source, 0., n_boot=40)[CELLS[0]]
    assert np.isnan(result["fsa"])
    assert result["fsa_lower"] == .5 and result["fsa_upper"] == .75
    assert result["n_accepted_censored"] == 1
    assert np.isnan(cell_statistics(rows(), source, 2., n_boot=40)[CELLS[0]]["fsa"])


def test_h4_shrink_contrast_is_paired_and_has_correct_direction():
    result = paired_statistics(rows(), rows("boundary"), source_metadata(roots()), 0., 0., n_boot=100)
    assert result["boundary_minus_random_fsa"][CELLS[0]]["point"] == -.5
    assert result["boundary_minus_random_fsa"][CELLS[-1]]["point"] == 0.
    for contrast in result["h4_advantage_change"].values():
        assert contrast["point"] == .5
        assert contrast["interpretation"] == "boundary_advantage_shrinks"
    corrupted = rows("boundary")
    corrupted[0]["unsafe_composite"] = True
    with pytest.raises(ValueError, match="observed truth"):
        paired_statistics(rows(), corrupted, source_metadata(roots()), 0., 0., n_boot=10)


def test_h4_does_not_claim_shrinking_advantage_when_none_is_demonstrated():
    result = paired_statistics(rows(), rows(), source_metadata(roots()), 0., 0., n_boot=40)
    assert all(c["interpretation"] == "no_demonstrated_familiar_cell_advantage"
               for c in result["h4_advantage_change"].values())


def test_row_hash_and_full_frozen_branch_coverage_are_required(tmp_path):
    path = tmp_path / "rows.npz"
    np.savez_compressed(path, rows=json.dumps({"test": rows()}))
    reference = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert load_rows(tmp_path, reference)["test"] == rows()
    data = rows()
    keys = {(r["root"], r["branch"], r["layout"]) for r in data}
    validate_rows(data, source_metadata(roots()), keys)
    with pytest.raises(ValueError, match="exactly once"):
        validate_rows(data[:-1], source_metadata(roots()), keys)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        load_rows(tmp_path, reference)


def test_incomplete_or_descriptive_only_study_cannot_claim_e4():
    report = {"status": "complete", "gate": {"e2_passed": True}, "split_audit": {"passes": True},
              "source_family_protocol": SOURCE_FAMILY_PROTOCOL, "rounds": [64, 64],
              "arms": {arm: {str(seed): {"curve": [{"budget_added": 64}, {"budget_added": 128}]}
                             for seed in range(3)} for arm in ("random", "boundary")}}
    assert check_acquisition(report) == (["0", "1", "2"], [64, 128])
    modified = copy.deepcopy(report)
    modified["arms"]["boundary"]["2"]["curve"].pop()
    with pytest.raises(ValueError, match="every planned"):
        check_acquisition(modified)
    report["source_family_protocol"] = {"version": "episode-split"}
    with pytest.raises(ValueError, match="qualified geometric"):
        check_acquisition(report)


def test_cpu_entrypoint_writes_separate_complete_transfer_artifacts(tmp_path, monkeypatch):
    import h5py
    from helpers.pushtReplay import Root
    from helpers.pushtRetention import RETENTION_PROTOCOL, TERMINATION_POLICY, case_identity
    from helpers.runManifest import file_sha256
    from helpers.splitIntegrity import bank_identity
    from scripts import pusht_e4_report as entrypoint

    bank_dir = tmp_path / "banks"
    root_data = roots()
    for name in ("dev", "test", "stress"):
        folder = bank_dir / name
        folder.mkdir(parents=True)
        rr = copy.deepcopy(root_data)
        if name == "dev":
            rr = rr[:1]
            rr[0]["root_id"] = "development"
            rr[0]["meta"]["episode"] = 100
        (folder / "roots.json").write_text(json.dumps({"roots": rr}))
        with h5py.File(folder / "branches.h5", "w") as bank:
            bank["root_index"] = np.repeat(np.arange(len(rr)), 2)
        layouts = [{"root_id": root["root_id"], "family": family,
                    "hazard": {"kind": "disc", "cx": 64. if family == "familiar" else 192.,
                               "cy": 64., "r": 20. if family == "familiar" else 30.}}
                   for root in rr for family in ("familiar", "heldout")]
        (folder / "layouts.json").write_text(json.dumps({"layouts": layouts}))
    cases = [Root(i, np.zeros(7), np.ones(7), np.zeros((10, 2)), f"retention-{i}",
                  {"episode": 200 + i, "role": "retention"}) for i in range(20)]
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps([case.to_dict() for case in cases]))
    digest = file_sha256(cases_path)
    retention_rows = [{"root_id": r.root_id, "episode": r.meta["episode"], "case_sha256": case_identity(r),
        "valid": True, "arena_exit": False, "censored": False, "censored_future": False,
        "terminated": False, "truncated": False, "completed_on_verified_goal": False,
        "requested_steps": 250, "executed_steps": 250, "final_coverage": .8,
        "final_pose_error_px": 15.} for r in cases]
    outcome = {"rows": retention_rows, "summary": {"n_blocks": 50, "steps_per_block": 5,
               "protocol": RETENTION_PROTOCOL, "termination_policy": TERMINATION_POLICY}}
    e2_path = tmp_path / "e2.json"
    e2_path.write_text(json.dumps({"goal_retention_cases": str(cases_path),
        "goal_retention_cases_sha256": digest, "goal_retention": {"n_blocks": 50}}))
    acquisition = {"run_id": "acq", "status": "complete", "gate": {"e2_passed": True},
        "split_audit": {"passes": True}, "source_family_protocol": SOURCE_FAMILY_PROTOCOL,
        "rounds": [64], "arms": {}, "banks_dir": str(bank_dir), "e2_report": str(e2_path),
        "e2_report_sha256": file_sha256(e2_path), "target_acceptance_rate_dev": 1.,
        "evaluation_bank_identity": {name: bank_identity(bank_dir / name) for name in ("dev", "test", "stress")},
        "no_update": {"margin_matched_dev": 0.}, "no_update_retention": {"latent_mse_tf": .01}}

    def save_rows(filename, arm):
        rr = rows(arm)
        for row in rr:
            row.update(kind="nominal", displacement_px=1., imag_displacement_px=1.,
                       rotation_deg=0., imag_rotation_deg=0.)
        path = tmp_path / filename
        np.savez_compressed(path, rows=json.dumps({"test": rr, "stress": rr}))
        return {"file": filename, "sha256": file_sha256(path)}

    acquisition["no_update_rows"] = save_rows("no-update-rows.npz", "random")
    retained = {"status": "complete", "acquisition_run": "acq", "case_file_sha256": digest,
                "n_blocks": 50, "checkpoints": {}}
    for arm in ("random", "boundary"):
        acquisition["arms"][arm] = {}
        for seed in ("0", "1", "2"):
            name = f"acq-{arm}-s{seed}-b64"
            weights_hash = f"weights-{arm}-{seed}"
            ref = save_rows(f"{arm}-s{seed}-b64-rows.npz", arm)
            acquisition["arms"][arm][seed] = {"curve": [{"budget_added": 64,
                "weights_sha256": weights_hash, "evaluation_rows": ref,
                "eval": {"margin_matched_dev": 0.}, "retention": {"latent_mse_tf": .01},
                "charged_steps": 6400}]}
            detail = tmp_path / f"{name}-retention.json"
            detail.write_text(json.dumps({"base": outcome, "adapted": outcome,
                "comparison": {"passes": True}, "repaired_weights_sha256": weights_hash,
                "case_file_sha256": digest, "n_blocks": 50}))
            retained["checkpoints"][name] = {"report": str(detail), "weights_sha256": weights_hash}
    acquisition_path, retained_path = tmp_path / "acquisition.json", tmp_path / "retention.json"
    acquisition_path.write_text(json.dumps(acquisition))
    retained_path.write_text(json.dumps(retained))
    output = tmp_path / "e4"
    monkeypatch.setattr(entrypoint, "build_manifest", lambda **kwargs: kwargs)
    monkeypatch.setattr(sys, "argv", ["e4", "--acquisition-report", str(acquisition_path),
        "--retention-report", str(retained_path), "--output-dir", str(output), "--bootstrap", "100"])
    assert entrypoint.main() == 0
    result = json.loads((output / "transfer.json").read_text())
    assert result["status"] == "complete"
    assert result["additional_simulator_steps"] == result["additional_model_queries"] == 0
    assert set(result["no_update"]) == {"test", "stress"}
    assert len(result["no_update"]["test"]["cells"]) == 4
    assert result["arms"]["boundary"]["0"]["64"]["goal_retention"]["status"] == "measured"
    assert (output / "transfer.md").is_file()
    assert (output / "transfer.png").stat().st_size > 0
    assert (output / "manifest.json").is_file()


@pytest.mark.parametrize("censored_cell", ["reference", "heldout"])
def test_undefined_full_data_contrast_cannot_gain_direction_from_usable_resamples(censored_cell, tmp_path):
    from helpers.pushtTransferReport import h4_conclusion
    from scripts.pusht_e4_report import plain_json, write_markdown

    random, boundary = rows(), rows("boundary")
    selected_root = 0 if censored_cell == "reference" else 2
    selected_layout = "familiar" if censored_cell == "reference" else "heldout"
    for data in (random, boundary):
        selected = next(row for row in data if row["root"] == selected_root
                        and row["layout"] == selected_layout and not row["unsafe_composite"])
        selected["censored"] = True  # Accepted, but its unobserved future may be unsafe.
    sources = source_metadata(roots())
    result = paired_statistics(random, boundary, sources, 0., 0., n_boot=200, seed=0)
    reference = result["boundary_minus_random_fsa"][CELLS[0]]
    change = result["h4_advantage_change"][CELLS[-1]]
    assert np.isnan(change["point"])
    # Replicates omitting the censored source still give an apparently clear shift.
    assert np.isfinite(change["lo"]) and np.isfinite(change["hi"])
    assert change["lo"] > 0
    assert 0 < change["n_boot_usable"] < change["n_boot_requested"] == 200
    assert change["n_boot_dropped"] == 200 - change["n_boot_usable"]
    assert change["point_defined"] is False
    assert change["interval_defined"] is True
    assert change["contrast_direction"] == change["interpretation"] == "undefined_outcome"
    if censored_cell == "reference":
        assert np.isnan(reference["point"]) and reference["hi"] < 0
        assert change["reference_boundary_advantage_demonstrated"] is False
    else:
        assert np.isfinite(reference["point"]) and reference["hi"] < 0
        assert change["reference_boundary_advantage_demonstrated"] is True

    paired = {"64": {"0": {"test": result, "stress": result}}}
    cells = cell_statistics(random, sources, 0., n_boot=40)
    report = {"run_id": "censored-e4", "no_update": {bank: {"cells": cells} for bank in ("test", "stress")},
              "arms": {}, "h4": h4_conclusion(paired, ["0"], 64), "paired_boundary_random": paired}
    output = tmp_path / "transfer.md"
    write_markdown(report, output)
    markdown = output.read_text()
    assert "Bootstrap usable/requested (FSA; AR)" in markdown
    assert f"{change['n_boot_usable']}/200" in markdown
    assert "undefined_outcome" in markdown
    saved = plain_json(result)
    assert saved["h4_advantage_change"][CELLS[-1]]["point"] is None
    assert saved["h4_advantage_change"][CELLS[-1]]["n_boot_dropped"] > 0
    json.dumps(saved, allow_nan=False)


def test_finite_full_data_point_without_usable_interval_is_not_directional_evidence():
    result = paired_statistics(rows(), rows("boundary"), source_metadata(roots()),
                               0., 0., n_boot=1, seed=4)
    reference = result["boundary_minus_random_fsa"][CELLS[0]]
    change = result["h4_advantage_change"][CELLS[-1]]
    assert reference["point"] == -.5 and reference["point_defined"]
    assert reference["n_boot_usable"] == 0 and not reference["interval_defined"]
    assert change["point"] == .5 and change["point_defined"]
    assert change["interpretation"] == "insufficient_bootstrap_evidence"
    assert change["contrast_direction"] == "insufficient_bootstrap_evidence"
    assert change["reference_boundary_advantage_demonstrated"] is False
