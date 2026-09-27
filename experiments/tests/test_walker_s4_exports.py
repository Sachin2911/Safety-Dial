"""Already-computed S4 row exports preserve evidence without model/simulator imports."""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/walker_s4_prospective.py"
# Load just this filesystem-only function so these tests cannot initialize a renderer/model.
module = ast.parse(SOURCE.read_text())
function = next(node for node in module.body
                if isinstance(node, ast.FunctionDef) and node.name == "save_evaluation_rows")
namespace = {"Path": Path, "hashlib": hashlib, "json": json}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
save_evaluation_rows = namespace["save_evaluation_rows"]


def destinations(tmp_path):
    dirs = [tmp_path / name for name in ("run", "results", "bank")]
    for directory in dirs:
        directory.mkdir()
    return dirs


def full_rows():
    return {bank: [{"bank": bank, "root_id": "root-2", "branch": 2,
                    "cmin_imagined_health": -0.0, "source_values": [1.25, None]},
                   {"bank": bank, "root_id": "root-1", "branch": 1,
                    "cmin_imagined_health": 0.125, "source_values": [2.5]}]
            for bank in ("dev", "test", "stress")}


def test_all_three_destinations_retain_exact_rows_order_and_referenced_bytes(tmp_path):
    dirs = destinations(tmp_path)
    rows = full_rows()
    original = deepcopy(rows)
    reference = save_evaluation_rows(dirs, "baseline_decomposition_rows.json", rows)
    payloads = [(d / reference["file"]).read_bytes() for d in dirs]
    assert payloads[0] == payloads[1] == payloads[2]
    assert reference["sha256"] == hashlib.sha256(payloads[0]).hexdigest()
    assert reference["bytes"] == len(payloads[0])
    assert json.loads(payloads[0]) == original == rows
    assert [r["branch"] for r in json.loads(payloads[0])["test"]] == [2, 1]
    assert b"-0.0" in payloads[0]


def test_baseline_and_no_update_evaluations_remain_separate_artifacts(tmp_path):
    dirs = destinations(tmp_path)
    baseline, evaluated = full_rows(), full_rows()
    evaluated["test"][0]["source_values"] = [9.0]
    base_ref = save_evaluation_rows(dirs, "baseline_decomposition_rows.json", baseline)
    eval_ref = save_evaluation_rows(dirs, "no_update_evaluation_rows.json", evaluated)
    assert base_ref["sha256"] != eval_ref["sha256"]
    for directory in dirs:
        assert json.loads((directory / base_ref["file"]).read_text()) == baseline
        assert json.loads((directory / eval_ref["file"]).read_text()) == evaluated


def test_serialization_is_deterministic_without_reordering_rows(tmp_path):
    dirs = destinations(tmp_path)
    rows = full_rows()
    first = save_evaluation_rows(dirs, "one.json", rows)
    reordered = {bank: [dict(reversed(list(row.items()))) for row in items]
                 for bank, items in reversed(list(rows.items()))}
    second = save_evaluation_rows(dirs, "two.json", reordered)
    assert first["sha256"] == second["sha256"]


def test_development_only_export_does_not_invent_final_test_evidence(tmp_path):
    dirs = destinations(tmp_path)
    rows = full_rows()["dev"]
    rows[0]["undefined_value"] = float("nan")
    ref = save_evaluation_rows(dirs, "development_rows.json", rows)
    for directory in dirs:
        saved = json.loads((directory / ref["file"]).read_text())
        assert isinstance(saved, list) and {r["bank"] for r in saved} == {"dev"}
        assert math.isnan(saved[0]["undefined_value"])
        assert [p.name for p in directory.iterdir()] == ["development_rows.json"]


def test_existing_destination_is_preserved_before_any_mirror_is_written(tmp_path):
    dirs = destinations(tmp_path)
    old = dirs[-1] / "rows.json"
    old.write_bytes(b"original evidence\n")
    with pytest.raises(FileExistsError, match="Preserving"):
        save_evaluation_rows(dirs, "rows.json", full_rows())
    assert old.read_bytes() == b"original evidence\n"
    assert not (dirs[0] / "rows.json").exists()
    assert not (dirs[1] / "rows.json").exists()


@pytest.mark.parametrize("filename", ["../escape.json", "/tmp/escape.json", "rows.csv"])
def test_export_rejects_paths_outside_the_bundle(filename, tmp_path):
    dirs = destinations(tmp_path)
    with pytest.raises(ValueError, match="single JSON filename"):
        save_evaluation_rows(dirs, filename, [])
    assert all(not list(directory.iterdir()) for directory in dirs)
