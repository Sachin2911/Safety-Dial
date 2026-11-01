"""Derive the frozen eight-case stress95 panel from JSON metadata, on the rented host."""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.lewm_navigation_stress import derive_case


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(protocol_path):
    start = time.monotonic()
    protocol = json.loads(protocol_path.read_text())
    if os.environ.get("RUNPOD_POD_ID") != protocol["execution_pod_id"]:
        raise RuntimeError("Stress-radius derivation belongs on the declared rented host")
    if protocol["schema"] != "lewm-navigation-stress-preparation-v1":
        raise ValueError("Unknown stress preparation protocol")
    for name, expected in protocol["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Frozen stress source changed: " + name)
    for name, expected in protocol["input_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Frozen JSON input changed: " + name)
    original = json.loads((ROOT / protocol["parent_panel_path"]).read_text())
    source = {case["case_id"]: case for case in original["cases"]}
    cases = []
    for case_id in protocol["parent_case_ids"]:
        case = source[case_id]
        if case["split"] != "development_training":
            raise ValueError("This stage admits only the original eight training pilot cases")
        cases.append(derive_case(case, protocol["geometry_rule"]))
    if len(cases) != 8 or len({case["case_id"] for case in cases}) != 8:
        raise ValueError("All eight original pilot cases must be retained exactly once")
    output = ROOT / protocol["output_namespace"]
    if output.exists():
        raise FileExistsError("Use a fresh stress metadata namespace")
    output.mkdir(parents=True)
    panel = {
        "schema": "lewm-navigation-stress-panel-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol_sha256": digest(protocol_path),
        "source_sha256": protocol["source_sha256"],
        "parent_panel": {
            "path": protocol["parent_panel_path"],
            "sha256": protocol["input_sha256"][protocol["parent_panel_path"]],
        },
        "parent_case_preparation": protocol["parent_case_preparation"],
        "geometry_transform": protocol["geometry_rule"],
        "cases": cases,
        "pilot_case_ids": [case["case_id"] for case in cases],
        "condition_semantics": "condition/original_condition retain the ORIGINAL radius stratum; each actual stress radius is explicit in case.shape.radius and stress_derivation",
        "sealed_final_episode_ids": original["sealed_final_episode_ids"],
        "final_pixels_read": 0,
        "selection_geometry_derived": False,
        "new_pixels_read": 0,
        "resources": {
            "environment_resets": 0,
            "environment_actions": 0,
            "render_calls": 0,
            "encoder_queries": 0,
            "predictor_queries": 0,
            "metadata_cases_derived": 8,
        },
        "elapsed_seconds": time.monotonic() - start,
        "limitations": "Endpoint admission and a12px radius cap do not prove global route feasibility. All original cases are retained without outcome filtering or replacement.",
    }
    (output / "panel.json").write_text(json.dumps(panel, indent=2, allow_nan=False) + "\n")
    (output / "source-freeze.json").write_text(
        json.dumps(
            {"protocol_sha256": digest(protocol_path), "sources": protocol["source_sha256"]},
            indent=2,
        )
        + "\n"
    )
    return {
        "panel": str(output / "panel.json"),
        "sha256": digest(output / "panel.json"),
        "cases": 8,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.protocol), indent=2))


if __name__ == "__main__":
    main()
