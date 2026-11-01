"""Freeze native/strong-fixed stress comparison from already-derived JSON metadata."""

import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def freeze(base_path, panel_path, output_path):
    base = json.loads(base_path.read_text())
    panel = json.loads(panel_path.read_text())
    if panel["schema"] != "lewm-navigation-stress-panel-v1" or len(panel["cases"]) != 8:
        raise ValueError("The frozen eight-case rented-host derivation is required")
    if [case["parent_case_id"] for case in panel["cases"]] != base["pilot_case_ids"]:
        raise ValueError("All original pilot cases must be retained in their frozen order")
    if panel["selection_geometry_derived"] or panel["final_pixels_read"]:
        raise ValueError("This stage must leave selection geometry and final pixels untouched")
    for name, expected in base["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Parent source changed: " + name)
    for name, expected in panel["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Stress derivation source changed: " + name)
    result = copy.deepcopy(base)
    result.update(
        name="navigation-stress95-target-pilot-v1",
        frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        output_namespace="results/navigation-stress95-target-pilot-v1",
        purpose="Development-only capped metadata stress and useful-authority landscape pilot, with all original cases retained; no optimizer claim",
        parent_protocol={"path": str(base_path.relative_to(ROOT)), "sha256": digest(base_path)},
        metadata_panel={
            "path": str(panel_path.relative_to(ROOT)),
            "sha256": digest(panel_path),
            "parent_image_panel_sha256": base["case_preparation"]["panel_sha256"],
            "meaning": "Only virtual geometry/IDs/seeds change. Existing actual image inputs and their original visual review remain exact.",
        },
        geometry_transform=panel["geometry_transform"],
        pilot_case_ids=panel["pilot_case_ids"],
        source_selection="All original eight training pilot cases, including both none controls and all six hazard cases; no source/case replacement or outcome-based exclusion. Selection geometry is not derived in this stage.",
    )
    result["evaluation"]["asset_sha256"][str(panel_path.relative_to(ROOT))] = digest(panel_path)
    result["evaluation"]["allowed_source_episodes"] = [
        case["source_episode"] for case in panel["cases"]
    ]
    parameters = base["controller_static_parameters"]["fixed_geometry_progress"]
    result["jobs"] = []
    for case in panel["cases"]:
        for method in ("native", "fixed_geometry_progress"):
            result["jobs"].append(
                {
                    "job_id": method + "_" + case["case_id"] + "_s" + str(case["seed"]),
                    "method": method,
                    "case": copy.deepcopy(case),
                    "environment_seed": case["seed"],
                    "policy_seed": case["seed"],
                    "parameters": parameters.copy(),
                    "max_actions": 150,
                    "temperature": 1.0,
                    "repeat_id": 0,
                }
            )
    result["maximum_episode_starts"] = result["maximum_native_resets"] = 16
    result["maximum_environment_actions"] = 2400
    result["evaluation"]["maximum_episode_actions"] = 150
    result["pairing"] = (
        "Each derived stress case uses fresh seed321000+source_episode identically across native/strongfixed, same SMALL planner/checker/readout/body/margin/allowance and original goal. Original condition labels identify parent radius strata; use shape.radius for actual stress geometry."
    )
    result["comparison_limits"] = (
        "Architecture stress pilot only. The 12px cap is prospectively fixed to avoid obvious whole-room barrier geometry, without establishing global route feasibility. All cases are retained; do not replace difficult or infeasible cases after outcomes. No selection/final interventions or optimizer efficacy claim."
    )
    sources = sorted(
        set(
            base["source_files"]
            + list(panel["source_sha256"])
            + [str(Path(__file__).relative_to(ROOT))]
        )
    )
    result["source_files"] = sources
    result["source_sha256"] = {name: digest(ROOT / name) for name in sources}
    if output_path.exists():
        raise FileExistsError("Never overwrite a frozen stress comparison")
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return {"protocol": str(output_path), "sha256": digest(output_path), "jobs": 16}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        default=ROOT / "configs/lewm_campaign_20261031/navigation-progress-target-pilot-v2.json",
    )
    parser.add_argument(
        "--panel", type=Path, default=ROOT / "results/navigation-stress95-cases-v1/panel.json"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "configs/lewm_campaign_20261031/navigation-stress95-target-pilot-v1.json",
    )
    args = parser.parse_args()
    print(json.dumps(freeze(args.base.resolve(), args.panel.resolve(), args.out), indent=2))


if __name__ == "__main__":
    main()
