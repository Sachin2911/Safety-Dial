"""Freeze a metadata-only V4 plan from the unchanged V3 plan/input binding."""

import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.lewm_navigation_final_relocation import authority, require, validate_binding


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reference(path):
    path = path.resolve()
    require(path.is_relative_to(ROOT), "Metadata reference must remain in canonical project")
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-plan", type=Path, required=True)
    parser.add_argument("--original-inputs", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--extra-source", type=Path, action="append", default=[])
    args = parser.parse_args()
    require(not args.out.exists(), "Never overwrite a plan")
    original = json.loads(args.original_plan.read_text())
    inputs = json.loads(args.original_inputs.read_text())
    require(
        original["version"] == 3
        and inputs["schema"] == "lewm-navigation-analysis-inputs-20261031-v3"
        and inputs["plan"] == reference(args.original_plan)
        and inputs["all_final_output_namespaces_absent_at_binding"] is True,
        "Exact original pre-final V3 plan and input binding required",
    )
    require(digest(args.authority) == args.authority_sha256, "Authority SHA differs")
    relocation_ref = reference(args.authority)
    value = authority(ROOT, relocation_ref)
    require(
        value["original_analysis_inputs"] == reference(args.original_inputs),
        "Continuation and plan bind different original inputs",
    )
    contract = validate_binding(ROOT, inputs, value)
    sources = [
        ROOT / "experiments/helpers/lewm_navigation_final_relocation.py",
        ROOT / "experiments/scripts/run_lewm_navigation_relocated_final.py",
        ROOT / original["analysis_script"],
        Path(__file__),
        args.launcher,
        *args.extra_source,
    ]
    result = copy.deepcopy(original)
    result.update(
        version=4,
        execution_enabled=True,
        frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        analysis_script="experiments/scripts/analyze_lewm_navigation_campaign_20261031_v4.py",
        original_analysis_plan=reference(args.original_plan),
        original_analysis_inputs=reference(args.original_inputs),
        final_execution_relocation=relocation_ref,
        continuation_contract=contract,
        amendment_reason=(
            "Explicit new user authority after original-host GPU-capacity failures. "
            "Existing three Nav core outcomes are known and retained unchanged. All remaining "
            "profiles move prospectively together for availability, without outcome-based "
            "selection, new learning or statistical-method change. Original V3 bytes remain intact."
        ),
        availability_continuation_scope=(
            "Capacity-driven continuation: three completed Nav profiles stay on their original "
            "host; the remaining core/auxiliary profiles execute on one new A40. Training and "
            "final hosts differ for relocated controllers. Auxiliary references are the lowest "
            "final repeat on the actual final host at the same endpoint, with exact common "
            "case/seeds checked prospectively. Repeated auxiliary blocks on the same new host "
            "are retained and are not independent hosts. Conditional fixed-controller case "
            "statistics remain unchanged; no host-population or pooled speed inference."
        ),
        new_relocated_final_outcomes_read_during_plan_creation=0,
        new_model_simulator_calls=0,
    )
    result["analysis_script_sha256"] = digest(ROOT / result["analysis_script"])
    result["source_sha256"].update(
        {str(path.resolve().relative_to(ROOT)): digest(path) for path in sources}
    )
    result["source_sha256"].update(value["source_sha256"])
    new_pod = value["actual_final_host"]["pod_id"]
    result["execution"]["allowed_pod_ids"] = sorted(
        {*result["execution"]["allowed_pod_ids"], new_pod}
    )
    result["resource_reporting"] += (
        " Continuation adds a fourth physical source pod: retain exactly one authoritative "
        "owning-host report per source pod, charge only new owning-host work, and never "
        "recount restored foreign artifacts. Preserve all legacy missing-counter caveats."
    )
    result["original_inference_scope"] = result["inference_scope"]
    result["inference_scope"] = (
        "Conditional on all three originally selected fixed controllers per endpoint, their "
        "unchanged training/selection pipeline, and the explicitly recorded ACTUAL final hosts. "
        "Original training-host labels are retained separately. Case bootstrap estimates only "
        "source-case uncertainty, never future-learner/host uncertainty. Relocated repeats on "
        "one shared new host do not become independent host replicates."
    )
    result["preserved_v3_synthetic_validation"] = result.pop("synthetic_validation", None)
    result["continuation_software_validation"] = (
        "New bounded guards require separate retained qualification."
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"plan": str(args.out), "sha256": digest(args.out), "new_outcomes_read": 0}))


if __name__ == "__main__":
    main()
