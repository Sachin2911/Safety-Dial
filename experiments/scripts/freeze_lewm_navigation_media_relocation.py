"""Freeze metadata-only media readiness amendments for the explicit relocation.

Only analysis/readiness/provenance fields change. Showcase and plotting functions
must have exactly the original source text. No final evidence arrays are opened.
"""

import argparse
import ast
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_PLAN_SHA = "e43271d8aff95350779e90e6ffff525060d584a29ba333e19de56c6171286e53"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def reference(path):
    return {"path": str(path.resolve().relative_to(ROOT)), "sha256": digest(path)}


def freeze(args):
    require(not args.out.exists(), "Never overwrite an original or amended media freeze")
    require(digest(args.original_plan) == ORIGINAL_PLAN_SHA, "Original media plan changed")
    old = json.loads(args.original_plan.read_text())
    analysis = json.loads(args.analysis_plan.read_text())
    require(
        old["schema"] == "lewm-navigation-scientific-media-plan-v2"
        and analysis["schema"] == "lewm-navigation-analysis-plan-20261031-v1"
        and analysis["version"] == 4
        and analysis["execution_enabled"] is True,
        "Require original media plan and enabled relocation-aware V4 analysis",
    )
    require(
        digest(ROOT / analysis["analysis_script"]) == analysis["analysis_script_sha256"],
        "Analysis source changed",
    )
    for name, expected in analysis["source_sha256"].items():
        require(digest(ROOT / name) == expected, "Frozen analysis dependency changed")
    sources = [
        ROOT / "experiments/scripts/render_lewm_navigation_scientific_media_v2.py",
        ROOT / "experiments/scripts/render_lewm_navigation_scientific_media_v3.py",
    ]
    functions = []
    for path in sources:
        source = path.read_text()
        functions.append(
            {
                node.name: ast.get_source_segment(source, node)
                for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef)
            }
        )
    preserved = {}
    for name, body in functions[0].items():
        if name in ("main", "complete_inputs"):
            continue
        require(functions[1].get(name) == body, "Showcase/plot arithmetic changed: " + name)
        preserved[name] = hashlib.sha256(body.encode()).hexdigest()
    result = copy.deepcopy(old)
    result.update(
        schema="lewm-navigation-scientific-media-plan-v3",
        frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        original_media_plan=reference(args.original_plan),
        analysis_plan=reference(args.analysis_plan),
        allowed_pod_ids=analysis["execution"]["allowed_pod_ids"],
        final_execution_relocation=analysis["final_execution_relocation"],
        unchanged_original_function_source_sha256=preserved,
        amendment_reason="Availability-driven actual-host/deadline continuation only. Original Nav repeat0 and first admitted case per condition remain fixed; every original main/aux profile and exact continuation authority must validate before arrays. Restore original showcase evidence outside canonical results only after own-host accounting.",
    )
    result["source_sha256"].update(analysis["source_sha256"])
    result["source_sha256"][analysis["analysis_script"]] = analysis["analysis_script_sha256"]
    result["source_sha256"][str(sources[1].relative_to(ROOT))] = digest(sources[1])
    result["source_sha256"][str(Path(__file__).resolve().relative_to(ROOT))] = digest(__file__)
    result["future_command"] = (
        "python experiments/scripts/render_lewm_navigation_scientific_media_v3.py --plan "
        + str(args.out)
        + " --inputs <V4-bound-inputs.json> --artifact-root "
        "<verified-assembled-evidence-root-after-own-accounting> --out <fresh-media-namespace>"
    )
    result["actual_final_data_read_during_plan_creation"] = 0
    result["new_model_simulator_queries_during_plan_creation"] = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return reference(args.out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-plan", type=Path, required=True)
    parser.add_argument(
        "--original-plan",
        type=Path,
        default=ROOT / "configs/lewm_campaign_20261031/navigation-scientific-media-plan-v2.json",
    )
    parser.add_argument("--out", type=Path, required=True)
    print(json.dumps(freeze(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
