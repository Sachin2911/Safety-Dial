"""Verify fresh-process matched inputs before running reserved comparisons."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unisafe_adaptation_evaluate import matched_prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.batch / "summary.json").read_text())
    assert summary["status"] == "complete" and len(summary["jobs"]) == 2
    paths = {j["name"]: Path(j["directory"]) / "episode-0000" for j in summary["jobs"]}
    baseline = next(p for n, p in paths.items() if n.startswith("baseline-"))
    candidate = next(p for n, p in paths.items() if not n.startswith("baseline-"))
    result = {
        **matched_prefix(baseline, candidate),
        "baseline": str(baseline), "candidate": str(candidate),
        "environment_steps": summary["completed_environment_steps"],
        "physics_steps": summary["completed_physics_steps"],
        "limits": "Engineering acceptance on development seed 8; not a generalization estimate or arbitrary simulator snapshot.",
    }
    (args.batch / "prefix_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    assert result["pass"], "Fresh-process inputs differ; investigate before reserved paired claims"


if __name__ == "__main__":
    main()
