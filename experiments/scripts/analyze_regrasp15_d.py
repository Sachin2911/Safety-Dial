"""Stage D contrast: declared projection arm (stage D) against the stage C native arm.

Primary: strict safe completion, projection minus native, one-sided exact Barnard test at
alpha 0.05 with Wilson intervals and a task-stratified bootstrap. Secondary: raw-violation
episodes, one-sided for fewer under projection. Reported separately from the frozen
stage C confirmation.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/scripts"))
import analyze_regrasp15 as base  # noqa: E402

DECLARATION = ROOT / "progress/receipts/15-regrasp-stage-d.json"


def violation_contrast(rows, arm):
    from scipy.stats import barnard_exact

    treated = [r for r in rows if r["arm"] == arm]
    native = [r for r in rows if r["arm"] == "native"]
    v_arm = sum(r["outcome"]["raw_violation"] for r in treated)
    v_nat = sum(r["outcome"]["raw_violation"] for r in native)
    # Columns are the samples (native, arm); "greater": native violates more often.
    table = [[v_nat, v_arm], [len(native) - v_nat, len(treated) - v_arm]]
    test = barnard_exact(table, alternative="greater")
    return {"arm_violations": [v_arm, len(treated)], "native_violations": [v_nat, len(native)],
            "barnard_one_sided_p": float(test.pvalue),
            "arm_wilson95": base.wilson(v_arm, len(treated)),
            "native_wilson95": base.wilson(v_nat, len(native))}


def main():
    declaration = json.loads(DECLARATION.read_text())
    arm = declaration["arm"]
    native = [r for r in base.episodes("stage_c") if r["arm"] == "native"]
    treated = base.episodes("stage_d")
    rows = native + treated
    result = {
        "declaration": declaration,
        "summary": base.summarize(rows),
        "primary": base.contrast(rows, arm),
        "violations": violation_contrast(rows, arm) if treated and native else None,
    }
    (base.OUT / "stage-d-analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("primary", "violations")}, indent=2))


if __name__ == "__main__":
    main()
