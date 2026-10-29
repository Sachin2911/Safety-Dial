"""Run a mechanical reset repair with the original PushT panels and job definitions."""

import hashlib
from pathlib import Path
import sys
import time

import numpy as np

from campaign_lewm18 import OUT, ROOT, main as campaign_main, read, run_stage
from panels_lewm18 import freeze
from helpers.dsrl06_infra import atomic_json, utc
from helpers.safecem10_planner import pd_response

ORIGINAL = Path("/workspace/Safety-Dial/outputs/lewm18")


def main():
    assert ROOT.name == "Safety-Dial-lewm18r1"
    assert not (OUT / "engineering-pusht-freeze.json").exists()
    panel = "pusht-panels.json"
    assert (OUT / panel).read_bytes() == (ORIGINAL / panel).read_bytes()
    old_engineering = read(ORIGINAL / "engineering-pusht-freeze.json")
    manifest = freeze(
        "engineering-pusht",
        old_engineering["jobs"],
        {
            "repair": "Use returned post-reset state for the first checker anchor",
            "panel_sha256": hashlib.sha256((OUT / panel).read_bytes()).hexdigest(),
            "original_engineering_sha256": hashlib.sha256(
                (ORIGINAL / "engineering-pusht-freeze.json").read_bytes()
            ).hexdigest(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    )
    # The two campaigns share one GPU. Keep serial timing uncontended by waiting
    # for the entire Cube campaign before starting this repaired cohort.
    while not (ORIGINAL / "campaign-complete-cube.json").exists():
        if (OUT / "STOP").exists():
            raise RuntimeError("Budget guard stopped the queue before repair execution")
        time.sleep(15)
    rows = run_stage(manifest)
    checks = []
    for row in rows:
        assert not row["error"] and not row["censored"]
        assert row["resources"]["reset_physics_steps"] == 2
        assert row["resources"]["reset_render_frames"] == 1
        initial = np.asarray(row["initial_state"])
        dataset = np.asarray(row["dataset_initial_state"])
        np.testing.assert_allclose(initial[:2], dataset[:2] + 0.01 * dataset[-2:], atol=1e-7)
        trace = np.load(OUT / "attempts" / row["job"]["id"] / "trace.npz")
        if row["actions"]:
            c, g = pd_response(row["actions"])
            predicted = initial[:2] + c[:, None] * initial[-2:] + g @ trace["actions"]
            error = float(np.abs(predicted - trace["dense_xy"]).max())
            assert error < 1e-6, (row["job"]["id"], error)
        else:
            error = 0.0
        checks.append({"job": row["job"]["id"], "max_dense_error": error})
    atomic_json(
        OUT / "reset-repair-audit.json",
        {
            "utc": utc(),
            "pass": True,
            "checks": checks,
            "extra_simulator_actions": 0,
            "note": "Audited recorded engineering traces; no extra rollouts",
        },
    )
    # Fix the same full final panel prospectively, as required by the repair
    # admission. Do not resize the experiment after the original final outcomes.
    admission = read(ORIGINAL / "schedule-admission-pusht.json")
    assert admission["selected"]["tier"]["name"] == "full"
    admission["original_admission_utc"] = admission["utc"]
    admission["utc"] = utc()
    admission["repair"] = "Same full panel, seeds and candidates as the original cohort"
    atomic_json(OUT / "schedule-admission-pusht.json", admission)
    sys.argv = ["campaign_lewm18.py", "run", "--env", "pusht"]
    campaign_main()


if __name__ == "__main__":
    main()
