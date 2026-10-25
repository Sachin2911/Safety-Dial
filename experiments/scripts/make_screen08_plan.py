"""Freeze all task-8 schedules, implementation and scoring before simulator outcomes."""

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.screen08_protocol import protocol


def main():
    preparation = json.loads((ROOT / "progress/receipts/08-screen-preparation.json").read_text())
    paths = {ROOT / name for name in preparation["source_sha256"]}
    for pattern in ("experiments/helpers/screen08_*.py", "experiments/scripts/*screen08*",
                    "experiments/tests/test_screen08*.py", "configs/screen08/*.yaml"):
        paths.update(ROOT.glob(pattern))
    paths.update(ROOT / name for name in (
        "experiments/helpers/accounted_recorder.py", "experiments/helpers/tracking.py",
        "progress/receipts/08-screen-preparation.json",
    ))
    plan = protocol()
    plan.update({
        "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "authorization": {"user_launch": True, "scope": "Task-8 execution screen"},
        "money": preparation["money"],
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(paths) if p.is_file()},
    })
    path = ROOT / "progress/receipts/08-screen-plan.json"
    with path.open("x") as stream:
        json.dump(plan, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"plan_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "sources": len(plan["source_sha256"]),
                      "engineering": len(plan["engineering_jobs"]),
                      "stage_a": len(plan["jobs"]), "potential_stage_b": len(plan["stage_b_jobs"])}))


if __name__ == "__main__":
    main()
