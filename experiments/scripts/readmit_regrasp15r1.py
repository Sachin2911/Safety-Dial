"""Re-admit the task 15 recovery stage C from the new host's measured engineering speed.

The speed factor is the new host's total elapsed time for the warm stage A episodes
(jobs 1 to 3; job 0 is the cold start) divided by what the original-host model predicts
for those same episodes' steps and candidate requests. admit_regrasp15r1.admit then
re-simulates the remaining schedule against the live book. Stage C is admitted only if
all 96 of its episodes fit with the 45-minute margin; stage D's predicted denominator is
recorded. The admission is written locally and copied atomically to the rental.

    .venv/bin/python experiments/scripts/readmit_regrasp15r1.py [--push]
"""

import argparse
import datetime as dt
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/scripts"))
import admit_regrasp15r1 as admit  # noqa: E402

MIRROR = ROOT / "outputs/regrasp15r1/remote"
RENTAL = ROOT / "progress/rental-regrasp15r1-20261026.json"
FREEZE = ROOT / "progress/receipts/15-regrasp-recovery-live-v1.json"
RECOVERY = ROOT / "configs/regrasp15r1/recovery.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measured_speed(model):
    events = [json.loads(x) for x in (MIRROR / "attempts_stage_a.jsonl").read_text()
              .splitlines() if x]
    done = [e for e in events if e["event"] == "validated"]
    warm = [e for e in done if e["job"] != 0]
    if len(warm) < 3:
        raise SystemExit("Need the three warm stage A episodes")
    actual = sum(e["elapsed_s"] for e in warm)
    predicted = sum(model["startup_s"] + model["per_step_s"] * e["outcome"]["steps"]
                    + model["per_request_s"] * (e["adapter"]["closest_passing"]
                                                + e["adapter"]["least_excess"])
                    for e in warm)
    episodes = [{"job": e["job"], "arm": e["arm"], "elapsed_s": e["elapsed_s"],
                 "steps": e["outcome"]["steps"],
                 "requests": e["adapter"]["closest_passing"] + e["adapter"]["least_excess"],
                 "median_decode_s": e["adapter"]["median_decode_s"]} for e in done]
    return actual / predicted, episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--allowance-seconds", type=float, default=300)
    args = parser.parse_args()
    rental = json.loads(RENTAL.read_text())
    rows = admit.historical_rows()
    model, loads = admit.fit_model(rows), admit.workload(rows)
    speed, episodes = measured_speed(model)
    now = dt.datetime.now(dt.timezone.utc)
    elapsed = (now - dt.datetime.fromisoformat(rental["created_utc"])).total_seconds()
    recovery = json.loads(RECOVERY.read_text())
    remaining = {"stage_c_jobs": recovery["stage_c_jobs"],
                 "stage_d_jobs": recovery["stage_d_jobs"]}
    result = admit.admit(rate=rental["offer"]["dph_total"],
                         setup_s=elapsed + args.allowance_seconds, engineering_s=0.0,
                         speed=speed, replacements=2, book_path=admit.BOOK, recovery=remaining,
                         model=model, loads=loads, elapsed_s=elapsed)
    admission = {
        "schema": "regrasp15-recovery-readmission-v1", "utc": now.isoformat(),
        "admit_stage_c": result["stage_c_complete"],
        "predicted_stage_d_admitted": result["admitted"].get("stage_d", 0),
        "freeze_sha256": sha256(FREEZE), "instance_id": rental["instance_id"],
        "speed_factor": speed, "engineering_episodes": episodes,
        "elapsed_since_create_s": elapsed, "model": model, "prediction": result,
    }
    text = json.dumps(admission, indent=2) + "\n"
    (ROOT / "outputs/regrasp15r1/admission-c.json").write_text(text)
    (ROOT / "progress/receipts/15-regrasp-recovery-readmission.json").write_text(text)
    print(text)
    if args.push and admission["admit_stage_c"]:
        remote = "/workspace/Safety-Dial/outputs/regrasp15r1"
        command = (f"cat > {remote}/.admission-c.tmp && "
                   f"mv {remote}/.admission-c.tmp {remote}/admission-c.json && "
                   f"sha256sum {remote}/admission-c.json")
        subprocess.run([sys.executable, str(ROOT / "experiments/scripts/manage_regrasp15r1.py"),
                        "ssh", f"bash -c {shlex.quote(command)} <<'JSON'\n{text}JSON"],
                       check=True)


if __name__ == "__main__":
    main()
