"""Serial stage supervisor with durable exits and timing, no evaluation retuning."""

import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[2]
name, stages = sys.argv[1], sys.argv[2:]
out = root / "outputs/lewm-preset-expansion-20261030" / name
out.mkdir(parents=True, exist_ok=True)
record = out / ("pipeline-" + "-".join(stages) + ".json")
rows = []
for stage in stages:
    started = time.time()
    row = {"stage": stage, "started_epoch": started, "status": "running"}
    rows.append(row)
    record.write_text(json.dumps(rows, indent=2) + "\n")
    # Import Triton's native extension before OSMesa loads its LLVM symbols.
    # The isolated Cube import path otherwise segfaults before environment creation.
    bootstrap = (
        "import triton,runpy,sys; "
        "sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
    )
    code = subprocess.call(
        [
            sys.executable,
            "-X",
            "faulthandler",
            "-u",
            "-c",
            bootstrap,
            str(root / "experiments/scripts/run_portable_lewm.py"),
            "--env",
            name,
            "--stage",
            stage,
        ],
        cwd=root,
    )
    row.update(
        exit_code=code,
        elapsed_seconds=time.time() - started,
        status="complete" if code == 0 else "failed",
    )
    record.write_text(json.dumps(rows, indent=2) + "\n")
    if code:
        sys.exit(code)
