"""Wait for an audited, packaged study, then render its saved-evidence report."""

import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    assert len(sys.argv) in [2, 3] and sys.argv[1] in ["dial", "evolution"]
    kind = sys.argv[1]
    assert socket.gethostname() == ("075cb160c9b3" if kind == "dial" else "8cf3a706cdd1")
    name = "safety-dial-library-20261102" if kind == "dial" else "evolution-budget-20261102"
    base = ROOT / "outputs" / name
    control = ROOT / "outputs/safety-dial-library-launch-20261102" if kind == "dial" else base
    complete = "pipeline-complete.json" if kind == "dial" else "packaging-complete.json"
    launch = "launch.json" if kind == "dial" else "packaging-launch.json"
    process = "launch_dial_library.py" if kind == "dial" else "package_evolution_when_ready.py"
    if len(sys.argv) == 2:
        assert not (base / "closeout-launch.json").exists()
        with (base / "closeout-watch.log").open("xb") as log:
            child = subprocess.Popen(
                [sys.executable, __file__, kind, "watch"],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        (base / "closeout-launch.json").write_text(
            json.dumps(dict(pid=child.pid, started=time.time())) + "\n"
        )
        print(json.dumps(dict(pid=child.pid)), flush=True)
        return
    assert sys.argv[2] == "watch"
    handle = json.loads((control / launch).read_text())
    until = time.monotonic() + (5 if kind == "dial" else 9) * 3600
    while not (control / complete).exists():
        path = Path("/proc") / str(handle["pid"]) / "cmdline"
        if not path.exists() or process.encode() not in path.read_bytes():
            raise RuntimeError("Study pipeline ended without audited packaging; no retry")
        if time.monotonic() > until:
            raise TimeoutError("Report-watcher observation limit; rental unchanged")
        time.sleep(30)
    assert json.loads((base / "run-v1/audit.json").read_text())["passed"]
    subprocess.run(
        [sys.executable, "experiments/scripts/render_development_closeout.py", kind],
        cwd=ROOT,
        check=True,
    )
    archive = ROOT / "outputs" / (name + "-closeout.tar.gz")
    assert not archive.exists()
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(base / "closeout", arcname="closeout")
    with archive.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    (base / "closeout-complete.json").write_text(
        json.dumps(
            dict(
                completed=time.time(),
                archive=str(archive),
                sha256=sha,
                bytes=archive.stat().st_size,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
