"""Install an isolated published AP runtime on the existing rented host."""

import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/pusht-ap-reproduction-20261102"


def main():
    assert socket.gethostname() == "c45db82630e6"
    if len(sys.argv) == 1:
        assert not (OUT / "setup-launch.json").exists()
        with (OUT / "setup.log").open("xb") as log:
            proc = subprocess.Popen(
                [sys.executable, __file__, "execute"], cwd=ROOT,
                env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
                stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True,
            )
        record = dict(pid=proc.pid, started=time.time(), rental_shutdown="manual")
        (OUT / "setup-launch.json").write_text(json.dumps(record) + "\n")
        print(json.dumps(record))
        return
    assert sys.argv[1:] == ["execute"]
    target = ROOT / "venv-ap-reproduction"
    assert not target.exists()
    commands = [
        [sys.executable, "-m", "venv", str(target)],
        [str(target / "bin/python"), "-m", "pip", "install", "--no-cache-dir",
         "torch==2.7.1", "torchvision==0.22.1", "--index-url",
         "https://download.pytorch.org/whl/cu126"],
        [str(target / "bin/python"), "-m", "pip", "install", "--no-cache-dir", "-r",
         str(ROOT / "assets/ap-reference/code/ap/requirements/native.txt")],
    ]
    for command in commands:
        assert shutil.disk_usage(ROOT).free > 2 * 1024**3
        subprocess.run(command, cwd=ROOT, check=True, timeout=3600)
    subprocess.run(
        [str(target / "bin/python"), "-m", "pip", "check"], check=True, timeout=60
    )
    with (OUT / "installed-packages.txt").open("x") as stream:
        subprocess.run(
            [str(target / "bin/python"), "-m", "pip", "freeze"],
            stdout=stream, check=True, timeout=60,
        )
    (OUT / "setup-complete.json").write_text(json.dumps(dict(completed=time.time())) + "\n")


if __name__ == "__main__":
    main()
