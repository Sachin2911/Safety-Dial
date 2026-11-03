"""Collect and verify rented-host reports without local experiment computation."""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["dial", "evolution"])
    args = parser.parse_args()
    kind = args.kind
    name = "safety-dial-library-20261102" if kind == "dial" else "evolution-budget-20261102"
    helper = ROOT / (
        "outputs/ppo-20261101/remote.py"
        if kind == "dial"
        else "outputs/runpod-setup-20261101/remote.py"
    )
    remote = "/workspace/SafeLeWM/outputs/" + name
    local = ROOT / "outputs" / name
    local.mkdir(parents=True, exist_ok=True)

    def call(*arguments, **kwargs):
        return subprocess.run(
            [sys.executable, str(helper), *arguments], check=True, **kwargs
        )

    code = (
        "import json;from pathlib import Path;p=Path(" + repr(remote) + ");"
        "h=json.loads((p/'closeout-launch.json').read_text());"
        "f=Path('/proc')/str(h['pid'])/'cmdline';"
        "live=f.exists() and b'finish_development_closeout.py' in f.read_bytes();"
        "print(json.dumps(dict(live=live,complete=(p/'closeout-complete.json').exists())))"
    )
    deadline = time.monotonic() + (5 if kind == "dial" else 9) * 3600
    while True:
        response = call("exec", "python -c " + shlex.quote(code), capture_output=True, text=True)
        state = json.loads(response.stdout.strip().splitlines()[-1])
        if state["complete"]:
            break
        if not state["live"]:
            raise RuntimeError("Report watcher ended without completion; no experiment restart")
        if time.monotonic() > deadline:
            raise TimeoutError("Collection observation limit; rentals unchanged")
        time.sleep(60)

    receipt_path = local / "closeout-complete.json"
    call("get", remote + "/closeout-complete.json", str(receipt_path))
    receipt = json.loads(receipt_path.read_text())
    assert receipt["archive"] == remote + "-closeout.tar.gz"
    archive = local / "closeout.tar.gz"
    if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest() != receipt["sha256"]:
        partial = archive.with_suffix(".partial")
        call("get", receipt["archive"], str(partial))
        assert partial.stat().st_size == receipt["bytes"]
        assert hashlib.sha256(partial.read_bytes()).hexdigest() == receipt["sha256"]
        partial.replace(archive)
    assert archive.stat().st_size == receipt["bytes"]

    # Read only named regular files. Never extract archive paths or links directly.
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        assert len(names) == len(set(names))
        manifest_member = tar.getmember("closeout/manifest.json")
        assert manifest_member.isfile()
        manifest_bytes = tar.extractfile(manifest_member).read()
        manifest = json.loads(manifest_bytes)
        expected = manifest["files"]
        assert all(Path(n).name == n and n not in (".", "..") for n in expected)
        assert set(names) == {"closeout", "closeout/manifest.json"} | {
            "closeout/" + n for n in expected
        }
        assert tar.getmember("closeout").isdir()
        verified = {}
        for name, entry in expected.items():
            member = tar.getmember("closeout/" + name)
            assert member.isfile() and member.size == entry["bytes"]
            data = tar.extractfile(member).read()
            assert hashlib.sha256(data).hexdigest() == entry["sha256"]
            verified[name] = data

    destination = local / "closeout"
    destination.mkdir(exist_ok=True)
    for name, data in {**verified, "manifest.json": manifest_bytes}.items():
        target = destination / name
        if target.exists():
            assert target.read_bytes() == data, "Existing report differs; preserve it for review"
        else:
            target.write_bytes(data)
    verification = dict(
        verified_at=time.time(),
        archive=str(archive.relative_to(ROOT)),
        sha256=receipt["sha256"],
        bytes=receipt["bytes"],
        files=len(verified),
        verification="Archive SHA256 and every report member SHA256 checked",
        editorial_review_pending=True,
    )
    (local / "closeout-local-preservation.json").write_text(
        json.dumps(verification, indent=2) + "\n"
    )
    print(json.dumps(verification), flush=True)


if __name__ == "__main__":
    main()
