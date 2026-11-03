"""Transfer and byte-verify completed evidence; no local experiment computation."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

BASE = Path(__file__).resolve().parents[2]
assert sys.argv[1:] in [["evolution"], ["dial"]]
kind = sys.argv[1]
study = "evolution-budget-20261102" if kind == "evolution" else "safety-dial-library-20261102"
OUT = BASE / "outputs" / study
OUT.mkdir(exist_ok=True, parents=True)
HELPER = BASE / (
    "outputs/runpod-setup-20261101/remote.py"
    if kind == "evolution"
    else "outputs/ppo-20261101/remote.py"
)
REMOTE = "/workspace/SafeLeWM/outputs/" + study
CONTROL = (
    REMOTE
    if kind == "evolution"
    else "/workspace/SafeLeWM/outputs/safety-dial-library-launch-20261102"
)
LAUNCH = "packaging-launch.json" if kind == "evolution" else "launch.json"
COMPLETE = "packaging-complete.json" if kind == "evolution" else "pipeline-complete.json"
PROCESS = "package_evolution_when_ready.py" if kind == "evolution" else "launch_dial_library.py"
ARCHIVE = BASE / "archive/evidence" / (study + ".tar.gz")


def call(*args, **kwargs):
    return subprocess.run([sys.executable, str(HELPER), *args], check=True, **kwargs)


def inspect(code):
    r = call("exec", "python -c " + shlex.quote(code), capture_output=True, text=True)
    return json.loads(r.stdout.strip().splitlines()[-1])


until = time.monotonic() + (9 * 3600 if kind == "evolution" else 5 * 3600)
while True:
    status = inspect(
        "from pathlib import Path;import json;p=Path("
        + repr(CONTROL)
        + ");launch=json.loads((p/"
        + repr(LAUNCH)
        + ").read_text());f=Path('/proc')/str(launch['pid'])/'cmdline';"
        + "live=f.exists() and "
        + repr(PROCESS.encode())
        + " in f.read_bytes();"
        + "print(json.dumps(dict(complete=(p/"
        + repr(COMPLETE)
        + ").exists(),live=live)))"
    )
    if status["complete"]:
        break
    if not status["live"]:
        raise RuntimeError("Pipeline is terminal without completion; no restart")
    if time.monotonic() > until:
        raise TimeoutError("Observation deadline; rental unchanged")
    time.sleep(60)
for remote, local in [
    (REMOTE + "-manifest.json", OUT / "archive-manifest.json"),
    (REMOTE + "/run-v1/audit.json", OUT / "audit.json"),
    (REMOTE + "/run-v1/analysis.json", OUT / "analysis.json"),
    (REMOTE + "/run-v1/summary.json", OUT / "summary.json"),
    (REMOTE + "/run-v1/freeze.json", OUT / "freeze.json"),
    (CONTROL + "/" + COMPLETE, OUT / "collection-ready.json"),
]:
    call("get", remote, str(local))
manifest = json.loads((OUT / "archive-manifest.json").read_text())
assert json.loads((OUT / "audit.json").read_text())["passed"]
archive_remote = REMOTE + "-evidence.tar.gz"
size = inspect(
    "from pathlib import Path;import json;print(json.dumps(Path("
    + repr(archive_remote)
    + ").stat().st_size))"
)
parts = OUT / "transfer-parts"
parts.mkdir(exist_ok=True)
block = 4 * 1024 * 1024
nblocks = (size + block - 1) // block


def transfer(i):
    first, last = nblocks * i // 8, nblocks * (i + 1) // 8
    path = parts / ("part-" + str(i))
    expected = min(last * block, size) - min(first * block, size)
    if path.exists() and path.stat().st_size == expected:
        return path
    with path.open("wb") as stream:
        call(
            "exec",
            "dd if="
            + shlex.quote(archive_remote)
            + f" bs={block} skip={first} count={last - first} status=none",
            stdout=stream,
        )
    assert path.stat().st_size == expected
    return path


with ThreadPoolExecutor(max_workers=8) as pool:
    ordered = list(pool.map(transfer, range(8)))
partial = ARCHIVE.with_suffix(".partial")
with partial.open("wb") as dst:
    for path in ordered:
        with path.open("rb") as src:
            for chunk in iter(lambda: src.read(8 * 1024 * 1024), b""):
                dst.write(chunk)
with partial.open("rb") as stream:
    sha = hashlib.file_digest(stream, "sha256").hexdigest()
archive_key = "evidence/" + study + ".tar.gz"
assert sha == manifest["archives"][archive_key]
expected = {v["member"]: v for v in manifest["original_files"].values()}
seen = set()
with tarfile.open(partial, "r:gz") as tar:
    for member in tar:
        assert member.isfile() and member.name in expected and member.name not in seen
        entry = expected[member.name]
        assert member.size == entry["bytes"]
        with tar.extractfile(member) as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == entry["sha256"]
        seen.add(member.name)
assert seen == set(expected)
partial.replace(ARCHIVE)
(BASE / "archive" / ("manifest-" + study + ".json")).write_bytes(
    (OUT / "archive-manifest.json").read_bytes()
)
receipt = dict(
    verified_at=time.time(),
    archive=str(ARCHIVE.relative_to(BASE)),
    bytes=size,
    sha256=sha,
    files=len(seen),
    verification="Every archive member SHA256 checked",
)
(OUT / "local-preservation.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt), flush=True)
