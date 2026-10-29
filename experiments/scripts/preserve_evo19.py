"""Verify that a task 19 host's outputs are completely mirrored locally, then mark it.

For one host: list every file under the remote outputs/evo19 with its size, run a final
rsync, and require every remote file to exist locally with the same size and every
result to have a local copy of its episode directory. Writes
outputs/evo19/remote/<host>/manifest.json (local sha256 of every file) and
preserved.json, which manage_evo19.py destroy requires.
"""

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments/scripts"))
import manage_evo19 as infra  # noqa: E402
from helpers.dsrl06_infra import atomic_json, ssh_endpoint, utc  # noqa: E402

REMOTE = "/workspace/Safety-Dial/outputs/evo19"


def remote_listing(row):
    address, port = ssh_endpoint(row)
    script = ("import os,json\n"
              f"R='{REMOTE}'\n"
              "out={}\n"
              "for d,_,fs in os.walk(R):\n"
              "  for f in fs:\n"
              "    p=os.path.join(d,f); out[os.path.relpath(p,R)]=os.path.getsize(p)\n"
              "print(json.dumps(out))\n")
    result = subprocess.run(["ssh", *infra.ssh_options(port), f"root@{address}",
                             "python3 -c " + shlex.quote(script)],
                            capture_output=True, text=True, timeout=180, check=True)
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("host")
    args = parser.parse_args()
    _, row = infra.host_row(args.host)
    if infra.sync_host(args.host, row, timeout=1800):
        raise SystemExit("Final rsync failed")
    listing = remote_listing(row)
    local = ROOT / "outputs/evo19/remote" / args.host
    volatile = ("logs/", "worker-events.jsonl", "network-cost.json", "server/")
    missing, size_diff = [], []
    for rel, size in listing.items():
        path = local / rel
        if not path.exists():
            missing.append(rel)
        elif path.stat().st_size != size and not rel.startswith(volatile):
            size_diff.append(rel)
    results = [p for p in listing if p.startswith("results/") and p.endswith(".json")]
    for rel in results:
        record = json.loads((local / rel).read_text())
        for attempt in record.get("attempts", []):
            leaf = attempt.get("leaf")
            if leaf and not (local / "episodes" / leaf / "job.json").exists():
                missing.append(f"episodes/{leaf}")
    if missing or size_diff:
        raise SystemExit(json.dumps({"missing": missing[:20], "size_diff": size_diff[:20]}))
    manifest = {str(p.relative_to(local)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(local.rglob("*")) if p.is_file()
                and p.name not in ("manifest.json", "preserved.json")}
    atomic_json(local / "manifest.json", {"host": args.host, "utc": utc(), "files": manifest})
    marker = {"host": args.host, "instance_id": row["id"], "utc": utc(),
              "remote_files": len(listing), "local_files": len(manifest),
              "results": len(results),
              "manifest_sha256": hashlib.sha256((local / "manifest.json").read_bytes()).hexdigest()}
    atomic_json(local / "preserved.json", marker)
    print(json.dumps(marker))


if __name__ == "__main__":
    main()
