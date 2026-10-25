"""Deploy the separately frozen screen runtime and start its one durable queue."""

import hashlib
import json
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.screen08_infra import instances, ssh_endpoint, ssh_options


def main():
    plan_path = ROOT / "progress/receipts/08-screen-plan.json"
    rental_path = ROOT / "progress/rental-screen08-20261025.json"
    plan, rental = json.loads(plan_path.read_text()), json.loads(rental_path.read_text())
    paths = [ROOT / name for name in plan["source_sha256"]]
    for path in paths:
        if hashlib.sha256(path.read_bytes()).hexdigest() != plan["source_sha256"][str(path.relative_to(ROOT))]:
            raise ValueError("Source differs from the full task-8 freeze")
    bundle = ROOT / "outputs/screen08/runtime-source.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        for path in [*paths, plan_path, rental_path]:
            archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    provider = next(r for r in instances() if r["id"] == rental["instance_id"])
    host, port = ssh_endpoint(provider)
    transport = shlex.join(["ssh", *ssh_options(port)])
    subprocess.run(["rsync", "-az", "--timeout=90", "-e", transport, str(bundle),
                    f"root@{host}:/tmp/screen08-runtime.tar.gz"],
                   capture_output=True, timeout=180, check=True)
    check = subprocess.run(["ssh", *ssh_options(port), f"root@{host}",
                            "sha256sum /tmp/screen08-runtime.tar.gz"],
                           capture_output=True, text=True, timeout=45, check=True)
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    if check.stdout.split()[0] != digest:
        raise ValueError("Runtime transfer hash differs")
    remote = """set -e
cd /workspace/Safety-Dial
test ! -e outputs/screen08/pipeline-started.txt
if tmux has-session -t run08 2>/dev/null; then exit 1; fi
tar -xzf /tmp/screen08-runtime.tar.gz -C /workspace/Safety-Dial
cp /tmp/screen08-runtime.tar.gz outputs/screen08/source/runtime-source.tar.gz
tmux new-session -d -s run08 'cd /workspace/Safety-Dial && bash experiments/scripts/launch_screen08.sh > outputs/screen08/logs/pipeline.log 2>&1'
tmux list-sessions
"""
    result = subprocess.run(["ssh", *ssh_options(port), f"root@{host}", remote],
                            capture_output=True, text=True, timeout=45, check=True)
    print(result.stdout, end="")
    print(json.dumps({"bundle_sha256": digest, "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
