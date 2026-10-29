"""Task-16 transport and queue launcher, with explicit rental identity checks."""

import argparse
import json
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_infra import ssh_endpoint
from helpers.lewm16_infra import RENTAL, current, remote, ssh_options, snapshot, utc
from helpers.lewm16_protocol import OUT, digest, write_json


def bundle(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    names = set(manifest["source_sha256"])
    names |= {
        str(manifest_path.relative_to(ROOT)),
        str(RENTAL.relative_to(ROOT)),
        "progress/receipts/16-lewm-authorization.json",
    }
    for name, expected in manifest["source_sha256"].items():
        assert digest(ROOT / name) == expected, name
    destination = OUT / "source" / f"{manifest_path.stem}.tar.gz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz") as tar:
        for name in sorted(names):
            tar.add(ROOT / name, arcname=name)
    return destination


def transport():
    rental, row = current()
    if row is None:
        raise RuntimeError("Task-16 rental absent")
    rental["provider_seen_utc"], rental["provider"] = utc(), row
    write_json(RENTAL, rental)
    host, port = ssh_endpoint(row)
    return host, port


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["status", "ssh", "upload", "sync", "launch"])
    parser.add_argument("--command")
    parser.add_argument("--manifest")
    parser.add_argument("--lanes", type=int, default=2)
    parser.add_argument("--mapping", default="0,1:2,3")
    args = parser.parse_args()
    if args.action == "status":
        rental, row = current()
        print(json.dumps({"provider": row, "finance": snapshot(rental)}))
    elif args.action == "ssh":
        result = remote(args.command, check=False, timeout=90)
        print(result.stdout)
        if result.returncode:
            print(result.stderr)
        return result.returncode
    elif args.action == "upload":
        manifest = (ROOT / args.manifest).resolve()
        source = bundle(manifest)
        host, port = transport()
        dest = "/tmp/lewm16-source.tar.gz"
        subprocess.run(
            [
                "scp",
                *["-P" if x == "-p" else x for x in ssh_options(port)],
                str(source),
                f"root@{host}:{dest}",
            ],
            check=True,
        )
        command = (
            "mkdir -p /workspace/Safety-Dial && "
            "tar -xzf /tmp/lewm16-source.tar.gz -C /workspace/Safety-Dial && "
            "mkdir -p /workspace/Safety-Dial/outputs/lewm16/{control,attempts,logs,source} && "
            f"cp /tmp/lewm16-source.tar.gz /workspace/Safety-Dial/outputs/lewm16/source/{source.name}"
        )
        result = remote(command)
        print(json.dumps({"bundle": str(source), "sha256": digest(source), "uploaded": True}))
    elif args.action == "sync":
        host, port = transport()
        destination = OUT / "remote"
        destination.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "rsync",
                "-az",
                "--partial",
                "-e",
                shlex.join(["ssh", *ssh_options(port)]),
                f"root@{host}:/workspace/Safety-Dial/outputs/lewm16/",
                str(destination) + "/",
            ],
            check=True,
        )
        print(json.dumps({"synced": str(destination)}))
    elif args.action == "launch":
        session = Path(args.manifest).stem.replace("16-", "lewm16-")
        gpus = sorted({int(x) for p in args.mapping.split(":") for x in p.split(",")})
        commands = []
        for gpu in gpus:
            for lane in range(args.lanes):
                worker = shlex.join(
                    [
                        "/venv/main/bin/python",
                        "-u",
                        "experiments/scripts/worker_lewm16.py",
                        "--manifest",
                        args.manifest,
                        "--gpu",
                        str(gpu),
                        "--lane",
                        str(lane),
                        "--lanes",
                        str(args.lanes),
                        "--mapping",
                        args.mapping,
                    ]
                )
                log = f"outputs/lewm16/logs/{session}-g{gpu}-l{lane}.log"
                commands.append(f"{worker} > {shlex.quote(log)} 2>&1 &")
        content = "cd /workspace/Safety-Dial\n" + "\n".join(commands) + "\nwait\n"
        command = f"tmux new-session -d -s {shlex.quote(session)} {shlex.quote('bash -lc ' + shlex.quote(content))}"
        result = remote(command)
        print(json.dumps({"session": session, "workers": len(commands)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
