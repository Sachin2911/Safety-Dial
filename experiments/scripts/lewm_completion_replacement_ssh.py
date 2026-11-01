"""Private SSH/SCP to the exact provider-verified replacement, with strict keys."""

import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/runpod-lewm-completion-20261031/replacement"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=("exec", "push", "pull"))
    p.add_argument("arguments", nargs="+")
    args = p.parse_args()
    connection = json.loads((OUT / "connection.json").read_text())
    session = json.loads((OUT / "session.json").read_text())
    if connection["pod_id"] != session["id"] or connection["provider_verified"] is not True:
        raise RuntimeError("Exact replacement identity is required")
    common = [
        "-i",
        str(Path.home() / ".ssh/runpod_ed25519"),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "UserKnownHostsFile=" + str(OUT / "known_hosts"),
        "-p" if args.operation == "exec" else "-P",
        str(connection["port"]),
    ]
    target = "root@" + connection["host"]
    if args.operation == "exec":
        command = ["ssh", *common, target, args.arguments[0]]
    else:
        if len(args.arguments) != 2:
            raise RuntimeError("Single-file source and destination required")
        source, destination = args.arguments
        if args.operation == "push":
            if any(
                part.startswith(".env") or part in (".ssh", ".runpod")
                for part in Path(source).parts
            ):
                raise RuntimeError("Credentials cannot be transferred")
            command = ["scp", *common, source, target + ":" + destination]
        else:
            command = ["scp", *common, target + ":" + source, destination]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
