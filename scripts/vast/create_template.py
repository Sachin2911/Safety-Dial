"""Create the private Vast platform template without renting an instance."""

import ast
import base64
import gzip
import json
from pathlib import Path
import shlex
import subprocess


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    config_path = root / "configs/vast/evo-safety-filter.json"
    config = json.loads(config_path.read_text())
    state_dir = "/workspace/.evo-safety-filter"
    chunks = ["#!/usr/bin/env bash\nset -euo pipefail\n", f"mkdir -p {state_dir}\n"]
    for source_key, target in (
        ("provision_script", "provision.sh"),
        ("requirements_file", "requirements-linux-py312.txt"),
        ("overrides_file", "uv-overrides.txt"),
    ):
        data = (root / config[source_key]).read_bytes()
        encoded = base64.b64encode(gzip.compress(data, mtime=0)).decode()
        chunks.append(
            f"base64 --decode <<'EVO_DATA' | gzip --decompress > {state_dir}/{target}\n"
            f"{encoded}\nEVO_DATA\n"
        )
    chunks.append(
        "cat > /etc/vast_boot.d/80-evo-safety-filter.sh <<'EVO_HOOK'\n"
        f"bash {state_dir}/provision.sh || "
        "printf 'evo safety filter provisioning failed; check provision.log\\n' >&2\n"
        "EVO_HOOK\n"
        "exec /opt/instance-tools/bin/entrypoint.sh\n"
    )
    onstart = "".join(chunks)
    subprocess.run(["bash", "-n"], input=onstart, text=True, check=True)

    query = "private=true name=" + config["name"].replace(" ", "_")
    existing = subprocess.run(
        ["vastai", "search", "templates", query], capture_output=True, text=True, check=True
    )
    rows = json.loads(existing.stdout)
    if isinstance(rows, dict):
        rows = rows.get("templates", [])
    if rows:
        raise SystemExit("A private template with this name already exists. Inspect it first.")

    env = " ".join(f"-e {shlex.quote(k + '=' + v)}" for k, v in config["env"].items())
    command = [
        "vastai",
        "create",
        "template",
        "--name",
        config["name"],
        "--image",
        config["image"],
        "--image_tag",
        config["image_tag"],
        "--desc",
        config["description"],
        "--env=" + env,
        "--ssh",
        "--jupyter",
        "--direct",
        "--jupyter-lab",
        "--jupyter-dir",
        config["jupyter_dir"],
        "--disk_space",
        str(config["disk_space_gb"]),
        "--search_params",
        config["search_params"],
        "--onstart-cmd",
        onstart,
        "--readme",
        (root / "configs/vast/README.md").read_text(),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    prefix = "New Template: "
    response = next((line for line in result.stdout.splitlines() if line.startswith(prefix)), None)
    if response is None:
        raise SystemExit("Vast did not confirm template creation: " + result.stdout)
    saved = ast.literal_eval(response.removeprefix(prefix))
    record = {k: saved.get(k) for k in ("id", "hash_id", "name")}
    config_path.with_name("created-template.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    print(f"Embedded startup command: {len(onstart.encode())} bytes. No instance rented.")


if __name__ == "__main__":
    main()
