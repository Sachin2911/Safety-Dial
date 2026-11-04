"""Validate the built image on rented compute, then publish it to private GHCR."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.error import HTTPError
from urllib.request import Request, urlopen


DEFAULT_IMAGE = "ghcr.io/sachin2911/lewm-ap:0.1.0"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    args = parser.parse_args()
    if not args.image.startswith("ghcr.io/sachin2911/lewm-ap:"):
        raise SystemExit("Expected the reviewed sachin2911/lewm-ap GHCR package.")
    token = os.environ.get("GHCR_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("Set GHCR_TOKEN in the remote environment with write:packages access.")
    request = Request(
        "https://api.github.com/user/packages/container/lewm-ap",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            package = json.load(response)
        if package.get("visibility") != "private":
            raise SystemExit("The existing GHCR package is not private; inspect its visibility first.")
    except HTTPError as error:
        if error.code != 404:
            raise SystemExit(f"GHCR package preflight returned HTTP {error.code}.") from None
        # GitHub creates new packages with private visibility.
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", args.image], text=True
    ).strip()
    subprocess.run([
        "docker", "run", "--rm", "--network", "none", "--gpus", "all",
        "--entrypoint", "/venv/main/bin/python", image_id,
        "/opt/lewm-runtime/verify_runtime.py", "--gpu", "--render",
    ], check=True)
    current = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", args.image], text=True
    ).strip()
    if current != image_id:
        raise SystemExit("Image tag changed during validation; refusing to publish.")

    # Docker obtains the token over its helper pipe. No credential is written to
    # Docker config, passed as an argument, copied into the image or printed here.
    with tempfile.TemporaryDirectory(prefix="lewm-ghcr-") as directory:
        root = Path(directory)
        helper = root / "docker-credential-lewm-env"
        helper.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "if sys.argv[1:] != ['get']: raise SystemExit(1)\n"
            "server = sys.stdin.read().strip()\n"
            "if server not in ('ghcr.io', 'https://ghcr.io'): raise SystemExit(1)\n"
            "json.dump({'Username': 'Sachin2911', 'Secret': os.environ['GHCR_TOKEN']}, sys.stdout)\n"
        )
        helper.chmod(0o700)
        (root / "config.json").write_text(json.dumps({"credHelpers": {"ghcr.io": "lewm-env"}}))
        env = dict(os.environ, GHCR_TOKEN=token, PATH=f"{root}:{os.environ.get('PATH', '')}")
        subprocess.run(["docker", "--config", str(root), "push", args.image], env=env, check=True)
        subprocess.run(["docker", "--config", str(root), "pull", args.image], env=env, check=True)
        published = json.loads(subprocess.check_output(
            ["docker", "image", "inspect", args.image], text=True
        ))[0]
        if published["Id"] != image_id:
            raise SystemExit("Pulled image differs from the image that passed validation.")
        digest = next(item for item in published["RepoDigests"]
                      if item.startswith("ghcr.io/sachin2911/lewm-ap@sha256:"))
        subprocess.run(["docker", "--config", str(root), "pull", digest], env=env, check=True)
    with urlopen(request, timeout=30) as response:
        visibility = json.load(response)["visibility"]
    if visibility != "private":
        raise SystemExit("Published package visibility differs from private.")
    print(json.dumps({"image": args.image, "digest": digest, "visibility": visibility,
                      "image_id": image_id, "registry_pull": "verified"}, indent=2))


if __name__ == "__main__":
    main()
