"""Script for registering a template on Vast"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DIRECTORY = Path(__file__).resolve().parent
TEMPLATE = DIRECTORY / "vast-template.json"
RECEIPT = DIRECTORY / "vast-template.receipt.json"
ENDPOINT = "https://console.vast.ai/api/v0/template/"


def api_key():
    value = os.environ.get("VAST_API_KEY", "").strip()
    if value:
        return value
    config = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    for path in (config / "vastai/vast_api_key", Path.home() / ".vast_api_key"):
        if path.is_file() and (value := path.read_text().strip()):
            return value
    raise SystemExit("Set VAST_API_KEY in the environment or configure the Vast CLI key file.")


def request(key, *, payload=None, template_hash=None):
    url = ENDPOINT
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
    else:
        url += "?" + urlencode({
            "select_filters": json.dumps({"hash_id": {"eq": template_hash}}),
            "select_cols": json.dumps(["*"]),
        })
    req = Request(url, data=data, headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    })
    try:
        with urlopen(req, timeout=30) as response:
            return json.load(response)
    except HTTPError as error:
        raise SystemExit(f"Vast returned HTTP {error.code}; credentials and response omitted.")
    except (URLError, TimeoutError):
        raise SystemExit("Vast request failed. Check saved registration status before retrying.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--register", action="store_true", help="Save the template in Vast")
    mode.add_argument("--verify", action="store_true", help="Only read back the saved template")
    args = parser.parse_args()
    payload = json.loads(TEMPLATE.read_text())
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if payload.get("private") is not True:
        raise SystemExit("This project template must remain private.")
    if payload.get("runtype") != "ssh" or not payload.get("ssh_direct"):
        raise SystemExit("Expected the reviewed direct-SSH template configuration.")
    if not args.register and not args.verify:
        print(json.dumps({
            "mode": "local preview",
            "name": payload["name"],
            "image": payload["image"] + ":" + payload["tag"],
            "disk_gb": payload["recommended_disk_space"],
            "template_sha256": fingerprint,
        }, indent=2))
        return

    key = api_key()
    receipt = json.loads(RECEIPT.read_text()) if RECEIPT.exists() else {}
    if receipt.get("template_sha256") != fingerprint:
        if args.verify:
            raise SystemExit("No registration receipt matches this configuration; use --register.")
        response = request(key, payload=payload)
        template = response.get("template", {})
        if not response.get("success") or not template.get("hash_id"):
            raise SystemExit("Vast did not confirm template creation; response omitted.")
        receipt = {
            "registered_at_utc": datetime.now(timezone.utc).isoformat(),
            "template_sha256": fingerprint,
            "template": {field: template.get(field) for field in ("id", "hash_id", "name")},
            "launch_settings_verified": False,
            "instance_created": False,
        }
        RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")

    template_hash = receipt["template"]["hash_id"]
    response = request(key, template_hash=template_hash)
    rows = [row for row in response.get("templates", []) if row.get("hash_id") == template_hash]
    if len(rows) != 1:
        raise SystemExit("Template registration recorded, but read-back did not find one match.")
    stored = rows[0]
    if isinstance(stored.get("extra_filters"), str):
        stored["extra_filters"] = json.loads(stored["extra_filters"])
    # Vast returns the README's MD5 content identifier rather than its text.
    reference_only = "readme" not in stored
    readme_matches = stored.get("readme") == payload.get("readme")
    if reference_only:
        expected_hash = hashlib.md5(payload["readme"].encode(), usedforsecurity=False).hexdigest()
        readme_matches = str(stored.get("readme_hash", "")).lstrip("0") == expected_hash.lstrip("0")
    mismatches = [
        field for field, expected in payload.items()
        if field != "readme" and stored.get(field) != expected
    ]
    if not readme_matches:
        mismatches.append("readme")
    receipt.pop("verified", None)
    receipt.pop("unverified_content_fields", None)
    receipt["launch_settings_verified"] = not mismatches
    receipt["readme_hash"] = stored.get("readme_hash")
    receipt["readme_verification"] = "MD5 content identifier" if reference_only else "text"
    receipt["readback_mismatched_fields"] = mismatches
    receipt["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
    receipt["url"] = f"https://cloud.vast.ai/?template_id={template_hash}"
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")
    if mismatches:
        raise SystemExit("Template saved; inspect normalized fields: " + ", ".join(mismatches))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
