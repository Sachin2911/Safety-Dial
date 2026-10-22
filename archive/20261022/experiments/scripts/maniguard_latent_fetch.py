"""Resume pinned public family assets with verified files and resumable Hub transport."""

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/maniguard"))

from phase2_download import digest, plan_download, verify_file


def acquire(family, kind, recovery):
    plan = plan_download(family, kind)
    destination = Path(plan["destination"])
    if destination.exists():
        previous = json.loads((destination / "download-provenance.json").read_text())
        if previous["plan"] != plan or not all(
                verify_file(destination / entry["relative"], entry) for entry in plan["files"]):
            raise ValueError("Existing asset destination differs from its verified pin")
        return previous
    staging = destination.with_name("." + destination.name + ".phase2-incomplete")
    staging.mkdir(parents=True, exist_ok=True)
    marker = staging / "download-plan.json"
    if marker.exists() and json.loads(marker.read_text()) != plan:
        raise ValueError("Unrecognized download staging plan")
    marker.write_text(json.dumps(plan, indent=2) + "\n")
    recovery.mkdir(parents=True, exist_ok=True)
    for part in staging.rglob("*.part"):
        relative = part.relative_to(staging)
        target = recovery / family / kind / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError("Refusing to replace preserved transfer failure")
        part.rename(target)
    missing = sum(entry["bytes"] for entry in plan["files"]
                  if not (staging / entry["relative"]).exists())
    if shutil.disk_usage(staging).free < missing + 5 * 1024**3:
        raise OSError("Insufficient space for missing assets plus 5 GiB reserve")

    def fetch(entry):
        target = staging / entry["relative"]
        if target.exists():
            if not verify_file(target, entry):
                raise ValueError("Previously downloaded file failed integrity: " + entry["relative"])
        else:
            source = Path(hf_hub_download(plan["repo"], entry["remote"],
                                          repo_type=plan["repo_type"], revision=plan["revision"])).resolve()
            if not verify_file(source, entry):
                raise ValueError("Hub download failed upstream digest: " + entry["remote"])
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, target)
            except OSError:
                shutil.copyfile(source, target)
        print("verified", family, kind, entry["relative"], flush=True)
        return entry["relative"], {"bytes": target.stat().st_size, "sha256": digest(target)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        files = dict(pool.map(fetch, plan["files"]))
    record = {"plan": plan, "files": files, "complete": True,
              "behavioral_evaluation_performed": False, "transport": "huggingface_hub",
              "at": time.time()}
    (staging / "download-provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    staging.rename(destination)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["jar", "clutter"], required=True)
    parser.add_argument("--kind", choices=["benchmark", "policy", "both"], default="both")
    parser.add_argument("--recovery", type=Path, required=True)
    args = parser.parse_args()
    for kind in ["benchmark", "policy"] if args.kind == "both" else [args.kind]:
        result = acquire(args.family, kind, args.recovery)
        print(json.dumps({"family": args.family, "kind": kind, "complete": True,
                          "files": len(result["files"])}), flush=True)


if __name__ == "__main__":
    main()
