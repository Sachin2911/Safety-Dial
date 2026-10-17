"""Plan or download pinned public family assets into new, verified destinations.

Only public HTTPS metadata and files are requested, without authentication. Existing
destinations are verified or refused, never overwritten. The benchmark selection
contains task scene/diagnostic JSON, not videos or review grids. No policy runs.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.maniguard_phase2_families import family_spec, load_sources


def digest(path, algorithm="sha256", *, git_blob=False):
    hasher = hashlib.new(algorithm)
    if git_blob:
        hasher.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def public_metadata(repo, revision, kind):
    category = "datasets" if kind == "dataset" else "models"
    url = f"https://huggingface.co/api/{category}/{repo}/revision/{revision}?blobs=true"
    with urllib.request.urlopen(url, timeout=60) as response:
        data = json.load(response)
    if data["sha"] != revision or data.get("private") or data.get("gated"):
        raise ValueError("Asset is unpinned, private or gated; public downloader refuses it")
    return data


def plan_download(family, kind, data_root=None):
    spec, sources = family_spec(family), load_sources()
    data_root = Path(data_root or ROOT / "data/maniguard")
    if kind == "benchmark":
        source, repo_type = sources["benchmark"], "dataset"
        prefix = spec["benchmark_dir"] + "/"
        destination = data_root / "benchmark" / spec["benchmark_dir"]
    elif kind == "policy":
        source, repo_type = sources["policies"][family], "model"
        prefix = source["step"] + "/"
        destination = data_root / "policies" / family
    else:
        raise ValueError(f"Unknown download kind: {kind}")
    metadata = public_metadata(source["repo"], source["revision"], repo_type)
    selected = []
    for entry in metadata["siblings"]:
        filename = entry["rfilename"]
        if kind == "benchmark":
            relative = filename.removeprefix(prefix)
            include = (filename.startswith(prefix + "task_")
                       and filename.endswith((".json", ".jsonl")))
        else:
            relative = filename
            include = filename.startswith(prefix) or filename in ("README.md", "LICENSE")
        if not include:
            continue
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Unsafe remote asset path")
        selected.append({"remote": filename, "relative": relative, "bytes": entry["size"],
                         "git_blob_sha1": entry["blobId"],
                         "sha256": entry.get("lfs", {}).get("sha256")})
    if not selected or (kind == "policy" and not any(
        item["relative"].startswith(prefix + "params/") for item in selected
    )):
        raise ValueError("Pinned repository lacks the requested runtime assets")
    return {"schema_version": 1, "family": family, "kind": kind, "repo": source["repo"],
            "revision": source["revision"], "repo_type": repo_type,
            "destination": str(destination), "files": selected,
            "bytes": sum(item["bytes"] for item in selected), "public_unauthenticated": True}


def verify_file(path, entry):
    if not path.is_file() or path.is_symlink() or path.stat().st_size != entry["bytes"]:
        return False
    if entry["sha256"]:
        return digest(path) == entry["sha256"]
    return digest(path, "sha1", git_blob=True) == entry["git_blob_sha1"]


def download(plan, workers=4, reserve_bytes=2 * 1024**3):
    destination = Path(plan["destination"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        marker = destination / "download-provenance.json"
        if not marker.is_file():
            raise FileExistsError(f"Existing assets have no Phase 2 manifest: {destination}")
        old = json.loads(marker.read_text())
        if old.get("plan") != plan:
            raise FileExistsError(f"Existing asset manifest differs: {destination}")
        if not all(verify_file(destination / entry["relative"], entry) for entry in plan["files"]):
            raise ValueError("Existing downloaded asset failed integrity verification")
        return old
    staging = destination.with_name("." + destination.name + ".phase2-incomplete")
    marker = staging / "download-plan.json"
    if staging.exists():
        if not marker.is_file() or json.loads(marker.read_text()) != plan:
            raise FileExistsError(f"Unrecognized staging assets: {staging}")
    else:
        staging.mkdir()
        marker.write_text(json.dumps(plan, indent=2) + "\n")
    remaining = sum(entry["bytes"] for entry in plan["files"]
                    if not (staging / entry["relative"]).exists())
    if shutil.disk_usage(destination.parent).free < remaining + reserve_bytes:
        raise OSError("Insufficient free disk for pinned files plus 2 GiB reserve")
    base = "https://huggingface.co/"
    if plan["repo_type"] == "dataset":
        base += "datasets/"
    base += plan["repo"] + "/resolve/" + plan["revision"] + "/"

    def fetch(entry):
        target = staging / entry["relative"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if not verify_file(target, entry):
                raise ValueError(f"Existing staging file failed hash: {entry['relative']}")
        else:
            temporary = target.with_name(target.name + ".part")
            if temporary.exists():
                raise FileExistsError(f"Incomplete file retained; inspect before retry: {temporary}")
            request = urllib.request.Request(base + entry["remote"],
                                             headers={"User-Agent": "Safety-Dial-public-assets/1"})
            with (urllib.request.urlopen(request, timeout=180) as response,
                  temporary.open("xb") as output):
                shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
            if not verify_file(temporary, entry):
                raise ValueError(f"Downloaded file failed hash: {entry['relative']}")
            os.rename(temporary, target)
        return entry["relative"], {"bytes": target.stat().st_size, "sha256": digest(target)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        hashes = dict(pool.map(fetch, plan["files"]))
    record = {"plan": plan, "files": hashes, "complete": True,
              "behavioral_evaluation_performed": False}
    (staging / "download-provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    if destination.exists():
        raise FileExistsError(f"Destination appeared during download: {destination}")
    staging.rename(destination)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=("jar", "stack", "lid", "clutter", "cabinet", "dusty"))
    parser.add_argument("kind", choices=("benchmark", "policy", "both"))
    parser.add_argument("--execute", action="store_true", help="Download after printing the plan")
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    kinds = ("benchmark", "policy") if args.kind == "both" else (args.kind,)
    plans = [plan_download(args.family, kind, args.data_root) for kind in kinds]
    print(json.dumps({"plans": [{key: value for key, value in plan.items() if key != "files"}
                                | {"file_count": len(plan["files"])} for plan in plans],
                      "total_bytes": sum(plan["bytes"] for plan in plans)}, indent=2), flush=True)
    if args.execute:
        for plan in plans:
            record = download(plan)
            print(json.dumps({"complete": record["complete"], "destination": plan["destination"],
                              "verified_files": len(record["files"])}), flush=True)


if __name__ == "__main__":
    main()
