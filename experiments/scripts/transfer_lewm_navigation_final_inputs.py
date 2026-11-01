"""Exact Nav final-input/config transfer after explicit root QA/release.

Standard library only. Never decodes arrays/images, starts a worker, changes an
original, or copies foreign results, models, raw datasets or host environments.
"""

import argparse
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile

ROOT = Path(__file__).resolve().parents[2]
NAV = "vukovc3t4o1dui"
DESTINATIONS = {"wq4ar2qx2ff6r4", "nj2fqkp2prgk8x"}
CONFIGS = "configs/lewm_campaign_20261031"
PREP = "results/navigation-final-cases-v1"
ACCESS = PREP + "-joint-access.json"
WORKER_ALIAS = "environment/final-preparation-origin-v1/source-worker.json"
RECEIPT = "environment/navigation-final-transfer-v1.json"
CONTROL = "FINAL-INPUT-TRANSFER.json"
SCHEMA = "lewm-navigation-final-input-transfer-v1"
PROOFS = {
    "results/evidence-audits/navigation-secondary-r2-unstarted-v2.json",
    "results/evidence-audits/navigation-terminal-mean-prefinal-absence-v1.json",
    "results/navigation-journal-parity-v1/qualification.json",
    "results/navigation-memory-cache-v3/memory-metadata.json",
}
SUFFIXES = {".json", ".jsonl", ".npz", ".png", ".jpg", ".jpeg", ".svg", ".webp",
            ".html", ".txt", ".csv", ".md"}
MAX_FILE_BYTES = 128 * 1024 * 1024


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text())


def safe_name(value):
    p = PurePosixPath(value)
    require(value and "\\" not in value and not p.is_absolute() and ".." not in p.parts
            and str(p) == value, "Unsafe noncanonical transfer path")
    return value


def allowed(value):
    value = safe_name(value)
    return (
        value.startswith(CONFIGS + "/") and value.endswith(".json")
        or value.startswith(PREP + "/") and Path(value).suffix.lower() in SUFFIXES
        or value in PROOFS | {ACCESS, WORKER_ALIAS}
    )


def local_name(value):
    p = Path(value)
    if p.is_absolute():
        require(p.resolve().is_relative_to(ROOT), "Frozen reference escaped canonical root")
        value = str(p.relative_to(ROOT))
    return safe_name(value)


def regular(path):
    require(not path.is_symlink() and stat.S_ISREG(path.stat().st_mode)
            and path.stat().st_size <= MAX_FILE_BYTES, "Bounded regular source file required")


def record(path, archive_name):
    regular(path)
    require(allowed(archive_name), "File is outside the explicit transfer whitelist")
    return {"sha256": sha(path), "bytes": path.stat().st_size}


def ref(value):
    require(allowed(value), "Origin reference is outside transfer whitelist")
    return {"path": value, "sha256": sha(ROOT / value)}


def retired(worker):
    p = Path("/proc") / str(worker["pid"]) / "stat"
    if not p.exists():
        return True
    raw = p.read_text()
    fields = raw[raw.rfind(")") + 2:].split()
    return fields[0] == "Z" or fields[19] != str(worker["start_ticks"])


def export(args):
    require(args.root_release_confirmed, "Root QA and all14-profile release must be explicit")
    require(ROOT == Path("/workspace/Safety-Dial") and os.environ.get("RUNPOD_POD_ID") == NAV,
            "Only actual canonical Nav may export original final inputs")
    out = args.out_dir.resolve()
    require(not out.is_relative_to(ROOT) and not out.exists(), "Fresh outside-project archive dir")
    main, mean = load(args.main_manifest), load(args.mean_manifest)
    require(main["schema"] == "lewm-navigation-joint-final-comparison-v2"
            and len(main["schedule"]) == 11
            and mean["schema"] == "lewm-terminal-mean-final-sidecar-v1"
            and len(mean["profiles"]) == 3, "All11 main and3 mean profiles must be bound")
    main_sha = sha(args.main_manifest)
    require(mean["main_manifest"]["sha256"] == main_sha, "Mean sidecar binds a different main freeze")
    panel_path, review_path = ROOT / PREP / "panel.json", ROOT / PREP / "visual-review.json"
    panel, review = load(panel_path), load(review_path)
    require(len(panel["cases"]) == 16 and review["approved"] is True
            and review["images_clean"] is True, "Root must approve the complete16-case gallery")
    status = load(ROOT / PREP / "status.json")
    require(not status["error"] and status["eligible_cases"] == 16
            and status["controller_episodes"] == status["new_model_queries"] == 0,
            "Final preparation must finish without technical failure or controller work")
    protocols = [item["protocol"] for item in main["schedule"]]
    protocols += [item["protocol"]["path"] for item in mean["profiles"]]
    for value in protocols:
        p = ROOT / local_name(value)
        v = load(p)
        preparation = v["case_preparation"]
        require(local_name(preparation["panel_path"]) == str(panel_path.relative_to(ROOT))
                and preparation["panel_sha256"] == sha(panel_path)
                and local_name(preparation["review_path"]) == str(review_path.relative_to(ROOT))
                and preparation["review_sha256"] == sha(review_path),
                "A bound final profile uses different prepared inputs/review")
    worker_path = args.source_worker_record.resolve()
    require(worker_path.is_relative_to(ROOT / "environment/workers"),
            "Read exactly one explicit original prep worker record")
    worker = load(worker_path)
    access = load(ROOT / ACCESS)
    joint_name = safe_name(worker["protocol"])
    joint_path = ROOT / joint_name
    joint = load(joint_path)
    primary_ref = joint["primary_preparation_protocol"]
    primary_name = safe_name(primary_ref["path"])
    primary = load(ROOT / primary_name)
    freeze = load(ROOT / PREP / "source-freeze.json")
    require(worker["pod_id"] == NAV and worker["out"] == PREP
            and worker["entrypoint"] == "experiments/scripts/prepare_lewm_navigation_joint_final_cases_v2.py"
            and retired(worker), "Exact original Nav preparation worker must be retired")
    require(worker.get("expected_protocol_sha256", worker.get("protocol_sha256"))
            == access["protocol_sha256"] == sha(joint_path)
            and access["all_six_repeat_selections_fixed"] is True
            and access["primary_preparation_protocol"] == primary_ref
            and sha(ROOT / primary_name) == primary_ref["sha256"]
            == freeze["protocol_sha256"]
            and primary["execution_enabled"] is True and primary["allowed_pod_ids"] == [NAV]
            and freeze["sources"] == primary["source_sha256"],
            "Original preparation source/worker/access/primary protocol chain differs")
    source_paths = {}
    for base in (ROOT / CONFIGS, ROOT / PREP):
        for p in sorted(base.rglob("*")):
            require(not p.is_symlink(), "No symlink may occur inside frozen transfer trees")
            if p.is_file() and (base.name != Path(CONFIGS).name or p.suffix == ".json"):
                source_paths[str(p.relative_to(ROOT))] = p
    source_paths[ACCESS] = ROOT / ACCESS
    source_paths[WORKER_ALIAS] = worker_path
    for value in args.proof:
        require(value in PROOFS, "Only explicitly listed global metadata/proofs are allowed")
        source_paths[value] = ROOT / value
    npz = [value for value in source_paths if value.startswith(PREP + "/") and value.endswith(".npz")]
    require(len(npz) == 16 and all(value.startswith(PREP + "/inputs/") for value in npz),
            "Exactly16 prepared input NPZ files required; no raw dataset/model arrays")
    for case in panel["cases"]:
        value = local_name(case["input_path"])
        require(value in npz and sha(source_paths[value]) == case["input_sha256"],
                "Prepared input does not match its frozen source/frame identity")
    files = {value: record(p, value) for value, p in source_paths.items()}
    origin = {
        "schema": "lewm-final-preparation-nav-origin-v1",
        "source_freeze": ref(PREP + "/source-freeze.json"),
        "primary_preparation_protocol": ref(primary_name),
        "joint_preparation_protocol": ref(joint_name), "joint_access": ref(ACCESS),
        "source_worker": {"path": WORKER_ALIAS, "sha256": sha(worker_path)},
    }
    manifest = {
        "schema": SCHEMA, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_pod_id": NAV, "files": files,
        "sha256": {value: item["sha256"] for value, item in files.items()},
        "final_preparation_origin": origin,
        "source_worker_original_path": str(worker_path.relative_to(ROOT)),
        "main_manifest_sha256": main_sha, "mean_manifest_sha256": sha(args.mean_manifest),
        "source_helper_sha256": sha(Path(__file__)),
        "new_simulator_actions": 0, "new_model_queries": 0,
    }
    out.mkdir(parents=True)
    archive = out / "navigation-final-transfer-v1.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=3) as stream:
        for value, p in sorted(source_paths.items()):
            before = p.stat()
            raw = p.read_bytes()
            require(p.stat() == before and hashlib.sha256(raw).hexdigest() == files[value]["sha256"],
                    "Original changed while exporting; originals are never edited")
            info = tarfile.TarInfo(value)
            info.size, info.mode = len(raw), 0o600
            stream.addfile(info, io.BytesIO(raw))
        raw = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
        info = tarfile.TarInfo(CONTROL)
        info.size, info.mode = len(raw), 0o600
        stream.addfile(info, io.BytesIO(raw))
    for value, p in source_paths.items():
        require(sha(p) == files[value]["sha256"], "Original source changed after export")
    require(verify_archive(archive, sha(archive)) == manifest, "Export archive readback differs")
    receipt = {"archive": str(archive), "archive_sha256": sha(archive),
               "manifest": manifest, "verified": True}
    (out / "export-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"archive": str(archive), "archive_sha256": receipt["archive_sha256"],
                      "files": len(files), "new_simulator_actions": 0, "new_model_queries": 0}))


def verify_archive(archive, expected):
    require(sha(archive) == expected, "Archive SHA differs from the exact Nav export")
    observed, manifest = {}, None
    with tarfile.open(archive, "r:*") as stream:
        for member in stream:
            value = safe_name(member.name)
            require(member.isfile() and not member.issym() and not member.islnk()
                    and value not in observed and member.size <= MAX_FILE_BYTES,
                    "Unsafe, duplicate or oversized archive member")
            raw = stream.extractfile(member).read()
            require(len(raw) == member.size, "Truncated archive member")
            observed[value] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
            if value == CONTROL:
                manifest = json.loads(raw)
            else:
                require(allowed(value), "Archive member is outside explicit input/config whitelist")
    require(manifest and manifest["schema"] == SCHEMA and manifest["source_pod_id"] == NAV,
            "Original Nav transfer manifest required")
    require(set(observed) == set(manifest["files"]) | {CONTROL}, "Archive member set differs")
    for value, item in manifest["files"].items():
        require(observed[value] == item and manifest["sha256"][value] == item["sha256"],
                "Archive member hash/size differs")
    return manifest


def install(args):
    pod = os.environ.get("RUNPOD_POD_ID")
    require(args.root_release_confirmed and ROOT == Path("/workspace/Safety-Dial")
            and pod in DESTINATIONS, "Explicit root release on an actual destination pod required")
    manifest = verify_archive(args.archive, args.archive_sha256)
    require(manifest["source_helper_sha256"] == sha(Path(__file__)),
            "Exporter and installer must use identical frozen helper bytes")
    receipt_path = ROOT / RECEIPT
    require(not receipt_path.exists(), "Never overwrite an earlier final-transfer receipt")
    for value, item in manifest["files"].items():
        p = ROOT / value
        require(p.resolve().is_relative_to(ROOT) and not p.is_symlink(), "Destination escapes root")
        if p.exists():
            regular(p)
            require(sha(p) == item["sha256"], "Existing frozen bytes differ; never overwrite: " + value)
    created, identical = [], []
    with tarfile.open(args.archive, "r:*") as stream:
        for member in stream:
            value = safe_name(member.name)
            if value == CONTROL:
                continue
            require(value in manifest["files"] and member.isfile(), "Archive changed during install")
            p = ROOT / value
            expected = manifest["sha256"][value]
            if p.exists():
                require(sha(p) == expected, "Existing source changed during installation")
                identical.append(value)
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            require(p.parent.resolve().is_relative_to(ROOT), "Destination parent escaped root")
            temporary = p.with_name("." + p.name + ".incoming-" + expected[:16])
            with temporary.open("xb") as output:
                source = stream.extractfile(member)
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())
            require(sha(temporary) == expected, "Staged bytes differ")
            try:
                os.link(temporary, p)
            except FileExistsError:
                require(sha(p) == expected, "Concurrent different frozen bytes; no overwrite attempted")
            temporary.unlink()
            require(sha(p) == expected, "Installed bytes differ")
            created.append(value)
    require(sha(args.archive) == args.archive_sha256, "Archive changed during install")
    for value, expected in manifest["sha256"].items():
        require(sha(ROOT / value) == expected, "Final whole-transfer readback failed")
    receipt = {
        "schema": SCHEMA, "verified": True, "source_pod_id": NAV,
        "destination_pod_id": pod, "installed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sha256": manifest["sha256"], "final_preparation_origin": manifest["final_preparation_origin"],
        "archive_sha256": args.archive_sha256, "archive_bytes": args.archive.stat().st_size,
        "created_files": created, "identical_existing_files_untouched": identical,
        "source_helper_sha256": manifest["source_helper_sha256"],
        "installer_source_sha256": sha(Path(__file__)),
        "new_simulator_actions": 0, "new_model_queries": 0,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"receipt": RECEIPT, "sha256": sha(receipt_path), "verified": True,
                      "created_files": len(created), "identical_files": len(identical)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    exporter = commands.add_parser("export")
    exporter.add_argument("--main-manifest", type=Path, required=True)
    exporter.add_argument("--mean-manifest", type=Path, required=True)
    exporter.add_argument("--source-worker-record", type=Path, required=True)
    exporter.add_argument("--out-dir", type=Path, required=True)
    exporter.add_argument("--proof", action="append", default=[], choices=sorted(PROOFS))
    exporter.add_argument("--root-release-confirmed", action="store_true")
    importer = commands.add_parser("install")
    importer.add_argument("--archive", type=Path, required=True)
    importer.add_argument("--archive-sha256", required=True)
    importer.add_argument("--root-release-confirmed", action="store_true")
    args = parser.parse_args()
    (export if args.command == "export" else install)(args)


if __name__ == "__main__":
    main()
