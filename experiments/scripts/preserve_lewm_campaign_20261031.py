"""Quiesced whitelist-only preservation for the NEW20261031 LeWM campaign.

Build is rented-host-only and never stops/uploads/deletes anything. Verify is a
stdlib-only local stream check. No account configuration, environment secrets or raw
HDF5/public dataset archive belongs in this package.
"""

import argparse
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tarfile
import time
import traceback

SCHEMA = "lewm-campaign-preservation-20261031-v1"
VOLATILE = {"environment/deadline-guard.json", "environment/deadline-guard.part"}
ROOTS = (
    "results",
    "environment",
    "configs/lewm_campaign_20261031",
    "configs/tworoom12",
    "assets",
    "experiments",
    "reference",
    "references",
    "ruff.toml",
    "docs/research/reports",
    "docs/research/research_notes/Evolutionary LeWM recovery strategies",
    "docs/research/research_notes/Evolutionary adapters for world models",
    "docs/research/honours-gecco",
    "AGENTS.md",
    "firstPrinciples.md",
    "docs/allocatedTopic.md",
    "docs/gecco2027.md",
)
FORBIDDEN_NAMES = {
    ".git",
    ".ssh",
    ".aws",
    ".config",
    ".cache",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "wandb",
    "credentials",
    "credentials.json",
    ".netrc",
    ".runpod",
    ".huggingface",
    ".npmrc",
}
RAW_SUFFIXES = (".h5", ".hdf5", ".hdf", "tworoom.tar.zst", "tworoom.tar.zstd")
TEXT_SUFFIXES = {
    ".json",
    ".jsonl",
    ".log",
    ".txt",
    ".md",
    ".py",
    ".sh",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".out",
    ".err",
    ".csv",
    ".tsv",
}
SECRET = re.compile(
    rb"(?:hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
    rb"|AKIA[A-Z0-9]{16}|(?:WANDB_API_KEY|HF_TOKEN|GITHUB_TOKEN)[\"']?\s*[:=]\s*[\"']?[a-fA-F0-9]{32,64})"
)
NAVIGATION_REQUIRED = (
    "results/navigation-memory-cache-v3/memory.npz",
    "results/navigation-memory-cache-v3/observations.npz",
    "results/navigation-memory-cache-v3/observation-embeddings.npy",
    "results/navigation-memory-cache-v3/frame-index.json",
    "results/navigation-memory-cache-v3/memory-metadata.json",
    "results/navigation-readout-v1/readout.npz",
    "results/navigation-readout-v1/readout-coefficients.json",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def safe_name(value):
    path = PurePosixPath(value)
    require(
        value and not path.is_absolute() and ".." not in path.parts and "." not in path.parts,
        "Unsafe archive/project member path",
    )
    require(str(path) == value and "\\" not in value, "Ambiguous member path")
    return path


def owner(pid):
    try:
        text = (Path("/proc") / str(pid) / "stat").read_text()
        # comm may contain spaces. Fields after its final ')' begin with state3.
        suffix = text[text.rfind(")") + 2 :].split()
        return {"pid": int(pid), "start_ticks": suffix[19], "state": suffix[0]}
    except FileNotFoundError:
        return None


def live(identity):
    current = owner(identity["pid"])
    return (
        current is not None
        and current["state"] != "Z"
        and current["start_ticks"] == str(identity["start_ticks"])
    )


def parse_identity(value):
    pid, ticks = value.split(":", 1)
    return {"pid": int(pid), "start_ticks": ticks}


def process_summary(pid):
    folder = Path("/proc") / str(pid)
    try:
        raw = (folder / "cmdline").read_bytes()
        argv = [part.decode(errors="replace") for part in raw.split(b"\0") if part]
        cwd = os.readlink(folder / "cwd")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    return {
        "identity": owner(pid),
        "argv0": Path(argv[0]).name if argv else "",
        "scripts": [Path(arg).name for arg in argv[1:] if arg.endswith(".py")],
        "cmdline_sha256": hashlib.sha256(raw).hexdigest(),
        "cwd": cwd,
        "raw": raw,
        "argv": argv,
    }


def quiescence(root, workers_path, analysis_owners, control_owners):
    worker_references = []
    if workers_path is not None:
        data = json.loads(workers_path.read_text())
        workers = data["workers"] if isinstance(data, dict) else data
        worker_references.append({"path": str(workers_path), "sha256": digest(workers_path)})
    else:
        records = sorted((root / "environment/workers").glob("*.json"))
        require(bool(records), "No registered experiment worker inventory found")
        workers = []
        for path in records:
            record = json.loads(path.read_text())
            require(
                "pid" in record and "start_ticks" in record,
                "Worker record lacks exact PID/start identity: " + path.name,
            )
            workers.append(
                {
                    "pid": record["pid"],
                    "start_ticks": record["start_ticks"],
                    "name": record.get("name"),
                    "record": str(path.relative_to(root)),
                }
            )
            worker_references.append({"path": str(path.relative_to(root)), "sha256": digest(path)})
    require(
        isinstance(workers, list) and bool(workers),
        "Explicit complete worker identity inventory required",
    )
    own = owner(os.getpid())
    allowed = [*analysis_owners, *control_owners, own]
    allowed_keys = {(i["pid"], str(i["start_ticks"])) for i in allowed}
    analysis_keys = {(i["pid"], str(i["start_ticks"])) for i in [*analysis_owners, own]}
    for identity in workers:
        require(
            "pid" in identity and "start_ticks" in identity,
            "Each worker requires PID and start ticks",
        )
        key = (identity["pid"], str(identity["start_ticks"]))
        require(
            not live(identity) or key in analysis_keys,
            "A declared experiment worker is still live: PID " + str(identity["pid"]),
        )
    # A declared analysis owner may retire during the snapshot. That is safe;
    # unknown replacement project processes are still caught by the full scan.
    for identity in control_owners:
        require(live(identity), "Declared deadline-control identity is not currently live")
    control_keys = {(i["pid"], str(i["start_ticks"])) for i in control_owners}
    active = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        process = process_summary(int(entry.name))
        if not process or not process["identity"] or process["identity"]["state"] == "Z":
            continue
        identity = process["identity"]
        key = (identity["pid"], identity["start_ticks"])
        argv0 = process["argv0"]
        python = argv0.startswith(("python", "pytest", "torchrun"))
        related = str(root).encode() in process["raw"] or Path(process["cwd"]).is_relative_to(root)
        worker_script = any(
            name.startswith(("run_lewm_", "prepare_lewm_", "qualify_lewm_"))
            for name in process["scripts"]
        )
        if related and (python or worker_script):
            summary = {k: v for k, v in process.items() if k not in ("raw", "argv")}
            if key not in allowed_keys:
                raise RuntimeError(
                    "Unretired campaign/unknown Python worker: PID " + str(identity["pid"])
                )
            require(
                not worker_script,
                "An experiment worker cannot be exempted as analysis/control owner",
            )
            if key in control_keys:
                require(
                    b"guard" in process["raw"] and b"deadline" in process["raw"],
                    "Control exemption is only for the independent deadline guard",
                )
            active.append(summary)
    return {
        "checked_utc": utc(),
        "declared_workers": workers,
        "workers_inventory_references": worker_references,
        "all_declared_workers_retired": all(not live(value) for value in workers),
        "all_declared_experiment_workers_retired_or_explicit_analysis_owner": True,
        "allowed_analysis_owners": analysis_owners,
        "allowed_deadline_guard_owners": control_owners,
        "packager_owner": own,
        "observed_allowed_processes": active,
    }


def control_snapshot(root):
    path = root / "environment/deadline-guard.json"
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as source:
        before = identity(os.fstat(source.fileno()))
        raw = source.read()
        require(
            identity(os.fstat(source.fileno())) == before, "Guard heartbeat changed within one read"
        )
    heartbeat = json.loads(raw)
    require(heartbeat["pod_id"] == os.environ.get("RUNPOD_POD_ID"), "Guard belongs to another pod")
    stamp = dt.datetime.fromisoformat(heartbeat["heartbeat_utc"])
    age = (dt.datetime.now(dt.timezone.utc) - stamp).total_seconds()
    require(
        0 <= age <= 90 and heartbeat["status"] == "armed",
        "Independent deadline guard must remain live/armed",
    )
    return {
        "source_path": "environment/deadline-guard.json",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "captured_utc": utc(),
        "source_identity": before,
        "heartbeat": heartbeat,
        "volatile_source_excluded_from_stable_inventory": True,
        "guard_remains_active_and_local_control_mirror_is_separately_retained": True,
    }


def excluded(name):
    parts = PurePosixPath(name).parts
    if any(part in FORBIDDEN_NAMES or part.startswith(".env") for part in parts):
        return "account/cache/env directory or file"
    basename = parts[-1].lower()
    if basename.endswith(RAW_SUFFIXES):
        return "raw public HDF5 or dataset archive, provenance retained separately"
    if basename.endswith((".pem", ".key", ".p12", ".pfx")) or basename.startswith(
        ("id_rsa", "id_ed25519")
    ):
        return "credential/key file"
    return None


def identity(info):
    return {
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
        "device": info.st_dev,
        "inode": info.st_ino,
        "mode": stat.S_IMODE(info.st_mode),
    }


def inventory(root):
    files, absent = {}, []
    skipped = [
        {
            "path": name,
            "reason": "Exact independently changing guard heartbeat/temp; captured in control_snapshot, guard remains active",
        }
        for name in sorted(VOLATILE)
    ]
    seen = set()

    def walk(path, name):
        safe_name(name)
        if name in VOLATILE:
            return
        if name in seen:
            return
        seen.add(name)
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            skipped.append(
                {"path": name, "reason": "symlink not followed", "target": os.readlink(path)}
            )
            return
        reason = excluded(name)
        if reason:
            skipped.append({"path": name, "reason": reason})
            return
        if stat.S_ISDIR(info.st_mode):
            for child in sorted(path.iterdir(), key=lambda p: p.name):
                walk(child, name + "/" + child.name)
        elif stat.S_ISREG(info.st_mode):
            files[name] = identity(info)
        else:
            skipped.append({"path": name, "reason": "nonregular file"})

    for name in ROOTS:
        path = root / name
        if path.exists() or path.is_symlink():
            walk(path, name)
        else:
            absent.append(name)
    return {"files": files, "skipped": skipped, "absent_whitelist_roots": absent}


def data_reference(path):
    payload = json.loads(path.read_text())
    pending = [payload]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            if all(key in value for key in ("repo_id", "revision", "archive_sha256")):
                require(
                    len(value["archive_sha256"]) == 64, "Raw dataset reference needs archive SHA256"
                )
                value = {
                    key: value[key]
                    for key in (
                        "repo_id",
                        "revision",
                        "archive",
                        "archive_bytes",
                        "archive_sha256",
                        "member",
                        "h5_sha256",
                        "h5_bytes",
                    )
                    if key in value
                }
                return {
                    "reference_path": str(path),
                    "reference_sha256": digest(path),
                    "dataset": value,
                    "raw_h5_included": False,
                    "provenance": "Pinned public archive SHA256/revision identifies the excluded HDF5 member; preserve any supplied member SHA too",
                }
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    raise RuntimeError("Provide pinned public dataset revision/archive SHA256 provenance")


class HashReader:
    def __init__(self, stream, check_secret):
        self.stream, self.check_secret = stream, check_secret
        self.hasher, self.tail, self.bytes = hashlib.sha256(), b"", 0

    def read(self, size=-1):
        data = self.stream.read(size)
        if self.check_secret and SECRET.search(self.tail + data):
            raise RuntimeError(
                "Credential-like bytes detected in selected source; file must be reviewed"
            )
        self.tail = data[-512:]
        self.hasher.update(data)
        self.bytes += len(data)
        return data


def verify(archive, *, expected_archive_sha256=None):
    if expected_archive_sha256:
        require(digest(archive) == expected_archive_sha256, "Archive SHA256 differs")
    observed, manifest = {}, None
    with tarfile.open(archive, "r|*") as stream:
        for member in stream:
            safe_name(member.name)
            if member.name != "MANIFEST-20261031.json":
                require(
                    any(
                        member.name == root or member.name.startswith(root + "/") for root in ROOTS
                    ),
                    "Archive member is outside the preservation whitelist",
                )
                require(
                    member.name not in VOLATILE,
                    "Live guard heartbeat must be a manifest control snapshot",
                )
                require(
                    excluded(member.name) is None,
                    "Archive contains excluded raw/credential/cache file",
                )
            require(
                member.isfile() and not member.issym() and not member.islnk(),
                "Archive contains unsafe nonregular member",
            )
            require(member.name not in observed, "Duplicate archive member")
            source = stream.extractfile(member)
            require(source is not None, "Archive member has no readable bytes")
            hasher, count = hashlib.sha256(), 0
            if member.name == "MANIFEST-20261031.json":
                raw = source.read()
                count = len(raw)
                hasher.update(raw)
                manifest = json.loads(raw)
            else:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    hasher.update(block)
                    count += len(block)
            require(count == member.size, "Truncated archive member")
            observed[member.name] = {"size": count, "sha256": hasher.hexdigest()}
    require(
        manifest is not None and manifest["schema"] == SCHEMA,
        "Preservation manifest absent/unknown",
    )
    expected = manifest["files"]
    require(
        set(observed) == set(expected) | {"MANIFEST-20261031.json"},
        "Archive/manifest member sets differ",
    )
    for name, record in expected.items():
        require(
            observed[name]["size"] == record["size"]
            and observed[name]["sha256"] == record["sha256"],
            "Member byte/hash verification failed: " + name,
        )
    return {
        "verified_utc": utc(),
        "archive_sha256": digest(archive),
        "members": len(observed),
        "every_member_streamed_and_hashed": True,
        "source_files": len(expected),
        "total_source_bytes": sum(record["size"] for record in expected.values()),
        "manifest_sha256": observed["MANIFEST-20261031.json"]["sha256"],
        "manifest": manifest,
    }


def build(args):
    root, out = args.project_root.resolve(), args.out_dir.resolve()
    require(
        os.environ.get("RUNPOD_POD_ID") == args.pod_id,
        "Build belongs on its declared current rented pod",
    )
    require(not out.exists(), "Use a fresh preservation namespace")
    for selected in ROOTS:
        require(not out.is_relative_to(root / selected), "Output may not be inside a selected root")
    require(args.pod_id and root.is_dir(), "Explicit current project and pod identity required")
    analysis = [parse_identity(value) for value in args.allow_analysis_owner]
    controls = [parse_identity(value) for value in args.deadline_guard_owner]
    before_quiet = quiescence(root, args.workers_json, analysis, controls)
    before_control = control_snapshot(root)
    raw_reference = data_reference(args.raw_data_reference)
    acquisition_path = root / "results/navigation-memory-cache-v3/complete.json"
    if acquisition_path.is_file():
        acquisition = json.loads(acquisition_path.read_text()).get("acquisition", {})
        if acquisition:
            require(
                acquisition["archive_sha256"] == raw_reference["dataset"]["archive_sha256"]
                and acquisition["dataset_revision"] == raw_reference["dataset"]["revision"],
                "HDF5 acquisition provenance differs from pinned public archive",
            )
            raw_reference["verified_h5_acquisition"] = acquisition
            raw_reference["acquisition_record_sha256"] = digest(acquisition_path)
    before = inventory(root)
    require(bool(before["files"]), "Whitelist selected no files")
    required = list(args.required)
    if args.require_navigation_bank:
        required.extend(NAVIGATION_REQUIRED)
    require(
        all(name in before["files"] for name in required),
        "A required bank/readout/source artifact is missing, excluded or a symlink",
    )
    source_bytes = sum(record["size"] for record in before["files"].values())
    needed = int(source_bytes * 1.03) + len(before["files"]) * 2048 + 16 * 1024 * 1024
    require(
        shutil.disk_usage(out.parent).free >= needed,
        "Insufficient free space for a conservative archive upper bound",
    )
    out.mkdir(parents=True)
    write(
        out / "preservation-owner.json",
        {"identity": owner(os.getpid()), "started_utc": utc(), "pod_id": args.pod_id},
    )
    archive = out / "lewm-campaign-20261031.tar.gz"
    partial = archive.with_suffix(archive.suffix + ".part")
    try:
        records = {}
        with tarfile.open(partial, "w:gz", compresslevel=args.compression_level) as stream:
            for index, (name, expected) in enumerate(sorted(before["files"].items())):
                fd = os.open(root / name, os.O_RDONLY | os.O_NOFOLLOW)
                with os.fdopen(fd, "rb") as source:
                    require(
                        identity(os.fstat(source.fileno())) == expected,
                        "Selected file changed before read: " + name,
                    )
                    reader = HashReader(source, True)
                    member = tarfile.TarInfo(name)
                    member.size, member.mode = expected["size"], expected["mode"] & 0o777
                    member.mtime = expected["mtime_ns"] // 1_000_000_000
                    member.uid = member.gid = 0
                    member.uname = member.gname = ""
                    stream.addfile(member, reader)
                    require(
                        reader.bytes == expected["size"]
                        and identity(os.fstat(source.fileno())) == expected,
                        "Selected file mutated during snapshot: " + name,
                    )
                    records[name] = {**expected, "sha256": reader.hasher.hexdigest()}
                if index % 100 == 0:
                    print(
                        json.dumps(
                            {
                                "phase": "archiving",
                                "files": index + 1,
                                "total_files": len(before["files"]),
                            }
                        ),
                        flush=True,
                    )
            after = inventory(root)
            require(before == after, "Source inventory/stat identities changed during snapshot")
            after_quiet = quiescence(root, args.workers_json, analysis, controls)
            manifest = {
                "schema": SCHEMA,
                "snapshot_utc": utc(),
                "pod_id": args.pod_id,
                "project_root": str(root),
                "source_builder_sha256": digest(Path(__file__)),
                "whitelist_roots": list(ROOTS),
                "files": records,
                "skipped": before["skipped"],
                "absent_whitelist_roots": before["absent_whitelist_roots"],
                "quiescence_before": before_quiet,
                "quiescence_after": after_quiet,
                "control_snapshot_before": before_control,
                "control_snapshot": control_snapshot(root),
                "raw_public_dataset": raw_reference,
                "all_failed_partial_evidence_retained": "Every selected regular result/environment file is included regardless of outcome/status; no result filtering",
                "snapshot_contract": "All source inventory/stat identities stable before/after; no symlink following; every archived member read back and hash checked",
                "new_model_simulator_calls": 0,
            }
            raw = (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
            info = tarfile.TarInfo("MANIFEST-20261031.json")
            info.size, info.mode, info.mtime = len(raw), 0o600, int(time.time())
            stream.addfile(info, io.BytesIO(raw))
        with partial.open("rb") as source:
            os.fsync(source.fileno())
        checked = verify(partial)
        # Re-check all live source hashes, not just metadata. This also detects a
        # same-size writer that restored mtime during the package operation.
        for name, record in records.items():
            fd = os.open(root / name, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, "rb") as source:
                require(
                    identity(os.fstat(source.fileno())) == before["files"][name],
                    "Source identity changed after snapshot: " + name,
                )
                require(
                    hashlib.file_digest(source, "sha256").hexdigest() == record["sha256"],
                    "Source bytes changed after snapshot: " + name,
                )
                require(
                    identity(os.fstat(source.fileno())) == before["files"][name],
                    "Source changed during final hash: " + name,
                )
        require(before == inventory(root), "Source tree changed during archive verification")
        final_quiet = quiescence(root, args.workers_json, analysis, controls)
        partial.replace(archive)
        receipt = {k: v for k, v in checked.items() if k != "manifest"}
        receipt.update(
            archive=str(archive),
            archive_bytes=archive.stat().st_size,
            completed_utc=utc(),
            final_quiescence=final_quiet,
            final_control_snapshot=control_snapshot(root),
            source_builder_sha256=digest(Path(__file__)),
            new_model_simulator_calls=0,
        )
        write(out / "manifest.json", manifest)
        write(out / "receipt.json", receipt)
        return receipt
    except BaseException:
        write(
            out / "failure.json",
            {
                "failed_utc": utc(),
                "error": traceback.format_exc(),
                "partial_archive_retained": str(partial),
                "new_model_simulator_calls": 0,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    package = commands.add_parser("build")
    package.add_argument("--project-root", type=Path, default=Path("/workspace/Safety-Dial"))
    package.add_argument("--out-dir", type=Path, required=True)
    package.add_argument("--pod-id", required=True)
    package.add_argument(
        "--workers-json",
        type=Path,
        help="Optional explicit experiment-only inventory; default reads every environment/workers/*.json record",
    )
    package.add_argument(
        "--allow-analysis-owner", action="append", default=[], metavar="PID:START_TICKS"
    )
    package.add_argument(
        "--deadline-guard-owner", action="append", default=[], metavar="PID:START_TICKS"
    )
    package.add_argument("--raw-data-reference", type=Path, required=True)
    package.add_argument("--require-navigation-bank", action="store_true")
    package.add_argument("--required", action="append", default=[])
    package.add_argument("--compression-level", type=int, choices=range(1, 10), default=3)
    check = commands.add_parser("verify")
    check.add_argument("archive", type=Path)
    check.add_argument("--expected-archive-sha256")
    args = parser.parse_args()
    if args.command == "build":
        result = build(args)
    else:
        result = verify(args.archive, expected_archive_sha256=args.expected_archive_sha256)
        result.pop("manifest")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
