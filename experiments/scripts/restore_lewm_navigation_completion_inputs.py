"""Restore only frozen science dependencies from three verified immutable archives.

No original episode/job/result/resource ledger, worker, ready record or live guard
is imported into the replacement's canonical project. Original full trees remain
outside it for later analysis assembly. No model, image array or simulator opens.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat

OWNERS = {
    "navigation": "vukovc3t4o1dui",
    "pusht": "wq4ar2qx2ff6r4",
    "recovery": "nj2fqkp2prgk8x",
}
TREE_ROOTS = (
    "experiments",
    "configs",
    "assets",
    "reference",
    "references",
)
EXACT_ROOT_FILES = {"AGENTS.md", "firstPrinciples.md", "ruff.toml"}
SHARED = (
    "results/navigation-memory-cache-v3/",
    "results/navigation-readout-v1/",
    "results/navigation-final-cases-v1/",
)
OMIT_BASENAMES = {
    "resources.json",
    "interactions.jsonl",
    "complete.json",
    "status.json",
    "memory-fetch-events.jsonl",
    "events.jsonl",
    "report.json",
    "failure.json",
    "live.json",
    "started.json",
    "state.json",
    "result.json",
    "journal-recovery.json",
}
SOFTWARE_GATES = {
    "environment/navigation-common-source-gate-v2/tests.xml",
    "environment/navigation-common-source-gate-v2/ruff.log",
    "environment/navigation-decoded-cost-qualified-v3/tests.xml",
    "environment/navigation-decoded-cost-qualified-v3/ruff.log",
    "environment/recovery-foundation-source-v1/tests.xml",
    "environment/recovery-foundation-source-v1/ruff.log",
}
FORBIDDEN_PARTS = {
    ".git",
    ".ssh",
    ".aws",
    ".config",
    ".cache",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "wandb",
    "credentials",
    "credentials.json",
    ".netrc",
}
REQUIRED_BANK = (
    "results/navigation-memory-cache-v3/memory.npz",
    "results/navigation-memory-cache-v3/observations.npz",
    "results/navigation-memory-cache-v3/observation-embeddings.npy",
    "results/navigation-memory-cache-v3/frame-index.json",
    "results/navigation-memory-cache-v3/memory-metadata.json",
    "results/navigation-readout-v1/readout.npz",
    "results/navigation-readout-v1/readout-coefficients.json",
)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_name(value):
    text = str(value)
    if "/Safety-Dial/" in text:
        text = text.split("/Safety-Dial/", 1)[1]
    path = PurePosixPath(text)
    require(
        text
        and not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in text
        and str(path) == text,
        "Unsafe original dependency path",
    )
    return text


def split_options(values):
    result = {}
    for value in values:
        name, path = value.split("=", 1)
        require(name in OWNERS and name not in result, "Exactly one entry per original pod")
        result[name] = Path(path).resolve()
    require(set(result) == set(OWNERS), "All three original archives are required")
    return result


def forbidden(name):
    path = PurePosixPath(name)
    return (
        any(part in FORBIDDEN_PARTS or part.startswith(".env") for part in path.parts)
        or path.name.lower().endswith(
            (".h5", ".hdf5", ".hdf", ".pem", ".key", ".p12", ".pfx", ".tar.zst")
        )
        or path.name.startswith(("id_rsa", "id_ed25519"))
    )


def selected(name, *, dependency=False):
    if forbidden(name):
        return False
    path = PurePosixPath(name)
    if name in EXACT_ROOT_FILES or any(name.startswith(prefix + "/") for prefix in TREE_ROOTS):
        return True
    if name in SOFTWARE_GATES:
        return True
    if name.startswith("results/evidence-audits/"):
        return path.suffix == ".json"
    if any(name.startswith(prefix) for prefix in SHARED):
        return (
            path.name not in OMIT_BASENAMES
            and "/jobs/" not in name
            and "/episode/" not in name
            and path.suffix not in (".jsonl", ".log", ".err", ".out")
        )
    # Qualification JSONs in environment are copied only when an exact original
    # frozen dependency requires them. Never import worker/control identities.
    if dependency and name.startswith("environment/"):
        return (
            path.suffix == ".json"
            and "/workers/" not in name
            and path.name not in {"ready.json", "deadline-guard.json", "deadline-guard.part"}
            and "guard" not in path.name.lower()
        )
    if dependency and name == "results/navigation-final-analysis-v3/inputs.json":
        return True
    return False


def original_file(origin, name):
    path = origin / canonical_name(name)
    require(path.resolve().is_relative_to(origin), "Origin symlink escaped extracted archive")
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode), "Original dependency must be regular, never symlink")
    return path


def load_origins(origins, receipts):
    result = {}
    for lane in OWNERS:
        root = origins[lane]
        manifest_path = root / "MANIFEST-20261031.json"
        manifest = json.loads(manifest_path.read_text())
        receipt = json.loads(receipts[lane].read_text())
        require(
            manifest["schema"] == "lewm-campaign-preservation-20261031-v1"
            and manifest["pod_id"] == OWNERS[lane]
            and receipt["every_member_streamed_and_hashed"] is True
            and receipt["manifest_sha256"] == digest(manifest_path),
            "Original archive lacks exact all-member verification provenance",
        )
        require(len(receipt["archive_sha256"]) == 64, "Original archive SHA required")
        result[lane] = {
            "root": root,
            "manifest": manifest,
            "receipt": receipt,
            "archive_sha256": receipt["archive_sha256"],
            "manifest_sha256": digest(manifest_path),
            "receipt_sha256": digest(receipts[lane]),
        }
    return result


def candidates(records, name, expected=None):
    output = []
    for lane in OWNERS:
        record = records[lane]
        file = record["manifest"]["files"].get(name)
        if file and (expected is None or file["sha256"] == expected):
            path = original_file(record["root"], name)
            require(path.stat().st_size == file["size"], "Extracted source size differs")
            output.append((lane, path, file))
    require(bool(output), "Missing exact original dependency: " + name)
    return output


def dependency_closure(records, authority_template):
    expected = {}

    def add(name, sha):
        name = canonical_name(name)
        require(selected(name, dependency=True), "Unsafe/non-science dependency rejected: " + name)
        require(name not in expected or expected[name] == sha, "Conflicting frozen SHA: " + name)
        expected[name] = sha

    for row in authority_template["profiles"]:
        ref = row["protocol"]
        name = canonical_name(ref["path"])
        matches = candidates(records, name, ref["sha256"])
        path = matches[0][1]
        require(digest(path) == ref["sha256"], "Original profile bytes differ")
        protocol = json.loads(path.read_text())
        add(name, ref["sha256"])
        for declaration in (protocol, protocol["evaluation"]):
            for field in ("source_sha256", "asset_sha256"):
                for dependency, sha in declaration.get(field, {}).items():
                    add(dependency, sha)
        for job in protocol["jobs"]:
            case = job["case"]
            if "input_path" in case:
                add(case["input_path"], case["input_sha256"])
    original_inputs = authority_template["original_analysis_inputs"]
    add(original_inputs["path"], original_inputs["sha256"])
    return expected


def copy_one(source, target, expected):
    require(not target.is_symlink(), "Never replace destination symlinks")
    if target.exists():
        require(target.is_file() and digest(target) == expected, "Existing destination differs")
        return "reused_exact_existing_bytes"
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".restore-part-" + str(os.getpid()))
    require(not partial.exists(), "Retain any prior partial restoration")
    before = source.stat()
    with source.open("rb") as incoming, partial.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
        os.fchmod(outgoing.fileno(), stat.S_IMODE(before.st_mode))
        outgoing.flush()
        os.fsync(outgoing.fileno())
    after = source.stat()
    require(
        (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        == (after.st_size, after.st_mtime_ns, after.st_ctime_ns),
        "Immutable extracted source changed during copy",
    )
    require(digest(partial) == expected, "Copied dependency SHA differs")
    os.link(partial, target)  # Atomic visibility with exclusive no-overwrite semantics.
    partial.unlink()
    require(digest(target) == expected, "Destination readback differs")
    return "copied_and_readback_verified"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", action="append", required=True)
    parser.add_argument("--archive-receipt", action="append", required=True)
    parser.add_argument("--authority-template", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("RUNPOD_POD_ID") == args.pod_id, "Restore only on the declared new pod")
    require(not args.out.exists(), "Never overwrite a restore receipt")
    destination = args.destination.resolve()
    require(destination == Path("/workspace/Safety-Dial"), "Explicit canonical replacement project")
    origins = split_options(args.origin)
    require(
        all(not root.is_relative_to(destination) for root in origins.values()),
        "Origins must stay outside canonical project",
    )
    records = load_origins(origins, split_options(args.archive_receipt))
    authority_template = json.loads(args.authority_template.read_text())
    require(len(authority_template["profiles"]) == 10, "All ten unstarted profiles required")
    expected = dependency_closure(records, authority_template)
    names = set(expected)
    for record in records.values():
        names.update(name for name in record["manifest"]["files"] if selected(name))
    require(set(REQUIRED_BANK) <= names, "Complete observed bank and readout must be restored")
    copied, conflicts = {}, []
    destination.mkdir(parents=True, exist_ok=True)
    for name in sorted(names):
        options = candidates(records, name, expected.get(name))
        lane, source, info = options[0]
        require(digest(source) == info["sha256"], "Extracted source SHA differs: " + name)
        all_versions = {
            record["manifest"]["files"][name]["sha256"]
            for record in records.values()
            if name in record["manifest"]["files"]
        }
        if len(all_versions) > 1:
            conflicts.append(
                {
                    "path": name,
                    "selected_sha256": info["sha256"],
                    "reason": "exact frozen dependency"
                    if name in expected
                    else "fixed origin priority navigation,pusht,recovery",
                    "all_preserved_versions": sorted(all_versions),
                }
            )
        target = destination / name
        require(target.resolve().is_relative_to(destination), "Destination escaped project")
        disposition = copy_one(source, target, info["sha256"])
        copied[name] = {
            "sha256": info["sha256"],
            "bytes": info["size"],
            "original_owner_pod_id": OWNERS[lane],
            "origin_archive_sha256": records[lane]["archive_sha256"],
            "source_relative_path": name,
            "disposition": disposition,
            "required_by_frozen_profiles": name in expected,
        }
    require(
        all(copied[name]["sha256"] == sha for name, sha in expected.items()),
        "Dependency closure incomplete",
    )
    result = {
        "schema": "lewm-completion-input-restoration-v1",
        "verified": True,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "destination_pod_id": args.pod_id,
        "destination": str(destination),
        "source_script_sha256": digest(Path(__file__)),
        "authority_template_sha256": digest(args.authority_template),
        "origins": {
            lane: {
                "pod_id": OWNERS[lane],
                "extracted_root": str(record["root"]),
                "archive_sha256": record["archive_sha256"],
                "manifest_sha256": record["manifest_sha256"],
                "receipt_sha256": record["receipt_sha256"],
            }
            for lane, record in records.items()
        },
        "sha256": {name: value["sha256"] for name, value in copied.items()},
        "files": copied,
        "conflicting_original_versions_preserved_outside_canonical": conflicts,
        "foreign_physical_episode_or_job_directories_copied": 0,
        "foreign_resources_or_interaction_ledgers_copied": 0,
        "old_workers_ready_or_live_guards_copied": 0,
        "new_model_queries": 0,
        "new_simulator_actions": 0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "verified": True,
                "files": len(copied),
                "required_dependencies": len(expected),
                "manifest": str(args.out),
                "sha256": digest(args.out),
            }
        )
    )


if __name__ == "__main__":
    main()
