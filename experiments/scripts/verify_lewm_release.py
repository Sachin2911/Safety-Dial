"""Inspect a saved release using only hashes and JSON, with no experiment execution."""

import argparse
import collections
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(folder):
    manifest = json.loads((folder / "release-manifest.json").read_text())
    expected = manifest["files"]
    actual = {
        str(p.relative_to(folder))
        for p in folder.rglob("*")
        if p.is_file() and p.name not in {"release-manifest.json", "verification.json"}
    }
    if actual != set(expected):
        raise RuntimeError("Unexpected or missing release files")
    for name, info in expected.items():
        path = folder / name
        if path.is_symlink() or path.stat().st_size != info["bytes"]:
            raise RuntimeError("File type or byte count mismatch: " + name)
        if digest(path) != info["sha256"]:
            raise RuntimeError("Release file hash mismatch: " + name)
    if manifest["file_count"] != len(expected):
        raise RuntimeError("Release manifest count mismatch")
    if manifest["total_bytes"] != sum(row["bytes"] for row in expected.values()):
        raise RuntimeError("Release manifest byte sum mismatch")
    evidence = json.loads((folder / "evidence-manifest.json").read_text())["members"]
    seen, reports, audits, metadata = set(), {}, {}, {}
    trial_counts = collections.Counter()
    base = "outputs/lewm-preset-expansion-20261030/"
    roots = {
        "pusht": "outputs/pusht-preset-20261029-v1/",
        "tworoom": base + "tworoom/",
        "reacher": base + "reacher/",
        "cube": base + "cube-v4/",
    }
    extra_json = {
        "outputs/pusht-budget150-20261030-v1/report.json": "pusht_budget",
        "configs/lewm_portable_20261030/design.json": "extension_design",
    }
    with tarfile.open(folder / "evidence.tar.gz", "r|gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if (
                not member.isfile()
                or path.is_absolute()
                or ".." in path.parts
                or member.name not in evidence
                or member.name in seen
                or path.name in {".env", "config.toml", "known_hosts"}
                or path.name.startswith(".env.")
                or path.suffix in {".pem", ".key"}
            ):
                raise RuntimeError("Unexpected, unsafe or duplicate archive member")
            info = evidence[member.name]
            if member.size != info["bytes"]:
                raise RuntimeError("Archive member byte count mismatch")
            stream = archive.extractfile(member)
            value = hashlib.sha256()
            chunks = []
            capture = member.name in extra_json or any(
                member.name == root + suffix
                for root in roots.values()
                for suffix in ("report.json", "independent-audit.json")
            )
            while block := stream.read(1024 * 1024):
                value.update(block)
                if capture:
                    chunks.append(block)
            if value.hexdigest() != info["sha256"]:
                raise RuntimeError("Archive hash mismatch: " + member.name)
            if member.name in extra_json:
                metadata[extra_json[member.name]] = json.loads(b"".join(chunks))
            for name, root in roots.items():
                if member.name == root + "report.json":
                    reports[name] = json.loads(b"".join(chunks))
                if member.name == root + "independent-audit.json":
                    audits[name] = json.loads(b"".join(chunks))
                if member.name.startswith(root) and member.name.endswith("/result.json"):
                    relative = member.name.removeprefix(root)
                    if relative.split("/", 1)[0] in {"development", "evaluation"}:
                        trial_counts[name] += 1
            seen.add(member.name)
    if seen != set(evidence) or set(reports) != set(roots):
        raise RuntimeError("Missing evidence or reports")
    summary = json.loads((folder / "results.json").read_text())
    if set(summary["studies"]) != set(roots):
        raise RuntimeError("Summary does not include all four environments")
    model_hashes = {
        "tworoom": "87aab38843a8e96dc3d6fb325969f6cabe4e11dcd3b1130df8f5e5806620b973",
        "reacher": "7f79c90fcf5ae6ff986852752e62b318ac812c12710a9dc85f8e4997277b34a2",
        "cube": "7e7592bcd43e2b137f14f1178d62c6199a03ce681ab88cc41014655a5ff7ab76",
    }
    checkpoint_names = {name + "-lewm_object.ckpt" for name in model_hashes}
    checkpoint_names.update(name + "-safety.pt" for name in roots)
    if {p.name for p in (folder / "checkpoints").iterdir()} != checkpoint_names:
        raise RuntimeError("Missing or unexpected exposed model checkpoints")
    for name, sha256 in model_hashes.items():
        if expected[f"checkpoints/{name}-lewm_object.ckpt"]["sha256"] != sha256:
            raise RuntimeError("Public world-model identity differs: " + name)
    design = metadata["extension_design"]
    if design["max_episode_actions"] != 150 or design["completion_checkpoints"] != [50, 100, 150]:
        raise RuntimeError("Approved action budget or checkpoints differ")
    budget = metadata["pusht_budget"]
    if not budget["all_original_prefixes_match"]:
        raise RuntimeError("PushT fixed-setting sensitivity differs in its original prefix")
    if summary["studies"]["pusht"]["budget_sensitivity"] != budget["summaries"]:
        raise RuntimeError("PushT sensitivity summary mismatch")
    for name, root in roots.items():
        report = reports[name]
        result_key = "evaluation_at50" if name == "pusht" else "evaluation"
        if summary["studies"][name][result_key] != report["evaluation"]:
            raise RuntimeError("Release summary differs from archived report: " + name)
        if name != "pusht" and not audits[name]["verified"]:
            raise RuntimeError("Independent audit is not verified: " + name)
        for conditions in report["evaluation"].values():
            for outcome in conditions.values():
                if outcome["episodes"] != 6:
                    raise RuntimeError("Reserved denominator differs: " + name)
                if name != "pusht" and set(outcome["completion_at"]) != {"50", "100", "150"}:
                    raise RuntimeError("Missing completion checkpoints: " + name)
        if name == "pusht":
            if not report["audited"]:
                raise RuntimeError("PushT primary audit failed")
            expected_trials = 120
            motion = "outputs/pusht-preset-motion-h5-20261029/model.pt"
        else:
            selected = report["selection"]["selected"]["id"]
            expected_trials = 108 if selected == "authors" else 120
            motion = root + "motion/model.pt"
        if trial_counts[name] != expected_trials:
            raise RuntimeError("Scored-trial count mismatch: " + name)
        if digest(folder / "checkpoints" / (name + "-safety.pt")) != evidence[motion]["sha256"]:
            raise RuntimeError("Exposed safety head differs from original: " + name)
        if len(list((folder / "media" / name).glob("*.mp4"))) != 12:
            raise RuntimeError("Missing paired replays: " + name)
    if not summary["accounting"]["all_three_extension_components_verified"]:
        raise RuntimeError("Combined campaign accounting is incomplete")
    verification = json.loads((folder / "verification.json").read_text())
    if verification["archive_sha256"] != expected["evidence.tar.gz"]["sha256"]:
        raise RuntimeError("Archive verification identity mismatch")
    if verification["archive_members_verified"] != len(seen):
        raise RuntimeError("Archive member count mismatch")
    if verification["payload_files"] != len(expected) + 2:
        raise RuntimeError("Payload count mismatch")
    return {
        "verified": True,
        "archive_members": len(seen),
        "trial_counts": dict(trial_counts),
        "paired_replay_count": 48,
        "payload_files": len(expected) + 2,
        "actual_total_bytes": sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()),
        "archive_sha256": expected["evidence.tar.gz"]["sha256"],
        "scope": "Saved file integrity and report consistency only; no simulation or model queries.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = verify(args.root)
    if args.receipt:
        args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
