#!/usr/bin/env python3
"""Verify saved AP source gates, manifests and journal bytes without model/simulator imports."""

import argparse
import gzip
import hashlib
import json
import pathlib
import re
import tarfile

from accounting import sha
from audit import read_json as read


def verify(base, status_path, config_path, final=False):
    status = read(status_path)
    original = read(config_path)
    official_path = pathlib.Path(__file__).parent / "evidence/official-source-verification.json"
    official = read(official_path)
    proofs = {r["run_id"]: r for r in status["runs"]}
    errors = []
    checks = []
    official_sources = {
        name: item["sha256"]
        for name, item in official["files"].items()
        if name.endswith(".py")
    }
    checked_files = checked_bytes = restored_journals = 0

    def require(value, message):
        if not value:
            errors.append(message)

    def norm(name):
        return re.sub(r"[-_.]+", "-", name.lower())

    require(
        official.get("passed") is True
        and official["revision"] == original["ap_revision"]
        and all(
            item["matches"] is True and item["sha256"] == item["recorded_runtime_sha256"]
            for item in official["files"].values()
        ),
        "Pinned official source verification is inconsistent",
    )
    roots = sorted(p.parent for p in base.glob("*/run.json"))
    missing_roots = sorted(set(proofs) - {root.name for root in roots})
    require(len(proofs) == len(status["runs"]), "Status contains duplicate run identities")
    if final:
        require(not missing_roots, "Final status contains undownloaded runs: " + str(missing_roots))
        require(
            bool(status.get("hosts"))
            and all(host.get("ready_to_destroy") is True for host in status["hosts"].values()),
            "Final host snapshots are not all idle with verified backups",
        )
    for root in roots:
        run_id = root.name
        record = proofs.get(run_id, {})
        upload = record.get("upload") or {}
        require(
            record.get("phase") == "finished"
            and not record.get("job_alive")
            and not record.get("supervisor_alive")
            and upload.get("verified") is True
            and not upload.get("error"),
            run_id + ": backup is not terminal and verified",
        )
        manifest = read(root / "manifest.json")
        spec = read(root / "run.json")
        require(
            sha(root / "manifest.json") == upload.get("manifest_sha256"),
            run_id + ": manifest differs from current verified proof",
        )
        receipts = [read(p) for p in root.glob("download-receipt-*.json")]
        valid = [
            r
            for r in receipts
            if r.get("selected_evidence_verified") is True
            and not r.get("failed_files")
            and r.get("revision") == upload.get("revision")
            and r.get("manifest_sha256") == upload.get("manifest_sha256")
        ]
        require(bool(valid), run_id + ": pinned selected-evidence download proof missing")
        verified_names = set()
        for receipt in valid:
            for item in receipt["verified_files"]:
                name = item["path"]
                if name in verified_names:
                    continue
                verified_names.add(name)
                path = root / name
                expected = manifest["files"].get(name)
                require(
                    expected == {"bytes": item["bytes"], "sha256": item["sha256"]},
                    run_id + ": receipt/manifest mismatch " + name,
                )
                require(
                    path.is_file()
                    and path.stat().st_size == item["bytes"]
                    and sha(path) == item["sha256"],
                    run_id + ": selected file changed " + name,
                )
                checked_files += 1
                checked_bytes += item["bytes"]
        require(
            spec.get("source_sha256") == sha(root / "source.tar.gz"),
            run_id + ": source archive differs from launch specification",
        )
        case_paths = sorted((root / "metrics/cases").glob("*.json"))
        checked = {
            "run_id": run_id,
            "revision": upload.get("revision"),
            "manifest_sha256": upload.get("manifest_sha256"),
            "local_verified_files": len(verified_names),
            "full_backup_verification": upload.get("verification"),
            "scientific_cases": len(case_paths),
        }
        if case_paths:
            gate = read(root / "artifacts/engineering-gate.json")
            prov = read(root / "artifacts/provenance.json")
            config = read(root / "config.json")
            task = read(case_paths[0])["compact"]["task"]
            require(
                gate.get("passed") is True and gate.get("task") == task,
                run_id + ": engineering gate failed/task differs",
            )
            require(
                gate["ap_revision"] == prov["ap_revision"] == original["ap_revision"],
                run_id + ": AP source revision differs",
            )
            source_root = base / gate["source_run"]
            require(
                source_root.is_dir()
                and gate == read(source_root / "artifacts/engineering-gate.json"),
                run_id + ": engineering gate differs from verified source run",
            )
            require(
                read(source_root / "run.json")["session_id"] == spec["session_id"]
                and read(source_root / "run.json")["instance_id"] == spec["instance_id"],
                run_id + ": gate belongs to a different host",
            )
            require(
                spec["image"].endswith(
                    "@sha256:7a63a5d1d11c429a82b3bb0cfef6d1e48e61ee8c28680ef0e5dd83950428bace"
                ),
                run_id + ": image differs",
            )
            require(
                config["task_specs"][task] == original["task_specs"][task],
                run_id + ": asset pins differ",
            )
            with tarfile.open(root / "source.tar.gz", "r:gz") as archive:
                for name in ("common.py", "instrumentation.py", "engineering.py", "evaluate.py"):
                    data = archive.extractfile(name).read()
                    actual = hashlib.sha256(data).hexdigest()
                    require(
                        actual
                        == gate["source_hashes"][name]
                        == prov["instrumentation_hashes"][name],
                        run_id + ": source gate/hash mismatch " + name,
                    )
            require(
                prov["source_hashes"] == official_sources,
                run_id + ": native AP source bytes differ from pinned official commit",
            )
            require(
                prov["native_protocol_sha256"]
                == official["files"]["protocols/main.json"]["sha256"]
                and prov["native_scalers_sha256"]
                == official["files"]["protocols/scalers.json"]["sha256"],
                run_id + ": native protocol/scalers differ from pinned official commit",
            )
            packages = {(norm(name), version) for name, version in prov["packages"]}
            require(
                all(
                    (norm(name), version) in packages for name, version in gate["versions"].items()
                ),
                run_id + ": gate package versions absent from recorded runtime",
            )
            expected_versions = ("3.10.0", "1.0.43") if task == "reacher" else ("3.12.0", "1.0.44")
            require(
                (gate["versions"]["mujoco"], gate["versions"]["dm-control"]) == expected_versions,
                run_id + ": task simulator versions differ",
            )
            cache = read(root / "artifacts/cache-receipt.json")
            cache_root = base / cache["source_run"]
            cache_manifest = read(cache_root / "manifest.json")
            require(
                cache == read(cache_root / "artifacts/cache.json"),
                run_id + ": cache receipt differs from source",
            )
            require(
                cache_manifest["files"].get("artifacts/retrieval-cache/" + task + ".npy")
                == {"bytes": cache["bytes"], "sha256": cache["sha256"]},
                run_id + ": full cache lacks verified backup",
            )
            require(
                cache["checkpoint_sha256"] == original["task_specs"][task]["weights_sha256"],
                run_id + ": cache checkpoint differs",
            )
            checked.update(
                task=task,
                gate_run=gate["source_run"],
                gate_source_hashes=gate["source_hashes"],
                cache_run=cache["source_run"],
            )
            for path in case_paths:
                case = path.stem
                names = manifest["files"]
                journals = [
                    name
                    for name in (
                        "metrics/journals/" + case + ".jsonl",
                        "metrics/journals/" + case + ".jsonl.gz",
                    )
                    if name in names
                ]
                require(
                    len(journals) == 1
                    and all(
                        name in names and names[name]["bytes"] > 0
                        for name in (
                            "metrics/cases/" + case + ".json",
                            "metrics/trajectories/" + case + ".json.gz",
                            "artifacts/videos/" + case + ".mp4",
                        )
                    ),
                    run_id + ": incomplete backed-up case evidence " + case,
                )
                if journals and journals[0].endswith(".gz"):
                    trajectory = read(root / "metrics/trajectories" / (case + ".json.gz"))
                    journal = trajectory["journal"]
                    journal_path = root / journals[0]
                    h = hashlib.sha256()
                    count = 0
                    require(
                        journal["path"] == journals[0]
                        and journal["compressed_bytes"] == names[journals[0]]["bytes"]
                        and journal["compressed_sha256"] == names[journals[0]]["sha256"],
                        run_id + ": compressed journal metadata mismatch " + case,
                    )
                    with gzip.open(journal_path, "rb") as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            h.update(block)
                            count += len(block)
                            if count > journal["original_bytes"]:
                                raise ValueError("Journal exceeds declared original bytes")
                    require(
                        count == journal["original_bytes"]
                        and h.hexdigest() == journal["original_sha256"],
                        run_id + ": journal restoration differs " + case,
                    )
                    restored_journals += 1
        checks.append(checked)
    report = {
        "passed": not errors,
        "errors": errors,
        "final_requested": final,
        "undownloaded_status_runs": missing_roots,
        "official_source_verification": official,
        "runs": checks,
        "verified_selected_files": checked_files,
        "verified_selected_bytes": checked_bytes,
        "restored_scientific_gzip_journals": restored_journals,
        "scope": "Local selected evidence rehashed against pinned manifests; full remote backup receipts retained, without downloading full latent caches or all videos. Engineering gates checked against original source-run records, packaged source bytes, runtime versions and asset/cache pins.",
    }
    report["verifier"] = {
        "path": "LeWM_Experiment1/ap_reproduction/verify_sources.py",
        "sha256": sha(pathlib.Path(__file__)),
    }
    report["inputs"] = {
        "run_root": str(base),
        "status": str(status_path),
        "status_sha256": sha(status_path),
        "config": str(config_path),
        "config_sha256": sha(config_path),
        "official_source_verification": str(official_path),
        "official_source_verification_sha256": sha(official_path),
    }
    return report


def main():
    project = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=pathlib.Path, default=project / "runs/AP")
    parser.add_argument(
        "--status",
        type=pathlib.Path,
        default=project / "computeConfigs/records/ap_reproduction/combined-session-status.json",
    )
    parser.add_argument(
        "--config", type=pathlib.Path, default=project / "ap_reproduction/assets.json"
    )
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    report = verify(args.run_root, args.status, args.config, args.final)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "passed",
                    "errors",
                    "verified_selected_files",
                    "verified_selected_bytes",
                    "restored_scientific_gzip_journals",
                )
            }
        )
    )
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
