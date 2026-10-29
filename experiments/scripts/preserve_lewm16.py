"""Immutable task-16 evidence snapshots in the existing private research dataset."""

import argparse
import concurrent.futures
import datetime as dt
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.lewm16_protocol import OUT, digest, write_json

REPO = "Sachioster/maniguard-jar-replication-20261022"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--stages", nargs="*")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    if not args.snapshot.replace("-", "").isalnum():
        raise ValueError("Snapshot name must be alphanumeric with optional hyphens")
    stage = OUT / "preservation" / args.snapshot
    if not stage.exists():
        stage.mkdir(parents=True)
        groups = {}
        for folder in sorted((OUT / "remote/attempts").iterdir()):
            if not folder.is_dir():
                continue
            ticket = json.loads((folder / "ticket.json").read_text())
            if args.stages and ticket["stage"] not in args.stages:
                continue
            if not (folder / "result.json").exists():
                raise ValueError(f"Attempt not terminal: {folder.name}")
            groups.setdefault(ticket["stage"], []).append(folder)
        raw_files, results = {}, []
        for name, folders in sorted(groups.items()):
            with tarfile.open(stage / f"attempts-{name}.tar.gz", "w:gz") as archive:
                for folder in folders:
                    archive.add(folder, arcname=folder.name)
                    results.append(json.loads((folder / "result.json").read_text()))
                    for path in sorted(folder.rglob("*")):
                        if path.is_file():
                            raw_files[str(path.relative_to(OUT / "remote"))] = {
                                "sha256": digest(path),
                                "bytes": path.stat().st_size,
                            }
        write_json(stage / "raw-files.json", raw_files)
        with (stage / "results.jsonl").open("w") as stream:
            for row in results:
                stream.write(json.dumps(row) + "\n")
        with tarfile.open(stage / "metadata-and-sources.tar.gz", "w:gz") as archive:
            for child in sorted((OUT / "remote").iterdir()):
                if child.name != "attempts":
                    archive.add(child, arcname="remote/" + child.name)
            for path in sorted((ROOT / "progress/receipts").glob("16-*")):
                if path.is_file():
                    archive.add(path, arcname=str(path.relative_to(ROOT)))
            for name in (
                "progress/16-lewm-safecem-campaign-plan.md",
                "progress/rental-lewm16-20261026.json",
            ):
                archive.add(ROOT / name, arcname=name)
            if args.snapshot.startswith("final-closeout"):
                for name in (
                    "progress/README.md",
                    "progress/receipts/README.md",
                    "progress/theOdessy.tex",
                    "progress/theOdessy.pdf",
                    "docs/research/research_notes/LeWM safe progress task 16.md",
                    "experiments/scripts/report_lewm16.py",
                    "experiments/scripts/closeout_lewm16.py",
                    "experiments/scripts/preserve_lewm16.py",
                ):
                    archive.add(ROOT / name, arcname=name)
            archive.add(ROOT / "configs/lewm16", arcname="configs/lewm16")
            if (OUT / "analysis").exists():
                archive.add(OUT / "analysis", arcname="analysis")
        manifest = {
            p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
            for p in sorted(stage.iterdir())
            if p.is_file()
        }
        write_json(
            stage / "index.json",
            {
                "schema": "lewm16-preservation-v1",
                "snapshot": args.snapshot,
                "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "stages": sorted(groups),
                "attempts": len(results),
                "files": manifest,
            },
        )
    index = json.loads((stage / "index.json").read_text())
    for name, item in index["files"].items():
        if digest(stage / name) != item["sha256"]:
            raise ValueError("Preservation snapshot hash changed")
    if not args.upload:
        print(json.dumps({"snapshot": args.snapshot, "attempts": index["attempts"]}))
        return
    from dotenv import dotenv_values
    from huggingface_hub import HfApi, hf_hub_download

    secret = dotenv_values(ROOT / ".env")
    sensitive = [v.encode() for k, v in secret.items() if v and ("TOKEN" in k or "KEY" in k)]
    for parent in (OUT / "remote", ROOT / "configs/lewm16"):
        for path in parent.rglob("*"):
            if path.is_file() and path.suffix in {".json", ".jsonl", ".log", ".txt", ".md", ".py"}:
                if any(value in path.read_bytes() for value in sensitive):
                    raise ValueError("Sensitive value in preservation input")
    api = HfApi(token=secret.get("HF_TOKEN"))
    if not api.repo_info(REPO, repo_type="dataset").private:
        raise ValueError("Preservation destination is not private")
    prefix = f"16_lewm16_campaign/{args.snapshot}"
    commit = api.upload_folder(
        folder_path=stage,
        path_in_repo=prefix,
        repo_id=REPO,
        repo_type="dataset",
        commit_message=f"task16: {args.snapshot} evidence",
    )
    revision = commit.oid

    def verify(path):
        copied = Path(
            hf_hub_download(
                REPO,
                prefix + "/" + path.name,
                repo_type="dataset",
                revision=revision,
                token=secret.get("HF_TOKEN"),
                cache_dir=OUT / "hf-verify",
            )
        )
        if digest(copied) != digest(path):
            raise ValueError("Private object hash differs")
        return path.name

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        checked = list(pool.map(verify, sorted(stage.iterdir())))
    receipt = dict(
        snapshot=args.snapshot,
        private=True,
        repo=REPO,
        prefix=prefix,
        revision=revision,
        verified_files=len(checked),
        attempts=index["attempts"],
        index_sha256=digest(stage / "index.json"),
    )
    write_json(ROOT / f"progress/receipts/16-preservation-{args.snapshot}.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error_type": type(exc).__name__}), file=sys.stderr)
        raise SystemExit(1) from None
