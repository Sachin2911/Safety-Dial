"""Package independently audited task-18 results for the existing private datasets."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package(root, output, kind):
    report = read(root / "analysis/summary.json")
    rows = [r for r in read(root / "analysis/scored-rows.json") if r["env"] == kind]
    target = output / kind / "campaigns/lewm18"
    if target.exists():
        raise FileExistsError(target)
    target.mkdir(parents=True)
    tables = [r for r in report["tables"] if r["env"] == kind]
    write(target / "results/tables.json", tables)
    write(target / "results/scored-rows.json", rows)
    write(target / "results/full-accounting.json", report)
    write(
        target / "results/benchmark-context.json",
        {
            "scope": "Twelve unfiltered dataset pairs with native 25-action execution blocks. "
            "This separate context panel is not a full published-benchmark replication.",
            "groups": [r for r in report["benchmark_context_strata"] if r["env"] == kind],
        },
    )
    write(
        target / "results/paired-contrasts.json",
        [r for r in report["contrasts"] if r["env"] == kind],
    )
    fields = [
        "margin",
        "method",
        "n",
        "safe",
        "margin_breach",
        "violation",
        "unfinished",
        "engaged",
        "controller_failure",
        "mean_J_safe",
        "safe_speed_n",
        "mean_safe_actions",
        "mean_safe_simulated_seconds",
    ]
    with (target / "results/final-summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in tables if r["stage"] == "H")
    attempts = []
    members = {}
    archive_path = target / "data/raw-attempts.tar.gz"
    archive_path.parent.mkdir()
    host_roots = sorted((root / "hosts").iterdir()) if (root / "hosts").exists() else [root]
    folders = [
        (host.name, host, folder)
        for host in host_roots
        for folder in sorted((host / "attempts").glob("*"))
        if folder.is_dir()
    ]
    with tarfile.open(archive_path, "w:gz") as archive:
        for host_id, host, folder in folders:
            candidates = [
                folder / name for name in ("result.json", "failed-start.json", "started.json")
            ]
            metadata = next((p for p in candidates if p.exists()), None)
            if metadata is None:
                raise ValueError(f"Unidentified attempt: {folder.name}")
            row = read(metadata)
            if row["job"]["env"] != kind:
                continue
            attempts.append(
                {
                    "cohort": host_id,
                    "physical_instance": host_id.split("-")[0],
                    "included_in_final_cohort": report["hosts_by_environment"][kind] == host_id,
                    **row,
                }
            )
            for path in sorted(folder.rglob("*")):
                if not path.is_file():
                    continue
                name = f"{host_id}/{path.relative_to(host / 'attempts')}"
                members[name] = {"bytes": path.stat().st_size, "sha256": sha(path)}
                archive.add(path, arcname=name)
    write(target / "data/raw-files.json", members)
    with tarfile.open(archive_path) as archive:
        found = set()
        for member in archive:
            assert member.isfile() and member.name not in found
            blob = archive.extractfile(member).read()
            assert hashlib.sha256(blob).hexdigest() == members[member.name]["sha256"]
            found.add(member.name)
        assert found == set(members)
    (target / "data/attempts.jsonl").write_text(
        "".join(json.dumps(r, allow_nan=False) + "\n" for r in attempts)
    )
    protocol_dir = target / "protocol"
    protocol_dir.mkdir()
    for path in sorted(root.glob("*.json")):
        shutil.copy2(path, protocol_dir / path.name)
    for dirname in ("source", "environment", "logs"):
        if (root / dirname).exists():
            shutil.copytree(root / dirname, protocol_dir / dirname)
    if (root / "hosts").exists():
        for host in host_roots:
            destination = protocol_dir / "hosts" / host.name
            destination.mkdir(parents=True)
            for path in host.glob("*.json"):
                shutil.copy2(path, destination / path.name)
            for path in host.glob("*.py"):
                shutil.copy2(path, destination / path.name)
            for dirname in ("source", "environment", "logs"):
                if (host / dirname).exists():
                    shutil.copytree(host / dirname, destination / dirname)
    shutil.copytree(ROOT / "configs/lewm18", protocol_dir / "configs")
    shutil.copytree(ROOT / "outputs/lewm18/source", protocol_dir / "local-preparation-source")
    code = target / "code"
    code.mkdir()
    for name in (
        "merge_lewm18_evidence.py",
        "analyze_lewm18.py",
        "media_lewm18.py",
        "package_lewm18_evidence.py",
        "upload_lewm18_evidence.py",
        "repair_lewm18_pusht.py",
        "write_lewm18_tables.py",
        "write_lewm18_report.py",
        "write_lewm18_paper.py",
        "publish_lewm18_closeout.py",
        "manage_lewm18.py",
        "preserve_lewm18_outputs.py",
    ):
        shutil.copy2(ROOT / "experiments/scripts" / name, code / name)
    for path in sorted((ROOT / "progress/receipts").glob("18-*")):
        if path.is_file() and path.suffix in (".json", ".md"):
            shutil.copy2(path, protocol_dir / path.name)
    for path in sorted((ROOT / "progress").glob("rental-lewm18*.json")):
        shutil.copy2(path, protocol_dir / path.name)
    media = [r for r in read(root / "media/index.json") if r["environment"] == kind]
    (target / "media").mkdir()
    for item in media:
        for key in ("video", "picture", "goal_image", "paper_picture"):
            shutil.copy2(root / "media" / item[key], target / "media" / item[key])
    galleries = sorted((root / "media").glob(f"{kind}-trajectory-gallery-*"))
    for gallery in galleries:
        shutil.copy2(gallery, target / "media" / gallery.name)
    write(target / "media/index.json", media)
    for name in ("comparison.png", "comparison.pdf"):
        if (root / "analysis" / name).exists():
            shutil.copy2(root / "analysis" / name, target / "results" / name)
    names = {
        "native": "Native LeWM",
        "canonical": "Original SafeCEM",
        "fixed": "Tuned fixed settings",
        "feedback": "Feedback adapter",
    }
    final = [r for r in tables if r["stage"] == "H"]
    highest_margin = max(r["margin"] for r in final)
    strict = {r["method"]: r for r in final if r["margin"] == highest_margin}
    n = strict["fixed"]["n"]
    unit = f"{highest_margin * 100:g} cm" if kind == "cube" else f"{highest_margin:g} pixels"
    headline = (
        f"At the largest clearance ({unit}), the original checked planner finished safely "
        f"in {strict['canonical']['safe']} of {n} attempts. Tuned fixed settings finished "
        f"in {strict['fixed']['safe']} of {n}, and feedback in {strict['feedback']['safe']} of {n}. "
        f"Their margin-breach counts were {strict['canonical']['margin_breach']}, "
        f"{strict['fixed']['margin_breach']} and {strict['feedback']['margin_breach']}, respectively."
    )
    interpretation = (
        "Better fixed settings recovered safe completion on this panel. "
        if strict["fixed"]["safe"] > strict["canonical"]["safe"]
        else "The tuned fixed settings did not improve the safe-completion count at this clearance. "
    ) + (
        "Feedback needs its own comparison against the tuned fixed setting; "
        "a numerical difference alone does not establish an advantage. "
        "These are 12 cases with three seeds each, not 36 independent task geometries."
    )
    lines = [
        f"# Task 18: {'original Cube object moving' if kind == 'cube' else 'PushT'}",
        "",
        "We compare four ways to plan with the same frozen world model. "
        "At each safety margin, the three checked planners use the same safety rule. "
        "Only their planning settings differ.",
        "",
        "For Cube, success means moving the object within 4 cm of its goal. "
        "This differs from the earlier Cube gripper-reaching study. "
        "These are LeWM dataset-pair goals, not the full official five-goal OGBench evaluation."
        if kind == "cube"
        else "PushT keeps the pinned benchmark's joint pusher/block position and block-angle "
        "success rule. The reported score is not shape coverage.",
        "",
        "## What happened",
        "",
        headline,
        "",
        interpretation,
        "",
        "## Final cases",
        "",
        "Safe means the task finished without crossing the requested keep-out margin. "
        "Safety checks cover the pusher centre or gripper pinch point, not the whole robot. "
        "Unfinished and unsafe attempts both receive the full 50-action penalty.",
        "",
        "| Margin | Planner | Safe completion | Margin breaches | Unfinished | Moved | Capped actions |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(final, key=lambda r: (r["margin"], list(names).index(r["method"]))):
        margin = f"{row['margin'] * 100:g} cm" if kind == "cube" else f"{row['margin']:g} px"
        n = row["n"]
        lines.append(
            f"| {margin} | {names[row['method']]} | {row['safe']}/{n} | "
            f"{row['margin_breach']}/{n} | {row['unfinished']}/{n} | "
            f"{row['engaged']}/{n} | {row['mean_J_safe']:.2f} |"
        )
    lines += [
        "",
        "Lower capped actions means better progress toward safe completion across all attempts. "
        "Successful-episode action counts and uncontended planning latency are separate "
        "columns in [the full tables](results/tables.json). Zero observed breaches does "
        "not establish a zero underlying violation rate. Cases are clustered; "
        "[paired intervals](results/paired-contrasts.json) retain whole case blocks.",
        "A zero-width empirical bootstrap interval can occur when all observed case effects "
        "are identical. It does not prove population equivalence or zero uncertainty.",
        "",
        "The [separate native context panel](results/benchmark-context.json) uses twelve "
        "unfiltered demonstration pairs. It reports cases that already met the goal at "
        "reset separately from those that did not. These are different cases and a "
        "different replanning interval from the matched comparison above.",
        "",
        "## Watch the same cases",
        "",
        "These are the first two final cases and first planner seed, selected before outcomes. "
        "Each video shows all four methods at the highest safety margin. Playback uses "
        "eight actions per second and does not show actual planner computation time. "
        "Images are saved simulator observations. The adjacent map explains the keep-out area. "
        "Red is forbidden; amber is the extra clearance. The map is for the viewer and "
        "is not supplied to LeWM. These examples can fail even when other cases succeed. "
        "Use the full table to judge performance.",
        "",
        "Cube videos also show the goal image and the object's three-dimensional distance "
        "to its goal. A target can be above the table. The faint cube-shaped marker in the "
        "scene belongs to data collection; this test uses the cube position in the goal "
        "image, with its top-down position marked in green on the map."
        if kind == "cube"
        else "The PushT goal includes both pusher and block position, plus "
        "block angle. Reaching the green circle with the pusher alone is not success.",
        "",
    ]
    for item in media:
        lines.append(
            f"- Case {item['case_index']}: [video](media/{item['video']}), "
            f"[picture](media/{item['picture']}), [goal image](media/{item['goal_image']})."
        )
    lines += [
        "",
        "The trajectory gallery shows **every final case at the first seed**, "
        "with all four planners side by side. Blue tracks the monitored pusher or gripper; "
        "the dashed path tracks the object. It includes successes and failures:",
        "",
    ]
    for gallery in galleries:
        if gallery.suffix == ".png":
            page = gallery.stem.rsplit("-", 1)[-1]
            lines.append(
                f"- Gallery page {page}: [picture](media/{gallery.name}) "
                f"or [PDF](media/{gallery.stem}.pdf)."
            )
    lines += [
        "",
        "## Check the evidence",
        "",
        "The first PushT cohort used a pre-reset state for its first safety check. "
        "Reset advances the simulator slightly. We corrected that anchor and reran "
        "the same panels, seeds and candidate settings. Only the corrected cohort "
        "supplies the final PushT comparisons. All original attempts remain in the "
        "archive and resource totals; the reset-repair receipt records the change.",
        "",
        "- `data/attempts.jsonl` records every attempt, including development and failures.",
        "- `data/raw-attempts.tar.gz` retains the exact traces, actions, saved frames and results.",
        "- `data/raw-files.json` gives the archive members' sizes and SHA-256 hashes.",
        "- `protocol/` contains stage freezes, source snapshots, input pins and host records.",
        "- `results/` contains independently rescored outcomes, uncertainty and work accounting.",
        "- [The small CSV table](results/final-summary.csv) opens in a spreadsheet. "
        "Margin units are pixels for PushT and metres for Cube. `safe_speed_n` is the "
        "number of safe successes used for the conditional speed means; a blank mean "
        "means there were none.",
        "",
        "For a final raw-data row, require `included_in_final_cohort` to be true and "
        '`job.stage` to equal `"H"`. `job.method` identifies the planner. The original '
        "and repaired PushT cohorts can contain the same job ID, so retain the `cohort` "
        "field when joining files. `scored-rows.json` contains the three margin views "
        "of each native final rollout; do not count those views as extra experiments.",
        "",
        "Native final rollouts are scored at three margins; these views are not independent "
        "new episodes. This campaign compares planning settings and feedback. It does not "
        "test or establish an evolutionary optimizer advantage.",
    ]
    (target / "README.md").write_text("\n".join(lines) + "\n")
    write(
        target / "sha256-manifest.json",
        {
            str(p.relative_to(target)): {"bytes": p.stat().st_size, "sha256": sha(p)}
            for p in sorted(target.rglob("*"))
            if p.is_file()
        },
    )
    return {"environment": kind, "attempts": len(attempts), "archive_members": len(members)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/lewm18-publication/stage")
    args = parser.parse_args()
    checks = [package(args.root, args.output, kind) for kind in ("pusht", "cube")]
    write(args.output.parent / "packaging-validation.json", checks)
    print(json.dumps(checks))


if __name__ == "__main__":
    main()
