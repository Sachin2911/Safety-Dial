"""Export public derived ledgers and a complete gallery from independently audited evidence."""

import json
import shutil
import time
from pathlib import Path

from followup_evidence import FIGURES, ROOT, RUN, atomic, audit_all, sha, verdict

PAPER = ROOT / "docs/research/reports/maniguard-phase3-part1-gecco"
NOTES = ROOT / "docs/research/research_notes"


def main():
    result = audit_all(RUN / "evidence")
    records = [r for r in result["records"] if r["part"] == 2]
    assert len(records) == 20 and sum(r["valid"] and r["full_episode"] for r in records) == 16
    source = RUN / "evidence/audit-snapshot-part2.json"
    shutil.copyfile(RUN / "evidence/audit.json", source)
    public = {
        "schema": "maniguard-part2-public-audit-v1",
        "at": time.time(),
        "clock": result["clock"],
        "records": records,
        "recording_scope": "Part2 only; Part3 progresses in a separate run subtree",
        "source_independent_audit_sha256": sha(source),
        "source_snapshot": str(source.relative_to(ROOT)),
        "protocol_sha256": sha(RUN / "part2/protocol.json"),
        "diagnostics_sha256": sha(RUN / "part2/diagnostics.json"),
        "export_script_sha256": sha(__file__),
    }
    atomic(NOTES / "ManiGuard phase3 Part2 audit.json", public)
    shutil.copyfile(
        RUN / "part2/diagnostics.json", NOTES / "ManiGuard phase3 Part2 diagnostics.json"
    )
    lines = [
        "# Parts 2 and 3 recorded evidence",
        "",
        "These are recorded simulator observations and native videos. Offline composition",
        "adds no physics step or policy query. Every episode in the frame-index snapshot appears below.",
        "Video links require access to the approved private dataset; pictures are in Git.",
        "",
        "Part 2 panels are A/B above C/D. A is identity at interval eight; B uses caps",
        "at interval eight; C uses the same caps at interval four; D uses correction-triggered",
        "refresh from four to eight. Part 3 recording panels are B/C, ordered by label.",
        "Native episode endings become black panels labelled ENDED rather than repeated physics.",
        "All videos retain chronological order at 20 fps.",
        "",
    ]
    for stage in ("part2", "part3-recording", "part3-comparison"):
        receipt_path = RUN / "evidence" / (stage + "-video-backup.json")
        if not receipt_path.exists():
            continue
        receipt = json.loads(receipt_path.read_text())
        assert receipt["private"] and receipt["download_verified"]
        lines += [
            f"## {stage}: complete matched videos",
            "",
            f"Pinned private revision: `{receipt['revision']}`. Every file was credential-scanned",
            "and re-downloaded for SHA256 verification.",
            "",
            "| Block | Overview | Wrist |",
            "| --- | --- | --- |",
        ]
        files = {Path(f["path"]).name: f for f in receipt["files"]}
        blocks = sorted({name.split("-")[0] for name in files if name.endswith(".mp4")})
        for block in blocks:
            links = [
                f"[{camera}]({files[block + '-' + camera + '.mp4']['url']})"
                for camera in ("overview", "wrist")
            ]
            lines.append(f"| {block} | {' | '.join(links)} |")
        lines.append("")
    frame_index = json.loads((FIGURES / "frames/index.json").read_text())
    by_episode = {}
    for item in frame_index["pictures"]:
        by_episode.setdefault(item["episode_id"], {})[item["camera"]] = item
    available = {r["episode_id"]: r for r in result["records"] if r["valid"] and r["full_episode"]}
    lines += [
        "## All event and episode pictures",
        "",
        "For every violating episode, the three panels show one second before, at, and one",
        "second after the first native violation, clipped to episode bounds. Every other",
        "episode shows its start, middle and final observation. Time and step labels are",
        "visible. Raw NPZ source hashes are in the [frame index](../../figures/ManiGuard/phase3-followup/frames/index.json).",
        "Selection is complete rather than restricted to favorable cases.",
        "",
        "| Episode | Outcome | Overview | Wrist |",
        "| --- | --- | --- | --- |",
    ]
    for episode, pictures in sorted(by_episode.items()):
        if episode not in available:
            continue
        links = [
            f"[{camera}](../../figures/ManiGuard/phase3-followup/frames/{Path(pictures[camera]['path']).name})"
            for camera in ("overview", "wrist")
        ]
        lines.append(f"| {episode} | {verdict(available[episode])} | {' | '.join(links)} |")
    lines += [
        "",
        "Part 3 comparison videos and evolved-controller lineage exist only if its frozen",
        "forecast audit permits the physical search. Unstarted search has no invented video or curve.",
        "",
    ]
    (PAPER / "followup-videos.md").write_text("\n".join(lines))
    print(
        json.dumps(
            {
                "part2_attempts": len(records),
                "gallery_episodes": len(by_episode),
                "source_snapshot_sha256": sha(source),
            }
        )
    )


if __name__ == "__main__":
    main()
