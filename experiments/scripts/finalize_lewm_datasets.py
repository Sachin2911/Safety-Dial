"""Validate and index the dedicated LeWM dataset additions before upload."""

from __future__ import annotations

import collections
import hashlib
import json
import shutil
import tarfile

import imageio.v2 as imageio
import numpy as np
from PIL import Image
import pyarrow.parquet as pq

from consolidate_lewm_media import BG, FPS, HEIGHT, OUT, ROOT, S, sha, write_json


def grid(dest, rows, name, gif=False):
    movies = []
    for row in rows:
        reader = imageio.get_reader(dest / row["video"])
        movies.append([frame for frame in reader])
        reader.close()
    cols = 2 if len(rows) <= 4 else 3
    height = HEIGHT * int(np.ceil(len(rows) / cols))
    frames = []
    for k in range(max(len(m) for m in movies)):
        panel = Image.new("RGB", (cols * S, height), BG)
        for i, movie in enumerate(movies):
            panel.paste(
                Image.fromarray(movie[min(k, len(movie) - 1)]),
                ((i % cols) * S, (i // cols) * HEIGHT),
            )
        frames.append(np.asarray(panel))
    video = dest / "videos/comparisons" / f"{name}.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        video,
        fps=FPS,
        codec="libx264",
        quality=7,
        ffmpeg_params=["-threads", "1"],
        macro_block_size=16,
    ) as writer:
        for frame in frames:
            writer.append_data(frame)
    pic = dest / "pictures" / f"{name}.png"
    Image.fromarray(frames[-1]).save(pic)
    if gif:
        ims = [
            Image.fromarray(f).resize((cols * S // 2, height // 2), Image.Resampling.LANCZOS)
            for f in frames
        ]
        ims[0].save(
            video.with_suffix(".gif"), save_all=True, append_images=ims[1:], duration=125, loop=0
        )
    return {
        "name": name,
        "episodes": [r["episode_id"] for r in rows],
        "video": str(video.relative_to(dest)),
        "picture": str(pic.relative_to(dest)),
        "gif": str(video.with_suffix(".gif").relative_to(dest)) if gif else None,
        "paired_case": rows[0]["case_id"],
        "paired_seed": rows[0]["seed"],
        "margin": rows[0]["margin"],
        "fps": FPS,
        "note": "Illustrative paired cases, not aggregate performance evidence",
    }


def campaign(env, task):
    dest = OUT / "stage" / env / "campaigns" / f"lewm{task}"
    media = json.loads((dest / "media-index.json").read_text())
    summary = json.loads((dest / "summary.json").read_text())
    manifest = json.loads((dest / "data/raw-files.json").read_text())
    seen = set()
    with tarfile.open(dest / "data/raw-attempts.tar.gz") as arc:
        for member in arc:
            assert member.isfile() and member.name not in seen
            blob = arc.extractfile(member).read()
            assert hashlib.sha256(blob).hexdigest() == manifest[member.name]["sha256"]
            seen.add(member.name)
    assert seen == set(manifest)
    for name, key in [
        ("episodes", "attempts"),
        ("trajectories", "control_rows"),
        ("dense-states", "dense_state_rows"),
    ]:
        assert pq.read_metadata(dest / "data" / f"{name}.parquet").num_rows == summary[key]
    rows = [json.loads(s) for s in (dest / "data/episodes.jsonl").read_text().splitlines()]
    assert len({r["episode_id"] for r in rows}) == summary["attempts"]
    assert len(media) == summary["gallery_episodes"]
    comparisons = []
    for stage in ["A", "B"] if task == 16 else ["H"]:
        eligible = [r for r in media if r["stage"] == stage]
        case = min(r["case_id"] for r in eligible)
        seed = min(r["seed"] for r in eligible)
        margin = max(r["margin"] for r in eligible)
        chosen = [
            r
            for r in eligible
            if r["case_id"] == case and r["seed"] == seed and r["margin"] == margin
        ]
        order = (
            ["native", "canonical", "fixed", "adaptive"]
            if task == 16
            else ["historical", "tuned", "random-r0", "reinforce-r0", "evolution-r0"]
        )
        chosen = sorted(
            (r for r in chosen if r["condition"] in order),
            key=lambda r: order.index(r["condition"]),
        )
        assert len(chosen) == (4 if stage == "A" else 2 if stage == "B" else 5)
        comparisons.append(
            grid(dest, chosen, f"{stage}-paired-high-margin", gif=stage in {"A", "H"})
        )
    write_json(dest / "comparison-index.json", comparisons)
    valid = [r for r in rows if r["status"] == "complete"]
    final = [r for r in valid if r["stage"] == ("B" if task == 16 else "H")]
    groups = collections.defaultdict(list)
    for r in final:
        j = json.loads(r["job_json"])
        method = j.get("optimizer", r["condition"])
        groups[(r["margin"], method)].append(r)
    table = [
        "| Margin | Method | Safe completion | Hazard violations | Margin breaches | Unfinished | Engaged |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for (margin, method), items in sorted(groups.items()):
        label = f"{margin:g} px" if env == "tworoom" else f"{margin * 100:g} cm"
        cols = [
            f"{sum(bool(x.get(k)) for x in items)}/{len(items)}"
            for k in [
                "requested_safe_completion",
                "any_violation",
                "margin_breach",
                "unfinished",
                "engaged",
            ]
        ]
        table.append("| " + " | ".join([label, method] + cols) + " |")
    report = "16-lewm-results.md" if task == 16 else "17-lewm-evolution-results.md"
    claim = (
        "Strong fixed planner settings recovered safe progress; a substantial adaptive benefit was not established."
        if task == 16
        else "The evolutionary completion-advantage criterion was not met. Shared preset anchors won validation in eight of ten evolution runs across both environments."
    )
    source = (
        "cf77289d39b19a21f68dc16c723ee3f55acb1745"
        if task == 16
        else "0e8001f31aa5ce2bfe07bd63d48491f439d0791f"
    )
    provenance = {
        "task": task,
        "environment": env,
        "source_repo": "Sachioster/maniguard-jar-replication-20261022",
        "source_revision": source,
        "all_raw_attempt_files_preserved": len(seen),
        "derived_tables": "Lossless result_json/job_json plus typed query columns; raw archives authoritative",
        "media": "Postprocessing from recorded states, 8 actions/second playback; no new policy queries or simulator action steps",
        "sample_selection": "First case and planner seed for every final method/margin, plus first failed completion per method/margin in lexical order",
        "gallery_not_all_episodes": True,
        "current_source_code": "code/consolidation",
        "invalid_record_note": "Four task-16 Reacher failures retained; four unrecorded physics integrations are reconstructed in accounting, not invented in dense-state rows.",
    }
    write_json(dest / "provenance.json", provenance)
    sections = [
        f"# Task {task}: {env} evidence",
        f"\n{claim}\n",
        f"{summary['attempts']} attempts: {summary['complete']} complete and {summary['failed']} engineering failures. "
        f"{summary['control_rows']} recorded action rows and {summary['dense_state_rows']} dense state rows. "
        f"{len(media)} illustrative episode videos and {len(comparisons)} paired comparison videos.\n",
        "## Fresh-case outcomes\n",
        "\n".join(table),
        "\nZero observed breaches is not a safety guarantee. Cases and trained controllers are clustered. "
        "Fixed references are evaluated once per case/seed; adaptive results pool five optimizer repeats in task 17. "
        "Equal allowances are not equal consumed interaction work.\n",
        f"[Full campaign report](protocol/{report}) includes both environments, uncertainty and accounting.\n",
        "## Watch the comparisons\n",
    ]
    for row in comparisons:
        sections.append(
            f"- [{row['name']}](" + row["video"] + f") ([picture]({row['picture']})). "
            f"Case {row['paired_case']}, planner seed {row['paired_seed']}."
        )
        if row["gif"]:
            sections.append(f"\n![{row['name']}]({row['gif']})\n")
    sections += [
        "\n## Data and reproduction\n",
        "- `data/episodes.parquet` and `episodes.jsonl`: every attempt and its result, stage, controller, counters and full job.\n"
        "- `data/trajectories.parquet`: states after each recorded environment action, the actual commands and reconstructed clipped controls.\n"
        "- `data/dense-states.parquet`: every saved physics integration for Reacher, or action state for TwoRoom.\n"
        "- `data/raw-attempts.tar.gz`: exact attempt folders, including traces, decisions, candidate snapshots, tickets and failures.\n"
        "- `data/raw-files.json`: per-member sizes and SHA-256 hashes.\n"
        "- `code/`, `protocol/`, `controllers/` when applicable: configuration, source and optimizer checkpoints. "
        "Existing repository `checkpoints/` retains the frozen LeWM weights; pinned identities remain in configs.\n"
        "- `results/`: original comparative figures and machine-readable statistics; figures may contain both environments.\n"
        "- `media-index.json`, `comparison-index.json`: exact episode identities, outcomes, replay provenance and selection reasons.\n",
        "Video frames are reconstructed with the simulator renderer from saved states. They are not live camera recordings or new rollouts. "
        "Red is the true hazard, amber is the requested inflated margin, blue is the recorded path, and green is the goal. "
        "For Reacher the green arm indicates the joint-pose goal; fingertip proximity alone is not its success rule. "
        "Footers count violations over recorded dense states. These illustrative clips include outcome-selected failures and do not estimate performance. "
        "The full tables above include all final cases.\n",
        "The task-16 invalid recorder attempts lack four physics integrations; their accounting reconstruction is preserved in the engineering invalidation receipt. "
        "Postprocessing never fills that gap with invented state observations.\n",
        "## Episode gallery\n",
        "| Episode | Outcome | Selected controller | Media |",
        "|---|---|---|---|",
    ]
    for r in media:
        outcome = (
            "safe completion"
            if r["requested_safe_completion"]
            else "unsafe completion"
            if not r["unfinished"]
            else "unfinished"
        )
        sections.append(
            f"| `{r['episode_id']}` | {outcome} | {r['selected_candidate'] or 'fixed/feedback recipe'} | "
            f"[video]({r['video']}) / [picture]({r['picture']}) |"
        )
    (dest / "README.md").write_text("\n".join(sections) + "\n")
    # Preview indexes are local development products, not a second gallery contract.
    preview = dest / "media-index-preview.json"
    if preview.exists():
        preview.unlink()
    code = dest / "code/consolidation"
    code.mkdir(parents=True, exist_ok=True)
    for name in ["consolidate_lewm_media.py", "finalize_lewm_datasets.py"]:
        shutil.copy2(ROOT / "experiments/scripts" / name, code / name)
    summary.update(comparison_videos=len(comparisons), episode_videos=len(media))
    write_json(dest / "summary.json", summary)
    write_json(
        dest / "sha256-manifest.json",
        {
            str(p.relative_to(dest)): {"bytes": p.stat().st_size, "sha256": sha(p)}
            for p in sorted(dest.rglob("*"))
            if p.is_file() and p.name != "sha256-manifest.json"
        },
    )
    return summary


def root_index(env, title, summaries):
    dest = OUT / "stage" / env
    original = (OUT / f"{title}-README-original.md").read_text()
    history = dest / "history/before-lewm16-17"
    history.mkdir(parents=True, exist_ok=True)
    (history / "README.md").write_text(original)
    shutil.copy2(OUT / f"{title}-manifest-original.json", history / "sha256-manifest.json")
    head, metadata, body = original.split("---", 2)
    assert not head.strip()
    for task in [16, 17]:
        for name in ["episodes", "trajectories", "dense-states"]:
            metadata += f"  - config_name: lewm{task}_{name.replace('-', '_')}\n    data_files: campaigns/lewm{task}/data/{name}.parquet\n"
    intro = f"""# SafetyDial-{title}

## Latest: tasks 16 and 17, 26 October 2026

This private dataset now includes the completed safe-progress and evolutionary-controller campaigns.
The earlier task {"12" if env == "tworoom" else "13"} data, videos, model weights and source remain at their original paths.

| Campaign | All attempts | Complete | Failed engineering starts | Episode videos | Paired videos |
|---|---:|---:|---:|---:|---:|
"""
    for s in summaries:
        intro += f"| [Task {s['task']}](campaigns/lewm{s['task']}/README.md) | {s['attempts']} | {s['complete']} | {s['failed']} | {s['episode_videos']} | {s['comparison_videos']} |\n"
    intro += """
**Current interpretation:** task 16 recovered safe progress with strong fixed planner settings.
Task 17 completed the evolution/random-search/REINFORCE comparison and did not establish an evolutionary completion advantage.
Safety margins were not relaxed. The campaign pages report completion, violations, unfinished episodes and engagement together.
Equal resource ceilings did not make actual consumed work equal; there is no search-efficiency claim.

- [Fixed versus canonical planning: paired replay](campaigns/lewm16/videos/comparisons/A-paired-high-margin.mp4)
- [Optimizer-selected controllers: paired replay](campaigns/lewm17/videos/comparisons/H-paired-high-margin.mp4)
- [Updated combined-paper framing](paper/paperCombination.pdf)
- [Task 16 full data and gallery](campaigns/lewm16/README.md)
- [Task 17 full data and gallery](campaigns/lewm17/README.md)

![Fixed versus canonical planning](campaigns/lewm16/videos/comparisons/A-paired-high-margin.gif)

Videos are reconstructed from recorded states, not new scored experiments. Galleries are illustrative subsets;
all attempted episodes and raw trajectories are included. Every new file is covered by the updated root checksum
manifest; raw archive members and campaign packages have separate manifests. Previous root files are unchanged
except this index and its checksum manifest; their prior versions are retained under `history/before-lewm16-17/`.

**Scope note on the historical narrative below:** zero observed violations is not a general safety guarantee.
Three optimizer runs remain three independent replications, not nine because all cross-pairings are compared.
Earlier “held-out layouts” were withheld from search but had appeared in development; tasks 16–17 use fresh cases.
The latest comparisons have stronger fixed and ordinary-learning controls. Preserve those distinctions when citing the old results.

## Historical campaign description
"""
    body = body.lstrip()
    heading = f"# SafetyDial-{title}"
    assert body.startswith(heading)
    body = body[len(heading) :].lstrip()
    (dest / "README.md").write_text("---" + metadata + "---\n\n" + intro + "\n" + body)
    paper = dest / "paper"
    paper.mkdir(exist_ok=True)
    for name in ["paperCombination.pdf", "paperCombination.tex"]:
        shutil.copy2(ROOT / "progress" / name, paper / name)
    # The PDF's relative evidence links continue to resolve from its location on the Hub.
    receipts = paper / "receipts"
    receipts.mkdir(exist_ok=True)
    for name in [
        "16-lewm-results.md",
        "16-final-statistics.json",
        "16-final-audit.json",
        "17-lewm-evolution-results.md",
        "17-final-statistics.json",
        "17-final-audit.json",
        "17-external-status-20261026.json",
        "09-unisafe-results.md",
        "15-regrasp-results.md",
    ]:
        shutil.copy2(ROOT / "progress/receipts" / name, receipts / name)
    doc = dest / "docs/research/reports"
    doc.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "docs/research/reports/Unifying safe adaptation research.md", doc)
    root_manifest = json.loads((OUT / f"{title}-manifest-original.json").read_text())
    for p in sorted(dest.rglob("*")):
        if p.is_file() and p != dest / "sha256-manifest.json":
            root_manifest[str(p.relative_to(dest))] = {"bytes": p.stat().st_size, "sha256": sha(p)}
    write_json(dest / "sha256-manifest.json", root_manifest)


def main():
    all_summaries = []
    for env, title in [("tworoom", "TwoRoom"), ("reacher", "Reacher")]:
        summaries = [campaign(env, task) for task in [16, 17]]
        root_index(env, title, summaries)
        all_summaries += summaries
    assert sum(s["attempts"] for s in all_summaries) == 10180
    assert sum(s["complete"] for s in all_summaries) == 10176
    write_json(
        OUT / "packaging-validation.json",
        {
            "campaigns": all_summaries,
            "total_attempts": 10180,
            "valid_episodes": 10176,
            "episode_videos": sum(s["episode_videos"] for s in all_summaries),
            "comparison_videos": sum(s["comparison_videos"] for s in all_summaries),
            "raw_members_sha256_verified": sum(s["raw_files"] for s in all_summaries),
        },
    )
    print(json.dumps(all_summaries), flush=True)


if __name__ == "__main__":
    main()
