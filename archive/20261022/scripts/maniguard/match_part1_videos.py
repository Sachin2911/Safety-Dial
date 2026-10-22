"""Create every available matched three-dial video block without replaying physics.

Native frames are retained in order. Shorter recordings get black frames labelled
"episode ended" up to the longest recording in the matched block. A matched seed
does not imply identical rendering or trajectories.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def duration(path):
    result = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        text=True,
    )
    return float(json.loads(result)["format"]["duration"])


def compose(run, index, output, purpose="validation"):
    run, output = Path(run).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    videos = json.loads(Path(index).read_text())
    groups = sorted({(v["scene"], v["seed"]) for v in videos if v["purpose"] == purpose})
    results = []
    for scene, seed in groups:
        for camera in ("main", "wrist"):
            rows = [
                v
                for v in videos
                if v["purpose"] == purpose
                and v["scene"] == scene
                and v["seed"] == seed
                and v["path"].endswith(f"_{camera}.mp4")
            ]
            selected = []
            for dial in (0, 0.5, 1):
                matches = [v for v in rows if v["dial"] == dial]
                if len(matches) != 1:
                    break
                selected.append(matches[0])
            if len(selected) != 3:
                results.append(
                    {
                        "scene": scene,
                        "seed": seed,
                        "camera": camera,
                        "status": "incomplete_or_ambiguous_native_block",
                    }
                )
                continue
            paths = [(run / v["path"]).resolve() for v in selected]
            for path, row in zip(paths, selected, strict=True):
                path.relative_to(run)
                if digest(path) != row["sha256"]:
                    raise ValueError("Native video hash changed")
            durations = [duration(path) for path in paths]
            end = max(durations)
            command = ["ffmpeg", "-nostdin", "-v", "error", "-n"]
            for path in paths:
                command += ["-i", str(path)]
            filters = []
            font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            for i, (dial, seconds) in enumerate(zip((0, 0.5, 1), durations, strict=True)):
                filters.append(
                    f"[{i}:v]setpts=PTS-STARTPTS,tpad=stop_mode=add:stop_duration={end - seconds}:"
                    f"color=black,drawtext=fontfile={font}:text='d={dial}':x=8:y=8:"
                    "fontsize=18:fontcolor=white:box=1:boxcolor=black@0.65,"
                    f"drawtext=fontfile={font}:text='episode ended':x=8:y=40:fontsize=16:"
                    f"fontcolor=white:enable='gte(t,{seconds})'[v{i}]"
                )
            filters.append("[v0][v1][v2]hstack=inputs=3:shortest=1[out]")
            target = output / f"{scene.replace('/', '-')}-seed{seed}-{camera}.mp4"
            command += [
                "-filter_complex_threads",
                "2",
                "-filter_complex",
                ";".join(filters),
                "-map",
                "[out]",
                "-t",
                str(end),
                "-an",
                "-c:v",
                "libx264",
                "-threads",
                "2",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                str(target),
            ]
            subprocess.run(command, check=True, timeout=300)
            results.append(
                {
                    "scene": scene,
                    "seed": seed,
                    "camera": camera,
                    "purpose": purpose,
                    "status": "composed",
                    "path": target.name,
                    "sha256": digest(target),
                    "duration_seconds": end,
                    "native_durations": durations,
                    "source_sha256": [v["sha256"] for v in selected],
                    "physics_replayed": False,
                }
            )
    (output / "index.json").write_text(json.dumps(results, indent=2) + "\n")
    print(
        json.dumps(
            {
                "composed": sum(r["status"] == "composed" for r in results),
                "index": str(output / "index.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("index", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--purpose", choices=("engineering", "validation"), default="validation")
    args = parser.parse_args()
    compose(args.run, args.index, args.output, args.purpose)
