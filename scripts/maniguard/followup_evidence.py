"""Audit and export recorded followup evidence; never advances physics or queries a VLA."""

import argparse
import csv
import json
import subprocess
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.maniguard_followup_budget import atomic, clock
from helpers.maniguard_part2_protocol import RUN
from helpers.maniguard_part2_results import COSTS, audit
from helpers.maniguard_phase3_protocol import sha

FIGURES = ROOT / "docs/research/figures/ManiGuard/phase3-followup"
COLORS = ["#287d66", "#bf5b34", "#b5bdc4", "#7e3654"]
CATEGORIES = ["Safe completion", "Unsafe completion", "Safe unfinished", "Unsafe unfinished"]


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def verdict(record):
    if not record["valid"]:
        return "INVALID / INCOMPLETE"
    return ("completed" if record["C_req"] else "unfinished") + ("; VIOLATION" if record["V"] else "; no violation")


def audit_all(directory):
    records = []
    for process_path in sorted(RUN.rglob("process.json")):
        process = json.loads(process_path.read_text())
        if process.get("role") != "episode":
            continue
        output = process_path.parent / "data"
        record = audit(output)
        record["process"] = process
        if process["status"] != "completed" or process.get("returncode") != 0:
            record.update(valid=False, engineering_valid=False, U=None)
        record["part"] = json.loads((output / "job.json").read_text()).get("part") if (output / "job.json").exists() else None
        record["raw_observations"] = len(list((output / "observations").glob("*.npz")))
        record["native_videos"] = [{"path": str(p.relative_to(RUN)), "sha256": sha(p),
                                    "bytes": p.stat().st_size} for p in sorted(output.rglob("*.mp4"))]
        if record["valid"]:
            job = json.loads((output / "job.json").read_text())
            native = rows(output / "episode/results.jsonl")[0]
            expected_seed = zlib.crc32(f"{job['seed']}:{native['scene_name']}".encode()) & 0x7fffffff
            if native["episode_seed"] != expected_seed:
                raise ValueError("Native policy sampling seed differs from CRC32 provenance")
            trace = rows(output / "trace.jsonl")
            queries = [row["step"] for row in trace if row["type"] == "query"]
            intervals = np.diff(queries)
            if len(intervals) and (intervals.min() < 4 or intervals.max() > 8):
                raise ValueError("A followup query escaped the four-to-eight-action bounds")
            # Independently require a query as soon as an allowed trigger is reached.
            since = accumulated = 0
            threshold = job["controller"]["correction_threshold"]
            minimum = job["controller"]["minimum_interval"]
            due = False
            for row in trace:
                if row["type"] == "query":
                    if row["step"] and not due:
                        raise ValueError("New query preceded its causal scheduling trigger")
                    since, accumulated, due = 0, 0.0, False
                elif row["type"] == "proposal" and due:
                    raise ValueError("A stale proposal executed after its refresh trigger")
                elif row["type"] == "correction":
                    since += 1
                    accumulated += row["normalized"]
                    due = since >= 8 or (threshold is not None and since >= minimum and accumulated >= threshold)
            if job["controller"]["free_step_limit"] is None and job["controller"]["grasp_step_limit"] is None:
                for row in trace:
                    if row["type"] == "proposal":
                        dtype = row["event"]["proposal_dtype"]
                        if np.asarray(row["original"], dtype=dtype).tobytes() != np.asarray(row["adapted"], dtype=dtype).tobytes():
                            raise ValueError("Identity output differs at the bit level")
                record["identity_bitwise_verified"] = True
            record["sampling_seed"] = expected_seed
            record["query_intervals"] = intervals.tolist()
            record["replan_reasons"] = {reason: sum(row["type"] == "replan" and row["reason"] == reason for row in trace)
                                        for reason in ("maximum_interval", "fixed_minimum", "correction_threshold")}
            record["correction_total"] = sum(row["normalized"] for row in trace if row["type"] == "correction")
            record["native_violation_step"] = json.loads(next((output / "episode").rglob("*_ltl.json")).read_text())["violation_step"]
            if record["raw_observations"] != record["steps"] + 1:
                raise ValueError("Passive raw observation capture is incomplete")
        records.append(record)
    total = {}
    for key in COSTS:
        known = [r["costs"].get(key) for r in records if r["costs"].get(key) is not None]
        total[key] = {"known_total": sum(known), "unknown_attempts": len(records) - len(known)}
    result = {"schema": "maniguard-followup-audit-v1", "at": time.time(), "clock": clock(),
              "records": records, "attempts": len(records), "costs": total,
              "native_video_files": sum(len(r["native_videos"]) for r in records),
              "pre_subscription_physics": "unknown for every app launch, never imputed as zero",
              "script_sha256": sha(__file__)}
    directory.mkdir(parents=True, exist_ok=True)
    atomic(directory / "audit.json", result)
    with (directory / "interactions.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["episode_id", "part", "purpose", "condition", "scene", "seed", "valid", "C_req", "S_req", "V", "steps", *COSTS])
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in writer.fieldnames if key not in COSTS} | record["costs"])
    return result


def outcome_counts(records):
    counts = [sum(bool(r["C_req"]) == completion and bool(r["V"]) == violation for r in records)
              for completion, violation in ((True, False), (True, True), (False, False), (False, True))]
    assert sum(counts) == len(records)
    return counts


def savefig(fig, stem, provenance):
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        target = FIGURES / (stem + "." + suffix)
        fig.savefig(target, dpi=160, bbox_inches="tight")
        provenance.append({"path": str(target.relative_to(ROOT)), "sha256": sha(target)})
    plt.close(fig)


def plots(result):
    FIGURES.mkdir(parents=True, exist_ok=True)
    atomic(FIGURES / "figure-data.json", result)
    atomic(RUN / "evidence/figure-input-audit.json", result)
    full = [r for r in result["records"] if r.get("full_episode") and r["valid"]]
    provenance = []
    part2 = [r for r in full if r["part"] == 2]
    if len(part2) == 16:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
        scenes = sorted({r["scene"] for r in part2})
        for axis, scene in zip(axes, [*scenes, None], strict=True):
            selected = part2 if scene is None else [r for r in part2 if r["scene"] == scene]
            values = np.asarray([outcome_counts([r for r in selected if r["condition"] == c]) for c in "ABCD"])
            bottom = np.zeros(4)
            for index, label in enumerate(CATEGORIES):
                axis.bar(list("ABCD"), values[:, index], bottom=bottom, color=COLORS[index], label=label)
                bottom += values[:, index]
            axis.set_title("Pooled (n=4/condition)" if scene is None else f"{scene} (n=2/condition)")
            axis.set_ylim(0, 4.5)
            axis.set_yticks(range(5))
            axis.set_xlabel("A identity/8  B cap/8\nC cap/4  D correction/4..8")
        axes[0].set_ylabel("Episodes")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=9, loc="upper center", bbox_to_anchor=(.5, 1.08), ncol=4)
        savefig(fig, "part2-outcomes", provenance)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
        for axis, key, label in zip(axes, ("policy_control_steps", "policy_queries", "elapsed_seconds"),
                                     ("Executed controls", "VLA queries", "In-process time (s)"), strict=True):
            values = [sum(r["costs"][key] for r in part2 if r["condition"] == c) for c in "ABCD"]
            axis.bar(list("ABCD"), values, color=["#336b87", "#7ca9bd", "#bf8a44", "#77567a"])
            axis.set_ylabel(label)
            for i, value in enumerate(values):
                axis.text(i, value, f"{value:.0f}", ha="center", va="bottom", fontsize=9)
            axis.margins(y=.15)
        savefig(fig, "part2-costs", provenance)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
        for c in "ABCD":
            selected = [r for r in part2 if r["condition"] == c]
            intervals = [v for r in selected for v in r["query_intervals"]]
            counts = [intervals.count(i) for i in range(4, 9)]
            axes[0].plot(range(4, 9), counts, marker="o", label=c)
            axes[1].scatter([sum(r["costs"]["policy_queries"] for r in selected)],
                            [sum(r["S_req"] for r in selected)], s=60, label=c)
            offsets = {"A": (16, 22), "B": (-20, -18), "C": (-15, 8), "D": (16, -4)}
            axes[1].annotate(c, (sum(r["costs"]["policy_queries"] for r in selected),
                                 sum(r["S_req"] for r in selected)), xytext=offsets[c],
                            textcoords="offset points", arrowprops={"arrowstyle": "-", "lw": .6})
        axes[0].set(xlabel="Executed actions between actual queries", ylabel="Inter-query intervals", xticks=range(4, 9))
        axes[0].legend()
        axes[1].set(xlabel="Total VLA queries (four episodes)", ylabel="Safe completions / 4", yticks=range(5))
        axes[1].set_ylim(-.25, 4.5)
        axes[1].margins(x=.15)
        savefig(fig, "part2-feedback", provenance)
    recordings = [r for r in full if r["episode_id"].startswith("part3-record-")]
    if len(recordings) == 8:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), sharey=True)
        for axis, scene in zip(axes, sorted({r["scene"] for r in recordings}), strict=True):
            selected = [r for r in recordings if r["scene"] == scene]
            values = np.asarray([outcome_counts([r for r in selected if r["condition"] == c]) for c in "BC"])
            bottom = np.zeros(2)
            for index, label in enumerate(CATEGORIES):
                axis.bar(list("BC"), values[:, index], bottom=bottom, color=COLORS[index], label=label)
                bottom += values[:, index]
            axis.set(title=scene + " (n=2/condition)", ylim=(0, 2.5), yticks=range(3),
                     xlabel="B cap/8; C cap/4")
        axes[0].set_ylabel("Common recording episodes")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.08), ncol=2, fontsize=8)
        savefig(fig, "part3-recording-outcomes", provenance)
    model = ROOT / "data/maniguard/models/phase3-parts2-3-20261018"
    if (model / "forecast-audit.json").exists():
        forecast = json.loads((model / "forecast-audit.json").read_text())
        m = forecast["metrics"]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        keys = ["persistence_4", "ridge_4", "action_shuffled_ensemble_4", "ensemble_4"]
        axes[0].bar(["Persist", "Ridge", "Shuf MLP", "MLP x5"], [m[k + "_changing_rmse"] for k in keys], color="#5d8398")
        title = f"Changing windows (n={m['changing_windows']})" if m["changing_windows"] else f"All windows; no changing support (n={m['audit_windows']})"
        axes[0].set(ylabel="Mean 4-action latent RMSE", title=title)
        # These three comparisons use every retained actual audit observation.
        # The separate visual-only metric uses query-start future windows instead.
        names = ["pca", "raw", "random"]
        axes[1].bar(range(len(names)), [m["readout_comparison"][n]["margin_mae_deg"] for n in names], color="#a06c69")
        axes[1].set(xticks=range(len(names)), xticklabels=["PCA+q", "Raw+q", "RP+q"], ylabel="Actual-state margin MAE (deg)", title="All retained audit observations")
        axes[2].bar(["M", "C"], [m["M_acceptance"], m["C_acceptance"]], color=["#9971a0", "#c58c58"])
        axes[2].set(ylim=(0, 1), ylabel="Accepted audit prefixes", title="Nominal calibration acceptance 0.5")
        savefig(fig, "part3-forecast", provenance)
        with np.load(model / "forecast-windows.npz") as d:
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
            axes[0].scatter(d["actual_target"][:, 0], d["forecast_target"][:, 0], s=6, alpha=.35)
            limits = [min(d["actual_target"][:, 0].min(), d["forecast_target"][:, 0].min()),
                      max(d["actual_target"][:, 0].max(), d["forecast_target"][:, 0].max())]
            axes[0].plot(limits, limits, color="black", lw=1)
            axes[0].axvline(0, color="red", ls="--", lw=1)
            axes[0].axhline(0, color="red", ls="--", lw=1)
            axes[0].set(xlabel="Measured min upright margin (deg)", ylabel="Four-action predicted margin (deg)")
            axes[1].scatter(d["correction"], d["error4"], s=7, alpha=.35)
            axes[1].set(xlabel="Factual normalized correction in prefix", ylabel="Four-action latent RMSE")
            savefig(fig, "part3-correction-error", provenance)
    atomic(FIGURES / "figure-provenance.json", {"at": time.time(), "source_audit_sha256": sha(FIGURES / "figure-data.json"),
                                                "source_snapshot": "figure-data.json",
                                                "model_audit_sha256": sha(model / "forecast-audit.json") if (model / "forecast-audit.json").exists() else None,
                                                "model_windows_sha256": sha(model / "forecast-windows.npz") if (model / "forecast-windows.npz").exists() else None,
                                                "files": provenance, "new_simulation_steps": 0, "VLA_queries": 0,
                                                "script_sha256": sha(__file__)})


def probe(path):
    value = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-show_entries", "stream=nb_frames,r_frame_rate", "-of", "json", str(path)], text=True)
    return json.loads(value)


def matched(records, stage):
    blocks = defaultdict(list)
    for r in records:
        if r["valid"] and r["full_episode"]:
            job = json.loads((Path(r["output"]) / "job.json").read_text())
            if stage == "part2" and job["part"] == 2 or stage == "part3-recording" and job.get("stage") == "common_recording" or stage == "part3-comparison" and job.get("stage") == "comparison":
                blocks[job["block"]].append((job, r))
    folder = RUN / "matched-videos" / stage
    folder.mkdir(parents=True, exist_ok=True)
    index = []
    for block, entries in sorted(blocks.items()):
        expected = 4 if stage == "part2" else 5 if stage == "part3-comparison" else 2
        if len(entries) != expected:
            continue
        condition_order = ["identity", "fixed", "E", "M", "C"] if stage == "part3-comparison" else list("ABCD")
        entries.sort(key=lambda item: condition_order.index(item[0]["condition"]))
        for camera, suffix in (("overview", "main"), ("wrist", "wrist")):
            destination = folder / f"block{block}-{camera}.mp4"
            inputs, sources, filters = [], [], []
            durations = []
            for job, record in entries:
                path = Path(record["output"]) / "episode" / (job["scene"] + f"_{suffix}.mp4")
                metadata = probe(path)
                duration = float(metadata["format"]["duration"])
                if int(metadata["streams"][0]["nb_frames"]) != record["steps"] + 1:
                    raise ValueError("Native video/observation alignment differs")
                inputs += ["-i", str(path)]
                durations.append(duration)
                sources.append({"episode_id": job["episode_id"], "camera": camera,
                                "path": str(path.relative_to(RUN)), "sha256": sha(path),
                                "duration": duration, "steps": record["steps"], "verdict": verdict(record)})
            maximum = max(durations)
            columns = 2 if expected <= 4 else 3
            for i, ((job, record), duration) in enumerate(zip(entries, durations, strict=True)):
                label = f"{job['condition']} | {verdict(record)}"
                case_label = f"{job['scene']} seed {job['seed']}"
                # Conditions and identifiers are generated internally and path-free.
                label = label.replace(";", " ")
                filters.append(f"[{i}:v]fps=20,scale=320:320,pad=320:360:0:40:color=black,drawtext=text='{label}':x=5:y=4:fontsize=12:fontcolor=white,drawtext=text='{case_label}':x=5:y=22:fontsize=11:fontcolor=white,tpad=stop_mode=add:stop_duration={maximum-duration+0.1}:color=black,drawtext=text='ENDED':fontcolor=white:fontsize=30:x=(w-tw)/2:y=(h-th)/2:enable='gte(t,{duration})'[v{i}]")
            layout = "|".join(f"{i % columns * 320}_{i // columns * 360}" for i in range(expected))
            filters.append("".join(f"[v{i}]" for i in range(expected)) + f"xstack=inputs={expected}:layout={layout}:fill=black,format=yuv420p[out]")
            command = ["ffmpeg", "-nostdin", "-v", "error", *inputs, "-filter_complex", ";".join(filters),
                       "-map", "[out]", "-an", "-t", str(maximum), "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-threads", "4", "-y", str(destination)]
            if not destination.exists():
                subprocess.run(command, check=True, timeout=180)
            index.append({"block": block, "camera": camera, "path": str(destination.relative_to(RUN)),
                          "sha256": sha(destination), "bytes": destination.stat().st_size,
                          "sources": sources, "ended_rule": "black panel labelled ENDED after native clip duration",
                          "composition": command})
    atomic(folder / "index.json", {"stage": stage, "videos": index, "new_physics": 0})
    return index


def pictures(result):
    output = FIGURES / "frames"
    output.mkdir(parents=True, exist_ok=True)
    index = []
    for record in result["records"]:
        if not record["valid"] or not record["full_episode"]:
            continue
        data = Path(record["output"])
        n = record["steps"]
        event = record["native_violation_step"]
        if event is not None:
            steps = [max(0, event - 20), event, min(n, event + 20)]
            reason = "every violation: one second before, at first violation, one second after, clipped to bounds"
        else:
            steps = [0, n // 2, n]
            reason = "every nonviolating episode: start, middle, final recorded observation"
        for camera in ("overview", "wrist"):
            strip = Image.new("RGB", (3 * 256, 300), "black")
            draw = ImageDraw.Draw(strip)
            sources = []
            for i, step in enumerate(steps):
                path = data / "observations" / f"{step:04d}.npz"
                with np.load(path) as raw:
                    img = Image.fromarray(raw[camera])
                strip.paste(img, (i * 256, 44))
                draw.text((i * 256 + 4, 22), f"step {step}  t={step/20:.2f}s", fill="white")
                sources.append({"step": step, "file_sha256": sha(path)})
            draw.text((4, 3), f"{record['episode_id']} {camera} | {verdict(record)}", fill="white")
            dest = output / f"{record['episode_id']}-{camera}.png"
            strip.save(dest)
            index.append({"path": str(dest.relative_to(ROOT)), "sha256": sha(dest),
                          "episode_id": record["episode_id"], "camera": camera,
                          "selection": reason, "first_violation_step": event, "sources": sources})
    atomic(output / "index.json", {"selection": "complete episode gallery, no outcome cherry-picking", "pictures": index})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "plots", "videos", "pictures", "all"))
    args = parser.parse_args()
    directory = RUN / "evidence"
    result = audit_all(directory) if args.stage in ("audit", "all") else json.loads((directory / "audit.json").read_text())
    if args.stage in ("plots", "all"):
        plots(result)
    if args.stage in ("videos", "all"):
        for stage in ("part2", "part3-recording", "part3-comparison"):
            matched(result["records"], stage)
    if args.stage in ("pictures", "all"):
        pictures(result)
    print(json.dumps({"stage": args.stage, "attempts": result["attempts"], "clock": clock()}))


if __name__ == "__main__":
    main()
