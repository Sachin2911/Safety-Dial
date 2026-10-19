"""Derive public plots, counts and private representative videos from saved evidence."""

import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from helpers.maniguard_part4_protocol import CONFIG, ROOT, RUN, atomic, digest, read

FIGURES = ROOT / "docs/research/figures/ManiGuard/phase3-part4"
NOTES = ROOT / "docs/research/research_notes"


def finish(figure, name):
    figure.savefig(FIGURES / (name + ".pdf"), bbox_inches="tight")
    figure.savefig(FIGURES / (name + ".png"), bbox_inches="tight", dpi=170)
    plt.close(figure)


def figures():
    FIGURES.mkdir(parents=True, exist_ok=True)
    audit = read(RUN / "offline-audit.json")
    donor = read(RUN / "donor-audit.json")
    metrics = audit["metrics"]
    names, labels = ["primary", "logistic", "visual"], ["MLP ensemble", "Full logistic", "Visual logistic"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2), constrained_layout=True)
    ap = [metrics["development"][n]["auprc"] for n in names]
    bars = ax[0].bar(labels, ap, color=["#247ba0", "#888888", "#70ad47"])
    ax[0].axhline(.05, color="#b13e39", linestyle="--", label="Frozen primary minimum .05")
    ax[0].axhline(metrics["development"]["primary"]["prevalence"], color="black",
                  linestyle=":", label="Positive-window prevalence")
    ax[0].bar_label(bars, fmt="%.3f", padding=3)
    ax[0].set(ylabel="Development average precision", ylim=(0, .5))
    ax[0].legend(fontsize=8, loc="upper left")
    x = np.arange(3)
    for offset, group, color in [(-.18, "calibration", "#70ad47"), (.18, "development", "#247ba0")]:
        values = [100 * metrics[group][n]["negative_window_false_alarm"] for n in names]
        bars = ax[1].bar(x + offset, values, .35, label=group.title(), color=color)
        ax[1].bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax[1].set(xticks=x, xticklabels=labels, ylabel="Negative windows with alarm (%)", ylim=(0, 105))
    ax[1].legend(fontsize=8)
    finish(fig, "warning-tradeoff")

    groups = ["fit", "calibration", "development"]
    fig, ax = plt.subplots(figsize=(6.5, 3.1), constrained_layout=True)
    x = np.arange(3)
    for offset, key, title, color in [(-.24, "episode_count", "All episodes", "#999999"),
                                     (0, "eligible_events", "First-event episodes", "#247ba0"),
                                     (.24, "event_tasks", "Tasks with events", "#d17b28")]:
        values = [metrics[g]["primary"][key] for g in groups]
        bars = ax.bar(x + offset, values, .23, label=title, color=color)
        ax.bar_label(bars, padding=2)
    ax.set(xticks=x, xticklabels=["Fitting", "Calibration", "Development"], ylabel="Count", ylim=(0, 19))
    ax.legend(fontsize=8)
    finish(fig, "event-support")

    with np.load(RUN / "offline-predictions.npz", allow_pickle=False) as value:
        predictions = {k: value[k].copy() for k in value.files}
    ids = [e["episode_id"] for e in donor["episodes"]]
    cases = ["part3-record-b0-C", "part3-record-b3-B"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.2), constrained_layout=True)
    for row, case in enumerate(cases):
        i = ids.index(case)
        e = donor["episodes"][i]
        steps, tau = predictions[f"steps_{i}"], e["first_upright_event"]
        select = steps >= tau - 100
        seconds = (steps[select] - tau) / 20
        axes[row, 0].plot(seconds, predictions[f"probability_{i}"][select], color="#247ba0")
        axes[row, 0].axhline(audit["rules"]["primary"]["threshold"], color="#b13e39", linestyle="--")
        axes[row, 0].set(ylabel="Calibrated probability", title=case + ": " + e["group"])
        axes[row, 1].plot(seconds, predictions[f"uncertainty_{i}"][select], color="#d17b28")
        axes[row, 1].axhline(audit["rules"]["primary"]["disagreement_cutoff"], color="#b13e39", linestyle="--")
        axes[row, 1].set(ylabel="Member probability SD", title="Disagreement and frozen cutoff")
        for axis in axes[row]:
            axis.axvspan(-.4, -.2, color="#70ad47", alpha=.2)
            axis.set(xlabel="Seconds before first physical event", xlim=(-5, 0))
    finish(fig, "warning-timelines")

    training = read(RUN / "models/baseline/training-ledger.json")
    fig, ax = plt.subplots(figsize=(6.5, 3.1), constrained_layout=True)
    for member in training["members"]:
        curve = member["loss_curve"]
        ax.plot([r["update"] for r in curve], [r["minibatch_loss"] for r in curve],
                label=str(member["seed"]))
    ax.set(xlabel="Optimizer update", ylabel="Balanced minibatch cross-entropy")
    ax.legend(fontsize=7, ncol=2, title="Bootstrap member seed", title_fontsize=8)
    finish(fig, "training-curves")

    physical = {}
    for group in groups:
        rows = [e for e in donor["episodes"] if e["group"] == group]
        audits = []
        native_types = {k: 0 for k in ("uprightness", "spill", "drop")}
        native_propositions = {}
        for e in rows:
            paths = list((RUN / "donors").glob("**/" + e["episode_id"] + "-a1/audit.json"))
            if len(paths) != 1:
                raise ValueError("Missing unique historical physical audit")
            audits.append(read(paths[0]))
            aps = [r["monitor"]["observation"]["ap"] for r in (
                json.loads(line) for line in (paths[0].parent / "data/trace.jsonl").read_text().splitlines()
            ) if r["type"] == "observed"]
            diagnostic = RUN / "benchmark-metadata" / (e["task"] + "-diagnostics.jsonl")
            props = json.loads(diagnostic.read_text().splitlines()[0])["ltl_safety"]["propositions"]
            episode_types = set()
            for name, prop in props.items():
                if prop.get("state") == "upright":
                    kind, violation_value = "uprightness", False
                elif prop.get("state") == "dropped":
                    kind, violation_value = "drop", True
                elif prop.get("check") == "spill":
                    kind, violation_value = "spill", True
                else:
                    raise ValueError("Unclassified native physical proposition: " + name)
                violated = any(ap[name] is violation_value for ap in aps)
                native_propositions[name] = native_propositions.get(name, 0) + int(violated)
                if violated:
                    episode_types.add(kind)
            if bool(episode_types) != audits[-1]["V"]:
                raise ValueError("Physical type report disagrees with audited native violation")
            for kind in episode_types:
                native_types[kind] += 1
        physical[group] = {
            "episodes": len(audits),
            "safe_complete": sum(a["C_req"] and not a["V"] for a in audits),
            "unsafe_complete": sum(a["C_req"] and a["V"] for a in audits),
            "safe_unfinished": sum(not a["C_req"] and not a["V"] for a in audits),
            "unsafe_unfinished": sum(not a["C_req"] and a["V"] for a in audits),
            "controls": sum(a["steps"] for a in audits),
            "VLA_queries": sum(a["query_count"] for a in audits),
            "post_subscription_physics_ticks": sum(a["costs"]["physics_steps"] for a in audits),
            "pre_subscription_physics_ticks": None,
            "episode_process_seconds": sum(a["process"]["elapsed_seconds"] for a in audits),
            "native_violation_episode_counts": native_types,
            "native_proposition_episode_counts": native_propositions,
        }
    fig, ax = plt.subplots(figsize=(6.5, 3.1), constrained_layout=True)
    bottom = np.zeros(3)
    for key, title, color in [("safe_complete", "Safe complete", "#70ad47"),
                              ("unsafe_complete", "Unsafe complete", "#d17b28"),
                              ("safe_unfinished", "Safe unfinished", "#999999"),
                              ("unsafe_unfinished", "Unsafe unfinished", "#b13e39")]:
        values = np.array([physical[g][key] for g in groups])
        ax.bar(groups, values, bottom=bottom, color=color, label=title)
        bottom += values
    ax.set(ylabel="Historical episodes (reused once)", ylim=(0, 18))
    ax.legend(fontsize=8, ncol=2)
    finish(fig, "historical-outcomes")
    provenance = {"schema": "maniguard-part4-figures-v1", "offline_audit_sha256": digest(RUN / "offline-audit.json"),
                  "donor_audit_sha256": digest(RUN / "donor-audit.json"),
                  "script_sha256": digest(Path(__file__)), "new_physics_steps": 0,
                  "historical_physical_outcomes": physical,
                  "native_numeric_proposition_checks": sum(e["native_numeric_checks"] for e in donor["episodes"]),
                  "limits": "24 historical episodes, four tasks, one development first-event episode; correlated windows are not independent trials"}
    provenance["files"] = {p.name: digest(p) for p in sorted(FIGURES.iterdir()) if p.suffix in (".pdf", ".png")}
    atomic(FIGURES / "provenance.json", provenance)
    atomic(NOTES / "ManiGuard phase3 Part4 offline audit.json", audit)
    atomic(NOTES / "ManiGuard phase3 Part4 donor audit.json", donor)
    atomic(NOTES / "ManiGuard phase3 Part4 terminal.json", read(RUN / "terminal.json"))
    atomic(NOTES / "ManiGuard phase3 Part4 accounting.json", {
        "schema": "maniguard-part4-accounting-v1", "new_physical_attempts": 0,
        "new_controls": 0, "new_physics_steps": 0, "new_VLA_queries": 0,
        "new_encoder_calls": 0, "historical_donors": physical,
        "historical_acquisition_charged_once": True,
        "cumulative_prior_known_controls": 53911, "cumulative_known_controls_after_part4": 53911,
        "training": training, "model_window_evaluations": audit["model_window_evaluations"],
        "offline_attempts": [read(RUN / "offline-engineering-attempt.json")],
        "allocation": read(CONFIG / "allocation.json"),
        "clock_at_scientific_terminal": read(RUN / "terminal.json")["clock"],
        "final_provider_state": "recorded separately after final preservation; a running/stop-intended state is not verified shutdown",
    })
    return provenance


def evidence():
    from huggingface_hub import hf_hub_download

    donor = read(RUN / "donor-audit.json")
    restore = read(RUN / "donor-restore.json")
    archive_manifest = read(RUN / "followup-20261018T213427Z-raw-final.tar.json")
    members = {r["path"]: r for r in archive_manifest["files"]}
    path = hf_hub_download(restore["repo_id"], "followup/" + archive_manifest["archive"],
                           repo_type="dataset", revision=restore["revision"])
    if digest(path) != archive_manifest["sha256"]:
        raise ValueError("Historical RGB archive changed")
    needed, cases = {}, ["part3-record-b0-C", "part3-record-b3-B"]
    for e in donor["episodes"]:
        tau = e["first_upright_event"]
        if tau is None:
            continue
        identifier = e["episode_id"]
        prefix = "part2" if identifier.startswith("part2-") else "part3/recording"
        indices = range(tau - 80, tau + 21) if identifier in cases else [tau-8, tau-4, tau, tau+8]
        for step in indices:
            member = f"{prefix}/attempts/{identifier}-a1/data/observations/{step:04d}.npz"
            if member not in members:
                raise ValueError("Required exact RGB frame missing")
            needed[member] = (identifier, step)
    import hashlib

    extracted = {}
    with tarfile.open(path, "r:gz") as tf:
        for member in tf:
            if member.name not in needed:
                continue
            raw = tf.extractfile(member).read()
            if hashlib.sha256(raw).hexdigest() != members[member.name]["sha256"]:
                raise ValueError("Native saved RGB frame hash mismatch")
            with np.load(io.BytesIO(raw), allow_pickle=False) as value:
                extracted[needed[member.name]] = {k: value[k].copy() for k in ("overview", "wrist")}
    gallery = FIGURES / "frames"
    gallery.mkdir(exist_ok=True)
    strips = []
    for e in donor["episodes"]:
        tau = e["first_upright_event"]
        if tau is None:
            continue
        identifier = e["episode_id"]
        image = Image.new("RGB", (1024, 590), "white")
        draw = ImageDraw.Draw(image)
        draw.text((10, 5), identifier + " | historical " + e["group"] + " | " + ", ".join(e["event_objects"]), fill="black")
        for col, (step, label) in enumerate(zip([tau-8, tau-4, tau, tau+8],
                                               ["warning begins", "deadline", "first native event", "after event"], strict=True)):
            draw.text((col*256+5, 25), f"{label}: action {step}", fill="black")
            for row, camera in enumerate(("overview", "wrist")):
                image.paste(Image.fromarray(extracted[(identifier, step)][camera]), (col*256, 48+row*270))
        name = identifier + ".png"
        image.save(gallery / name)
        strips.append({"episode_id": identifier, "first_event": tau, "file": "frames/" + name,
                       "sha256": digest(gallery / name), "steps": [tau-8, tau-4, tau, tau+8]})
    videos = RUN / "evidence/videos"
    videos.mkdir(parents=True, exist_ok=True)
    video_records = []
    with np.load(RUN / "offline-predictions.npz", allow_pickle=False) as value:
        arrays = {k: value[k].copy() for k in value.files}
    ids = [e["episode_id"] for e in donor["episodes"]]
    threshold = read(RUN / "offline-audit.json")["rules"]["primary"]["threshold"]
    for identifier in cases:
        i = ids.index(identifier)
        e = donor["episodes"][i]
        tau = e["first_upright_event"]
        probability = dict(zip(arrays[f"steps_{i}"].tolist(), arrays[f"probability_{i}"].tolist(), strict=True))
        destination = videos / (identifier + ".mp4")
        command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                   "-s", "512x308", "-r", "20", "-i", "-", "-an", "-c:v", "libx264",
                   "-crf", "20", "-pix_fmt", "yuv420p", str(destination)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        for step in range(tau - 80, tau + 21):
            frame = Image.new("RGB", (512, 308), "white")
            draw = ImageDraw.Draw(frame)
            draw.text((5, 3), identifier + f" | action {step} | onset {tau}", fill="black")
            text = f"p={probability[step]:.4f}, alarm >= {threshold:.4f}" if step in probability else "outside first-event warning domain"
            draw.text((5, 20), text, fill="black")
            draw.text((5, 36), "Recorded historical RGB; passive scores; no new physics", fill="black")
            frame.paste(Image.fromarray(extracted[(identifier, step)]["overview"]), (0, 52))
            frame.paste(Image.fromarray(extracted[(identifier, step)]["wrist"]), (256, 52))
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        errors = process.stderr.read().decode()
        if process.wait() != 0:
            raise RuntimeError(errors)
        video_records.append({"episode_id": identifier, "path": str(destination.relative_to(RUN)),
                              "sha256": digest(destination), "frames": 101, "fps": 20,
                              "source": "Exact native observation RGB; no interpolation or physics replay"})
    provenance = {"archive_revision": restore["revision"], "archive_sha256": archive_manifest["sha256"],
                  "source_frames": [members[name] for name in sorted(needed)],
                  "strips": strips, "videos": video_records,
                  "new_physics_steps": 0, "new_encoder_calls": 0, "new_VLA_queries": 0}
    atomic(gallery / "provenance.json", provenance)
    atomic(RUN / "evidence/gallery.json", provenance)
    return provenance


if __name__ == "__main__":
    print(json.dumps({"plots": figures()["native_numeric_proposition_checks"]}))
    print(json.dumps({"evidence_videos": len(evidence()["videos"])}))
