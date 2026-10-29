"""Build the task 19 picture strip from one matched confirmation pair's recorded video.

Task 3, seed 193016: the native attempt tips the jar before closing it; the shielded
attempt retreats four times in its first eight seconds as the policy pushes again, then
closes and carries the jar and finishes safely. Frames come from the
third-person camera (base_main.mp4, 20 frames per second, frame k + 1 shows the scene
after control k) and are cropped around the jar. Tilt labels are computed from the jar
quaternion in each attempt's recording. Writes assets/evo19-frames.pdf.

This pair was chosen because the retreat is visible; it illustrates the mechanism and is
not a representative sample (fig:jar and tab:jar-detail in the thesis give the full
results).
"""

import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "experiments/scripts"))
from revalidate_evo19 import counted_episode, finished_runs  # noqa: E402

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8})
RED, TEAL, DARK = "#b33b39", "#137c82", "#263d50"
CROP = (64, 40, 192, 168)  # x0, y0, x1, y1 on the 256-pixel main camera
STEPS = [76, 88, 96, 112]
LAST = {"native": 400, "evolved": 1020}


def tilt_deg(quat):
    quat = np.asarray(quat, dtype=np.float64)
    return np.degrees(np.arccos(np.clip(1 - 2 * (quat[:, 0] ** 2 + quat[:, 1] ** 2), -1, 1)))


def load(job):
    episode = counted_episode(job, finished_runs())
    with np.load(next(episode.glob("recordings/*_base.npz"))) as archive:
        tilt = tilt_deg(archive["jar_quat"])
    outcome = json.loads((episode / "validation.json").read_text())["outcome"]
    retreats = [e["step"] for e in map(json.loads, (episode / "adapter.jsonl").read_text().splitlines())
                if e.get("event") == "decision" and e.get("action") == "rewound"]
    video = next(episode.glob("eval/task_*/base_main.mp4"))
    return tilt, outcome, retreats, video


def frames(video, steps):
    wanted = {s + 1: s for s in steps}
    out = {}
    reader = imageio.get_reader(video)
    assert reader.get_meta_data()["fps"] == 20
    for i, frame in enumerate(reader):
        if i in wanted:
            out[wanted[i]] = frame[CROP[1]:CROP[3], CROP[0]:CROP[2]]
        if i > max(wanted):
            break
    reader.close()
    return out


native = load("b-native-t3-s193016")
shield = load("b-evolved-t3-s193016")
assert native[1]["violation_step"] == 106 and not native[1]["strict_safe"]
assert native[1]["first_closed_step"] is None  # the tip happens before closure
assert shield[1]["strict_safe"] and shield[1]["violation_step"] is None
assert shield[2][:4] == [88, 112, 136, 160]  # every retreat
assert shield[1]["steps"] - 1 >= LAST["evolved"]

fig, axes = plt.subplots(2, 5, figsize=(7.0, 3.05))
rows = [("native", "Native $\\pi_{0.5}$", native, RED), ("evolved", "With tip shield", shield, TEAL)]
for r, (arm, label, (tilt, outcome, retreats, video), color) in enumerate(rows):
    steps = STEPS + [LAST[arm]]
    images = frames(video, steps)
    for c, step in enumerate(steps):
        ax = axes[r, c]
        ax.imshow(images[step], interpolation="lanczos")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("0.75")
        note = f"tilt {tilt[step]:.0f}°"
        if arm == "native" and step >= outcome["violation_step"]:
            note += ", tipped"
        if arm == "evolved" and step in retreats:
            note += ", retreat" if step == retreats[0] else ", retreat again"
        if step == LAST[arm]:
            note = f"{step / 20:.0f} s: " + ("still tipped" if arm == "native"
                                               else "carried, finished")
        ax.text(0.03, 0.04, note, transform=ax.transAxes, fontsize=7, color="white",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": color if (
                    "tipped" in note or "retreat" in note or "finished" in note) else DARK,
                    "edgecolor": "none", "alpha": 0.9})
        if r == 0:
            ax.set_title(f"{step / 20:.1f} s (control {step})" if c < 4 else "Later",
                         fontsize=8)
    axes[r, 0].set_ylabel(label, fontsize=8, color=color)
fig.tight_layout(w_pad=0.4, h_pad=0.9)
fig.savefig(HERE / "assets/evo19-frames.pdf", dpi=300)
plt.close(fig)
print("Task 19 frame strip generated from the matched task 3, seed 193016 pair.")
