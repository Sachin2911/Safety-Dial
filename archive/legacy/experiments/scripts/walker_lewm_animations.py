#!/usr/bin/env python3
"""Walker2d animations driven by the trained LeWM checkpoint.

LeWM is a JEPA: it predicts latents and has no pixel decoder. Imagination is therefore shown
two ways, both computed from the checkpoint itself:

1. Probe readouts: the frozen S3 probe reads torso height and pitch from the imagined latents.
   Forward speed is NOT drawn because S3 did not qualify the speed readout.
2. Latent retrieval: for each imagined latent, the nearest real frame (L2 in latent space) from
   a gallery of probe-split frames. The same retrieval applied to the encoded REAL frame is
   shown beside it, so gallery coverage can be told apart from imagination error.

Clips:
  <bank>_<category>   one saved branch per bank (test, stress) and decision category of the
                      health rule at the saved matched margin: safe_accepted, fall_missed,
                      fall_caught, false_alarm. Examples are chosen by a fixed rule from the
                      saved S4 no-update rows, never by eye (see `select_example`).
  long_horizon        a held-out test-role episode imagined 30 blocks (2.4 s) ahead with its
                      recorded actions, three times the 0.8 s horizon evaluated in S3/S4.

Every real frame is the MuJoCo renderer replaying a stored simulator state. Model inputs at
block endpoints are the frames STORED in the bank, exactly what S4 evaluated, and the script
requires the recomputed clearances to equal the saved S4 row. The displayed frames in between
are re-rendered from the stored states. Stored frames were rendered straight after a physics
step, when MuJoCo's kinematics lag the state by one 0.002 s substep, so a re-render differs
from a stored frame in a few pixels (well under 1% of pixels, more for fast motion; the
measured difference is recorded per clip). These are illustrations, not new statistics.

    uv run python experiments/scripts/walker_lewm_animations.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from helpers.threads import apply_torch, pin_threads  # noqa: E402

pin_threads()

import h5py  # noqa: E402
import hdf5plugin  # noqa: E402,F401
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

apply_torch()

from helpers.locoData import RenderContext, render_fingerprint  # noqa: E402
from helpers.poseProbes import load_probe  # noqa: E402
from helpers.walkerBank import WalkerBank, interp_steps, iter_episodes  # noqa: E402
from helpers.walkerLewm import WalkerImaginer, load_walker_model  # noqa: E402
from helpers.walkerRules import (  # noqa: E402
    DT,
    FRAMESKIP,
    HEALTHY_ANGLE,
    HEALTHY_Z,
    HISTORY,
    HORIZON_BLOCKS,
    health_clearance,
)

MODEL_RUN = REPO_ROOT / "runs" / "walker2d-lewm-a-recovery-20260927-1"
PROBES_RUN = REPO_ROOT / "runs" / "walker2d-probes-recovery-20260927-1"
S4_RUN = REPO_ROOT / "runs" / "walker2d-s4-recovery-20260927-1"
DATA_DIR = REPO_ROOT / "data" / "study" / "walker2d" / "continuation-20260926-2"
S4_DATA = DATA_DIR / "s4" / "walker2d-s4-recovery-20260927-1"
OUT_ROOT = REPO_ROOT / "docs" / "mainPlan" / "results" / "walker-animations"

CATEGORIES = {
    "safe_accepted": "Truly safe, and the model accepts",
    "fall_missed": "Truly unsafe, but the model accepts (false safe)",
    "fall_caught": "Truly unsafe, and the model rejects",
    "false_alarm": "Truly safe, but the model rejects",
}
SHORT = {"safe_accepted": "safe and accepted", "fall_missed": "fall missed (false safe)", "fall_caught": "fall caught", "false_alarm": "false alarm"}
FPS = 25  # one video frame per 0.008 s env step -> 0.2x real time
HOLD_START_S, HOLD_END_S = 1.0, 2.0
GIF_WIDTH, GIF_FPS = 800, 12.5
LONG_BLOCKS = 30
LONG_MIN_STEPS = 420
LONG_ROOT_STEP = 100
C_TRUE, C_REAL, C_IMAG = "#1a1a1a", "#1f77b4", "#d62728"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------------------
# selection from SAVED rows (pure functions)
# --------------------------------------------------------------------------------------
def categorise(row: dict, margin: float, rule: str = "health") -> str:
    unsafe = row[f"cmin_dense_{rule}"] < 0
    accept = row[f"cmin_imagined_{rule}"] >= margin
    if unsafe:
        return "fall_missed" if accept else "fall_caught"
    return "safe_accepted" if accept else "false_alarm"


def first_block_below(cumulative_min: list, threshold: float) -> int | None:
    """1-based block at which the running minimum clearance first drops below the threshold."""
    below = np.asarray(cumulative_min) < threshold
    return int(np.argmax(below)) + 1 if below.any() else None


def violation_blocks(row: dict, margin: float) -> tuple[int | None, int | None]:
    return first_block_below(row["cmin_by_block_imagined_health"], margin), first_block_below(row["cmin_by_block_dense_health"], 0.0)


TIMING_TOLERANCE_BLOCKS = 2


def select_example(rows: list[dict], category: str, margin: float) -> tuple[dict, dict]:
    """The fixed rule, applied to saved rows only.

    1. Keep the rows of the decision category.
    2. `fall_caught` only: prefer rows whose imagined violation arrives within
       TIMING_TOLERANCE_BLOCKS of the true one. A rejection that fires many blocks early, from a
       borderline root reading, is a correct decision but not an imagined fall.
    3. Unsafe categories: prefer rows where the torso really dropped below the healthy height
       (`fell`), so the clip shows a fall rather than a lean.
    4. Take the row with the MEDIAN true clearance (ties: lowest branch index).
    Every preference falls back to the larger pool when it would leave nothing.
    """
    pool = [r for r in rows if categorise(r, margin) == category]
    info = {"category": category, "n_in_category": len(pool), "n_rows": len(rows)}
    if not pool:
        return None, info
    if category == "fall_caught":
        timed = [r for r in pool if None not in violation_blocks(r, margin) and abs(np.subtract(*violation_blocks(r, margin))) <= TIMING_TOLERANCE_BLOCKS]
        info["n_timing_consistent"] = len(timed)
        info["timing_tolerance_blocks"] = TIMING_TOLERANCE_BLOCKS
        pool = timed or pool
    if category in ("fall_missed", "fall_caught"):
        fell = [r for r in pool if r.get("fell")]
        info["n_fell_in_pool"] = len(fell)
        pool = fell or pool
    pool = sorted(pool, key=lambda r: (r["cmin_dense_health"], r["branch"]))
    info["n_in_final_pool"] = len(pool)
    info["rule"] = "category; fall_caught prefers timing-consistent rows; unsafe categories prefer rows that fell; then the median true health clearance (ties by branch index)"
    return pool[len(pool) // 2], info


# --------------------------------------------------------------------------------------
# retrieval gallery (probe-split frames only)
# --------------------------------------------------------------------------------------
class Gallery:
    def __init__(self, im: WalkerImaginer, ctx: RenderContext, h5_path: Path, n_frames: int, stride: int = 5):
        t0 = time.time()
        Z, QP, QV, EP = [], [], [], []
        got = 0
        for ei, ep in iter_episodes(h5_path):
            idx = np.arange(0, len(ep["qpos"]), stride)
            frames = ctx.render_many(ep["qpos"][idx], ep["qvel"][idx])
            Z.append(im.encode(frames))
            QP.append(ep["qpos"][idx])
            QV.append(ep["qvel"][idx])
            EP.append(np.full(len(idx), ei))
            got += len(idx)
            if got >= n_frames:
                break
        self.z = torch.cat(Z)
        self.qpos, self.qvel, self.episode = np.concatenate(QP), np.concatenate(QV), np.concatenate(EP)
        self.ctx = ctx
        self.seconds = time.time() - t0

    def __len__(self) -> int:
        return len(self.qpos)

    @torch.inference_mode()
    def nearest(self, z: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        d = torch.cdist(z.float().reshape(-1, z.shape[-1]), self.z)
        dist, idx = d.min(1)
        return idx.cpu().numpy(), dist.cpu().numpy()

    def frames(self, idx) -> np.ndarray:
        return self.ctx.render_many(self.qpos[idx], self.qvel[idx])


# --------------------------------------------------------------------------------------
# one clip worth of data
# --------------------------------------------------------------------------------------
def build_clip_data(im, probe, ctx, gallery, *, hist_qpos, hist_qvel, hist_actions, tape, qpos, qvel, n_blocks, stored_end_frames=None) -> dict:
    """qpos/qvel: (n_blocks*FRAMESKIP + 1, nq) dense truth from the root state onward.

    `stored_end_frames` (bank branches) are used for every model input at block endpoints so
    the numbers equal the saved S4 evaluation; re-rendered frames are for display only.
    """
    z_hist = im.encode(ctx.render_many(hist_qpos, hist_qvel))
    z_imag = im.rollout(z_hist, hist_actions, np.asarray(tape, np.float64).reshape(1, n_blocks, FRAMESKIP, 6))[0]
    dense_frames = ctx.render_many(qpos, qvel)
    end_frames = dense_frames[::FRAMESKIP] if stored_end_frames is None else np.asarray(stored_end_frames)
    z_real = im.encode(end_frames)
    y0 = probe.predict(z_hist[-1:])[0]
    pred = np.vstack([y0, probe.predict(z_imag)])  # (K+1, 3) imagined readout at endpoints
    real = np.vstack([y0, probe.predict(z_real[1:])])  # (K+1, 3) readout of real frames
    lat = torch.cat([z_hist[-1:], z_imag])  # latent the model holds at each endpoint
    nn_imag, d_imag = gallery.nearest(lat)
    nn_real, d_real = gallery.nearest(z_real)
    t_dense = np.arange(len(qpos)) * DT
    c_true = health_clearance(qpos[1:, 1], qpos[1:, 2])
    c_imag = health_clearance(interp_steps(pred[:, 0]), interp_steps(pred[:, 1]))
    c_real = health_clearance(interp_steps(real[:, 0]), interp_steps(real[:, 1]))
    latent_err = torch.linalg.norm(lat - z_real, dim=1).cpu().numpy()
    return {"dense_frames": dense_frames, "end_frames": end_frames, "pred": pred, "real": real, "t_dense": t_dense,
            "t_end": np.arange(n_blocks + 1) * FRAMESKIP * DT, "height": qpos[:, 1], "pitch": qpos[:, 2],
            "c_true": c_true, "c_imag": c_imag, "c_real": c_real, "nn_imag": nn_imag, "nn_real": nn_real, "d_imag": d_imag, "d_real": d_real,
            "nn_imag_frames": gallery.frames(nn_imag), "nn_real_frames": gallery.frames(nn_real), "latent_err": latent_err, "n_blocks": n_blocks}


# --------------------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------------------
def render_clip(data: dict, out_stem: Path, *, title: str, subtitle: str, verdict: str, margin: float | None, evaluated_blocks: int = HORIZON_BLOCKS) -> dict:
    K = data["n_blocks"]
    T = K * FRAMESKIP
    t_max = T * DT
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1.0], left=0.055, right=0.985, top=0.80, bottom=0.08, hspace=0.30, wspace=0.2)
    fig.text(0.055, 0.962, title, fontsize=14.5, fontweight="bold", va="center")
    fig.text(0.055, 0.922, subtitle, fontsize=10, va="center", color="#333333")
    fig.text(0.055, 0.886, verdict, fontsize=10, va="center", color="#111111", family="monospace")
    clock = fig.text(0.985, 0.962, "", fontsize=10.5, ha="right", va="center", family="monospace", color="#333333")
    ax_img = [fig.add_subplot(gs[0, i]) for i in range(3)]
    names = ["Real simulator (MuJoCo, stored state)", "Nearest real frame to the ENCODED real frame", "Nearest real frame to the IMAGINED latent"]
    ims, caps = [], []
    for ax, name, col in zip(ax_img, names, (C_TRUE, C_REAL, C_IMAG)):
        ax.set_title(name, fontsize=10, color=col, pad=5)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(col)
            s.set_linewidth(1.6)
        ims.append(ax.imshow(data["dense_frames"][0], interpolation="lanczos"))
        caps.append(ax.text(0.5, -0.035, "", transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color="#333333"))
    ax_h, ax_p, ax_c = (fig.add_subplot(gs[1, i]) for i in range(3))
    series = (("height", 0, ax_h, "torso height (m)", HEALTHY_Z), ("pitch", 1, ax_p, "torso pitch (rad)", HEALTHY_ANGLE))
    cursors = []
    for key, col, ax, label, band in series:
        ax.axhspan(band[0], band[1], color="#2ca02c", alpha=0.08, lw=0)
        for b in band:
            ax.axhline(b, color="#2ca02c", lw=0.9, ls=":")
        ax.plot(data["t_dense"], data[key], color=C_TRUE, lw=1.8, label="true (every 0.008 s)")
        ax.plot(data["t_end"], data["real"][:, col], color=C_REAL, lw=0, marker="o", ms=4.5, label="probe on real frames")
        ax.plot(data["t_end"], data["pred"][:, col], color=C_IMAG, lw=1.3, ls="--", marker="s", ms=4.5, label="probe on imagined latents")
        lo = min(data[key].min(), data["pred"][:, col].min(), data["real"][:, col].min(), band[0])
        hi = max(data[key].max(), data["pred"][:, col].max(), data["real"][:, col].max(), min(band[1], data[key].max() + 0.4))
        pad = 0.08 * (hi - lo + 1e-9)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(0, t_max)
        ax.set_ylabel(label, fontsize=9.5)
        ax.set_xlabel("time from the root (s)", fontsize=9.5)
        ax.tick_params(labelsize=8.5)
        ax.grid(alpha=0.25)
        cursors.append(ax.axvline(0, color="#555555", lw=1.0))
    ax_h.legend(fontsize=8, loc="best", framealpha=0.9)
    tt = data["t_dense"][1:]
    ax_c.axhspan(min(data["c_true"].min(), data["c_imag"].min(), -0.2) - 0.2, 0, color="#d62728", alpha=0.07, lw=0)
    ax_c.axhline(0, color="#d62728", lw=0.9, ls=":")
    ax_c.plot(tt, data["c_true"], color=C_TRUE, lw=1.8, label="true")
    ax_c.plot(tt, data["c_real"], color=C_REAL, lw=1.1, label="from real frames")
    ax_c.plot(tt, data["c_imag"], color=C_IMAG, lw=1.3, ls="--", label="imagined")
    if margin is not None:
        ax_c.axhline(margin, color=C_IMAG, lw=0.8, ls="-.", alpha=0.7)
    ax_c.set_xlim(0, t_max)
    ax_c.set_ylim(min(data["c_true"].min(), data["c_imag"].min(), data["c_real"].min(), -0.2) - 0.15, max(data["c_true"].max(), data["c_imag"].max(), data["c_real"].max()) + 0.15)
    ax_c.set_ylabel("health clearance (scaled; < 0 is unsafe)", fontsize=9.5)
    ax_c.set_xlabel("time from the root (s)", fontsize=9.5)
    ax_c.tick_params(labelsize=8.5)
    ax_c.grid(alpha=0.25)
    ax_c.legend(fontsize=8, loc="best", framealpha=0.9)
    cursors.append(ax_c.axvline(0, color="#555555", lw=1.0))
    if K > evaluated_blocks:
        for ax in (ax_h, ax_p, ax_c):
            ax.axvspan(evaluated_blocks * FRAMESKIP * DT, t_max, color="#888888", alpha=0.10, lw=0)
            ax.text(t_max - 0.02, ax.get_ylim()[0], "beyond the evaluated 0.8 s horizon ", fontsize=7.5, va="bottom", ha="right", color="#555555")

    def draw(step: int):
        k = step // FRAMESKIP
        ims[0].set_data(data["dense_frames"][step])
        ims[1].set_data(data["nn_real_frames"][k])
        ims[2].set_data(data["nn_imag_frames"][k])
        caps[0].set_text(f"height {data['height'][step]:.2f} m, pitch {data['pitch'][step]:+.2f} rad")
        caps[1].set_text(f"block {k}: probe reads {data['real'][k, 0]:.2f} m, {data['real'][k, 1]:+.2f} rad")
        caps[2].set_text(f"block {k}: probe reads {data['pred'][k, 0]:.2f} m, {data['pred'][k, 1]:+.2f} rad (latent error {data['latent_err'][k]:.1f})")
        for c in cursors:
            c.set_xdata([step * DT, step * DT])
        clock.set_text(f"t = {step * DT:5.3f} s  block {k:2d}/{K}  0.2x speed")
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()

    steps = [0] * int(HOLD_START_S * FPS) + list(range(T + 1)) + [T] * int(HOLD_END_S * FPS)
    mp4, gif, png = out_stem.with_suffix(".mp4"), out_stem.with_suffix(".gif"), out_stem.with_suffix(".png")
    first = draw(0)
    h, w = first.shape[:2]
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
                             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(mp4)], stdin=subprocess.PIPE)
    cache = {}
    last = None
    for s in steps:
        if s not in cache:
            cache = {s: draw(s)}
        last = cache[s]
        proc.stdin.write(last.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {mp4}")
    plt.imsave(png, last)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vf",
                    f"fps={GIF_FPS},scale={GIF_WIDTH}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                    str(gif)], check=True)
    plt.close(fig)
    return {"mp4": mp4.name, "gif": gif.name, "poster": png.name, "video_frames": len(steps), "fps": FPS, "seconds": len(steps) / FPS,
            "mp4_bytes": mp4.stat().st_size, "gif_bytes": gif.stat().st_size, "size": [int(w), int(h)]}


def readme(rec: dict) -> str:
    """The folder README, written from the record so text and numbers cannot drift apart."""
    L = [f"# Walker2d animations from the trained LeWM ({rec['model_run']})", "",
         f"Checkpoint at {rec['model_step']:,} optimizer updates (`weights.pt` sha256 `{rec['weights_sha256'][:16]}...`), read out with the frozen S3 probe "
         f"`{rec['probe']}`. Rebuild with `uv run python experiments/scripts/walker_lewm_animations.py` (about four minutes on the 5090).", "",
         "**These are illustrations, not new statistics.** Each clip is one saved branch. The counts beside each clip say how common its outcome is in the saved S4 rows; "
         "the S4 report holds the rates, intervals and costs.", "",
         "## How to read a clip", "",
         "LeWM predicts latents and has no image decoder, so imagination is shown two ways.", "",
         "- **Left image:** the MuJoCo render of the stored simulator state, one video frame per 0.008 s environment step, played at 0.2x speed.",
         "- **Middle image:** the nearest real frame, by L2 distance in the 192-d latent, to the encoded real frame at the last block endpoint. It shows what retrieval looks like when the latent is right.",
         f"- **Right image:** the nearest real frame to the latent the model imagined for that endpoint. The gallery is {rec['gallery']['n_frames']:,} frames from {rec['gallery']['n_episodes']} probe-split episodes, disjoint from the test roots.",
         "- **Traces:** torso height and pitch. Black is the truth at every step, blue dots are the probe on real endpoint frames, red squares are the probe on imagined latents. Green bands are the healthy range.",
         "- **Clearance panel:** the health rule's scaled clearance. Below zero is unsafe. The model accepts a tape when its imagined minimum stays at or above the margin.", "",
         "The model sees three real frames (0.16 s of history) and the action tape, then imagines ten blocks of 0.08 s. Images update once per block because imagination exists only at block endpoints.", "",
         "## Clips", ""]
    for bank, b in rec["banks"].items():
        c = b["category_counts"]
        L += [f"### {bank.capitalize()} bank ({b['n_rows']} saved rows, margin {b['margin_health']:+.2f})", "",
              f"Outcomes in the saved rows: {c['safe_accepted']} safe and accepted, {c['fall_missed']} unsafe but accepted, {c['fall_caught']} unsafe and rejected, {c['false_alarm']} safe but rejected.", "",
              "| Clip | Outcome | Count | Branch | True min | Imagined min | First violation (true / imagined block) |", "|---|---|---|---|---|---|---|"]
        for name, clip in rec["clips"].items():
            if clip.get("bank") != bank or clip.get("skipped"):
                continue
            L.append(f"| [`{name}`]({clip['mp4']}) ([gif]({clip['gif']}), [poster]({clip['poster']})) | {CATEGORIES[clip['category']]} | {clip['n_in_category']} | {clip['branch']} of root `{clip['root_id']}` "
                     f"| {clip['cmin_true']:+.2f} | {clip['cmin_imagined']:+.2f} | {clip['first_violation_block_true'] or '-'} / {clip['first_violation_block_imagined'] or '-'} |")
        caught = rec["clips"].get(f"{bank}_fall_caught", {})
        if caught and not caught.get("skipped"):
            L += ["", f"Of the {caught['n_in_category']} rejected unsafe rows, {caught['n_timing_consistent']} have the imagined violation within {caught['timing_tolerance_blocks']} blocks of the true one; "
                      f"the clip is drawn from those ({caught['n_fell_in_pool']} of them end with the torso below the healthy height)."]
        L.append("")
    lh = rec["clips"].get("long_horizon")
    if lh:
        L += ["### Long horizon", "",
              f"[`long_horizon`]({lh['mp4']}) ([gif]({lh['gif']}), [poster]({lh['poster']})): test-role episode {lh['episode']} of `roots.h5` from step {lh['root_step']}, imagined {lh['n_blocks']} blocks (2.4 s) ahead with its recorded actions. "
              f"Height error is {lh['height_abs_error_by_block'][lh['evaluated_horizon_blocks']]:.3f} m at 0.8 s and {lh['height_abs_error_by_block'][-1]:.3f} m at 2.4 s; pitch error is "
              f"{lh['pitch_abs_error_by_block'][lh['evaluated_horizon_blocks']]:.3f} and {lh['pitch_abs_error_by_block'][-1]:.3f} rad. Everything past 0.8 s lies outside the horizon evaluated in S3 and S4 and is shaded grey.", "",
              f"Selection: {lh['selection_rule']} ({lh['n_eligible_episodes']} eligible episodes; this one moves {lh['forward_displacement_m']:.2f} m forward in the window). It is ordinary walking under the policy that produced the training data, which is the easy case.", ""]
    L += ["## How examples were chosen", "",
          "From the saved S4 no-update rows only, by a fixed rule, never by eye:", "",
          "1. Keep the rows of the outcome category at the saved matched margin.",
          f"2. For rejected unsafe rows, prefer those whose imagined violation arrives within {TIMING_TOLERANCE_BLOCKS} blocks of the true one. A rejection that fires many blocks early from a borderline root reading is a correct decision, not an imagined fall.",
          "3. For the two unsafe outcomes, prefer rows where the torso really dropped below the healthy height, so the clip shows a fall and not a lean.",
          "4. Take the row with the median true clearance, ties broken by branch index.", "",
          "## Caveats", "",
          "- **Speed is not shown.** S3 qualified the health rule only; the speed readout from real frames did not track the truth well enough.",
          "- **Missed falls are the common failure.** At this margin the model accepts almost every tape, so most unsafe tapes are accepted. The rejected examples are rare and should not be read as typical.",
          "- **Retrieval is an illustration of a latent, not a decoded image.** The right-hand image is a real frame from another episode that happens to lie nearest in latent space. The floor pattern and exact limb pose can differ.",
          "- **Probe readings of fallen poses are unreliable.** The blue dots scatter once the robot is on the ground, which is outside the range where the probe was accurate.",
          "- **Stored frames lag the stored state by one physics substep.** The bank rendered each endpoint straight after a simulation step, when MuJoCo's kinematics are 0.002 s behind the state. "
          "Every model input here uses those stored frames, so the numbers equal the saved S4 rows. The displayed frames are re-rendered from the stored states and differ from the stored endpoint frames in at most "
          f"{max((c['displayed_rerender_vs_stored_endpoints']['max_frac_pixels_differing'] for c in rec['clips'].values() if 'displayed_rerender_vs_stored_endpoints' in c), default=0):.3%} of pixels.",
          "- MP4 is lossy and GIF is resized and palette encoded. Initial and final holds repeat a frame; they are not new observations.", "",
          "Full provenance, hashes and per-block numbers are in [`animations.json`](animations.json).", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--banks", nargs="+", default=["test", "stress"], choices=["test", "stress"])
    ap.add_argument("--gallery-frames", type=int, default=24000)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    t_start = time.time()
    out = OUT_ROOT / MODEL_RUN.name
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    model, scaler = load_walker_model(MODEL_RUN, device)
    im = WalkerImaginer(model, scaler, device)
    probe, probe_stats = load_probe(PROBES_RUN / "walker_mlp.pt", device)
    gate = json.loads((PROBES_RUN / "gate.json").read_text())
    study = json.loads((S4_RUN / "study.json").read_text())
    all_rows = json.loads((S4_RUN / "no_update_evaluation_rows.json").read_text())
    ctx = RenderContext()
    fp = render_fingerprint(ctx)
    with h5py.File(DATA_DIR / "roots.h5", "r") as f:
        fp_data = f.attrs.get("render_fingerprint")
        fp_data = fp_data.decode() if isinstance(fp_data, bytes) else fp_data
    if fp_data and fp != fp_data:
        raise RuntimeError(f"render fingerprint {fp} differs from the dataset's {fp_data}; frames would not match the training stack")
    gallery = Gallery(im, ctx, DATA_DIR / "probe.h5", args.gallery_frames)
    print(f"[anim] gallery: {len(gallery):,} probe-split frames from {len(np.unique(gallery.episode))} episodes in {gallery.seconds:.0f}s; fingerprint {fp}")
    record = {"model_run": MODEL_RUN.name, "model_step": json.loads((MODEL_RUN / "train_state.json").read_text())["step"], "weights_sha256": sha256(MODEL_RUN / "weights.pt"),
              "probe": str((PROBES_RUN / "walker_mlp.pt").relative_to(REPO_ROOT)), "probe_sha256": sha256(PROBES_RUN / "walker_mlp.pt"),
              "probe_validation": {k: probe_stats["val"][k] for k in ("height_r2", "pitch_r2", "speed_r2", "height_rmse", "pitch_rmse") if k in probe_stats.get("val", {})},
              "gate_active_rules": gate["gate"]["active_rules"], "speed_rule_ok": gate["gate"]["speed_rule_ok"],
              "rows_source": str((S4_RUN / "no_update_evaluation_rows.json").relative_to(REPO_ROOT)), "render_fingerprint": fp,
              "gallery": {"source": str((DATA_DIR / "probe.h5").relative_to(REPO_ROOT)), "n_frames": len(gallery), "n_episodes": int(len(np.unique(gallery.episode))), "stride": 5, "metric": "L2 in the 192-d latent"},
              "playback": {"fps": FPS, "speed": 0.2, "hold_start_s": HOLD_START_S, "hold_end_s": HOLD_END_S}, "banks": {}, "clips": {}}

    for bank_name in args.banks:
        rows = all_rows[bank_name]
        margin = float(study["decomposition"]["health"][bank_name]["imagined"]["m"])
        bank = WalkerBank(S4_DATA / bank_name)
        counts = {c: sum(categorise(r, margin) == c for r in rows) for c in CATEGORIES}
        record["banks"][bank_name] = {"n_rows": len(rows), "margin_health": margin, "category_counts": counts, "branches_sha256": sha256(S4_DATA / bank_name / "branches.h5")}
        print(f"[anim] {bank_name} bank at margin {margin:.4f}: {counts}")
        for cat, text in CATEGORIES.items():
            name = f"{bank_name}_{cat}"
            if args.only and name not in args.only:
                continue
            row, info = select_example(rows, cat, margin)
            if row is None:
                print(f"[anim] {name}: no rows in this category; skipped")
                record["clips"][name] = {"skipped": True, "bank": bank_name, **info}
                continue
            j = int(row["branch"])
            root = bank.roots[int(bank.h5["root_index"][j])]
            assert root.root_id == row["root_id"], (root.root_id, row["root_id"])
            qpos, qvel, tape = bank.h5["qpos"][j], bank.h5["qvel"][j], bank.h5["tape"][j]
            stored = bank.h5["frames"][j]
            d = build_clip_data(im, probe, ctx, gallery, hist_qpos=root.history_qpos, hist_qvel=root.history_qvel, hist_actions=root.history_actions, tape=tape, qpos=qpos, qvel=qvel,
                                n_blocks=HORIZON_BLOCKS, stored_end_frames=stored)
            px = np.abs(d["dense_frames"][::FRAMESKIP].astype(int) - stored.astype(int)).reshape(len(stored), -1)
            rerender = {"max_abs_diff": int(px.max()), "max_frac_pixels_differing": float((px > 0).mean(1).max()), "mean_abs_diff": float(px.mean()), "root_frame_equal": bool(px[0].max() == 0)}
            cmin_imag, cmin_true = float(d["c_imag"].min()), float(d["c_true"].min())
            diff = abs(cmin_imag - row["cmin_imagined_health"])
            # The one-substep kinematics lag moves more pixels in fast stress tapes (0.5% measured);
            # a wrong render stack differs in about 90% of pixels, so 5% separates the two.
            if rerender["max_frac_pixels_differing"] > 0.05:
                raise RuntimeError(f"{name}: re-rendered endpoints differ from stored frames in {rerender['max_frac_pixels_differing']:.2%} of pixels; the render stack is not the one that built the bank")
            if diff > 5e-3 or abs(cmin_true - row["cmin_dense_health"]) > 1e-9:
                raise RuntimeError(f"{name}: recomputed clearance differs from the saved row (imagined {cmin_imag} vs {row['cmin_imagined_health']}, true {cmin_true} vs {row['cmin_dense_health']})")
            kind = bank.h5["kind"][j]
            kind = kind.decode() if isinstance(kind, bytes) else str(kind)
            decision = "ACCEPT" if cmin_imag >= margin else "REJECT"
            truth = "UNSAFE" if cmin_true < 0 else "SAFE"
            b_imag, b_true = violation_blocks(row, margin)
            when = "" if b_true is None and b_imag is None else f"    first violation: true block {b_true or '-'}, imagined block {b_imag or '-'}"
            media = render_clip(d, out / name, title=f"Walker2d LeWM imagination: {SHORT[cat]} ({bank_name} bank)",
                                subtitle=f"{text}. Saved branch {j} ({kind} tape), root {root.root_id}. This outcome: {info['n_in_category']} of {len(rows)} saved {bank_name} rows.",
                                verdict=f"truth: {truth} (min {cmin_true:+.2f})    model: {decision} (imagined min {cmin_imag:+.2f}, margin {margin:+.2f}){when}", margin=margin)
            record["clips"][name] = {"bank": bank_name, **info, **media, "branch": j, "root_id": root.root_id, "tape_kind": kind, "fell": bool(row.get("fell")), "truth": truth, "decision": decision,
                                     "cmin_true": cmin_true, "cmin_imagined": cmin_imag, "cmin_imagined_saved_row": row["cmin_imagined_health"], "recompute_abs_diff": diff,
                                     "cmin_real_readout": float(d["c_real"].min()), "first_violation_block_true": b_true, "first_violation_block_imagined": b_imag,
                                     "displayed_rerender_vs_stored_endpoints": rerender,
                                     "imagined_readout": d["pred"][:, :2].round(4).tolist(), "real_readout": d["real"][:, :2].round(4).tolist(),
                                     "latent_error_by_block": d["latent_err"].round(4).tolist(), "gallery_distance_imagined": d["d_imag"].round(4).tolist(), "gallery_distance_real": d["d_real"].round(4).tolist()}
            print(f"[anim] {name}: branch {j} root {root.root_id} truth {truth} {cmin_true:+.3f} model {decision} {cmin_imag:+.3f} (saved row diff {diff:.1e}); violation blocks true {b_true} imagined {b_imag}; "
                  f"re-render differs in at most {rerender['max_frac_pixels_differing']:.3%} of pixels; mp4 {media['mp4_bytes'] / 1e6:.1f} MB gif {media['gif_bytes'] / 1e6:.1f} MB")

    name = "long_horizon"
    if not args.only or name in args.only:
        test_eps = set(json.loads((DATA_DIR / "splits.json").read_text())["roots"]["test"])
        t_lo, t_hi = LONG_ROOT_STEP, LONG_ROOT_STEP + LONG_BLOCKS * FRAMESKIP
        eligible = sorted(((float(ep["qpos"][t_hi, 0] - ep["qpos"][t_lo, 0]), ei) for ei, ep in iter_episodes(DATA_DIR / "roots.h5")
                           if ei in test_eps and len(ep["qpos"]) >= LONG_MIN_STEPS))
        chosen = None
        if eligible:
            disp, pick = eligible[len(eligible) // 2]
            chosen = next((ei, ep) for ei, ep in iter_episodes(DATA_DIR / "roots.h5") if ei == pick)
        if chosen is None:
            print("[anim] long_horizon: no test episode is long enough; skipped")
        else:
            ei, ep = chosen
            t0 = LONG_ROOT_STEP
            T = LONG_BLOCKS * FRAMESKIP
            hidx = np.arange(t0 - (HISTORY - 1) * FRAMESKIP, t0 + 1, FRAMESKIP)
            d = build_clip_data(im, probe, ctx, gallery, hist_qpos=ep["qpos"][hidx], hist_qvel=ep["qvel"][hidx],
                                hist_actions=np.asarray(ep["action"][hidx[0] : t0], float).reshape(HISTORY - 1, FRAMESKIP, 6),
                                tape=np.asarray(ep["action"][t0 : t0 + T], float), qpos=ep["qpos"][t0 : t0 + T + 1], qvel=ep["qvel"][t0 : t0 + T + 1], n_blocks=LONG_BLOCKS)
            err_h = np.abs(d["pred"][:, 0] - d["height"][::FRAMESKIP])
            err_p = np.abs(d["pred"][:, 1] - d["pitch"][::FRAMESKIP])
            media = render_clip(d, out / name, title="Walker2d LeWM imagination: 2.4 s open loop",
                                subtitle=f"Held-out test-role episode {ei} from step {t0}, imagined {LONG_BLOCKS} blocks ahead with its recorded actions: 3x the 0.8 s horizon evaluated in S3/S4.",
                                verdict=f"|height error| {err_h[HORIZON_BLOCKS]:.2f} m at 0.8 s, {err_h[-1]:.2f} m at 2.4 s    |pitch error| {err_p[HORIZON_BLOCKS]:.2f} rad at 0.8 s, {err_p[-1]:.2f} rad at 2.4 s", margin=None)
            record["clips"][name] = {"bank": None, **media, "episode": int(ei), "root_step": t0, "n_blocks": LONG_BLOCKS, "n_eligible_episodes": len(eligible), "forward_displacement_m": disp,
                                     "selection_rule": f"among test-role episodes in roots.h5 with at least {LONG_MIN_STEPS} steps, the one with the median forward displacement over the window; root at step {LONG_ROOT_STEP}",
                                     "height_abs_error_by_block": err_h.round(4).tolist(), "pitch_abs_error_by_block": err_p.round(4).tolist(), "latent_error_by_block": d["latent_err"].round(4).tolist(),
                                     "evaluated_horizon_blocks": HORIZON_BLOCKS, "note": "Blocks beyond 10 lie outside the horizon evaluated in S3 and S4."}
            print(f"[anim] {name}: episode {ei} step {t0}; height error {err_h[HORIZON_BLOCKS]:.3f} m at 0.8 s, {err_h[-1]:.3f} m at 2.4 s; {media['seconds']:.1f}s mp4 {media['mp4_bytes'] / 1e6:.1f} MB gif {media['gif_bytes'] / 1e6:.1f} MB")
    record["wall_clock_s"] = time.time() - t_start
    (out / "animations.json").write_text(json.dumps(record, indent=1) + "\n")
    (out / "README.md").write_text(readme(record))
    print(f"[anim] wrote {out} in {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
