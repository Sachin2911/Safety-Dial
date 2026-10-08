#!/usr/bin/env python3
"""Stage 3b of the evolution study (docs/evoPlan/README.md): calibrate imagination noise.

sigma is the per-dimension standard deviation of one-step teacher-forced residuals (predicted
minus real encoded latent) on tuning episodes: every development-role source episode of
roots.h5, rendered at block ends (stride 10, phase 0). Each window holds three real context
latents, the next real latent and the three action blocks between them. Also recorded, for
reference only: open-loop error growth over a 10-block horizon on the same episodes, sigma
from the S4 development bank's branches (perturbed tapes), and the imagined violation rate
of theta_BC on the tuning roots at each declared noise level.

    uv run python experiments/scripts/evo_calibrate_noise.py [--run-id ID] [--no-upload]
"""

from __future__ import annotations

import argparse
import os
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
import numpy as np  # noqa: E402
import torch  # noqa: E402

apply_torch()

from helpers.evoImagine import ClosedLoopImaginer, one_step_residuals, open_loop_errors, sigma_from_residuals  # noqa: E402
from helpers.evoInputs import EvoRun, fetch_inputs, input_revisions, load_stage_config  # noqa: E402
from helpers.evoPolicy import LinearLatentPolicy  # noqa: E402
from helpers.evoRanking import segment_metrics  # noqa: E402
from helpers.evoRoots import encode_histories, tuning_roots  # noqa: E402
from helpers.locoData import RenderContext  # noqa: E402
from helpers.poseProbes import load_probe  # noqa: E402
from helpers.runManifest import build_manifest, file_sha256  # noqa: E402
from helpers.walkerBank import WalkerBank  # noqa: E402
from helpers.walkerLewm import FRAMESKIP, HISTORY, load_walker_model  # noqa: E402
from helpers.walkerRules import HORIZON_BLOCKS  # noqa: E402
from helpers.walkerValidation import episode_role, verify_render_fingerprint  # noqa: E402


def normalise(blocks: np.ndarray, scaler) -> np.ndarray:
    """(..., F, 6) raw -> (..., F*6) float32, the LeWM action input (WalkerImaginer._flat)."""
    a = torch.as_tensor(np.asarray(blocks, dtype=np.float32))
    a = (a - torch.as_tensor(scaler[0], dtype=torch.float32)) / torch.as_tensor(scaler[1], dtype=torch.float32)
    return a.reshape(*a.shape[:-2], -1).numpy()


def episode_sequences(roots_h5: Path, role: str, ctx: RenderContext, encode, scaler):
    """Per episode of the role: block-end latents (T, D) and normalised blocks (T-1, 60)."""
    out = []
    with h5py.File(roots_h5, "r") as f:
        off, ln = f["ep_offset"][:].astype(np.int64), f["ep_len"][:].astype(np.int64)
        for e in range(len(ln)):
            if episode_role(e) != role:
                continue
            a, b = int(off[e]), int(off[e] + ln[e])
            rows = np.arange(0, b - a, FRAMESKIP)
            act = f["action"][a:b].astype(np.float64)
            n_blocks = 0
            while n_blocks < len(rows) - 1 and np.isfinite(act[rows[n_blocks] : rows[n_blocks] + FRAMESKIP]).all() \
                    and rows[n_blocks] + FRAMESKIP <= b - a:
                n_blocks += 1
            rows = rows[: n_blocks + 1]
            if len(rows) < HISTORY + 1:
                continue
            z = encode(ctx.render_many(f["qpos"][a:b][rows], f["qvel"][a:b][rows])).cpu().numpy()
            blocks = np.stack([act[t : t + FRAMESKIP] for t in rows[:-1]])
            out.append((e, z, normalise(blocks, scaler)))
    return out


def windows(seqs, span: int):
    """All windows of `span` latents and `span - 1` blocks, plus their episode ids."""
    Z, A, E = [], [], []
    for e, z, a in seqs:
        for i in range(len(z) - span + 1):
            Z.append(z[i : i + span])
            A.append(a[i : i + span - 1])
            E.append(e)
    return np.asarray(Z, np.float32), np.asarray(A, np.float32), np.asarray(E)


def bank_windows(bank: WalkerBank, encode, ctx, scaler):
    """S4 development bank: 3 history latents + 10 branch latents, 2 + 10 blocks per branch."""
    Z, A = [], []
    for ri, root in enumerate(bank.roots):
        z_hist = encode(ctx.render_many(root.history_qpos, root.history_qvel)).cpu().numpy()
        for j in bank.indices_for_root(ri):
            z_branch = encode(bank.h5["frames"][int(j)][1:]).cpu().numpy()
            tape = bank.h5["tape"][int(j)].astype(np.float64).reshape(HORIZON_BLOCKS, FRAMESKIP, 6)
            Z.append(np.concatenate([z_hist, z_branch]))
            A.append(normalise(np.concatenate([root.history_actions, tape]), scaler))
    return np.asarray(Z, np.float32), np.asarray(A, np.float32)


def main() -> int:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    cfg = load_stage_config("stage3")
    paths = fetch_inputs(cfg.inputs)
    run = EvoRun.create(cfg, "s3-noise", args.run_id)
    print(f"[evo-s3-noise] run {run.run_id}", flush=True)
    device = "cuda"
    model, scaler = load_walker_model(paths["model"], device)
    probe, _ = load_probe(paths["probes"] / "walker_mlp.pt", device)
    im = ClosedLoopImaginer(model, scaler, probe, device=device)
    ctx = RenderContext()
    roots_h5 = paths["data"] / "roots.h5"
    verify_render_fingerprint(ctx, roots_h5)

    seqs = episode_sequences(roots_h5, cfg.noise.tuning_role, ctx, im.encode, scaler)
    z1, a1, e1 = windows(seqs, HISTORY + 1)
    with torch.inference_mode():
        res = torch.cat([one_step_residuals(model, torch.as_tensor(z1[i : i + 4096], device=device),
                                            torch.as_tensor(a1[i : i + 4096], device=device)) for i in range(0, len(z1), 4096)])
    res = res.cpu().numpy().astype(np.float64)
    sigma = sigma_from_residuals(res)
    zk, ak, _ = windows(seqs, HISTORY + HORIZON_BLOCKS)
    growth = open_loop_errors(model, torch.as_tensor(zk, device=device), torch.as_tensor(ak, device=device), HORIZON_BLOCKS)
    step_rms = float(np.sqrt(((z1[:, 3] - z1[:, 2]) ** 2).sum(1).mean()))

    bank = WalkerBank(paths["banks"] / "dev")
    zb, ab = bank_windows(bank, im.encode, ctx, scaler)
    bank.h5.close()
    zb1 = np.concatenate([zb[:, i : i + HISTORY + 1] for i in range(zb.shape[1] - HISTORY)])
    ab1 = np.concatenate([ab[:, i : i + HISTORY] for i in range(zb.shape[1] - HISTORY)])
    res_b = one_step_residuals(model, torch.as_tensor(zb1, device=device), torch.as_tensor(ab1, device=device)).cpu().numpy()
    sigma_bank = sigma_from_residuals(res_b.astype(np.float64))

    # Reference only: theta_BC on the tuning roots at each noise level (n samples, CRN seed).
    bc = np.load(REPO_ROOT / cfg.bc_policy.file)
    policy = LinearLatentPolicy.from_state(bc, device=device)
    im.sigma = torch.as_tensor(sigma, dtype=torch.float32, device=device)
    tune = encode_histories(tuning_roots(paths["banks"]), im.encode, ctx)
    ctx.close()
    by_k = {}
    for k in cfg.noise.levels:
        n = 1 if k == 0 else cfg.noise.n_samples
        out = im.rollout_policies(policy, bc["theta"][None], torch.as_tensor(tune.z_hist, device=device), tune.hist_blocks,
                                  n_samples=n, k=float(k), seed=cfg.noise.reference_seed)
        m = segment_metrics(out["readout"][0])
        by_k[str(k)] = {"n_samples": n, "violation_rate": float(m["violated"].mean()), "return_mean": float(m["ret"].mean())}

    np.save(run.run_dir / "sigma.npy", sigma)
    np.save(run.results_dir / "sigma.npy", sigma)
    report = {"run_id": run.run_id, "stage": "3-noise", "tuning_role": cfg.noise.tuning_role,
              "n_episodes": len(seqs), "n_one_step_windows": int(len(z1)), "n_open_loop_windows": int(len(zk)),
              "sigma": {"mean": float(sigma.mean()), "median": float(np.median(sigma)), "min": float(sigma.min()),
                        "max": float(sigma.max()), "l2": float(np.linalg.norm(sigma))},
              "one_step_residual_rms": float(np.sqrt((res ** 2).sum(1).mean())),
              "latent_step_rms": step_rms, "open_loop_rms_by_block": np.asarray(growth).tolist(),
              "sigma_bank_reference": {"n_windows": int(len(zb1)), "l2": float(np.linalg.norm(sigma_bank)),
                                       "ratio_to_sigma_median": float(np.median(sigma_bank / sigma))},
              "theta_bc_tuning_by_k": by_k, "imagined_rows": int(im.counters.imagined_rows),
              "wall_clock_s": time.time() - t0}
    run.write_json("noise.json", report)
    manifest = build_manifest(run_id=run.run_id, kind="evo-s3-noise", seeds={"reference_seed": cfg.noise.reference_seed},
                              data={"sigma_sha256": file_sha256(run.run_dir / "sigma.npy"),
                                    "bc_policy_sha256": file_sha256(REPO_ROOT / cfg.bc_policy.file)},
                              upstream_revisions=input_revisions(paths), metrics={"sigma_l2": report["sigma"]["l2"]},
                              costs={"real_steps": 0, "imagined_rows": report["imagined_rows"]}, started_at=t0)
    run.finish(manifest, upload=not args.no_upload,
               readme=f"# {run.run_id}\n\nStage 3 noise calibration of docs/evoPlan: sigma.npy is the per-dimension std of one-step teacher-forced latent residuals on tuning episodes.\n")
    print(f"[evo-s3-noise] sigma l2 {report['sigma']['l2']:.4f}, one-step rms {report['one_step_residual_rms']:.4f}, "
          f"latent step rms {step_rms:.4f}; {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
