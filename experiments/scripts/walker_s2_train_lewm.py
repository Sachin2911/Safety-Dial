#!/usr/bin/env python3
"""S2: train LeWM-A (or -B) on Walker2d set A with the upstream recipe on the 5090.

Upstream defaults (le-wm config/train/lewm.yaml at the pinned commit): AdamW lr 5e-5,
weight decay 1e-3, batch 128, bf16, grad clip 1.0, linear warmup + cosine, SIGReg weight
0.1 (1,024 projections, 17 knots), history 3, one prediction. Data: frameskip 10, 224 px,
lazy EGL rendering in spawn workers. Logs to W&B; pushes intermediate checkpoints to
<ns>/safetydial-walker2d:lewm-a/<run_id> every --push-every steps so a recycled instance
loses at most one interval.

    uv run python experiments/scripts/walker_s2_train_lewm.py --smoke                  # 300 steps
    uv run python experiments/scripts/walker_s2_train_lewm.py --epochs 10 --name lewm-a
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from helpers.threads import apply_torch, pin_threads  # noqa: E402

pin_threads()

import numpy as np  # noqa: E402
import torch  # noqa: E402

apply_torch()

from helpers.hfStore import HFStore  # noqa: E402
from helpers.locoData import make_loco_dataloader  # noqa: E402
from helpers.runManifest import build_manifest, make_run_id, write_manifest  # noqa: E402
from helpers.walkerLewm import (  # noqa: E402
    SIGReg,
    action_scaler,
    build_model,
    lewm_loss,
    make_dataset,
    model_config,
    preprocess_batch,
    save_checkpoint,
)

DATA = REPO_ROOT / "data" / "study" / "walker2d"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA / "setA.h5"))
    ap.add_argument("--name", default="lewm-a")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--clip-stride", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--push-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=3072)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--n", type=int, default=1)
    args = ap.parse_args()
    t_start = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    run_id = make_run_id("walker2d", args.name, "smoke" if args.smoke else "", n=args.n)
    run_dir = REPO_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    ds = make_dataset(Path(args.data), clip_stride=args.clip_stride)
    scaler = action_scaler(ds)
    n_val = max(1, int(0.02 * len(ds)))
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(ds), generator=g).tolist()
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    train_loader = make_loco_dataloader(torch.utils.data.Subset(ds, tr_idx), batch_size=args.batch, num_workers=args.workers, drop_last=True, pin_memory=True, prefetch_factor=3)
    val_loader = make_loco_dataloader(torch.utils.data.Subset(ds, val_idx[:512]), batch_size=args.batch, num_workers=max(1, args.workers // 2), shuffle=False)
    print(f"[s2] dataset {len(ds):,} clips ({len(tr_idx):,} train); render fingerprint {ds.attrs.get('render_fingerprint')}")

    cfg = model_config()
    model = build_model(cfg).to(device)
    sigreg = SIGReg(knots=17, num_proj=1024).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = len(train_loader)
    total = args.max_steps or (300 if args.smoke else args.epochs * steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(total, 1)))))
    print(f"[s2] model {n_params / 1e6:.1f}M params; {steps_per_epoch} steps/epoch; total {total} steps")
    wandb = None
    if not args.no_wandb and os.environ.get("WANDB_API_KEY"):
        import wandb as _wandb

        wandb = _wandb
        wandb.init(project="safetydial-walker2d", name=run_id, config={**vars(args), "n_params": n_params, "steps_per_epoch": steps_per_epoch})
    store = None if args.no_upload else HFStore()
    manifest_base = dict(run_id=run_id, kind=args.name, seeds={"seed": args.seed}, data={"path": args.data, "n_clips": len(ds), "render_fingerprint": ds.attrs.get("render_fingerprint"), "frameskip": 10, "history": 3})

    step, t0, hist = 0, time.time(), []
    black_checked = False
    done = False
    while not done:
        for batch in train_loader:
            if not black_checked:
                px = batch["pixels"]
                print(f"[s2] first batch pixels {tuple(px.shape)} {px.dtype} mean {float(px.float().mean()):.1f} (black frames would be ~0)")
                assert float(px.float().mean()) > 5.0, "black frames: EGL context problem"
                black_checked = True
            b = preprocess_batch(batch, scaler[0], scaler[1], device)
            model.train()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = lewm_loss(model, sigreg, b)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 20 == 0:
                rec = {"step": step, "loss": float(out["loss"]), "pred_loss": float(out["pred_loss"]), "sigreg_loss": float(out["sigreg_loss"]), "lr": sched.get_last_lr()[0], "it_per_s": step / (time.time() - t0)}
                hist.append(rec)
                if wandb:
                    wandb.log(rec, step=step)
                if step % 100 == 0:
                    print(f"[s2] step {step}/{total} loss {rec['loss']:.4f} pred {rec['pred_loss']:.4f} sig {rec['sigreg_loss']:.4f} {rec['it_per_s']:.2f} it/s")
            if step % args.push_every == 0 or step >= total:
                model.eval()
                vl = []
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for vb in val_loader:
                        vl.append(float(lewm_loss(model, sigreg, preprocess_batch(vb, scaler[0], scaler[1], device))["pred_loss"]))
                val_pred = float(np.mean(vl))
                if wandb:
                    wandb.log({"val_pred_loss": val_pred}, step=step)
                save_checkpoint(model, run_dir, cfg=cfg, scaler=scaler, step=step, extra={"val_pred_loss": val_pred, "history": hist[-50:]})
                write_manifest(run_dir, build_manifest(**manifest_base, metrics={"step": step, "val_pred_loss": val_pred, "train_loss": hist[-1]["loss"] if hist else None}, started_at=t_start))
                (run_dir / "README.md").write_text(f"# {run_id}\n\nLeWM trained from scratch on Walker2d set A (frameskip 10, history 3, 224 px). Step {step}. Not a released checkpoint. See manifest.json.\n")
                print(f"[s2] step {step}: val pred loss {val_pred:.4f}; checkpoint saved")
                if store is not None:
                    store.upload_run("walker2d", args.name, run_dir, run_id=run_id, tag=step >= total)
            if step >= total:
                done = True
                break
    out_json = REPO_ROOT / "docs" / "mainPlan" / "results" / "s2"
    out_json.mkdir(parents=True, exist_ok=True)
    (out_json / f"{run_id}.json").write_text(json.dumps({"run_id": run_id, "steps": step, "history": hist, "wall_clock_s": time.time() - t_start, "n_clips": len(ds), "n_params": n_params}, indent=1) + "\n")
    print(f"[s2] done: {step} steps in {(time.time() - t_start) / 60:.1f} min -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
