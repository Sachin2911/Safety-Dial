#!/usr/bin/env python3
"""S2: train LeWM-A (or -B) on Walker2d set A with the upstream recipe on the 5090.

Upstream defaults (le-wm config/train/lewm.yaml at the pinned commit): AdamW lr 5e-5,
weight decay 1e-3, batch 128, bf16, grad clip 1.0, linear warmup + cosine, SIGReg weight
0.1 (1,024 projections, 17 knots), history 3, one prediction. Data: frameskip 10, 224 px,
lazy EGL rendering in spawn workers. Logs to W&B; pushes intermediate checkpoints to
<ns>/safetydial-walker2d:lewm-a/<run_id> every --push-every steps so a recycled instance
loses at most one interval.

    uv run python experiments/scripts/walker_s2_train_lewm.py --smoke                  # 1,000 steps
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
from helpers.storageBudget import checkpoint_storage_budget, private_storage_profile, require_storage_budget
from helpers.walkerProtocol import episode_partition, file_sha256, indices_for_episode_roles, training_action_scaler
from helpers.walkerTrainingState import RestartableBatchSampler, restore_training_state, save_training_state, snapshot_sources, upload_checkpoint, verify_resume_cadence
from helpers.locoData import make_loco_dataloader  # noqa: E402
from helpers.runManifest import build_manifest, make_run_id, write_manifest  # noqa: E402
from helpers.walkerLewm import (  # noqa: E402
    SIGReg,
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
    ap.add_argument("--episode-splits", type=Path, help="explicit whole-episode roles, including S5 remapping")
    ap.add_argument("--match-run", type=Path, help="match A config, normalisers and optimizer count for S5")
    ap.add_argument("--run-id", help="explicit immutable run name for managed pipelines")
    ap.add_argument("--resume-from", type=Path, help="atomic optimizer.pt from the identical unfinished recipe; write to a fresh run-id")
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
    ap.add_argument("--push-every", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=3072)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--n", type=int, default=1)
    args = ap.parse_args()
    if args.match_run:
        from omegaconf import OmegaConf

        if args.smoke or not args.episode_splits:
            ap.error("--match-run requires --episode-splits and a full comparison run")
        reference_config = OmegaConf.to_container(OmegaConf.load(args.match_run / "config.yaml"))
        for name in ("batch", "lr", "wd", "warmup", "epochs", "seed", "clip_stride"):
            setattr(args, name, reference_config[name])
        args.max_steps = json.loads((args.match_run / "train_state.json").read_text())["step"]
    t_start = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    run_id = args.run_id or make_run_id("walker2d", args.name, "smoke" if args.smoke else "", n=args.n)
    if Path(run_id).name != run_id or run_id in (".", ".."):
        ap.error("run-id must be a single directory name")
    run_dir = REPO_ROOT / "runs" / run_id
    if args.batch < 2 or args.workers < 0 or args.push_every < 1 or args.warmup < 1:
        ap.error("batch >= 2, workers >= 0, push-every >= 1 and warmup >= 1 required")
    if args.epochs < 1 or args.max_steps < 0:
        ap.error("epochs must be positive and max-steps nonnegative")
    if args.resume_from:
        verify_resume_cadence(args.resume_from, args.push_every)
    run_dir.mkdir(parents=True, exist_ok=False)
    source_hashes = snapshot_sources(REPO_ROOT, run_dir)

    ds = make_dataset(Path(args.data), clip_stride=args.clip_stride)
    if args.episode_splits:
        episode_splits = json.loads(args.episode_splits.read_text())
        tr_idx, val_idx = indices_for_episode_roles(ds.clip_indices, episode_splits)
    else:
        tr_idx, val_idx, episode_splits = episode_partition(ds.clip_indices, seed=args.seed)
    if args.match_run:
        with np.load(args.match_run / "scalers.npz") as z:
            scaler = (z["action_mean"], z["action_std"])
    else:
        scaler = training_action_scaler(ds, episode_splits["training"])
    if len(tr_idx) < args.batch:
        raise ValueError("fewer training clips than a full batch; lower --batch")
    val_idx = np.random.default_rng(args.seed).choice(val_idx, min(512, len(val_idx)), replace=False).tolist()
    loader_options = {"prefetch_factor": 3} if args.workers else {}
    batch_sampler = RestartableBatchSampler(len(tr_idx), args.batch, args.seed)
    train_loader = make_loco_dataloader(torch.utils.data.Subset(ds, tr_idx), batch_size=1, shuffle=False,
        batch_sampler=batch_sampler, num_workers=args.workers, pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed + 91000), **loader_options)
    val_loader = make_loco_dataloader(torch.utils.data.Subset(ds, val_idx), batch_size=args.batch,
        num_workers=args.workers // 2, shuffle=False, generator=torch.Generator().manual_seed(args.seed + 92000))
    print(f"[s2] dataset {len(ds):,} clips ({len(tr_idx):,} train); render fingerprint {ds.attrs.get('render_fingerprint')}")

    cfg = json.loads((args.match_run / "config.json").read_text()) if args.match_run else model_config()
    model = build_model(cfg).to(device)
    sigreg = SIGReg(knots=17, num_proj=1024).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = batch_sampler.steps_per_epoch
    total = args.max_steps or (1000 if args.smoke else args.epochs * steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(total, 1)))))
    print(f"[s2] model {n_params / 1e6:.1f}M params; {steps_per_epoch} steps/epoch; total {total} steps")
    store = None if args.no_upload else HFStore()
    storage_budget = None
    if store is not None:
        state_bytes = sum(v.numel() * v.element_size() for v in model.state_dict().values())
        parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        checkpoint_bytes = 3 * state_bytes + 2 * parameter_bytes + 1_000_000
        storage_budget = checkpoint_storage_budget(private_storage_profile(store), total_steps=total,
            push_every=args.push_every, checkpoint_bytes=checkpoint_bytes,
            model_runs=2 if args.name == "lewm-a" and not args.smoke else 1)
        (run_dir / "storage_budget.json").write_text(json.dumps(storage_budget, indent=2) + "\n")
        require_storage_budget(storage_budget)
        print(f"[s2] private checkpoint storage projection {storage_budget['projected_history_bytes'] / 1e9:.2f} GB; remaining after reserve {storage_budget['remaining_after_reserve_bytes'] / 1e9:.2f} GB")
    wandb = None
    if not args.no_wandb and os.environ.get("WANDB_API_KEY"):
        import wandb as _wandb

        wandb = _wandb
        wandb.init(project="safetydial-walker2d", name=run_id, config={**vars(args), "n_params": n_params, "steps_per_epoch": steps_per_epoch})
    from omegaconf import OmegaConf

    resolved = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "total_steps": total, "model": cfg, "episode_splits": episode_splits, "data_order_version": 1, "source_hashes": source_hashes}
    data_sha256 = file_sha256(args.data)
    training_identity = {"data_sha256": data_sha256, "model": cfg, "episode_splits": episode_splits,
        "scaler": [np.asarray(a).tolist() for a in scaler], "total_steps": total, "data_order_version": 1,
        "recipe": {k: getattr(args, k) for k in ("batch", "lr", "wd", "warmup", "seed", "clip_stride")}}
    OmegaConf.save(OmegaConf.create(resolved), run_dir / "config.yaml")
    (run_dir / "splits.json").write_text(json.dumps(episode_splits, indent=2) + "\n")
    data_receipt = Path(args.data).parent / "upload_receipt.json"
    upstream = json.loads(data_receipt.read_text()) if data_receipt.is_file() else {}
    manifest_base = dict(run_id=run_id, kind=args.name, seeds={"seed": args.seed}, data={"path": args.data, "sha256": data_sha256, "n_clips": len(ds), "episode_splits": episode_splits, "normalisation_role": "training", "source_hashes": source_hashes, "render_fingerprint": ds.attrs.get("render_fingerprint"), "frameskip": 10, "history": 3}, upstream_revisions=upstream)

    step, t0, hist = 0, time.time(), []
    if args.resume_from:
        step, hist = restore_training_state(args.resume_from, model=model, optimizer=opt, scheduler=sched, identity=training_identity)
        manifest_base["upstream_revisions"] = {**upstream, "resume_bundle": str(args.resume_from), "resume_sha256": file_sha256(args.resume_from), "resume_step": step}
        print(f"[s2] restored atomic checkpoint at step {step}/{total}; writing fresh run {run_id}")
    initial_step = step
    black_checked = False
    render_validation = None
    done = False
    while not done:
        batch_sampler.set_step(step)
        for batch in train_loader:
            if not black_checked:
                px = batch["pixels"]
                print(f"[s2] first batch pixels {tuple(px.shape)} {px.dtype} mean {float(px.float().mean()):.1f} (black frames would be ~0)")
                pixel_mean = float(px.float().mean())
                frame_std = px.float().flatten(start_dim=2).std(dim=-1)
                assert pixel_mean > 5.0 and float(frame_std.min()) > 1.0, "black/constant frames: EGL context problem"
                render_validation = {"pixel_mean": pixel_mean, "min_frame_std": float(frame_std.min()), "shape": list(px.shape), "render_fingerprint": ds.attrs.get("render_fingerprint"), "passed": True}
                black_checked = True
            b = preprocess_batch(batch, scaler[0], scaler[1], device)
            model.train()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = lewm_loss(model, sigreg, b)
            if not torch.isfinite(out["loss"]):
                raise RuntimeError(f"nonfinite training loss at step {step}")
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            sched.step()
            step += 1
            if step % 20 == 0:
                rec = {"step": step, "loss": float(out["loss"].detach()), "pred_loss": float(out["pred_loss"]), "sigreg_loss": float(out["sigreg_loss"]), "lr": sched.get_last_lr()[0], "it_per_s": (step - initial_step) / (time.time() - t0)}
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
                if not math.isfinite(val_pred):
                    raise RuntimeError("nonfinite validation loss")
                if wandb:
                    wandb.log({"val_pred_loss": val_pred}, step=step)
                save_training_state(run_dir / "optimizer.pt", model=model, optimizer=opt, scheduler=sched,
                    step=step, identity=training_identity, history=hist)
                save_checkpoint(model, run_dir, cfg=cfg, scaler=scaler, step=step, extra={"val_pred_loss": val_pred, "history": hist[-50:], "recovery_bundle": "optimizer.pt", "recovery_sha256": file_sha256(run_dir / "optimizer.pt")})
                write_manifest(run_dir, build_manifest(**manifest_base, metrics={"step": step, "val_pred_loss": val_pred, "train_loss": hist[-1]["loss"] if hist else None}, started_at=t_start))
                (run_dir / "README.md").write_text(f"# {run_id}\n\nLeWM trained from scratch on {Path(args.data).name} (frameskip 10, history 3, 224 px). Step {step}. Not a released checkpoint. See manifest.json.\n")
                print(f"[s2] step {step}: val pred loss {val_pred:.4f}; checkpoint saved")
                if store is not None:
                    revision = upload_checkpoint(store, "walker2d", args.name, run_dir, run_id=run_id, final=step >= total)
                    (run_dir / "upload_receipt.json").write_text(json.dumps({"repo_id": store.repo_id("walker2d"), "revision": revision, "path": f"{args.name}/{run_id}", "step": step}, indent=2) + "\n")
            if step >= total:
                done = True
                break
    out_json = REPO_ROOT / "docs" / "mainPlan" / "results" / "s2"
    out_json.mkdir(parents=True, exist_ok=True)
    receipt = json.loads((run_dir / "upload_receipt.json").read_text()) if store else None
    (out_json / f"{run_id}.json").write_text(json.dumps({"run_id": run_id, "steps": step, "history": hist, "wall_clock_s": time.time() - t_start, "n_clips": len(ds), "n_params": n_params, "episode_splits": episode_splits, "checkpoint": receipt, "render_validation": render_validation, "steps_per_epoch": steps_per_epoch, "storage_budget": storage_budget}, indent=1) + "\n")
    if wandb:
        wandb.finish()
    print(f"[s2] done: {step} steps in {(time.time() - t_start) / 60:.1f} min -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
