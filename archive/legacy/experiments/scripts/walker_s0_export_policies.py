#!/usr/bin/env python3
"""S0: export OmniSafe actor checkpoints to plain MLP state dicts (run in the OmniSafe venv).

    /workspace/omnisafe/.venv/bin/python experiments/scripts/walker_s0_export_policies.py

Writes data/study/walker2d/policies/<algo>-e<epoch>.pt with torch.nn-only contents.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(REPO_ROOT / "runs" / "omnisafe"))
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "study" / "walker2d" / "policies"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for run_dir in sorted(glob.glob(f"{args.runs}/*/seed-*")):
        cfg = json.load(open(Path(run_dir) / "config.json"))
        algo = cfg["exp_name"].split("-")[0] if "exp_name" in cfg else Path(run_dir).parent.name.split("-")[0]
        actor = cfg["model_cfgs"]["actor"]
        for ck in sorted(glob.glob(f"{run_dir}/torch_save/epoch-*.pt"), key=lambda p: int(Path(p).stem.split("-")[1])):
            epoch = int(Path(ck).stem.split("-")[1])
            blob = torch.load(ck, map_location="cpu", weights_only=False)
            sd = {k[len("mean."):]: v.clone() for k, v in blob["pi"].items() if k.startswith("mean.")}
            norm = blob.get("obs_normalizer", {})
            export = {"algo": algo, "epoch": epoch, "hidden_sizes": list(actor["hidden_sizes"]), "activation": actor["activation"],
                      "state_dict": sd, "obs_mean": norm["_mean"].numpy() if "_mean" in norm else None,
                      "obs_var": norm["_var"].numpy() if "_var" in norm else None, "source": ck,
                      "total_steps_at_epoch": epoch * cfg["algo_cfgs"]["steps_per_epoch"]}
            torch.save(export, out / f"{algo}-e{epoch:03d}.pt")
            n += 1
    print(f"[export] wrote {n} policies to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
