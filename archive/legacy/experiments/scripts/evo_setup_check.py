#!/usr/bin/env python3
"""Stage 0 of the evolution study (docs/evoPlan/README.md): pinned assets and S4 numbers.

Downloads the Walker LeWM, probes, state data, S4 banks and policy ladder at the revisions
in configs/evo/inputs.yaml, checks that each asset loads, then recomputes the S4 imagined
health clearances of the saved no-update evaluation rows with the S4 recipe written out:
render a root's three history frames, encode, `WalkerImaginer.rollout` the saved tape, read
the probe at every block end, prepend the root readout, interpolate to environment steps
and take the minimum clearance. Passing proves the assets and preprocessing are the ones
the earlier study used (tolerance and banks declared in configs/evo/stage0.yaml).

    uv run python experiments/scripts/evo_setup_check.py [--run-id ID] [--no-upload]
"""

from __future__ import annotations

import argparse
import json
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

apply_torch()

from helpers.evoInputs import EvoRun, fetch_inputs, input_revisions, load_stage_config  # noqa: E402
from helpers.locoData import RenderContext  # noqa: E402
from helpers.locoEnv import make_loco_env  # noqa: E402
from helpers.poseProbes import load_probe  # noqa: E402
from helpers.runManifest import build_manifest, file_sha256  # noqa: E402
from helpers.walkerAssets import load_exported_actor  # noqa: E402
from helpers.walkerBank import WalkerBank, interp_steps  # noqa: E402
from helpers.walkerLewm import FRAMESKIP, WalkerImaginer, load_walker_model  # noqa: E402
from helpers.walkerReporting import cumulative_clearance  # noqa: E402
from helpers.walkerRules import HORIZON_BLOCKS, health_clearance, rule_unsafe  # noqa: E402
from helpers.walkerValidation import load_source_split, verify_render_fingerprint  # noqa: E402


def imagined_rows(bank: WalkerBank, im: WalkerImaginer, probe, ctx: RenderContext) -> dict:
    """(bank, branch) -> imagined health clearance (min and per-block running min)."""
    out = {}
    for ri, root in enumerate(bank.roots):
        idx = bank.indices_for_root(ri)
        if len(idx) == 0:
            continue
        z_hist = im.encode(ctx.render_many(root.history_qpos, root.history_qvel))
        tapes = np.stack([bank.h5["tape"][int(j)] for j in idx]).astype(np.float64)
        z_imag = im.rollout(z_hist, root.history_actions, tapes.reshape(len(idx), HORIZON_BLOCKS, FRAMESKIP, 6))
        n, k, d = z_imag.shape
        pred = probe.predict(z_imag.reshape(-1, d)).reshape(n, k, 3)
        y0 = probe.predict(z_hist[-1:])[0]
        for a, j in enumerate(idx):
            ends = np.vstack([y0, pred[a]])
            clearance = health_clearance(interp_steps(ends[:, 0]), interp_steps(ends[:, 1]))
            out[(bank.dir.name, int(j))] = {"cmin": float(clearance.min()), "by_block": cumulative_clearance(clearance)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    cfg = load_stage_config("stage0")
    paths = fetch_inputs(cfg.inputs)
    run = EvoRun.create(cfg, "s0", args.run_id)
    print(f"[evo-s0] run {run.run_id}", flush=True)
    checks = {}

    # ---- identities: the saved rows, the source bank and the split --------------------
    rows_path = REPO_ROOT / cfg.check.rows_file
    checks["rows_sha256_matches"] = file_sha256(rows_path) == cfg.check.rows_sha256
    roots_h5 = paths["data"] / "roots.h5"
    checks["roots_h5_is_bank_source"] = file_sha256(roots_h5) == cfg.check.banks_source_sha256
    load_source_split(paths["data"])
    checks["source_split_protocol_v2"] = True

    # ---- every asset loads ---------------------------------------------------------------
    model, scaler = load_walker_model(paths["model"], "cuda")
    im = WalkerImaginer(model, scaler, "cuda")
    probe, _ = load_probe(paths["probes"] / "walker_mlp.pt", "cuda")
    env = make_loco_env("Walker2d", "v1", render=False, terminate_when_unhealthy=False, six_tuple=True)
    env.reset(seed=0)
    policies = sorted(paths["policies"].glob("*.pt"))
    for p in policies:
        actor = load_exported_actor(p, env.action_space)
        actor.reset()
        a = actor.act(np.zeros(env.observation_space.shape, dtype=np.float64))
        if np.shape(a) != (6,) or not np.isfinite(a).all():
            raise ValueError(f"policy {p.name} did not produce a finite 6-d action")
    env.close()
    checks["policies_loaded"] = len(policies)
    checks["policies_count_ok"] = len(policies) == cfg.check.expected_policies
    ctx = RenderContext()
    checks["render_fingerprint"] = verify_render_fingerprint(ctx, roots_h5)
    banks_dir = paths["banks"]
    acquisition = json.loads((banks_dir / "acquisition_roots.json").read_text())
    checks["acquisition_roots"] = len(acquisition["roots"])

    # ---- reproduce the saved imagined health clearances ------------------------------------
    saved = json.loads(rows_path.read_text())
    rule = cfg.check.rule
    per_bank = {}
    worst = 0.0
    all_rows_ok = True
    for name in cfg.check.banks:
        bank = WalkerBank(banks_dir / name)
        got = imagined_rows(bank, im, probe, ctx)
        bank.h5.close()
        rows = saved[name]
        diffs, block_diffs, bitwise, decisions = [], [], 0, 0
        missing = 0
        for row in rows:
            key = (row["bank"], int(row["branch"]))
            if key not in got:
                missing += 1
                continue
            ref, new = row[f"cmin_imagined_{rule}"], got[key]["cmin"]
            diffs.append(abs(new - ref))
            block_diffs.append(float(np.max(np.abs(np.asarray(got[key]["by_block"]) - np.asarray(row[f"cmin_by_block_imagined_{rule}"])))))
            bitwise += int(new == ref)
            decisions += int(bool(rule_unsafe(rule, new)) == bool(rule_unsafe(rule, ref)))
        n = len(rows)
        max_diff = max(diffs) if diffs else float("nan")
        ok = missing == 0 and len(got) == n and decisions == n and max_diff <= cfg.check.atol
        all_rows_ok &= ok
        worst = max(worst, max_diff)
        per_bank[name] = {"n_rows": n, "n_reproduced": len(diffs), "missing": missing, "max_abs_diff": max_diff,
                          "mean_abs_diff": float(np.mean(diffs)) if diffs else float("nan"),
                          "max_abs_diff_by_block": max(block_diffs) if block_diffs else float("nan"),
                          "bitwise_equal": bitwise, "decisions_identical": decisions, "pass": bool(ok)}
        print(f"[evo-s0] {name}: {len(diffs)}/{n} rows, max |diff| {max_diff:.3e}, bitwise {bitwise}, decisions {decisions}/{n}", flush=True)
    ctx.close()

    identity_ok = all(checks[k] for k in ("rows_sha256_matches", "roots_h5_is_bank_source", "policies_count_ok"))
    passed = bool(identity_ok and all_rows_ok)
    report = {"run_id": run.run_id, "stage": 0, "pass": passed, "atol": cfg.check.atol, "rule": rule,
              "checks": checks, "reproduction": per_bank, "max_abs_diff": worst,
              "wall_clock_s": time.time() - t0}
    run.write_json("check.json", report)
    manifest = build_manifest(run_id=run.run_id, kind="evo-s0", seeds={}, data={"rows_file": cfg.check.rows_file},
                              upstream_revisions=input_revisions(paths), metrics={"pass": passed, "max_abs_diff": worst},
                              costs={"real_steps": 0, "imagined_rows": int(sum(b["n_rows"] for b in per_bank.values()) * HORIZON_BLOCKS)},
                              started_at=t0)
    run.finish(manifest, upload=not args.no_upload,
               readme=f"# {run.run_id}\n\nStage 0 of docs/evoPlan: pinned inputs load and the saved S4 imagined health clearances are reproduced.\n")
    print(f"[evo-s0] {'PASS' if passed else 'FAIL'} in {time.time() - t0:.0f}s", flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
