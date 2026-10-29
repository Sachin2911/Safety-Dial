"""Recompute the archived SafeCEM Push-T pilot table from its committed state arrays.

Reads ``results.json``, ``results_states.npz``, ``arena_fix.json`` and
``arena_fix_states.npz`` from the archived results directory, recomputes every
per-episode quantity from the saved simulator states, checks them against the JSON
rows, and prints the per-setting table used in ``results/safecem-preliminary-experiment.tex``.
No simulation or model evaluation is run.

Usage::

    .venv/bin/python -I experiments/scripts/safecem_pilot_verify.py \
        archive/legacy/docs/safeDial/results
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ARENA_LO, ARENA_HI = 0.0, 512.0


def inside(xy: np.ndarray, box: list[float]) -> np.ndarray:
    """Pusher-centre inside the true hazard box, per saved state."""
    return (xy[:, 0] >= box[0]) & (xy[:, 0] <= box[1]) & (xy[:, 1] >= box[2]) & (xy[:, 1] <= box[3])


def out_of_arena(xy: np.ndarray) -> np.ndarray:
    return (xy < ARENA_LO).any(axis=1) | (xy > ARENA_HI).any(axis=1)


def main(root: Path) -> int:
    results = json.loads((root / "results.json").read_text())
    states = np.load(root / "results_states.npz")
    cfg = results["config"]
    box = cfg["hazard_box"]
    goal_block = np.array(cfg["goal_block"])
    start_block = np.array(cfg["start_block"])

    mismatches = 0
    rows: dict[tuple[str, float], list[dict]] = {}
    for arm_name, arm in results["arms"].items():
        for i, ep in enumerate(arm["episodes"]):
            s = states[f"{arm_name}_{i}"]
            pxy, bxy = s[:, 0:2], s[:, 2:4]
            n = len(s)
            n_haz = int(inside(pxy, box).sum())
            n_oob = int(out_of_arena(pxy).sum())
            err = float(np.linalg.norm(bxy[-1] - goal_block))
            moved = float(np.linalg.norm(bxy[-1] - start_block))
            agrees = (
                n_haz == ep["n_violating_steps"]
                and n_oob == ep["oob_steps"]
                and abs(err - ep["final_block_err_px"]) < 1e-6
                and abs(n_haz / n - ep["frac_violating"]) < 1e-9
            )
            mismatches += not agrees
            key = (arm_name, ep["lam"] if arm["key"] == "lam" else ep["dial"])
            rows.setdefault(key, []).append(
                dict(n=n, haz=n_haz, oob=n_oob, err=err, moved=moved,
                     success=bool(ep["success"]), feasible=ep["final_frac_feasible"])
            )
    total = sum(len(arm["episodes"]) for arm in results["arms"].values())
    print(f"main sweep: {total - mismatches}/{total} episodes agree with the JSON rows on "
          "hazard steps, out-of-arena steps, violating fraction and final block error")
    print()
    print(f"{'arm':16}{'setting':>8}{'entries':>9}{'haz/states':>12}{'oob/states':>12}"
          f"{'block err':>16}{'moved':>14}{'success':>9}{'feasible':>10}")
    for (arm_name, setting), eps in rows.items():
        errs = np.array([e["err"] for e in eps])
        moved = np.array([e["moved"] for e in eps])
        n_states = sum(e["n"] for e in eps)
        print(f"{arm_name:16}{setting:>8g}{sum(e['haz'] > 0 for e in eps):>7}/{len(eps)}"
              f"{sum(e['haz'] for e in eps):>8}/{n_states:<4}"
              f"{sum(e['oob'] for e in eps):>7}/{n_states:<4}"
              f"{errs.mean():>9.2f} ± {errs.std():.2f}"
              f"{moved.min():>8.1f}..{moved.max():.1f}"
              f"{sum(e['success'] for e in eps):>7}/{len(eps)}"
              f"{np.mean([e['feasible'] for e in eps]):>10.3f}")
    print()

    arena = json.loads((root / "arena_fix.json").read_text())
    arena_states = np.load(root / "arena_fix_states.npz")
    for variant, block in arena["variants"].items():
        errs, moved, hazards, oobs, counts = [], [], [], [], []
        for i, ep in enumerate(block["episodes"]):
            s = arena_states[f"{variant}_{i}"]
            pxy, bxy = s[:, 0:2], s[:, 2:4]
            err = float(np.linalg.norm(bxy[-1] - goal_block))
            mv = float(np.linalg.norm(bxy[-1] - start_block))
            assert abs(err - ep["final_block_err_px"]) < 1e-6
            assert abs(mv - ep["block_moved_px"]) < 1e-6
            assert abs(out_of_arena(pxy).sum() / len(s) - ep["oob_frac"]) < 1e-9
            errs.append(err)
            moved.append(mv)
            hazards.append(int(inside(pxy, box).sum()))
            oobs.append(int(out_of_arena(pxy).sum()))
            counts.append(len(s))
        per_seed = ", ".join(f"{e:.2f}" for e in errs)
        print(f"arena d=60 {variant:13} hazard {sum(hazards)}/{sum(counts)}  "
              f"oob {sum(oobs)}/{sum(counts)}  block err {np.mean(errs):.2f} ± {np.std(errs):.2f} "
              f"(per seed {per_seed})  moved {', '.join(f'{m:.1f}' for m in moved)}  "
              f"success field present: {'success' in block['episodes'][0]}")
    escape_x = min(arena_states[f"probe_based_{i}"][:, 0].min() for i in range(3))
    escape_y = min(arena_states[f"probe_based_{i}"][:, 1].min() for i in range(3))
    print(f"probe-based escape extremes: x min {escape_x:.0f}, y min {escape_y:.0f}")
    print("states per episode, main sweep:", sorted({len(states[k]) for k in states.files}))
    print("probe stats:", cfg["probe_stats"])
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
