"""Calibrate the task 15 knock check on existing recordings, before any new episode.

For every policy query inside the closure gate of the preserved task 4, 5, 7 and 8
recordings, compute the executed native chunk's predicted near-jar hand speed and
whether a tip followed within the next two seconds. Reports the distribution that the
frozen dial values are chosen from. Read-only; no simulator or model call.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/helpers"))
from regrasp15_check import CheckConfig, Gate, features  # noqa: E402
from regrasp15_kinematics import fit_base  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments/scripts"))
from diagnose_regrasp15 import SOURCES, arm_of, benchmark_row  # noqa: E402

HORIZON = 40  # controls after a query in which a tip is attributed to it


def gated_queries(path, source):
    d = np.load(path)
    arm, _ = arm_of(path, source)
    result = benchmark_row(path)
    violation = result.get("ltl_violation_step") if result.get("ltl_violated") else None
    ap = d["ap"].astype(bool)
    q = d["q"][:, :7].astype(np.float64)
    eef = d["eef_pos"].astype(np.float64)
    idx = np.linspace(0, len(q) - 1, 200).astype(int)
    yaw, _, error = fit_base(q[idx], eef[idx])
    if error > 1e-3:
        raise ValueError(f"Base fit error {error:.4f} m")
    config = CheckConfig(mode="select", speed_limit_mps=1.0, candidates=1)
    gate = Gate(config)
    rows = []
    steps = d["proposal_step"].astype(int)
    for k, step in enumerate(steps):
        for t in range(gate.seen, step + 1):
            gate.update(t, ap[t, 0], ap[t, 2], ap[t, 3], ap[t, 1])
        if not gate.open:
            continue
        feat = features(d["proposal_chunk"][k], q[step], eef[step], d["jar_pos"][step], yaw,
                        config.near_radius_m)
        tip_soon = violation is not None and step <= violation <= step + HORIZON
        rows.append({
            "source": source, "arm": arm, "leaf": path.parents[1].name,
            "task": int(path.stem.split("_")[1]), "step": int(step),
            "since_gate": int(step - gate.latched_at),
            "max_near_speed_mps": feat["max_near_speed_mps"],
            "min_horizontal_m": feat["min_horizontal_m"],
            "tip_within_2s": bool(tip_soon),
            "gripper_open_cmd": bool((d["proposal_chunk"][k][:8, 7] > 0).all()),
        })
    return rows, yaw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/regrasp15/design")
    args = parser.parse_args()
    rows, yaws = [], {}
    for source, pattern in SOURCES.items():
        for path in sorted(ROOT.glob(pattern)):
            try:
                found, yaw = gated_queries(path, source)
            except (KeyError, ValueError) as error:
                print("skip", path.parents[1].name, type(error).__name__, error)
                continue
            rows.extend(found)
            yaws[f"{source}/{path.parents[1].name}"] = yaw
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "gated_queries.json").write_text(json.dumps(rows) + "\n")
    speed = np.array([r["max_near_speed_mps"] for r in rows])
    tip = np.array([r["tip_within_2s"] for r in rows])
    native = np.array([r["arm"] == "native" for r in rows])
    print("gated queries", len(rows), "native", int(native.sum()), "tip-preceding", int(tip.sum()))
    for label, mask in (("all", np.ones_like(tip)), ("native", native)):
        s = speed[mask]
        print(label, "near speed percentiles 50/75/90/95/99:",
              np.round(np.percentile(s, [50, 75, 90, 95, 99]), 3))
    print("tip-preceding near speed percentiles 10/25/50/75/90:",
          np.round(np.percentile(speed[tip], [10, 25, 50, 75, 90]), 3))
    print("non-tip near speed percentiles 50/75/90/95:",
          np.round(np.percentile(speed[~tip], [50, 75, 90, 95]), 3))
    for limit in (0.30, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04):
        veto = speed > limit
        print(f"limit {limit:.2f}: native veto rate {veto[native].mean():.3f}, "
              f"tip-preceding vetoed {veto[tip].mean():.3f}, non-tip vetoed {veto[~tip].mean():.3f}")


if __name__ == "__main__":
    main()
