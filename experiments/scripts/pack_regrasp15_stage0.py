"""Pack the task 7 query records that stage 0 decodes on the rental.

Selects task 7 policy queries inside the closure gate: every record whose executed
chunk exceeded 0.06 m/s near the jar plus a seeded random sample of the others, capped
at 720 records (task 15 plan, stage 0). For each record it gathers the stored policy
inputs, the stored noise and chunks, and the measured arm state, hand and jar positions
and fitted base yaw that the knock check needs. Read-only on the preserved task 7 data;
no simulator or model call.
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/helpers"))
from regrasp15_check import CheckConfig, features  # noqa: E402
from regrasp15_kinematics import fit_base  # noqa: E402

TASK7 = ROOT / "outputs/dsrl07/remote-final"
GATED = ROOT / "outputs/regrasp15/design/gated_queries.json"
CAP = 720
FAST = 0.06
SEED = 15


def select(rows):
    fast = [r for r in rows if r["max_near_speed_mps"] > FAST]
    rest = [r for r in rows if r["max_near_speed_mps"] <= FAST]
    order = np.random.default_rng(SEED).permutation(len(rest))
    chosen = fast + [rest[i] for i in sorted(order[: max(0, CAP - len(fast))])]
    reasons = ["fast"] * len(fast) + ["random"] * (len(chosen) - len(fast))
    return chosen[:CAP], reasons[:CAP]


def query_index(folder):
    """Map control step to query file for one task 7 job folder."""
    index = {}
    for path in sorted(folder.glob("query_*.npz")):
        with np.load(path) as z:
            meta = json.loads(str(z["meta"]))
        if meta.get("status") == "returned":
            if meta["step"] in index:
                raise ValueError(f"Two returned queries at step {meta['step']} in {folder}")
            index[meta["step"]] = path
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/regrasp15/stage0/records.npz")
    args = parser.parse_args()
    rows = [r for r in json.loads(GATED.read_text()) if r["source"] == "dsrl07"]
    chosen, reasons = select(rows)
    by_leaf = defaultdict(list)
    for i, row in enumerate(chosen):
        by_leaf[row["leaf"]].append(i)
    arrays = defaultdict(lambda: [None] * len(chosen))
    manifest = [None] * len(chosen)
    config = CheckConfig(mode="select", speed_limit_mps=1.0, candidates=1)
    for leaf, members in sorted(by_leaf.items()):
        recording = next((TASK7 / "episodes" / leaf / "recordings").glob("task_*_base.npz"))
        d = np.load(recording)
        q = d["q"][:, :7].astype(np.float64)
        eef = d["eef_pos"].astype(np.float64)
        idx = np.linspace(0, len(q) - 1, 200).astype(int)
        yaw, _, error = fit_base(q[idx], eef[idx])
        if error > 1e-3:
            raise ValueError(f"Base fit error {error:.4f} m in {leaf}")
        steps = d["proposal_step"].astype(int)
        index = query_index(TASK7 / "queries" / leaf.split(".")[0])
        for i in members:
            row = chosen[i]
            step = row["step"]
            k = int(np.flatnonzero(steps == step)[0])
            path = index[step]
            with np.load(path) as z:
                record = {name: z[name] for name in z.files}
            executed = d["proposal_chunk"][k]
            if not np.allclose(executed, record["actions"], atol=1e-6):
                raise ValueError(f"Recorded chunk differs from query record at {leaf} {step}")
            feat = features(executed, q[step], eef[step], d["jar_pos"][step], yaw,
                            config.near_radius_m)
            if abs(feat["max_near_speed_mps"] - row["max_near_speed_mps"]) > 1e-9:
                raise ValueError(f"Near speed differs from calibration at {leaf} {step}")
            values = {
                "image_left": record["observation_image_left"],
                "wrist_image": record["observation_wrist_image"],
                "obs_state": record["observation_state"],
                "noise": record["noise"],
                "actions_native": record["actions_native"],
                "actions": record["actions"],
                "q_now": q[step],
                "eef_now": eef[step],
                "jar_pos": d["jar_pos"][step].astype(np.float64),
                "yaw": np.float64(yaw),
            }
            for name, value in values.items():
                arrays[name][i] = np.asarray(value)
            meta = json.loads(str(record["meta"]))
            manifest[i] = {
                "row": i,
                "reason": reasons[i],
                "leaf": leaf,
                "arm": row["arm"],
                "task": row["task"],
                "step": step,
                "since_gate": row["since_gate"],
                "tip_within_2s": row["tip_within_2s"],
                "executed_near_speed_mps": row["max_near_speed_mps"],
                "query_file": str(path.relative_to(ROOT)),
                "query_mode": meta["mode"],
                "noise_origin": meta.get("noise_origin"),
                "prompt": str(record["prompt"]),
            }
    stacked = {name: np.stack(values) for name, values in arrays.items()}
    stacked["prompt"] = np.asarray([m["prompt"] for m in manifest])
    stacked["manifest"] = np.asarray(json.dumps(manifest))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **stacked)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    summary = {
        "records": len(manifest),
        "fast": reasons.count("fast"),
        "random": reasons.count("random"),
        "arms": {a: sum(m["arm"] == a for m in manifest) for a in sorted({m["arm"] for m in manifest})},
        "tasks": {t: sum(m["task"] == t for m in manifest) for t in sorted({m["task"] for m in manifest})},
        "out": str(args.out.relative_to(ROOT)),
        "sha256": digest,
        "bytes": args.out.stat().st_size,
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
