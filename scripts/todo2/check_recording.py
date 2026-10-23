"""Check an item 7 recording: one entry per step, the expected fields, scoring unchanged.

    python3 -I scripts/todo2/check_recording.py <episode.npz> <results.jsonl row file> [--reference progress/receipts/01b-smoke.jsonl]

Prints the field table (name, shape, dtype), checks N+1 state rows for N executed steps,
N executed actions, ceil(N/8) proposals, the monitor propositions against the sidecar if
given, and compares the result row's keys and scoring fields to a reference smoke row.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

STATE_FIELDS = ["step", "wall_s", "recorder_s", "q", "dq", "eef_pos", "eef_quat", "gripper_q", "policy_state",
                "jar_pos", "jar_quat", "jar_link_pose", "jar_joint_q", "jar_joint_dq", "jar_lin_vel", "jar_ang_vel",
                "jar_robot_contacts", "jar_finger_contact", "grasping", "other_obj_pos", "ap", "ap_valid",
                "monitor_doomed"]
SCORING = ["scene_name", "seed", "episode_seed", "status", "steps", "success", "success_step", "success_first_step",
           "ltl_violated", "ltl_violation_step", "ltl_violation_count", "first_contact_step", "ever_grasped",
           "grasp_steps", "outcome", "counted_violation", "safety_evaluated"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("row")
    ap.add_argument("--sidecar")
    ap.add_argument("--reference", default="progress/receipts/01b-smoke.jsonl")
    args = ap.parse_args()
    z = np.load(args.npz, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    row = [json.loads(l) for l in open(args.row) if l.strip()][-1]
    n = row["steps"]
    bad = []
    print(f"{'field':24s} {'shape':22s} dtype")
    for k in z.files:
        if k == "meta":
            continue
        print(f"{k:24s} {str(z[k].shape):22s} {z[k].dtype}")
    for k in STATE_FIELDS:
        if k not in z.files:
            bad.append(f"missing {k}")
        elif z[k].shape[0] != n + 1:
            bad.append(f"{k}: {z[k].shape[0]} rows, expected steps + 1 = {n + 1}")
    if "exec_raw" not in z.files or z["exec_raw"].shape[0] != n:
        bad.append(f"exec_raw rows {z['exec_raw'].shape[0] if 'exec_raw' in z.files else None} != steps {n}")
    horizon = meta.get("execute_horizon", 8)
    if "proposal_chunk" not in z.files or z["proposal_chunk"].shape[0] != math.ceil(n / horizon):
        bad.append(f"proposals {z['proposal_chunk'].shape[0] if 'proposal_chunk' in z.files else None} != ceil({n}/{horizon})")
    if not (z["step"] == np.arange(n + 1)).all():
        bad.append("step index is not 0..N")
    if not z["ap_valid"].all():
        bad.append(f"monitor record invalid at {int((~z['ap_valid']).sum())} steps")
    if args.sidecar:
        log = json.load(open(args.sidecar))["log"]
        names = meta["ap_names"]
        side = np.array([[e["ap"][k] for k in names] for e in log], bool)
        if side.shape != z["ap"].shape or not (side == z["ap"]).all():
            bad.append("propositions differ from the sidecar")
        else:
            print(f"propositions match the sidecar at all {len(log)} steps ({names})")
    if os.path.exists(args.reference):
        ref = json.loads(open(args.reference).readline())
        missing = sorted(set(ref) - set(row))
        extra = sorted(set(row) - set(ref))
        print("row keys: missing vs reference", missing, "extra", extra)
        if missing or extra:
            bad.append("row keys differ from the reference smoke row")
        print("scoring fields:")
        for k in SCORING:
            print(f"  {k:22s} this {row.get(k)!r:>14}   reference {ref.get(k)!r}")
    print(f"steps {n}, state rows {z['step'].shape[0]}, executed actions {z['exec_raw'].shape[0]}, "
          f"proposals {z['proposal_chunk'].shape[0]}, file {os.path.getsize(args.npz) / 1e6:.2f} MB, "
          f"recorder overhead {meta.get('recorder_overhead_s', 0):.1f} s "
          f"({1000 * meta.get('recorder_overhead_s', 0) / max(n, 1):.1f} ms per step)")
    print("closed at", int(np.argmax(z["ap"][:, meta["ap_names"].index("jar_closed")])) if z["ap"][:, meta["ap_names"].index("jar_closed")].any() else None,
          "first finger contact", int(np.argmax(z["jar_finger_contact"])) if z["jar_finger_contact"].any() else None,
          "first grasp", int(np.argmax(z["grasping"] == 1)) if (z["grasping"] == 1).any() else None,
          "jar hinge range", float(z["jar_joint_q"].min()), float(z["jar_joint_q"].max()))
    for b in bad:
        print("BAD", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
