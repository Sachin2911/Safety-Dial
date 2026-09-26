#!/usr/bin/env python3
"""Add one recorded-action contact root without changing the original E0 qualification."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import numpy as np

from helpers.acquisitionSafety import MeteredEnv, require_new_paths
from helpers.branchBank import Bank
from helpers.e0Validation import compare_replays
from helpers.pushtContactReplay import CONTACT_COUNTER, execute_branch, make_env
from helpers.pushtReplay import Root, StepLedger


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bank", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require_new_paths([args.out])
    original = json.loads(args.qualification.read_text())
    if original.get("status") != "complete" or original.get("all_replays_bitwise") is not True or len(original["roots"]) != 50:
        raise ValueError("Supplement requires the complete deterministic 50-root qualification")
    if any(digest(args.source_bank / name) != sha for name, sha in original["source_sha256"].items()):
        raise ValueError("The archived source bank changed since contact qualification")
    source = Bank(args.source_bank)
    source_index = next(i for i, root in enumerate(source.roots) if root.root_id == "e0-r001")
    parent = source.roots[source_index]
    if parent.root_id in original["mid_pusher_contact_deterministic_root_ids"]:
        raise ValueError("This supplement is intended to cover an additional source case")
    branch_index = int(source.indices_for_root(source_index)[0])
    tape = source.branch(branch_index)["tape"].astype(float).reshape(-1, 2)
    root = Root.from_dict(parent.to_dict())
    root.root_id = "e0-r001-recorded-contact-extension-10"
    root.prefix = np.concatenate([parent.prefix, tape[:10]])
    root.meta = {"episode": parent.meta["episode"], "source_root_id": parent.root_id,
                 "source_branch_index": branch_index, "recorded_prefix_extension_steps": 10,
                 "purpose": "infrastructure contact/replay validation only"}
    ledger = StepLedger()
    env = MeteredEnv(make_env(), ledger, "supplement_prefix_and_suffix_replay")
    started = time.perf_counter()
    contexts, logs = [], []
    try:
        for _ in range(3):
            ctx, log = execute_branch(env, root, tape)
            contexts.append(ctx)
            logs.append(log)
            ledger.add("supplement_prefix_and_suffix_replay", 0, branches=1)
    finally:
        env.close()
        source.h5.close()
    prefix = contexts[0].prefix_log
    before, after = compare_replays([ctx.prefix_log for ctx in contexts]), compare_replays(logs)
    typed = all(np.array_equal(getattr(prefix, field), getattr(ctx.prefix_log, field)) and
                np.array_equal(getattr(logs[0], field), getattr(log, field))
                for field in ("pusher_block_contacts", "block_wall_contacts")
                for ctx, log in zip(contexts[1:], logs[1:]))
    deterministic = before["bitwise"] and before["frames_equal"] and after["bitwise"] and after["frames_equal"] and typed
    mid, strict = bool((prefix.pusher_block_contacts[-5:] > 0).any()), bool(prefix.pusher_block_contacts[-1] > 0)
    root.meta.update(state_at_root=contexts[0].state.tolist(), contact_kind="pusher_block", contact_counter=CONTACT_COUNTER,
                     in_contact_last_block=mid, contact_at_final_prefix_step=strict)
    combined_ids = original["mid_pusher_contact_deterministic_root_ids"] + ([root.root_id] if mid and deterministic else [])
    combined_costs = copy.deepcopy(original["ledger"])
    for key, steps in ledger.counts.items():
        combined_costs["steps"][key] = steps
    for key, count in ledger.branches.items():
        combined_costs["branches"][key] = count
    combined_costs["total_steps"] += ledger.total
    report = {"status": "complete", "original_qualification": str(args.qualification),
              "original_qualification_sha256": digest(args.qualification),
              "original_counts": original["counts"], "original_gate": original["gate"],
              "augmentation_reason": "The original any-collision count included wall contacts; add one recorded-action root to retain the predeclared 15 pusher-contact coverage target",
              "supplement": {"root": root.to_dict(), "repeats": 3, "suffix_actions": tape.tolist(),
                  "suffix_policy": "same stored 25-action tape replayed from the extended recorded prefix",
                  "prefix_replay": before, "suffix_replay": after, "typed_counts_bitwise": bool(typed),
                  "pusher_block_last_prefix_block": mid, "pusher_block_final_prefix_step": strict,
                  "prefix_pusher_contact_points": prefix.pusher_block_contacts.tolist(),
                  "prefix_wall_contact_points": prefix.block_wall_contacts.tolist(),
                  "ledger": ledger.to_dict(), "world_model_queries": 0},
              "n_roots": 51, "n_unique_source_episodes": len({row["source_episode"] for row in original["roots"]}),
              "n_mid_pusher_contact_deterministic_roots": len(combined_ids),
              "mid_pusher_contact_deterministic_root_ids": combined_ids,
              "n_strict_pusher_contact_deterministic_roots": len(original["strict_pusher_contact_deterministic_root_ids"]) + int(strict and deterministic),
              "gate": {"passes": bool(deterministic and len(combined_ids) >= 15),
                       "requirement": "at least 50 roots, with three bitwise repeats and at least 15 pusher-contact roots in the last prefix block"},
              "combined_ledger": combined_costs,
              "interpretation": "51 reset-and-prefix roots from the same 50 source episodes; infrastructure replay coverage, not 51 statistically independent trajectories",
              "supplement_wall_clock_seconds": time.perf_counter() - started}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"n_roots": 51, "pusher_contact_roots": len(combined_ids), "strict_pusher_contact_roots": report["n_strict_pusher_contact_deterministic_roots"],
                      "supplement_steps": ledger.total, "gate": report["gate"]}, indent=1))
    return 0 if report["gate"]["passes"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
