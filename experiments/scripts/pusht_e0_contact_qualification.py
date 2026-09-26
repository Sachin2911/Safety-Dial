#!/usr/bin/env python3
"""Bounded CPU qualification of body-specific contact at the 50 archived E0 roots.

Replays one fixed stored suffix per root three times. Records both last-prefix-block
and final-prefix-step contact definitions. Does not change the archived bank/report.
"""
from __future__ import annotations

import argparse
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
from helpers.pushtReplay import StepLedger


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bank", type=Path, required=True)
    parser.add_argument("--replay-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require_new_paths([args.out])
    source = Bank(args.source_bank)
    if len(source.roots) != 50:
        raise ValueError("This qualification is predeclared for the 50 archived E0 roots")
    report = {"status": "running", "contact_counter": CONTACT_COUNTER,
              "source_bank": str(args.source_bank), "original_replay_report": str(args.replay_report),
              "source_sha256": {name: digest(args.source_bank / name) for name in ("roots.json", "branches.h5")},
              "replay_report_sha256": digest(args.replay_report),
              "protocol": {"repeats": 3, "root_count": 50, "suffix": "first stored branch at each root",
                           "mid_contact_definition": "at least one pusher-T contact in the final five prefix steps",
                           "strict_root_contact_definition": "pusher-T contact during the final prefix step",
                           "simulator_only": True, "world_model_queries": 0}, "roots": []}
    ledger = StepLedger()
    env = MeteredEnv(make_env(), ledger, "typed_prefix_and_suffix_replay")
    started = time.perf_counter()
    try:
        for index, root in enumerate(source.roots):
            branch_indices = source.indices_for_root(index)
            if not len(branch_indices):
                raise ValueError(f"Root {root.root_id} has no archived suffix")
            branch_index = int(branch_indices[0])
            branch = source.branch(branch_index)
            tape = branch["tape"].astype(float).reshape(-1, 2)
            contexts, logs = [], []
            for _ in range(3):
                context, log = execute_branch(env, root, tape)
                contexts.append(context)
                logs.append(log)
                ledger.add("typed_prefix_and_suffix_replay", 0, branches=1)
            prefix = contexts[0].prefix_log
            prefix_check = compare_replays([context.prefix_log for context in contexts])
            suffix_check = compare_replays(logs)
            typed_equal = all(np.array_equal(getattr(logs[0], field), getattr(log, field))
                              and np.array_equal(getattr(prefix, field), getattr(context.prefix_log, field))
                              for field in ("pusher_block_contacts", "block_wall_contacts")
                              for context, log in zip(contexts[1:], logs[1:]))
            mid = bool((prefix.pusher_block_contacts[-5:] > 0).any())
            root_step = bool(prefix.pusher_block_contacts[-1] > 0)
            wall = bool((prefix.block_wall_contacts[-5:] > 0).any())
            row = {"root_id": root.root_id, "source_branch_index": branch_index, "branch_kind": branch["kind"],
                   "source_episode": root.meta.get("episode"), "prefix_steps": len(root.prefix),
                   "pusher_block_last_prefix_block": mid, "pusher_block_final_prefix_step": root_step,
                   "block_wall_last_prefix_block": wall, "wall_only_last_prefix_block": wall and not mid,
                   "any_collision_last_prefix_block": bool((prefix.n_contacts[-5:] > 0).any()),
                   "prefix_pusher_contact_steps": int((prefix.pusher_block_contacts > 0).sum()),
                   "prefix_wall_contact_steps": int((prefix.block_wall_contacts > 0).sum()),
                   "suffix_pusher_contact_steps": int((logs[0].pusher_block_contacts[logs[0].observed] > 0).sum()),
                   "suffix_wall_contact_steps": int((logs[0].block_wall_contacts[logs[0].observed] > 0).sum()),
                   "prefix_replay": prefix_check, "suffix_replay": suffix_check, "typed_counts_bitwise": bool(typed_equal),
                   "deterministic": bool(prefix_check["bitwise"] and prefix_check["frames_equal"] and
                                         suffix_check["bitwise"] and suffix_check["frames_equal"] and typed_equal)}
            report["roots"].append(row)
            if (index + 1) % 10 == 0:
                print(f"[contact-qualification] {index + 1}/50 roots, {ledger.total} env steps", flush=True)
        rows = report["roots"]
        keys = ("pusher_block_last_prefix_block", "pusher_block_final_prefix_step", "block_wall_last_prefix_block",
                "wall_only_last_prefix_block", "any_collision_last_prefix_block")
        report["counts"] = {key: sum(row[key] for row in rows) for key in keys}
        report["mid_pusher_contact_deterministic_root_ids"] = [row["root_id"] for row in rows if row["pusher_block_last_prefix_block"] and row["deterministic"]]
        report["strict_pusher_contact_deterministic_root_ids"] = [row["root_id"] for row in rows if row["pusher_block_final_prefix_step"] and row["deterministic"]]
        report["all_replays_bitwise"] = all(row["deterministic"] for row in rows)
        report["gate"] = {"passes": report["all_replays_bitwise"] and len(report["mid_pusher_contact_deterministic_root_ids"]) >= 15,
                          "requirement": "50 roots with three bitwise repeats and at least 15 roots contacting the pusher in the last prefix block"}
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="incomplete", error_type=type(exc).__name__, gate={"passes": False})
        raise
    finally:
        env.close()
        source.h5.close()
        report["ledger"] = ledger.to_dict()
        report["wall_clock_seconds"] = time.perf_counter() - started
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    print(json.dumps({"counts": report["counts"], "gate": report["gate"], "seconds": report["wall_clock_seconds"]}, indent=1))
    return 0 if report["gate"]["passes"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
