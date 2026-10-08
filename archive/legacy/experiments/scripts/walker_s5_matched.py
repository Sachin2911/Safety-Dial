#!/usr/bin/env python3
"""S5: replace exactly N ordinary training transitions with S4 random-arm experience.

Reuse the actually acquired branches at the largest additional budget. No new simulator
queries or root-history prefixes are hidden in the replacement count. Validation source
episodes remain unchanged; partial ordinary episodes retain a contiguous prefix.
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

from helpers.threads import pin_threads

pin_threads()

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

from helpers.hfStore import HFStore
from helpers.locoCollect import SCHEMA, StateHDF5Writer
from helpers.locoEnv import make_loco_env
from helpers.runManifest import build_manifest, make_run_id, write_manifest
from helpers.walkerProtocol import exact_replacement, file_sha256


def acquired_rows(bank_dir, seed, selected_ids):
    """Locate only selected additional branches; final bank metadata defines row order."""
    found = {}
    for path in sorted(bank_dir.glob(f"random_s{seed}_b*")):
        meta = json.loads((path / "roots.json").read_text())
        ids = meta["meta"]["selected_ids"]
        with h5py.File(path / "branches.h5", "r") as f:
            if len(ids) != len(f["tape"]):
                raise ValueError(f"branch/ID count mismatch in {path}")
        for row, candidate_id in enumerate(ids):
            if candidate_id in selected_ids:
                if candidate_id in found:
                    raise ValueError(f"duplicate acquired branch {candidate_id}")
                found[candidate_id] = (path, row)
    if set(found) != set(selected_ids):
        raise ValueError("selected random experience is incomplete on disk")
    return found


def branch_columns(path, row, candidate, root, env, episode_idx):
    """Copy the measured transitions, preserving the collector's pre-action state schema."""
    with h5py.File(path / "branches.h5", "r") as f:
        qp, qv = f["qpos"][row], f["qvel"][row]
        tape, velocity = f["tape"][row], f["x_velocity"][row]
        root_index = int(f["root_index"][row])
    bank_meta = json.loads((path / "roots.json").read_text())
    stored_root = bank_meta["roots"][root_index]
    for key in ("root_id", "episode", "step"):
        if stored_root[key] != root[key]:
            raise ValueError(f"S4 branch root identity mismatch: {key}")
    for key in ("qpos", "qvel", "history_qpos", "history_qvel", "history_actions"):
        np.testing.assert_array_equal(stored_root[key], root[key])
    np.testing.assert_array_equal(qp[0], root["qpos"])
    np.testing.assert_array_equal(qv[0], root["qvel"])
    if int(root["episode"]) % 4 not in (2, 3):
        raise ValueError("S5 branch is not from the acquisition role")
    np.testing.assert_allclose(tape, candidate["tape"], rtol=0, atol=1e-7)
    n = len(tape)
    obs = []
    for p, v in zip(qp[:-1], qv[:-1]):
        env.unwrapped.set_state(p, v)
        obs.append(env.unwrapped._get_obs())
    healthy = ((qp[:-1, 1] > 0.8) & (qp[:-1, 1] < 2.) & (np.abs(qp[:-1, 2]) < 1.)).astype("u1")
    truncated = np.zeros(n, "u1")
    truncated[-1] = 1
    # Branch banks do not contain rewards. Mark them missing rather than invent values;
    # LeWM trains on the state-rendered frames and actions only.
    columns = {"qpos": qp[:-1], "qvel": qv[:-1], "action": tape,
               "observation": np.asarray(obs), "x_velocity": velocity,
               "reward": np.full(n, np.nan), "cost": (velocity > 2.3415).astype("u1"),
               "healthy": healthy, "terminated": np.zeros(n, "u1"), "truncated": truncated,
               "episode_idx": np.full(n, episode_idx), "step_idx": np.arange(n),
               "policy_id": np.full(n, 200)}
    return {k: np.asarray(v, dtype=SCHEMA[k][0]) for k, v in columns.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--acquisition-bank", type=Path, required=True)
    ap.add_argument("--model-a", type=Path, required=True, help="completed LeWM-A run with splits.json")
    ap.add_argument("--out", type=Path, required=True, help="fresh set-B directory")
    ap.add_argument("--seed", type=int, default=0, help="paired S4 acquisition seed")
    ap.add_argument("--replacement-seed", type=int, default=20261027)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--n", type=int, default=1)
    args = ap.parse_args()
    started = time.time()
    if args.out.exists():
        ap.error("--out must be a fresh directory")
    selection_path = args.acquisition_bank / f"random_s{args.seed}_selection.json"
    selection = json.loads(selection_path.read_text())
    if not selection.get("selection_complete"):
        raise ValueError("S4 largest planned budget has not completed")
    if selection["budget"] != max(selection["planned_budgets"]):
        raise ValueError("S5 requires the largest planned additional budget")
    roots_file = (args.acquisition_bank / selection["root_file"]).resolve()
    if not roots_file.is_relative_to(args.acquisition_bank.resolve()):
        raise ValueError("root_file must be inside the S4 bank")
    roots = json.loads(roots_file.read_text())["roots"]
    if file_sha256(args.data_dir / "roots.h5") != selection["source_identity"]["sha256"]:
        raise ValueError("S4 root source differs from the specified data run")
    selected = selection["selected_candidates"]
    ids = [c["id"] for c in selected]
    if len(set(ids)) != len(ids) or len(ids) != selection["budget"]:
        raise ValueError("candidate budget or identity mismatch")
    if set(ids) & {c["id"] for c in selection["seed_candidates"]}:
        raise ValueError("additional data includes common seed branches")
    steps = sum(len(c["tape"]) for c in selected)
    if steps != selection["branch_steps"]:
        raise ValueError("S4 transition budget does not match stored tapes")
    locations = acquired_rows(args.acquisition_bank, args.seed, ids)
    a_manifest = json.loads((args.model_a / "manifest.json").read_text())
    if a_manifest["data"]["sha256"] != file_sha256(args.data_dir / "setA.h5"):
        raise ValueError("LeWM-A was trained on a different ordinary dataset")
    roles = json.loads((args.model_a / "splits.json").read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    run_id = make_run_id("walker2d", "data", "matched-b", n=args.n)
    env = make_loco_env("Walker2d", "v1", render=False, terminate_when_unhealthy=False)
    env.reset(seed=0)
    new_roles = {"training": [], "validation": []}
    origins = []
    with h5py.File(args.data_dir / "setA.h5", "r") as source:
        lengths, offsets = source["ep_len"][:], source["ep_offset"][:]
        kept = exact_replacement(lengths, roles["training"], steps, seed=args.replacement_seed)
        dims = {k: source[k].shape[1] for k in ("qpos", "qvel", "action", "observation")}
        attrs = {**dict(source.attrs), "set": "B", "run_id": run_id,
                 "replacement_steps": steps, "branch_rewards": "missing (NaN)",
                 "diagnostic_continuations": True}
        with StateHDF5Writer(args.out / "setB.h5", dims, attrs) as writer:
            for ep, count in enumerate(kept):
                if count == 0:
                    continue
                start, end = int(offsets[ep]), int(offsets[ep] + count)
                columns = {k: source[k][start:end] for k in SCHEMA}
                new_ep = len(writer.ep_lens)
                columns["episode_idx"][:] = new_ep
                if count != lengths[ep]:
                    columns["terminated"][-1] = 0
                    columns["truncated"][-1] = 1
                writer.write_episode(columns)
                role = "training" if ep in roles["training"] else "validation"
                new_roles[role].append(new_ep)
                origins.append({"episode": new_ep, "source_episode": ep,
                                "rows_kept": int(count), "role": role})
            for candidate in selected:
                path, row = locations[candidate["id"]]
                new_ep = len(writer.ep_lens)
                columns = branch_columns(path, row, candidate, roots[candidate["root_index"]], env, new_ep)
                writer.write_episode(columns)
                new_roles["training"].append(new_ep)
                origins.append({"episode": new_ep, "candidate_id": candidate["id"],
                                "root_index": candidate["root_index"], "role": "training"})
            if writer.n != int(lengths.sum()):
                raise AssertionError("A/B transition counts differ")
            total = writer.n
    env.close()
    (args.out / "splits.json").write_text(json.dumps(new_roles, indent=2) + "\n")
    (args.out / "origins.json").write_text(json.dumps(origins, indent=2) + "\n")
    (args.out / "config.yaml").write_text("set: B\nsource: actual_S4_random_acquisition\n")
    (args.out / "README.md").write_text(f"# {run_id}\n\nExactly {steps} ordinary training transitions replaced by the same S4 random-arm transitions. Validation sources are unchanged. Total rows: {total}. Branch rewards are missing and must not be used as evaluation data. Train B with --match-run pointing to the LeWM-A run and --episode-splits pointing to splits.json; this fixes optimizer count, initial seed, architecture and normalisation.\n")
    metrics = {"ordinary_steps_replaced": steps, "branch_steps_added": steps,
               "total_A_steps": total, "total_B_steps": total, "additional_simulator_steps": 0,
               "original_acquisition_steps": selection.get("charged_steps"),
               "n_branches": len(selected)}
    manifest = build_manifest(run_id=run_id, kind="data", seeds={"acquisition": args.seed, "replacement": args.replacement_seed}, data={"selection_sha256": file_sha256(selection_path), "setA_sha256": a_manifest["data"]["sha256"], "setB_sha256": file_sha256(args.out / "setB.h5"), "model_a": str(args.model_a), "episode_splits": new_roles}, metrics=metrics, started_at=started)
    write_manifest(args.out, manifest)
    receipt = None
    if not args.no_upload:
        store = HFStore()
        revision = store.upload_run("walker2d-data", "data", args.out, run_id=run_id)
        receipt = {"repo_id": store.repo_id("walker2d-data"), "revision": revision, "path": f"data/{run_id}"}
        (args.out / "upload_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    result = REPO_ROOT / "docs/mainPlan/results/s5" / f"{run_id}.json"
    result.parent.mkdir(parents=True, exist_ok=True)
    if result.exists():
        raise FileExistsError(result)
    result.write_text(json.dumps({**metrics, "checkpoint": receipt, "manifest": manifest}, indent=2) + "\n")
    print(f"[s5] done: {steps} rows replaced, {total} rows in both A and B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
