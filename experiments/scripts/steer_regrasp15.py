"""Stage 0 of task 15: can pi0.5's own resamples avoid fast pushes near the jar?

For each packed task 7 query inside the closure gate, replay the stored noise through
the ordinary Policy.infer path and through the batched candidate path and report both
differences from the stored chunk, then decode K fresh latents on the same observation.
Each candidate's predicted near-jar hand speed is checked at 0.15, 0.10 and 0.06 m/s.
The rescue rate at a limit is the fraction of records whose executed chunk fails that
have at least one passing candidate. Frozen rule: K = 16 for the live stages if the
rescue rate at 0.10 m/s is at least 0.5, otherwise K = 32. Model only; no simulator.
Resumable: rows already in the output are skipped.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/helpers"))
from regrasp15_check import features, joint_distance  # noqa: E402
from regrasp15_sampler import CandidateSampler, candidate_noise, load_policy  # noqa: E402

LIMITS = (0.15, 0.10, 0.06)
RULE_LIMIT = 0.10
RULE_RESCUE = 0.5
NEAR_RADIUS = 0.22


def observation(records, i):
    return {
        "observation/image_left": records["image_left"][i],
        "observation/wrist_image": records["wrist_image"][i],
        "observation/state": records["obs_state"][i],
        "prompt": str(records["prompt"][i]),
    }


def evaluate(records, i, sampler, policy, k):
    obs = observation(records, i)
    stored = records["actions_native"][i]
    saved = policy._rng
    start = time.monotonic()
    replay = np.asarray(policy.infer(dict(obs), noise=records["noise"][i])["actions"])
    replay_s = time.monotonic() - start
    policy._rng = saved
    latents = np.concatenate([records["noise"][i][None], candidate_noise(i, k)])
    chunks, decode_s = sampler.decode(obs, latents)
    executed = records["actions"][i]
    state = (records["q_now"][i], records["eef_now"][i], records["jar_pos"][i], records["yaw"][i])
    pool = [executed, *chunks[1:]]
    speeds = [features(c, *state, NEAR_RADIUS)["max_near_speed_mps"] for c in pool]
    row = {
        "row": i,
        "identity_infer_max_abs": float(np.abs(replay - stored).max()),
        "identity_batch_max_abs": float(np.abs(chunks[0] - stored).max()),
        "executed_near_speed_mps": speeds[0],
        "candidate_near_speed_mps": speeds[1:],
        "candidate_joint_rms": [joint_distance(c, executed) for c in chunks[1:]],
        "candidate_gripper_open": [bool((c[:8, 7] > 0).all()) for c in chunks[1:]],
        "replay_s": replay_s,
        "decode_s": decode_s,
        "noise_seed": i,
    }
    return row, chunks


def summarize(rows, manifest, k):
    summary = {"records": len(rows), "k": k, "limits": {}}
    for limit in LIMITS:
        failing = [r for r in rows if r["executed_near_speed_mps"] > limit]
        rescued = [r for r in failing if min(r["candidate_near_speed_mps"]) <= limit]
        pass_fraction = [np.mean(np.array(r["candidate_near_speed_mps"]) <= limit) for r in rows]
        closest = []
        for r in rescued:
            ok = [d for d, s in zip(r["candidate_joint_rms"], r["candidate_near_speed_mps"])
                  if s <= limit]
            closest.append(min(ok))
        by_arm = {}
        for arm in sorted({m["arm"] for m in manifest}):
            members = [r for r in failing if manifest[r["row"]]["arm"] == arm]
            by_arm[arm] = {
                "failing": len(members),
                "rescued": sum(min(r["candidate_near_speed_mps"]) <= limit for r in members),
            }
        summary["limits"][f"{limit:.2f}"] = {
            "executed_failing": len(failing),
            "rescued": len(rescued),
            "rescue_rate": len(rescued) / len(failing) if failing else None,
            "mean_pass_fraction_all": float(np.mean(pass_fraction)),
            "mean_pass_fraction_failing": float(
                np.mean([np.mean(np.array(r["candidate_near_speed_mps"]) <= limit)
                         for r in failing])
            ) if failing else None,
            "median_closest_passing_joint_rms": float(np.median(closest)) if closest else None,
            "by_arm": by_arm,
        }
    rule = summary["limits"][f"{RULE_LIMIT:.2f}"]["rescue_rate"]
    summary["live_k"] = 16 if rule is not None and rule >= RULE_RESCUE else 32
    summary["identity_infer_max_abs"] = max(r["identity_infer_max_abs"] for r in rows)
    summary["identity_batch_max_abs"] = max(r["identity_batch_max_abs"] for r in rows)
    summary["identity_infer_median_abs"] = float(
        np.median([r["identity_infer_max_abs"] for r in rows]))
    summary["identity_batch_median_abs"] = float(
        np.median([r["identity_batch_max_abs"] for r in rows]))
    summary["median_decode_s"] = float(np.median([r["decode_s"] for r in rows]))
    summary["median_replay_s"] = float(np.median([r["replay_s"] for r in rows]))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path,
                        default=ROOT / "outputs/regrasp15/stage0/records.npz")
    parser.add_argument("--sha256", required=True, help="expected records file digest")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "third_party/checkpoints/jar/7400")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/regrasp15/stage0")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--limit", type=int, help="first N records only (smoke test)")
    args = parser.parse_args()
    if hashlib.sha256(args.records.read_bytes()).hexdigest() != args.sha256:
        raise SystemExit("Stage 0 records differ from the frozen digest")
    records = dict(np.load(args.records))
    manifest = json.loads(str(records["manifest"]))
    count = len(manifest) if args.limit is None else min(args.limit, len(manifest))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "chunks").mkdir(exist_ok=True)
    results = args.out / "steer.jsonl"
    done = set()
    if results.exists():
        done = {json.loads(line)["row"] for line in results.read_text().splitlines() if line}
    start = time.monotonic()
    policy = load_policy(args.checkpoint)
    sampler = CandidateSampler(policy)
    print(json.dumps({"event": "policy_loaded", "seconds": time.monotonic() - start}), flush=True)
    with results.open("a") as handle:
        for i in range(count):
            if i in done:
                continue
            row, chunks = evaluate(records, i, sampler, policy, args.k)
            np.savez_compressed(args.out / "chunks" / f"row_{i:04d}.npz", chunks=chunks,
                                noise_seed=np.int64(i))
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            if i % 20 == 0:
                print(json.dumps({"event": "progress", "row": i,
                                  "identity_infer": row["identity_infer_max_abs"],
                                  "identity_batch": row["identity_batch_max_abs"],
                                  "decode_s": row["decode_s"]}), flush=True)
    rows = [json.loads(line) for line in results.read_text().splitlines() if line]
    rows = sorted((r for r in rows if r["row"] < count), key=lambda r: r["row"])
    summary = summarize(rows, manifest, args.k)
    summary["records_sha256"] = args.sha256
    summary["complete"] = len(rows) == len(manifest)
    name = "summary.json" if summary["complete"] else f"summary-first{len(rows)}.json"
    (args.out / name).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "summary", **{k: v for k, v in summary.items() if k != "limits"},
                      "rescue": {k: v["rescue_rate"] for k, v in summary["limits"].items()}}),
          flush=True)


if __name__ == "__main__":
    main()
