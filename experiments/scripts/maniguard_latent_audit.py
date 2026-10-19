"""Execute the 320-call paired decoder authority audit on saved development inputs."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.maniguard_client import StudyClient
from helpers.maniguard_latent import CODE_DIM, FEATURE_DIM, arm_basis
from helpers.maniguard_latent_results import read_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, nargs=2, required=True,
                        help="Two full development episodes for the served family")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    client = StudyClient(host="127.0.0.1", port=args.port)
    metadata = client.get_server_metadata()
    if metadata.get("audit_service") is not True or metadata.get("model_shape") != [16, 32]:
        raise ValueError("Decoder probes require the explicitly enabled audit service")
    inputs = []
    for episode in args.episodes:
        read_episode(episode, require_full=True)
        job = json.loads((episode / "job.json").read_text())
        paths = sorted((episode / "query_observations").glob("*.npz"))[:4]
        if len(paths) != 4 or job["policy"]["revision"] != metadata["checkpoint_revision"]:
            raise ValueError("Need four observed native query inputs from each matching full episode")
        for path in paths:
            inputs.append((path, job))
    basis = arm_basis()
    baseline_actions = {}
    for index, (path, job) in enumerate(inputs):
        with np.load(path, allow_pickle=False) as saved:
            observation = {"observation/image_left": saved["overview"],
                           "observation/wrist_image": saved["wrist"],
                           "observation/state": saved["state"], "prompt": str(saved["prompt"]),
                           "latent_features": saved["features"],
                           "latent_weights": np.zeros((CODE_DIM, FEATURE_DIM), dtype=np.float32)}
        for draw in range(2):
            offsets = [("native", None), ("same-noise", np.zeros((16, 32), dtype=np.float32))]
            offsets += [(f"d{direction}-{sign:+d}", np.broadcast_to(
                .35 * sign * basis[:, direction], (16, 32)).copy())
                for direction in range(4) for sign in [-1, 1]]
            for label, offset in offsets:
                name = f"o{index:02d}-n{draw}-{label}"
                receipt_path = args.output / (name + ".json")
                actions_path = args.output / (name + ".npz")
                if receipt_path.exists():
                    receipt = json.loads(receipt_path.read_text())
                    with np.load(actions_path, allow_pickle=False) as data:
                        actions = data["actions"]
                else:
                    # A started-but-unreturned request consumes allowance. It must
                    # be reconciled with the server ledger before any resumption.
                    ledger = args.output / "requests.jsonl"
                    started = [json.loads(line) for line in ledger.read_text().splitlines()] \
                        if ledger.exists() else []
                    if any(row["id"] == name for row in started):
                        raise RuntimeError("Unresolved decoder request; reconcile server receipt: " + name)
                    if len(started) >= 160:
                        raise RuntimeError("Family's 160-call authority allowance exhausted")
                    payload = {**observation, "study_episode_id": str(args.output / name),
                               "episode_seed": 61019000 + 2 * index + draw}
                    if offset is not None:
                        payload["latent_audit_offset"] = offset
                    with ledger.open("a") as stream:
                        stream.write(json.dumps({"id": name, "started_epoch": time.time()}) + "\n")
                    began = time.monotonic()
                    result = client.infer(payload)
                    actions = np.asarray(result["actions"], dtype=np.float32)
                    if actions.shape != (16, 8) or not np.isfinite(actions).all():
                        raise ValueError("Decoder did not return a finite native 16x8 physical chunk")
                    receipt = {"id": name, "source_observation": str(path),
                               "observation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "family": job["family"], "scene": job["scene"], "label": label,
                               "draw": draw, "elapsed_seconds": time.monotonic() - began,
                               "server_receipt": result["latent_receipt"]}
                    np.savez_compressed(actions_path, actions=actions)
                    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
                if label == "native":
                    baseline_actions[(index, draw)] = actions
                else:
                    reference = baseline_actions[(index, draw)]
                    if label == "same-noise" and not np.allclose(
                            reference, actions, atol=1e-5, rtol=1e-5):
                        raise ValueError("Explicit same-noise inference differs from native inference")
                    receipt["prefix_arm_rms_rad"] = float(np.sqrt(np.mean(
                        (actions[:8, :7] - reference[:8, :7])**2)))
                    receipt["full_arm_rms_rad"] = float(np.sqrt(np.mean(
                        (actions[:, :7] - reference[:, :7])**2)))
                    receipt["gripper_max_change"] = float(np.max(abs(actions[:, 7] - reference[:, 7])))
                    receipt["gripper_saturated_fraction"] = float(np.mean(abs(actions[:, 7]) >= .999))
                    def command(values):
                        return np.where(abs(values) > .01, np.sign(values), -1.)

                    receipt["prefix_gripper_command_changes"] = int(np.sum(
                        command(actions[:8, 7]) != command(reference[:8, 7])))
                    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
                print(name, "verified", flush=True)
    rows = [json.loads(p.read_text()) for p in args.output.glob("o*.json")]
    passed = []
    for direction in range(4):
        subset = [r for r in rows if r["label"].startswith(f"d{direction}-")]
        # Two signs and two native noise draws per observation; half of the
        # observation-level means must cross the prospective authority threshold.
        values = [np.mean([r["prefix_arm_rms_rad"] for r in subset
                           if r["id"].startswith(f"o{index:02d}-")]) for index in range(8)]
        passed.append(bool(sum(value >= .001 for value in values) >= 4))
    summary = {"family": job["family"], "decoder_calls": len(rows),
               "directions_passing": passed, "authority_pass": sum(passed) >= 2,
               "same_noise_identity_pass": True,
               "native_resampling_prefix_rms": [float(np.sqrt(np.mean(
                   (baseline_actions[(i, 0)][:8, :7] - baseline_actions[(i, 1)][:8, :7])**2)))
                   for i in range(8)],
               "interpretation": "Decoder influence only; physical benefit and transfer untested"}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
