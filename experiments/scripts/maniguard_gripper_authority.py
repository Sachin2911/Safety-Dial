"""Test gripper-noise authority on saved development observations, without physics."""

import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.maniguard_client import StudyClient
from helpers.maniguard_latent_policy import array_hash


def command(values):
    return np.where(abs(values) > .01, np.sign(values), -1.)


def offsets():
    yield "native", None
    yield "same-noise", np.zeros((16, 32), np.float32)
    for amplitude in (.5, 2.):
        for sign in (-1, 1):
            offset = np.zeros((16, 32), np.float32)
            offset[:, 7] = sign * amplitude
            yield f"a{amplitude:g}-s{sign:+d}", offset


def plan(protocol):
    inputs = protocol["observations"]
    if len(inputs) != 8 or {x["family"] for x in inputs} != {"jar", "clutter"}:
        raise ValueError("Require eight saved development inputs across Jar and Clutter")
    if any(sum(x["family"] == family for x in inputs) != 4 for family in ("jar", "clutter")):
        raise ValueError("Require four observations per development family")
    rows = []
    for index, item in enumerate(inputs):
        path = ROOT / item["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("A prospectively pinned saved observation changed")
        with np.load(path, allow_pickle=False) as data:
            if (data["overview"].shape != (640, 640, 3)
                    or data["wrist"].shape != (640, 640, 3) or data["state"].shape != (8,)
                    or data["features"].shape != (44,)):
                raise ValueError("Saved native observation contract differs")
        for label, _ in offsets():
            rows.append({"id": f"o{index:02d}-{label}", "index": index,
                         "family": item["family"], "label": label, "observation": item})
    if len(rows) != protocol["maximum_new_decoder_calls"] or len(rows) != 48:
        raise ValueError("The diagnostic must have exactly 48 prospectively declared calls")
    return rows


def assert_admission(ledger, request_id, maximum=48):
    started = [json.loads(line) for line in ledger.read_text().splitlines()] \
        if ledger.exists() else []
    if any(row["id"] == request_id for row in started):
        raise RuntimeError("Started request unresolved; reconcile it before any repeat")
    if len(started) >= maximum:
        raise RuntimeError("Supplemental decoder allowance exhausted")


def admit_request(protocol, request_id, output, family, protocol_hash):
    """One locked, durable allowance shared by all outputs and family clients."""
    ledger = ROOT / protocol["admission_ledger"]
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if ledger.exists() and not ledger.read_bytes().endswith(b"\n"):
            raise RuntimeError("Incomplete admission tail requires explicit reconciliation")
        prefix = protocol.get("prior_ledger_prefix")
        if prefix and hashlib.sha256(ledger.read_bytes()[:prefix["bytes"]]).hexdigest() != prefix["sha256"]:
            raise ValueError("Already charged request history changed")
        assert_admission(ledger, request_id)
        row = {"id": request_id, "family": family, "started_epoch": time.time(),
               "output": str(output.resolve()), "protocol_sha256": protocol_hash}
        with ledger.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def legacy_protocol_binding(protocol, ledger_bytes, request_id):
    """Bind an inherited request to the exact superseded protocol and prefix."""
    prefix = protocol.get("prior_ledger_prefix", {})
    prior_hash = protocol.get("supersedes_execution_protocol_sha256")
    if not prior_hash or protocol.get("accepted_prior_protocol_sha256") != [prior_hash]:
        raise ValueError("Missing or ambiguous original protocol binding")
    contents = ledger_bytes[:prefix.get("bytes", 0)]
    if (not contents.endswith(b"\n")
            or hashlib.sha256(contents).hexdigest() != prefix.get("sha256")):
        raise ValueError("The inherited admission prefix differs from its frozen binding")
    records = [json.loads(line) for line in contents.splitlines()]
    identifiers = [row["id"] for row in records]
    if (len(records) != prefix.get("requests") or len(identifiers) != len(set(identifiers))
            or identifiers.count(request_id) != 1
            or any("protocol_sha256" in row for row in records)):
        raise ValueError("Legacy request has no unique membership in the frozen original prefix")
    return prior_hash


def reconcile(protocol, request_id, output, server_ledger):
    """Recover saved action bytes or classify a charged return without querying."""
    ledger = ROOT / protocol["admission_ledger"]
    admitted = [json.loads(line) for line in ledger.read_text().splitlines()
                if json.loads(line)["id"] == request_id]
    if len(admitted) != 1 or Path(admitted[0].get("output", ledger.parent)).resolve() != output.resolve():
        raise ValueError("Reconciliation needs one admission for this exact output")
    original_protocol_hash = admitted[0].get("protocol_sha256")
    if original_protocol_hash is None:
        original_protocol_hash = legacy_protocol_binding(protocol, ledger.read_bytes(), request_id)
    planned = next(row for row in plan(protocol) if row["id"] == request_id)
    expected_server = ROOT / protocol["server_ledgers"][planned["family"]]
    if server_ledger.resolve() != expected_server.resolve():
        raise ValueError("Use the declared family server ledger")
    episode = str(output.resolve() / request_id)
    matches = [json.loads(line) for line in server_ledger.read_text().splitlines()
               if json.loads(line).get("episode") == episode]
    if len(matches) != 1:
        raise ValueError("Need exactly one attributable server return")
    server = matches[0]
    result = {"id": request_id, "charged": True, "replay_allowed": False,
              "server_ledger_sha256": hashlib.sha256(server_ledger.read_bytes()).hexdigest(),
              "status": "returned_without_action_bytes" if server["status"] == "returned"
              else "failed_and_charged"}
    actions_path = output / (request_id + ".npz")
    if server["status"] == "returned" and actions_path.exists():
        with np.load(actions_path, allow_pickle=False) as saved:
            if array_hash(saved["actions"]) != server["actions_sha256"]:
                raise ValueError("Interrupted client action bytes differ from server receipt")
        receipt = {**planned, "protocol_sha256": original_protocol_hash,
                   "server_receipt": server,
                   "metadata": json.loads((ledger.parent / (planned["family"] + "-metadata.json")).read_text()),
                   "reconciliation": result}
        target = output / (request_id + ".json")
        with target.open("x") as stream:
            stream.write(json.dumps(receipt, indent=2) + "\n")
        result["status"] = "client_artifact_recovered_without_inference"
    with (ledger.parent / "reconciliation.jsonl").open("a") as stream:
        stream.write(json.dumps(result) + "\n")
    return result


def run(protocol, protocol_hash, family, output, port):
    if protocol["status"] != "authorized_frozen":
        raise RuntimeError("The exhausted original decoder allowance needs explicit new authorization")
    if not protocol.get("authorization", {}).get("user_message"):
        raise RuntimeError("Record the authorization before executing the frozen diagnostic")
    if protocol["script_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("Diagnostic source differs from the prospectively frozen version")
    rows = [row for row in plan(protocol) if row["family"] == family]
    output.mkdir(parents=True, exist_ok=True)
    client = StudyClient(host="127.0.0.1", port=port)
    try:
        metadata = client.get_server_metadata()
        policy = protocol["policies"][family]
        if (metadata.get("audit_service") is not True or metadata.get("model_shape") != [16, 32]
                or metadata.get("checkpoint_revision") != policy["revision"]
                or metadata.get("serve_config") != policy["config"]):
            raise ValueError("Require the matching frozen checkpoint in an audit-only service")
        if any(metadata.get("source_hashes", {}).get(name) != value
               for name, value in protocol["actor_source_hashes"].items()):
            raise ValueError("The diagnostic sampler differs from the verified native-noise hook")
        canonical = (ROOT / protocol["admission_ledger"]).parent
        canonical.mkdir(parents=True, exist_ok=True)
        metadata_path = canonical / (family + "-metadata.json")
        if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
            raise ValueError("The saved diagnostic service metadata changed")
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        for row in rows:
            name = row["id"]
            receipt_path, actions_path = output / (name + ".json"), output / (name + ".npz")
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                with np.load(actions_path, allow_pickle=False) as data:
                    actions = data["actions"]
                if (receipt["protocol_sha256"] not in [protocol_hash, *protocol.get("accepted_prior_protocol_sha256", [])]
                        or array_hash(actions) != receipt["server_receipt"]["actions_sha256"]):
                    raise ValueError("Existing diagnostic evidence differs")
                continue
            with np.load(ROOT / row["observation"]["path"], allow_pickle=False) as data:
                payload = {"observation/image_left": data["overview"],
                           "observation/wrist_image": data["wrist"],
                           "observation/state": data["state"], "prompt": str(data["prompt"]),
                           "latent_features": data["features"],
                           "latent_weights": np.zeros((4, 44), np.float32),
                           "study_episode_id": str(output.resolve() / name),
                           "episode_seed": protocol["noise_seed"] + row["index"]}
            offset = dict(offsets())[row["label"]]
            if offset is not None:
                payload["latent_audit_offset"] = offset
            admit_request(protocol, name, output, family, protocol_hash)
            response = client.infer(payload)
            actions = np.asarray(response["actions"])
            server = response["latent_receipt"]
            if (actions.shape != (16, 8) or not np.isfinite(actions).all()
                    or array_hash(actions) != server["actions_sha256"]
                    or server["seed"] != payload["episode_seed"] or server["query_index"] != 0
                    or server["episode"] != payload["study_episode_id"]
                    or (offset is not None
                        and server.get("audit_offset_sha256") != array_hash(offset))
                    or server["audit"] is not (offset is not None) or server["status"] != "returned"):
                raise ValueError("Invalid returned diagnostic chunk or sampler receipt")
            receipt = {**row, "protocol_sha256": protocol_hash, "server_receipt": server,
                       "metadata": metadata}
            np.savez_compressed(actions_path, actions=actions)
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
            print(json.dumps({"completed": name, "family": family}), flush=True)
    finally:
        client._ws.close()
    # Reconstruct paired results independently from the persisted returned chunks.
    comparisons = []
    for index in sorted({row["index"] for row in rows}):
        reference_path = output / f"o{index:02d}-native.npz"
        with np.load(reference_path, allow_pickle=False) as data:
            reference = data["actions"]
        native = json.loads(reference_path.with_suffix(".json").read_text())["server_receipt"]
        for label, _ in list(offsets())[1:]:
            path = output / f"o{index:02d}-{label}.npz"
            with np.load(path, allow_pickle=False) as data:
                actions = data["actions"]
            receipt = json.loads(path.with_suffix(".json").read_text())["server_receipt"]
            if any(receipt[key] != native[key] for key in
                   ("native_noise_sha256", "rng_before", "rng_after", "features_sha256")):
                raise ValueError("Paired input, native noise or RNG transition differs")
            if label == "same-noise" and not np.allclose(reference, actions, atol=1e-5, rtol=1e-5):
                raise ValueError("The same-noise native identity check failed")
            comparisons.append({"index": index, "label": label,
                                "prefix_gripper_command_changes": int(np.sum(
                                    command(actions[:8, 7]) != command(reference[:8, 7]))),
                                "prefix_arm_rms_rad": float(np.sqrt(np.mean(
                                    (actions[:8, :7] - reference[:8, :7])**2))),
                                "prefix_arm_max_difference_rad": float(np.max(abs(
                                    actions[:8, :7] - reference[:8, :7]))),
                                "raw_gripper_max_difference": float(np.max(abs(
                                    actions[:, 7] - reference[:, 7])))})
    passes = {}
    for amplitude in (.5, 2.):
        count = sum(any(row["index"] == index and row["label"].startswith(f"a{amplitude:g}-")
                        and row["prefix_gripper_command_changes"] > 0 for row in comparisons)
                    for index in sorted({row["index"] for row in rows}))
        passes[str(amplitude)] = {"observations_with_gripper_command_change": count,
                                 "denominator": 4, "influence_criterion_pass": count >= 2}
    summary = {"family": family, "decoder_calls": len(rows), "comparisons": comparisons,
               "same_noise_identity_pass": True, "amplitudes": passes,
               "new_physical_controls": 0, "new_encoder_calls": 0,
               "scope": "Decoder influence on saved development inputs only. No physical benefit, grasp repair, safety or transfer result. No new method comparison is authorized by this summary."}
    (output / (family + "-summary.json")).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "comparisons"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--family", choices=("jar", "clutter"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--port", type=int, default=18002)
    parser.add_argument("--reconcile-request")
    parser.add_argument("--server-ledger", type=Path)
    args = parser.parse_args()
    raw = args.protocol.read_bytes()
    protocol = json.loads(raw)
    rows = plan(protocol)
    if args.plan_only:
        print(json.dumps({"planned_decoder_calls": len(rows), "new_physical_controls": 0,
                          "executed_decoder_calls": 0, "status": protocol["status"]}))
        return
    if args.reconcile_request:
        if args.output is None or args.server_ledger is None:
            parser.error("reconciliation requires --output and --server-ledger")
        print(json.dumps(reconcile(protocol, args.reconcile_request, args.output, args.server_ledger)))
        return
    if args.family is None or args.output is None:
        parser.error("execution requires --family and --output")
    run(protocol, hashlib.sha256(raw).hexdigest(), args.family, args.output, args.port)


if __name__ == "__main__":
    main()
