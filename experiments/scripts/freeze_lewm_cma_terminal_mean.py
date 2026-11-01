"""Freeze all three already-computed secondary CMA terminal centers before final access.

Only trusted own-host checkpoints are loaded. No ask/tell, fitting, sampling,
controller episode, model query or final-case input is performed.
"""

import argparse
import datetime as dt
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import struct

ROOT = Path(__file__).resolve().parents[2]
GROUP = "configs/lewm_campaign_20261031"
NAV = "vukovc3t4o1dui"
REPEATS = {
    0: ("navigation", NAV, "navigation-budget30-comparison-v1"),
    1: ("pusht", "wq4ar2qx2ff6r4", "navigation-budget30-comparison-v1"),
    2: ("recovery", "nj2fqkp2prgk8x", "navigation-budget30-relocated-v1"),
}
ABSENT = (
    "results/navigation-final-cases-v1",
    "results/navigation-final-cases-v1-joint-access.json",
    GROUP + "/navigation-final-cases-v1.json",
    GROUP + "/navigation-joint-final-cases-v2.json",
    GROUP + "/navigation-joint-final-comparison-v2",
)
PROOF = "results/evidence-audits/navigation-terminal-mean-prefinal-absence-v1.json"
MANIFEST = GROUP + "/navigation-budget30-relocated-v1/manifest.json"
MANIFEST_SHA = "6cbf1c890c1451bd79e9702db3f7bbed69d783cb3df6496b3ad44c48d62fbf40"
CONVERSION = "es.gp.pheno(es.mean, into_bounds=es.boundary_handler.repair)"


def require(value, message):
    if not value:
        raise ValueError(message)


def utc():
    return dt.datetime.now(dt.timezone.utc)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text())


def ref(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return ref(path)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def absent_paths():
    require(all(not (ROOT / name).exists() for name in ABSENT), "Final access/metadata already began")
    return [{"path": name, "exists": False} for name in ABSENT]


def host_guard(pod):
    require(
        ROOT.resolve() == Path("/workspace/Safety-Dial")
        and os.environ.get("RUNPOD_POD_ID") == pod,
        "Only the exact owning rented host may freeze this checkpoint",
    )


def absence():
    host_guard(NAV)
    value = {
        "schema": "lewm-terminal-mean-prefinal-absence-v1",
        "pod_id": NAV, "observed_utc": utc().isoformat(),
        "absent_paths": absent_paths(), "extractor_source": ref(Path(__file__).resolve()),
        "final_data_read": 0, "new_model_simulator_calls": 0,
    }
    return write(ROOT / PROOF, value)


def fp32(values):
    return [struct.unpack("!f", struct.pack("!f", value))[0] for value in values]


def extract(repeat):
    host, pod, group = REPEATS[repeat]
    host_guard(pod)
    own_absent = absent_paths()
    proof_path = ROOT / PROOF
    proof = load(proof_path)
    require(
        proof["schema"] == "lewm-terminal-mean-prefinal-absence-v1"
        and proof["pod_id"] == NAV
        and proof["absent_paths"] == own_absent
        and proof["extractor_source"]["sha256"] == sha(Path(__file__))
        and proof["final_data_read"] == proof["new_model_simulator_calls"] == 0,
        "Exact authoritative Nav pre-final absence proof required",
    )
    age = utc() - dt.datetime.fromisoformat(proof["observed_utc"])
    require(dt.timedelta(0) <= age < dt.timedelta(minutes=10), "Require fresh Nav absence proof")
    manifest_path = ROOT / MANIFEST
    require(sha(manifest_path) == MANIFEST_SHA, "Authoritative secondary manifest changed")
    protocol_path = ROOT / GROUP / group / f"training-cma-r{repeat}.json"
    protocol = load(protocol_path)
    manifest = load(manifest_path)
    declared = [item for item in manifest["files"] if Path(item["path"]).name == protocol_path.name]
    require(len(declared) == 1 and sha(protocol_path) == declared[0]["sha256"], "Protocol differs")
    require(
        protocol["method"] == "cma" and protocol["repeat_id"] == repeat
        and protocol["allowed_pod_ids"] == [pod]
        and protocol["evaluation"]["execution_pod_id"] == pod,
        "Wrong method/repeat/physical owner",
    )
    stage = ROOT / protocol["output_namespace"]
    state_path, source_path, checkpoint_path = (
        stage / "state.json", stage / "source-freeze.json", stage / "checkpoint.pkl"
    )
    state, source = load(state_path), load(source_path)
    require(
        state["phase"] == "complete" and state["active"] is None and state["pending"] is None
        and state["optimizer_updates"] == state["generation"] == 6
        and state["candidate_panels_completed"] == 48
        and not state["unfinalized_job_ids"] and state["action_accounting_exact"]
        and source["protocol"] == protocol and source["protocol_sha256"] == canonical(protocol)
        and source["identity"] == state["identity"],
        "Require the immutable completed six-update CMA run",
    )
    updates = []
    for index, history in enumerate(state["history"]):
        path = stage / f"generation-{index:03}.json"
        require(
            load(path) == history and history["generation"] == index
            and history["genuine_evolutionary_update"] is True,
            "Actual completed update record differs",
        )
        updates.append({**ref(path), "generation": index, "completed_updates_after": index + 1})
    require(len(updates) == 6, "Exactly six completed updates required")
    before = sha(checkpoint_path)
    with checkpoint_path.open("rb") as stream:
        checkpoint = pickle.load(stream)  # Only this completed campaign's trusted own checkpoint.
    es = checkpoint["es"]
    require(
        checkpoint["state"] == state and checkpoint["identity"] == state["identity"]
        and es.countiter == 6,
        "Checkpoint is not the final six-tell optimizer object",
    )

    import cma
    import numpy as np

    raw = np.asarray(es.mean, dtype=np.float64).copy()
    bounded = np.asarray(
        es.gp.pheno(raw.copy(), into_bounds=es.boundary_handler.repair), dtype=np.float64
    )
    require(
        raw.shape == bounded.shape == (10,) and np.isfinite(raw).all()
        and np.isfinite(bounded).all() and (bounded >= -6).all() and (bounded <= 6).all()
        and np.array_equal(raw, es.mean) and before == sha(checkpoint_path),
        "Invalid bounded center or checkpoint/mean mutation",
    )
    parameters, raw_mean = bounded.tolist(), raw.tolist()
    effective = fp32(parameters)
    duplicates = []
    for generation, history in enumerate(state["history"]):
        for candidate, vector in enumerate(history["parameters"]):
            if vector == parameters or fp32(vector) == effective:
                duplicates.append({
                    "generation": generation, "candidate_index": candidate,
                    "exact_float64_equal": vector == parameters,
                    "effective_float32_equal": fp32(vector) == effective,
                })
    package_sources = {}
    for label, cls in (("evolution_strategy", type(es)), ("genotype_phenotype", type(es.gp)),
                       ("boundary_handler", type(es.boundary_handler))):
        path = Path(inspect.getfile(cls))
        package_sources[label] = {"file": str(path), "sha256": sha(path)}
    value = {
        "schema": "lewm-secondary-terminal-mean-lock-v1",
        "frozen_utc": utc().isoformat(), "repeat_id": repeat,
        "host": {"host": host, "pod_id": pod},
        "parameters": parameters, "raw_mean": raw_mean, "parameters_float32": effective,
        "optimizer_updates": 6, "conversion": CONVERSION,
        "mapping_is_identity": bool(np.array_equal(raw, bounded)),
        "boundary_handler": type(es.boundary_handler).__module__ + "."
        + type(es.boundary_handler).__name__,
        "cma_package": {"version": cma.__version__, "numpy_version": np.__version__,
                        "source_files": package_sources},
        "sigma": float(es.sigma),
        "checkpoint": {"path": str(checkpoint_path.relative_to(ROOT)), "sha256": before},
        "state": ref(state_path), "source_freeze": ref(source_path),
        "protocol": ref(protocol_path), "update_records": updates,
        "producer_identity": state["identity"],
        "harness_source": ref(ROOT / "experiments/helpers/lewm_campaign_training.py"),
        "authoritative_secondary_manifest": ref(manifest_path),
        "duplicate_flags": {
            "exact_equals_anchor": parameters == state["anchor"]["parameters"],
            "effective_equals_anchor": effective == fp32(state["anchor"]["parameters"]),
            "matches_evaluated_candidates": duplicates,
            "every_repeat_retained_regardless_of_duplicates": True,
        },
        "prefinal_absence": {"navigation_proof": ref(proof_path),
                             "own_host_observed_utc": utc().isoformat(),
                             "absent_paths": absent_paths()},
        "extractor_source": ref(Path(__file__).resolve()),
        "new_optimizer_update_calls": 0, "new_ask_sample_calls": 0,
        "new_model_simulator_calls": 0, "dev_probes": 0, "final_data_read": 0,
        "original_six_selection_locks_still_mandatory": True,
        "original_eleven_profile_comparison_unchanged": True,
        "exploratory_scope": (
            "Prospective supplemental terminal-center study motivated before final access by "
            "development-selection regression and an archive winner from generation0. All three "
            "centers after six completed updates use the same mapping, without mean/repeat "
            "selection, perturbation or development evaluation. Report alongside the unchanged "
            "archive-selected comparison regardless of outcomes. No optimizer-superiority claim."
        ),
    }
    output = ROOT / GROUP / f"navigation-cma-terminal-mean-r{repeat}-v1.json"
    return write(output, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("absence")
    worker = commands.add_parser("extract")
    worker.add_argument("--repeat-id", type=int, choices=tuple(REPEATS), required=True)
    args = parser.parse_args()
    result = absence() if args.command == "absence" else extract(args.repeat_id)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
