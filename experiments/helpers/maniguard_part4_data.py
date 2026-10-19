"""Verified exact donors and causal first-upright-event warning windows.

Numerical object state is an offline audit target only. Predictor inputs are built
from a separate explicit allowlist, ending before the forecasted physical event.
"""

import fnmatch
import json
from pathlib import Path

import numpy as np

from helpers.maniguard_part4_protocol import CONFIG, atomic, assert_partitioning, digest, read


def warning_labels(upright, feature_count, horizon=8, history=4):
    upright = np.asarray(upright, dtype=bool)
    if upright.ndim != 1 or len(upright) != feature_count + 1:
        raise ValueError("Require one initial and one post-action observation per control")
    if horizon < 1 or history < 0:
        raise ValueError("Invalid warning window")
    bad = np.flatnonzero(~upright)
    onset = int(bad[0]) if len(bad) else None
    steps = np.arange(history, min(feature_count, len(upright) - horizon), dtype=int)
    if onset is not None:
        steps = steps[steps < onset]
    y = np.zeros(len(steps), dtype=int)
    if onset is not None:
        y = ((steps < onset) & (onset <= steps + horizon)).astype(int)
    return {"steps": steps, "y": y, "onset": onset,
            "initial_violation": onset == 0,
            "censored_tail_inputs": min(horizon - 1, feature_count)}


def native_agreement(trace, propositions):
    props = {k: v for k, v in propositions.items() if v.get("state") == "upright"}
    if not props:
        raise ValueError("No native uprightness proposition")
    observed = {r["step"]: r["monitor"] for r in trace if r["type"] == "observed"}
    measured = {r["step"]: r for r in trace if r["type"] == "measurement"}
    if set(observed) != set(measured) or sorted(observed) != list(range(len(observed))):
        raise ValueError("Missing aligned native/numerical observations")
    upright, event_objects, checks = [], [], 0
    for step in sorted(observed):
        monitor = observed[step]
        if not monitor.get("valid") or monitor.get("error"):
            raise ValueError("Invalid native monitor record")
        ap = monitor["observation"]["ap"]
        objects = measured[step]["protected_objects"]
        bad_names = []
        for name, prop in props.items():
            selected = [o for o in objects if any(
                fnmatch.fnmatchcase(o["name"], pattern) for pattern in prop["over"]
            )]
            if not selected:
                raise ValueError("Missing protected scope for " + name)
            for obj in selected:
                numbers = [obj[k] for k in ("angle_deg", "threshold_deg", "margin_deg")]
                if not np.isfinite(numbers).all():
                    raise ValueError("Nonfinite physical target")
                if obj["threshold_deg"] != prop["params"]["max_tilt_deg"]:
                    raise ValueError("Native and numerical threshold differ")
                if not np.isclose(numbers[1] - numbers[0], numbers[2], atol=1e-6):
                    raise ValueError("Inconsistent uprightness margin")
            numerical = all(o["margin_deg"] >= 0 for o in selected)
            if numerical != ap[name]:
                raise ValueError(f"Native/numerical uprightness disagree at {step}: {name}")
            bad_names += [o["name"] for o in selected if o["margin_deg"] < 0]
            checks += 1
        upright.append(all(ap[name] for name in props))
        event_objects.append(sorted(set(bad_names)))
    return np.asarray(upright), event_objects, checks


def load_donors(run):
    run = Path(run)
    protocol = read(CONFIG / "offline-protocol.json")
    assert_partitioning(protocol["partitions"])
    restored = read(run / "donor-restore.json")
    if not restored.get("archive_download_hash_verified") or not restored.get("private"):
        raise ValueError("Verified private donor restore required")
    if restored["revision"] != protocol["donors"]["revision"]:
        raise ValueError("Unexpected donor revision")
    donor_root = run / "donors"
    expected = {r["path"]: r["sha256"] for r in restored["files"]}
    for name, sha in expected.items():
        if digest(donor_root / name) != sha:
            raise ValueError("Restored donor changed: " + name)
    features = donor_root / "part3/features"
    corpus = read(features / "corpus.json")
    jobs = {}
    for job_path in donor_root.glob("**/data/job.json"):
        job = read(job_path)
        if job["episode_id"] in jobs:
            raise ValueError("Duplicate donor episode")
        jobs[job["episode_id"]] = (job, job_path.parent)
    metadata = read(run / "benchmark-metadata/manifest.json")
    props = {}
    for item in metadata["files"]:
        path = run / "benchmark-metadata" / (item["task"] + "-diagnostics.jsonl")
        if digest(path) != item["sha256"]:
            raise ValueError("Native proposition source changed")
        props[item["task"]] = json.loads(path.read_text().splitlines()[0])[
            "ltl_safety"]["propositions"]
    episodes, inventory = [], []
    for item in corpus["episodes"]:
        identifier = item["episode_id"]
        job, output = jobs[identifier]
        audit = read(output.parent / "audit.json")
        if not audit.get("valid") or not audit.get("full_episode"):
            raise ValueError("Invalid or incomplete historical donor")
        base = item["scene"].split("/")[0]
        groups = [g for g, tasks in protocol["partitions"].items() if base in tasks]
        if len(groups) != 1:
            raise ValueError("Donor lacks unique whole-task partition")
        trace = [json.loads(line) for line in (output / "trace.jsonl").read_text().splitlines()]
        upright, objects, checks = native_agreement(trace, props[base])
        feature_path = features / (identifier + ".npz")
        with np.load(feature_path, allow_pickle=False) as value:
            arrays = {k: value[k].copy() for k in ("visual", "robot", "action", "proposed", "query")}
        n = len(arrays["visual"])
        if n != audit["steps"] or any(len(a) != n for a in arrays.values()):
            raise ValueError("Feature/control count differs")
        proposed = {r["step"]: r for r in trace if r["type"] == "proposal"}
        executed = {r["step"]: r for r in trace if r["type"] == "executed"}
        queries = {r["step"] for r in trace if r["type"] == "query"}
        for t in range(n):
            frame = proposed[t]["frame"]
            for actual, source in (
                (arrays["robot"][t], frame["joints"] + frame["fingers"]),
                (arrays["proposed"][t], proposed[t]["original"]),
                (arrays["action"][t], executed[t]["action"]),
            ):
                if not np.array_equal(actual, np.asarray(source, np.float32)):
                    raise ValueError("Exact causal feature/action alignment failed")
            if arrays["query"][t] != (t in queries):
                raise ValueError("Query age alignment failed")
        label = warning_labels(upright, n, protocol["horizon_actions"], protocol["history_actions"])
        cfg = job["controller"]
        age, last = [], None
        for t in range(n):
            if t in queries:
                last = t
            if last is None:
                raise ValueError("No initial policy query")
            age.append(t - last)
        controller = []
        for key in ("free_step_limit", "grasp_step_limit", "correction_threshold"):
            controller.extend([float(cfg[key] is not None), float(cfg[key] or 0)])
        controller.extend([cfg["minimum_interval"], job["effective_horizon"], 1.0])
        episode = {**arrays, **label, "episode_id": identifier, "task": base,
                   "scene": item["scene"], "group": groups[0],
                   "controller": np.asarray(controller, dtype=np.float64),
                   "query_age": np.asarray(age), "audit": audit,
                   "event_objects": objects[label["onset"]] if label["onset"] is not None else []}
        if any(not np.isfinite(a).all() for a in arrays.values()):
            raise ValueError("Nonfinite predictor input")
        episodes.append(episode)
        inventory.append({"episode_id": identifier, "task": base, "group": groups[0],
                          "steps": n, "first_upright_event": label["onset"],
                          "event_objects": episode["event_objects"],
                          "initial_violation": label["initial_violation"],
                          "eligible_windows": len(label["steps"]),
                          "positive_windows": int(label["y"].sum()),
                          "native_numeric_checks": checks,
                          "feature_sha256": digest(feature_path),
                          "trace_sha256": digest(output / "trace.jsonl"),
                          "audit_sha256": digest(output.parent / "audit.json")})
    atomic(run / "donor-audit.json", {
        "schema": "maniguard-part4-causal-donor-audit-v1", "episodes": inventory,
        "protocol_sha256": digest(CONFIG / "offline-protocol.json"),
        "native_numeric_agreement": True, "new_physical_controls": 0,
        "initializations": 0, "new_encoder_calls": 0,
        "scope": "Historical exposed development evidence; no Part4 fresh audit opened",
    })
    return episodes


def causal_design(episode, pca, visual_only=False, history=4):
    visual = pca.transform(episode["visual"])
    if visual_only:
        return visual[episode["steps"]]
    rows = []
    for t in episode["steps"]:
        if t < history:
            raise ValueError("Insufficient causal history")
        # Current observation plus four preceding observations. The command at t
        # exists already; actual environment inputs stop strictly at t-1.
        rows.append(np.concatenate([
            visual[t-history:t+1].ravel(), episode["robot"][t-history:t+1].ravel(),
            episode["proposed"][t], episode["action"][t-history:t].ravel(),
            episode["controller"], [episode["query_age"][t]],
        ]))
    width = (history + 1) * (pca.n_components_ + 9) + 8 + history * 8 + 10
    return np.asarray(rows).reshape(-1, width)
