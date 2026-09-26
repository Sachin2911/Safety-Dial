"""Scientific gates and source roles shared by the new Walker study.

Gating uses development episodes only. A readout that accepts every unsafe branch
cannot pass merely because most examples are safe.
"""

import numpy as np


def episode_role(episode: int) -> str:
    return ("development", "test", "acquisition", "acquisition")[episode % 4]


def decision_diagnostics(clearance, unsafe) -> dict:
    c, u = np.asarray(clearance), np.asarray(unsafe, dtype=bool)
    predicted_unsafe = c < 0
    nu, ns = int(u.sum()), int((~u).sum())
    tp = int((predicted_unsafe & u).sum())
    tn = int((~predicted_unsafe & ~u).sum())
    return {"n_unsafe": nu, "n_safe": ns, "unsafe_detected": tp,
            "safe_accepted": tn, "unsafe_recall": tp / nu if nu else None,
            "specificity": tn / ns if ns else None}


def gate_decision(probes: dict, errors: dict, diagnostics: dict, *,
                  min_unsafe: int = 10, min_recall: float = 0.8,
                  min_specificity: float = 0.8) -> dict:
    """Provisional thresholds fixed before new development outcomes are collected."""
    tracked = {}
    for rule in ("health", "speed"):
        d = diagnostics[rule]
        tracked[rule] = bool(d["n_unsafe"] >= min_unsafe and d["n_safe"] >= min_unsafe
                             and d["unsafe_recall"] is not None
                             and d["unsafe_recall"] >= min_recall
                             and d["specificity"] is not None
                             and d["specificity"] >= min_specificity)
    finite = {k: bool(len(errors[k]) > 1 and np.isfinite(errors[k]).all())
              for k in ("height", "pitch", "speed")}
    grows = {k: bool(finite[k] and errors[k][-1] > errors[k][0]) for k in finite}
    useful = {"health": bool(finite["height"] and finite["pitch"]
                             and errors["height"][-1] < 0.3 and errors["pitch"][-1] < 0.6),
              "speed": bool(finite["speed"] and errors["speed"][-1] < 1.0)}
    height = bool(probes["height_r2"] >= 0.9)
    pitch = bool(probes["pitch_r2"] >= 0.9)
    speed = bool(probes["speed_r2"] >= 0.8)
    health_ok = height and pitch and useful["health"] and grows["height"] and grows["pitch"] and tracked["health"]
    speed_ok = speed and useful["speed"] and grows["speed"] and tracked["speed"]
    return {"height_pass": height, "pitch_pass": pitch, "speed_pass": speed,
            "imagined_error_grows": grows, "imagined_error_useful": useful,
            "real_readout_tracks_truth": tracked, "health_rule_ok": bool(health_ok),
            "speed_rule_ok": bool(speed_ok), "go": bool(health_ok),
            "active_rules": (["health"] + (["speed"] if speed_ok else [])) if health_ok else [],
            "thresholds": {"min_unsafe": min_unsafe, "min_recall": min_recall,
                           "min_specificity": min_specificity},
            "status": "pass" if health_ok else "no_go"}


def load_source_split(data_dir):
    """Require the fresh, recorded episode split before consuming any model or frames."""
    import json
    from pathlib import Path

    path = Path(data_dir) / "splits.json"
    split = json.loads(path.read_text())
    if split.get("protocol_version") != 2:
        raise ValueError("A fresh protocol-v2 source run is required; historical roots were consumed during smoke gating")
    groups = split["roots"]
    seen = set()
    for role in ("development", "test", "acquisition"):
        ids = groups[role]
        if len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError("Source roles overlap or repeat episodes")
        if any(episode_role(e) != role for e in ids):
            raise ValueError("Source split differs from the fixed episode-role protocol")
        seen.update(ids)
    if not seen or seen != set(range(max(seen) + 1)):
        raise ValueError("Source split must cover every recorded root episode")
    return split


def decomposition_gate(rows, rules, *, min_unsafe=10, min_false_safe=5,
                       min_imagination=3, min_share=0.25):
    """Require measurable imagination error on development tapes before final test.

    Counts and the share threshold are provisional, recorded before collection.
    Attribution identifies the first earlier source that already accepts the unsafe tape.
    """
    from helpers.walkerRules import rule_unsafe

    evidence = {}
    for rule in rules:
        u = rule_unsafe(rule, np.array([row[f"cmin_dense_{rule}"] for row in rows]))
        accepted = {source: np.array([row[f"cmin_{source}_{rule}"] >= 0 for row in rows])
                    for source in ("endpoint", "real_readout", "imagined")}
        fs = u & accepted["imagined"]
        imagination = fs & ~accepted["endpoint"] & ~accepted["real_readout"]
        n_false_safe, n_imagination = int(fs.sum()), int(imagination.sum())
        share = n_imagination / n_false_safe if n_false_safe else None
        passed = bool(int(u.sum()) >= min_unsafe and n_false_safe >= min_false_safe
                      and n_imagination >= min_imagination and share is not None and share >= min_share)
        evidence[rule] = {"n": len(rows), "n_unsafe": int(u.sum()), "n_false_safe": n_false_safe,
                          "n_imagination": n_imagination, "imagination_share": share, "pass": passed}
    repair_rules = [rule for rule in rules if evidence[rule]["pass"]]
    return {"go": bool(repair_rules), "repair_rules": repair_rules, "evidence": evidence,
            "role": "development", "thresholds": {"min_unsafe": min_unsafe,
            "min_false_safe": min_false_safe, "min_imagination": min_imagination,
            "min_share": min_share}, "status": "pass" if repair_rules else "diagnostic_stop"}


def upload_reference(run_dir):
    """Read a pinned remote receipt when present, without ever resolving 'latest'."""
    import json
    from pathlib import Path

    for name in ("hf_upload.json", "upload_receipt.json"):
        path = Path(run_dir) / name
        if path.is_file():
            record = json.loads(path.read_text())
            return {k: record[k] for k in ("repo_id", "revision", "path") if k in record}
    return None


def verify_render_fingerprint(ctx, source_path):
    """Reject a changed renderer before mixing saved frames and newly rendered roots."""
    import h5py
    from helpers.locoData import render_fingerprint

    with h5py.File(source_path, "r") as source:
        expected = source.attrs.get("render_fingerprint")
    if not expected or render_fingerprint(ctx) != expected:
        raise ValueError("Walker evaluation render fingerprint differs from the source collection")
    return expected
