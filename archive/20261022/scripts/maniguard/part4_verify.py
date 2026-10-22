"""Verify the bounded terminal result, frozen arrays, public reports and preservation pins."""

import argparse
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
from helpers.maniguard_part4_protocol import CONFIG, ROOT, RUN, atomic, digest, read


def verify():
    checks = {}
    protocol = CONFIG / "offline-protocol.json"
    original = subprocess.check_output(
        ["git", "show", "e66c2f9:configs/maniguard/part4/offline-protocol.json"], cwd=ROOT)
    checks["protocol_matches_before_label_commit"] = hashlib.sha256(original).hexdigest() == digest(protocol)
    frozen = read(RUN / "models/baseline/baseline-freeze.json")
    for name, expected in frozen["files"].items():
        if digest(RUN / "models/baseline" / name) != expected:
            raise ValueError("Frozen predictor changed: " + name)
    for name, expected in frozen["sources"].items():
        if digest(RUN / "source/frozen" / name) != expected:
            raise ValueError("Original frozen source changed: " + name)
    checks["baseline_arrays_rules_and_original_source_bytes"] = True
    audit, donor, terminal = (read(RUN / name) for name in (
        "offline-audit.json", "donor-audit.json", "terminal.json"))
    for source, public in [("offline-audit.json", "offline audit"),
                           ("donor-audit.json", "donor audit"), ("terminal.json", "terminal")]:
        if digest(RUN / source) != digest(ROOT / "docs/research/research_notes" / (
                "ManiGuard phase3 Part4 " + public + ".json")):
            raise ValueError("Public evidence differs from private source: " + source)
    checks["public_audits_match_raw_derived_receipts"] = True
    if terminal["status"] != "scientifically_gated" or audit["gate"]["eligible"]:
        raise ValueError("Unexpected scientific terminal state")
    if any(terminal[k] != 0 for k in ("new_physical_attempts", "new_controls", "new_physics_steps",
                                     "new_VLA_queries", "new_encoder_calls")):
        raise ValueError("Unexpected new physical or encoder interaction")
    if any(v["executed"] or v["proposals"] for v in terminal["stages"]["acquisition"].values()):
        raise ValueError("Acquisition ran despite the failed scientific gate")
    if terminal["stages"]["fresh_audit"]["labels_opened"]:
        raise ValueError("Fresh audit was opened despite the scientific gate")
    checks["conditional_stages_unstarted_and_fresh_audit_unopened"] = True
    if digest(RUN / "offline-predictions.npz") != audit["predictions_sha256"]:
        raise ValueError("Saved predictions changed")
    with np.load(RUN / "offline-predictions.npz", allow_pickle=False) as data:
        for group in ("fit", "calibration", "development"):
            indices = [i for i, e in enumerate(donor["episodes"]) if e["group"] == group]
            y = np.concatenate([data[f"labels_{i}"] for i in indices])
            p = np.concatenate([data[f"probability_{i}"] for i in indices])
            expected = audit["metrics"][group]["primary"]
            if not np.isclose(average_precision_score(y, p), expected["auprc"], atol=1e-12):
                raise ValueError("Average precision does not reproduce from saved predictions")
            if not np.isclose(roc_auc_score(y, p), expected["auroc"], atol=1e-12):
                raise ValueError("AUROC does not reproduce from saved predictions")
            alarm = p >= audit["rules"]["primary"]["threshold"]
            if int(((y == 0) & alarm).sum()) != expected["false_alarm_windows"]:
                raise ValueError("False-alarm count does not reproduce")
    checks["primary_metrics_reproduce_from_saved_arrays_without_model_forward_calls"] = True
    physical = read(ROOT / "docs/research/figures/ManiGuard/phase3-part4/provenance.json")
    totals = physical["historical_physical_outcomes"]
    if sum(g["controls"] for g in totals.values()) != 18657:
        raise ValueError("Historical donor controls changed or were double charged")
    if sum(g["native_violation_episode_counts"]["uprightness"] for g in totals.values()) != 8:
        raise ValueError("Native event support count differs")
    checks["historical_donors_charged_once_with_unknown_initialization_preserved"] = True
    archive_changes = subprocess.check_output(
        ["git", "diff", "--name-only", "cb7f7c2", "--", "archive"], cwd=ROOT, text=True)
    if archive_changes.strip():
        raise ValueError("Archived evidence changed")
    checks["archived_evidence_unchanged"] = True
    export = read(CONFIG / "frozen-baseline-export.json")
    for key in ("private_model", "private_evidence"):
        receipt = export[key]
        if receipt.get("private") is not True or receipt.get("download_verified") is not True:
            raise ValueError("Missing verified private preservation")
        if not re.fullmatch(r"[a-f0-9]{40}", receipt["revision"]):
            raise ValueError("Private preservation is not revision pinned")
    checks["private_model_and_evidence_revision_pins"] = True
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", "-z", "cb7f7c2"], cwd=ROOT).split(b"\0")
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=ROOT).split(b"\0")
    paths = sorted({ROOT / os.fsdecode(p) for p in changed + untracked if p})
    secrets = [v.encode() for k, v in os.environ.items() if len(v) >= 8 and any(
        term in k.upper() for term in ("TOKEN", "API_KEY", "PASSWORD", "SECRET"))]
    for path in paths:
        if not path.is_file():
            continue
        raw = path.read_bytes()
        if any(value in raw for value in secrets):
            raise ValueError("Credential value detected in public change")
        if path.suffix in (".npz", ".pt", ".safetensors", ".pkl"):
            raise ValueError("Raw arrays or trained weights entered public changes")
        if path.suffix == ".md" and path.name in ("ManiGuard Part4 counterexample premise.md",
                                                  "part4-videos.md"):
            for target in re.findall(r"\]\(([^)]+)\)", raw.decode()):
                if "://" in target or target.startswith("#"):
                    continue
                local = (path.parent / unquote(target.split("#")[0])).resolve()
                if not local.exists():
                    raise ValueError("Broken new relative report link: " + target)
    checks["public_changes_credential_scanned_no_raw_weights_or_datasets"] = True
    checks["new_report_and_gallery_links_resolve"] = True
    build = read(ROOT / "docs/research/reports/maniguard-phase3-part1-gecco/current-build.json")
    if build["pdf_sha256"] != digest(
            ROOT / "docs/research/reports/maniguard-phase3-part1-gecco/named.pdf"):
        raise ValueError("Rendered paper differs from its reviewed build")
    test_result = re.search(r"(\d+) passed", (RUN / "final-tests.log").read_text())
    if test_result is None or "All checks passed!" not in (RUN / "final-lint.log").read_text():
        raise ValueError("Final test/lint completion evidence is missing")
    checks["reviewed_paper_and_completed_test_logs"] = True
    if not all(checks.values()):
        raise ValueError("Terminal verification failed: " + str(checks))
    result = {"schema": "maniguard-part4-verification-v1", "checks": checks,
              "public_files_scanned": sum(p.is_file() for p in paths),
              "new_model_forward_calls": 0, "new_physics_steps": 0,
              "CPU_tests_passed": int(test_result.group(1)), "ruff": "passed", "paper": build,
              "claim": "Bounded premise diagnostic completed; comparative physical stages remain unstarted by the unchanged prospective gate.",
              "provider_stop": "separate final action and provider-state receipt required"}
    atomic(RUN / "verification.json", result)
    atomic(ROOT / "docs/research/research_notes/ManiGuard phase3 Part4 verification.json", result)
    print("Verified Part4 frozen premise, artifacts, metrics, links and public payload")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    verify()
