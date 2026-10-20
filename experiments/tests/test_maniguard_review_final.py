"""Hand-computed paired examples and failed evidence joins, with no physical data."""

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from helpers.maniguard_review_final import CONDITIONS, summarize


@pytest.fixture
def inputs():
    digest = "a" * 64
    finalists = {name: {"candidate_id": name, "sha256": f"{index:064x}"}
                 for index, name in enumerate(CONDITIONS)}
    finalists["random"]["sha256"] = finalists["fixed"]["sha256"]
    protocol = {"status": "capacity_selected_frozen", "workers": 2, "family": "jar",
                "audit_blocks": [{"scene": f"case{block}", "seed": 100 + block,
                                  "order": list(CONDITIONS[block:] + CONDITIONS[:block])}
                                 for block in range(4)]}
    # (requested completion, raw completion, violation). Each condition has four episodes.
    outcomes = {
        "native": [(False, True, False)] + [(False, False, False)] * 3,
        "fixed": [(False, False, False), (True, True, True),
                  (False, False, True), (True, True, False)],
        "ea": [(True, True, False), (True, True, False),
               (True, True, True), (False, False, False)],
        "random": [(False, False, False), (True, True, True),
                   (False, False, True), (True, True, False)],
    }
    rows = []
    for condition in CONDITIONS:
        for block, (completed, raw, violation) in enumerate(outcomes[condition]):
            prefix = "unsafe_" if violation else "safe_"
            rows.append({"attempt": f"audit-b{block}-{condition}-a0",
                         "block": block, "condition": condition, "stage": "independent_audit",
                         "status": "completed", "valid_full_episode": True, "family": "jar",
                         "scene": f"case{block}", "seed": 100 + block,
                         "worker": protocol["audit_blocks"][block]["order"].index(condition) % 2,
                         "controller_sha256": finalists[condition]["sha256"],
                         "requested_completion": completed, "benchmark_completion": raw,
                         "violation": violation, "score": int(completed and not violation)
                         - 2 * int(violation), "ever_contacted_task_object": True,
                         "ever_grasped_task_object": raw,
                         "requested_category": prefix + ("completion" if completed else "unfinished"),
                         "benchmark_category": prefix + ("completion" if raw else "unfinished")})
    prior = [{"attempt": f"synthetic-training-{i}", "valid_full_episode": True}
             for i in range(44)]
    return {"report": {"physical_comparison_complete": True,
                       "accounting": {"observed_epoch": 500.}, "audit": rows,
                       "attempts": prior + rows},
            "design": {"physical_design_complete": True, "errors": [], "pending": [],
                       "verified_full_jobs": 60, "expected_full_jobs": 60,
                       "protocol_sha256": digest, "report_observed_epoch": 500.},
            "protocol": protocol, "seal": {"protocol_sha256": digest, "finalists": finalists},
            "protocol_sha256": digest}


def test_all_categories_pairs_and_identical_genotypes_are_retained(inputs):
    result = summarize(**inputs)
    assert len(result["episodes"]) == 16
    native = result["conditions"]["native"]
    assert native["safe_unfinished"] == 4 and native["safe_completion"] == 0
    assert native["raw_completions"] == native["grasped_episodes"] == 1
    fixed = result["conditions"]["fixed"]
    assert [fixed[key] for key in ("safe_completion", "unsafe_completion",
                                   "safe_unfinished", "unsafe_unfinished")] == [1, 1, 1, 1]
    assert fixed["episodes"] == 4 and fixed["violations"] == 2
    assert fixed["raw_completions"] == fixed["grasped_episodes"] == 2
    assert fixed["contacted_episodes"] == 4 and fixed["mean_utility"] == -.75
    pairs = {(p["reference"], p["candidate"]): p for p in result["paired_contrasts"]}
    assert len(pairs) == 6
    native_ea = pairs["native", "ea"]
    assert [row["utility"] for row in native_ea["blocks"]] == [1, 1, -2, 0]
    assert native_ea["mean_difference"] == {"utility": 0., "safe_completion": .5, "violation": .25}
    assert pairs["fixed", "random"]["same_genotype"]
    assert not pairs["native", "ea"]["same_genotype"]


@pytest.mark.parametrize("change", ["incomplete", "design", "protocol", "date", "missing",
                                   "duplicate", "case", "genotype", "worker", "score", "label",
                                   "incomplete_export", "disagree_export"])
def test_incomplete_or_misbound_independent_evidence_is_rejected(inputs, change):
    value = copy.deepcopy(inputs)
    if change == "incomplete":
        value["report"]["physical_comparison_complete"] = False
    elif change == "design":
        value["design"]["pending"] = ["Remaining job"]
    elif change == "protocol":
        value["seal"]["protocol_sha256"] = "b" * 64
    elif change == "date":
        value["design"]["report_observed_epoch"] = 501.
    elif change == "missing":
        value["report"]["audit"].pop()
    elif change == "duplicate":
        value["report"]["audit"].append(value["report"]["audit"][0])
    elif change == "incomplete_export":
        value["report"]["attempts"].pop(0)
    elif change == "disagree_export":
        value["report"]["audit"] = copy.deepcopy(value["report"]["audit"])
        value["report"]["audit"][0]["ever_contacted_task_object"] = False
    else:
        key, replacement = {"case": ("seed", 999), "genotype": ("controller_sha256", "b" * 64),
                            "worker": ("worker", 1), "score": ("score", 1),
                            "label": ("violation", None)}[change]
        value["report"]["audit"][0][key] = replacement
    with pytest.raises(ValueError):
        summarize(**value)


def test_cli_binds_exact_input_bytes_preserves_output_and_rejects_partial_report(inputs, tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/maniguard_review_final.py"
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(inputs["protocol"]))
    expected = hashlib.sha256(protocol.read_bytes()).hexdigest()
    inputs["seal"]["protocol_sha256"] = inputs["design"]["protocol_sha256"] = expected
    command = [sys.executable, str(script)]
    for name in ("report", "design", "protocol", "seal"):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(inputs[name]))
        command.extend(["--" + name, str(path)])
    output = tmp_path / "summary.json"
    complete = subprocess.run(command + ["--output", str(output)], check=False,
                              capture_output=True, timeout=10)
    assert complete.returncode == 0, complete.stderr.decode()
    first_bytes = output.read_bytes()
    result = json.loads(first_bytes)
    assert result["input_sha256"]["protocol"] == result["protocol_sha256"] == expected
    assert len(result["paired_contrasts"]) == 6 and len(result["episodes"]) == 16
    repeated = subprocess.run(command + ["--output", str(output)], check=False,
                              capture_output=True, timeout=10)
    assert repeated.returncode != 0 and output.read_bytes() == first_bytes
    report_path = tmp_path / "report.json"
    report = json.loads(report_path.read_text())
    report["physical_comparison_complete"] = False
    report_path.write_text(json.dumps(report))
    partial = tmp_path / "partial.json"
    failed = subprocess.run(command + ["--output", str(partial)], check=False,
                            capture_output=True, timeout=10)
    assert failed.returncode != 0 and not partial.exists()
