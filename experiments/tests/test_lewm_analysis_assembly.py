"""One rented-Nav dummy JSON integration of owner-bound analysis transfer.

Fake host identities/roots are monkeypatched only inside this software fixture.
It never opens real final data, arrays, model files or a simulator.
"""

import copy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_three_host_json_assembly_and_refusals(tmp_path, monkeypatch):
    assert os.environ.get("RUNPOD_POD_ID") == "vukovc3t4o1dui"
    source = ROOT / "experiments/scripts/assemble_lewm_navigation_analysis_inputs.py"
    spec = importlib.util.spec_from_file_location("assembly", source)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    source_root, destination = tmp_path / "Safety-Dial", tmp_path / "analysis-only"
    source_root.mkdir()
    monkeypatch.setattr(tool, "SOURCE_PROJECT", source_root)
    monkeypatch.setattr(tool, "DESTINATION", destination)

    def fake_rented(pod):
        assert os.environ.get("RUNPOD_POD_ID") == tool.NAV and pod in tool.PODS

    monkeypatch.setattr(tool, "rented", fake_rented)

    def put(name, payload):
        path = source_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
        return path

    def reference(path):
        return {"path": str(path.relative_to(source_root)), "sha256": tool.sha(path)}

    plan = put(
        "configs/plan.json",
        {
            "analysis_script_sha256": tool.sha(
                ROOT / "experiments/scripts/analyze_lewm_navigation_campaign_20261031_v3.py"
            ),
            "execution": {"allowed_pod_ids": sorted(tool.PODS)},
            "source_sha256": {},
        },
    )
    monkeypatch.setattr(tool, "PLAN_SHA256", tool.sha(plan))
    pods = sorted(tool.PODS)
    training, profiles = [], []

    def stage(index, kind, method, owner):
        namespace = "results/synthetic-owned-" + str(index)
        protocol = {
            "evaluation": {"execution_pod_id": owner},
            "output_namespace": namespace,
            "synthetic": True,
        }
        protocol_path = put("configs/protocol-" + str(index) + ".json", protocol)
        inner = put(
            namespace + "/jobs/job0/episode/result.json",
            {
                "episode_seconds": 0.1,
                "planning_seconds": 0.05,
                "synthetic": True,
            },
        )
        row = {
            "job_id": "job0",
            "environment_actions": 1,
            "extra": {
                "episode_result": "/workspace/Safety-Dial/" + str(inner.relative_to(source_root)),
                "episode_result_sha256": tool.sha(inner),
            },
        }
        put(namespace + "/jobs/job0/result.json", row)
        put(namespace + "/jobs/job0/started.json", {"synthetic": True, "job_id": "job0"})
        state = {
            "phase": "complete",
            "active": None,
            "identity": "dummy-identity",
            "completed_job_ids": ["job0"],
            "unfinalized_job_ids": [],
            "action_accounting_exact": True,
            "jobs": {"job0": row},
        }
        put(namespace + "/state.json", state)
        if kind == "training" and method != "tuned_fixed":
            freeze = {"protocol": protocol, "identity": "dummy-identity"}
            nominee = {"job_ids": ["job0"], "provenance": {"synthetic": True}}
            put(namespace + "/candidate-000-000.json", nominee)
            put(namespace + "/update-000.json", {"nominee": nominee})
            put(namespace + "/generation-000.json", {"synthetic": True})
        else:
            freeze = {"protocol_sha256": tool.sha(protocol_path), "pod_id": owner}
        put(namespace + "/source-freeze.json", freeze)
        # These forbidden/unnecessary payloads must never be imported or opened.
        (source_root / namespace / "checkpoint.pkl").write_bytes(b"dummy unneeded checkpoint")
        (source_root / namespace / "jobs/job0/episode/trace.npz").write_bytes(b"not an array")
        put(namespace + "/jobs/job0/episode/selector-history.json", {"unneeded": True})
        if kind == "training":
            return {
                "family": "primary50",
                "repeat_id": index % 3,
                "method": method,
                "protocol": reference(protocol_path),
                "output_namespace": namespace,
            }
        return {
            "family": "primary50",
            "repeat_id": index % 3,
            "host": {"pod_id": owner},
            "protocol": reference(protocol_path),
            "state_path": namespace + "/state.json",
        }

    for index in range(24):
        method = ("cma", "random", "reinforce", "tuned_fixed")[index % 4]
        training.append(stage(index, "training", method, pods[index % 3]))
    for index in range(24, 35):
        profiles.append(stage(index, "final_profile", None, pods[index % 3]))
    shared = put("configs/shared-proof.json", {"synthetic": True})
    inputs = put(
        "configs/bound-inputs.json",
        {
            "schema": "lewm-navigation-analysis-inputs-20261031-v3",
            "plan": reference(plan),
            "training_runs": training,
            "profiles": profiles,
            "all_final_output_namespaces_absent_at_binding": True,
            "shared_proof": reference(shared),
            "final_outcomes_read": 0,
        },
    )
    reports = []
    for pod in pods:
        reports.append(
            put(
                "accounting/host-" + pod + ".json",
                {
                    "owning_pod_id": pod,
                    "complete_final_accounting": True,
                    "unresolved": [],
                },
            )
        )
    aggregate = put(
        "accounting/aggregate.json",
        {
            "schema": "synthetic-accounting-aggregate",
            "complete_final_accounting": True,
            "unresolved": [],
            "source_reports": [{"path": str(path), "sha256": tool.sha(path)} for path in reports],
        },
    )
    freeze_args = SimpleNamespace(
        pod_id=tool.NAV,
        project_root=source_root,
        inputs=inputs,
        plan=plan,
        accounting_merge=aggregate,
        host_report=reports,
        out=tmp_path / "request.json",
    )
    incomplete = put(
        "accounting/incomplete.json",
        {
            **tool.load(aggregate),
            "complete_final_accounting": False,
        },
    )
    bad_freeze = copy.copy(freeze_args)
    bad_freeze.accounting_merge, bad_freeze.out = incomplete, tmp_path / "bad-request.json"
    with pytest.raises(RuntimeError, match="BEFORE"):
        tool.freeze_request(bad_freeze)
    tool.freeze_request(freeze_args)
    request_sha = tool.sha(freeze_args.out)
    archives = []
    for pod, report in zip(pods, reports, strict=True):
        export_args = SimpleNamespace(
            pod_id=pod,
            project_root=source_root,
            request=freeze_args.out,
            request_sha256=request_sha,
            accounting_merge=aggregate,
            host_report=report,
            out_dir=tmp_path / ("export-" + pod),
        )
        tool.export(export_args)
        archives.append(
            (
                export_args.out_dir / ("analysis-json-" + pod + ".tar.gz"),
                export_args.out_dir / "receipt.json",
            )
        )
    archive, receipt = archives[0]
    import_args = SimpleNamespace(
        pod_id=tool.NAV,
        request=freeze_args.out,
        request_sha256=request_sha,
        archive=archive,
        receipt=receipt,
        destination=destination,
    )
    bad = copy.copy(import_args)
    bad.destination = source_root
    with pytest.raises(RuntimeError, match="outside canonical"):
        tool.import_archive(bad)
    bad = copy.copy(import_args)
    bad.request_sha256 = "0" * 64
    with pytest.raises(RuntimeError, match="SHA changed"):
        tool.import_archive(bad)
    bad_receipt = tmp_path / "wrong-owner-receipt.json"
    tool.write_new(bad_receipt, {**tool.load(receipt), "owning_pod_id": pods[1]})
    bad = copy.copy(import_args)
    bad.receipt = bad_receipt
    with pytest.raises(RuntimeError, match="ownership/accounting"):
        tool.import_archive(bad)
    wrong_hash = tmp_path / "wrong-hash-receipt.json"
    tool.write_new(wrong_hash, {**tool.load(receipt), "archive_sha256": "0" * 64})
    bad.receipt = wrong_hash
    with pytest.raises(RuntimeError, match="archive hash"):
        tool.import_archive(bad)
    for archive, receipt in archives:
        import_args.archive, import_args.receipt = archive, receipt
        tool.import_archive(import_args)
    with pytest.raises(RuntimeError, match="same owner twice"):
        tool.import_archive(import_args)
    validate_args = SimpleNamespace(
        pod_id=tool.NAV,
        request=freeze_args.out,
        request_sha256=request_sha,
        destination=destination,
        out=tmp_path / "validation.json",
    )
    tool.validate(validate_args)
    assert tool.load(validate_args.out)["namespace_count"] == 35
    assert not list(destination.rglob("*.npz")) and not list(destination.rglob("*.pkl"))
    assert not list(destination.rglob("selector-history.json"))
    assert len(list((destination / ".ownership").glob("*.json"))) == 3
    tool.write_new(
        tmp_path / "synthetic-qualification.json",
        {
            "qualified": True,
            "synthetic_only": True,
            "actual_execution_pod": tool.NAV,
            "helper_sha256": tool.sha(source),
            "fixture_sha256": tool.sha(Path(__file__)),
            "new_model_queries": 0,
            "new_simulator_calls": 0,
            "real_final_data_read": 0,
            "checks": [
                "35namespaces/3owners roundtrip",
                "canonical-root import refused",
                "wrong request/archive hash refused",
                "wrong owning pod refused",
                "incomplete accounting refused",
                "duplicate import refused",
                "no binaries or selector histories imported",
            ],
        },
    )
