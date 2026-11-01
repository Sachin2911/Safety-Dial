"""Rented-only stdlib synthetic JSON qualification; no real campaign outcomes/arrays."""

import copy
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    path.write_text(raw)
    return {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def fixture(root):
    target = "synthetic-new-final-pod"
    helper = module(
        ROOT / "experiments/helpers/lewm_navigation_final_relocation.py", "relocation_guard"
    )
    wrapper_path = ROOT / "experiments/scripts/run_lewm_navigation_relocated_final.py"
    wrapper = module(wrapper_path, "relocation_wrapper")
    wrapper_ref = put(root, str(wrapper_path.relative_to(ROOT)), wrapper_path.read_text())
    evaluator_name = "experiments/scripts/run_lewm_navigation_campaign_v2.py"
    evaluator = put(root, evaluator_name, "# Synthetic evaluator bytes only, never imported\n")
    profiles, moves, mean_profiles = [], [], []
    case = {"case_id": "synthetic-case-5000", "source_episode": 5000}
    for family, cap, repeats in (("primary50", 50, 3), ("secondary30", 30, 3)):
        for repeat in range(repeats):
            original = "old-nav" if repeat == 0 or (cap == 50 and repeat == 2) else f"old-{repeat}"
            name = f"{family}-r{repeat}"
            jobs = [
                {
                    "arm": "native",
                    "case": case,
                    "environment_seed": 718000,
                    "policy_seed": 718000,
                    "max_actions": cap,
                }
            ]
            protocol = {
                "schema": "lewm-navigation-comparison-v1",
                "name": name,
                "host": {"host": original, "pod_id": original},
                "evaluation": {
                    "execution_pod_id": original,
                    "deadline_utc": wrapper.ORIGINAL_DEADLINE,
                },
                "source_sha256": {},
                "jobs": jobs,
                "output_namespace": "results/" + name,
            }
            ref = put(root, "configs/" + name + ".json", protocol)
            profiles.append(
                {
                    "family": family,
                    "repeat_id": repeat,
                    "host": protocol["host"],
                    "protocol": ref,
                    "state_path": "results/" + name + "/state.json",
                }
            )
            if not (repeat == 0 or (cap == 50 and repeat == 2)):
                moves.append(
                    {
                        "protocol": ref,
                        "entrypoint": evaluator,
                        "output_namespace": protocol["output_namespace"],
                        "original_pod_id": original,
                        "pod_id": target,
                    }
                )
    for cap, originals in ((50, ("old-nav", "old-1")), (30, ("old-nav", "old-1", "old-2"))):
        for original in originals:
            name = f"aux-{cap}-{original}"
            protocol = {
                "schema": "lewm-navigation-comparison-v1",
                "name": name,
                "host": {"host": original, "pod_id": original},
                "evaluation": {
                    "execution_pod_id": original,
                    "deadline_utc": wrapper.ORIGINAL_DEADLINE,
                },
                "source_sha256": {},
                "output_namespace": "results/" + name,
                "jobs": [
                    {
                        "arm": "decoded_native",
                        "case": {**case, "task_cost_backend": "decoded-current-target-v1"},
                        "environment_seed": 718000,
                        "policy_seed": 718000,
                        "max_actions": cap,
                    }
                ],
            }
            ref = put(root, "configs/" + name + ".json", protocol)
            profiles.append(
                {
                    "family": f"aux-decoded-native{cap}",
                    "repeat_id": None,
                    "host": protocol["host"],
                    "protocol": ref,
                }
            )
            moves.append(
                {
                    "protocol": ref,
                    "entrypoint": evaluator,
                    "output_namespace": protocol["output_namespace"],
                    "original_pod_id": original,
                    "pod_id": target,
                }
            )
    for repeat in range(3):
        original = "old-nav" if repeat == 0 else f"old-{repeat}"
        name = f"mean-r{repeat}"
        protocol = {
            "schema": "lewm-navigation-comparison-v1",
            "name": name,
            "host": {"host": original, "pod_id": original},
            "evaluation": {"execution_pod_id": original, "deadline_utc": wrapper.ORIGINAL_DEADLINE},
            "source_sha256": {},
            "output_namespace": "results/" + name,
        }
        ref = put(root, "configs/" + name + ".json", protocol)
        mean_profiles.append({"repeat_id": repeat, "protocol": ref})
        if repeat:
            moves.append(
                {
                    "protocol": ref,
                    "entrypoint": evaluator,
                    "output_namespace": protocol["output_namespace"],
                    "original_pod_id": original,
                    "pod_id": target,
                }
            )
    inputs = {
        "profiles": profiles,
        "schema": "lewm-navigation-analysis-inputs-20261031-v3",
        "all_final_output_namespaces_absent_at_binding": True,
    }
    inputs_ref = put(root, "configs/original-inputs.json", inputs)
    means_ref = put(
        root,
        "configs/original-means.json",
        {
            "schema": "lewm-terminal-mean-final-sidecar-v1",
            "profiles": mean_profiles,
            "all_main_outputs_absent_at_binding": True,
        },
    )
    now = dt.datetime.now(dt.timezone.utc)
    authority = {
        "schema": "lewm-frozen-final-relocation-authority-v1",
        "execution_enabled": True,
        "maximum_duration_seconds": 7200,
        "total_campaign_cap_usd": 13,
        "no_new_training": True,
        "no_outcome_based_changes": True,
        "no_profile_selection_from_outcomes": True,
        "reason": "original_host_gpu_capacity_unavailable",
        "capacity_failure_evidence": put(root, "configs/capacity.json", {"synthetic": True}),
        "first_restart_utc": (now - dt.timedelta(seconds=1)).isoformat(),
        "new_deadline_utc": (now + dt.timedelta(minutes=119)).isoformat(),
        "original_deadline_utc": wrapper.ORIGINAL_DEADLINE,
        "actual_final_host": {"host": "completion", "pod_id": target},
        "profiles": moves,
        "source_sha256": {
            wrapper_ref["path"]: wrapper_ref["sha256"],
            evaluator["path"]: evaluator["sha256"],
        },
        "original_analysis_inputs": inputs_ref,
        "original_terminal_mean_manifest": means_ref,
    }
    authority_ref = put(root, "configs/authority.json", authority)
    return helper, wrapper, inputs, authority, authority_ref


class RelocationQualification(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix="synthetic-lewm-relocation-")
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.helper, self.wrapper, self.inputs, self.authority, self.ref = fixture(self.root)

    def test_full_coverage_actual_host_and_common_seed_guards(self):
        checked = self.helper.authority(self.root, self.ref)
        result = self.helper.validate_binding(self.root, self.inputs, checked)
        self.assertEqual(result["relocated_main_aux_profile_count"], 8)
        self.assertTrue(result["all_case_seed_pairs_checked"])
        missing = copy.deepcopy(checked)
        missing["profiles"].pop()
        with self.assertRaises(ValueError):
            self.helper.validate_binding(self.root, self.inputs, missing)
        changed = copy.deepcopy(self.inputs)
        auxiliary = next(e for e in changed["profiles"] if e["family"].startswith("aux-"))
        protocol = self.helper.load_ref(self.root, auxiliary["protocol"])
        protocol["jobs"][0]["policy_seed"] += 1
        original_ref = auxiliary["protocol"]
        auxiliary["protocol"] = put(self.root, original_ref["path"], protocol)
        updated = copy.deepcopy(checked)
        next(row for row in updated["profiles"] if row["protocol"] == original_ref)["protocol"] = (
            auxiliary["protocol"]
        )
        with self.assertRaises(ValueError):
            self.helper.validate_binding(self.root, changed, updated)

    def test_wrapper_exact_two_field_diff_hash_and_non_replay(self):
        row = self.authority["profiles"][0]
        args = type(
            "Args",
            (),
            {
                "authority": self.ref["path"],
                "authority_sha256": self.ref["sha256"],
                "entrypoint": row["entrypoint"]["path"],
                "protocol": row["protocol"]["path"],
                "out": row["output_namespace"],
            },
        )()
        wrapper_path = self.root / "experiments/scripts/run_lewm_navigation_relocated_final.py"
        with (
            mock.patch.object(self.wrapper, "ROOT", self.root),
            mock.patch.object(self.wrapper, "__file__", str(wrapper_path)),
            mock.patch.dict(os.environ, {"RUNPOD_POD_ID": row["pod_id"]}),
        ):
            path, output, _, effective, _, record = self.wrapper.prepare(args)
            original_bytes = path.read_bytes()
            original = json.loads(original_bytes)
            restored = copy.deepcopy(effective)
            restored["evaluation"] = copy.deepcopy(original["evaluation"])
            self.assertEqual(restored, original)
            self.assertEqual(len(record["effective_override"]), 2)
            self.assertEqual(path.read_bytes(), original_bytes)
            output.mkdir(parents=True)
            with self.assertRaises(RuntimeError):
                self.wrapper.prepare(args)
            output.rmdir()
            args.authority_sha256 = "0" * 64
            with self.assertRaises(RuntimeError):
                self.wrapper.prepare(args)

    def test_explicit_execution_sidecar_and_actual_pod(self):
        row = self.authority["profiles"][0]
        args = type(
            "Args",
            (),
            {
                "authority": self.ref["path"],
                "authority_sha256": self.ref["sha256"],
                "entrypoint": row["entrypoint"]["path"],
                "protocol": row["protocol"]["path"],
                "out": row["output_namespace"],
            },
        )()
        wrapper_path = self.root / "experiments/scripts/run_lewm_navigation_relocated_final.py"
        with (
            mock.patch.object(self.wrapper, "ROOT", self.root),
            mock.patch.object(self.wrapper, "__file__", str(wrapper_path)),
            mock.patch.dict(os.environ, {"RUNPOD_POD_ID": row["pod_id"]}),
        ):
            _, _, _, _, _, record = self.wrapper.prepare(args)
        entry = next(e for e in self.inputs["profiles"] if e["protocol"] == row["protocol"])
        protocol = self.helper.load_ref(self.root, row["protocol"])
        source = {
            "pod_id": row["pod_id"],
            "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "continuation": put(self.root, "environment/execution.json", record),
        }
        plan = {
            "final_execution_relocation": self.ref,
            "source_sha256": self.authority["source_sha256"],
        }
        self.assertEqual(
            self.helper.verify_execution(self.root, entry, protocol, source, plan),
            self.authority["actual_final_host"],
        )
        wrong_host = {**source, "pod_id": row["original_pod_id"]}
        with self.assertRaises(ValueError):
            self.helper.verify_execution(self.root, entry, protocol, wrong_host, plan)
        record["effective_override"]["planner.iterations"] = {"effective": 1}
        source["continuation"] = put(self.root, "environment/tampered-execution.json", record)
        with self.assertRaises(ValueError):
            self.helper.verify_execution(self.root, entry, protocol, source, plan)

    def test_only_declared_backend_label_can_differ(self):
        checked = self.helper.authority(self.root, self.ref)
        self.helper.validate_binding(self.root, self.inputs, checked)
        entry = next(e for e in self.inputs["profiles"] if e["family"].startswith("aux-"))
        original_ref = entry["protocol"]
        original = self.helper.load_ref(self.root, original_ref)
        for changes in ({"task_cost_backend": "wrong-backend"}, {"unrelated_case_field": 1}):
            inputs, value = copy.deepcopy(self.inputs), copy.deepcopy(checked)
            protocol = copy.deepcopy(original)
            protocol["jobs"][0]["case"].update(changes)
            changed_ref = put(self.root, original_ref["path"], protocol)
            next(e for e in inputs["profiles"] if e["protocol"] == original_ref)["protocol"] = (
                changed_ref
            )
            next(r for r in value["profiles"] if r["protocol"] == original_ref)["protocol"] = (
                changed_ref
            )
            with self.assertRaises(ValueError):
                self.helper.validate_binding(self.root, inputs, value)

    def test_root_time_revision_accepts7200_and_rejects_larger_bound(self):
        checked = self.helper.authority(self.root, self.ref)
        self.assertEqual(checked["maximum_duration_seconds"], 7200)
        exact = copy.deepcopy(self.authority)
        first = dt.datetime.fromisoformat(exact["first_restart_utc"])
        exact["new_deadline_utc"] = (first + dt.timedelta(seconds=7200)).isoformat()
        exact_ref = put(self.root, "configs/exact7200-authority.json", exact)
        self.assertEqual(
            self.helper.authority(self.root, exact_ref)["maximum_duration_seconds"], 7200
        )
        for duration, declared in ((7201, 7200), (7200, 7201)):
            changed = copy.deepcopy(self.authority)
            changed["maximum_duration_seconds"] = declared
            first = dt.datetime.fromisoformat(changed["first_restart_utc"])
            changed["new_deadline_utc"] = (first + dt.timedelta(seconds=duration)).isoformat()
            ref = put(self.root, "configs/too-long-authority.json", changed)
            with self.assertRaises(ValueError):
                self.helper.authority(self.root, ref)


if __name__ == "__main__":
    if not os.environ.get("RUNPOD_POD_ID"):
        raise RuntimeError("Run this synthetic qualification only on an authorized rented host")
    unittest.main()
