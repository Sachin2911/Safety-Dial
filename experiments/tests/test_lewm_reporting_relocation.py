"""Rented-only scalar metadata checks for fixed-repeat versus actual-host grouping."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "reporting_relocation", SCRIPTS / "report_lewm_navigation_terminal_mean_sidecar_v2.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@unittest.skipUnless(os.environ.get("RUNPOD_POD_ID"), "Run only on an authorized rented host")
class ReportingRelocation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pod = os.environ["RUNPOD_POD_ID"]

    def write(self, name, data):
        path = self.root / name
        path.write_text(json.dumps(data))
        return path

    def fixture(self, changed=False):
        manifest_path = self.write("manifest.json", {"dummy": True})
        manifest_ref = module.ref(self.root, manifest_path)
        original = ["DUMMY-Nav", "DUMMY-PushT", "DUMMY-Recovery"]
        actual = "DUMMY-newGPU"
        accounts = [
            self.write(
                "account" + str(i) + ".json",
                {
                    "schema": module.core.ACCOUNTING,
                    "owning_pod_id": pod,
                    "complete_final_accounting": True,
                    "unresolved": [],
                },
            )
            for i, pod in enumerate([*original, actual])
        ]
        aggregate = self.write(
            "aggregate.json",
            {
                "schema": module.core.ACCOUNTING + "-aggregate",
                "complete_final_accounting": True,
                "unresolved": [],
                "source_reports": [module.ref(self.root, p) for p in accounts],
            },
        )
        profiles = [
            {
                "repeat_id": i,
                "host": {"pod_id": original[i]},
                "protocol": {"path": "DUMMY-protocol" + str(i), "sha256": str(i) * 64},
            }
            for i in range(3)
        ]
        relocation = {
            "actual_final_host": {"pod_id": actual},
            "profiles": [{"protocol": p["protocol"]} for p in profiles[1:]],
        }
        plan = {
            "execution": {"allowed_pod_ids": [self.pod]},
            "profiles": {
                "secondary30": {"host_assignment": {str(i): original[i] for i in range(3)}}
            },
            "source_sha256": {"DUMMY-code": "a" * 64},
            "final_execution_relocation": {"path": "DUMMY-authority", "sha256": "b" * 64},
        }
        reports = []
        for repeat in range(3):
            rows = []
            for arm in (module.MEAN, *module.core.analysis.ARMS):
                for case in range(16):
                    row = {
                        "arm": arm,
                        "source_episode": -case - 1,
                        "case_id": "DUMMY-case" + str(case),
                        "condition": module.core.analysis.CONDITIONS[case % 4],
                        "safe_completion": 0,
                        "violation": 0,
                        "unfinished": 1,
                        "engaged": 1,
                        "progress": 0.5,
                        "environment_actions": 20,
                        "resources": {"actual_cost_call_attempts": 10},
                    }
                    if changed and repeat == 2 and arm == module.MEAN and case == 0:
                        row.update(
                            safe_completion=1, unfinished=0, progress=0.75, environment_actions=19
                        )
                    rows.append(row)
            report = {
                "schema": "lewm-terminal-mean-own-host-report-v1"
                if repeat == 0
                else "lewm-terminal-mean-repeat-report-v2",
                "repeat_id": repeat,
                "manifest": manifest_ref,
                "owning_pod_id": original[0] if repeat == 0 else actual,
                "accounting_report": module.ref(self.root, accounts[0 if repeat == 0 else 3]),
                "complete_own_host_final_accounting": True,
                "all_ordinary96_and_mean16_physical_episodes_hash_bound_in_accounting": True,
                "optimizer_updates": 6,
                "source_sha256": plan["source_sha256"],
                "final_execution_relocation": plan["final_execution_relocation"],
                "paired_descriptive_contrasts": module.core.contrasts(rows),
                "totals_by_arm": {
                    arm: module.core.totals([r for r in rows if r["arm"] == arm])
                    for arm in (module.MEAN, *module.core.analysis.ARMS)
                },
                "rows": rows,
            }
            reports.append(self.write("report" + str(repeat) + ".json", report))
        args = argparse.Namespace(
            project_root=self.root,
            manifest=manifest_path,
            reports=reports,
            report_sha256=[module.digest(p) for p in reports],
            accounting_merge=aggregate,
            accounting_reports=accounts,
            plan=manifest_path,
        )
        return (
            args,
            (self.root, plan, relocation, {"profiles": profiles}, {}),
            reports[0],
            accounts[0],
        )

    def invoke(self, args, context, nav, account):
        with (
            patch.object(module, "context", return_value=context),
            patch.object(module, "RETAINED_NAV_REPORT", module.digest(nav)),
            patch.object(module, "RETAINED_NAV_ACCOUNTING", module.digest(account)),
        ):
            return module.merge(args)

    def test_two_actual_hosts_three_repeats_and_four_source_accounts(self):
        args, context, nav, account = self.fixture(changed=True)
        result = self.invoke(args, context, nav, account)
        self.assertEqual(result["source_case_clusters"], 16)
        self.assertEqual(result["fixed_controller_repeats"], 3)
        self.assertEqual(result["actual_final_host_count"], 2)
        for item in result["all_five_paired_descriptive_contrasts"].values():
            self.assertEqual(
                item["raw_count_differences_across_fixed_repeats"]["safe_completion"], 1
            )
            self.assertEqual(
                item["equally_averaged_fixed_repeat_rate_differences"]["safe_completion"], 1 / 48
            )
            self.assertEqual(item["actual_action_difference_across_fixed_repeats"], -1)

    def test_missing_fixed_repeat_is_rejected(self):
        args, context, nav, account = self.fixture()
        args.reports[2] = args.reports[1]
        args.report_sha256 = [module.digest(p) for p in args.reports]
        with self.assertRaises(ValueError):
            self.invoke(args, context, nav, account)

    def test_omitted_training_only_source_account_is_rejected(self):
        args, context, nav, account = self.fixture()
        args.accounting_reports.pop(1)
        with self.assertRaises(ValueError):
            self.invoke(args, context, nav, account)

    def test_retained_nav_bytes_cannot_be_substituted(self):
        args, context, nav, account = self.fixture()
        pinned = module.digest(nav)
        nav.write_text(nav.read_text() + " ")
        args.report_sha256 = [module.digest(p) for p in args.reports]
        with (
            patch.object(module, "context", return_value=context),
            patch.object(module, "RETAINED_NAV_REPORT", pinned),
            patch.object(module, "RETAINED_NAV_ACCOUNTING", module.digest(account)),
        ):
            with self.assertRaises(ValueError):
                module.merge(args)


if __name__ == "__main__":
    unittest.main()
