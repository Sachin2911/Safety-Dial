"""Rented-only synthetic integrity checks for the four-source JSON assembler."""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]


def load_module():
    path = ROOT / "experiments/scripts/assemble_lewm_navigation_completion_json.py"
    spec = importlib.util.spec_from_file_location("new_completion_assembler", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


class AssemblyIntegrity(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix="synthetic-four-source-assembly-")
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.tool = load_module()

    def test_four_account_coverage_and_incomplete_duplicate_rejection(self):
        target = "synthetic-replacement"
        owners = [*self.tool.OLD_PODS.values(), target]
        paths, hashes = [], []
        for owner in owners:
            row = {
                "schema": self.tool.ACCOUNTING,
                "owning_pod_id": owner,
                "complete_final_accounting": True,
                "unresolved": [],
            }
            if owner == self.tool.OLD_PODS["recovery"]:
                row.update(
                    collection_execution_pod_id=target,
                    recorded_source_owner_pod_id=owner,
                    numeric_collection_executed_on_explicit_authorized_replacement=True,
                )
            path, sha = put(self.root, owner + ".json", row)
            paths.append(path)
            hashes.append(sha)
        with mock.patch.multiple(
            self.tool, NAV_REPORT=hashes[0], PUSHT_REPORT=hashes[1], RECOVERY_REPORT=hashes[2]
        ):
            self.assertEqual(len(self.tool.accounting_reports(paths, hashes, target)), 4)
            with self.assertRaises(RuntimeError):
                self.tool.accounting_reports(paths[:3], hashes[:3], target)
            with self.assertRaises(RuntimeError):
                self.tool.accounting_reports(
                    [*paths[:3], paths[0]], [*hashes[:3], hashes[0]], target
                )
            changed = json.loads(paths[-1].read_text())
            changed["complete_final_accounting"] = False
            path, sha = put(self.root, "incomplete.json", changed)
            with self.assertRaises(RuntimeError):
                self.tool.accounting_reports([*paths[:3], path], [*hashes[:3], sha], target)

    def test_exclusive_copy_hash_readback_and_unsafe_path_rejection(self):
        source, sha = put(self.root, "source.json", {"only": "synthetic JSON"})
        destination = self.root / "sibling/a.json"
        self.tool.copy_checked(source, destination, sha)
        self.assertEqual(destination.read_bytes(), source.read_bytes())
        with self.assertRaises(RuntimeError):
            self.tool.copy_checked(source, destination, sha)
        with self.assertRaises(RuntimeError):
            self.tool.copy_checked(source, self.root / "bad-hash.json", "0" * 64)
        with self.assertRaises(RuntimeError):
            self.tool.relative("../escape.json")

    def test_old_extraction_owner_and_manifest_hash_guards(self):
        origins, receipts = [], []
        for lane, pod in self.tool.OLD_PODS.items():
            base = self.root / ("origin-" + lane)
            base.mkdir()
            manifest, sha = put(base, "MANIFEST-20261031.json", {"pod_id": pod, "files": {}})
            receipt, _ = put(
                self.root,
                lane + "-receipt.json",
                {
                    "schema": "lewm-preserved-origin-extraction-v1",
                    "verified": True,
                    "original_owner_pod_id": pod,
                    "destination": str(base),
                    "every_extracted_member_hashed_and_readback_verified": True,
                    "embedded_manifest_sha256": sha,
                    "origin_files_read_only": True,
                    "archive": {"sha256": "1" * 64},
                },
            )
            self.assertTrue(manifest.is_file())
            origins.append(lane + "=" + str(base))
            receipts.append(lane + "=" + str(receipt))
        self.assertEqual(
            set(self.tool.origins(origins, receipts)), set(self.tool.OLD_PODS.values())
        )
        bad = json.loads(Path(receipts[0].split("=", 1)[1]).read_text())
        bad["embedded_manifest_sha256"] = "0" * 64
        path, _ = put(self.root, "wrong-extraction.json", bad)
        altered = copy.deepcopy(receipts)
        altered[0] = "navigation=" + str(path)
        with self.assertRaises(RuntimeError):
            self.tool.origins(origins, altered)


if __name__ == "__main__":
    if not os.environ.get("RUNPOD_POD_ID"):
        raise RuntimeError("Run synthetic checks only on an authorized rented host")
    unittest.main()
