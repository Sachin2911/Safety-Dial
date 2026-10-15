"""Package drift checks must reject incomplete or incorrectly sourced installations."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location(
    "check_platform", Path(__file__).resolve().parents[1] / "check_platform.py"
)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class PackageChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = Path(self.temp.name)
        self.patch = patch.object(checker, "CONFIG", self.config)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def requirements(self, text):
        (self.config / "requirements-linux-py312.txt").write_text(text)

    def test_wrong_cuda_build_is_rejected(self):
        self.requirements("torch==2.11.0+cu130\n")
        with (
            patch.object(checker.metadata, "version", return_value="2.11.0+cu128"),
            self.assertRaisesRegex(RuntimeError, "expected 2.11.0\\+cu130"),
        ):
            checker.check_packages()

    def test_missing_package_is_rejected(self):
        self.requirements("torch==2.11.0+cu130\n")
        with (
            patch.object(
                checker.metadata, "version", side_effect=checker.metadata.PackageNotFoundError
            ),
            self.assertRaisesRegex(RuntimeError, "torch: missing"),
        ):
            checker.check_packages()

    def test_matching_snapshot_passes(self):
        self.requirements("# snapshot\n\ntorch==2.11.0+cu130\n    # via platform\n")
        with patch.object(checker.metadata, "version", return_value="2.11.0+cu130"):
            self.assertEqual(checker.check_packages(), {"torch": "2.11.0+cu130"})

    def test_lab_source_requires_correct_commit_and_subdirectory(self):
        repository = "https://github.com/isaac-sim/IsaacLab.git"
        revision = "a" * 40
        self.requirements(
            f"isaaclab @ git+{repository}@{revision}#subdirectory=tools/wheel_builder\n"
        )
        for commit, subdirectory, passes in [
            (revision, "tools/wheel_builder", True),
            ("b" * 40, "tools/wheel_builder", False),
            (revision, "another-package", False),
        ]:
            with self.subTest(commit=commit, subdirectory=subdirectory):
                dist = Mock(version="3.0.0rc1")
                dist.read_text.return_value = json.dumps(
                    {
                        "url": repository,
                        "vcs_info": {"commit_id": commit},
                        "subdirectory": subdirectory,
                    }
                )
                with patch.object(checker.metadata, "distribution", return_value=dist):
                    if passes:
                        self.assertEqual(checker.check_packages(), {"isaaclab": "3.0.0rc1"})
                    else:
                        with self.assertRaisesRegex(RuntimeError, "pinned revision"):
                            checker.check_packages()

    def test_unpinned_requirement_is_rejected(self):
        self.requirements("torch>=2\n")
        with self.assertRaisesRegex(RuntimeError, "Unsupported requirement"):
            checker.check_packages()


class HostChecks(unittest.TestCase):
    def check_gpu(self, name, driver="595.104.02"):
        with (
            patch.object(checker.sys, "version_info", (3, 12, 14)),
            patch.object(checker.platform, "machine", return_value="x86_64"),
            patch.object(
                checker.platform,
                "freedesktop_os_release",
                return_value={
                    "ID": "ubuntu",
                    "VERSION_ID": "24.04",
                },
            ),
            patch.object(
                checker.subprocess,
                "run",
                return_value=Mock(
                    stdout=f"{name}, {driver}\n",
                ),
            ),
        ):
            checker.check_host()

    def test_a100_is_rejected_before_installation(self):
        with self.assertRaisesRegex(RuntimeError, "lack RT cores"):
            self.check_gpu("NVIDIA A100-PCIE-40GB")

    def test_rtx_host_passes_preflight(self):
        self.check_gpu("NVIDIA RTX PRO 5000 Blackwell")

    def test_old_driver_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Host driver"):
            self.check_gpu("NVIDIA RTX PRO 5000 Blackwell", "570.10.01")


if __name__ == "__main__":
    unittest.main()
