"""Check template selection and updates without contacting Vast or installing packages."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "create_template", ROOT / "scripts/vast/create_template.py"
)
template = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(template)


class TemplateTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        config_dir = self.root / "configs/vast"
        config_dir.mkdir(parents=True)
        for name in (
            "evo-safety-filter.json",
            "unisafe-platform.json",
            "created-template.json",
            "README.md",
            "UNISAFE.md",
        ):
            (config_dir / name).write_bytes((ROOT / "configs/vast" / name).read_bytes())
        self.config = json.loads((config_dir / "evo-safety-filter.json").read_text())
        self.record_path = config_dir / "created-template.json"
        self.record = json.loads(self.record_path.read_text())

    def invoke(self, args, run):
        with (
            mock.patch.object(
                template, "__file__", str(self.root / "scripts/vast/create_template.py")
            ),
            mock.patch("sys.argv", ["create_template.py", *args]),
            mock.patch.object(template.subprocess, "run", run),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            template.main()

    def test_dry_run_never_contacts_api_or_changes_record(self):
        run = mock.Mock()
        self.invoke(["--update", "--dry-run"], run)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0], ["bash", "-n"])
        self.assertEqual(json.loads(self.record_path.read_text()), self.record)

    def update_responses(self, name=None):
        row = {**self.record, "name": name or self.record["name"]}
        saved = {**self.record, "hash_id": "new-template-hash"}
        return mock.Mock(
            side_effect=[
                subprocess.CompletedProcess([], 0),
                subprocess.CompletedProcess([], 0, json.dumps([row])),
                subprocess.CompletedProcess([], 0, "updated template: " + json.dumps(saved)),
            ]
        )

    def test_update_targets_recorded_template_and_keeps_light_startup(self):
        run = self.update_responses()
        self.invoke(["--update"], run)
        command = run.call_args.args[0]
        self.assertEqual(command[:4], ["vastai", "update", "template", self.record["hash_id"]])
        self.assertEqual(command[command.index("--image_tag") + 1], self.config["image_tag"])
        self.assertEqual(
            command[command.index("--onstart-cmd") + 1],
            "exec /opt/instance-tools/bin/entrypoint.sh",
        )
        self.assertIn("SAFETY_DIAL_PLATFORM=maniguard", " ".join(command))
        self.assertEqual(json.loads(self.record_path.read_text())["hash_id"], "new-template-hash")

    def test_alternate_config_uses_preserved_image_and_readme(self):
        run = self.update_responses()
        self.invoke(["--update", "--config", "configs/vast/unisafe-platform.json"], run)
        command = run.call_args.args[0]
        self.assertIn("cuda-13.0.3", command[command.index("--image_tag") + 1])
        self.assertTrue(command[command.index("--readme") + 1].startswith("# Preserved UNISafe"))

    def test_wrong_template_identity_prevents_update(self):
        run = self.update_responses(name="another template")
        with self.assertRaisesRegex(SystemExit, "Could not identify"):
            self.invoke(["--update"], run)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(json.loads(self.record_path.read_text()), self.record)

    def test_unisafe_installer_refuses_maniguard_before_host_checks(self):
        result = subprocess.run(
            ["bash", str(ROOT / "scripts/vast/setup.sh"), "--dry-run"],
            env={**os.environ, "SAFETY_DIAL_PLATFORM": "maniguard"},
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("For the ManiGuard base", result.stderr)


if __name__ == "__main__":
    unittest.main()
