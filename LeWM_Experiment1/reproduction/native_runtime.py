"""Select an isolated pre-3.12 MuJoCo overlay on the rented host."""

import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

PINS = {"mujoco": "3.11.0", "dm-control": "1.0.43"}


def main():
    cfg = json.loads(Path(os.environ["LEWM_CONFIG"]).read_text())
    root = Path(cfg["asset_root"])
    if not str(root).startswith("/workspace/LeWM_Experiment1/"):
        raise ValueError("Runtime preparation must run on the rented host")
    overlay = root / "runtime-mujoco311"
    if sys.argv[1:] == ["--setup"]:
        from run_io import publish_artifact

        if overlay.exists():
            raise ValueError("Refusing to overwrite an existing runtime overlay")
        report = Path(os.environ["LEWM_RUN_DIR"]) / ".control/pip-report.json"
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps",
                        "--only-binary=:all:", "--target", str(overlay),
                        "--report", str(report),
                        *[f"{name}=={version}" for name, version in PINS.items()]],
                       check=True, timeout=600)
        publish_artifact("artifacts/pip-report.json", source=report)
        sys.path.insert(0, str(overlay))
        import mujoco
        import torch

        actual = {name: importlib.metadata.version(name) for name in PINS}
        assert actual == PINS
        assert Path(mujoco.__file__).is_relative_to(overlay)
        assert str(torch.__file__).startswith("/venv/main/")
        record = {"pins": actual, "interpreter": sys.executable,
                  "mujoco_file": mujoco.__file__, "torch_file": torch.__file__,
                  "torch_version": torch.__version__, "simulator_actions": 0,
                  "model_queries": 0}
        publish_artifact("artifacts/runtime-overlay.json", data=json.dumps(record, indent=2))
        print(json.dumps(record), flush=True)
        return
    if not overlay.is_dir():
        raise ValueError("Prepare the pinned overlay first")
    sys.path.insert(0, str(overlay))
    assert {name: importlib.metadata.version(name) for name in PINS} == PINS
    sys.argv = sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
