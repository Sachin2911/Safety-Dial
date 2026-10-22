"""Checked Part 1 source hooks; the installed upstream evaluator stays untouched."""

import hashlib
import json
from pathlib import Path

from helpers.maniguard_source import prepare_runtime as prepare_base_runtime


def prepare_runtime(output):
    path = prepare_base_runtime(output)
    from helpers.maniguard_source import ROOT

    original = ROOT / "third_party/ManiGuard/maniguard/eval/benchmark.py"
    (Path(output) / "upstream_benchmark.py").write_bytes(original.read_bytes())
    source = path.read_text()
    for before, after in {
        "from helpers.maniguard_runtime import STUDY": "from helpers.maniguard_phase3_runtime import STUDY",
        "STUDY.start(scene_info, obs)": "STUDY.start(scene_info, obs, env, robot, monitor)",
        "action_clipped = np.clip(ctrl_action, action_space.low, action_space.high)": "action_clipped = STUDY.clip_action(step_idx, ctrl_action, action_space)",
    }.items():
        if source.count(before) != 1:
            raise ValueError("Nonunique Part 1 source anchor: " + before)
        source = source.replace(before, after, 1)
    compile(source, str(path), "exec")
    path.write_text(source)
    receipt_path = Path(output) / "runtime-provenance.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["base_runtime_sha256"] = receipt.pop("runtime_sha256")
    receipt["runtime_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    receipt["phase"] = "phase3_part1"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return path
