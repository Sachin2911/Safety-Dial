"""Checked additive phase-two hooks over the preserved instrumented evaluator."""
import hashlib
import json
from pathlib import Path

from helpers.maniguard_source import prepare_runtime as prepare_legacy_runtime


def prepare_runtime(output):
    path = prepare_legacy_runtime(output)
    source = path.read_text()
    replacements = {
        "from helpers.maniguard_runtime import STUDY":
            "from helpers.maniguard_phase2_runtime import STUDY",
        "STUDY.start(scene_info, obs)":
            "STUDY.start(scene_info, obs, env, robot, monitor)",
    }
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise ValueError("Nonunique phase-two runtime anchor: " + before)
        source = source.replace(before, after, 1)
    compile(source, "benchmark_phase2.py", "exec")
    path.write_text(source)
    provenance_path = Path(output) / "runtime-provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["phase1_runtime_sha256"] = provenance.pop("runtime_sha256")
    provenance["runtime_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    provenance["phase"] = "phase2"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    return path
