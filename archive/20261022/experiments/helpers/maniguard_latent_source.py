"""Checked additive evaluator hooks for the current latent-noise study."""

import hashlib
import json
from pathlib import Path

from helpers.maniguard_source import prepare_runtime as prepare_base


def prepare_runtime(output):
    path = prepare_base(output)
    source = path.read_text()
    for old, new in {
        "from helpers.maniguard_runtime import STUDY":
            "from helpers.maniguard_latent_runtime import STUDY",
        "STUDY.start(scene_info, obs)": "STUDY.start(scene_info, obs, env, robot, monitor)",
        'out["study_episode_id"] = STUDY.job["episode_id"]':
            'out["study_episode_id"] = STUDY.job["episode_id"]\n'
            '    out.update(STUDY.policy_payload)',
    }.items():
        if source.count(old) != 1:
            raise ValueError("Nonunique latent-study source anchor: " + old)
        source = source.replace(old, new, 1)
    compile(source, str(path), "exec")
    path.write_text(source)
    receipt = Path(output) / "runtime-provenance.json"
    data = json.loads(receipt.read_text())
    data["base_runtime_sha256"] = data.pop("runtime_sha256")
    data.update(runtime_sha256=hashlib.sha256(source.encode()).hexdigest(),
                study="latent-evolution-v1")
    receipt.write_text(json.dumps(data, indent=2) + "\n")
    return path
