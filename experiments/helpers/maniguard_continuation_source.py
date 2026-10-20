"""Add only compound study callbacks to a checked native evaluator copy."""

import hashlib
import json
from pathlib import Path

from helpers.maniguard_latent_source import prepare_runtime as prepare_latent

REQUIRED_SOURCES = tuple("experiments/helpers/" + name + ".py" for name in (
    "maniguard_latent", "maniguard_latent_policy", "maniguard_latent_runtime",
    "maniguard_latent_source", "maniguard_runtime", "maniguard_source", "maniguard_client",
    "maniguard_adaptation", "maniguard_phase2_features", "maniguard_phase2_outcomes",
    "maniguard_results", "maniguard_latent_completion", "maniguard_continuation_handoff",
    "maniguard_continuation_evidence", "maniguard_continuation_runtime",
    "maniguard_continuation_source", "maniguard_continuation_results",
    "maniguard_continuation_protocol",
)) + (
    "experiments/scripts/maniguard_continuation_episode.py",
    "configs/maniguard/phase2/encoder.json", "configs/maniguard/phase2/completion.json",
    "configs/maniguard/phase2/sources.json",
)


def prepare_runtime(output):
    path = prepare_latent(output)
    source = path.read_text()
    old = "from helpers.maniguard_latent_runtime import STUDY"
    new = "from helpers.maniguard_continuation_runtime import STUDY"
    if source.count(old) != 1:
        raise ValueError("Compound runtime import anchor is not unique")
    source = source.replace(old, new, 1)
    compile(source, str(path), "exec")
    receipt = Path(output) / "runtime-provenance.json"
    data = json.loads(receipt.read_text())
    data["latent_runtime_sha256"] = data["runtime_sha256"]
    data.update(runtime_sha256=hashlib.sha256(source.encode()).hexdigest(),
                study="continuation-feasibility-v1")
    path.write_text(source)
    receipt.write_text(json.dumps(data, indent=2) + "\n")
    return path
