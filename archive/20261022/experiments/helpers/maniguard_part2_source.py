"""Additive checked followup hooks, without changing the frozen Part 1 source."""

import json
from pathlib import Path

from helpers.maniguard_phase3_protocol import sha
from helpers.maniguard_phase3_source import prepare_runtime as prepare_part1


def prepare_runtime(output):
    path = prepare_part1(output)
    source = path.read_text()
    before = "from helpers.maniguard_phase3_runtime import STUDY"
    if source.count(before) != 1:
        raise ValueError("Nonunique followup runtime anchor")
    source = source.replace(before, "from helpers.maniguard_part2_runtime import STUDY", 1)
    compile(source, str(path), "exec")
    path.write_text(source)
    receipt_path = Path(output) / "runtime-provenance.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["part1_runtime_sha256"] = receipt["runtime_sha256"]
    receipt.update(runtime_sha256=sha(path), phase="phase3_followup")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return path
