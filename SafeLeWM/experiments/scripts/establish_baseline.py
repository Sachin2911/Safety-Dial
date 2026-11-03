"""Evaluate the already selected fixed recipe on the frozen 50-case panel."""

import collections
import json
import os
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_baselines as previous  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "outputs/baseline-20261101/run-v2"
OUT = ROOT / "outputs/fixed-baseline-20261101/run-v1"


def main():
    assert os.environ.get("RUNPOD_POD_ID") == "yvrg8ow7i9n9u6"
    from loguru import logger

    logger.disable("stable_worldmodel")
    OUT.mkdir(parents=True, exist_ok=False)
    freeze = json.loads((OLD / "freeze.json").read_text())
    for name, expected in freeze["sources"].items():
        assert previous.digest(ROOT / name) == expected, name
    for name, expected in freeze["upstream_sources"].items():
        assert previous.digest(name) == expected, name
    assert previous.digest(OLD / "panel.json") == freeze["panel_sha256"]
    assert json.loads((OLD / "preflight.json").read_text())["passed"]
    panel = [c for c in json.loads((OLD / "panel.json").read_text()) if c["panel"] == "dataset"]
    assert len(panel) == 50
    for case in panel:
        assert previous.digest(ROOT / case["file"]) == case["sha256"]
        case["profiles"] = ["random05"]
    previous.save(OUT / "panel.json", panel)
    previous.save(
        OUT / "freeze.json",
        dict(
            utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            source_comparison_freeze_sha256=previous.digest(OLD / "freeze.json"),
            sources={
                str(p.relative_to(ROOT)): previous.digest(p)
                for p in [ROOT / "FixedBaseline.md", Path(__file__).resolve()]
            },
            panel_sha256=previous.digest(OUT / "panel.json"),
            checkpoint_sha256=previous.digest(ROOT / "assets/weights.pt"),
            reused_preflight_sha256=previous.digest(OLD / "preflight.json"),
        ),
    )
    torch.set_num_threads(4)
    torch.manual_seed(5100)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pair = previous.models(ROOT)
    previous.OUT = OUT
    totals = collections.Counter()
    rows = []
    started = time.monotonic()
    for case in panel:
        # Worst possible next episode: 150 actions, 30 solves * 9,000 model steps.
        assert totals["environment_actions"] + 150 <= 7500
        assert totals["candidate_model_steps"] + 270000 <= 13500000
        row = previous.episode(case, "random05", pair, totals, started + 1800)
        rows.append(row)
        previous.save(
            OUT / "summary.json",
            dict(
                results=rows,
                resources=dict(totals),
                scheduled=50,
                finished=len(rows),
                seconds=time.monotonic() - started,
            ),
        )
        if row["error"]:
            raise RuntimeError(row["error"])


if __name__ == "__main__":
    main()
