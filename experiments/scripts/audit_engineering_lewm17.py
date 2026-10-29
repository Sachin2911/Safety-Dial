"""Audit every engineering attempt without selecting on controller success."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np

from helpers.lewm17_protocol import OUT, write_json
from helpers.lewm17_selector import reinforce_gradient, utility


def main():
    rows = [json.loads(p.read_text()) for p in (OUT / "attempts").glob("*/result.json")]
    rows = [r for r in rows if r["job"]["stage"] in ("E", "E2")]
    assert len(rows) == 48
    parity, render, diagnostic = [], [], {}
    for row in rows:
        j = row["job"]
        assert row["status"] == "complete"
        assert row["replay_error_max"] <= (1e-3 if j["env"] == "tworoom" else 1e-7)
        if j["condition"] == "native":
            assert row["native_parity_max"] <= 1e-5
        path = OUT / "attempts" / j["id"]
        trace = np.load(path / "states.npz")
        assert len(trace["actions"]) == row["counters"]["environment_actions"]
        assert len(trace["fine_states"]) == row["counters"]["executed_physics_steps"]
        assert all(np.isfinite(trace[key]).all() for key in trace.files)
        if len(trace["frames"]):
            frame = trace["frames"][0]
            assert frame.shape == (224, 224, 3) and frame.std() > 5
            render.append(
                {"job": j["id"], "shape": list(frame.shape), "pixel_std": float(frame.std())}
            )
        if j["condition"] == "fixed":
            other = next(
                r
                for r in rows
                if r["job"]["block"] == j["block"]
                and r["job"]["stage"] == j["stage"]
                and r["job"]["condition"] == "selector"
            )
            reference = np.load(OUT / "attempts" / other["job"]["id"] / "states.npz")
            for key in ("states", "actions", "fine_states"):
                np.testing.assert_array_equal(trace[key], reference[key])
            parity.append(j["block"])
    for env in ("tworoom", "reacher"):
        selected = [
            r for r in rows if r["job"]["env"] == env and r["job"]["condition"] == "stochastic"
        ]
        episodes = [
            (
                utility(r),
                json.loads((OUT / "attempts" / r["job"]["id"] / "decisions.json").read_text()),
            )
            for r in selected
        ]
        gradient, entropy = reinforce_gradient(episodes, 0.0)
        diagnostic[env] = {
            "episodes": len(episodes),
            "reward_variance": float(np.var([x[0] for x in episodes])),
            "entropy": entropy,
            "gradient_norm": float(np.linalg.norm(gradient)),
            "mode_counts": np.bincount(
                [d["mode"] for _, ds in episodes for d in ds], minlength=6
            ).tolist(),
            "interpretation": "Development diagnostic only, no coefficients updated or hyperparameters selected",
        }
    receipt = {
        "passed": True,
        "episodes": len(rows),
        "constant_parity_blocks": parity,
        "native_max": max(r.get("native_parity_max") or 0 for r in rows),
        "render_checks": render,
        "learner_diagnostics": diagnostic,
        "replay_max": {
            e: max(r["replay_error_max"] for r in rows if r["job"]["env"] == e)
            for e in ("tworoom", "reacher")
        },
    }
    write_json(ROOT / "progress/receipts/17-engineering-final-audit.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
