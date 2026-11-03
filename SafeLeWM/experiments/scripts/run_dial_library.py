"""Prospective saved-library development evaluation on the existing 4090."""

import collections
import json
from pathlib import Path
import socket
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/safety-dial-library-20261102"
OUT = BASE / "run-v1"
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "experiments/scripts")]
import numpy as np  # noqa: E402
import adapter_pilot_v2 as qualified  # noqa: E402
import run_matched_ppo as prior  # noqa: E402
from run_mechanism import spec  # noqa: E402
from helpers.pilot_common import digest, save  # noqa: E402


def main():
    import torch
    from helpers.recurrent_adapter import AdapterSpec, RecurrentActionAdapter

    config = json.loads((ROOT / "configs/safety-dial-library/run.json").read_text())
    assert socket.gethostname() == config["host"]
    assert torch.__version__ == "2.4.1+cu124"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    preparation = json.loads((BASE / "preparation.json").read_text())
    remaining = config["run_seconds"] - (time.time() - preparation["started"])
    assert remaining > 0
    started = time.monotonic()
    deadline = started + remaining
    qualified.OUT = prior.OUT = OUT
    qualified.freeze()
    frozen = json.loads((OUT / "freeze.json").read_text())
    files = [
        ROOT / "SafetyDialLibrary.md",
        Path(__file__),
        ROOT / "experiments/scripts/prepare_dial_library.py",
        ROOT / "experiments/scripts/audit_dial_library.py",
        ROOT / "experiments/scripts/launch_dial_library.py",
        ROOT / "experiments/scripts/preserve_dial_library.py",
        ROOT / "experiments/scripts/run_matched_ppo.py",
        ROOT / "experiments/scripts/run_mechanism.py",
        ROOT / "experiments/scripts/ppo_worker.py",
        ROOT / "experiments/helpers/ppo_adapter.py",
        ROOT / "experiments/helpers/mechanism.py",
        ROOT / "experiments/helpers/replication_budget.py",
        BASE / "library.json",
        BASE / "preparation.json",
        *sorted((BASE / "selection-inputs").glob("*")),
        *sorted((BASE / "weights").glob("*.npz")),
        *sorted((ROOT / "configs/safety-dial-library").glob("*.json")),
    ]
    for path in files:
        rel = str(path.relative_to(ROOT))
        frozen["sources"][rel] = digest(path)
        target = OUT / "sources" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    frozen.update(
        instance_id=config["pod_id"],
        machine_id=None,
        host=config["host"],
        study="prospective development of a selected policy library",
        deadline=preparation["started"] + config["run_seconds"],
    )
    save(OUT / "freeze.json", frozen)
    panel = json.loads((ROOT / "configs/safety-dial-library/panel.json").read_text())
    library = json.loads((BASE / "library.json").read_text())
    summary = dict(
        status="running",
        phase="engineering",
        results=[],
        resources={},
        host=dict(name=config["host"], action_cap=config["action_cap"]),
        prior_acquisition_actions=library["prior_acquisition_actions"],
    )
    counts = collections.Counter()
    pool = None

    def record(row):
        summary["results"].append(row)
        counts.update(row["resources"])
        assert counts["environment_actions"] <= config["action_cap"]
        assert counts["environment_resets"] <= config["reset_cap"]
        summary.update(resources=dict(counts), seconds=time.monotonic() - started)
        save(OUT / "summary.json", summary)

    try:
        pool = prior.SerialPool(deadline)
        actor = RecurrentActionAdapter(
            AdapterSpec(396, 2, initialization_seed=config["engineering_seed"])
        ).eval()
        neutral = actor.parameters_vector()
        (OUT / "initialization").mkdir()
        np.savez(OUT / "initialization/engineering.npz", neutral=neutral)
        for case in panel["training"]:
            names = []
            for arm, parameters in [("native", None), ("neutral", neutral)]:
                job = spec(
                    "engineering", case, arm, panel["training_seed"], "parity-" + arm, "engineering"
                )
                names.append(job["id"])
                record(pool.episode(job, parameters))
            qualified.parity(*names)
        save(OUT / "engineering.json", dict(passed=True, episodes=16))
        for pid in library["schedule"]:
            policy = library["policies"][pid]
            summary["phase"] = "development-" + pid
            parameters = None
            if policy["original_path"]:
                path = BASE / "weights" / (pid + ".npz")
                assert digest(path) == library["weights"][pid]["sha256"]
                parameters = np.load(path)["parameters"].copy()
            for case in panel["evaluation"]:
                for seed in panel["evaluation_seeds"]:
                    job = spec(
                        "development",
                        case,
                        policy["arm"],
                        seed,
                        "dev-" + pid,
                        policy["repeat"],
                        policy["candidate"],
                        "head",
                        policy["search"],
                    )
                    job["library_policy"] = pid
                    record(pool.episode(job, parameters))
            print(
                json.dumps(
                    dict(
                        policy=pid,
                        completed=len(summary["results"]),
                        resources=dict(counts),
                        seconds=time.monotonic() - started,
                    )
                ),
                flush=True,
            )
        assert len(summary["results"]) == config["development_episodes"] + 16
        summary["status"] = "complete"
    except BaseException:
        summary.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        if pool is not None:
            pool.close()
        summary.update(resources=qualified.resource_summary(), seconds=time.monotonic() - started)
        save(OUT / "summary.json", summary)


if __name__ == "__main__":
    main()
