"""Bounded, durably charged synthetic server preparation, without simulator scenes."""

import argparse
import datetime as dt
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/helpers"))
from maniguard_evolution_episode import atomic_json, reset_sampler, utc


def report(directory, ready=False):
    attempts = [json.loads(p.read_text()) for p in directory.glob("*/preseed-query.json")]
    return {"schema": "maniguard-v3-server-preparation-v1", "updated_utc": utc(),
            "attempt_directory": str(directory), "ready": ready,
            "model_query_attempts": sum(row["attempted"] for row in attempts),
            "model_query_returns": sum(row["completed"] for row in attempts),
            "model_query_seconds": sum(row.get("seconds", 0) for row in attempts),
            "connection_or_query_starts": len(attempts),
            "new_simulator_interactions": 0, "episode_starts": 0}


def prime(config):
    output = Path(config["server_preparation_accounting"])
    output.parent.mkdir(parents=True, exist_ok=True)
    directory = output.parent / "policy-primer"
    directory.mkdir(exist_ok=True)
    if output.exists() and json.loads(output.read_text()).get("ready"):
        return json.loads(output.read_text())
    end = dt.datetime.fromisoformat(config["limits"]["deadline_utc"]).timestamp()
    latest = end - config["limits"]["minimum_remaining_seconds_before_first_episode"]
    startup_end = min(latest, time.time() + 360)
    while time.time() < startup_end:
        value = report(directory)
        atomic_json(output, value)
        if value["model_query_attempts"] >= config["limits"][
                "max_server_preparation_model_query_attempts"]:
            raise RuntimeError("The four server-preparation query attempts are exhausted")
        index = len(list(directory.glob("attempt-*")))
        job = directory / f"attempt-{index:03d}"
        job.mkdir(exist_ok=False)
        try:
            reset_sampler(job, config["evaluator"]["host"], config["evaluator"]["port"])
        except Exception as error:
            atomic_json(job / "error.json", {"error_type": type(error).__name__,
                                              "recorded_utc": utc()})
            atomic_json(output, report(directory))
            time.sleep(3)
        else:
            value = report(directory, ready=True)
            atomic_json(output, value)
            return value
    raise RuntimeError("Server preparation exceeded its startup or first-episode window")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prime(json.loads(args.config.read_text()))), flush=True)
