"""Invoke the pinned native bf16 encoder, preserving full original row order."""

import argparse
from collections import Counter
import os
from pathlib import Path
import time
from types import SimpleNamespace

from common import (Ledger, TASKS, assets, check_deadline, configuration, configure,
                    provenance, publish, read, sha)
from run_io import publish_artifact, report_progress


def main(task):
    config = configuration()
    ap_root, runtime = configure(config, task, require_overlay=False)
    asset = assets(config, task)
    if asset["latent_cache"].exists():
        raise ValueError("Retrieval cache already exists; do not recompute it")
    import h5py
    from anchored_planning import model

    with h5py.File(asset["dataset"], "r") as handle:
        total = len(handle["pixels"])
    ledger = Ledger(config)
    identity = f"cache:{task}:attempt{int(config.get('cache_attempt', 0))}"
    ledger.reserve(identity, "cache", encoder_images=total)
    counters = Counter()
    started, last = time.monotonic(), 0
    original = model.encode_retrieval
    root = Path(os.environ["LEWM_RUN_DIR"])
    publish("artifacts/provenance.json", provenance(config, ap_root))

    def counted(model_instance, frames, device):
        nonlocal last
        check_deadline(config, started)
        counters["encoder_calls"] += 1
        counters["encoder_images_attempted"] += len(frames)
        output = original(model_instance, frames, device)
        counters["encoder_images_completed"] += len(frames)
        if time.monotonic() - last > 60 or counters["encoder_images_completed"] == total:
            last = time.monotonic()
            publish(f"metrics/cache-{counters['encoder_images_completed']:08d}.json",
                    {"task": task, "counts": counters, "elapsed_seconds": last - started})
            report_progress(counters["encoder_images_completed"], total, unit="encoded images")
        return output

    model.encode_retrieval = counted
    completed = False
    try:
        args = SimpleNamespace(task=task, batch_size=int(config.get("cache_batch_size", 128)),
                               device="cuda:0")
        runtime.encode_cache(args, asset)
        if counters["encoder_images_completed"] != total:
            raise ValueError("Incomplete full-row retrieval cache")
        receipt = read(asset["latent_cache"].with_suffix(".json"))
        receipt.update(sha256=sha(asset["latent_cache"]),
                       bytes=asset["latent_cache"].stat().st_size,
                       dataset=read(asset["dataset"].with_suffix(".receipt.json")),
                       checkpoint_sha256=sha(asset["checkpoint"]),
                       counts=dict(counters), source_run=root.name)
        from common import write
        write(asset["latent_cache"].with_suffix(".receipt.json"), receipt)
        publish("artifacts/cache.json", receipt)
        publish_artifact(f"artifacts/retrieval-cache/{task}.npy", source=asset["latent_cache"])
        completed = True
    finally:
        model.encode_retrieval = original
        ledger.finish(identity, counters, completed)
        publish("metrics/accounting.json", {"completed": completed, "counts": counters,
                "elapsed_seconds": time.monotonic() - started,
                "simulator_actions": 0})
        publish("metrics/study-attempt-ledger.json", read(ledger.path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=TASKS)
    main(parser.parse_args().task)
