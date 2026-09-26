"""Atomic LeWM recovery with a deterministic, restartable training batch order."""
from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Sampler


class RestartableBatchSampler(Sampler):
    """Shuffle each full epoch independently and resume at an optimizer-step boundary.

    Prefetched batches do not advance the persisted cursor: only the caller's completed
    optimizer steps do. Dataset positions refer to the fixed training-only subset.
    """
    def __init__(self, size, batch_size, seed):
        self.size, self.batch_size, self.seed = int(size), int(batch_size), int(seed)
        self.steps_per_epoch = self.size // self.batch_size
        if self.steps_per_epoch < 1:
            raise ValueError("not enough examples for one complete training batch")
        self.set_step(0)

    def set_step(self, completed_steps):
        if completed_steps < 0:
            raise ValueError("completed steps cannot be negative")
        self.epoch, self.offset = divmod(int(completed_steps), self.steps_per_epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.size, generator=generator).tolist()
        for batch in range(self.offset, self.steps_per_epoch):
            begin = batch * self.batch_size
            yield order[begin:begin + self.batch_size]

    def __len__(self):
        return self.steps_per_epoch - self.offset


def atomic_torch_save(payload, path):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_training_state(path, *, model, optimizer, scheduler, step, identity, history):
    """The single recovery authority; model exports may be recreated from this bundle."""
    atomic_torch_save({
        "protocol_version": 1, "step": int(step), "identity": identity,
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
        "history": history,
    }, path)


def verify_resume_cadence(path, push_every):
    """The adjacent immutable recipe preserves legacy bundles without changing identity.

    Validation evaluates SIGReg and therefore consumes the model RNG. A different
    checkpoint interval changes the subsequent training trajectory even with restored RNG.
    """
    from omegaconf import OmegaConf

    source_config = Path(path).parent / "config.yaml"
    if not source_config.is_file():
        raise ValueError("resume requires sibling config.yaml to verify the original checkpoint interval")
    original = OmegaConf.to_container(OmegaConf.load(source_config)).get("push_every")
    if type(original) is not int or original < 1 or original != push_every:
        raise ValueError(f"resume checkpoint interval differs: use original --push-every {original!r}")
    return original


def restore_training_state(path, *, model, optimizer, scheduler, identity):
    # Only explicitly supplied locally produced bundles are loaded; never arbitrary input.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("protocol_version") != 1 or state.get("identity") != identity:
        raise ValueError("resume checkpoint data, split, architecture or training recipe differs")
    if not 0 < state["step"] < identity["total_steps"]:
        raise ValueError("resume checkpoint must be an unfinished training step")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"]:
        if not torch.cuda.is_available() or len(state["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("resume CUDA device count differs")
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    return state["step"], state["history"]


def upload_checkpoint(store, environment, name, run_dir, *, run_id, final):
    """Bounded retries; local atomic recovery state survives an exhausted upload."""
    for attempt in range(3):
        try:
            return store.upload_run(environment, name, run_dir, run_id=run_id, tag=final)
        except Exception as exc:
            if attempt == 2:
                raise
            print(f"[s2] checkpoint upload attempt {attempt + 1} failed ({type(exc).__name__}); retrying", flush=True)
            time.sleep(5 * (attempt + 1))


def snapshot_sources(repo_root, run_dir):
    """Keep the executed local training sources beside each durable checkpoint."""
    relative_paths = ["experiments/scripts/walker_s2_train_lewm.py"] + [
        f"experiments/helpers/{name}.py" for name in (
            "walkerTrainingState", "walkerLewm", "walkerProtocol", "locoData", "locoEnv",
            "threads", "hfStore", "runManifest", "storageBudget",
        )
    ]
    hashes = {}
    for relative in relative_paths:
        data = (Path(repo_root) / relative).read_bytes()
        output = Path(run_dir) / "source_snapshot" / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(data)
        hashes[relative] = hashlib.sha256(data).hexdigest()
    (Path(run_dir) / "source_snapshot" / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    return hashes
