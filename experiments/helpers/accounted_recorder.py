"""Task 2's passive recorder, with a crash-surviving event and state journal.

The evaluator adds scripts/todo2 to PYTHONPATH. No action is modified, and no
additional simulator step, render or robot-state read is introduced here.
"""

import json
import time
from pathlib import Path

import numpy as np
from record_rollout import RolloutRecorder


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


class AccountedRecorder(RolloutRecorder):
    def __init__(self, cfg, scene):
        super().__init__(cfg, scene)
        directory = Path(cfg.recording_output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        name = scene["name"].replace("/", "_")
        self.journal = (directory / f"{name}.journal.jsonl").open("x", buffering=1)
        self.journal_overhead_s = 0.0
        self.emit("observer_created")

    def emit(self, event, **values):
        started = time.monotonic()
        self.journal.write(json.dumps({"event": event, **values}, default=serializable) + "\n")
        self.journal_overhead_s += time.monotonic() - started

    def start(self, *args, **kwargs):
        super().start(*args, **kwargs)
        self.emit("start", meta=self.meta, state=self.rows[-1])

    def proposal(self, *args, **kwargs):
        super().proposal(*args, **kwargs)
        self.emit("proposal", **self.proposals[-1])

    def before_action(self, *args, **kwargs):
        super().before_action(*args, **kwargs)
        self.emit("before_action", **self.pending)

    def applied(self, step):
        super().applied(step)
        self.emit("applied", step=step)

    def oracle(self, *args, **kwargs):
        super().oracle(*args, **kwargs)
        self.emit("state", state=self.rows[-1])

    def finish(self, result):
        super().finish(result)
        self.emit("finish", result=result, journal_overhead_s=self.journal_overhead_s)
        self.journal.close()


def create_observer(cfg, scene):
    return AccountedRecorder(cfg, scene)
