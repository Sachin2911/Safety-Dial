"""Four frozen execution interventions using proposals and proprioception only."""

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ScreenConfig:
    mode: str = "native"
    scale: float = 0.5
    blend_start: float = 0.5
    quiet_intervals: int = 16
    quiet_range_rad: float = 0.02

    def __post_init__(self):
        if self.mode not in {"native", "h4", "slow", "smooth", "gripwait"}:
            raise ValueError("Unknown screen arm")
        if (self.scale, self.blend_start, self.quiet_intervals, self.quiet_range_rad) != (
            0.5, 0.5, 16, 0.02
        ):
            raise ValueError("Task-8 presets are frozen")

    @property
    def horizon(self):
        return 4 if self.mode == "h4" else 8


class Controller:
    def __init__(self, config):
        self.config = config
        self.current = None
        self.previous = None
        self.chunk_step = None
        self.next_step = 0
        self.scaling_active = False
        self.positions = deque(maxlen=config.quiet_intervals + 1)

    def proposal(self, step, chunk):
        value = np.asarray(chunk)
        if value.shape != (16, 8) or not np.isfinite(value).all():
            raise ValueError("Invalid native chunk")
        if step != self.next_step or step % self.config.horizon:
            raise ValueError("Proposal is out of sequence")
        if self.chunk_step == step:
            raise ValueError("Duplicate proposal")
        self.previous, self.current = self.current, value.copy()
        self.chunk_step = step

    def apply(self, step, command, q):
        native = np.asarray(command)
        q = np.asarray(q, dtype=np.float64)
        if native.shape != (8,) or q.shape != (7,):
            raise ValueError("Wrong command or joint-position shape")
        if not np.isfinite(native).all() or not np.isfinite(q).all():
            raise ValueError("Nonfinite control input")
        if native.dtype.kind != "f":
            raise ValueError("Floating command dtype is required")
        if step != self.next_step or self.chunk_step is None:
            raise ValueError("Control step is out of sequence")
        offset = step - self.chunk_step
        if not 0 <= offset < self.config.horizon:
            raise ValueError("Missing fresh proposal")
        expected = self.current[offset].copy()
        expected[-1] = np.sign(expected[-1]) if abs(expected[-1]) > 0.01 else -1.0
        if not np.array_equal(native, expected):
            raise ValueError("Native conversion differs from the raw proposal")
        self.positions.append(q.copy())
        result = native.copy()
        denied = False
        if self.config.mode == "slow":
            self.scaling_active |= bool(native[-1] <= 0)
            if self.scaling_active:
                result[:7] = q + self.config.scale * (native[:7] - q)
        elif self.config.mode == "smooth" and self.previous is not None:
            weight = self.config.blend_start + (1 - self.config.blend_start) * offset / 7
            result[:7] = weight * native[:7] + (1 - weight) * self.previous[8 + offset, :7]
        elif self.config.mode == "gripwait" and native[-1] <= 0:
            quiet = len(self.positions) == self.positions.maxlen and (
                np.ptp(np.asarray(self.positions), axis=0).max() <= self.config.quiet_range_rad
            )
            if not quiet:
                result[-1] = 1.0
                denied = True
        self.next_step += 1
        trace = {
            "step": step,
            "offset": offset,
            "native_command": native.copy(),
            "modified_command": result.copy(),
            "q": q,
            "command_dtype": str(native.dtype),
            "changed": bool(not np.array_equal(result, native)),
            "gripper_close_denied": denied,
            "scaling_active": bool(self.scaling_active),
            "position_history_length": len(self.positions),
        }
        return result, trace

    def settings(self):
        return asdict(self.config)
