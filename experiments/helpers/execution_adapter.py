"""Bounded, causal joint-reference correction inspired by APEX (2606.16504).

This is our specified discrete implementation, not verified author code. See the
adapter05 protocol for its normalized projected update and all departures.
"""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class AdapterConfig:
    mode: str = "native"
    dt: float = 0.05
    correction_limit: float = 0.10
    fixed_coefficients: tuple = (0.2, 0.0, 0.0)
    learning_rates: tuple = (30.0, 0.1, 0.1)
    coefficient_lower: tuple = (0.0, -0.05, -0.01)
    coefficient_upper: tuple = (0.6, 0.05, 0.01)
    sliding_gain: float = 30.0
    reference_velocity_limit: float = 2.0
    reference_acceleration_limit: float = 20.0

    def __post_init__(self):
        if self.mode not in {"native", "fixed", "adaptive"}:
            raise ValueError("Unknown adapter mode")
        values = [v for v in asdict(self).values() if not isinstance(v, str)]
        if not all(np.isfinite(np.asarray(v)).all() for v in values):
            raise ValueError("Adapter parameters must be finite")
        if self.dt != 0.05 or self.correction_limit < 0:
            raise ValueError("Expected 20 Hz and a nonnegative correction bound")
        for key in (
            "fixed_coefficients",
            "learning_rates",
            "coefficient_lower",
            "coefficient_upper",
        ):
            if np.shape(getattr(self, key)) != (3,):
                raise ValueError(f"{key} must have three entries")
        if (
            np.any(np.asarray(self.coefficient_lower) > self.coefficient_upper)
            or np.any(np.asarray(self.learning_rates) < 0)
            or min(
                self.sliding_gain, self.reference_velocity_limit, self.reference_acceleration_limit
            )
            <= 0
        ):
            raise ValueError("Invalid adaptive bounds or derivative limits")


class ExecutionAdapter:
    """One controller per episode, seven arm joints, untouched eighth coordinate.

    The correction uses coefficients from the previous update. Current feedback
    updates the coefficients for the next control, never using a future state.
    """

    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.coefficients = np.zeros(3, dtype=np.float64)
        self.previous_reference = None
        self.previous_reference_velocity = None
        self.next_step = 0

    def apply(self, command, q, dq, step):
        command = np.asarray(command)
        q, dq = np.asarray(q, dtype=np.float64), np.asarray(dq, dtype=np.float64)
        if command.shape != (8,) or q.shape != (7,) or dq.shape != (7,):
            raise ValueError("Expected an 8-D command and seven measured arm joints")
        if command.dtype.kind != "f":
            raise ValueError("Commands must use a floating dtype")
        if not all(np.isfinite(v).all() for v in (command, q, dq)):
            raise ValueError("Non-finite command or robot feedback")
        if step != self.next_step:
            raise ValueError("Non-contiguous control step; reset between episodes")
        cfg = self.config
        reference = command[:7].astype(np.float64)
        error = reference - q
        velocity = np.zeros(7)
        acceleration = np.zeros(7)
        if self.previous_reference is not None:
            velocity = np.clip(
                (reference - self.previous_reference) / cfg.dt,
                -cfg.reference_velocity_limit,
                cfg.reference_velocity_limit,
            )
        if self.previous_reference_velocity is not None:
            acceleration = np.clip(
                (velocity - self.previous_reference_velocity) / cfg.dt,
                -cfg.reference_acceleration_limit,
                cfg.reference_acceleration_limit,
            )
        features = np.stack((error, dq, acceleration), axis=1)
        coefficients = self.coefficients.copy()
        if cfg.mode == "fixed":
            coefficients = np.asarray(cfg.fixed_coefficients, dtype=np.float64)
        raw_correction = features @ coefficients
        correction = np.clip(raw_correction, -cfg.correction_limit, cfg.correction_limit)
        output = command.copy()
        if cfg.mode != "native":
            output[:7] = reference + correction
        else:
            correction[:] = 0
            raw_correction[:] = 0
        if cfg.mode == "adaptive":
            sliding_error = dq - velocity - cfg.sliding_gain * error
            gradient = np.mean(features * sliding_error[:, None], axis=0)
            normalization = 1.0 + np.mean(features**2, axis=0)
            self.coefficients = np.clip(
                coefficients - cfg.dt * np.asarray(cfg.learning_rates) * gradient / normalization,
                cfg.coefficient_lower,
                cfg.coefficient_upper,
            )
        self.previous_reference = reference.copy()
        self.previous_reference_velocity = velocity.copy()
        self.next_step += 1
        return output, {
            "step": step,
            "mode": cfg.mode,
            "reference": reference,
            "q": q,
            "dq": dq,
            "tracking_error": error,
            "reference_velocity": velocity,
            "reference_acceleration": acceleration,
            "coefficients": coefficients,
            "next_coefficients": self.coefficients.copy(),
            "raw_correction": raw_correction,
            "correction": correction,
            "saturated": np.abs(raw_correction) > cfg.correction_limit,
            "native_command": command.copy(),
            "corrected_command": output.copy(),
        }
