"""Mechanical PushT inputs for the shared recurrent action-adapter family."""

import numpy as np


def features(current, goal, state, hazard, base, previous, displacement, step):
    def normalize(vector):
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        return (vector - vector.mean()) / np.sqrt(vector.var() + 1e-5)

    state = np.asarray(state, dtype=np.float32)
    value = np.concatenate(
        [
            normalize(current),
            normalize(goal),
            state[:2] / 256 - 1,
            (state[2:4] - state[:2]) / 512,
            np.asarray([np.sin(state[4]), np.cos(state[4])], dtype=np.float32),
            state[5:7] / 100,
            (np.asarray(hazard, dtype=np.float32) - state[:2]) / 512,
            np.asarray([33 / 512], dtype=np.float32),
            base,
            previous,
            np.asarray(displacement, dtype=np.float32) / 100,
            np.asarray([step / 280], dtype=np.float32),
        ]
    ).astype(np.float32)
    assert value.shape == (402,) and np.isfinite(value).all()
    return value


def distinct_start(a, b):
    angle = abs(float(a[4] - b[4])) % (2 * np.pi)
    angle = min(angle, 2 * np.pi - angle)
    return np.linalg.norm(np.asarray(a[:4]) - b[:4]) >= 40 or angle >= np.pi / 9
