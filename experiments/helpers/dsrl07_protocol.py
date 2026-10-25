"""Fresh task-7 schedule and numerical comparisons, without model imports."""

import hashlib
import json

import numpy as np

from helpers.dsrl06_protocol import CAPS as CAPS
from helpers.dsrl06_protocol import benchmark_episode_seed, frozen_jobs as previous_jobs


def frozen_jobs():
    jobs = previous_jobs()
    for group in jobs.values():
        for job in group:
            job["seed"] += 10000
            seed = benchmark_episode_seed(job["task"], job["seed"])
            for key in ("episode_seed", "policy_rng_seed", "numpy_seed"):
                job[key] = seed
            payload = json.dumps(["dsrl07-v1", job["job_id"], "actor"], separators=(",", ":"))
            job["actor_rng_seed"] = (
                int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") & 0x7FFFFFFF
            )
            if job["job_id"] in ("E4", "E5"):
                job["admission_requires"] = ["numerical_integration_passed"]
    return jobs


def array_difference(left, right, *, atol, rtol=0):
    a, b = np.asarray(left), np.asarray(right)
    finite = bool(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all())
    if not finite:
        return {"finite_same_shape": False, "pass": False, "maximum_absolute_difference": None}
    difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {
        "finite_same_shape": True,
        "pass": bool(np.allclose(a, b, atol=atol, rtol=rtol)),
        "exact": bool(np.array_equal(a, b)),
        "maximum_absolute_difference": float(difference.max(initial=0)),
        "rms_difference": float(np.sqrt(np.mean(difference**2))) if difference.size else 0.0,
    }


def decoder_comparison(reference, replay):
    result = array_difference(reference, replay, atol=1e-5, rtol=1e-5)
    a, b = np.asarray(reference), np.asarray(replay)
    shape_valid = a.shape == (16, 8) and b.shape == (16, 8)
    result["gripper_prefix_equal"] = bool(
        shape_valid and np.array_equal(a[:8, 7] > 0.01, b[:8, 7] > 0.01)
    )
    result["pass"] &= result["gripper_prefix_equal"]
    return result


def candidate_summary(candidates):
    values = np.asarray(candidates)
    if values.shape != (14, 16, 8) or not np.isfinite(values).all():
        raise ValueError("Expected fourteen finite candidate chunks")
    grip = values[:, :8, 7]
    rms = np.sqrt(np.mean((values[:, :8, :7] - values[0, :8, :7]) ** 2, axis=(1, 2)))
    positions = np.flatnonzero((grip.max(0) >= 0.02) & (grip.min(0) <= 0))
    return {
        "sensitive_gripper_positions": positions.tolist(),
        "gripper_sensitive": bool(len(positions)),
        "arm_rms_from_zero": rms.tolist(),
        "arm_influence": bool(np.max(rms) > 1e-5),
        "gripper_min": float(grip.min()),
        "gripper_max": float(grip.max()),
    }
