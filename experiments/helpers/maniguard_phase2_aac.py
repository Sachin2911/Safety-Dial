"""Pure AAC-joint scheduling; no policy calls, simulator or outcome predicates.

This is an explicitly adapted implementation of Adaptive Action Chunking,
arXiv:2604.04161v2. The author source uses Cartesian deltas, separate position
and rotation Gaussians, and a Cartesian movement floor. Here one Gaussian
covers seven absolute joint targets; movement is endpoint displacement from
measured joints. Settings are proposals pending training-only feasibility.

Prefix means, first argmax indexing, strict movement-threshold crossing and
sample-zero execution follow the pinned pi0.5 selector. A singleton chunk
executes one action, an extension of the upstream undefined singleton case.
Float64 prefix means deliberately retain the upstream rounding behavior even
for constant step entropy. This is neither a safety filter nor a reproduction
of the paper's published benchmark results.

Upstream provenance: Adaptive-Action-Chunking/libero at
ff8c8e421f7805b5845a72d07fbd8f1171df74bf,
action_optimization/action_entropy_pi05.py. Repository license:

MIT License
Copyright (c) 2023 Lifelong Robot Learning

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SOURCE_REVISION = "ff8c8e421f7805b5845a72d07fbd8f1171df74bf"
SOURCE_SELECTOR_SHA256 = "afccedc7198187373dfba9a967544f46f482328a9578933d65967ddfdb75ae74"
SOURCE_HELPER_SHA256 = "d162661c8f5a4ace9c014247a53ddcd9bdc14686dffbfd8ac4d0e9af6f217b36"
CONFIG_SCHEMA = "maniguard-phase2-aac-joint-v1"
COMMON_PREFIX = 16


def _real(value, name, *, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float))
            or not np.isfinite(value) or (value <= 0 if positive else value < 0)):
        bound = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a finite {bound} scalar")
    return float(value)


def _array(value, name):
    result = np.asarray(value)
    if result.dtype.kind not in "fiu" or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite real numeric data")
    return result


@dataclass(frozen=True)
class AACConfig:
    """Shared, unfrozen engineering defaults, not empirically selected settings.

    With scale .2 radians, movement threshold 1 means an endpoint joint-space
    L2 displacement of .2 radians, before the optional gripper contribution.
    A missing threshold disables the movement floor for diagnostic tests only.
    """

    sample_count: int = 20
    max_prefix: int = COMMON_PREFIX
    joint_scale_radians: float = 0.2
    covariance_epsilon: float = 1e-6
    movement_threshold: float | None = 1.0
    gripper_movement_weight: float = 0.2
    gripper_entropy_weight: float = 1.0

    def __post_init__(self):
        if type(self.sample_count) is not int or self.sample_count < 2:
            raise ValueError("sample_count must be an integer of at least two")
        if type(self.max_prefix) is not int or self.max_prefix != COMMON_PREFIX:
            raise ValueError("AAC-joint uses the common maximum prefix of sixteen actions")
        for name in ("joint_scale_radians", "covariance_epsilon"):
            _real(getattr(self, name), name, positive=True)
        for name in ("gripper_movement_weight", "gripper_entropy_weight"):
            _real(getattr(self, name), name)
        if self.movement_threshold is not None:
            _real(self.movement_threshold, "movement_threshold")

    def as_dict(self):
        return asdict(self)


def load_config(path):
    """Read the proposed settings with an explicit schema and upstream source pin."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise TypeError("AAC-joint configuration must be an object")
    if data.get("schema") != CONFIG_SCHEMA:
        raise ValueError("Unsupported AAC-joint configuration schema")
    source = data.get("source", {})
    if not isinstance(source, dict):
        raise TypeError("AAC-joint source must be an object")
    if (source.get("revision") != SOURCE_REVISION
            or source.get("selector_sha256") != SOURCE_SELECTOR_SHA256
            or source.get("helper_sha256") != SOURCE_HELPER_SHA256):
        raise ValueError("AAC-joint source pin mismatch")
    if not isinstance(data.get("settings"), dict):
        raise TypeError("AAC-joint configuration requires a settings object")
    try:
        return AACConfig(**data["settings"])
    except TypeError as error:
        raise ValueError("Invalid AAC-joint setting names") from error


@dataclass(frozen=True)
class EntropyProfile:
    continuous: np.ndarray
    gripper: np.ndarray
    total: np.ndarray
    cumulative_mean: np.ndarray


def entropy_profile(continuous, binary_gripper, *, group_sizes, covariance_epsilon=1e-6,
                    gripper_weight=1.0):
    """Gaussian block entropies plus Bernoulli entropy, using sample covariance.

    Inputs are normalized continuous samples [N,T,D] and binary gripper
    samples [N,T]. Groups (3,3) allow native-format parity; AAC-joint uses (7,).
    No variance surrogate, fitted density, or task-specific normalization is used.
    """
    values = _array(continuous, "continuous samples").astype(np.float64)
    grip = _array(binary_gripper, "binary gripper")
    if values.ndim != 3 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("continuous samples must have shape [N>=2,T>=1,D]")
    if grip.shape != values.shape[:2] or not np.isin(grip, [0, 1]).all():
        raise ValueError("binary gripper must have shape [N,T] and values zero or one")
    if (not isinstance(group_sizes, tuple) or not group_sizes
            or any(type(size) is not int or size < 1 for size in group_sizes)
            or sum(group_sizes) != values.shape[2]):
        raise ValueError("group_sizes must partition the continuous dimensions")
    epsilon = _real(covariance_epsilon, "covariance_epsilon", positive=True)
    weight = _real(gripper_weight, "gripper_weight")
    arm = np.zeros(values.shape[1], dtype=np.float64)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            start = 0
            for size in group_sizes:
                for t in range(values.shape[1]):
                    samples = values[:, t, start:start + size]
                    covariance = np.atleast_2d(np.cov(samples, rowvar=False))
                    covariance += epsilon * np.eye(size)
                    factor = np.linalg.cholesky(covariance)
                    logdet = 2 * np.log(np.diag(factor)).sum()
                    arm[t] += 0.5 * (size * np.log(2 * np.pi * np.e) + logdet)
                start += size
            probability = np.clip(grip.mean(axis=0), 1e-9, 1 - 1e-9)
            gripper = -probability * np.log(probability)
            gripper -= (1 - probability) * np.log(1 - probability)
            total = arm + weight * gripper
            # np.mean over each prefix matches the author's floating-point path.
            means = np.array([np.mean(total[:k]) for k in range(1, len(total) + 1)])
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise ValueError("Entropy computation failed for the supplied samples") from error
    _array(total, "computed entropy")
    _array(means, "computed cumulative mean entropy")
    return EntropyProfile(arm, gripper, total, means)


def select_horizon(cumulative_mean, movement_magnitudes, movement_threshold):
    """Return (chosen, entropy, movement) horizons, all one-based action counts.

    movement_magnitudes[k-1] describes the prefix of k actions. Like upstream,
    search floor crossings from k=2, use strict >, and fall back to the full
    available horizon if movement never crosses the threshold. Disabled floors
    use min(2,T). No near-tie tolerance alters the author's first np.argmax.
    """
    means = _array(cumulative_mean, "cumulative mean entropy").astype(np.float64)
    movement = _array(movement_magnitudes, "movement magnitudes").astype(np.float64)
    if means.ndim != 1 or not len(means) or movement.shape != means.shape:
        raise ValueError("Entropy means and movement must be matching nonempty vectors")
    if np.any(movement < 0):
        raise ValueError("Movement magnitudes must be nonnegative")
    if movement_threshold is not None:
        _real(movement_threshold, "movement_threshold")
    if len(means) == 1:
        return 1, 1, 1
    with np.errstate(over="ignore"):
        differences = np.diff(means)
    _array(differences, "entropy mean differences")
    entropy_horizon = max(2, int(np.argmax(differences)) + 1)
    movement_horizon = 2
    if movement_threshold is not None:
        crossings = np.flatnonzero(movement[1:] > movement_threshold)
        movement_horizon = int(crossings[0]) + 2 if len(crossings) else len(means)
    return max(entropy_horizon, movement_horizon), entropy_horizon, movement_horizon


@dataclass(frozen=True)
class AACSelection:
    prefix: np.ndarray
    horizon: int
    entropy_horizon: int
    movement_horizon: int
    entropy: EntropyProfile
    movement_magnitudes: np.ndarray
    config: AACConfig

    def record(self):
        """Finite, JSON-compatible scheduling diagnostics; no future outcomes."""
        return {"method": "AAC-joint", "sample_index": 0, "horizon": self.horizon,
                "entropy_horizon": self.entropy_horizon,
                "movement_horizon": self.movement_horizon,
                "available_prefix": len(self.entropy.total),
                "sample_count": self.config.sample_count,
                "arm_entropy": self.entropy.continuous.tolist(),
                "gripper_entropy": self.entropy.gripper.tolist(),
                "total_entropy": self.entropy.total.tolist(),
                "cumulative_mean_entropy": self.entropy.cumulative_mean.tolist(),
                "movement_magnitudes": self.movement_magnitudes.tolist(),
                "settings": self.config.as_dict(), "source_revision": SOURCE_REVISION}


def select_joint_prefix(samples, current_joints, config=None):
    """Select a copied, otherwise bit-exact sample-zero prefix from [N,T,8].

    Only the first min(T,16) proposals are inspected. No averaging, clipping,
    smoothing, padding, four-step quantization, or gripper rewriting occurs.
    Entropy and movement use ManiGuard's executed open state: command > .01.
    Its enabled gripper_binarize closes zero and every command <= .01.
    Proposal generation, independent RNG streams and query accounting belong
    to a future integration, not this pure core.
    """
    config = AACConfig() if config is None else config
    if not isinstance(config, AACConfig):
        raise TypeError("config must be an AACConfig")
    proposals = np.asarray(samples)
    if (proposals.ndim != 3 or proposals.shape[0] != config.sample_count
            or proposals.shape[1] < 1 or proposals.shape[2] != 8):
        raise ValueError("samples must have shape [configured sample_count,T>=1,8]")
    proposals = _array(proposals[:, :COMMON_PREFIX], "accessible proposals")
    joints = _array(current_joints, "current joints").astype(np.float64)
    if joints.shape != (7,):
        raise ValueError("current joints must have shape [7]")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            relative = (proposals[:, :, :7].astype(np.float64) - joints)
            relative /= config.joint_scale_radians
            binary = (proposals[:, :, 7] > 0.01).astype(np.int64)
            movement = np.linalg.norm(relative[0], axis=1)
            toggled = np.r_[False, np.maximum.accumulate(np.diff(binary[0]) != 0)]
            movement += config.gripper_movement_weight * toggled
    except FloatingPointError as error:
        raise ValueError("Joint normalization or movement calculation overflowed") from error
    _array(relative, "normalized joint proposals")
    _array(movement, "movement magnitudes")
    profile = entropy_profile(relative, binary, group_sizes=(7,),
                              covariance_epsilon=config.covariance_epsilon,
                              gripper_weight=config.gripper_entropy_weight)
    horizon, entropy_horizon, movement_horizon = select_horizon(
        profile.cumulative_mean, movement, config.movement_threshold)
    return AACSelection(proposals[0, :horizon].copy(), horizon, entropy_horizon,
                        movement_horizon, profile, movement, config)
