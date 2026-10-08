"""Episode-level training partitions and immutable inputs for new Walker2d runs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def episode_partition(clip_indices, *, seed: int, val_fraction: float = 0.02):
    """Keep every overlapping clip and sibling of an episode in the same role."""
    if not 0 < val_fraction < 1:
        raise ValueError("validation fraction must be strictly between zero and one")
    episode_ids = np.array(sorted({int(ep) for ep, _ in clip_indices}), dtype=np.int64)
    if len(episode_ids) < 2:
        raise ValueError("training needs at least two episodes with usable clips")
    shuffled = np.random.default_rng(seed).permutation(episode_ids)
    n_val = min(len(shuffled) - 1, max(1, int(np.ceil(len(shuffled) * val_fraction))))
    validation = set(map(int, shuffled[:n_val]))
    training = set(map(int, shuffled[n_val:]))
    tr_idx, val_idx = [], []
    for idx, (ep, _) in enumerate(clip_indices):
        (val_idx if int(ep) in validation else tr_idx).append(idx)
    return tr_idx, val_idx, {"training": sorted(training), "validation": sorted(validation)}


def training_action_scaler(dataset, episode_ids):
    """Fit normalisation using only training episodes, in contiguous slices."""
    actions = dataset.get_col_data("action")
    count = 0
    total = np.zeros(actions.shape[1], dtype=np.float64)
    squares = np.zeros_like(total)
    for ep in episode_ids:
        start = int(dataset.offsets[ep])
        end = start + int(dataset.lengths[ep])
        a = np.asarray(actions[start:end], dtype=np.float64)
        a = a[np.isfinite(a).all(axis=1)]
        count += len(a)
        total += a.sum(axis=0)
        squares += np.square(a).sum(axis=0)
    if count == 0:
        raise ValueError("no finite training actions")
    mean = total / count
    std = np.sqrt(np.maximum(squares / count - mean * mean, 0))
    return mean, np.maximum(std, 1e-8)


def root_roles(n_episodes: int) -> dict[str, list[int]]:
    return {
        "development": [e for e in range(n_episodes) if e % 4 == 0],
        "test": [e for e in range(n_episodes) if e % 4 == 1],
        "acquisition": [e for e in range(n_episodes) if e % 4 in (2, 3)],
    }


def exact_replacement(lengths, training_episodes, replacement_steps: int, *, seed: int):
    """Remove exactly N training rows; retain validation and episode boundaries.

    A partially retained source contributes only its initial contiguous prefix.
    Replacing it never joins unrelated trajectories into a model training clip.
    """
    kept = np.asarray(lengths, dtype=np.int64).copy()
    episodes = np.asarray(sorted(set(map(int, training_episodes))), dtype=np.int64)
    if replacement_steps < 1 or replacement_steps > int(kept[episodes].sum()):
        raise ValueError("replacement exceeds available ordinary training steps")
    remaining = int(replacement_steps)
    for ep in np.random.default_rng(seed).permutation(episodes):
        remove = min(remaining, int(kept[ep]))
        kept[ep] -= remove
        remaining -= remove
        if remaining == 0:
            break
    assert int(np.asarray(lengths).sum() - kept.sum()) == replacement_steps
    return kept


def indices_for_episode_roles(clip_indices, roles):
    training = set(map(int, roles["training"]))
    validation = set(map(int, roles["validation"]))
    if not training or not validation or training & validation:
        raise ValueError("training and validation episode roles must be nonempty and disjoint")
    available = {int(ep) for ep, _ in clip_indices}
    if not available <= training | validation:
        raise ValueError("explicit episode roles omit usable source episodes")
    return ([i for i, (ep, _) in enumerate(clip_indices) if ep in training],
            [i for i, (ep, _) in enumerate(clip_indices) if ep in validation])
