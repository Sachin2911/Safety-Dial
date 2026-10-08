"""Recover strictly preceding action blocks for the original cached cloning rows."""
import h5py
import hdf5plugin  # noqa: F401
import numpy as np


def cached_previous_actions(path, episodes, targets, *, stride=5):
    episodes, targets = np.asarray(episodes), np.asarray(targets)
    if targets.shape != (len(episodes), 60):
        raise ValueError('expected 60-action targets for every cached row')
    previous = np.empty_like(targets, dtype=np.float32)
    with h5py.File(path, 'r') as f:
        offsets, lengths = f['ep_offset'][:], f['ep_len'][:]
        for episode in np.unique(episodes):
            lo, n = int(offsets[episode]), int(lengths[episode])
            actions = f['action'][lo:lo+n]
            rows = np.arange(0, n, stride)
            times = rows[(rows >= 10) & (rows + 10 <= n)]
            idx = np.flatnonzero(episodes == episode)
            if len(idx) != len(times):
                raise ValueError('cached row count does not match episode/stride reconstruction')
            wanted = np.stack([actions[t:t+10].reshape(60) for t in times]).astype(np.float32)
            if not np.array_equal(wanted, targets[idx].astype(np.float32)):
                raise ValueError('cached targets do not match reconstructed temporal alignment')
            previous[idx] = np.stack([actions[t-10:t].reshape(60) for t in times])
    return previous
