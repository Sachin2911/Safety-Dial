"""Pin CPU thread pools to the container's cgroup quota. Import BEFORE torch.

Why this exists
---------------
Vast containers report every host core through `nproc` and `sched_getaffinity`, but
`/sys/fs/cgroup/cpu.max` caps the container far lower (this instance: 128 visible cores,
a 15.36-core quota). Torch, OpenMP and MKL size their pools from the visible count, so
they oversubscribe the quota and thrash: a CEM solve once took 15 s instead of 1 s with
the GPU idle ([notes/safeCEM.md]). `safe_dial_pusht.py` hard-coded 8 threads; this
module reads the quota so the same code behaves on any box.

Usage, as the very first import of every entry point:

    from helpers.threads import pin_threads
    pin_threads()          # sets OMP/MKL/OPENBLAS env vars, returns the thread count
    import torch           # torch.set_num_threads is applied lazily by `apply_torch`

Then, after torch is imported: `apply_torch()` (idempotent).
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
_OVERRIDE = "SAFETYDIAL_THREADS"


def cpu_quota_cores() -> float | None:
    """Cores allowed by the cgroup v2 quota, or None when unlimited/unknown."""
    p = Path("/sys/fs/cgroup/cpu.max")
    if not p.is_file():
        return None
    try:
        quota, period = p.read_text().split()
    except ValueError:
        return None
    if quota == "max":
        return None
    return int(quota) / int(period)


def visible_cores() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def choose_threads(max_threads: int = 8) -> int:
    """Thread count: an explicit override, else min(quota, visible, max_threads)."""
    override = os.environ.get(_OVERRIDE)
    if override:
        return max(1, int(override))
    quota = cpu_quota_cores()
    n = visible_cores()
    if quota is not None:
        n = min(n, int(quota))
    return max(1, min(n, max_threads))


def pin_threads(max_threads: int = 8) -> int:
    """Set the pool env vars (only if unset) and return the chosen count."""
    n = choose_threads(max_threads)
    for k in _ENV_KEYS:
        os.environ.setdefault(k, str(n))
    return n


def apply_torch(n: int | None = None) -> int:
    """Call after `import torch`. Uses the pinned count unless `n` is given."""
    import torch

    n = n or int(os.environ.get("OMP_NUM_THREADS", choose_threads()))
    torch.set_num_threads(n)
    return n


def describe() -> dict:
    return {
        "cpu_quota_cores": cpu_quota_cores(),
        "visible_cores": visible_cores(),
        "threads": int(os.environ.get("OMP_NUM_THREADS", choose_threads())),
    }
