"""State-only trajectory collection for the locomotion velocity tasks.

Why state and not pixels
------------------------
Rendering 224x224x3 uint8 costs 150 KB per frame, so a 3.25M step dataset is 489 GB. There is
roughly 60 GB free on a shared overlay filesystem, so that does not fit, and even a JPEG variant
would be 40 GB.

MuJoCo rendering is a pure function of `(qpos, qvel)`: `set_state` calls `mj_forward`, and a
replay from a restored state reproduces the original observations to ~1e-13 (see
`experiments/scripts/verify_replay.py`). Walker2d has `na == 0`, so `(qpos, qvel)` really is the
complete state, with no actuator activations to carry. Storing state costs 264 bytes per step,
so the same 3.25M steps is ~312 MB raw and ~180 MB compressed, and frames are regenerated on
demand at ~3100 per second.

`x_velocity` is stored as float64 deliberately. It makes `cost` re-derivable at any threshold, so
if Gate 0 fails at the v1 constant of 2.3415 the cost channel can be recomputed at the v0
constant, or at a measured quantile of the trained policy's velocity distribution, without
recollecting anything. It is the cheapest insurance in the design.

The layout is flat timestep arrays plus `ep_offset` and `ep_len`, matching the Push-T convention
documented in `notes/pushTDataExp.md`, so `swm.data.HDF5Dataset` opens these files unmodified.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np

# Column name -> (dtype, per-step shape). `None` shape means scalar per step.
SCHEMA: dict[str, tuple[str, tuple[int, ...] | None]] = {
    "qpos": ("f8", None),  # f8 is required: f4 breaks bit-reproducible re-rendering
    "qvel": ("f8", None),
    "action": ("f4", None),
    "observation": ("f4", None),
    "x_velocity": ("f8", ()),
    "reward": ("f4", ()),
    "cost": ("u1", ()),
    "healthy": ("u1", ()),  # is_healthy AT this row. The real Mode B signal; see below.
    "terminated": ("u1", ()),
    "truncated": ("u1", ()),
    "episode_idx": ("i4", ()),
    "step_idx": ("i4", ()),
    "policy_id": ("u1", ()),
}

#: Written by `annotate_steps_to_termination`, not during the rollout itself.
DERIVED = {"steps_to_termination": ("i4", ())}

_BLOSC2 = dict(
    hdf5plugin.Blosc2(cname="zstd", clevel=5, filters=hdf5plugin.Blosc2.BITSHUFFLE)
)


def disk_guard(path: str | Path, need_gb: float = 5.0) -> float:
    """Refuse to start a collection that would fill the disk. Returns free GB."""
    free_gb = shutil.disk_usage(Path(path).parent).free / 1e9
    if free_gb < need_gb:
        raise RuntimeError(
            f"only {free_gb:.1f} GB free at {path}, need {need_gb:.1f} GB. "
            "First reclaim: data/raw/pusht_expert_train.h5.zst (13 GB, re-downloadable)."
        )
    return free_gb


class StateHDF5Writer:
    """Append-only writer for the state schema.

    `swm.data.HDF5Writer` is not used because its `_init_schema` hardcodes `chunks=(1, *shape)`
    and no compression, which at a million steps means a million single-row chunks per column:
    pathological B-tree overhead and no compression at all. This writes the byte-identical key
    layout with 8192-row chunks and Blosc2/zstd instead.
    """

    def __init__(self, path, dims: dict[str, int], attrs: dict | None = None, chunk=8192):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dims = dims
        self.chunk = int(chunk)
        self.f = h5py.File(self.path, "w")
        self.n = 0
        self.ep_lens: list[int] = []
        for name, (dtype, shape) in SCHEMA.items():
            per_step = (dims[name],) if shape is None else shape
            self.f.create_dataset(
                name,
                shape=(0, *per_step),
                maxshape=(None, *per_step),
                dtype=dtype,
                chunks=(self.chunk, *per_step),
                **_BLOSC2,
            )
        for k, v in (attrs or {}).items():
            self.f.attrs[k] = v

    def write_episode(self, cols: dict[str, np.ndarray]) -> int:
        """Append one episode. `cols` holds one array per SCHEMA key, all the same length."""
        lengths = {len(v) for v in cols.values()}
        if len(lengths) != 1:
            raise ValueError(f"ragged episode: column lengths {lengths}")
        n_new = lengths.pop()
        if n_new == 0:
            return 0
        for name in SCHEMA:
            ds = self.f[name]
            ds.resize(self.n + n_new, axis=0)
            ds[self.n : self.n + n_new] = cols[name]
        self.n += n_new
        self.ep_lens.append(n_new)
        return n_new

    def close(self) -> None:
        # Truncate every column to the number of rows actually committed. `write_episode`
        # resizes and then assigns, so an exception between those two steps leaves the dataset
        # longer than `self.n` and the file reads back ragged, with garbage rows that no
        # `ep_offset` points at. Truncating here makes a partial write a short file rather than
        # a corrupt one.
        for name in SCHEMA:
            if self.f[name].shape[0] != self.n:
                self.f[name].resize(self.n, axis=0)

        lens = np.asarray(self.ep_lens, dtype="i4")
        offsets = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype("i8") if len(lens) else (
            np.zeros(0, "i8")
        )
        self.f.create_dataset("ep_len", data=lens)
        self.f.create_dataset("ep_offset", data=offsets)
        self.f.attrs["n_steps"] = int(self.n)
        self.f.attrs["n_episodes"] = int(len(lens))
        self.f.attrs["schema_version"] = 1
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.f:
            self.close()
        return False


def rollout_episode(
    env,
    policy,
    seed: int | None = None,
    *,
    stop_after_unhealthy: int | None = None,
) -> dict[str, np.ndarray]:
    """Run one episode, returning the state schema columns.

    Records the state *before* each action, so row `t` holds `(s_t, a_t)` and the outcome of
    applying `a_t`. That is the convention the world model and the oracle both want: the oracle
    restores `qpos`/`qvel` from row `t` and asks what could have happened instead.

    **`terminated[t]` therefore means "action t made the robot unhealthy", while `qpos[t]` is
    still the last HEALTHY state.** This is easy to get wrong and it matters enormously here: an
    earlier version of the oracle validation scored states at `terminated == 1` and found 96% of
    them recoverable, which looked like the project's premise collapsing. They were in fact
    healthy robots standing at a 55 degree lean, and of course those recover. Use the `healthy`
    column, not `terminated`, to identify failure states.

    Worse, under the benchmark's own termination the episode STOPS at that point, so the
    unhealthy state is never recorded at all: every row in such a dataset is healthy. A project
    about irreversibility cannot be built on data that contains no failures.

    `stop_after_unhealthy` is the fix. Build the env with `terminate_when_unhealthy=False` and
    pass a step count; the rollout then continues that many steps past the moment the benchmark
    would have stopped it, so the trajectory actually reaches the ground and the genuinely
    irrecoverable states exist in the data. `terminated` is then all-zero and `healthy` carries
    the Mode B signal, from which the benchmark's flag is exactly recoverable.
    """
    u = env.unwrapped
    obs, _ = env.reset(seed=seed)
    policy.reset()

    rec: dict[str, list] = {k: [] for k in SCHEMA}
    terminated = truncated = False
    step_idx = 0
    first_unhealthy = -1

    while not (terminated or truncated):
        qpos = u.data.qpos.copy()
        qvel = u.data.qvel.copy()
        healthy = bool(getattr(u, "is_healthy", True))
        if not healthy and first_unhealthy < 0:
            first_unhealthy = step_idx
        if (
            stop_after_unhealthy is not None
            and first_unhealthy >= 0
            and step_idx - first_unhealthy >= stop_after_unhealthy
        ):
            break

        action = policy.act(obs)
        nxt, reward, terminated, truncated, info = env.step(action)

        rec["qpos"].append(qpos)
        rec["qvel"].append(qvel)
        rec["action"].append(action)
        rec["observation"].append(obs)
        rec["x_velocity"].append(info["x_velocity"])
        rec["reward"].append(reward)
        rec["cost"].append(info["cost"])
        rec["healthy"].append(healthy)
        rec["terminated"].append(terminated)
        rec["truncated"].append(truncated)
        rec["step_idx"].append(step_idx)
        rec["policy_id"].append(policy.policy_id)
        rec["episode_idx"].append(0)  # filled in by the caller

        obs = nxt
        step_idx += 1

    return {k: np.asarray(v, dtype=SCHEMA[k][0]) for k, v in rec.items()}


def collect(
    env,
    policy,
    writer: StateHDF5Writer,
    *,
    n_episodes: int | None = None,
    n_steps: int | None = None,
    seed: int = 0,
    progress_every: int = 50,
    verbose: bool = True,
) -> dict:
    """Roll out until `n_episodes` or `n_steps` is reached, writing each episode as it ends."""
    if (n_episodes is None) == (n_steps is None):
        raise ValueError("pass exactly one of n_episodes or n_steps")

    t0 = time.time()
    ep = 0
    total = 0
    returns, lengths, falls, costs = [], [], [], []

    while True:
        if n_episodes is not None and ep >= n_episodes:
            break
        if n_steps is not None and total >= n_steps:
            break

        cols = rollout_episode(env, policy, seed=seed + ep)
        cols["episode_idx"][:] = ep
        writer.write_episode(cols)

        n = len(cols["reward"])
        total += n
        returns.append(float(cols["reward"].sum()))
        lengths.append(n)
        # A fall is termination, not truncation: the time limit is not a safety event.
        falls.append(bool(cols["terminated"][-1]))
        costs.append(int(cols["cost"].sum()))
        ep += 1

        if verbose and progress_every and ep % progress_every == 0:
            rate = total / max(time.time() - t0, 1e-9)
            print(
                f"[collect] ep {ep} steps {total} "
                f"fall_rate {np.mean(falls):.3f} cost_rate {np.sum(costs)/total:.4f} "
                f"{rate:.0f} steps/s"
            )

    elapsed = time.time() - t0
    stats = {
        "n_episodes": ep,
        "n_steps": total,
        "fall_rate": float(np.mean(falls)) if falls else 0.0,
        "cost_rate": float(np.sum(costs) / total) if total else 0.0,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "elapsed_s": elapsed,
        "steps_per_s": total / max(elapsed, 1e-9),
    }
    if verbose:
        print(f"[collect] done: {json.dumps(stats, indent=None)}")
    return stats


def annotate_steps_to_failure(path) -> None:
    """Add `steps_to_failure`: rows until this episode's first unhealthy state.

    Counts down to the first row with `healthy == 0`. Episodes that never go unhealthy get a
    sentinel of -1, because their last row is not a failure and treating it as one would poison
    every label downstream. Rows at or after the first unhealthy state get 0 or negative values,
    so "already failed" is distinguishable from "about to fail".

    Defined on `healthy` rather than `terminated` deliberately. Under termination-disabled
    collection `terminated` is all-zero, and even under the benchmark's own termination it is
    offset by one row from the actual failure (see `rollout_episode`).

    This column is an EVALUATION label. It must never be fed to an estimator.
    """
    with h5py.File(path, "a") as f:
        n = int(f.attrs["n_steps"])
        ep_len = f["ep_len"][:]
        ep_off = f["ep_offset"][:]
        healthy = f["healthy"][:] if "healthy" in f else None
        terminated = f["terminated"][:]

        stf = np.full(n, -1, dtype="i4")
        for off, ln in zip(ep_off, ep_len):
            if ln == 0:
                continue
            sl = slice(int(off), int(off) + int(ln))
            if healthy is not None and (healthy[sl] == 0).any():
                first = int(np.argmax(healthy[sl] == 0))
            elif terminated[int(off) + int(ln) - 1]:
                # Benchmark-terminated episode with no `healthy` column: the failure happened one
                # step after the last recorded row.
                first = int(ln)
            else:
                continue
            stf[sl] = first - np.arange(int(ln), dtype="i4")

        if "steps_to_failure" in f:
            del f["steps_to_failure"]
        # h5py rejects a chunk shape larger than the dataset, so an empty or tiny file needs the
        # chunk clamped and compression dropped entirely at zero rows.
        if n == 0:
            f.create_dataset("steps_to_failure", data=stf)
        else:
            f.create_dataset(
                "steps_to_failure", data=stf, chunks=(min(65536, n),), **_BLOSC2
            )
        f.attrs["has_steps_to_failure"] = True


#: Old name, kept so existing call sites do not silently skip the annotation pass.
annotate_steps_to_termination = annotate_steps_to_failure
