"""Lazy-render dataset: stores MuJoCo state, produces pixels on demand.

The trade
---------
A 224x224x3 uint8 frame is 150,528 bytes; the state that generates it is 260. Storing pixels for
3.25M steps would be 489 GB against roughly 60 GB of free disk. Storing state is 0.84 GB, and
re-rendering costs about 0.32 ms per frame (measured: 3101 frames/s for `set_state` + `render`,
which is faster than collection because it skips physics entirely).

This is only sound because MuJoCo rendering is a pure function of `(qpos, qvel)`:
`MujocoEnv.set_state` calls `mj_forward`, and Walker2d has `na == 0` so there are no actuator
activations to carry. `experiments/scripts/verify_replay.py` checks both halves of that claim.

The rendering stack is the catch
--------------------------------
"Deterministic given the state" holds only within a fixed stack. A different driver, a different
EGL device, or an unpinned framebuffer size produces different frames, and nothing raises. Two
defences, both cheap:

  - `render_fingerprint()` hashes a fixed set of canonical frames. It is written into the HDF5
    attributes at collection time and re-checked when the dataset is opened and in every worker.
  - Width, height and camera are pinned at `make_loco_env` time, never inherited from the XML.

The other silent failure is an EGL context inherited across `fork()`. It does not raise; it
yields black or stale frames, which surfaces two days later as "the world model will not learn"
and reads as a modelling problem rather than an infrastructure one. Hence the lazy context, the
owner-pid assertion, and `spawn` as the default multiprocessing context.
"""

from __future__ import annotations

import hashlib
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import h5py  # noqa: E402
import hdf5plugin  # noqa: F401,E402  (required before reading Blosc2-compressed datasets)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_worldmodel.data import Dataset as SWMDataset  # noqa: E402

from helpers.locoEnv import make_loco_env  # noqa: E402

#: Columns rendered rather than read. `pixels` is the only synthetic one.
RENDERED = ("pixels",)

class RenderContext:
    """One MuJoCo env per process, built on first use.

    Never constructed eagerly, and never shared across a fork. `_owner_pid` is what turns an
    inherited context from a silent wrong answer into a loud assertion.
    """

    def __init__(self, robot="Walker2d", version="v1", width=224, height=224):
        self.robot, self.version = robot, version
        self.width, self.height = int(width), int(height)
        self._env = None
        self._owner_pid = None
    def close(self) -> None:
        env, self._env = self._env, None
        if env is not None and self._owner_pid == os.getpid():
            try:
                env.close()
            except Exception:  # noqa: BLE001  teardown is best effort
                pass
        self._owner_pid = None

    @property
    def env(self):
        pid = os.getpid()
        if self._env is not None:
            assert self._owner_pid == pid, (
                f"MuJoCo/EGL context was created in pid {self._owner_pid} and is being used in "
                f"pid {pid}. A GL context inherited across fork() renders black or stale frames "
                f"without raising. Use multiprocessing_context='spawn', or build the context "
                f"inside the worker."
            )
            return self._env
        self._env = make_loco_env(
            self.robot, self.version, render=True, width=self.width, height=self.height,
            time_limit=False, six_tuple=True,
        )
        self._env.reset(seed=0)
        self._owner_pid = pid
        return self._env

    def render_state(self, qpos, qvel) -> np.ndarray:
        u = self.env.unwrapped
        u.set_state(np.asarray(qpos, np.float64), np.asarray(qvel, np.float64))
        return self.env.render()

    def render_many(self, qpos_arr, qvel_arr) -> np.ndarray:
        return np.stack([self.render_state(p, v) for p, v in zip(qpos_arr, qvel_arr)])

    def __getstate__(self):
        # Drop the context so the object pickles into DataLoader workers, mirroring how
        # swm's HDF5Dataset drops its open h5 file handle.
        state = self.__dict__.copy()
        state["_env"] = None
        state["_owner_pid"] = None
        return state


def render_fingerprint(ctx: RenderContext, n: int = 64) -> str:
    """Hash frames rendered from a fixed, computed set of states.

    The states are generated deterministically rather than sampled, so the fingerprint depends
    only on the rendering stack, never on the dataset it is being compared against.
    """
    u = ctx.env.unwrapped
    nq, nv = u.model.nq, u.model.nv
    rng = np.random.Generator(np.random.PCG64(12345))
    h = hashlib.blake2b(digest_size=16)
    for _ in range(n):
        qpos = np.zeros(nq)
        qvel = np.zeros(nv)
        qpos[1] = rng.uniform(0.8, 1.4)
        qpos[2] = rng.uniform(-0.9, 0.9)
        qpos[3:] = rng.uniform(-0.6, 0.6, nq - 3)
        qvel[:] = rng.uniform(-2.0, 2.0, nv)
        h.update(ctx.render_state(qpos, qvel).tobytes())
    return h.hexdigest()


class MujocoStateDataset(SWMDataset):
    """Reads the state schema from HDF5 and renders `pixels` on demand.

    Subclasses `stable_worldmodel.data.Dataset` so that `clip_indices`, `__len__`, `__getitem__`
    and the `action.reshape(num_steps, -1)` behaviour are inherited unchanged, which is what
    makes this a drop-in for le-wm's training loop.

    `_load_slice` replicates `HDF5Dataset._load_slice` exactly, including the rule that every
    column EXCEPT `action` is subsampled by `frameskip`. Action stays at full rate and is
    reshaped by the base class into a frameskip-sized block; that is the invariant behind the
    project rule that `PlanConfig.action_block` must equal the frameskip.

    Pixels are rendered only for the subsampled rows: 4 renders per sample at the default
    config rather than 20, a 5x saving that makes lazy rendering cheaper than the GPU it feeds.
    """

    def __init__(
        self,
        path,
        *,
        frameskip: int = 1,
        num_steps: int = 1,
        keys_to_load: list[str] | None = None,
        transform=None,
        robot: str = "Walker2d",
        version: str = "v1",
        width: int = 224,
        height: int = 224,
        clip_stride: int = 1,
        check_fingerprint: bool = True,
    ):
        self.path = str(path)
        with h5py.File(self.path, "r") as f:
            self.attrs = dict(f.attrs)
            lengths = f["ep_len"][:].astype(np.int64)
            offsets = f["ep_offset"][:].astype(np.int64)
            available = [k for k in f.keys() if k not in ("ep_len", "ep_offset")]
            keys = list(keys_to_load) if keys_to_load else list(available) + ["pixels"]
            self._stored_keys = [k for k in keys if k in available]
            self._render_pixels = any(k in RENDERED for k in keys)
            # The whole non-pixel file is a few hundred MB, so caching it removes the random
            # fancy-indexing cost that AGENTS.md flags as a known Push-T wart.
            self._cache = {k: f[k][:] for k in set(self._stored_keys) | {"qpos", "qvel"}}

        super().__init__(lengths, offsets, frameskip=frameskip, num_steps=num_steps,
                         transform=transform)

        if clip_stride > 1:
            # Adjacent clips overlap by span-1 frames, so a stride loses almost no information
            # and cuts epoch time proportionally.
            self.clip_indices = self.clip_indices[::clip_stride]

        self.ctx = RenderContext(robot, version, width, height)
        self._check_fingerprint = check_fingerprint
        self._fingerprint_checked = False

    @property
    def column_names(self) -> list[str]:
        return list(self._stored_keys) + (["pixels"] if self._render_pixels else [])

    def _verify_stack(self) -> None:
        """Compare the live rendering stack against the one that produced the file."""
        if self._fingerprint_checked or not self._check_fingerprint:
            return
        self._fingerprint_checked = True
        expected = self.attrs.get("render_fingerprint")
        if not expected:
            return
        actual = render_fingerprint(self.ctx)
        if actual != expected:
            raise RuntimeError(
                f"render fingerprint mismatch for {self.path}.\n"
                f"  file was collected with: {expected}\n"
                f"  this process renders   : {actual}\n"
                "Frames will not match the ones the stored latents were computed from. Check "
                "MUJOCO_GL, the GPU/driver, mujoco version, and the pinned render size."
            )

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        g_start = int(self.offsets[ep_idx]) + int(start)
        g_end = int(self.offsets[ep_idx]) + int(end)

        steps: dict = {}
        for col in self._stored_keys:
            data = self._cache[col][g_start:g_end]
            if col != "action":
                data = data[:: self.frameskip]
            steps[col] = torch.from_numpy(np.ascontiguousarray(data))

        if self._render_pixels:
            self._verify_stack()
            qpos = self._cache["qpos"][g_start:g_end][:: self.frameskip]
            qvel = self._cache["qvel"][g_start:g_end][:: self.frameskip]
            frames = self.ctx.render_many(qpos, qvel)  # (T, H, W, C) uint8
            steps["pixels"] = torch.from_numpy(frames).permute(0, 3, 1, 2)

        return self.transform(steps) if self.transform else steps

    def get_col_data(self, col: str) -> np.ndarray:
        if col in self._cache:
            return self._cache[col]
        with h5py.File(self.path, "r") as f:
            return f[col][:]

    def get_dim(self, col: str) -> int:
        arr = self.get_col_data(col)
        return int(arr.shape[1]) if arr.ndim > 1 else 1

    def get_row_data(self, row_idx):
        return {c: self._cache[c][row_idx] for c in self._stored_keys}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fingerprint_checked"] = False
        return state


def worker_init_fn(worker_id: int) -> None:
    """Pin an EGL device per worker and force the fingerprint check inside the worker.

    Passed to `DataLoader(worker_init_fn=...)`. The fingerprint check matters more here than in
    the parent: a worker with a broken context renders black frames and reports nothing.
    """
    n_gpu = int(os.environ.get("SAFETYDIAL_N_GPU", "1"))
    if n_gpu > 1:
        os.environ["EGL_DEVICE_ID"] = str(worker_id % n_gpu)
    info = torch.utils.data.get_worker_info()
    ds = getattr(info, "dataset", None)
    if ds is not None and hasattr(ds, "_verify_stack"):
        ds._verify_stack()


def make_loco_dataloader(dataset, batch_size=64, num_workers=4, shuffle=True, **kwargs):
    """DataLoader with the settings that keep lazy rendering safe.

    `spawn` is the default because `fork` silently breaks EGL: a context created in the parent
    (by a figure, a smoke test, a fingerprint check) is inherited dead by every worker.
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn if num_workers else None,
        multiprocessing_context=kwargs.pop("multiprocessing_context", "spawn")
        if num_workers
        else None,
        persistent_workers=bool(num_workers),
        **kwargs,
    )
