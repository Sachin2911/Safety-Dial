"""Start-state sets for evolution inside the Walker LeWM (docs/evoPlan/README.md, "Start-state
roles"). The roles reuse the S4 banks, which split whole source episodes
(`walkerValidation.episode_role`), so no source episode feeds two roles:

    fitness = fitness_roots(banks_dir)            # 96 acquisition roots, scored during evolution
    tuning = tuning_roots(banks_dir)              # 24 development roots: noise, steps, penalty
    evaluation = evaluation_roots(banks_dir, roots_h5, seed=s)   # 64 test roots + 3 each = 256
    evaluation = encode_histories(evaluation, imaginer.encode, ctx, cache_path=p, identity=ident)

Evaluation extras come from one `np.random.default_rng(seed)` (an int seed, never a stateful
Generator, so the set is a function of the seed's value), drawn in test-file order,
uniformly without replacement over the representative range [MIN_PREFIX, n - HORIZON_STEPS) of
the root's source episode (the range S4 drew its test roots from), never the existing step. Each
test root must equal its episode in `roots_h5` bit for bit, so a wrong data file cannot quietly
supply extras from other episodes. `rootset_digest` pins a set's identity (every array of every
root, recorded policy tapes included) for manifests and caches.

History latents are encoded one root at a time, `encode_fn(ctx.render_many(history_qpos,
history_qvel))`, exactly as the S4 analysis and the Stage 0 check did: a root's latents then never
depend on which other roots share a batch, so subsets and cache hits equal a fresh encoding
bitwise. The npz cache is keyed by the ordered root ids, the set digest and the caller's identity
(model weights and renderer); any mismatch or unreadable file is rebuilt, never reused.
"""

from __future__ import annotations

import hashlib
import json
import operator
import os
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from helpers.walkerBank import HORIZON_STEPS, MIN_PREFIX
from helpers.walkerRules import FRAMESKIP, HISTORY, WalkerRoot, roots_from_episode
from helpers.walkerValidation import episode_role

ACTION_DIM = 6
DIGEST_VERSION = b"evoRoots.rootset_digest.v2"  # v2: the policy tape joined the hashed arrays
CACHE_VERSION = "evoRoots.history_cache.v1"
# every array of a root: the source check compares them all and the digest hashes them all (Gate 0
# replays the recorded policy tapes, so a stamped digest must pin them too)
_ROOT_ARRAYS = ("qpos", "qvel", "history_qpos", "history_qvel", "history_actions", "policy_tape")
_EPISODE_KEYS = ("qpos", "qvel", "action", "x_velocity")  # what roots_from_episode reads
_CACHE_FIELDS = ("version", "root_ids", "digest", "identity", "z_hist")


@dataclass(eq=False)
class RootSet:
    name: str
    roots: list[WalkerRoot]
    hist_blocks: np.ndarray  # (N, 2, 10, 6) float64 raw history actions
    source_episode: np.ndarray  # (N,) int
    z_hist: np.ndarray | None = None  # (N, 3, D) float32 once encoded

    def __post_init__(self):
        self.hist_blocks = np.asarray(self.hist_blocks)
        self.source_episode = np.asarray(self.source_episode)
        self.z_hist = None if self.z_hist is None else np.asarray(self.z_hist)
        n = len(self.roots)
        rows = [len(self.hist_blocks), len(self.source_episode)]
        if any(r != n for r in rows + ([] if self.z_hist is None else [len(self.z_hist)])):
            raise ValueError(f"RootSet {self.name}: arrays do not have one row per root ({n})")

    @classmethod
    def from_roots(cls, name: str, roots, z_hist=None) -> "RootSet":
        """Stack the history actions and source episodes of `roots`; root ids must be unique."""
        roots = list(roots)
        ids = [r.root_id for r in roots]
        if len(set(ids)) != len(ids):
            raise ValueError(f"RootSet {name}: duplicate root ids")
        shape = (HISTORY - 1, FRAMESKIP, ACTION_DIM)
        blocks = np.empty((0, *shape))
        if roots:
            blocks = np.stack([np.asarray(r.history_actions, dtype=np.float64) for r in roots])
        if blocks.shape[1:] != shape:
            raise ValueError(f"RootSet {name}: history actions must be {shape}, got {blocks.shape[1:]}")
        return cls(name, roots, blocks, np.array([int(r.episode) for r in roots], dtype=np.int64), z_hist)

    def __len__(self) -> int:
        return len(self.roots)

    @property
    def root_ids(self) -> list[str]:
        return [r.root_id for r in self.roots]

    def subset(self, idx) -> "RootSet":
        """Rows `idx` in the given order (integer positions, or a boolean mask over the set)."""
        idx = np.asarray(idx)
        if idx.dtype == np.bool_:
            if idx.shape != (len(self),):
                raise ValueError("a boolean mask must cover every root")
            idx = np.flatnonzero(idx)
        idx = idx.astype(np.int64) if idx.size == 0 else idx
        if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer):
            raise ValueError("subset takes a 1-d integer index array or a boolean mask")
        z = None if self.z_hist is None else self.z_hist[idx]
        roots = [self.roots[i] for i in idx]
        return RootSet(self.name, roots, self.hist_blocks[idx], self.source_episode[idx], z)


# --------------------------------------------------------------------------------------
# the three roles
# --------------------------------------------------------------------------------------
def load_roots_json(path) -> list[WalkerRoot]:
    """Roots of a `{"roots": [...]}` file (an S4 bank's roots.json, acquisition_roots.json)."""
    return [WalkerRoot.from_dict(d) for d in json.loads(Path(path).read_text())["roots"]]


def _require_role(roots, role: str, source) -> None:
    bad = sorted({int(r.episode) for r in roots if episode_role(int(r.episode)) != role})
    if bad:
        found = sorted({episode_role(e) for e in bad})
        raise ValueError(f"{source}: source episodes {bad[:8]} have roles {found}, not the {role} role")


def _role_set(name: str, path: Path, role: str) -> RootSet:
    roots = load_roots_json(path)
    _require_role(roots, role, path)
    return RootSet.from_roots(name, roots)


def fitness_roots(banks_dir) -> RootSet:
    """S4 acquisition roots (96 reserved): scoring candidates during evolution."""
    return _role_set("fitness", Path(banks_dir) / "acquisition_roots.json", "acquisition")


def tuning_roots(banks_dir) -> RootSet:
    """S4 development roots (24): noise calibration, step sizes, penalty scale."""
    return _role_set("tuning", Path(banks_dir) / "dev" / "roots.json", "development")


def _require_source(root: WalkerRoot, ep: dict, source) -> None:
    """A bank root must be its source episode's rows bit for bit (the same roots.h5)."""
    ref = roots_from_episode(ep, [root.step], episode=int(root.episode))
    if not ref or not all(np.array_equal(getattr(ref[0], k), getattr(root, k)) for k in _ROOT_ARRAYS):
        raise ValueError(f"{root.root_id} does not match episode {root.episode} of {source}")


def _fixed_seed(seed) -> int:
    """An int seed >= 0, so the evaluation set is a function of its value: None, a Generator or a
    BitGenerator would draw from state that differs between calls."""
    try:
        value = operator.index(seed)
    except TypeError:
        value = -1
    if value < 0:
        raise ValueError(f"evaluation extras need a fixed int seed >= 0, got {seed!r}")
    return value


def evaluation_roots(banks_dir, roots_h5, *, seed, extra_per_episode=3) -> RootSet:
    """S4 test roots (one per test-role source episode), each followed by `extra_per_episode`
    more roots from its own episode, sorted by step. Every number the study reports uses these."""
    import h5py
    import hdf5plugin  # noqa: F401  (roots.h5 is Blosc2-compressed)

    rng, extra = np.random.default_rng(_fixed_seed(seed)), operator.index(extra_per_episode)
    if extra < 0:
        raise ValueError(f"extra_per_episode must be nonnegative, got {extra}")
    path = Path(banks_dir) / "test" / "roots.json"
    base = load_roots_json(path)
    _require_role(base, "test", path)
    episodes = [int(r.episode) for r in base]
    if len(set(episodes)) != len(episodes):
        raise ValueError(f"{path}: expected one root per source episode")
    out = []
    with h5py.File(roots_h5, "r") as f:
        ln, off = f["ep_len"][:], f["ep_offset"][:]
        for root, e in zip(base, episodes):
            if not 0 <= e < len(ln):
                raise ValueError(f"{root.root_id}: episode {e} is not in {roots_h5}")
            a, b = int(off[e]), int(off[e] + ln[e])
            ep = {k: f[k][a:b] for k in _EPISODE_KEYS}
            _require_source(root, ep, roots_h5)
            pool = np.arange(MIN_PREFIX, b - a - HORIZON_STEPS)
            pool = pool[pool != root.step]
            if len(pool) < extra:
                raise ValueError(f"episode {e} has {len(pool)} eligible steps besides {root.root_id}, "
                                 f"needs {extra}")
            steps = np.sort(rng.choice(pool, extra, replace=False))
            extras = roots_from_episode(ep, steps, episode=e, prefix="eval")
            if len(extras) != extra:
                raise ValueError(f"episode {e}: roots_from_episode dropped an eligible step")
            out += [root, *extras]
    return RootSet.from_roots("evaluation", out)


# --------------------------------------------------------------------------------------
# identity and encoded histories
# --------------------------------------------------------------------------------------
def rootset_digest(rootset) -> str:
    """sha256 over the ordered root ids (with episode and step) and every array of each root (qpos,
    qvel, the history arrays and the policy tape) as little-endian float64 with their shapes. Name,
    meta and z_hist are not identity."""
    h = hashlib.sha256(DIGEST_VERSION)
    h.update(len(rootset.roots).to_bytes(8, "little"))
    for r in rootset.roots:
        head = json.dumps([r.root_id, int(r.episode), int(r.step)]).encode()
        h.update(len(head).to_bytes(8, "little") + head)
        for k in _ROOT_ARRAYS:
            a = np.ascontiguousarray(getattr(r, k), dtype="<f8")
            h.update(np.asarray([a.ndim, *a.shape], dtype="<i8").tobytes() + a.tobytes())
    return h.hexdigest()


def _encode_root(root: WalkerRoot, encode_fn, ctx) -> np.ndarray:
    z = encode_fn(ctx.render_many(root.history_qpos, root.history_qvel))
    if hasattr(z, "detach"):  # a torch tensor on any device
        z = z.detach().float().cpu().numpy()
    z = np.asarray(z, dtype=np.float32)
    if z.ndim != 2 or len(z) != HISTORY:
        raise ValueError(f"{root.root_id}: encode_fn must return ({HISTORY}, D) latents, got {z.shape}")
    return z


def _read_cache(path: Path, key: dict) -> np.ndarray | None:
    """The cached z_hist if `path` is an npz written for exactly `key`, else None (rebuild)."""
    if not path.is_file():
        return None
    try:  # zipfile, zlib and numpy report damage with many exception types; each one means rebuild
        f = np.load(path, allow_pickle=False)
        if not isinstance(f, np.lib.npyio.NpzFile):  # a lone .npy array, say
            return None
        with f:
            stored = {k: f[k] for k in _CACHE_FIELDS}
    except Exception:
        return None
    z = stored["z_hist"]
    same = (str(stored["version"]) == CACHE_VERSION and stored["root_ids"].tolist() == key["root_ids"]
            and str(stored["digest"]) == key["digest"] and str(stored["identity"]) == key["identity"])
    if not same or z.dtype != np.float32 or z.ndim != 3 or z.shape[:2] != (len(key["root_ids"]), HISTORY):
        return None
    return z


def _write_cache(path: Path, key: dict, z: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as fh:  # a file handle, so np.savez cannot append its own suffix
            np.savez(fh, version=np.asarray(CACHE_VERSION), digest=np.asarray(key["digest"]),
                     identity=np.asarray(key["identity"]), z_hist=z,
                     root_ids=np.asarray(key["root_ids"], dtype=np.str_))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def encode_histories(rootset, encode_fn, ctx, *, cache_path=None, identity=None) -> RootSet:
    """A copy of `rootset` with z_hist (N, 3, D) float32: each root's three history states rendered
    with `ctx.render_many` and encoded by `encode_fn` (frames -> torch (3, D)), one root per call.

    With `cache_path`, an npz written for the same ordered root ids, set digest and `identity`
    (JSON-serialisable, e.g. model weights sha256 and render fingerprint) is read back instead;
    any other file there (another key, damaged, not an npz) is rebuilt. A cache without an
    identity is refused.
    """
    if not len(rootset):
        raise ValueError(f"RootSet {rootset.name} has no roots to encode")
    if cache_path is not None and identity is None:
        raise ValueError("a cached encoding needs an identity (model weights and renderer)")
    key = {"root_ids": rootset.root_ids, "digest": rootset_digest(rootset),
           "identity": json.dumps(identity, sort_keys=True)}
    path = None if cache_path is None else Path(cache_path)
    z = None if path is None else _read_cache(path, key)
    if z is None:
        z = np.stack([_encode_root(r, encode_fn, ctx) for r in rootset.roots])
        if path is not None:
            _write_cache(path, key, z)
    return replace(rootset, z_hist=z)
