"""Evolution start-state sets: S4 roles, deterministic evaluation extras, digest and history cache."""
from __future__ import annotations

from dataclasses import replace
import io
import json
from pathlib import Path
import struct
import sys

import h5py
import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.evoRoots import (  # noqa: E402
    RootSet,
    encode_histories,
    evaluation_roots,
    fitness_roots,
    load_roots_json,
    rootset_digest,
    tuning_roots,
)
from helpers.walkerBank import HORIZON_STEPS, MIN_PREFIX, iter_episodes  # noqa: E402
from helpers.walkerRules import WalkerRoot, roots_from_episode  # noqa: E402
from helpers.walkerValidation import episode_role  # noqa: E402

BANKS = ROOT / "data/hf/safetydial-walker2d-data/banks/walker2d-s4-recovery-20260927-1"
ROOTS_H5 = ROOT / "data/hf/safetydial-walker2d-data/data/walker2d-data-20260926-3/roots.h5"
ARRAYS = ("qpos", "qvel", "history_qpos", "history_qvel", "history_actions", "policy_tape")
LENGTHS = (150, 190, 140, 160, 175, 210, 150, 165, 145, 170)  # episodes 0..9; test role: 1, 5, 9
TEST_PICKS = ((5, 60), (1, 40), (9, 25))  # (episode, step) in file order, deliberately unsorted
D = 8


def write_h5(path, lengths=LENGTHS, seed=0):
    rng = np.random.default_rng(seed)
    total = int(sum(lengths))
    with h5py.File(path, "w") as f:
        f["ep_len"] = np.asarray(lengths, np.int32)
        f["ep_offset"] = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
        f["qpos"] = rng.normal(size=(total, 9))
        f["qvel"] = rng.normal(size=(total, 9))
        f["action"] = rng.uniform(-1, 1, (total, 6)).astype(np.float32)
        f["x_velocity"] = rng.normal(size=total)
        f["healthy"] = np.ones(total, np.uint8)
        f["terminated"] = np.zeros(total, np.uint8)
    return path


def write_roots(path, h5, picks, prefix):
    """Roots built as the S4 banks were, saved as {"roots": [...]}."""
    eps = dict(iter_episodes(h5))
    roots = [roots_from_episode(eps[e], [t], episode=e, prefix=prefix)[0] for e, t in picks]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"roots": [r.to_dict() for r in roots], "meta": {}}))
    return roots


@pytest.fixture
def bank(tmp_path):
    h5 = write_h5(tmp_path / "roots.h5")
    banks = tmp_path / "banks"
    write_roots(banks / "test" / "roots.json", h5, TEST_PICKS, "test")
    write_roots(banks / "dev" / "roots.json", h5, ((0, 30), (4, 50), (8, 35)), "dev")
    write_roots(banks / "acquisition_roots.json", h5, ((2, 30), (3, 21), (6, 40), (7, 44)), "acq")
    return banks, h5


class FakeCtx:
    """Records the states it is asked to render; frames are zeros."""

    def __init__(self):
        self.calls = []

    def render_many(self, qpos_arr, qvel_arr):
        self.calls.append((np.array(qpos_arr), np.array(qvel_arr)))
        return np.zeros((len(qpos_arr), 224, 224, 3), np.uint8)


class FakeEncoder:
    """Stands in for WalkerImaginer.encode (frames -> torch (N, D)). `version` plays the weights
    and calls are numbered, so a z_hist row shows which call (and so which root) produced it."""

    def __init__(self, version):
        self.version, self.calls = version, 0

    def __call__(self, frames):
        self.calls += 1
        return expected_z(self.version, self.calls, len(frames))


def expected_z(version, call, n=3):
    z = torch.arange(n * D, dtype=torch.float32).reshape(n, D)
    return z + 1e5 * version + 100.0 * call


def same_root(a, b):
    return a.root_id == b.root_id and all(np.array_equal(getattr(a, k), getattr(b, k)) for k in ARRAYS)


def cached_encode(rootset, path, identity, version):
    """encode_histories through the cache at `path`: (z_hist, encode calls made, 0 on a cache hit)."""
    ctx, enc = FakeCtx(), FakeEncoder(version)
    z = encode_histories(rootset, enc, ctx, cache_path=path, identity=identity).z_hist
    assert enc.calls == len(ctx.calls)
    return z, enc.calls


def test_load_roots_json_reads_bank_roots_exactly(bank):
    banks, h5 = bank
    eps = dict(iter_episodes(h5))
    roots = load_roots_json(banks / "test" / "roots.json")
    assert [r.root_id for r in roots] == ["test-e5-t60", "test-e1-t40", "test-e9-t25"]
    for r in roots:
        ref = roots_from_episode(eps[r.episode], [r.step], episode=r.episode, prefix="test")[0]
        assert same_root(r, ref) and all(getattr(r, k).dtype == np.float64 for k in ARRAYS)


def test_fitness_and_tuning_sets_keep_their_roles(bank):
    banks, _ = bank
    fit, tune = fitness_roots(banks), tuning_roots(banks)
    assert (fit.name, len(fit), tune.name, len(tune)) == ("fitness", 4, "tuning", 3)
    assert fit.source_episode.tolist() == [2, 3, 6, 7] and tune.source_episode.tolist() == [0, 4, 8]
    for rs, role in ((fit, "acquisition"), (tune, "development")):
        assert {episode_role(int(e)) for e in rs.source_episode} == {role}
        assert rs.hist_blocks.shape == (len(rs), 2, 10, 6) and rs.hist_blocks.dtype == np.float64
        assert np.array_equal(rs.hist_blocks, np.stack([r.history_actions for r in rs.roots]))
        assert rs.z_hist is None
    cases = ((fitness_roots, "acquisition_roots.json", "dev", "development", "acquisition"),
             (tuning_roots, "dev/roots.json", "test", "test", "development"))
    for load, rel, intruder, found, role in cases:
        blob = json.loads((banks / rel).read_text())  # one root of another role, mid-file
        blob["roots"].insert(1, json.loads((banks / intruder / "roots.json").read_text())["roots"][0])
        (banks / rel).write_text(json.dumps(blob))
        with pytest.raises(ValueError, match=rf"\['{found}'\], not the {role} role"):
            load(banks)


def test_subset_keeps_rows_aligned(bank):
    banks, _ = bank
    fit = fitness_roots(banks)
    enc = encode_histories(fit, FakeEncoder(1), FakeCtx())
    assert len({row.tobytes() for row in enc.z_hist}) == len(fit)  # every row distinguishable
    sub = enc.subset(np.array([3, 0]))
    assert sub.root_ids == [enc.root_ids[3], enc.root_ids[0]] and sub.name == "fitness"
    assert np.array_equal(sub.hist_blocks, enc.hist_blocks[[3, 0]])
    assert sub.source_episode.tolist() == [7, 2]
    assert np.array_equal(sub.z_hist, enc.z_hist[[3, 0]])
    assert enc.subset(np.array([True, False, True, False])).root_ids == enc.root_ids[::2]
    assert len(enc.subset([])) == 0 and fit.subset([1]).z_hist is None
    with pytest.raises(ValueError):
        RootSet("bad", fit.roots, fit.hist_blocks[:2], fit.source_episode)
    with pytest.raises(ValueError, match="duplicate"):
        RootSet.from_roots("bad", fit.roots + fit.roots[:1])


def test_evaluation_roots_layout_ranges_and_exact_construction(bank):
    banks, h5 = bank
    eps = dict(iter_episodes(h5))
    base = load_roots_json(banks / "test" / "roots.json")
    rs = evaluation_roots(banks, h5, seed=7)
    assert rs.name == "evaluation" and len(rs) == 4 * len(base)
    assert rs.source_episode.tolist() == [r.episode for r in base for _ in range(4)]
    assert {episode_role(int(e)) for e in rs.source_episode} == {"test"}
    assert np.array_equal(rs.hist_blocks, np.stack([r.history_actions for r in rs.roots]))
    for i, orig in enumerate(base):
        group = rs.roots[4 * i : 4 * i + 4]
        assert same_root(group[0], orig) and group[0].meta == orig.meta
        steps = [r.step for r in group[1:]]
        n = len(eps[orig.episode]["qpos"])
        assert steps == sorted(steps) and len(set(steps)) == 3 and orig.step not in steps
        assert all(MIN_PREFIX <= t < n - HORIZON_STEPS for t in steps)
        want = roots_from_episode(eps[orig.episode], steps, episode=orig.episode, prefix="eval")
        for got, ref in zip(group[1:], want):
            assert same_root(got, ref) and got.meta == ref.meta
            assert got.root_id == f"eval-e{orig.episode}-t{got.step}"


def test_evaluation_extras_are_drawn_once_from_the_seed(bank):
    banks, h5 = bank
    a, b, c = (evaluation_roots(banks, h5, seed=s) for s in (3, 3, 4))
    assert a.root_ids == b.root_ids and rootset_digest(a) == rootset_digest(b)
    assert a.root_ids != c.root_ids and rootset_digest(a) != rootset_digest(c)
    eps = dict(iter_episodes(h5))
    rng = np.random.default_rng(3)  # one generator, test roots in file order
    want = []
    for r in load_roots_json(banks / "test" / "roots.json"):
        pool = np.arange(MIN_PREFIX, len(eps[r.episode]["qpos"]) - HORIZON_STEPS)
        drawn = np.sort(rng.choice(pool[pool != r.step], 3, replace=False))
        want += [r.root_id] + [f"eval-e{r.episode}-t{t}" for t in drawn]
    assert a.root_ids == want
    assert evaluation_roots(banks, h5, seed=np.int64(3)).root_ids == want  # any int type, by value
    # a stateful source would give another "fixed" set on every call, so only an int seed is taken
    for bad in (None, np.random.default_rng(3), np.random.PCG64(3), -1, 3.5):
        with pytest.raises(ValueError, match="fixed int seed"):
            evaluation_roots(banks, h5, seed=bad)


def test_extras_stay_distinct_and_in_range_for_any_seed_and_count(bank):
    banks, h5 = bank
    for extra in (0, 1, 5):
        for seed in range(10):
            rs = evaluation_roots(banks, h5, seed=seed, extra_per_episode=extra)
            assert len(rs) == len(TEST_PICKS) * (1 + extra) and len(set(rs.root_ids)) == len(rs)
            for g, (e, t0) in enumerate(TEST_PICKS):
                steps = [r.step for r in rs.roots[g * (1 + extra) : (g + 1) * (1 + extra)]]
                assert steps[0] == t0 and t0 not in steps[1:] and steps[1:] == sorted(set(steps[1:]))
                assert all(MIN_PREFIX <= t < LENGTHS[e] - HORIZON_STEPS for t in steps[1:])


def test_evaluation_refuses_a_non_test_episode(bank):
    banks, h5 = bank
    blob = json.loads((banks / "test" / "roots.json").read_text())
    blob["roots"].append(json.loads((banks / "dev" / "roots.json").read_text())["roots"][0])
    (banks / "test" / "roots.json").write_text(json.dumps(blob))
    with pytest.raises(ValueError, match=r"\['development'\], not the test role"):
        evaluation_roots(banks, h5, seed=0)


def test_evaluation_refuses_a_different_source_file(bank, tmp_path):
    banks, _ = bank
    with pytest.raises(ValueError, match="does not match"):
        evaluation_roots(banks, write_h5(tmp_path / "other.h5", seed=1), seed=0)


def test_evaluation_refuses_a_source_file_one_ulp_off(bank, tmp_path):
    """A roots.h5 equal to the bank's except one ulp in one value the last test root was built from:
    a history action (rows t-20..t-1), a policy-tape action (rows t..t+99), a state at a history
    frame (t-20, t-10) or the root's own state (t)."""
    banks, h5 = bank
    e, t = TEST_PICKS[-1]
    want = evaluation_roots(banks, h5, seed=0).root_ids
    copy = write_h5(tmp_path / "copy.h5")  # the bank's file written again: accepted, the same set
    assert evaluation_roots(banks, copy, seed=0).root_ids == want
    for key, offset in (("action", -20), ("action", -1), ("action", 0), ("action", 99), ("qpos", -10),
                        ("qvel", -20), ("qvel", 0)):
        other = write_h5(tmp_path / f"{key}{offset}.h5")
        with h5py.File(other, "r+") as f:
            row = int(f["ep_offset"][e]) + t + offset
            v = f[key][row]
            v[-1] = np.nextafter(v[-1], np.inf)  # in the dataset's own dtype
            f[key][row] = v
        with pytest.raises(ValueError, match=f"test-e{e}-t{t} does not match"):
            evaluation_roots(banks, other, seed=0)


def test_evaluation_needs_one_root_per_episode_and_enough_steps(tmp_path):
    h5 = write_h5(tmp_path / "roots.h5", lengths=(150, 123, 150, 150, 150, 170))
    write_roots(tmp_path / "short" / "test" / "roots.json", h5, ((1, 21),), "test")
    with pytest.raises(ValueError, match="eligible"):  # [20, 23) without step 21 leaves two
        evaluation_roots(tmp_path / "short", h5, seed=0)
    two = evaluation_roots(tmp_path / "short", h5, seed=0, extra_per_episode=2)
    assert [r.step for r in two.roots] == [21, 20, 22]
    write_roots(tmp_path / "dup" / "test" / "roots.json", h5, ((5, 30), (5, 40)), "test")
    with pytest.raises(ValueError, match="one root per source episode"):
        evaluation_roots(tmp_path / "dup", h5, seed=0)
    with pytest.raises(ValueError, match="seed"):
        evaluation_roots(tmp_path / "short", h5, seed=None)
    with pytest.raises(ValueError, match="nonnegative"):
        evaluation_roots(tmp_path / "short", h5, seed=0, extra_per_episode=-1)


def toy_roots(n=3):
    out = []
    for i in range(n):
        base = np.arange(9.0) + i  # exact binary fractions: the digest cannot depend on the platform
        out.append(WalkerRoot(f"toy-e{i}-t{20 + i}", i, 20 + i, base / 8, -base / 4,
                              np.stack([base / 2] * 3), np.stack([base] * 3),
                              np.arange(120.0).reshape(2, 10, 6) / 64 + i,
                              np.arange(600.0).reshape(100, 6) / 32 - i, {"note": i}))
    return out


def test_rootset_digest_is_stable_and_content_sensitive():
    roots = toy_roots()
    rs = RootSet.from_roots("toy", roots)
    digest = rootset_digest(rs)
    assert digest == "4b815c63abd63c791931d12be3355b1a9422c43c8d91bdcca1353bd20b45a8e9"  # frozen scheme
    reloaded = [WalkerRoot.from_dict(json.loads(json.dumps(r.to_dict()))) for r in roots]
    assert rootset_digest(RootSet.from_roots("renamed", reloaded)) == digest
    assert rootset_digest(replace(rs, z_hist=np.ones((3, 3, D), np.float32))) == digest
    assert rootset_digest(RootSet.from_roots("toy", [replace(r, meta={}) for r in roots])) == digest
    assert rootset_digest(rs.subset([0, 2, 1])) != digest

    def middle_changed(**change):
        changed = [roots[0], replace(roots[1], **change), roots[2]]
        return rootset_digest(RootSet.from_roots("toy", changed))

    for k in ARRAYS:  # one ulp anywhere in any array of a root changes it, the policy tape included
        a = getattr(roots[1], k).copy()
        a.flat[-1] = np.nextafter(a.flat[-1], np.inf)
        assert middle_changed(**{k: a}) != digest
    for change in ({"root_id": "toy-x"}, {"step": 99}, {"episode": 7}):
        assert middle_changed(**change) != digest


def test_encode_histories_renders_each_root_history_in_order(bank):
    banks, _ = bank
    rs = tuning_roots(banks)
    ctx, enc = FakeCtx(), FakeEncoder(1)
    out = encode_histories(rs, enc, ctx)
    assert rs.z_hist is None and out.z_hist.shape == (len(rs), 3, D) and out.z_hist.dtype == np.float32
    assert enc.calls == len(ctx.calls) == len(rs)
    for (qp, qv), r in zip(ctx.calls, rs.roots):
        assert np.array_equal(qp, r.history_qpos) and np.array_equal(qv, r.history_qvel)
    for i in range(len(rs)):  # one encode call per root, in root order: row i is call i + 1
        assert np.array_equal(out.z_hist[i], expected_z(1, i + 1).numpy())
    assert out.root_ids == rs.root_ids and np.array_equal(out.hist_blocks, rs.hist_blocks)


def test_history_cache_round_trip_and_rejection(bank, tmp_path):
    banks, h5 = bank
    rs = evaluation_roots(banks, h5, seed=1)
    path = tmp_path / "cache" / "evaluation_hist.npz"
    w1 = {"weights_sha256": "w1", "render_fingerprint": "r1"}
    w2 = {**w1, "weights_sha256": "w2"}

    def run(rootset, version, identity):
        return cached_encode(rootset, path, identity, version)

    z1, calls = run(rs, 1, w1)
    assert calls == len(rs) and [p.name for p in path.parent.iterdir()] == [path.name]  # no tmp left
    z, calls = run(rs, 2, dict(w1))  # same ids, digest and identity: read back, nothing rendered
    assert calls == 0 and np.array_equal(z, z1) and z.dtype == np.float32
    z2, calls = run(rs, 2, w2)  # new weights: rejected and rebuilt
    assert calls == len(rs) and not np.array_equal(z2, z1)
    z, calls = run(rs, 3, w2)  # the rebuilt cache now serves the new identity
    assert calls == 0 and np.array_equal(z, z2)
    moved = replace(rs.roots[1], history_qpos=rs.roots[1].history_qpos + 1e-9)
    same_ids = RootSet.from_roots(rs.name, [rs.roots[0], moved, *rs.roots[2:]])
    assert same_ids.root_ids == rs.root_ids
    assert run(same_ids, 3, w2)[1] == len(rs)  # same ids, different states: rejected
    sub = rs.subset(range(4))
    assert run(sub, 3, w2)[1] == len(sub)  # different ids: rejected
    assert run(sub, 4, w2)[1] == 0
    path.write_bytes(b"not an npz")
    z, calls = run(sub, 5, w2)  # unreadable: rebuilt, and readable again afterwards
    assert calls == len(sub)
    z_again, calls = run(sub, 6, w2)
    assert calls == 0 and np.array_equal(z_again, z)
    with pytest.raises(ValueError, match="identity"):
        encode_histories(rs, FakeEncoder(1), FakeCtx(), cache_path=path)
    before = sorted(tmp_path.rglob("*"))
    assert encode_histories(rs, FakeEncoder(1), FakeCtx()).z_hist.shape == (len(rs), 3, D)
    assert sorted(tmp_path.rglob("*")) == before  # no cache_path, nothing written


def test_history_cache_rebuilds_a_damaged_or_foreign_file(bank, tmp_path):
    """Anything at the cache path short of a cache for exactly this key is rebuilt, never read and
    never a crash: zip header damage that zipfile reports as NotImplementedError or RuntimeError,
    a lone .npy array, and npz files with this key but another version, a field missing, or
    latents of the wrong dtype or shape."""
    banks, _ = bank
    rs, path, ident = tuning_roots(banks), tmp_path / "hist.npz", {"weights_sha256": "w1"}
    z, calls = cached_encode(rs, path, ident, 1)
    good = path.read_bytes()
    with np.load(path) as f:
        fields = dict(f)
    cd = struct.unpack_from("<I", good, good.rindex(b"PK\x05\x06") + 16)[0]  # central directory
    assert calls == len(rs) and good[cd : cd + 4] == b"PK\x01\x02"

    def patched(offset, fmt, value):  # one field of the first member's central directory entry
        blob = bytearray(good)
        struct.pack_into(fmt, blob, cd + offset, value)
        return bytes(blob)

    def npz(**change):  # a well-formed file with fields changed (None drops one)
        buf = io.BytesIO()
        np.savez(buf, **{k: v for k, v in {**fields, **change}.items() if v is not None})
        return buf.getvalue()

    lone = io.BytesIO()
    np.save(lone, z)
    damaged = {"needs zip version 17.3": patched(6, "<B", 173), "encrypted": patched(8, "<H", 1),
               "patched data": patched(8, "<H", 1 << 5), "strong encryption": patched(8, "<H", 1 << 6),
               "unknown compression": patched(10, "<H", 99), "lone npy": lone.getvalue(),
               "old version": npz(version=np.asarray("evoRoots.history_cache.v0")),
               "no identity": npz(identity=None), "float64 latents": npz(z_hist=z.astype(np.float64)),
               "two history frames": npz(z_hist=z[:, :2]), "a root short": npz(z_hist=z[:-1])}
    for version, (what, blob) in enumerate(damaged.items(), start=2):
        path.write_bytes(blob)
        z_new, calls = cached_encode(rs, path, ident, version)
        assert calls == len(rs), what  # rebuilt with this call's encoder
        assert np.array_equal(z_new, encode_histories(rs, FakeEncoder(version), FakeCtx()).z_hist), what
        z_again, calls = cached_encode(rs, path, ident, 99)
        assert calls == 0 and np.array_equal(z_again, z_new), what  # and a valid cache again


def test_real_s4_banks_give_96_24_256():
    if not (BANKS.is_dir() and ROOTS_H5.is_file()):
        pytest.skip("S4 banks or roots.h5 not downloaded")
    cfg = yaml.safe_load((ROOT / "configs/evo/roots.yaml").read_text())["roots"]
    fit, tune = fitness_roots(BANKS), tuning_roots(BANKS)
    ev = evaluation_roots(BANKS, ROOTS_H5, seed=cfg["evaluation_seed"],
                          extra_per_episode=cfg["extra_per_episode"])
    assert (len(fit), len(tune), len(ev)) == (96, 24, 256)
    # the configured sets, pinned: new banks, roots.h5, seed or numpy draws all show up here
    assert [rootset_digest(rs) for rs in (fit, tune, ev)] == [
        "eee9cc0e4ec65cb34a9922204ee0dde34d0b199dfe9e99cd3a0f971652244c09",
        "d3171be52e65adf90ab0cc705a32d1beafb637fa762be0aac94a69b68996d515",
        "107e848387c8d4d37bbb854cbdfbd01d06f116342d5baf5cf9816cc23e80e303"]
    assert {episode_role(int(e)) for e in fit.source_episode} == {"acquisition"}
    assert {episode_role(int(e)) for e in tune.source_episode} == {"development"}
    assert {episode_role(int(e)) for e in ev.source_episode} == {"test"}
    test_ids = [r.root_id for r in load_roots_json(BANKS / "test" / "roots.json")]
    assert ev.root_ids[::4] == test_ids and len(set(ev.root_ids)) == 256
    assert sorted(np.unique(ev.source_episode, return_counts=True)[1].tolist()) == [4] * 64
