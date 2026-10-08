from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import pushtReplay as legacy
from helpers.acquisitionSafety import MeteredEnv
from helpers.pushtContactReplay import (CONTACT_COUNTER, ContactDenseLog, execute_branch,
                                      make_env, reset_root, run_actions)


def root_fixture(start=None):
    if start is None:
        start = [250, 170, 250, 200, 0, 0, 0]
    return legacy.Root(seed=31, start_state=np.asarray(start, float),
                       goal_state=np.asarray([420, 420, 370, 350, 1.2, 0, 0]),
                       prefix=np.zeros((10, 2)), root_id="contact-fixture")


def test_passive_contact_observer_preserves_full_replay_bitwise_and_counts_substeps():
    old, new = legacy.make_env(), make_env()
    root = root_fixture()
    tape = np.tile([0, .08], (25, 1))
    try:
        _, before = legacy.execute_branch(old, root, tape)
        ctx, after = execute_branch(new, root, tape)
        _, again = execute_branch(new, root, tape)
        for field in ("states", "block_vel", "block_ang_vel", "n_contacts", "actions",
                      "observed", "observation_valid", "terminated", "truncated", "frames"):
            assert np.array_equal(getattr(before, field), getattr(after, field)), field
            assert np.array_equal(getattr(after, field), getattr(again, field)), field
        assert isinstance(ctx.prefix_log, ContactDenseLog)
        assert after.pusher_block_contacts.sum() > 0
        assert after.block_wall_contacts.sum() == 0
        assert after.pusher_block_contacts.sum() > after.n_contacts.sum()
        assert np.array_equal(after.pusher_block_contacts, again.pusher_block_contacts)
        assert after.contact_counter == CONTACT_COUNTER
    finally:
        old.close()
        new.close()


def test_wall_contact_does_not_become_pusher_block_contact():
    env = make_env()
    root = root_fixture([100, 350, 250, 390, 0, 0, 0])
    try:
        ctx = reset_root(env, root)
        log = run_actions(env, np.zeros((10, 2)))
        assert ctx.prefix_log.block_wall_contacts.sum() + log.block_wall_contacts.sum() > 0
        assert log.n_contacts.sum() > 0
        assert ctx.prefix_log.pusher_block_contacts.sum() == log.pusher_block_contacts.sum() == 0
    finally:
        env.close()


def test_legacy_environment_retains_unknown_contact_types_and_metering_is_preserved():
    old = legacy.make_env()
    ledger = legacy.StepLedger()
    new = MeteredEnv(make_env(), ledger, "observed")
    try:
        ctx = reset_root(old, root_fixture())
        assert getattr(ctx.prefix_log, "pusher_block_contacts", None) is None
        ctx, log = execute_branch(new, root_fixture(), np.zeros((25, 2)))
        assert ledger.total == len(root_fixture().prefix) + log.executed_steps
        assert isinstance(log, ContactDenseLog)
    finally:
        old.close()
        new.close()


def test_live_typed_and_legacy_contacts_roundtrip_without_relabelling_unknown(tmp_path):
    from helpers.branchBank import Bank, BankWriter, Branch, Proposal, execute_proposals
    from helpers.decomposition import contact_metadata

    root = root_fixture()
    tape = np.tile([0, .08], (25, 1)).reshape(5, 5, 2)
    env, old = make_env(), legacy.make_env()
    ledger = legacy.StepLedger()
    try:
        branches = execute_proposals(env, root, [Proposal(tape, "fixture", {})], ledger=ledger)
        _, unknown = legacy.execute_branch(old, root, tape.reshape(-1, 2))
    finally:
        env.close()
        old.close()
    writer = BankWriter(tmp_path / "bank")
    writer.add_root(root)
    writer.add_branches([*branches, Branch(root.root_id, tape, "legacy_fixture", {}, unknown)])
    writer.finish(ledger)
    bank = Bank(tmp_path / "bank")
    try:
        typed, untyped = bank.branch(0), bank.branch(1)
        assert typed["contact_kind"] == "pusher_block"
        assert typed["contact_counter"] == CONTACT_COUNTER
        assert np.array_equal(typed["pusher_block_contacts"], branches[0].log.pusher_block_contacts)
        assert np.array_equal(typed["block_wall_contacts"], branches[0].log.block_wall_contacts)
        assert np.array_equal(typed["n_contacts"], unknown.n_contacts)
        assert contact_metadata(typed)["contact"] is True
        assert untyped["contact_kind"] == "any_collision"
        assert untyped["pusher_block_contacts"] is None
        assert contact_metadata(untyped)["contact"] is None
        assert contact_metadata(untyped)["contact_mechanism_identifiable"] is False
        assert np.array_equal(untyped["n_contacts"], unknown.n_contacts)
    finally:
        bank.h5.close()


def test_typed_contacts_survive_nominal_blocks_and_include_idle_root_prefix():
    from types import SimpleNamespace

    from helpers.branchBank import build_root

    class FixedPlanner:
        def __init__(self, action):
            self.blocks = np.broadcast_to(np.asarray(action), (5, 5, 2)).copy()

        def plan(self, *args, **kwargs):
            return SimpleNamespace(blocks=self.blocks.copy(), cost=0., solve_time=0., frac_feasible=1.)

    for k in (0, 2):
        env = make_env()
        initial = root_fixture([250, 190, 250, 200, 0, 0, 0])
        pair = {"start": initial.start_state, "goal": initial.goal_state,
                "episode": 7, "t0": 0, "goal_offset": 25}
        ledger = legacy.StepLedger()
        try:
            root, context, nominal = build_root(env, FixedPlanner([0, .08]), pair,
                                               seed=31, k=k, root_id=f"typed-k{k}", ledger=ledger)
            assert isinstance(nominal.log, ContactDenseLog)
            assert len(nominal.log.pusher_block_contacts) == k * 5 + 1
            assert isinstance(context.prefix_log, ContactDenseLog)
            contacts = context.prefix_log.pusher_block_contacts
            assert contacts.sum() > 0
            assert root.meta["contact_kind"] == "pusher_block"
            assert root.meta["contact_steps_prefix"] == int((contacts > 0).sum())
            assert root.meta["in_contact_last_block"] == bool((contacts[-5:] > 0).any())
            assert ledger.total == 2 * len(root.prefix)
        finally:
            env.close()

    env = make_env()
    initial = root_fixture([100, 350, 250, 390, 0, 0, 0])
    try:
        root, context, nominal = build_root(env, FixedPlanner([0, 0]),
            {"start": initial.start_state, "goal": initial.goal_state, "episode": 9, "t0": 0, "goal_offset": 25},
            seed=31, k=2, root_id="wall-only-nominal")
        assert nominal.log.block_wall_contacts.sum() > 0
        assert nominal.log.pusher_block_contacts.sum() == 0
        assert root.meta["in_contact_last_block"] is False
        assert root.meta["contact_steps_prefix"] == 0
        assert root.meta["any_collision_steps_prefix"] > 0
    finally:
        env.close()
