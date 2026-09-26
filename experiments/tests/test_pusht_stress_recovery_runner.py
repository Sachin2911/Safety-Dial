"""CPU-only runner contracts; all environment calls are explicit fake transitions."""
from __future__ import annotations

from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import pusht_stress_recovery as runner
from helpers.branchBank import Bank, Branch
from helpers.pushtBankSampling import SAMPLING_PROTOCOL, ProgressBankWriter, atomic_json
from helpers.pushtGeometry import Box
from helpers.pushtLayouts import Layout
from helpers.pushtReplay import DenseLog, Root, StepLedger
from helpers.runManifest import file_sha256
from helpers.splitIntegrity import bank_identity


def log_for(root, tape):
    return DenseLog(np.tile(root.start_state, (26, 1)), np.zeros((26, 2)), np.zeros(26),
        np.zeros(26, int), tape.reshape(-1, 2),
        [np.zeros((224, 224, 3), np.uint8) for _ in range(6)],
        observed=np.ones(26, bool), observation_valid=np.ones(26, bool))


def manifest(**kw):
    return {"status": "complete", "run_id": kw["run_id"], "kind": kw.get("kind"),
            "data": kw.get("data", {}), "costs": kw.get("costs", {}), "metrics": kw.get("metrics", {})}


@pytest.fixture
def recovery(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "TEST_ROOTS", 2)
    monkeypatch.setattr(runner, "DEV_ROOTS", 1)
    monkeypatch.setattr(runner, "FAILED_STRESS_STEPS", 560)
    monkeypatch.setattr(runner, "build_manifest", manifest)
    source = tmp_path / "original"
    source.mkdir()
    source_hashes = {}
    for relative in runner.NEW_SOURCES:
        target = source / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(runner.REPO_ROOT / relative, target)
        source_hashes[relative] = file_sha256(target)
    candidates, roots = {}, {}
    for role, episode, family, xy in (("dev", 0, "familiar", (192, 192)),
                                      ("test-familiar", 1, "familiar", (192, 192)),
                                      ("test-heldout", 2, "heldout", (64, 192))):
        state = np.array([256., 256., *xy, 0., 0., 0.])
        c = {"root_id": role, "episode": episode, "t0": 0, "goal_offset": 25,
             "k": 0, "pair_draw": 0, "candidate_round": 0, "candidate_index": 0,
             "source_position": 0, "root_seed": episode + 7,
             "start": state.tolist(), "goal": state.tolist()}
        candidates[role] = c
        roots[role] = Root(c["root_seed"], state, state.copy(), np.zeros((10, 2)), role,
            {**{k: c[k] for k in ("episode", "t0", "goal_offset", "k", "pair_draw", "candidate_round", "candidate_index", "source_position")},
             "source_family": family, "state_at_root": state.tolist(), "nominal_plan": np.zeros((5, 5, 2)).tolist()})
    plan = {"protocol_version": SAMPLING_PROTOCOL["version"], "sampling_protocol": SAMPLING_PROTOCOL,
            "settings": {"seed": 20260927, "dev_roots": 1, "test_roots": 1, "dev_tapes": 8, "test_tapes": 16},
            "source_roles": {"dev": {"familiar": [0]}, "test": {"familiar": [1], "heldout": [2]}},
            "candidate_schedule": {"dev": {"familiar": [candidates["dev"]]},
                "test": {"familiar": [candidates["test-familiar"]], "heldout": [candidates["test-heldout"]]}},
            "source_snapshot_sha256": source_hashes, "input_sha256": {}}
    atomic_json(source / "sampling-plan.json", plan)
    plan_hash = file_sha256(source / "sampling-plan.json")
    for root in roots.values():
        root.meta["sampling_plan_sha256"] = plan_hash
    for name, group, count in (("dev", [roots["dev"]], 8),
                              ("test", [roots["test-familiar"], roots["test-heldout"]], 16)):
        writer = ProgressBankWriter(source / name, bank_name=name, plan_path=source / "sampling-plan.json")
        ledger = StepLedger()
        for root in group:
            writer.add_root(root)
            families = ("familiar",) if name == "dev" else ("familiar", "heldout")
            writer.layouts.extend(Layout(Box(320, 340, 240, 260).to_dict(), family, root.root_id, .5, 0, 30, 30) for family in families)
            branches = []
            for i in range(count):
                tape = np.full((5, 5, 2), i * .001, np.float32)
                branches.append(Branch(root.root_id, tape, "random", {"sigma": .1}, log_for(root, tape)))
            writer.add_branches(branches)
            ledger.add("branch", count * 35, branches=count)
        m = manifest(run_id="original-" + name, kind="bank", costs=ledger.to_dict(),
                     data={"source_roles": plan["source_roles"], "sampling_plan_sha256": plan_hash})
        writer.finish(ledger, m)
        runner.write_manifest(source / name, m)
    atomic_json(source / "dev/hf_upload.json", {"repo_id": "private/banks", "repo_type": "dataset",
                "path": "banks/original-dev", "revision": "a" * 40, "run_id": "original-dev"})
    failed = ProgressBankWriter(source / "stress", bank_name="stress", plan_path=source / "sampling-plan.json")
    root = roots["test-familiar"]
    failed.add_root(root)
    failed.add_branches([Branch(root.root_id, np.zeros((5, 5, 2), np.float32), "stress", {},
                               log_for(root, np.zeros((5, 5, 2), np.float32))) for _ in range(16)])
    old_ledger = StepLedger()
    old_ledger.add("branch", 560, branches=16)
    failed.progress["attempts_completed"] = 1
    failed.event({"event": "completed", "charged_steps": 560, "ledger": old_ledger.to_dict()})
    failed.fail(old_ledger, RuntimeError("bounded proposal shortfall"))
    witness_dir = tmp_path / "original-witness"
    witness_dir.mkdir()
    witness = witness_dir / "witnesses.json"
    atomic_json(witness, {"run_id": "original-witness", "frozen_sampling_plan": {
        "path": str(source / "sampling-plan.json"), "sha256": plan_hash},
        "ledger": {"total_steps": 30, "steps": {"witness": 30}, "branches": {}},
        "witnesses": [{"retained": "original evidence"}], "gate": {"passes": True},
        "generation_phase": "before_final_banks"})
    calls = []
    def gate(path, bank, *, sampling_plan):
        report = runner.read_json(path)
        assert report["gate"]["passes"] and report["frozen_sampling_plan"]["path"] == str(sampling_plan.resolve())
        assert report["frozen_sampling_plan"]["sha256"] == file_sha256(sampling_plan)
        calls.append((str(path), str(bank)))
        return {"passes": True, "report": str(path), "sha256": file_sha256(path)}
    monkeypatch.setattr(runner, "require_feasibility_report", gate)
    args = SimpleNamespace(source_banks_dir=source, source_feasibility_report=witness,
        output_dir=tmp_path / "recovered", results_dir=tmp_path / "results",
        feasibility_output=tmp_path / "relocated-witness", run_id="recovery", no_upload=True)
    return SimpleNamespace(args=args, source=source, witness=witness, calls=calls, roots=roots)


def install_fake_execution(monkeypatch, *, mismatch=False, fail_at=None):
    class Env:
        def __init__(self):
            self.calls = 0
            self.closed = False
        def step(self, action):
            self.calls += 1
            if self.calls == fail_at:
                raise RuntimeError("fixture step failed")
        def close(self):
            self.closed = True
    env = Env()
    monkeypatch.setattr(runner, "make_env", lambda: env)
    def execute(metered, root, proposals):
        p = proposals[0]
        for action in list(root.prefix) + list(p.tape.reshape(-1, 2)):
            metered.step(action)
        log = log_for(root, p.tape)
        if mismatch:
            log.states[0, 0] += 1
        return [Branch(root.root_id, p.tape, p.kind, p.params, log)]
    monkeypatch.setattr(runner, "execute_proposals", execute)
    return env


def test_copy_relocation_preserves_bytes_and_only_rebases_witness(recovery):
    a = recovery.args
    inputs = runner.preflight(a.source_banks_dir, a.source_feasibility_report)
    original = runner.read_json(recovery.witness)
    runner.copy_inputs(inputs, a.output_dir, a.source_feasibility_report, a.feasibility_output, a.run_id)
    for name, hashes in inputs["trees"].items():
        assert runner.tree_hashes(a.output_dir / name) == hashes == runner.tree_hashes(recovery.source / name)
        assert (a.output_dir / name).stat().st_ino != (recovery.source / name).stat().st_ino
    relocated = runner.read_json(a.feasibility_output / "witnesses.json")
    provenance = relocated.pop("relocation")
    relocated["frozen_sampling_plan"]["path"] = original["frozen_sampling_plan"]["path"]
    assert relocated == original
    assert provenance["original_report_sha256"] == file_sha256(recovery.witness)
    assert provenance["new_witness_simulator_steps"] == 0
    assert len(recovery.calls) == 2
    assert not (a.feasibility_output / "hf_upload.json").exists()


def test_full_mock_recovery_freezes_before_environment_and_counts_old_failure(recovery, monkeypatch):
    a = recovery.args
    env = install_fake_execution(monkeypatch)
    def make_env():
        assert (a.output_dir / "stress-controls/controls-plan.json").is_file()
        assert runner.read_json(a.output_dir / "stress-controls/controls-plan.json")["branch_count"] == 32
        assert (a.feasibility_output / "witnesses.json").is_file()
        return env
    monkeypatch.setattr(runner, "make_env", make_env)
    original = runner.tree_hashes(recovery.source)
    report = runner.run(a)
    assert report["status"] == "complete" and env.closed and env.calls == 32 * 35
    costs = report["costs"]
    assert costs["new_stress"]["total_steps"] == 1120
    assert costs["failed_stress"]["total_steps"] == 560
    assert costs["total_study"]["total_steps"] == 280 + 1120 + 560 + 30 + 1120
    assert costs["unaccounted_history"]["v2_failed_bank_steps"] is None
    assert costs["new_stress"]["branches"]["proposals_attempted"] >= 32
    assert runner.tree_hashes(recovery.source) == original
    assert bank_identity(a.output_dir / "test") == bank_identity(recovery.source / "test")
    assert not list((a.output_dir / "stress").rglob("*.py"))
    archived = a.output_dir / "prior-failed-stress"
    assert runner.read_json(archived / "manifest.json")["status"] == "incomplete"
    assert runner.read_json(archived / "roots.json")["status"] == "incomplete"
    for name, sha in runner.tree_hashes(recovery.source / "stress").items():
        assert file_sha256(archived / name) == sha
    bank = Bank(a.output_dir / "stress")
    try:
        assert len(bank) == 32
        assert [r.root_id for r in bank.roots] == ["test-familiar", "test-heldout"]
    finally:
        bank.h5.close()


@pytest.mark.parametrize("failure", ["initial_state", "step"])
def test_paid_trace_is_retained_before_replay_mismatch_or_later_exception(recovery, monkeypatch, failure):
    a = recovery.args
    env = install_fake_execution(monkeypatch, mismatch=failure == "initial_state", fail_at=40 if failure == "step" else None)
    with pytest.raises((ValueError, RuntimeError), match="initial state|step failed"):
        runner.run(a)
    report = runner.read_json(a.results_dir / "banks.json")
    assert report["status"] == "incomplete" and env.closed
    assert report["costs"]["new_stress"]["total_steps"] == (35 if failure == "initial_state" else 40)
    assert runner.read_json(a.output_dir / "stress/progress.json")["status"] == "incomplete"
    with h5py.File(a.output_dir / "stress/branches.h5", "r") as h5:
        assert len(h5["tape"]) == 1
    with pytest.raises(ValueError, match="incomplete"):
        Bank(a.output_dir / "stress")
    assert not (a.results_dir / "manifest.json").exists()


def test_preflight_failure_queries_nothing_and_preserves_inputs(recovery, monkeypatch):
    a = recovery.args
    monkeypatch.setattr(runner, "make_env", lambda: pytest.fail("environment created during preflight"))
    monkeypatch.setattr(runner, "freeze_controls", lambda *args: (_ for _ in ()).throw(ValueError("pool shortfall")))
    before = runner.tree_hashes(recovery.source)
    with pytest.raises(ValueError, match="pool shortfall"):
        runner.run(a)
    assert not (a.output_dir / "stress").exists()
    assert runner.read_json(a.results_dir / "banks.json")["costs"]["new_stress"]["total_steps"] == 0
    assert runner.tree_hashes(recovery.source) == before


def test_frozen_control_tampering_fails_before_environment(recovery, monkeypatch):
    a = recovery.args
    inputs = runner.preflight(a.source_banks_dir, a.source_feasibility_report)
    runner.copy_inputs(inputs, a.output_dir, a.source_feasibility_report, a.feasibility_output, a.run_id)
    roots, _, directory = runner.freeze_controls(inputs, a.output_dir)
    pools, _ = runner.load_frozen_controls(directory, roots)
    assert len(pools) == 2
    (directory / "tapes.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        runner.load_frozen_controls(directory, roots)


def test_refuses_reuse_links_and_nested_output(recovery):
    a = recovery.args
    a.output_dir.mkdir()
    with pytest.raises(FileExistsError):
        runner.run(a)
    a.output_dir = recovery.source / "new-child"
    with pytest.raises(ValueError, match="preserved input"):
        runner.run(a)
    linked = recovery.source / "dev/linked"
    linked.symlink_to(recovery.source / "test/roots.json")
    with pytest.raises(ValueError, match="Symlinks"):
        runner.tree_hashes(recovery.source / "dev")


def test_bad_failed_cost_or_source_bank_completion_fails_preflight(recovery):
    source = recovery.source
    p = source / "stress/progress.json"
    progress = runner.read_json(p)
    progress["ledger"]["total_steps"] -= 1
    atomic_json(p, progress)
    with pytest.raises(ValueError, match="charged terminal"):
        runner.preflight(source, recovery.witness)
    p = source / "test/progress.json"
    progress = runner.read_json(p)
    progress["status"] = "running"
    atomic_json(p, progress)
    with pytest.raises(ValueError, match="coherently completed"):
        runner.preflight(source, recovery.witness)


def test_success_uploads_relocated_witness_and_reuses_identical_dev_receipt(recovery, monkeypatch):
    a = recovery.args
    a.no_upload = False
    install_fake_execution(monkeypatch)
    calls = []
    class Store:
        def reference_run(self, key, kind, directory):
            assert directory.name == "dev"
            return {"revision": "a" * 40}
        def upload_run(self, key, kind, directory, *, run_id):
            calls.append((kind, run_id))
            receipt = {"revision": "b" * 40, "run_id": run_id}
            atomic_json(directory / "hf_upload.json", receipt)
            return receipt["revision"]
    monkeypatch.setattr(runner, "HFStore", Store)
    report = runner.run(a)
    assert calls == [("prior-failed-stress", "recovery-prior-failed-stress"),
                     ("banks", "original-test"), ("banks", "recovery-stress"),
                     ("development-witness", "recovery-witness")]
    assert report["hf_revisions"]["dev"] == "a" * 40
    assert runner.read_json(a.feasibility_output / "hf_upload.json")["run_id"] == "recovery-witness"
    assert not (recovery.witness.parent / "hf_upload.json").exists()


def test_successful_partial_upload_revision_survives_later_upload_error(recovery, monkeypatch):
    a = recovery.args
    a.no_upload = False
    install_fake_execution(monkeypatch)
    class Store:
        def reference_run(self, *args):
            return {"revision": "a" * 40}
        def upload_run(self, key, kind, directory, *, run_id):
            if directory.name == "stress":
                raise RuntimeError("upload failed")
            atomic_json(directory / "hf_upload.json", {"revision": "b" * 40})
            return "b" * 40
    monkeypatch.setattr(runner, "HFStore", Store)
    with pytest.raises(RuntimeError, match="upload failed"):
        runner.run(a)
    report = runner.read_json(a.results_dir / "banks.json")
    assert report["status"] == "incomplete" and report["hf_revisions"]["test"] == "b" * 40
    assert runner.read_json(a.output_dir / "stress/progress.json")["status"] == "complete"
