"""CPU-only regression checks for confirmed split, metric and durability failures."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from helpers import hfStore  # noqa: E402
from helpers.dialMetrics import auc_dial, fsa, margin_for_acceptance  # noqa: E402
from helpers.hfStore import HFStore  # noqa: E402
from helpers.runManifest import sanitise_remote, write_manifest  # noqa: E402
from helpers.splitIntegrity import (  # noqa: E402
    inspect_bank_splits,
    partition_source_roles,
    validate_bank_splits,
)


def bank(tmp_path, name, episodes):
    out = tmp_path / name
    out.mkdir()
    roots = [{"root_id": f"{name}-{i}", "meta": {"episode": ep}}
             for i, ep in enumerate(episodes)]
    (out / "roots.json").write_text(json.dumps({"roots": roots}))
    return out


def fake_store(*, private=True):
    store = HFStore.__new__(HFStore)
    store.api = MagicMock()
    store.namespace = "test-owner"
    store.token = "test-only"
    store.api.repo_info.return_value = SimpleNamespace(private=private, sha="a" * 40)
    return store


def test_splits_detect_shared_episodes_despite_distinct_root_ids(tmp_path):
    paths = {"dev": bank(tmp_path, "dev", [10, 11]),
             "test": bank(tmp_path, "test", [11, 12])}
    report = inspect_bank_splits(paths)
    assert not report["passes"]
    assert report["overlaps"][0]["source_episodes"] == [11]
    with pytest.raises(ValueError, match="1 shared source episodes"):
        validate_bank_splits(paths)


def test_stress_may_share_test_sources_but_never_development(tmp_path):
    paths = {"dev": bank(tmp_path, "dev", [1]),
             "test": bank(tmp_path, "test", [2]),
             "stress": bank(tmp_path, "stress", [2])}
    assert validate_bank_splits(paths)["passes"]


def test_partition_occurs_before_root_generation_and_is_reproducible():
    families = {"familiar": list(range(20)), "heldout": list(range(20, 30))}
    roles = partition_source_roles(families, 17)
    assert roles == partition_source_roles(families, 17)
    dev = set(roles["dev"]["familiar"])
    test = set(roles["test"]["familiar"] + roles["test"]["heldout"])
    assert not dev & test
    assert dev | test == set(range(30))


def test_zero_acceptance_and_empty_population_are_undefined():
    result = fsa([1.0, 2.0], [1, 0], float("inf"))
    assert result["n_accepted"] == 0
    assert np.isnan(result["fsa"])
    assert np.isnan(fsa([], [], 0.0)["acceptance_rate"])


@pytest.mark.parametrize("clearance,truth", [([float("nan")], [0]), ([1.0], [None]),
                                             ([1.0], [float("nan")]), ([1.0], [0, 1])])
def test_metrics_require_explicitly_observed_valid_rows(clearance, truth):
    with pytest.raises(ValueError):
        fsa(clearance, truth, 0.0)


def test_margin_respects_from_above_contract_and_large_floats():
    c = np.arange(10, dtype=float)
    assert np.mean(c >= margin_for_acceptance(c, 0.25)) == 0.3
    c = np.array([1e20, 1e20])
    assert not (c >= margin_for_acceptance(c, 0.0)).any()
    with pytest.raises(ValueError):
        margin_for_acceptance([], 0.5)


def test_auc_uses_reachable_acceptance_including_ties():
    c, u = np.array([2., 2., 1., 0.]), np.array([0, 1, 1, 0])
    # Actual dial points are (0.5, 0.5), (0.75, 2/3), (1, 0.5).
    assert auc_dial(c, u, 0.5, 1.) == pytest.approx(7 / 12)
    assert np.isnan(auc_dial([1., 1.], [0, 1], 0.2, 0.9))


def test_existing_public_repository_is_rejected_before_upload(tmp_path):
    store = fake_store(private=False)
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(RuntimeError, match="not a private repository"):
        store.upload_run("pusht", "adapted", tmp_path)
    store.api.upload_folder.assert_not_called()


def test_successful_upload_persists_its_commit_receipt(tmp_path):
    store = fake_store()
    store.api.upload_folder.return_value = SimpleNamespace(oid="b" * 40)
    (tmp_path / "manifest.json").write_text("{}")
    sha = store.upload_run("pusht", "adapted", tmp_path, run_id="unique-run")
    record = json.loads((tmp_path / "hf_upload.json").read_text())
    assert record["revision"] == sha == "b" * 40
    assert record["path"] == "adapted/unique-run"


@pytest.mark.parametrize("revision", ["", "main", "master", "latest", "HEAD", "moving-branch"])
def test_download_refuses_moving_revisions(tmp_path, revision):
    store = fake_store()
    store.api.list_repo_refs.return_value = SimpleNamespace(tags=[])
    with pytest.raises(ValueError):
        store.download_run("pusht", "assets/run", revision=revision, local_root=tmp_path)


def test_download_resolves_run_tag_to_immutable_commit(tmp_path, monkeypatch):
    store = fake_store()
    store.api.list_repo_refs.return_value = SimpleNamespace(tags=[SimpleNamespace(name="run-tag")])
    download = MagicMock()
    monkeypatch.setattr(hfStore, "snapshot_download", download)
    (tmp_path / "assets" / "run").mkdir(parents=True)
    out = store.download_run("pusht", "assets/run", revision="run-tag", local_root=tmp_path)
    assert download.call_args.kwargs["revision"] == "a" * 40
    assert json.loads((out / "hf_download.json").read_text())["revision"] == "a" * 40


def test_manifest_redacts_url_credentials_and_writes_atomically(tmp_path):
    remote = "https://user:secret@example.com/owner/repo.git?token=secret#secret"
    assert sanitise_remote(remote) == "https://example.com/owner/repo.git"
    assert sanitise_remote("git@example.com:owner/repo.git") == "git@example.com:owner/repo.git"
    path = write_manifest(tmp_path, {"run_id": "test"})
    assert json.loads(path.read_text()) == {"run_id": "test"}
    assert not path.with_suffix(".json.tmp").exists()


class FakePushT:
    def __init__(self, *, terminal_step=None, truncated_step=None, exit_step=None):
        self.unwrapped = self
        self.block = SimpleNamespace(velocity=(0., 0.), angular_velocity=0.)
        self.steps = 0
        self.terminal_step, self.truncated_step, self.exit_step = terminal_step, truncated_step, exit_step

    def _get_obs(self):
        x = 513. if self.exit_step is not None and self.steps >= self.exit_step else 256.
        return np.array([x, 256., 256., 256., 0., 0., 0.])

    def render(self):
        return np.full((224, 224, 3), self.steps, dtype=np.uint8)

    def step(self, action):
        self.steps += 1
        return None, 0., self.steps == self.terminal_step, self.steps == self.truncated_step, {}


def test_terminal_padding_is_explicit_and_not_executed():
    from helpers.pushtReplay import run_actions

    env = FakePushT(terminal_step=2)
    log = run_actions(env, np.zeros((25, 2)), record_frames=True)
    assert env.steps == log.executed_steps == 2
    assert log.censored and not log.valid_for_training
    assert log.observed.sum() == 3
    assert not log.observed[3:].any()
    assert len(log.frames) == 6
    assert log.terminated[2]


def test_final_step_termination_is_a_complete_observed_horizon():
    from helpers.pushtReplay import run_actions

    log = run_actions(FakePushT(terminal_step=25), np.zeros((25, 2)))
    assert log.executed_steps == 25
    assert not log.censored
    assert log.valid_for_training


def test_domain_exit_retains_dense_geometry_but_invalidates_image_targets():
    from helpers.decomposition import outcome_metadata
    from helpers.pushtReplay import run_actions

    log = run_actions(FakePushT(exit_step=2), np.zeros((25, 2)))
    assert log.executed_steps == 25 and not log.censored
    assert not log.valid_for_training
    row = outcome_metadata(log.states, np.ones(26), observed=log.observed,
                           observation_valid=log.observation_valid)
    assert row["unsafe_composite"] and row["observation_domain_exit"]
    assert not row["hazard_unsafe_observed"]
    assert fsa([1.], [row["unsafe_composite"]], 0.)["fsa"] == 1.


def test_unseen_padded_future_is_neither_hazard_safe_nor_an_observed_violation():
    from helpers.decomposition import outcome_metadata
    from helpers.pushtReplay import run_actions

    log = run_actions(FakePushT(truncated_step=2), np.zeros((25, 2)))
    c = np.ones(26)
    c[3:] = -1.  # An arbitrary padded value must not manufacture a true future.
    row = outcome_metadata(log.states, c, observed=log.observed)
    assert row["censored"] and row["hazard_only_unsafe"] is None
    assert not row["unsafe_composite"]
    metric = fsa([1.], [row["unsafe_composite"]], 0., censored=[True])
    assert np.isnan(metric["fsa"])
    assert metric["n_accepted_censored"] == 1
    assert metric["fsa_lower"] == 0. and metric["fsa_upper"] == 1.


def test_known_failure_resolves_censored_binary_unsafe_event():
    metric = fsa([1., 1.], [True, False], 0., censored=[True, False])
    assert metric["fsa"] == 0.5
    assert metric["n_accepted_censored"] == 0


def test_bank_storage_roundtrips_masks_and_rejects_censored_training(tmp_path):
    from helpers.branchBank import Bank, BankWriter, Branch
    from helpers.predictorAdapt import branch_clips
    from helpers.pushtReplay import Root, StepLedger, run_actions

    log = run_actions(FakePushT(terminal_step=2), np.zeros((25, 2)), record_frames=True)
    root = Root(1, log.states[0], log.states[0], np.zeros((10, 2)), "r", {"episode": 1})
    writer = BankWriter(tmp_path)
    writer.add_root(root)
    writer.add_branches([Branch("r", np.zeros((5, 5, 2)), "random", {}, log)])
    writer.finish(StepLedger())
    stored = Bank(tmp_path)
    assert np.array_equal(stored.branch(0)["observed"], log.observed)
    assert not len(stored.valid_training_indices())
    with pytest.raises(ValueError, match="No complete"):
        branch_clips(None, stored, [0], env=object())
    stored.h5.close()


def test_failed_root_generation_keeps_every_actual_step_in_ledger(monkeypatch):
    from helpers import branchBank
    from helpers.pushtReplay import StepLedger

    def rejected(env, planner, pair, **kwargs):
        env.step(np.zeros(2))
        env.step(np.zeros(2))
        raise ValueError("censored or out-of-domain prefix")

    monkeypatch.setattr(branchBank, "_build_root", rejected)
    ledger = StepLedger()
    with pytest.raises(ValueError):
        branchBank.build_root(FakePushT(), None, {}, seed=1, k=0, root_id="rejected", ledger=ledger)
    assert ledger.total == 2


def test_legacy_bank_without_masks_reconstructs_domain_exit(tmp_path):
    import h5py

    from helpers.branchBank import Bank, BankWriter, Branch
    from helpers.pushtReplay import Root, StepLedger, run_actions

    log = run_actions(FakePushT(exit_step=2), np.zeros((25, 2)), record_frames=True)
    root = Root(1, log.states[0], log.states[0], np.zeros((10, 2)), "r", {"episode": 1})
    writer = BankWriter(tmp_path)
    writer.add_root(root)
    writer.add_branches([Branch("r", np.zeros((5, 5, 2)), "random", {}, log)])
    writer.finish(StepLedger())
    with h5py.File(tmp_path / "branches.h5", "a") as f:
        for key in ("observed", "observation_valid", "terminated", "truncated"):
            del f[key]
    bank = Bank(tmp_path)
    assert not bank.branch_valid_for_training(0)
    bank.h5.close()


def test_existing_bundle_receipt_pins_revision_and_local_weight_hash(tmp_path):
    from helpers.runManifest import file_sha256

    store = fake_store()
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "weights.pt").write_bytes(b"mock weights")
    (tmp_path / "hf_upload.json").write_text(json.dumps({
        "repo_id": "test-owner/safetydial-pusht", "path": "probes/older-run",
        "revision": "c" * 40,
    }))
    reference = store.reference_run("pusht", "probes", tmp_path)
    assert reference["revision"] == "c" * 40
    assert reference["files_sha256"]["weights.pt"] == file_sha256(tmp_path / "weights.pt")
    store.api.repo_info.assert_not_called()


@pytest.mark.parametrize("script", ["pusht_e1_decompose.py", "pusht_e2_repair.py"])
def test_evaluation_entrypoints_block_leaked_sources_before_model_loading(tmp_path, monkeypatch, script):
    import importlib.util

    bank(tmp_path, "dev", [1, 2])
    bank(tmp_path, "test", [2, 3])
    bank(tmp_path, "stress", [2, 3])
    path = Path(__file__).resolve().parents[1] / "experiments" / "scripts" / script
    spec = importlib.util.spec_from_file_location("audit_entrypoint", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    load_model = MagicMock(side_effect=AssertionError("GPU model must not load before split validation"))
    monkeypatch.setattr(module, "load_model", load_model)
    args = [script, "--banks-dir", str(tmp_path), "--results-dir", str(tmp_path / "result")]
    if script == "pusht_e2_repair.py":
        args += ["--feasibility-report", str(tmp_path / "not-read-before-split-check-witness.json"), "--study-dir", str(tmp_path / "study"), "--decomposition-report", str(tmp_path / "not-read-before-split-check.json")]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(ValueError, match="shared source episodes"):
        module.main()
    load_model.assert_not_called()
    assert not (tmp_path / "result").exists()


def test_probe_branch_labels_keep_the_original_expert_source_id(tmp_path, monkeypatch):
    import importlib.util

    from helpers import imagination
    from helpers.branchBank import BankWriter, Branch
    from helpers.pushtReplay import Root, StepLedger, run_actions

    log = run_actions(FakePushT(), np.zeros((25, 2)), record_frames=True)
    root = Root(1, log.states[0], log.states[0], np.zeros((10, 2)), "probe-root", {"episode": 7})
    writer = BankWriter(tmp_path)
    writer.add_root(root)
    writer.add_branches([Branch(root.root_id, np.zeros((5, 5, 2)), "random", {}, log)])
    writer.finish(StepLedger())
    path = Path(__file__).resolve().parents[1] / "experiments/scripts/pusht_e1_probes.py"
    spec = importlib.util.spec_from_file_location("audit_probes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Encoded:
        def cpu(self):
            return self

        def numpy(self):
            return np.zeros((6, 3), dtype=np.float32)

    monkeypatch.setattr(imagination, "encode", lambda *args: Encoded())
    _, _, episodes, _ = module.probe_bank_latents(
        None, {"roles": {"probe": [7]}}, 1, 1, np.random.default_rng(0), "cpu", bank_dir=tmp_path)
    assert set(episodes.tolist()) == {7}


def test_exact_push_t_contact_is_an_observed_violation():
    from helpers.decomposition import outcome_metadata

    states = np.array([[256., 256., 256., 256., 0., 0., 0.]])
    row = outcome_metadata(states, [0.])
    assert row["hazard_unsafe_observed"]
    assert fsa([0.], [row["unsafe_composite"]], 0.)["fsa"] == 1.


def test_development_decomposition_gate_requires_counts_and_imagination_share():
    from helpers.splitIntegrity import decomposition_gate

    assert decomposition_gate({"attribution": {"temporal": 1, "readout": 1, "imagination": 3}})["passes"]
    assert not decomposition_gate({"attribution": {"temporal": 2, "readout": 20, "imagination": 3}})["passes"]
    assert not decomposition_gate({"attribution": {"imagination": 2}})["passes"]


def test_decomposition_gate_is_bound_to_exact_banks_and_frozen_probe(tmp_path):
    from helpers.runManifest import file_sha256
    from helpers.splitIntegrity import bank_identity, validate_decomposition_gate

    for name in ("dev", "test", "stress"):
        folder = tmp_path / name
        folder.mkdir()
        for filename in ("roots.json", "branches.h5", "layouts.json"):
            (folder / filename).write_bytes(f"{name}/{filename}".encode())
    probes = tmp_path / "probes"
    probes.mkdir()
    weights = probes / "block_pose_mlp.pt"
    weights.write_bytes(b"frozen readout")
    report = {"gate": {"passes": True, "role": "development"}, "probe": "block_pose_mlp",
              "bank_identities": {name: bank_identity(tmp_path / name) for name in ("dev", "test", "stress")},
              "probe_sha256": file_sha256(weights)}
    assert validate_decomposition_gate(report, tmp_path, probes) == "block_pose_mlp"
    weights.write_bytes(b"different readout")
    with pytest.raises(ValueError, match="probe identities differ"):
        validate_decomposition_gate(report, tmp_path, probes)


def test_checkpoint_history_budget_preserves_every_intermediate_and_fails_early():
    from helpers.storageBudget import checkpoint_storage_budget, require_storage_budget

    profile = {"complete": True, "used_bytes": 100, "quota_bytes": 1000}
    budget = checkpoint_storage_budget(profile, total_steps=10, push_every=4,
        checkpoint_bytes=100, model_runs=2, reserve_bytes=100, overhead_fraction=0.)
    assert budget["versions_per_model"] == 3
    assert budget["raw_history_bytes"] == 600
    assert require_storage_budget(budget)["passes"]
    profile["quota_bytes"] = 500
    failed = checkpoint_storage_budget(profile, total_steps=10, push_every=4,
        checkpoint_bytes=100, model_runs=2, reserve_bytes=100, overhead_fraction=0.)
    with pytest.raises(RuntimeError, match="insufficient"):
        require_storage_budget(failed)


@pytest.mark.parametrize("run_id", ["../unsafe", "path/child", "", ".", "..", "a b", "a\\b"])
def test_explicit_run_ids_cannot_escape_the_run_directory(run_id):
    from helpers.runManifest import validate_run_id

    with pytest.raises(ValueError):
        validate_run_id(run_id)


def test_explicit_run_id_stays_stable_for_multiday_workflows():
    from helpers.runManifest import validate_run_id

    run_id = "pusht-probes-continuation-20260926-2"
    assert validate_run_id(run_id) == run_id
