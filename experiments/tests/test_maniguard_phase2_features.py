"""Causal feature-contract checks, with no learned encoder or simulator execution."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.maniguard_adaptation import Frame
from helpers.maniguard_phase2_features import (
    combine_features,
    finite_vector,
    project,
    projection_matrices,
    robot_features,
)


def frame(**kwargs):
    return Frame(np.zeros(7), np.array([.01, .01]), np.zeros(3), **kwargs)


def test_initial_features_do_not_invent_unobserved_history():
    state = frame()
    proposal = np.r_[np.full(7, .05), -1.0]
    actual = robot_features(state, proposal)
    np.testing.assert_array_equal(actual[9:19], np.zeros(10))
    np.testing.assert_array_equal(actual[26:33], np.zeros(7))
    np.testing.assert_allclose(actual[19:26], .25)
    assert actual.shape == (40,) and np.isfinite(actual).all()
    assert actual[33] == -1 and actual[34] == -1


def test_tracking_uses_applied_command_and_measured_joints_not_previous_proposal():
    proposal = np.r_[np.full(7, .1), 1.0]
    old_proposal = np.r_[np.full(7, .9), -1.0]
    applied = np.r_[np.full(7, .04), -1.0]
    actual = robot_features(frame(), proposal, last_executed=applied, last_proposal=old_proposal)
    np.testing.assert_allclose(actual[12:19], .2)
    np.testing.assert_array_equal(actual[26:33], np.full(7, -1.0))
    assert actual[33] == 1 and actual[34] == -1
    np.testing.assert_array_equal(applied, np.r_[np.full(7, .04), -1.0])


def test_requested_motion_uses_controller_feasible_targets_without_changing_proposal():
    bounds = {"joint_lower_limits": np.full(7, -.1), "joint_upper_limits": np.full(7, .1),
              "joint_has_limits": np.array([True, False, True, False, True, False, True])}
    proposal = np.r_[np.full(7, .3), -1.0]
    actual = robot_features(frame(**bounds), proposal)
    np.testing.assert_allclose(actual[19:26], [.5, 1, .5, 1, .5, 1, .5])
    np.testing.assert_array_equal(proposal, np.r_[np.full(7, .3), -1.0])


def test_end_effector_delta_uses_previous_decision_observation():
    actual = robot_features(frame(), np.zeros(8), last_eef=np.array([-.01, .02, 0.0]))
    np.testing.assert_allclose(actual[9:12], [.2, -.4, 0])


@pytest.mark.parametrize("field", ["prerequisite", "target_engaged"])
def test_privileged_false_labels_are_rejected_as_well_as_true(field):
    for value in (False, True):
        with pytest.raises(ValueError, match="Privileged"):
            robot_features(frame(**{field: value}), np.zeros(8))


def test_fixed_projection_is_reproducible_normalized_and_does_not_fit_observations():
    visual, text = projection_matrices()
    repeated_visual, repeated_text = projection_matrices()
    np.testing.assert_array_equal(visual, repeated_visual)
    np.testing.assert_array_equal(text, repeated_text)
    np.testing.assert_allclose(visual.T @ visual, np.eye(8), atol=1e-6)
    assert not np.array_equal(visual, text)
    embedding = np.random.default_rng(2).normal(size=512).astype(np.float32)
    expected = project(embedding, visual)
    np.testing.assert_allclose(project(embedding * 3, visual), expected, atol=1e-7)
    assert expected.shape == (8,) and np.max(np.abs(expected)) <= 1
    with pytest.raises(ValueError, match="Zero"):
        project(np.zeros(512), visual)


def test_visual_and_task_context_remain_separate_and_combination_copies_inputs():
    inputs = [np.zeros(40), np.ones(8) * .1, np.ones(8) * .2, np.ones(8) * .3]
    actual = combine_features(*inputs)
    assert actual.shape == (64,)
    np.testing.assert_allclose(actual[40:48], .1)
    np.testing.assert_allclose(actual[48:56], .2)
    np.testing.assert_allclose(actual[56:64], .3)
    inputs[0][:] = 1
    np.testing.assert_array_equal(actual[:40], np.zeros(40))


@pytest.mark.parametrize("value", [np.ones(8, dtype=bool), np.array(["1"] * 8),
                                  np.full(8, 1 + 2j), np.full(8, np.nan)])
def test_feature_validation_rejects_non_real_and_nonfinite_arrays(value):
    with pytest.raises(ValueError):
        finite_vector(value, (8,), "fixture")


@pytest.mark.parametrize("kwargs", [{"since_query": -1}, {"opening_steps": -1},
                                    {"choices": [0, 3, 0, 0]}, {"choices": [0, 0, 0]}])
def test_feature_history_rejects_invalid_counter_and_choice_domains(kwargs):
    with pytest.raises(ValueError):
        robot_features(frame(), np.zeros(8), **kwargs)


def test_failed_encoder_attempts_preserve_elapsed_work_and_do_not_count_successes():
    from contextlib import nullcontext
    from types import SimpleNamespace

    from helpers.maniguard_phase2_features import FrozenContextEncoder

    def fail(*args, **kwargs):
        raise RuntimeError("injected encoder failure")

    encoder = FrozenContextEncoder.__new__(FrozenContextEncoder)
    encoder.device = "cpu"
    encoder.torch = SimpleNamespace(inference_mode=nullcontext)
    encoder.model = SimpleNamespace(encode_text=fail)
    encoder.tokenizer = lambda prompts: SimpleNamespace(to=lambda device: None)
    encoder.preprocess = fail
    encoder.text_features = np.zeros(8)
    encoder.image_calls = encoder.image_count = encoder.text_calls = 0
    encoder.image_attempts = encoder.text_attempts = 0
    encoder.elapsed = 0.0
    with pytest.raises(RuntimeError, match="injected encoder"):
        encoder.begin_episode("actual instruction")
    text_elapsed = encoder.elapsed
    assert text_elapsed > 0
    with pytest.raises(RuntimeError, match="injected encoder"):
        encoder.encode_images(np.zeros((4, 4, 3), dtype=np.uint8),
                              np.zeros((4, 4, 3), dtype=np.uint8))
    ledger = encoder.ledger()
    assert ledger["encoder_image_attempts"] == ledger["encoder_text_attempts"] == 1
    assert ledger["encoder_image_calls"] == ledger["encoder_text_calls"] == 0
    assert ledger["encoder_images"] == 0
    assert ledger["encoder_seconds"] > text_elapsed


def _package_fixture(tmp_path, monkeypatch, *, module_version='1.2.3', metadata_version='1.2.3',
                     record_module=True):
    import types

    site = tmp_path / 'loaded'
    package = site / 'phase2_version_fixture'
    package.mkdir(parents=True)
    origin = package / '__init__.py'
    origin.write_text(f'__version__ = {module_version!r}\n')
    metadata = site / f'phase2_version_fixture-{metadata_version}.dist-info'
    metadata.mkdir()
    (metadata / 'METADATA').write_text(
        f'Metadata-Version: 2.1\nName: phase2-version-fixture\nVersion: {metadata_version}\n')
    records = [f'{metadata.name}/METADATA,,', f'{metadata.name}/RECORD,,']
    if record_module:
        records.append('phase2_version_fixture/__init__.py,,')
    (metadata / 'RECORD').write_text('\n'.join(records) + '\n')
    module = types.ModuleType('phase2_version_fixture')
    module.__file__ = str(origin)
    module.__version__ = module_version
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.syspath_prepend(str(site))
    return module, metadata


def test_loaded_version_survives_later_metadata_shadowing(tmp_path, monkeypatch):
    import importlib.metadata

    from helpers.maniguard_phase2_features import loaded_package_provenance

    module, owner = _package_fixture(tmp_path, monkeypatch)
    shadow = tmp_path / 'vendor'
    metadata = shadow / 'phase2_version_fixture-0.0.1.dist-info'
    metadata.mkdir(parents=True)
    (metadata / 'METADATA').write_text(
        'Metadata-Version: 2.1\nName: phase2-version-fixture\nVersion: 0.0.1\n')
    monkeypatch.syspath_prepend(str(shadow))
    assert importlib.metadata.version('phase2-version-fixture') == '0.0.1'
    actual = loaded_package_provenance('phase2-version-fixture', module.__name__, '1.2.3')
    assert actual['loaded_version'] == actual['distribution_version'] == '1.2.3'
    assert actual['global_metadata_version'] == '0.0.1' and actual['global_metadata_shadowed']
    assert actual['module_file'] == module.__file__
    assert actual['distribution_metadata_file'] == str(owner / 'METADATA')
    assert len(actual['module_sha256']) == len(actual['distribution_metadata_sha256']) == 64


def test_matching_global_metadata_cannot_mask_wrong_loaded_version(tmp_path, monkeypatch):
    import importlib.metadata

    from helpers.maniguard_phase2_features import loaded_package_provenance

    module, _ = _package_fixture(tmp_path, monkeypatch, module_version='9.9.9')
    assert importlib.metadata.version('phase2-version-fixture') == '1.2.3'
    with pytest.raises(ValueError, match='loaded runtime version mismatch'):
        loaded_package_provenance('phase2-version-fixture', module.__name__, '1.2.3')


def test_matching_module_version_cannot_mask_wrong_owning_distribution(tmp_path, monkeypatch):
    from helpers.maniguard_phase2_features import loaded_package_provenance

    module, _ = _package_fixture(tmp_path, monkeypatch, metadata_version='9.9.9')
    with pytest.raises(ValueError, match='owning distribution version mismatch'):
        loaded_package_provenance('phase2-version-fixture', module.__name__, '1.2.3')


def test_unowned_module_cannot_borrow_adjacent_distribution_version(tmp_path, monkeypatch):
    from helpers.maniguard_phase2_features import loaded_package_provenance

    module, _ = _package_fixture(tmp_path, monkeypatch, record_module=False)
    with pytest.raises(ValueError, match='owning distribution RECORD'):
        loaded_package_provenance('phase2-version-fixture', module.__name__, '1.2.3')


def test_actual_loaded_numpy_can_ignore_only_unrelated_metadata(tmp_path, monkeypatch):
    import importlib.metadata

    from helpers.maniguard_phase2_features import loaded_package_provenance

    before = loaded_package_provenance('numpy', 'numpy', np.__version__)
    metadata = tmp_path / 'numpy-0.0.0.dist-info'
    metadata.mkdir()
    (metadata / 'METADATA').write_text('Metadata-Version: 2.1\nName: numpy\nVersion: 0.0.0\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    assert importlib.metadata.version('numpy') == '0.0.0'
    after = loaded_package_provenance('numpy', 'numpy', np.__version__)
    assert after['loaded_version'] == before['loaded_version']
    assert after['module_file'] == before['module_file']
    assert after['distribution_metadata_sha256'] == before['distribution_metadata_sha256']
    assert after['global_metadata_shadowed'] and after['global_metadata_version'] == '0.0.0'


@pytest.mark.parametrize('problem', ['missing_dependency', 'extra_dependency', 'implementation_pin'])
def test_encoder_contract_cannot_silently_drop_or_change_dependencies(problem):
    import json

    from helpers.maniguard_phase2_features import ROOT, encoder_runtime_provenance

    spec = json.loads((ROOT / 'configs/maniguard/phase2/encoder.json').read_text())
    if problem == 'missing_dependency':
        del spec['runtime_versions']['numpy']
    elif problem == 'extra_dependency':
        spec['runtime_versions']['unknown'] = '1.0'
    else:
        spec['implementation_version'] = '0.0'
    with pytest.raises(ValueError, match='contract|pins disagree'):
        encoder_runtime_provenance(spec)
