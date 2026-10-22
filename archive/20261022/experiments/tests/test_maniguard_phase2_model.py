"""Actor contracts only; passing these does not establish robot performance."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.maniguard_phase2_model import (
    DEFAULT_SPEC,
    ModelSpec,
    decode,
    forward,
    genome_hash,
    genome_sha256,
    identity_genome,
    load_model,
    log_prob,
    make_torch_actor,
    pack_genome,
    random_genome,
    save_model,
    torch_entropy,
    torch_log_prob,
    unpack_genome,
)


def test_identity_is_exact_for_arbitrary_feature_histories_and_reset():
    genome = identity_genome()
    assert DEFAULT_SPEC.parameter_count == 692
    rng = np.random.default_rng(19)
    features = rng.normal(size=(40, 64))
    hidden = None
    for feature in features:
        logits, hidden = forward(genome, feature, hidden)
        np.testing.assert_array_equal(decode(logits), np.zeros(4, dtype=int))
        np.testing.assert_array_equal(logits, np.tile([0.25, 0, 0], (4, 1)))
    _, reset_hidden = forward(genome, features[0])
    _, repeated_hidden = forward(genome, features[0], np.zeros(8))
    np.testing.assert_array_equal(reset_hidden, repeated_hidden)
    assert np.linalg.norm(hidden - reset_hidden) > 0
    np.testing.assert_array_equal(genome, identity_genome())


def test_mutating_output_weights_activates_preexisting_context_representation():
    parts = unpack_genome(identity_genome())
    assert np.count_nonzero(parts["W_in"]) == 512
    assert np.count_nonzero(parts["W_rec"]) == 64
    # A single output weight can expose an existing context feature; no input
    # mutation or task-specific branch is needed to escape identity decisions.
    parts["W_out"][1, 0] = 2.0
    genome = pack_genome(parts)
    direction = parts["W_in"][0] / np.linalg.norm(parts["W_in"][0])
    positive, _ = forward(genome, direction * 3)
    negative, _ = forward(genome, -direction * 3)
    assert decode(positive)[0] == 1
    assert decode(negative)[0] == 0


def test_recurrence_changes_decisions_and_memory_is_explicit():
    parts = {name: np.zeros(shape) for name, shape in DEFAULT_SPEC.shapes.items()}
    parts["W_in"][0, 0] = 1
    parts["W_rec"][0, 0] = 1
    parts["W_out"][1, 0] = 2
    parts["b_out"][0] = 0.25
    genome = pack_genome(parts)
    _, memory = forward(genome, np.eye(64)[0])
    continued, _ = forward(genome, np.zeros(64), memory)
    reset, _ = forward(genome, np.zeros(64))
    assert decode(continued)[0] == 1
    assert decode(reset)[0] == 0


def test_random_initialization_can_make_context_dependent_choices():
    rng = np.random.default_rng(223)
    features = rng.normal(size=(32, 64))
    genomes = [random_genome(rng) for _ in range(7)]
    diversity = [len({tuple(decode(forward(g, x)[0])) for x in features}) for g in genomes]
    assert min(diversity) > 1


def test_sampling_log_probability_and_tie_break_are_consistent():
    logits = np.tile([0.25, 0.0, 0.0], (4, 1))
    first, second = np.random.default_rng(4), np.random.default_rng(4)
    samples = [decode(logits, first) for _ in range(200)]
    np.testing.assert_array_equal(samples, [decode(logits, second) for _ in range(200)])
    assert len({tuple(x) for x in samples}) > 10
    expected = 4 * (0.25 - np.log(np.exp(0.25) + 2))
    assert log_prob(logits, np.zeros(4, dtype=int)) == pytest.approx(expected)
    np.testing.assert_array_equal(decode(np.zeros((4, 3))), np.zeros(4, dtype=int))
    large = logits + 1000
    assert log_prob(large, np.zeros(4, dtype=int)) == pytest.approx(expected)


@pytest.mark.parametrize("field,value", [
    ("features", np.zeros(63)), ("features", np.full(64, np.nan)),
    ("features", np.ones(64, dtype=complex)), ("features", np.array(["0"] * 64)),
    ("hidden", np.zeros(7)), ("hidden", np.full(8, np.inf)),
    ("genome", np.zeros(691)), ("genome", np.full(692, np.nan)),
])
def test_numpy_rejects_invalid_inputs(field, value):
    kwargs = {"genome": identity_genome(), "features": np.zeros(64), "hidden": np.zeros(8)}
    kwargs[field] = value
    with pytest.raises(ValueError):
        forward(**kwargs)


def test_invalid_choices_specs_and_rng_are_rejected():
    with pytest.raises(ValueError):
        ModelSpec(input_dim=True)
    with pytest.raises(ValueError):
        ModelSpec(num_heads=3)
    for choice in (np.zeros(4), np.full(4, 3), np.zeros(3, dtype=int)):
        with pytest.raises(ValueError):
            log_prob(np.zeros((4, 3)), choice)
    with pytest.raises(TypeError):
        decode(np.zeros((4, 3)), rng=8)
    with pytest.raises(ValueError):
        identity_genome(bias=0)


def test_checkpoint_roundtrip_is_exact_and_hash_covers_metadata_and_weights(tmp_path):
    genome = random_genome(np.random.default_rng(11))
    target = tmp_path / "actor.npz"
    checksum = save_model(target, genome, {"seed": 11, "scope": "fixture"})
    restored, spec, metadata = load_model(target)
    np.testing.assert_array_equal(restored, genome)
    assert spec == DEFAULT_SPEC
    assert metadata == {"seed": 11, "scope": "fixture"}
    assert len(checksum) == 64
    assert genome_sha256(restored) == genome_sha256(genome)
    with pytest.raises(FileExistsError):
        save_model(target, genome)
    with np.load(target, allow_pickle=False) as data:
        fields = {name: data[name] for name in data.files}
    fields["genome"] = fields["genome"].copy()
    fields["genome"][0] += 0.01
    tampered = tmp_path / "tampered.npz"
    np.savez(tampered, **fields)
    with pytest.raises(ValueError, match="SHA256"):
        load_model(tampered)
    fields["genome"] = genome
    fields["manifest_json"] = np.asarray(str(fields["manifest_json"]).replace("fixture", "other"))
    np.savez(tampered, **fields)
    with pytest.raises(ValueError, match="SHA256"):
        load_model(tampered)


def test_checkpoint_rejects_schema_drift_and_nonfinite_metadata(tmp_path):
    target = tmp_path / "actor.npz"
    with pytest.raises(ValueError):
        save_model(target, identity_genome(), {"bad": float("nan")})
    assert not target.exists()
    save_model(target, identity_genome())
    with np.load(target, allow_pickle=False) as data:
        fields = {name: data[name] for name in data.files}
    fields["manifest_json"] = np.asarray(str(fields["manifest_json"]).replace("/v1", "/v2"))
    changed = tmp_path / "changed.npz"
    np.savez(changed, **fields)
    with pytest.raises(ValueError, match="schema"):
        load_model(changed)


def test_torch_numpy_parity_and_reset_mask_across_independent_episodes():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(91)
    genome = random_genome(rng)
    actor = make_torch_actor(genome, dtype=torch.float64)
    features = rng.normal(size=(2, 7, 64))
    resets = np.zeros((2, 7), dtype=bool)
    resets[0, 3] = True
    resets[1, 5] = True
    actual, final = actor.forward_sequence(features, reset_mask=resets)
    for batch in range(2):
        hidden = None
        for step in range(7):
            hidden = None if resets[batch, step] else hidden
            expected, hidden = forward(genome, features[batch, step], hidden)
            np.testing.assert_allclose(actual[batch, step].detach().numpy(), expected, atol=1e-12)
        np.testing.assert_allclose(final[batch].detach().numpy(), hidden, atol=1e-12)
    one, _ = actor.forward_sequence(features[0], reset_mask=resets[0])
    np.testing.assert_allclose(one.detach().numpy(), actual[0].detach().numpy(), atol=1e-12)
    np.testing.assert_array_equal(actor.export_genome(), genome)
    assert tuple(dict(actor.named_parameters())) == tuple(DEFAULT_SPEC.shapes)


def test_torch_sequence_retains_recurrent_gradients_and_categorical_likelihood():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(32)
    actor = make_torch_actor(random_genome(rng), dtype=torch.float64)
    features = torch.tensor(rng.normal(size=(5, 64)), dtype=torch.float64, requires_grad=True)
    logits, _ = actor.forward_sequence(features)
    choices = torch.tensor([[0, 1, 2, 0]] * 5)
    likelihood = torch_log_prob(logits, choices)
    for step in range(5):
        assert likelihood[step].item() == pytest.approx(log_prob(
            logits[step].detach().numpy(), choices[step].numpy(),
        ))
    assert bool((torch_entropy(logits) > 0).all())
    (-likelihood[-1]).backward()
    assert features.grad[0].abs().sum().item() > 0
    assert actor.W_rec.grad.abs().sum().item() > 0


def test_torch_identity_and_shape_checks():
    torch = pytest.importorskip("torch")
    actor = make_torch_actor(identity_genome())
    logits, hidden = actor(np.ones(64))
    assert tuple(hidden.shape) == (8,)
    np.testing.assert_array_equal(logits.argmax(-1).detach().numpy(), np.zeros(4))
    assert tuple(actor.initial_state(2).shape) == (2, 8)
    for features in (np.zeros((2, 63)), np.full((2, 64), np.nan)):
        with pytest.raises(ValueError):
            actor(features)
    with pytest.raises(ValueError):
        actor(np.zeros((2, 64)), torch.zeros(8))
    with pytest.raises(ValueError):
        actor.forward_sequence(np.zeros((2, 64)), reset_mask=[0, 1])
    with pytest.raises(ValueError):
        actor.forward_sequence(np.zeros((0, 64)))


def test_runtime_hash_alias_matches_canonical_serialized_actor_identity(tmp_path):
    genome = random_genome(np.random.default_rng(91))
    target = tmp_path / "actor.npz"
    save_model(target, genome, {"method": "test fixture"})
    restored, spec, _ = load_model(target)
    assert genome_hash(restored, spec) == genome_sha256(genome, DEFAULT_SPEC)
    different_schema = ModelSpec(input_dim=63, hidden_dim=8)
    assert different_schema.parameter_count != DEFAULT_SPEC.parameter_count
    with pytest.raises(ValueError):
        genome_hash(genome, different_schema)
