"""Shared recurrent phase-two actor, independent of the simulator and task family.

The first option of every head is identity. Logits define four independent
categorical distributions; evaluation uses deterministic first-index argmax.
The runtime, not this module, implements the physical execution semantics.
Torch is imported only when an ordinary-learning actor is requested.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SCHEMA = "maniguard_phase2_model/v1"
HEAD_NAMES = ("query", "cap", "smoothing", "release")
HEAD_OPTIONS = (
    ("baseline8", "now", "continue16"),
    ("identity", "0.03", "0.06"),
    ("1.0", "0.75", "0.5"),
    ("0", "2", "4"),
)
PARAMETER_NAMES = ("W_in", "W_rec", "b_hidden", "W_out", "b_out")


@dataclass(frozen=True)
class ModelSpec:
    input_dim: int = 64
    hidden_dim: int = 8
    num_heads: int = 4
    num_choices: int = 3

    def __post_init__(self):
        for name in ("input_dim", "hidden_dim", "num_heads", "num_choices"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_heads != len(HEAD_NAMES) or self.num_choices != 3:
            raise ValueError("This schema requires four categorical heads with three choices")

    @property
    def shapes(self):
        h, d, o = self.hidden_dim, self.input_dim, self.num_heads * self.num_choices
        return {"W_in": (h, d), "W_rec": (h, h), "b_hidden": (h,),
                "W_out": (o, h), "b_out": (o,)}

    @property
    def parameter_count(self):
        return sum(int(np.prod(shape)) for shape in self.shapes.values())


DEFAULT_SPEC = ModelSpec()


def _finite_array(value, shape, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.shape != shape:
        raise ValueError(f"{name} must be a real numeric array with shape {shape}")
    array = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def validate_genome(genome, spec=DEFAULT_SPEC):
    return _finite_array(genome, (spec.parameter_count,), "genome")


def unpack_genome(genome, spec=DEFAULT_SPEC):
    """Return named reshaped views in the fixed serialization/optimizer order."""
    genome = validate_genome(genome, spec)
    parts, offset = {}, 0
    for name, shape in spec.shapes.items():
        size = int(np.prod(shape))
        parts[name] = genome[offset:offset + size].reshape(shape)
        offset += size
    return parts


def pack_genome(parts, spec=DEFAULT_SPEC):
    if set(parts) != set(PARAMETER_NAMES):
        raise ValueError("Parameter names differ from the fixed actor layout")
    return np.concatenate([
        _finite_array(parts[name], shape, name).ravel()
        for name, shape in spec.shapes.items()
    ])


def random_genome(rng, spec=DEFAULT_SPEC):
    """Fixed initialization distribution; a fresh generator controls every draw.

    Input scale is 1/sqrt(input width), recurrent/output scale is
    0.5/sqrt(hidden width), and bias noise has standard deviation 0.05.
    Output choice zero additionally receives 0.25, matching the identity bias.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an explicit numpy Generator")
    parts = {}
    for name, shape in spec.shapes.items():
        scale = (1 / np.sqrt(spec.input_dim) if name == "W_in" else
                 0.5 / np.sqrt(spec.hidden_dim) if name in ("W_rec", "W_out") else 0.05)
        parts[name] = rng.normal(0.0, scale, size=shape)
    parts["b_out"].reshape(spec.num_heads, spec.num_choices)[:, 0] += 0.25
    return pack_genome(parts, spec)


def identity_genome(seed=20261017, bias=0.25, spec=DEFAULT_SPEC):
    """Deterministically select identity while retaining a mutable representation.

    Finite identity logits intentionally permit stochastic PPO exploration.
    Zero output weights keep identity decisions independent of every input and
    recurrent state; nonzero input/recurrent weights avoid a dead representation.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not np.isscalar(bias) or not np.isfinite(bias) or bias <= 0:
        raise ValueError("Identity bias must be positive and finite")
    parts = unpack_genome(random_genome(np.random.default_rng(seed), spec), spec)
    parts["W_out"].fill(0.0)
    parts["b_out"].fill(0.0)
    parts["b_out"].reshape(spec.num_heads, spec.num_choices)[:, 0] = bias
    return pack_genome(parts, spec)


def forward(genome, features, hidden=None, spec=DEFAULT_SPEC):
    """One NumPy actor step, returning (heads x choices logits, next hidden)."""
    weights = unpack_genome(genome, spec)
    features = _finite_array(features, (spec.input_dim,), "features")
    hidden = (np.zeros(spec.hidden_dim) if hidden is None else
              _finite_array(hidden, (spec.hidden_dim,), "hidden"))
    with np.errstate(over="raise", invalid="raise"):
        try:
            next_hidden = np.tanh(weights["W_in"] @ features
                                  + weights["W_rec"] @ hidden + weights["b_hidden"])
            logits = weights["W_out"] @ next_hidden + weights["b_out"]
        except FloatingPointError as exc:
            raise ValueError("Actor arithmetic produced nonfinite values") from exc
    if not np.isfinite(logits).all():
        raise ValueError("Actor produced nonfinite logits")
    return logits.reshape(spec.num_heads, spec.num_choices), next_hidden


def _log_probabilities(logits, spec=DEFAULT_SPEC):
    logits = _finite_array(logits, (spec.num_heads, spec.num_choices), "logits")
    with np.errstate(over="raise", invalid="raise"):
        try:
            shifted = logits - logits.max(axis=-1, keepdims=True)
            return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        except FloatingPointError as exc:
            raise ValueError("Logits exceed the stable numeric range") from exc


def decode(logits, rng=None, spec=DEFAULT_SPEC):
    """Argmax evaluation, or independent categorical sampling with explicit RNG."""
    log_probabilities = _log_probabilities(logits, spec)
    if rng is None:
        return np.argmax(log_probabilities, axis=-1)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an explicit numpy Generator")
    return np.asarray([rng.choice(spec.num_choices, p=np.exp(row))
                       for row in log_probabilities], dtype=np.int64)


def log_prob(logits, choices, spec=DEFAULT_SPEC):
    """Joint log probability of the four categorical decisions."""
    choices = np.asarray(choices)
    if (choices.shape != (spec.num_heads,) or choices.dtype.kind not in "iu"
            or np.any(choices < 0) or np.any(choices >= spec.num_choices)):
        raise ValueError("choices must contain four in-range integer indices")
    return float(_log_probabilities(logits, spec)[np.arange(spec.num_heads), choices].sum())


def _manifest(spec, metadata):
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a JSON object")
    return {"schema": SCHEMA, "spec": asdict(spec), "parameter_names": list(PARAMETER_NAMES),
            "head_names": list(HEAD_NAMES), "head_options": [list(x) for x in HEAD_OPTIONS],
            "metadata": metadata}


def _checksum(manifest_json, genome):
    digest = hashlib.sha256(manifest_json.encode("utf-8"))
    digest.update(b"\0")
    digest.update(np.asarray(genome, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def genome_sha256(genome, spec=DEFAULT_SPEC):
    genome = validate_genome(genome, spec)
    manifest_json = json.dumps(_manifest(spec, {}), sort_keys=True, separators=(",", ":"))
    return _checksum(manifest_json, genome)


def genome_hash(genome, spec=DEFAULT_SPEC):
    """Compatibility name for the canonical schema-aware actor digest."""
    return genome_sha256(genome, spec)


def save_model(path, genome, metadata=None, spec=DEFAULT_SPEC):
    """Exclusively create an NPZ checkpoint; hash covers schema, metadata and weights."""
    genome = validate_genome(genome, spec)
    manifest_json = json.dumps(_manifest(spec, {} if metadata is None else metadata),
                               sort_keys=True, separators=(",", ":"), allow_nan=False)
    checksum = _checksum(manifest_json, genome)
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, genome=genome, manifest_json=np.asarray(manifest_json),
                            sha256=np.asarray(checksum))
    return checksum


def load_model(path):
    """Load only numeric/string arrays, validate the exact schema and payload hash."""
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"genome", "manifest_json", "sha256"}:
            raise ValueError("Unexpected checkpoint fields")
        text = payload["manifest_json"]
        digest = payload["sha256"]
        if text.shape != () or digest.shape != () or text.dtype.kind != "U" or digest.dtype.kind != "U":
            raise ValueError("Checkpoint manifest and hash must be scalar strings")
        manifest_json = str(text.item())
        manifest = json.loads(manifest_json, parse_constant=reject_constant)
        if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
            raise ValueError("Unknown checkpoint schema")
        try:
            spec = ModelSpec(**manifest["spec"])
        except (KeyError, TypeError) as exc:
            raise ValueError("Invalid checkpoint model specification") from exc
        if manifest != _manifest(spec, manifest.get("metadata")):
            raise ValueError("Checkpoint layout or decision domains differ from this schema")
        genome = validate_genome(payload["genome"], spec).copy()
        if _checksum(manifest_json, genome) != str(digest.item()):
            raise ValueError("Checkpoint SHA256 mismatch")
    return genome, spec, manifest["metadata"]


def make_torch_actor(genome, spec=DEFAULT_SPEC, device="cpu", dtype=None):
    """Create a differentiable actor with the exact NumPy weight layout.

    ``forward`` handles one step (features D or B,D). ``forward_sequence``
    handles T,D or B,T,D, returning all logits and the final hidden state.
    A boolean reset mask resets memory *before* its corresponding timestep.
    """
    import torch
    from torch import nn

    dtype = torch.float32 if dtype is None else dtype
    if not dtype.is_floating_point:
        raise ValueError("Actor dtype must be floating point")
    weights = unpack_genome(genome, spec)

    class TorchActor(nn.Module):
        def __init__(self):
            super().__init__()
            self.spec = spec
            self.input_dim = spec.input_dim
            self.hidden_dim = spec.hidden_dim
            for name in PARAMETER_NAMES:
                self.register_parameter(name, nn.Parameter(torch.tensor(
                    weights[name].copy(), dtype=dtype, device=device,
                )))

        def initial_state(self, batch_size=None, device=None):
            if batch_size is not None and (type(batch_size) is not int or batch_size <= 0):
                raise ValueError("batch_size must be a positive integer or None")
            shape = (self.hidden_dim,) if batch_size is None else (batch_size, self.hidden_dim)
            return torch.zeros(shape, dtype=self.W_in.dtype,
                               device=self.W_in.device if device is None else device)

        def _tensor(self, value, name):
            tensor = torch.as_tensor(value, device=self.W_in.device)
            if tensor.is_complex() or tensor.dtype == torch.bool:
                raise ValueError(f"{name} must contain real numeric values")
            tensor = tensor.to(dtype=self.W_in.dtype)
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} must contain only finite values")
            return tensor

        def forward(self, features, hidden=None):
            features = self._tensor(features, "features")
            if features.ndim not in (1, 2) or features.shape[-1] != self.input_dim:
                raise ValueError("Features must have shape D or B,D")
            if features.ndim == 2 and features.shape[0] == 0:
                raise ValueError("Feature batches must be nonempty")
            expected = (*features.shape[:-1], self.hidden_dim)
            hidden = (torch.zeros(expected, dtype=self.W_in.dtype, device=self.W_in.device)
                      if hidden is None else self._tensor(hidden, "hidden"))
            if tuple(hidden.shape) != expected:
                raise ValueError("Hidden shape does not match the feature batch")
            activation = features @ self.W_in.T + hidden @ self.W_rec.T + self.b_hidden
            if not bool(torch.isfinite(activation).all()):
                raise ValueError("Actor arithmetic produced nonfinite values")
            next_hidden = torch.tanh(activation)
            logits = next_hidden @ self.W_out.T + self.b_out
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("Actor produced nonfinite logits")
            return logits.reshape(*features.shape[:-1], spec.num_heads, spec.num_choices), next_hidden

        def forward_sequence(self, features, hidden=None, reset_mask=None):
            features = self._tensor(features, "features")
            if features.ndim not in (2, 3) or features.shape[-1] != self.input_dim:
                raise ValueError("Sequence features must have shape T,D or B,T,D")
            unbatched = features.ndim == 2
            features = features.unsqueeze(0) if unbatched else features
            batch, steps, _ = features.shape
            if not batch or not steps:
                raise ValueError("Feature sequences must be nonempty")
            if hidden is not None:
                hidden = self._tensor(hidden, "hidden")
                hidden = hidden.unsqueeze(0) if unbatched else hidden
            else:
                hidden = self.initial_state(batch)
            if tuple(hidden.shape) != (batch, self.hidden_dim):
                raise ValueError("Hidden shape does not match the sequence batch")
            if reset_mask is not None:
                reset_mask = torch.as_tensor(reset_mask, device=self.W_in.device)
                reset_mask = reset_mask.unsqueeze(0) if unbatched else reset_mask
                if reset_mask.dtype != torch.bool or tuple(reset_mask.shape) != (batch, steps):
                    raise ValueError("Reset mask must be boolean with shape T or B,T")
            outputs = []
            for step in range(steps):
                if reset_mask is not None:
                    hidden = torch.where(reset_mask[:, step, None], torch.zeros_like(hidden), hidden)
                logits, hidden = self(features[:, step], hidden)
                outputs.append(logits)
            outputs = torch.stack(outputs, dim=1)
            return (outputs[0], hidden[0]) if unbatched else (outputs, hidden)

        def export_genome(self):
            return pack_genome({name: getattr(self, name).detach().cpu().numpy()
                                for name in PARAMETER_NAMES}, self.spec)

    return TorchActor()


def torch_log_prob(logits, choices):
    """Differentiable joint categorical log probability over the last head axis."""
    import torch

    return torch.distributions.Categorical(logits=logits).log_prob(choices).sum(dim=-1)


def torch_entropy(logits):
    """Entropy of the product of the categorical head distributions."""
    import torch

    return torch.distributions.Categorical(logits=logits).entropy().sum(dim=-1)
