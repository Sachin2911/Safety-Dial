"""Shared causal features without simulator goal, task-family or safety-label inputs."""
import hashlib
import importlib
import importlib.metadata
import json
import re
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ROBOT_FEATURES = 40
FEATURE_DIM = 64
RUNTIME_MODULES = {
    "numpy": "numpy", "torch": "torch", "torchvision": "torchvision", "Pillow": "PIL",
    "open_clip_torch": "open_clip", "ftfy": "ftfy", "timm": "timm",
    "safetensors": "safetensors",
}


def loaded_package_provenance(distribution_name, module_name, expected_version):
    """Validate the loaded module and its own wheel metadata, not sys.path precedence.

    Isaac adds vendored distribution metadata after NumPy has already been imported.
    A global metadata lookup can then describe an unused version. Check the module's
    version and bind its origin to the adjacent distribution RECORD; retain the global
    lookup separately as diagnostic evidence. This supports the installed wheel layout
    used by the frozen encoder, not editable or namespace-package installations.
    """
    module = importlib.import_module(module_name)
    loaded_version = getattr(module, "__version__", None)
    if not isinstance(loaded_version, str) or loaded_version != expected_version:
        raise ValueError(f"Encoder loaded runtime version mismatch: {distribution_name}; "
                         f"expected={expected_version!r}, loaded={loaded_version!r}, "
                         f"module={getattr(module, '__file__', None)!r}")
    origin_value = getattr(module, "__file__", None)
    if not isinstance(origin_value, str):
        raise TypeError(f"Encoder module origin is unavailable: {distribution_name}")
    origin = Path(origin_value).resolve(strict=True)
    site = origin.parent.parent
    relative = origin.relative_to(site).as_posix()
    canonical = re.sub(r"[-_.]+", "-", distribution_name).lower()
    owners = []
    for distribution in importlib.metadata.distributions(path=[str(site)]):
        name = distribution.metadata.get("Name", "")
        if re.sub(r"[-_.]+", "-", name).lower() != canonical:
            continue
        files = distribution.files or []
        if any(entry.as_posix() == relative for entry in files):
            owners.append(distribution)
    if len(owners) != 1:
        raise ValueError(f"Encoder module lacks one owning distribution RECORD: "
                         f"{distribution_name} at {origin}")
    owner = owners[0]
    if owner.version != expected_version:
        raise ValueError(f"Encoder owning distribution version mismatch: {distribution_name}; "
                         f"expected={expected_version!r}, metadata={owner.version!r}, "
                         f"module={str(origin)!r}, distribution_root={str(site)!r}")
    metadata_entries = [entry for entry in owner.files if entry.name == "METADATA"
                        and entry.parent.name.endswith(".dist-info")]
    if len(metadata_entries) != 1:
        raise ValueError(f"Encoder owning distribution metadata is ambiguous: {distribution_name}")
    metadata_path = Path(owner.locate_file(metadata_entries[0])).resolve(strict=True)
    try:
        global_version = importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        global_version = None
    return {
        "module": module_name, "loaded_version": str(loaded_version),
        "module_file": str(origin), "module_sha256": hashlib.sha256(origin.read_bytes()).hexdigest(),
        "distribution_version": owner.version, "distribution_metadata_file": str(metadata_path),
        "distribution_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "global_metadata_version": global_version,
        "global_metadata_shadowed": global_version != owner.version,
    }


def encoder_runtime_provenance(spec):
    """Check every frozen encoder dependency against the modules actually in use."""
    versions = spec.get("runtime_versions")
    if not isinstance(versions, dict) or set(versions) != set(RUNTIME_MODULES):
        raise ValueError("Encoder runtime contract must pin every declared dependency")
    if (spec.get("implementation") != "open_clip_torch"
            or spec.get("implementation_version") != versions["open_clip_torch"]):
        raise ValueError("Encoder implementation and runtime version pins disagree")
    return {name: loaded_package_provenance(name, RUNTIME_MODULES[name], version)
            for name, version in versions.items()}


def finite_vector(value, shape, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real numeric array")
    result = raw.astype(np.float32)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def projection_matrices(seed=20261017):
    rng = np.random.default_rng(seed)
    return tuple(np.linalg.qr(rng.normal(size=(512, 8)))[0].astype(np.float32)
                 for _ in range(2))


def project(embedding, matrix):
    embedding = finite_vector(embedding, (512,), "CLIP embedding")
    norm = float(np.linalg.norm(embedding))
    if norm < 1e-8:
        raise ValueError("Zero CLIP embedding")
    return np.clip((embedding / norm) @ matrix * np.sqrt(512.0), -5, 5) / 5


def robot_features(frame, proposal, *, last_executed=None, last_proposal=None,
                   last_eef=None, since_query=0, opening_steps=0, choices=(0, 0, 0, 0)):
    frame.validate()
    if frame.prerequisite is not None or frame.target_engaged is not None:
        raise ValueError("Privileged labels cannot enter phase-two runtime features")
    choice_array = np.asarray(choices)
    if (choice_array.shape != (4,) or choice_array.dtype.kind not in "iu"
            or np.any(choice_array < 0) or np.any(choice_array > 2)):
        raise ValueError("Previous choices must be four integer indices in [0, 2]")
    for name, value in (("since_query", since_query), ("opening_steps", opening_steps)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    proposal = finite_vector(proposal, (8,), "proposal")
    joints = np.asarray(frame.joints)
    previous = proposal if last_proposal is None else finite_vector(
        last_proposal, (8,), "previous proposal")
    executed = np.r_[joints, proposal[-1]] if last_executed is None else finite_vector(
        last_executed, (8,), "previous environment input")
    previous_eef = frame.eef_position if last_eef is None else finite_vector(
        last_eef, (3,), "previous eef position")
    feasible = proposal[:7]
    if frame.joint_lower_limits is not None:
        feasible = np.where(frame.joint_has_limits, np.clip(
            feasible, frame.joint_lower_limits, frame.joint_upper_limits), feasible)
    result = np.r_[
        joints / np.pi, np.asarray(frame.fingers) / .04,
        (np.asarray(frame.eef_position) - previous_eef) / .05,
        (executed[:7] - joints) / .2, (feasible - joints) / .2,
        (proposal[:7] - previous[:7]) / .2,
        1.0 if proposal[-1] >= 0 else -1.0,
        1.0 if executed[-1] >= 0 else -1.0,
        since_query / 16.0, min(opening_steps, 4) / 4.0,
        choices[0] / 2.0, choices[1] / 2.0, choices[2] / 2.0,
    ]
    return np.clip(finite_vector(result, (ROBOT_FEATURES,), "robot features"), -1, 1)


def combine_features(robot, overview, wrist, text):
    result = np.concatenate([
        finite_vector(robot, (40,), "robot features"),
        finite_vector(overview, (8,), "overview features"),
        finite_vector(wrist, (8,), "wrist features"),
        finite_vector(text, (8,), "text features"),
    ])
    return finite_vector(result, (FEATURE_DIM,), "combined features")


class FrozenContextEncoder:
    """Both modalities share frozen CLIP; projections never fit validation/test data."""
    def __init__(self, manifest=None, device=None):
        import open_clip
        import torch

        self.torch = torch
        path = ROOT / "configs/maniguard/phase2/encoder.json" if manifest is None else Path(manifest)
        self.spec = json.loads(path.read_text())
        if self.spec["schema"] != "maniguard-phase2-encoder-v1":
            raise ValueError("Unknown encoder contract")
        self.runtime_provenance = encoder_runtime_provenance(self.spec)
        checkpoint = ROOT / self.spec["checkpoint"]
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if digest != self.spec["checkpoint_sha256"]:
            raise ValueError("Encoder checkpoint checksum mismatch")
        self.device = device or self.spec["device"]
        from open_clip.openai import load_openai_model
        from open_clip.transform import image_transform
        self.model = load_openai_model(str(checkpoint), device=self.device, precision="fp32")
        self.preprocess = image_transform(
            self.model.visual.image_size, is_train=False,
            mean=self.model.visual.image_mean, std=self.model.visual.image_std,
            interpolation="bicubic", resize_mode="shortest",
        )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer(self.spec["architecture"])
        self.visual_projection, self.text_projection = projection_matrices(
            self.spec["projection_seed"])
        actual_projections = [hashlib.sha256(x.astype("<f4").tobytes()).hexdigest()
                              for x in (self.visual_projection, self.text_projection)]
        if actual_projections != self.spec["projection_sha256"]:
            raise ValueError("Fixed feature projection checksum mismatch")
        self.image_calls = 0
        self.image_count = 0
        self.text_calls = 0
        self.image_attempts = 0
        self.text_attempts = 0
        self.elapsed = 0.0
        self.text_features = None
        self.text_hash = None

    def _sync(self):
        if str(self.device).startswith("cuda"):
            self.torch.cuda.synchronize()

    def begin_episode(self, prompt):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("The actual task instruction is required")
        started = time.perf_counter()
        self.text_attempts += 1
        try:
            self._sync()
            with self.torch.inference_mode():
                embedding = self.model.encode_text(self.tokenizer([prompt]).to(self.device))[0]
            self._sync()
            self.text_features = project(embedding.float().cpu().numpy(), self.text_projection)
            self.text_hash = hashlib.sha256(prompt.encode()).hexdigest()
            self.text_calls += 1
            return self.text_features.copy()
        finally:
            self.elapsed += time.perf_counter() - started

    def encode_images(self, overview, wrist):
        from PIL import Image

        if self.text_features is None:
            raise RuntimeError("Begin the episode before encoding images")
        started = time.perf_counter()
        self.image_attempts += 1
        try:
            self._sync()
            images, hashes = [], []
            for value in (overview, wrist):
                value = np.asarray(value)
                if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8:
                    raise ValueError("Encoder requires original uint8 RGB observations")
                hashes.append(hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())
                images.append(self.preprocess(Image.fromarray(value)))
            with self.torch.inference_mode():
                values = self.model.encode_image(self.torch.stack(images).to(self.device))
            self._sync()
            projected = [project(x.float().cpu().numpy(), self.visual_projection) for x in values]
            self.image_calls += 1
            self.image_count += 2
            return projected[0], projected[1], {
                "image_hashes": hashes, "text_sha256": self.text_hash,
                "checkpoint_sha256": self.spec["checkpoint_sha256"],
            }
        finally:
            self.elapsed += time.perf_counter() - started

    def ledger(self):
        return {"encoder_image_calls": self.image_calls, "encoder_images": self.image_count,
                "encoder_text_calls": self.text_calls, "encoder_seconds": self.elapsed,
                "encoder_image_attempts": self.image_attempts,
                "encoder_text_attempts": self.text_attempts}
