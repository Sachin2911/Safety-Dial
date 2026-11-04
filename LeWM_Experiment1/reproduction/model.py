"""Load the original JEPA code and released weights; no AP model or planner."""

import hashlib
import importlib.util
import json
from pathlib import Path

import torch
from transformers import ViTConfig, ViTModel


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_original(root, task, spec):
    root = Path(root)
    source = root / "official-source"
    jepa = import_file("original_jepa", source / "jepa.py")
    module = import_file("original_module", source / "module.py")
    directory = root / "models" / task
    cfg = json.loads((directory / "config.json").read_text())
    weights = directory / "weights.pt"
    with weights.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != spec["weights_sha256"]:
            raise ValueError("Checkpoint checksum mismatch")
    if cfg["encoder"]["size"] != "tiny":
        raise ValueError("Unexpected encoder architecture")
    # The stable-pretraining ViT-tiny factory uses the standard HF architecture.
    # Constructing it directly avoids installing training-only dependencies.
    encoder = ViTModel(
        ViTConfig(hidden_size=192, num_hidden_layers=12, num_attention_heads=3,
                  intermediate_size=768, image_size=cfg["encoder"]["image_size"],
                  patch_size=cfg["encoder"]["patch_size"]),
        add_pooling_layer=False, use_mask_token=False,
    )
    encoder.config.interpolate_pos_encoding = True

    def clean(key):
        return {k: v for k, v in cfg[key].items() if not k.startswith("_")}

    def mlp(key):
        return module.MLP(input_dim=cfg[key]["input_dim"],
                          hidden_dim=cfg[key]["hidden_dim"],
                          output_dim=cfg[key]["output_dim"], norm_fn=torch.nn.BatchNorm1d)

    model = jepa.JEPA(encoder=encoder, predictor=module.ARPredictor(**clean("predictor")),
                      action_encoder=module.Embedder(**clean("action_encoder")),
                      projector=mlp("projector"), pred_proj=mlp("pred_proj"))
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.cuda().eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    return model, {"model_revision": spec["model_revision"],
                   "weights_sha256": spec["weights_sha256"],
                   "strict_load": True, "parameters": sum(p.numel() for p in model.parameters()),
                   "checkpoint_config": cfg, "encoder_config": encoder.config.to_dict()}


class CountedCost(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.base = model
        self.counts = {"cost_calls": 0, "candidate_rollouts": 0,
                       "predicted_latent_steps": 0, "encoder_frames": 0}
        model.encoder.register_forward_pre_hook(self._encoder)
        model.predictor.register_forward_pre_hook(self._predictor)

    def _encoder(self, module, args):
        self.counts["encoder_frames"] += int(args[0].shape[0])

    def _predictor(self, module, args):
        self.counts["predicted_latent_steps"] += int(args[0].shape[0])

    def get_cost(self, info, candidates):
        self.counts["cost_calls"] += 1
        self.counts["candidate_rollouts"] += int(candidates.shape[0] * candidates.shape[1])
        costs = self.base.get_cost(info, candidates)
        if not torch.isfinite(costs).all():
            raise ValueError("Nonfinite CEM cost")
        return costs
