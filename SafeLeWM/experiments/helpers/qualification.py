"""TwoRoom qualification components. Execute only on the authorised rented host."""

import collections
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def segment_distance(start, end, centre):
    start, end, centre = map(lambda x: np.asarray(x, dtype=float), (start, end, centre))
    delta = end - start
    t = np.clip(np.dot(centre - start, delta) / max(np.dot(delta, delta), 1e-20), 0, 1)
    return float(np.linalg.norm(start + t * delta - centre))


def first_event_costs(hits):
    seen = False
    costs = []
    for hit in hits:
        costs.append(int(hit and not seen))
        seen |= bool(hit)
    return costs


def load_model(weights):
    from stable_worldmodel.wm.lewm import LeWM
    from stable_worldmodel.wm.lewm.module import Embedder, MLP, Predictor
    from transformers import ViTConfig, ViTModel

    expected = "566f223624ea4bfb39dbfe6ae731198dd6ea73b7b8919fed6b1ecafca810f7dd"
    assert hashlib.sha256(Path(weights).read_bytes()).hexdigest() == expected
    encoder = ViTModel(
        ViTConfig(hidden_size=192, num_hidden_layers=12, num_attention_heads=3,
                  intermediate_size=768, image_size=224, patch_size=14),
        add_pooling_layer=False, use_mask_token=False,
    )
    model = LeWM(
        encoder=encoder,
        predictor=Predictor(num_frames=3, input_dim=192, hidden_dim=192, output_dim=192,
                            depth=6, heads=16, mlp_dim=2048, dim_head=64,
                            dropout=0.1, emb_dropout=0.0),
        action_encoder=Embedder(input_dim=10, emb_dim=192),
        projector=MLP(192, 2048, 192, norm_fn=torch.nn.BatchNorm1d),
        pred_proj=MLP(192, 2048, 192, norm_fn=torch.nn.BatchNorm1d),
    )
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    model = model.cuda().eval()
    model.requires_grad_(False)
    return model


class Planner:
    """Hazard-blind, cold-start CEM over normalized five-action blocks."""

    def __init__(self, model, scalers, seed, counters):
        self.model, self.counters = model, counters
        spec = json.loads(Path(scalers).read_text())["columns"]["action"]
        self.mean = torch.tensor(spec["mean"], device="cuda", dtype=torch.float32)
        self.scale = torch.tensor(spec["scale"], device="cuda", dtype=torch.float32)
        self.low = ((-1 - self.mean) / self.scale).repeat(5)
        self.high = ((1 - self.mean) / self.scale).repeat(5)
        self.rng = torch.Generator(device="cuda").manual_seed(seed)

    @torch.inference_mode()
    def encode(self, image):
        self.counters["encoder_frames"] += 1
        pixels = torch.tensor(np.array(image), device="cuda").permute(2, 0, 1).float() / 255
        mean = pixels.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        std = pixels.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        pixels = ((pixels - mean) / std)[None, None]
        return pixels, self.model.encode({"pixels": pixels})["emb"]

    @torch.inference_mode()
    def plan(self, image, target_image):
        pixels, current = self.encode(image)
        goal_pixels, goal = self.encode(target_image)
        mean = torch.zeros(1, 3, 10, device="cuda")
        std = torch.ones_like(mean)
        for _ in range(8):
            candidates = mean[:, None] + std[:, None] * torch.randn(
                1, 300, 3, 10, device="cuda", generator=self.rng
            )
            candidates = candidates.clamp(self.low, self.high)
            info = {
                "pixels": pixels[:, None].expand(1, 300, -1, -1, -1, -1),
                "goal": goal_pixels[:, None].expand(1, 300, -1, -1, -1, -1),
                "emb": current[:, None].expand(1, 300, -1, -1),
                "goal_emb": goal,
            }
            self.counters["model_batch_calls"] += 1
            self.counters["model_candidate_rollouts"] += 300
            self.counters["candidate_model_steps"] += 900
            cost = self.model.get_cost(info, candidates)
            if not torch.isfinite(cost).all():
                raise ValueError("Nonfinite planner cost")
            elites = candidates[0, cost[0].topk(30, largest=False).indices]
            mean = elites.mean(0, keepdim=True)
            std = elites.std(0, unbiased=False, keepdim=True).clamp_min(0.05)
        normalized = mean[0, 0].reshape(5, 2)
        proposed = normalized * self.scale + self.mean
        executed = proposed.clamp(-1, 1)
        self.counters["replans"] += 1
        return proposed.cpu().numpy(), executed.cpu().numpy()


def new_counters():
    return collections.defaultdict(int)
