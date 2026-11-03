"""Released-model baseline and historical TwoRoom recipe, without safety filtering."""

import importlib.util
import json
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from stable_worldmodel.solver import CEMSolver

from helpers.qualification import Planner, load_model


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def models(root):
    """Load identical weights into both implementations; original positional slots retained."""
    current = load_model(root / "assets/weights.pt")
    ref = root / "assets/baseline-reference"
    jepa = import_file("baseline_jepa", ref / "jepa.py")
    mod = import_file("baseline_module", ref / "module.py")
    cfg = json.loads((ref / "model-config.json").read_text())

    def clean(name):
        return {k: v for k, v in cfg[name].items() if not k.startswith("_")}

    def mlp(name):
        return mod.MLP(
            cfg[name]["input_dim"],
            cfg[name]["hidden_dim"],
            cfg[name]["output_dim"],
            norm_fn=torch.nn.BatchNorm1d,
        )

    original = jepa.JEPA(
        encoder=current.encoder,
        predictor=mod.ARPredictor(**clean("predictor")),
        action_encoder=mod.Embedder(**clean("action_encoder")),
        projector=mlp("projector"),
        pred_proj=mlp("pred_proj"),
    )
    original.load_state_dict(current.state_dict(), strict=True)
    original.cuda().eval().requires_grad_(False)
    return current, original


def pixels(image):
    value = torch.as_tensor(np.array(image), device="cuda").permute(2, 0, 1).float() / 255
    mean = value.new_tensor([0.485, 0.456, 0.406])[:, None, None]
    std = value.new_tensor([0.229, 0.224, 0.225])[:, None, None]
    return ((value - mean) / std)[None, None]


class CountedCost(torch.nn.Module):
    def __init__(self, base, counters, scaler):
        super().__init__()
        self.base, self.counters = base, counters
        self.mean = torch.tensor(scaler["mean"], device="cuda")
        self.scale = torch.tensor(scaler["scale"], device="cuda")

    def get_cost(self, info, candidates):
        b, n, h, _ = candidates.shape
        self.counters["model_batch_calls"] += 1
        self.counters["model_candidate_rollouts"] += b * n
        self.counters["candidate_model_steps"] += b * n * h
        raw = candidates.reshape(b, n, h, 5, 2) * self.scale + self.mean
        self.counters["out_of_bounds_model_coordinates"] += int((raw.abs() > 1).sum())
        result = self.base.get_cost(info, candidates)
        if not torch.isfinite(result).all():
            raise ValueError("Nonfinite model cost")
        return result


class BaselinePlanner:
    def __init__(self, model, profile, scaler, seed, counters):
        self.profile, self.counters = profile, counters
        self.scaler = scaler
        self.model = CountedCost(model, counters, scaler)
        self.rng = torch.Generator(device="cuda").manual_seed(seed)
        self.memory, self.remaining = None, None
        if profile == "paper":
            self.solver = CEMSolver(
                self.model,
                batch_size=1,
                num_samples=300,
                n_steps=10,
                topk=30,
                var_scale=1,
                device="cuda",
                seed=seed,
            )
            self.solver.configure(
                action_space=gym.spaces.Box(-1, 1, (1, 2), dtype=np.float32),
                n_envs=1,
                config=SimpleNamespace(horizon=5, receding_horizon=5, action_block=5),
            )

    @torch.inference_mode()
    def plan(self, image, goal):
        info = {
            "pixels": pixels(image),
            "goal": pixels(goal),
            "action": torch.zeros(1, 1, 10, device="cuda"),
        }
        if self.profile == "paper":
            actions = self.solver.solve(info)["actions"][0].reshape(25, 2).cpu().numpy()
        else:
            actions = self.selected(info)
        self.counters["replans"] += 1
        raw = actions * np.asarray(self.scaler["scale"]) + np.asarray(self.scaler["mean"])
        return raw, np.clip(raw, -1, 1)

    def selected(self, info):
        """Task-only reduction of historical random-05 with empty hazard constraints."""
        h, n = 2, 150
        current = self.model.base.encode({"pixels": info["pixels"]})["emb"]
        goal = self.model.base.encode({"pixels": info["goal"]})["emb"]
        expanded = {k: v[:, None].expand(1, n, *v.shape[1:]) for k, v in info.items()}
        expanded.update(emb=current[:, None].expand(1, n, -1, -1), goal_emb=goal)
        mean = torch.zeros(1, h, 10, device="cuda")
        if self.remaining is not None:
            mean[:, :1] = self.remaining
        std = torch.full_like(mean, 2.0)
        memory = None
        if self.memory is not None:
            memory = torch.zeros_like(self.memory)
            memory[:, :, :1] = self.memory[:, :, 1:2]
        for _ in range(30):
            candidates = mean[:, None] + std[:, None] * torch.randn(
                1, n, h, 10, device="cuda", generator=self.rng
            )
            candidates[:, 0] = mean
            if memory is not None:
                count = min(memory.shape[1], int(n * 0.25), n - 1)
                candidates[:, 1 : count + 1] = memory[:, :count]
            costs = self.model.get_cost(expanded, candidates)
            elite = candidates[0, costs[0].topk(30, largest=False).indices]
            mean, std = elite.mean(0, keepdim=True), elite.std(0, keepdim=True)
            memory = candidates[:, costs[0].argsort()[: n // 2]].detach()
        self.memory = memory
        self.remaining = mean[:, 1:2].detach()
        return mean[0, 0].reshape(5, 2).cpu().numpy()


def make_planner(root, models_pair, profile, seed, counters):
    current, original = models_pair
    scaler = json.loads((root / "configs/tworoom-scalers.json").read_text())["columns"]["action"]
    if profile == "qualification":
        return Planner(current, root / "configs/tworoom-scalers.json", seed, counters)
    # Use original code for paper; validated cached equivalent for historical retention.
    return BaselinePlanner(
        original if profile == "paper" else current, profile, scaler, seed, counters
    )
