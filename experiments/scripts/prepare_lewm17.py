"""Prepare only pinned checkpoints and preserved scalers; no historical rollout jobs."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
os.environ["STABLEWM_HOME"] = str(ROOT / "data/lewm17/stablewm")

from helpers.lewm16_protocol import digest, write_json

CONFIG = ROOT / "configs/lewm17"


def main():
    import torch
    from huggingface_hub import hf_hub_download
    from hydra.utils import instantiate

    records = {}
    for env in ("tworoom", "reacher"):
        spec = json.loads((CONFIG / f"{env}.json").read_text())
        pins = spec["pins"]
        folder = ROOT / f"data/lewm17/{env}"
        for filename in ("config.json", "weights.pt"):
            hf_hub_download(
                pins["model_repo"], filename, revision=pins["model_revision"], local_dir=str(folder)
            )
        assert digest(folder / "weights.pt") == pins["model_weights_sha256"]
        assert digest(CONFIG / f"{env}-scalers.json") == spec["scaler_sha256"]
        cfg = json.loads((folder / "config.json").read_text())
        model = instantiate(cfg)
        sd = torch.load(folder / "weights.pt", map_location="cpu", weights_only=False)
        remapped = {}
        for key, val in sd.items():
            key = key.replace("encoder.encoder.layer.", "encoder.layers.")
            for old, new in [
                ("attention.attention.query", "attention.q_proj"),
                ("attention.attention.key", "attention.k_proj"),
                ("attention.attention.value", "attention.v_proj"),
                ("attention.output.dense", "attention.o_proj"),
                ("intermediate.dense", "mlp.fc1"),
                (".output.dense", ".mlp.fc2"),
            ]:
                key = key.replace(old, new)
            remapped[key] = val
        try:
            model.load_state_dict(remapped, strict=True)
        except RuntimeError:
            model.load_state_dict(sd, strict=True)
        target = ROOT / f"data/lewm17/stablewm/checkpoints/{env}/lewm_object.ckpt"
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model, target)
        records[env] = {
            "checkpoint_sha256": digest(target),
            "parameters": sum(p.numel() for p in model.parameters()),
            "scalers_sha256": spec["scaler_sha256"],
            "source_weights_sha256": pins["model_weights_sha256"],
        }
        del model, sd, remapped
    write_json(ROOT / "outputs/lewm17/environment/prepared.json", records)
    print(json.dumps(records), flush=True)


if __name__ == "__main__":
    main()
