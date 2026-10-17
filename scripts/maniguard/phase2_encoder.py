"""Fetch the public immutable CLIP checkpoint and record its phase-two feature contract."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
URL = f"https://openaipublic.azureedge.net/clip/models/{SHA256}/ViT-B-32.pt"


def main():
    folder = ROOT / "data/maniguard/phase2/encoder"
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / "ViT-B-32.pt"
    if not destination.exists():
        temporary = folder / "ViT-B-32.download"
        with urllib.request.urlopen(URL, timeout=120) as source, temporary.open("wb") as target:
            while block := source.read(8 * 1024 * 1024):
                target.write(block)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != SHA256:
            raise ValueError("Downloaded encoder failed the publisher's SHA256")
        temporary.rename(destination)
    if hashlib.sha256(destination.read_bytes()).hexdigest() != SHA256:
        raise ValueError("Existing encoder differs from the declared pin")
    record = {
        "schema": "maniguard-phase2-encoder-v1", "architecture": "ViT-B-32",
        "implementation": "open_clip_torch", "implementation_version": "3.2.0",
        "source_url": URL, "checkpoint_sha256": SHA256,
        "checkpoint": "data/maniguard/phase2/encoder/ViT-B-32.pt",
        "projection_seed": 20261017, "embedding_dimension": 512,
        "overview_features": 8, "wrist_features": 8, "text_features": 8,
        "projection": "fixed Gaussian QR; sqrt(512) scale; clip[-5,5]/5",
        "input": "overview and wrist uint8 RGB; actual task instruction",
        "schedule": "images at every four-action decision; text once per episode",
        "training": "frozen weights and projections; no data fitting",
        "device": "cuda", "precision": "fp32",
    }
    record.update({'loader': 'open_clip.openai.load_openai_model; official TorchScript weights rebuilt as eager module', 'activation': 'QuickGELU', 'preprocess': 'bicubic shortest-side resize then center crop; OpenAI mean/std; RGB', 'projection_algorithm': 'NumPy float64 QR cast to float32', 'timing': 'wall time including preprocessing, device transfer and projection; synchronized GPU'})
    record.update({'projection_sha256': ['be89829839b7f6cf569d769d0138cc212f989d8ac04e5f3931cf8d8d6ac9a6c3', '7053045c89a23896de3aac41730e57201a77732941a4431b10ecbede3c07bead'], 'runtime_versions': {'numpy': '1.26.4', 'torch': '2.6.0+cu124', 'torchvision': '0.21.0+cu124', 'Pillow': '11.0.0', 'open_clip_torch': '3.2.0', 'ftfy': '6.3.1', 'timm': '1.0.30', 'safetensors': '0.8.0'}})
    manifest = ROOT / "configs/maniguard/phase2/encoder.json"
    text = json.dumps(record, indent=2) + "\n"
    if manifest.exists() and manifest.read_text() != text:
        raise ValueError("Refusing to replace a different encoder contract")
    manifest.write_text(text)
    print(json.dumps({"checkpoint_sha256": SHA256, "bytes": destination.stat().st_size}))


if __name__ == "__main__":
    main()
