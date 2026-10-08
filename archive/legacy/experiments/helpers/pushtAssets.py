"""Push-T assets: the released model, fitted normalisers and the image transform.

Everything that the historical planner (`experiments/scripts/safe_dial_pusht.py`) built
inline, rebuilt once on the fresh instance and stored with the Hugging Face asset
bundle ([infrastructure.md]). Old files are imported or copied, never edited.

    from helpers.pushtAssets import load_model, build_scalers, save_scalers, load_scalers
    model = load_model(device)                     # AutoCostModel("pusht/lewm"), once
    process = build_scalers(H5_PATH)               # dict of fitted StandardScalers
    save_scalers(process, run_dir / "scalers.npz")
    process = load_scalers(run_dir / "scalers.npz")

`load_model` assigns STABLEWM_HOME (never setdefault) to `data/stablewm` unless the
caller already exported it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
STABLEWM_HOME = REPO_ROOT / "data" / "stablewm"
H5_PATH = REPO_ROOT / "data" / "processed" / "pusht_expert_train.h5"
LEWM_DIR = REPO_ROOT / "third_party" / "le-wm"
PUBLIC_MODEL_REPO = "quentinll/lewm-pusht"
SCALER_COLS = ("action", "proprio", "state")

# Checked against the installed stable_worldmodel 0.1.1 Push-T env (pushT.md, "Pinned facts").
CONTROL_HZ = 10
PHYSICS_DT = 0.01
SUBSTEPS_PER_STEP = 10
ACTION_BLOCK = 5  # env steps per model step (frameskip)
HORIZON_BLOCKS = 5
HISTORY_FRAMES = 3
ARENA = (0.0, 512.0)
ACTION_SCALE = 100.0


def ensure_paths() -> None:
    """STABLEWM_HOME for the checkpoint loader and le-wm on sys.path for its utils."""
    home = os.environ.get("STABLEWM_HOME")
    if not home:
        os.environ["STABLEWM_HOME"] = str(STABLEWM_HOME)
    if str(LEWM_DIR) not in sys.path and LEWM_DIR.is_dir():
        sys.path.insert(0, str(LEWM_DIR))


def load_model(device: str = "cuda"):
    """The released Push-T LeWM, in eval mode. Load ONCE outside any episode loop."""
    ensure_paths()
    import stable_worldmodel as swm

    model = swm.policy.AutoCostModel("pusht/lewm").to(device).eval()
    return model


def build_scalers(h5_path: Path = H5_PATH) -> dict:
    """Fitted StandardScalers on the full expert columns, plus goal_ duplicates.

    Copied from `experiments/scripts/safe_dial_pusht.py::build_process` (read-only
    historical file) so the normalisation matches the pilot exactly.
    """
    ensure_paths()
    import stable_worldmodel as swm
    from sklearn import preprocessing

    dataset = swm.data.HDF5Dataset(path=str(h5_path), keys_to_cache=list(SCALER_COLS))
    process = {}
    for col in SCALER_COLS:
        proc = preprocessing.StandardScaler()
        data = dataset.get_col_data(col)
        data = data[~np.isnan(data).any(axis=1)]
        proc.fit(data)
        process[col] = proc
        if col != "action":
            process[f"goal_{col}"] = proc
    return process


def save_scalers(process: dict, path: Path) -> Path:
    blobs = {}
    for col in SCALER_COLS:
        sc = process[col]
        blobs[f"{col}_mean"] = np.asarray(sc.mean_, dtype=np.float64)
        blobs[f"{col}_scale"] = np.asarray(sc.scale_, dtype=np.float64)
        blobs[f"{col}_var"] = np.asarray(sc.var_, dtype=np.float64)
        blobs[f"{col}_n"] = np.asarray(sc.n_samples_seen_, dtype=np.int64)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **blobs)
    return path


def load_scalers(path: Path) -> dict:
    """Rebuild the `process` dict (with goal_ duplicates) from `scalers.npz`."""
    from sklearn import preprocessing

    z = np.load(path)
    process = {}
    for col in SCALER_COLS:
        sc = preprocessing.StandardScaler()
        sc.mean_ = z[f"{col}_mean"]
        sc.scale_ = z[f"{col}_scale"]
        sc.var_ = z[f"{col}_var"]
        sc.n_samples_seen_ = int(z[f"{col}_n"])
        sc.n_features_in_ = int(sc.mean_.shape[0])
        process[col] = sc
        if col != "action":
            process[f"goal_{col}"] = sc
    return process


def scaler_fingerprint(process: dict) -> dict:
    return {
        col: {
            "mean": np.asarray(process[col].mean_).round(6).tolist(),
            "scale": np.asarray(process[col].scale_).round(6).tolist(),
        }
        for col in SCALER_COLS
    }


def make_transform():
    """The pixel/goal transform dict the historical planner used (le-wm preprocessor)."""
    ensure_paths()
    from utils import get_img_preprocessor  # le-wm

    from helpers.linProbeHelpers import wm_transform

    pixel_prep = get_img_preprocessor("pixels", "pixels", img_size=224)
    goal_prep = get_img_preprocessor("goal", "goal", img_size=224)
    return {
        "pixels": wm_transform(pixel_prep, "pixels"),
        "goal": wm_transform(goal_prep, "goal"),
    }


def normalise_actions(process: dict, actions: np.ndarray) -> np.ndarray:
    """Raw env actions (..., 2) -> the model's normalised action space."""
    shape = actions.shape
    flat = actions.reshape(-1, shape[-1]).astype(np.float64)
    return process["action"].transform(flat).reshape(shape).astype(np.float32)


def denormalise_actions(process: dict, actions: np.ndarray) -> np.ndarray:
    shape = actions.shape
    flat = actions.reshape(-1, shape[-1]).astype(np.float64)
    return process["action"].inverse_transform(flat).reshape(shape).astype(np.float32)
