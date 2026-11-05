"""Small I/O helpers for the separately authorized, rented-host AP reproduction."""

from contextlib import contextmanager
import base64
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import zlib

from run_io import publish_artifact

AP_REVISION = "984327424ca7a2e52c75a8279197750d9d0b7c8e"
PROTOCOL_SHA256 = "3243f4ec7c55d349a3f7a73b093664ca7ae695d369f6f718824b54ea47b894b6"
TASKS = ("tworoom", "reacher", "pusht", "cube")
IMAGE_TAG = "__lewm_rgb_image__"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def encode_rgb_bytes(raw, shape):
    """Losslessly pack an RGB uint8 image without changing any environment object."""
    shape = list(shape)
    if (len(shape) != 3 or any(type(size) is not int or size <= 0 for size in shape)
            or shape[2] != 3 or not isinstance(raw, bytes) or len(raw) != math.prod(shape)):
        raise ValueError("Expected exact C-order uint8 RGB bytes and shape")
    return {IMAGE_TAG: 1, "encoding": "zlib-base64", "shape": shape, "dtype": "|u1",
            "order": "C", "byte_count": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "data": base64.b64encode(zlib.compress(raw, level=1)).decode("ascii")}


def decode_rgb_bytes(value, *, max_bytes=64 * 1024 * 1024):
    """Decode saved RGB bytes, checking shape, dtype, exact length and SHA-256.

    This standard-library decoder requires no model, simulator, or NumPy import.
    Dimensions remain in value['shape']; bytes are stored in C (HWC) order.
    """
    keys = {IMAGE_TAG, "encoding", "shape", "dtype", "order", "byte_count", "sha256", "data"}
    if not isinstance(value, dict) or set(value) != keys or type(value[IMAGE_TAG]) is not int:
        raise ValueError("Invalid saved RGB image record")
    if (value[IMAGE_TAG] != 1 or value["encoding"] != "zlib-base64" or value["dtype"] != "|u1"
            or value["order"] != "C"):
        raise ValueError("Unsupported saved RGB image encoding")
    shape = value["shape"]
    if (not isinstance(shape, list) or len(shape) != 3
            or any(type(size) is not int or size <= 0 for size in shape) or shape[2] != 3):
        raise ValueError("Invalid saved RGB image shape")
    expected = math.prod(shape)
    if (type(value["byte_count"]) is not int or value["byte_count"] != expected
            or not 0 < expected <= max_bytes):
        raise ValueError("Saved RGB image byte count differs or exceeds the decoder bound")
    encoded = value["data"]
    if not isinstance(encoded, str) or len(encoded) > 2 * max_bytes:
        raise ValueError("Saved RGB payload exceeds the decoder bound")
    compressed = base64.b64decode(encoded, validate=True)
    decoder = zlib.decompressobj()
    raw = decoder.decompress(compressed, expected + 1)
    if (len(raw) != expected or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail):
        raise ValueError("Saved RGB payload length or compressed stream differs")
    if hashlib.sha256(raw).hexdigest() != value["sha256"]:
        raise ValueError("Saved RGB image checksum differs")
    return raw


def serial(value):
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if (getattr(value, "ndim", None) == 3 and str(getattr(value, "dtype", "")) == "uint8"
            and tuple(value.shape)[-1] == 3):
        return encode_rgb_bytes(value.tobytes(order="C"), list(value.shape))
    if hasattr(value, "tolist"):
        return serial(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    if isinstance(value, Path):
        return str(value)
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(serial(value), stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def publish(relative, value):
    return publish_artifact(relative, data=json.dumps(serial(value), allow_nan=False) + "\n")


def configuration():
    config = read(os.environ["LEWM_CONFIG"])
    root = Path(config["asset_root"]).resolve()
    if not root.is_relative_to("/workspace/LeWM_Experiment1/assets"):
        raise ValueError("This entrypoint only runs on the rented experiment workspace")
    if config.get("ap_revision", AP_REVISION) != AP_REVISION:
        raise ValueError("AP source revision differs from the reviewed implementation")
    return config


def configure(config, task, *, require_overlay=True):
    """Load the exact native runtime; the Reacher overlay is installed separately."""
    overlay = config.get("reacher_overlay")
    if task == "reacher" and overlay:
        if not Path(overlay).is_dir():
            if require_overlay:
                raise ValueError("Missing frozen Reacher dependency overlay")
        else:
            sys.path.insert(0, str(overlay))
    root = Path(config.get("ap_root", "/opt/ap-reference/code/ap"))
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                     text=True).strip()
    if actual != AP_REVISION:
        raise ValueError("Unexpected AP checkout revision")
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain",
                                "--untracked-files=no"], text=True).strip():
        raise ValueError("Tracked AP source was modified")
    if sha(root / "protocols/main.json") != PROTOCOL_SHA256:
        raise ValueError("Native protocol checksum differs")
    sys.path.insert(0, str(root))
    from anchored_planning import runtime
    runtime.configure_runtime()
    return root, runtime


def assets(config, task):
    root = Path(config["asset_root"])
    return {"checkpoint": root / "models" / task / "weights.pt",
            "dataset": root / "datasets" / (task + ".h5"),
            "latent_cache": root / "latents" / (task + ".npy"),
            "state_key": "state" if task == "pusht" else "observation"}


class Ledger:
    """Reserve attempts before work. A scientific case can never be silently repeated.

    Reservation totals are conservative upper bounds, not measured interactions.
    Measured per-case accounting is stored independently, including failed attempts.
    A shared lock makes scientific sharding safe across processes on this host.
    """

    def __init__(self, config):
        self.path = Path(config["asset_root"]) / "attempt-ledger.json"
        self.limits = config.get("budgets", {})

    @contextmanager
    def locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            value = read(self.path) if self.path.exists() else {"attempts": {}, "reserved": {}}
            yield value
            write(self.path, value)

    def reserve(self, identity, phase, **amounts):
        with self.locked() as value:
            if identity in value["attempts"]:
                raise ValueError("An attempt already exists; do not rerun it: " + identity)
            for key, amount in amounts.items():
                name = phase + "_" + key
                total = value["reserved"].get(name, 0) + int(amount)
                if name not in self.limits:
                    raise ValueError("A finite budget is required: " + name)
                if total > self.limits[name]:
                    raise ValueError("Study reservation budget exhausted: " + name)
                value["reserved"][name] = total
            value["attempts"][identity] = {"phase": phase, "reserved": amounts,
                "run": Path(os.environ["LEWM_RUN_DIR"]).name, "started_epoch": time.time()}

    def finish(self, identity, counts, completed):
        with self.locked() as value:
            value["attempts"][identity].update(counts=serial(counts), completed=bool(completed),
                                               finished_epoch=time.time())


def check_deadline(config, started):
    limit = float(config.get("max_seconds", 18000))
    if not 0 < limit <= 21600:
        raise ValueError("A positive runtime of at most six hours is required")
    if time.monotonic() - started > limit:
        raise TimeoutError("Declared process runtime exceeded")


def provenance(config, ap_root):
    import importlib.metadata

    return {"config": config, "ap_revision": AP_REVISION,
            "source_hashes": {str(p.relative_to(ap_root)): sha(p)
                              for p in sorted(ap_root.rglob("*.py"))},
            "native_protocol_sha256": sha(ap_root / "protocols/main.json"),
            "native_scalers_sha256": sha(ap_root / "protocols/scalers.json"),
            "instrumentation_hashes": {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
            "packages": sorted((p.metadata["Name"], p.version)
                               for p in importlib.metadata.distributions())}
