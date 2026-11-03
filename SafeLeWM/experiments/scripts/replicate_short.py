"""Run the unchanged authors' evaluator with passive accounting on the rented host."""

import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback
import types

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "assets/ap-reference"
SOURCE = REF / "code/lewm/protocol_check/official_source"
sys.path.insert(0, str(REF / "code/ap"))
from anchored_planning.model import load_model  # noqa: E402


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    return value


def main():
    assert os.environ.get("RUNPOD_POD_ID") == "yvrg8ow7i9n9u6"
    out = ROOT / "outputs/anchored-20261101/short-v1"
    out.mkdir(parents=True, exist_ok=False)
    cache = out / "cache"
    (cache / "short").mkdir(parents=True)
    (cache / "tworoom.h5").symlink_to(ROOT / "data/tworoom.h5")
    os.environ["STABLEWM_HOME"] = str(cache)
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with h5py.File(ROOT / "data/tworoom.h5", "r") as data:
        lengths, offsets = data["ep_len"][:], data["ep_offset"][:]
        valid = np.concatenate([
            np.arange(o, o + n - 25) for o, n in zip(offsets, lengths) if n > 25
        ])
        rows = np.sort(valid[np.random.default_rng(42).choice(len(valid) - 1, 50, False)])
        panel = [dict(row=int(r), episode=int(np.searchsorted(offsets, r, side="right") - 1),
                      goal_row=int(r + 24)) for r in rows]
    save(out / "panel.json", panel)
    sources = [ROOT / "AnchoredQualification.md", Path(__file__), *SOURCE.rglob("*.py"),
               *SOURCE.rglob("*.yaml"), *(REF / "code/ap/anchored_planning").glob("*.py")]
    save(out / "freeze.json", dict(
        utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        sources={str(p.relative_to(ROOT)): digest(p) for p in sources},
        panel_sha256=digest(out / "panel.json"),
        checkpoint_sha256=digest(ROOT / "assets/weights.pt"),
        reference_revision="984327424ca7a2e52c75a8279197750d9d0b7c8e",
    ))
    report = dict(status="setup", resources={}, source_modified=False)
    # eval.py uses SPT only for these constants. Load its actual source module
    # without importing unrelated training callbacks and optional integrations.
    stats_path = ROOT / "ap-deps/stable_pretraining/data/dataset_stats.py"
    spec = importlib.util.spec_from_file_location("spt_eval_dataset_stats", stats_path)
    stats = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stats)
    spt = types.ModuleType("stable_pretraining")
    spt.data = types.SimpleNamespace(dataset_stats=stats)
    sys.modules["stable_pretraining"] = spt
    report["dependency_shim"] = dict(
        scope="SPT ImageNet constants only; unchanged installed dataset_stats.py",
        source_sha256=digest(stats_path), constants=stats.ImageNet,
    )
    save(out / "run.json", report)
    model = load_model(ROOT / "assets/weights.pt", torch.device("cpu")).official
    torch.save(model, cache / "short/lewm_object.ckpt")
    restored = torch.load(cache / "short/lewm_object.ckpt", weights_only=False)
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
    del model, restored
    counts = collections.Counter()
    traces, identifiers = {}, {}
    started = time.monotonic()

    def profile(frame, event, arg):
        filename, name = frame.f_code.co_filename, frame.f_code.co_name
        local = frame.f_locals
        if "/two_room/" in filename and name in {"step", "reset"}:
            env = local["self"]
            index = identifiers.setdefault(id(env), len(identifiers))
            trace = traces.setdefault(index, dict(before=[], after=[], actions=[], goals=[], term=[]))
            if event == "call":
                counts["environment_" + ("resets" if name == "reset" else "actions")] += 1
                if counts["environment_actions"] > 2500 or counts["environment_resets"] > 100:
                    raise RuntimeError("Frozen interaction budget exceeded")
                if name == "step":
                    trace["before"].append(env.agent_position.numpy().copy())
                    trace["goals"].append(env.target_position.numpy().copy())
                    trace["actions"].append(np.array(local["action"]).copy())
            elif event == "return" and name == "step" and arg is not None:
                trace["after"].append(env.agent_position.numpy().copy())
                trace["term"].append(bool(arg[2]))
        elif event == "call" and name == "get_cost" and filename.endswith("/jepa.py"):
            b, n, h, _ = local["action_candidates"].shape
            counts["model_batch_calls"] += 1
            counts["candidate_model_steps"] += b * n * h
            if counts["candidate_model_steps"] > 6000000:
                raise RuntimeError("Frozen model budget exceeded")
        elif event == "call" and name == "encode" and filename.endswith("/jepa.py"):
            counts["encoder_frames"] += int(np.prod(local["info"]["pixels"].shape[:2]))
        elif event == "return" and name == "step" and filename.endswith("/world.py"):
            counts["world_steps"] += 1
            if counts["world_steps"] % 5 == 0:
                report.update(status="running", resources=dict(counts), seconds=time.monotonic()-started)
                save(out / "run.json", report)
                print(json.dumps(report), flush=True)
        elif event == "return" and name == "run" and Path(filename) == SOURCE / "eval.py":
            if "metrics" in local:
                np.testing.assert_array_equal(local["random_episode_indices"], rows)
                report.update(status="complete", metrics=plain(local["metrics"]))
        if time.monotonic() - started > 1800:
            raise TimeoutError("Short-goal stage deadline")

    sys.path.insert(0, str(SOURCE))
    sys.argv = [str(SOURCE / "eval.py"), "--config-name=tworoom", "policy=short/lewm",
                "seed=42", "eval.num_eval=50", f"output.filename={out}/official_output.txt",
                f"hydra.run.dir={out}/hydra"]
    try:
        sys.setprofile(profile)
        runpy.run_path(str(SOURCE / "eval.py"), run_name="__main__")
    except BaseException:
        report.update(status="error", error=traceback.format_exc())
        raise
    finally:
        sys.setprofile(None)
        report.update(resources=dict(counts), seconds=time.monotonic()-started)
        save(out / "run.json", report)
        for i, trace in traces.items():
            np.savez_compressed(out / f"trace-{i:02d}.npz", **trace)
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
