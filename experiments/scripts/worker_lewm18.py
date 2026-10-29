"""Execute frozen task-18 jobs, retaining each failed start and complete trace."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
from helpers.dsrl06_infra import atomic_json, utc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--lanes", type=int, default=1)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    for name, expected in manifest["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Frozen source changed: {name}")
    import torch
    from helpers.lewm18_runtime import Runtime

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    runtimes = {}
    out = ROOT / "outputs/lewm18"
    for i, job in enumerate(manifest["jobs"]):
        if i % args.lanes != args.lane:
            continue
        if (out / "STOP").exists():
            break
        directory = out / "attempts" / job["id"]
        if directory.exists():
            print(json.dumps({"event": "retained_existing", "job": job["id"]}), flush=True)
            continue
        try:
            if job["env"] not in runtimes:
                runtimes[job["env"]] = Runtime(job["env"])
                if job["stage"] == "T":
                    runtime = runtimes[job["env"]]
                    dim = runtime.env.action_space.shape[0]
                    started = time.perf_counter()
                    with torch.inference_mode():
                        runtime.model.get_cost(
                            {
                                "pixels": torch.zeros(1, 300, 1, 3, 224, 224, device="cuda"),
                                "goal": torch.ones(1, 300, 1, 3, 224, 224, device="cuda"),
                                "action": torch.zeros(1, 300, 1, dim, device="cuda"),
                            },
                            torch.zeros(1, 300, 5, 5 * dim, device="cuda"),
                        )
                    torch.cuda.synchronize()
                    atomic_json(
                        out / f"serial-warmup-{job['env']}.json",
                        {
                            "utc": utc(),
                            "seconds": time.perf_counter() - started,
                            "resources": {
                                "encoder_frames": 2,
                                "model_batch_calls": 1,
                                "model_candidate_rollouts": 300,
                                "predictor_steps": 1500,
                            },
                            "note": "Same synthetic warm-up before timing each task, no simulator actions",
                        },
                    )
            result = runtimes[job["env"]].run(job, directory)
            print(
                json.dumps(
                    {
                        "job": job["id"],
                        "safe": result["requested_safe_completion"],
                        "actions": result["actions"],
                        "seconds": result["episode_seconds"],
                        "failure": result["controller_failure"],
                        "error": bool(result["error"]),
                    }
                ),
                flush=True,
            )
        except Exception:
            directory.mkdir(parents=True, exist_ok=True)
            atomic_json(
                directory / "failed-start.json",
                {"utc": utc(), "job": job, "error": traceback.format_exc()},
            )
            print(json.dumps({"job": job["id"], "failed_start": True}), flush=True)
            raise
    atomic_json(out / f"{manifest['name']}-lane-{args.lane}-done.json", {"utc": utc()})


if __name__ == "__main__":
    main()
