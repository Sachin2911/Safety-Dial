"""Passive tracing and video capture around the unmodified native AP evaluator."""

from collections import Counter
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from common import check_deadline, publish, serial
from run_io import publish_artifact


def compress_closed_journal(source, destination):
    """Preserve exact closed journal bytes in gzip and verify a fresh decompression."""
    source, destination = Path(source), Path(destination)
    before = source.stat()
    original_hash, original_bytes = hashlib.sha256(), 0
    with source.open("rb") as incoming, destination.open("xb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output,
                           compresslevel=3, mtime=0) as compressed:
            for block in iter(lambda: incoming.read(1024 * 1024), b""):
                original_hash.update(block)
                original_bytes += len(block)
                compressed.write(block)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size, after.st_mtime_ns, after.st_ino) or original_bytes != before.st_size:
        raise RuntimeError("Closed journal changed while compressing")
    restored_hash, restored_bytes = hashlib.sha256(), 0
    with gzip.open(destination, "rb") as restored:
        for block in iter(lambda: restored.read(1024 * 1024), b""):
            restored_hash.update(block)
            restored_bytes += len(block)
            if restored_bytes > original_bytes:
                raise RuntimeError("Decompressed journal exceeds its original size")
    if restored_bytes != original_bytes or restored_hash.digest() != original_hash.digest():
        raise RuntimeError("Journal compression round-trip differs from original bytes")
    with destination.open("rb") as compressed:
        compressed_hash = hashlib.file_digest(compressed, "sha256").hexdigest()
    return {"encoding": "gzip", "compression_level": 3, "original_filename": source.name,
            "original_bytes": original_bytes, "original_sha256": original_hash.hexdigest(),
            "compressed_bytes": destination.stat().st_size, "compressed_sha256": compressed_hash,
            "verification": "Fresh decompression matched original byte count and SHA-256",
            "raw_retention": "Original closed file remains in the run's excluded .control scratch area"}


class Observer:
    def __init__(self, config, task):
        self.config, self.task = config, task
        self.total, self.case = Counter(), None
        self.phase = "setup"
        self.case_limits = {}
        self.started = time.monotonic()
        self.patches, self.handles = [], []

    def add(self, name, amount=1):
        self.total[name] += amount
        if self.case is not None:
            self.case[name] += amount
            if name in self.case_limits and self.case[name] > self.case_limits[name]:
                raise ValueError("Reserved case compute budget exceeded: " + name)

    def patch(self, owner, name, replacement):
        self.patches.append((owner, name, getattr(owner, name)))
        setattr(owner, name, replacement)

    def guard(self):
        check_deadline(self.config, self.started)

    def install(self):
        import gymnasium as gym
        import mujoco
        import pymunk
        from anchored_planning import environments, model
        from anchored_planning.observation_bank import ObservationBank

        step = mujoco.mj_step

        def physics(*args, **kwargs):
            count = int(kwargs.get("nstep", args[2] if len(args) > 2 else 1))
            self.add(self.phase + "_physics_steps", count)
            result = step(*args, **kwargs)
            self.add(self.phase + "_physics_steps_completed", count)
            return result

        self.patch(mujoco, "mj_step", physics)
        # DM-Control's legacy Euler path integrates once through mj_step2,
        # then uses mj_step for any remaining substeps. mj_step1 does not integrate.
        step2 = mujoco.mj_step2

        def physics_step2(*args, **kwargs):
            self.add(self.phase + "_physics_steps")
            result = step2(*args, **kwargs)
            self.add(self.phase + "_physics_steps_completed")
            return result

        self.patch(mujoco, "mj_step2", physics_step2)
        space_step = pymunk.Space.step

        def pymunk_step(space, dt):
            self.add(self.phase + "_physics_steps")
            result = space_step(space, dt)
            self.add(self.phase + "_physics_steps_completed")
            return result

        self.patch(pymunk.Space, "step", pymunk_step)
        if self.task == "pusht":
            cls = environments.PushT
        else:
            entry = gym.spec({"tworoom": "swm/TwoRoom-v1", "reacher": "swm/ReacherDMControl-v0",
                              "cube": "swm/OGBCube-v0"}[self.task]).entry_point
            if isinstance(entry, str):
                module, name = entry.split(":")
                cls = getattr(importlib.import_module(module), name)
            else:
                cls = entry
        native_step = cls.step

        def controller(instance, *args, **kwargs):
            self.add(self.phase + "_controller_calls")
            result = native_step(instance, *args, **kwargs)
            self.add(self.phase + "_controller_calls_completed")
            return result

        self.patch(cls, "step", controller)
        load = model.load_model

        def load_model(*args, **kwargs):
            instance = load(*args, **kwargs)

            def encoder(_module, inputs):
                self.guard()
                self.add("encoder_calls")
                self.add("encoder_images", int(inputs[0].shape[0]))

            def predictor(_module, inputs):
                self.guard()
                self.add("predictor_calls")
                self.add("predictor_candidate_blocks", int(inputs[0].shape[0]))

            self.handles.extend([
                instance.official.encoder.register_forward_pre_hook(encoder),
                instance.official.predictor.register_forward_pre_hook(predictor)])
            return instance

        self.patch(model, "load_model", load_model)
        nearest = ObservationBank.nearest_topk

        def nearest_topk(instance, *args, **kwargs):
            self.guard()
            self.add("retrieval_queries")
            return nearest(instance, *args, **kwargs)

        self.patch(ObservationBank, "nearest_topk", nearest_topk)

    def close(self):
        for handle in self.handles:
            handle.remove()
        for owner, name, original in reversed(self.patches):
            setattr(owner, name, original)


class Video:
    """Stream existing render results to H.264, beside the fixed dataset goal."""

    def __init__(self, path, goal, title, config):
        from PIL import Image

        self.path, self.title = Path(path), title
        self.goal = Image.fromarray(goal).resize((448, 448))
        self.frames, self.frame_steps = 0, []
        ffmpeg = config.get("ffmpeg_path") or shutil.which("ffmpeg")
        if not ffmpeg:
            raise FileNotFoundError("ffmpeg must be installed on the rented host before evaluation")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "896x520", "-r", "2",
            "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE)

    def add(self, rgb, step):
        from PIL import Image, ImageDraw

        canvas = Image.new("RGB", (896, 520), "#f5f6f8")
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), self.title, fill="#15394a")
        draw.text((12, 28), f"Planner observation after {step} actions", fill="#15394a")
        draw.text((460, 28), "Fixed dataset goal", fill="#15394a")
        canvas.paste(Image.fromarray(rgb).resize((448, 448)), (0, 50))
        canvas.paste(self.goal, (448, 50))
        draw.text((12, 503),
                  "Original evaluation frames; 5-action spacing; last frame may precede stop",
                  fill="#15394a")
        self.process.stdin.write(canvas.tobytes())
        self.frames += 1
        self.frame_steps.append(step)

    def close(self):
        pipe_error = None
        try:
            self.process.stdin.close()
        except BrokenPipeError as error:
            pipe_error = error
        try:
            code = self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            raise
        if code:
            raise RuntimeError("Video encoding failed with exit code " + str(code))
        if pipe_error:
            raise pipe_error


class Recorder:
    """Return the exact native objects; observe the actions and existing render calls."""

    def __init__(self, env, task, case, initial, goal_image, observer, output, config,
                 on_close=None, video=True):
        from anchored_planning import environments

        self.env, self.task, self.case = env, task, case
        self.observer, self.output, self.config = observer, Path(output), config
        self.on_close, self.closed = on_close, False
        self.steps, self.render_hashes = [], []
        self.initial = serial(dict(initial, snapshot=environments._snapshot(task, env)))
        self.counts = observer.case
        self.path = self.output / "observations" / case
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.journal = self.path.with_suffix(".jsonl").open("x")
        self.event({"event": "initial", **self.initial})
        self.video = Video(self.path.with_suffix(".mp4"), goal_image,
                           f"{task.upper()} | {case}", config) if video else None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def event(self, data):
        self.journal.write(json.dumps(serial(data), allow_nan=False) + "\n")
        self.journal.flush()

    def render(self, *args, **kwargs):
        import numpy as np

        self.observer.guard()
        rgb = self.env.render(*args, **kwargs)
        frame = np.ascontiguousarray(rgb)
        digest = hashlib.sha256(frame.tobytes()).hexdigest()
        self.render_hashes.append({"action_index": len(self.steps), "sha256": digest})
        if self.video:
            self.video.add(frame, len(self.steps))
        return rgb

    def step(self, action):
        import numpy as np
        from anchored_planning import environments

        self.observer.guard()
        if len(self.steps) >= self.initial["action_budget"]:
            raise ValueError("Case action budget exceeded")
        if not np.isfinite(action).all():
            raise ValueError("Non-finite native action")
        if np.any(action < self.action_space.low) or np.any(action > self.action_space.high):
            raise ValueError("Native action outside physical domain")
        self.observer.phase = "policy"
        self.observer.add("policy_actions_attempted")
        self.event({"event": "action_attempt", "action": action,
                    "counts": self.counts})
        result = self.env.step(action)
        obs, reward, terminated, truncated, info = result
        row = serial({"action": action, "observation": obs, "reward": reward,
                      "terminated": terminated, "truncated": truncated, "info": info,
                      "snapshot": environments._snapshot(self.task, self.env)})
        self.steps.append(row)
        self.observer.add("policy_actions_completed")
        self.event({"event": "action_complete", "step": len(self.steps),
                    "transition": row, "counts": self.counts})
        return result

    def close(self):
        if self.closed:
            return
        self.closed = True
        failure = None
        trace = {"task": self.task, "case": self.case, "initial": self.initial,
                 "steps": self.steps, "render_hashes": self.render_hashes,
                 "counts": dict(self.counts),
                 "video": {"frame_steps": self.video.frame_steps,
                           "frames": self.video.frames,
                           "capture": "native planner render calls only"}
                          if self.video else None}
        try:
            if self.video:
                self.video.close()
        except Exception as error:
            failure = error
        try:
            self.event({"event": "close", "counts": self.counts})
            self.journal.close()
            journal = self.path.with_suffix(".jsonl.gz")
            trace["journal"] = {"path": f"metrics/journals/{self.case}.jsonl.gz",
                                **compress_closed_journal(self.path.with_suffix(".jsonl"), journal)}
            compressed = self.path.with_suffix(".json.gz")
            with gzip.open(compressed, "wt") as stream:
                json.dump(trace, stream, allow_nan=False)
            publish_artifact(f"metrics/trajectories/{self.case}.json.gz", source=compressed)
            publish_artifact(trace["journal"]["path"], source=journal)
            if self.video and failure is None:
                publish_artifact(f"artifacts/videos/{self.case}.mp4",
                                 source=self.path.with_suffix(".mp4"))
        except Exception as error:
            failure = failure or error
        finally:
            if not self.journal.closed:
                self.journal.close()
            try:
                self.env.close()
            except Exception as error:
                failure = failure or error
            finally:
                if self.on_close:
                    self.on_close(trace, failure)
        if failure:
            raise failure


def publish_failure(observer, error):
    publish("metrics/failure.json", {"error_type": type(error).__name__, "error": str(error),
            "counts": observer.total, "elapsed_seconds": time.monotonic() - observer.started,
            "run": Path(os.environ["LEWM_RUN_DIR"]).name})
