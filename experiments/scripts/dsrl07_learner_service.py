"""Local-only JSON service for the pinned author DSRL learner on its own GPU."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import sys
import threading
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.dsrl06_learner import LearnerBridge


def serve(bridge, host, port, *, actor_only=False):
    class Handler(BaseHTTPRequestHandler):
        def response(self, status, value):
            data = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/status":
                self.response(200, bridge.status())
            else:
                self.response(404, {"error": "unknown route"})

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 2000000:
                    raise ValueError("Invalid JSON request length")
                request = json.loads(self.rfile.read(length))
                if actor_only and self.path not in ("/act", "/shutdown"):
                    self.response(404, {"error": "unknown actor-only route"})
                    return
                if self.path == "/act":
                    result = bridge.act(request)
                elif self.path == "/append":
                    result = bridge.append(request)
                elif self.path == "/train":
                    result = bridge.train(request)
                elif self.path == "/publish":
                    result = bridge.publish(request["round_id"])
                elif self.path == "/initialize":
                    result = bridge.initialize(request["phase"])
                elif self.path == "/profile":
                    result = bridge.profile(request)
                elif self.path == "/save":
                    result = bridge.save(request["name"])
                elif self.path == "/export-actor":
                    result = bridge.export_actor(request["name"])
                elif self.path == "/load":
                    result = bridge.load(request["path"])
                elif self.path == "/shutdown":
                    if bridge.active_training:
                        raise ValueError("Wait for the active training request before shutdown")
                    result = {"shutdown": True, "status": bridge.status()}
                    self.response(200, result)
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                    return
                else:
                    self.response(404, {"error": "unknown route"})
                    return
                self.response(200, result)
            except (ValueError, KeyError, TypeError, FileNotFoundError) as error:
                self.response(409, {"error": type(error).__name__, "message": str(error)})
            except Exception as error:
                traceback.print_exc()
                self.response(500, {"error": type(error).__name__, "message": str(error)})

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    print(
        json.dumps({"learner_service": f"http://{host}:{port}", "status": bridge.status()}),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "third_party/dsrl_pi0")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "outputs/dsrl07/learner")
    parser.add_argument("--data-root", type=Path, default=ROOT / "outputs/dsrl07")
    parser.add_argument("--phase", choices=("engineering", "training"), default="engineering")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--prior-engineering-updates", type=int, default=0)
    parser.add_argument("--prior-engineering-receipt")
    parser.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1",))
    parser.add_argument("--port", type=int, default=9009)
    args = parser.parse_args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("MPLCONFIGDIR", str(args.work_dir / "matplotlib-cache"))
    numerical = json.loads((args.data_root / "numerical-summary.json").read_text())
    if numerical.get("admit_live_integration") is not True:
        raise ValueError("Task-7 numerical diagnosis has not passed")
    modes = {
        "deterministic": "--xla_gpu_deterministic_ops=true",
        "deterministic-no-autotune": "--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0",
    }
    os.environ["XLA_FLAGS"] = modes[numerical["learner_mode"]]
    bridge = LearnerBridge(
        args.work_dir,
        args.source,
        data_root=args.data_root,
        prior_engineering_updates=args.prior_engineering_updates,
        prior_engineering_receipt=args.prior_engineering_receipt,
    )
    if args.resume:
        bridge.load(args.resume)
    else:
        bridge.initialize(args.phase)
    serve(bridge, args.host, args.port)


if __name__ == "__main__":
    main()
