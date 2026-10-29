"""Local-only JSON service for the task-11 learners on one GPU, or CPU actor exports."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

DETERMINISTIC = "--xla_gpu_deterministic_ops=true"


def serve(bridge, host, port, *, actor_only=False):
    routes = {
        "/act": bridge.act,
        "/initialize": lambda r: bridge.initialize(r["phase"]),
        "/load-bank": getattr(bridge, "load_bank", None),
        "/append": getattr(bridge, "append", None),
        "/train": getattr(bridge, "train", None),
        "/publish": getattr(bridge, "publish", None),
        "/profile": getattr(bridge, "profile", None),
        "/save": lambda r: bridge.save(r["name"]),
        "/export-actor": getattr(bridge, "export_actor", None),
        "/load": lambda r: bridge.load(r["path"]),
    }
    routes = {path: route for path, route in routes.items() if route is not None}
    if actor_only:
        routes = {"/act": bridge.act}

    class Handler(BaseHTTPRequestHandler):
        def response(self, status, value):
            data = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):  # noqa: A002 - BaseHTTPRequestHandler API
            if not self.path.startswith(("/act", "/status")):
                super().log_message(format, *args)

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
                if self.path == "/shutdown":
                    if bridge.active_training:
                        raise ValueError("Wait for the active training request before shutdown")
                    self.response(200, {"shutdown": True, "status": bridge.status()})
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                    return
                route = routes.get(self.path)
                if route is None:
                    self.response(404, {"error": "unknown route"})
                    return
                self.response(200, route(request))
            except (ValueError, KeyError, TypeError, FileNotFoundError) as error:
                self.response(409, {"error": type(error).__name__, "message": str(error)})
            except Exception as error:
                traceback.print_exc()
                self.response(500, {"error": type(error).__name__, "message": str(error)})

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    print(json.dumps({"service": f"http://{host}:{port}", "status": bridge.status()}), flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()


def selected_config(path):
    from helpers.dsrl11_learner import Dsrl11LearnerConfig

    selection = json.loads(Path(path).read_text())
    if selection.get("schema") != "dsrl11-learner-selection-v1" or not selection.get("selected"):
        raise ValueError("The learner requires the frozen A1 selection receipt")
    utd, reduction = selection["selected"]
    return Dsrl11LearnerConfig(updates_per_transition=utd, critic_reduction=reduction)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "third_party/dsrl_pi0")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--phase", choices=("engineering", "learning"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--actor-export", action="append", default=[], metavar="CONDITION=PATH")
    parser.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1",))
    parser.add_argument("--port", type=int, default=9011)
    args = parser.parse_args()
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("MPLCONFIGDIR", str(args.work_dir / "matplotlib-cache"))
    if args.actor_export:
        # CPU actor exports: set before the first lazy JAX import; no GPU is claimed.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        from helpers.dsrl11_learner import FrozenResidualActors

        exports = dict(item.split("=", 1) for item in args.actor_export)
        bridge = FrozenResidualActors(args.work_dir, args.source, exports)
        serve(bridge, args.host, args.port, actor_only=True)
        return
    os.environ["XLA_FLAGS"] = DETERMINISTIC
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    from helpers.dsrl11_learner import Dsrl11LearnerBridge

    bridge = Dsrl11LearnerBridge(
        args.work_dir, args.source, selected_config(args.selection), data_root=args.data_root
    )
    if args.resume:
        bridge.load(args.resume)
    else:
        bridge.initialize(args.phase)
    serve(bridge, args.host, args.port)


if __name__ == "__main__":
    main()
