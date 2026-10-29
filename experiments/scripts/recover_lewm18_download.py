"""Resume the stalled public Cube archive; only a full pinned SHA-256 admits it."""

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_infra import atomic_json, utc

OUT = ROOT / "outputs/lewm18"
RAW = ROOT / "data/lewm18/cube/raw"
STATE = OUT / "environment/cube-download-recovery.json"
BLOCK = 32 * 1024 * 1024
WORKERS = 16


def main():
    pins = json.loads((ROOT / "configs/lewm18/cube.json").read_text())["pins"]
    total = pins["dataset_bytes"]
    expected_sha = pins["dataset_sha256"]
    target = RAW / pins["dataset_file"]
    staging = target.with_suffix(target.suffix + ".recovery")
    url = (
        f"https://huggingface.co/datasets/{pins['dataset_repo']}/resolve/"
        f"{pins['dataset_revision']}/{pins['dataset_file']}"
    )
    if target.exists():
        raise FileExistsError("Final archive already exists; verify preparation instead")
    if STATE.exists():
        state = json.loads(STATE.read_text())
        assert state["expected_sha256"] == expected_sha and state["total_bytes"] == total
        assert staging.exists()
        state["resumed_utc"] = utc()
    else:
        candidates = list(RAW.rglob("*.incomplete"))
        assert len(candidates) == 1 and not staging.exists()
        partial = candidates[0]
        pid = 1352
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
        assert b"prepare_lewm18.py" in command
        assert Path(f"/proc/{pid}/cwd").resolve() == ROOT
        os.kill(pid, signal.SIGSTOP)
        try:
            for _ in range(100):
                status = Path(f"/proc/{pid}/status").read_text()
                if "State:\tT" in status:
                    break
                time.sleep(0.01)
            else:
                raise RuntimeError("Downloader did not stop")
            prefix = partial.stat().st_size
            assert 0 < prefix < total
            os.link(partial, staging)
        except BaseException:
            os.kill(pid, signal.SIGCONT)
            raise
        os.kill(pid, signal.SIGKILL)
        state = {
            "started_utc": utc(),
            "expected_sha256": expected_sha,
            "total_bytes": total,
            "existing_prefix_bytes": prefix,
            "completed_chunks": [],
            "network_bytes": 0,
            "requests": 0,
            "retries": 0,
            "workers": WORKERS,
            "chunk_bytes": BLOCK,
            "status": "resuming",
            "old_downloader_pid": pid,
            "partial_source_path": str(partial.relative_to(RAW)),
            "prefix_probe": "64 KiB at start, midpoint and end matched HTTP ranges before recovery",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    prefix = state["existing_prefix_bytes"]
    chunks = [(start, min(start + BLOCK, total)) for start in range(prefix, total, BLOCK)]
    done = set(state["completed_chunks"])
    lock = threading.Lock()
    cancelled = threading.Event()
    monitoring_done = threading.Event()
    descriptor = os.open(staging, os.O_RDWR)

    def checkpoint():
        with lock:
            state["completed_chunks"] = sorted(done)
            state["heartbeat_utc"] = utc()
            state["verified_range_bytes"] = prefix + sum(chunks[i][1] - chunks[i][0] for i in done)
            atomic_json(STATE, state)

    def monitor():
        while not monitoring_done.wait(5):
            checkpoint()

    def fetch(index):
        start, end = chunks[index]
        for attempt in range(5):
            if cancelled.is_set() or (OUT / "STOP").exists():
                raise InterruptedError("Campaign stopped")
            received = 0
            try:
                with lock:
                    state["requests"] += 1
                with requests.get(
                    url,
                    headers={"Range": f"bytes={start}-{end - 1}", "Accept-Encoding": "identity"},
                    stream=True,
                    timeout=(20, 45),
                ) as response:
                    expected_range = f"bytes {start}-{end - 1}/{total}"
                    if (
                        response.status_code != 206
                        or response.headers.get("Content-Range") != expected_range
                    ):
                        raise ValueError("Server did not honor the exact requested range")
                    for data in response.iter_content(chunk_size=256 * 1024):
                        if cancelled.is_set() or (OUT / "STOP").exists():
                            raise InterruptedError("Campaign stopped")
                        if received + len(data) > end - start:
                            raise ValueError("Server sent more than the requested range")
                        offset = 0
                        while offset < len(data):
                            written = os.pwrite(
                                descriptor, data[offset:], start + received + offset
                            )
                            if written <= 0:
                                raise OSError("Incomplete positional write")
                            offset += written
                        received += len(data)
                        with lock:
                            state["network_bytes"] += len(data)
                if received != end - start:
                    raise ValueError("Truncated range response")
                with lock:
                    done.add(index)
                return
            except Exception as exc:
                with lock:
                    state["last_error_type"] = type(exc).__name__
                    state["last_error_chunk"] = index
                    state["retries"] += 1
                if isinstance(exc, InterruptedError) or attempt == 4:
                    cancelled.set()
                    raise
                time.sleep(min(2**attempt, 8))

    checkpoint()
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        pending = [i for i in range(len(chunks)) if i not in done]
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = [pool.submit(fetch, i) for i in pending]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        assert len(done) == len(chunks)
        os.ftruncate(descriptor, total)
        os.fsync(descriptor)
        with lock:
            state["status"] = "checking_full_pinned_hash"
        checkpoint()
        digest = hashlib.sha256()
        with staging.open("rb") as source:
            for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
                digest.update(block)
        actual = digest.hexdigest()
        with lock:
            state["actual_sha256"] = actual
        if actual != expected_sha:
            raise ValueError("Full archive checksum mismatch; file is not admitted")
        os.replace(staging, target)
        atomic_json(RAW / "verified.json", {"utc": utc(), "sha256": actual})
        partial_source = RAW / state["partial_source_path"]
        if partial_source.exists():
            assert os.path.samefile(partial_source, target)
            partial_source.unlink()
        with lock:
            state["status"] = "verified_complete"
            state["finished_utc"] = utc()
        print(json.dumps({"status": "verified_complete", "sha256": actual}), flush=True)
    except BaseException as exc:
        cancelled.set()
        with lock:
            state["status"] = "failed_retained"
            state["error_type"] = type(exc).__name__
        raise
    finally:
        monitoring_done.set()
        thread.join()
        os.close(descriptor)
        checkpoint()


if __name__ == "__main__":
    main()
