#!/usr/bin/env python3
"""Check lossless RGB trace packing using saved JSON/JSONL evidence only.

No simulator, model, or NumPy is imported. Existing evidence is never rewritten.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "LeWM_Experiment1/computeConfigs/scripts"))
from common import IMAGE_TAG, decode_rgb_bytes, encode_rgb_bytes, serial  # noqa: E402


def objects(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        if path.name.endswith((".jsonl", ".jsonl.gz")):
            for index, line in enumerate(stream):
                if line.strip():
                    yield f"line:{index + 1}", json.loads(line)
        else:
            yield "$", json.load(stream)


def images(value, location):
    if isinstance(value, dict):
        if IMAGE_TAG in value:
            yield location, value, None
        else:
            for key, item in value.items():
                yield from images(item, location + "." + key)
    elif isinstance(value, list):
        candidate = (value and isinstance(value[0], list) and value[0]
                     and isinstance(value[0][0], list) and len(value[0][0]) == 3)
        if candidate:
            width = len(value[0])
            if all(isinstance(row, list) and len(row) == width and all(
                    isinstance(pixel, list) and len(pixel) == 3
                    and all(type(channel) is int and 0 <= channel <= 255 for channel in pixel)
                    for pixel in row) for row in value):
                raw = bytes(channel for row in value for pixel in row for channel in pixel)
                yield location, encode_rgb_bytes(raw, [len(value), width, 3]), raw
                return
        for index, item in enumerate(value):
            yield from images(item, location + f"[{index}]")


def main(args):
    if args.limit < 1:
        raise ValueError("Image check limit must be positive")
    started = time.monotonic()
    checked = []
    for location, obj in objects(args.input):
        for field, encoded, legacy_raw in images(obj, location):
            decoded = decode_rgb_bytes(encoded)
            if legacy_raw is not None and decoded != legacy_raw:
                raise AssertionError("Saved legacy RGB pixels changed during lossless packing")
            if decode_rgb_bytes(encode_rgb_bytes(decoded, encoded["shape"])) != decoded:
                raise AssertionError("RGB codec round trip changed pixel bytes")
            if serial(encoded) != encoded:
                raise AssertionError("Repeated record serialization changed packed evidence")

            class SavedRGB:
                ndim, dtype = 3, "uint8"
                shape = tuple(encoded["shape"])

                def tobytes(self, order):
                    if order != "C":
                        raise AssertionError("Unexpected image byte order")
                    return decoded

            if decode_rgb_bytes(serial(SavedRGB())) != decoded:
                raise AssertionError("Recorder serialization branch changed pixel bytes")
            damaged = dict(encoded, sha256="0" * 64)
            try:
                decode_rgb_bytes(damaged)
            except ValueError:
                pass
            else:
                raise AssertionError("Decoder accepted an incorrect checksum")
            checked.append({"field": field, "shape": encoded["shape"], "dtype": encoded["dtype"],
                            "sha256": hashlib.sha256(decoded).hexdigest(),
                            "raw_bytes": len(decoded),
                            "packed_json_bytes": len(json.dumps(encoded).encode()),
                            "legacy_pixels_compared": legacy_raw is not None})
            if len(checked) == args.limit:
                break
        if len(checked) == args.limit:
            break
    if not checked:
        raise ValueError("No saved RGB image was found; no equivalence claim can be made")
    with args.input.open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {"input": str(args.input.resolve()), "input_sha256": source_hash,
              "passed": True, "images_checked": checked,
              "scope": "Lossless saved-evidence codec check; no simulation or model execution",
              "simulator_actions": 0, "model_queries": 0,
              "elapsed_seconds": time.monotonic() - started}
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(text)
    print(text, end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=8)
    main(parser.parse_args())
