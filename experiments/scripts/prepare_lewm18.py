"""Pinned public inputs in fresh task-18 paths; no scoring or historical job execution."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
os.environ["STABLEWM_HOME"] = str(ROOT / "data/lewm18/stablewm")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
from helpers.dsrl06_infra import atomic_json, utc
import prepare_cube14 as cube_utils
import prepare_safecem10 as push_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=("cube", "pusht"))
    parser.add_argument("--keep-archive", action="store_true")
    args = parser.parse_args()
    out = ROOT / "outputs/lewm18"
    out.mkdir(parents=True, exist_ok=True)
    for env in [args.env] if args.env else ("cube", "pusht"):
        spec = json.loads((ROOT / f"configs/lewm18/{env}.json").read_text())
        spec["pins"]["checkpoint_name"] = f"{env}/lewm"
        data = ROOT / f"data/lewm18/{env}"
        cube_utils.DATA = data
        cube_utils.OUT = out / env
        push_utils.DATA = data
        push_utils.OUT = out / env
        print(json.dumps({"event": "prepare", "environment": env, "utc": utc()}), flush=True)
        folder = cube_utils.fetch_model(spec)
        model = cube_utils.convert(folder, spec)
        if env == "cube":
            cube_utils.unpack_dataset(spec)
            dataset = cube_utils.dataset_file()
            scalers = cube_utils.build_scalers(dataset)
        else:
            dataset = push_utils.fetch_dataset(spec)
            scalers = push_utils.build_scalers(dataset)
        import h5py
        import hdf5plugin  # noqa: F401

        with h5py.File(dataset, "r") as f:
            columns = {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in f.items()}
        atomic_json(
            out / f"{env}-prepared.json",
            {
                "utc": utc(),
                "model": str(model),
                "dataset": str(dataset),
                "scalers": str(scalers),
                "columns": columns,
                "pins": spec["pins"],
            },
        )
        # The verified extracted dataset is retained; release its downloaded archive.
        if not args.keep_archive:
            (data / "raw" / spec["pins"]["dataset_file"]).unlink(missing_ok=True)
        print(json.dumps({"event": "prepared", "environment": env, "utc": utc()}), flush=True)
    if all((out / f"{kind}-prepared.json").exists() for kind in ("cube", "pusht")):
        atomic_json(out / "prepared.json", {"utc": utc(), "environments": ["cube", "pusht"]})


if __name__ == "__main__":
    main()
