"""Compare rendering backends on a fixed engineering state, with zero action steps."""

import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("backend", choices=["osmesa", "egl"])
args = parser.parse_args()
os.environ["MUJOCO_GL"] = args.backend
os.environ["PYOPENGL_PLATFORM"] = args.backend

import h5py  # noqa: E402
import hdf5plugin  # noqa: E402, F401
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import stable_worldmodel as swm  # noqa: E402

from evaluate import Ledger, publish, serial  # noqa: E402
from run_io import publish_artifact  # noqa: E402


def main():
    cfg = json.loads(Path(os.environ["LEWM_CONFIG"]).read_text())
    root = Path(cfg["asset_root"])
    run_id = Path(os.environ["LEWM_RUN_DIR"]).name
    ledger = Ledger(root, "engineering")
    ledger.reserve("reacher-render-" + args.backend + "-" + run_id, 1)
    cases_path = (root.parents[1] / "runs/repro-20261104-engineering-reacher-02/artifacts/"
                  "reacher-engineering-seed0-cem10/cases.json")
    case = json.loads(cases_path.read_text())["cases"][0]
    spec = cfg["tasks"]["reacher"]
    with h5py.File(root / "datasets" / (spec["dataset_name"] + ".h5"), "r") as handle:
        row = case["row"]
        start = {k: handle[k][row] for k in handle if k not in ("ep_len", "ep_offset")}
        goal_qpos = handle["qpos"][row + 25]
    variations = {k.removeprefix("variation."): v for k, v in start.items()
                  if k.startswith("variation.")}
    options = {"variation": list(variations), "variation_values": variations} if variations else {}
    report = {"backend": args.backend, "case": case, "simulator_actions": 0,
              "model_queries": 0, "status": "incomplete", "variation_values": variations}
    world = None
    try:
        world = swm.World(env_name="swm/ReacherDMControl-v0", num_envs=1,
                          image_shape=(224, 224), max_episode_steps=100,
                          task="qpos_match", verbose=0)
        seed = None if "seed" not in start else [int(start["seed"])]
        world.reset(seed=seed, options=[options])
        base = world.envs.unwrapped.envs[0].unwrapped
        base.set_state(start["qpos"], start["qvel"])
        base.set_target_qpos(goal_qpos)
        frame = base.render()
        expected = start["pixels"]
        if expected.shape[0] == 3:
            expected = expected.transpose(1, 2, 0)
        report.update(status="completed", mean_absolute_pixel_error=float(
            np.abs(frame.astype(float) - expected).mean() / 255),
            floor_colors=base.variation_space["floor"]["color"].value,
            qpos_error=float(np.abs(base.env.physics.data.qpos - start["qpos"]).max()))
        for name, array in [("rendered", frame), ("dataset", expected)]:
            path = Path(os.environ["LEWM_RUN_DIR"]) / ".control" / (name + ".png")
            Image.fromarray(array).save(path)
            publish_artifact("artifacts/" + name + ".png", source=path)
        publish_artifact("artifacts/reacher-model.xml", data=base._mjcf_model.to_xml_string())
        physics_model = base.env.physics.model
        for key in ("tex_rgb", "tex_data", "mat_texrepeat", "mat_texuniform"):
            if hasattr(physics_model, key):
                value = np.asarray(getattr(physics_model, key))
                report[key] = {"shape": value.shape, "min": float(value.min()),
                               "max": float(value.max())}
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        publish("artifacts/render-check.json", report)
        print(json.dumps(serial(report)), flush=True)
        if world:
            world.close()


if __name__ == "__main__":
    main()
