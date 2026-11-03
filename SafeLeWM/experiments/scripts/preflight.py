"""One counted simulator reset and two actions, before the qualification episodes."""

import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.qualification import Planner, load_model, new_counters  # noqa: E402


def main():
    assert os.environ.get("RUNPOD_POD_ID") == "yvrg8ow7i9n9u6"
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    root = Path(__file__).resolve().parents[2]
    out = root / "outputs/qualification-20261101/preflight.json"
    assert not out.exists(), "Do not repeat a counted preflight"
    counters = new_counters()
    result = {"passed": False, "resources": dict(counters)}
    out.write_text(json.dumps(result))
    torch.set_num_threads(4)
    model = load_model(root / "assets/weights.pt")
    env = TwoRoomEnv()
    try:
        counters["environment_resets"] += 1
        env.reset(seed=4199, options={"state": np.array([164., 90.]),
                                     "target_state": np.array([164., 170.])})
        image = env.render()
        counters["render_frames"] += 1
        before = env.agent_position.clone()
        target = env._render_frame(torch.tensor([134., 90.])).permute(1, 2, 0).numpy()
        counters["target_render_frames"] += 1
        assert torch.equal(env.agent_position, before)
        assert np.array_equal(env.target_position.numpy(), [164., 170.])
        p = Planner(model, root / "configs/tworoom-scalers.json", 4199, counters)
        proposed, actions = p.plan(image, target)
        assert actions.shape == (5, 2) and np.isfinite(actions).all()
        assert (np.abs(actions) <= 1).all()
        assert np.allclose(proposed, actions, atol=1e-6)
        for action in [np.array([0., 1.]), np.zeros(2)]:
            counters["environment_actions"] += 1
            _, reward, terminated, _, _ = env.step(action)
            assert reward == 0 and not terminated
        assert np.array_equal(env.agent_position.numpy(), [164., 95.])
        result["passed"] = True
    finally:
        env.close()
        result["resources"] = dict(counters)
        out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
