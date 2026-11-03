"""Counted AP inference/geometry preflight; run only on the authorised pod."""

import os
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anchored_pipeline as pipeline  # noqa: E402


def main():
    assert os.environ.get("RUNPOD_POD_ID") == "yvrg8ow7i9n9u6"
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    torch.set_num_threads(4)
    out = pipeline.BASE / "ap-preflight.json"
    assert not out.exists()
    report = dict(environment_resets=0, environment_actions=0, candidate_model_steps=0,
                  encoder_frames=0, passed=False, source_sha256=pipeline.digest(Path(__file__)))
    pipeline.save(out, report)
    _, native = pipeline.native_model()

    def hook(_module, args):
        report["encoder_frames"] += int(args[0].shape[0])

    handle = native.model.encoder.register_forward_pre_hook(hook)
    env = TwoRoomEnv()
    try:
        report["environment_resets"] += 1
        pipeline.save(out, report)
        env.reset(seed=5101, options={"state": np.array([164.,90.]),
                                     "target_state": np.array([164.,170.])})
        np.testing.assert_array_equal(env.agent_position.numpy(), [164.,90.])
        rgb = env.render()
        goal = env._target_img.permute(1, 2, 0).numpy()
        with torch.inference_mode():
            current, target = native.encode_rgb(rgb), native.encode_rgb(goal)
            actions = torch.randn(1, 8, 1, 10, device="cuda",
                                  generator=torch.Generator(device="cuda").manual_seed(5101))
            pixels = native.pixels(rgb)[:, None].expand(1,8,-1,-1,-1,-1)
            info = dict(pixels=pixels, goal=native.pixels(goal)[:, None].expand(1,8,-1,-1,-1,-1),
                        action=torch.zeros(1,8,1,10,device="cuda"))
            report["candidate_model_steps"] += 8
            expected = native.model.get_cost(info, actions)
            cached = dict(pixels=pixels, current_latent=current[:,None,None,:].expand(1,8,1,-1),
                          subgoal_latent=target[:,None,:].expand(1,8,-1))
            report["candidate_model_steps"] += 8
            actual = native.latent_cost.get_cost(cached, actions)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-5)
            report["maximum_cost_difference"] = float((actual-expected).abs().max())
        report["environment_actions"] += 1
        pipeline.save(out, report)
        env.step(np.array([0., 1.], dtype=np.float32))
        np.testing.assert_array_equal(env.agent_position.numpy(), [164.,95.])
        for x in [156,164,172]:
            points = [[x,90],[130,90],[130,170],[x,170]]
            assert all(pipeline.segment_distance(a,b,[x,130]) > 18
                       for a,b in zip(points[:-1], points[1:]))
            assert pipeline.segment_distance([x,90],[x,170],[x,130]) == 0
        assert pipeline.segment_distance([0,18],[10,18],[5,0]) == 18
        report["passed"] = True
    finally:
        handle.remove()
        env.close()
        pipeline.save(out, report)
    print(report)


if __name__ == "__main__":
    main()
