"""Small adapters preserving UNISafe observation semantics on Isaac Lab 3."""


def camera_rgb(env, sensor_cfg, data_type="rgb", normalize=False):
    if data_type != "rgb" or normalize:
        raise ValueError("This adapter is for UNISafe's unnormalized RGB observations only.")
    sensor = env.scene.sensors[sensor_cfg.name]
    if hasattr(env, "episode_length_buf") and bool((env.episode_length_buf == 0).any()):
        # On this headless RTX stack, render() alone does not capture a camera.
        # Flush two actual captures after reset, with no physics integration.
        env.sim.forward()
        for _ in range(2):
            env.sim.render()
            sensor.update(0.0, force_recompute=True)
    output = sensor.data.output[data_type]
    # Isaac Lab 3 wraps camera arrays in a backend-neutral tensor container.
    return output.torch[..., :3]


def normalized_to_environment(action):
    """UNISafe NormalizeActions followed by DreamerVecEnvWrapper's clipping.

    Six relative Cartesian pose commands followed by a binary gripper command.
    The environment's IK action term subsequently multiplies the six arm values by 0.5.
    """
    return (action * 0.15).clamp(-0.15, 0.15)
