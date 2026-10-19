"""Explicit device and endpoint assignments on the authorized two-GPU host."""

GPU_UUIDS = ["GPU-0bfb3ed5-af55-472e-3199-058727ff80a7",
             "GPU-4f452774-6f8b-6fcc-4ce1-2d4d190e6d07"]


def assignment(worker=None):
    if worker is None:
        return {"simulation_device": 0, "policy_device": 1, "port": 18000}
    if type(worker) is not int or worker not in [0, 1]:
        raise ValueError("A declared worker must be 0 or 1")
    return {"simulation_device": worker, "policy_device": worker, "port": 18000 + worker}
