"""Franka Panda forward kinematics for predicting where a joint-target chunk moves the hand.

The standard Panda modified-DH chain plus the 0.107 m flange and a 0.105 m offset to
OmniGibson's long-finger eef link reproduces recorded `robot.get_eef_position()` to
below a micrometre once each scene's base pose is fitted (task 15 design evidence).
Live prediction avoids the base position altogether: it adds the base-frame
displacement between the current joints and each target, rotated by the base yaw, to
the measured hand position.
"""

import numpy as np

DH = (
    (0.0, 0.333, 0.0),
    (0.0, 0.0, -np.pi / 2),
    (0.0, 0.316, np.pi / 2),
    (0.0825, 0.0, np.pi / 2),
    (-0.0825, 0.384, -np.pi / 2),
    (0.0, 0.0, np.pi / 2),
    (0.088, 0.0, np.pi / 2),
)
FLANGE_D = 0.107
EEF_OFFSET = np.array([0.0, 0.0, 0.105])


def flange(q):
    """Base-to-flange transforms for joint arrays shaped (..., 7)."""
    q = np.asarray(q, dtype=np.float64)
    shape = q.shape[:-1]
    q = q.reshape(-1, 7)
    n = q.shape[0]
    transform = np.tile(np.eye(4), (n, 1, 1))
    for i, (a, d, alpha) in enumerate(DH):
        ca, sa = np.cos(alpha), np.sin(alpha)
        ct, st = np.cos(q[:, i]), np.sin(q[:, i])
        link = np.zeros((n, 4, 4))
        link[:, 0, 0], link[:, 0, 1], link[:, 0, 3] = ct, -st, a
        link[:, 1, 0], link[:, 1, 1], link[:, 1, 2], link[:, 1, 3] = st * ca, ct * ca, -sa, -sa * d
        link[:, 2, 0], link[:, 2, 1], link[:, 2, 2], link[:, 2, 3] = st * sa, ct * sa, ca, ca * d
        link[:, 3, 3] = 1.0
        transform = transform @ link
    tip = np.eye(4)
    tip[2, 3] = FLANGE_D
    return (transform @ tip).reshape(*shape, 4, 4)


def eef_local(q):
    """Eef-link positions in the robot base frame, shaped like q[..., :3]."""
    frame = flange(q)
    return frame[..., :3, :3] @ EEF_OFFSET + frame[..., :3, 3]


def yaw_matrix(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def yaw_from_quaternion(xyzw):
    """Heading of a base orientation given as an (x, y, z, w) quaternion."""
    x, y, z, w = (float(v) for v in xyzw)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def fit_base(q, eef):
    """Least-squares base yaw and position from recorded joints and eef positions."""
    from scipy.optimize import least_squares

    local = eef_local(np.asarray(q, dtype=np.float64)[:, :7])
    eef = np.asarray(eef, dtype=np.float64)

    def residual(x):
        return ((local @ yaw_matrix(x[0]).T) + x[1:] - eef).ravel()

    best = None
    for start in (-np.pi / 2, 0.0, np.pi / 2, np.pi):
        result = least_squares(residual, np.array([start, 0.0, 0.0, 0.6]), method="lm")
        if best is None or result.cost < best.cost:
            best = result
    error = np.linalg.norm(residual(best.x).reshape(-1, 3), axis=1)
    return float(best.x[0]), best.x[1:].copy(), float(error.max())


def predict_path(q_now, eef_now, targets, yaw):
    """World eef positions after each joint target, assuming the targets are tracked.

    q_now is the current 7-joint arm position, eef_now the measured world eef
    position, targets an (n, 7) array of absolute joint targets and yaw the robot
    base heading. Pre-contact tracking on this plant is tight (task 4), so the
    commanded path is the prediction; contact can make the realized path differ.
    """
    q_now = np.asarray(q_now, dtype=np.float64)[:7]
    targets = np.asarray(targets, dtype=np.float64)[:, :7]
    base_now = eef_local(q_now)
    displacement = eef_local(targets) - base_now
    return np.asarray(eef_now, dtype=np.float64) + displacement @ yaw_matrix(yaw).T
