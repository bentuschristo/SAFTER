from __future__ import annotations

import math
import numpy as np


def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quat_wxyz(quat: np.ndarray) -> float:
    """Return yaw from quaternion [w, x, y, z]."""
    w, x, y, z = (float(v) for v in quat)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)
