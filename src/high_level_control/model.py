from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .config import ModelConfig


@dataclass(frozen=True)
class HighLevelState:
    base_pose_world: np.ndarray       # [x_c, y_c, yaw]
    base_velocity_body: np.ndarray    # [v_x, v_y, omega_z]


class HighLevelKinematicModel:
    """Planar high-level model used in the previous model-based controller.

    Xdot = B(theta) U,
    X = [x, y, theta], U = [vx_body, vy_body, omega].
    The controlled point may be offset from the base origin by (c_x,c_y).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg

    def controlled_pose(self, base_pose_world: np.ndarray) -> np.ndarray:
        xc, yc, theta = (float(v) for v in base_pose_world)
        c, s = math.cos(theta), math.sin(theta)
        px, py = self.cfg.controlled_point_x, self.cfg.controlled_point_y
        return np.array(
            [xc + c * px - s * py, yc + s * px + c * py, theta],
            dtype=np.float64,
        )

    def input_matrix(self, theta: float) -> np.ndarray:
        c, s = math.cos(theta), math.sin(theta)
        cx, cy = self.cfg.controlled_point_x, self.cfg.controlled_point_y
        return np.array(
            [
                [c, -s, -cx * s - cy * c],
                [s,  c,  cx * c - cy * s],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def world_planar_velocity(self, pose: np.ndarray, command_body: np.ndarray) -> np.ndarray:
        return self.input_matrix(float(pose[2])) @ command_body
