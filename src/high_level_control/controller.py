from __future__ import annotations

import numpy as np

from .config import ControllerConfig
from .math_utils import wrap_to_pi
from .model import HighLevelKinematicModel
from .trajectories import ReferenceState


class FeedbackLinearizationController:
    def __init__(self, model: HighLevelKinematicModel, cfg: ControllerConfig) -> None:
        self.model = model
        self.cfg = cfg
        self.K = np.diag([cfg.kp_x, cfg.kp_y, cfg.kp_yaw]).astype(np.float64)

    def compute(self, pose: np.ndarray, ref: ReferenceState) -> np.ndarray:
        error = np.asarray(ref.pose - pose, dtype=np.float64)
        error[2] = wrap_to_pi(float(error[2]))

        desired_pose_rate = np.asarray(ref.velocity, dtype=np.float64) + self.K @ error
        B = self.model.input_matrix(float(pose[2]))
        command = np.linalg.solve(B, desired_pose_rate)

        command[0] = np.clip(command[0], self.cfg.min_vx, self.cfg.max_vx)
        command[1] = np.clip(command[1], self.cfg.min_vy, self.cfg.max_vy)
        command[2] = np.clip(command[2], self.cfg.min_wz, self.cfg.max_wz)
        return command
