from __future__ import annotations

from typing import Any
import numpy as np
import torch

from .math_utils import yaw_from_quat_wxyz
from .model import HighLevelState


def _to_numpy_first(value: Any, env_index: int) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.ndim >= 2:
        arr = arr[env_index]
    return np.asarray(arr, dtype=np.float64)


def _first_attr(obj: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(f"None of {names} exists on {type(obj).__name__}")


class MjlabHighLevelStateInterface:
    """Read x-y-yaw and planar body velocity from mjlab's robot entity.

    The adapter tries both current and older mjlab data-field names. If your local
    version differs, print dir(env.unwrapped.scene[\"robot\"].data) and add the
    corresponding names below.
    """

    def __init__(self, env: Any, robot_name: str = "robot", env_index: int = 0) -> None:
        self.env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.robot_name = robot_name
        self.env_index = env_index

    def _robot(self) -> Any:
        scene = self.env.scene
        try:
            return scene[self.robot_name]
        except Exception:
            if hasattr(scene, "entities"):
                return scene.entities[self.robot_name]
            raise

    def read(self) -> HighLevelState:
        data = self._robot().data
        pos = _to_numpy_first(
            _first_attr(data, ("root_pos_w", "root_link_pos_w", "base_pos_w")),
            self.env_index,
        )
        quat = _to_numpy_first(
            _first_attr(data, ("root_quat_w", "root_link_quat_w", "base_quat_w")),
            self.env_index,
        )

        # unitree_rl_mjlab/mjlab uses scalar-first quaternions in the exposed state buffers.
        yaw = yaw_from_quat_wxyz(quat[:4])

        try:
            lin_b = _to_numpy_first(
                _first_attr(data, ("root_lin_vel_b", "root_link_lin_vel_b", "base_lin_vel_b")),
                self.env_index,
            )
        except AttributeError:
            lin_b = np.zeros(3, dtype=np.float64)

        try:
            ang_b = _to_numpy_first(
                _first_attr(data, ("root_ang_vel_b", "root_link_ang_vel_b", "base_ang_vel_b")),
                self.env_index,
            )
        except AttributeError:
            ang_b = np.zeros(3, dtype=np.float64)

        return HighLevelState(
            base_pose_world=np.array([pos[0], pos[1], yaw], dtype=np.float64),
            base_velocity_body=np.array([lin_b[0], lin_b[1], ang_b[2]], dtype=np.float64),
        )
