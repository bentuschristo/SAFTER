# NOTE: this is a DUPLICATE of src/high_level_rl/raycast_patterns.py, kept
# in sync manually -- also load-bearing for map-based scripts that stay
# outside SAFTER_src. See SAFTER_src/README.md.

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import torch


def horizontal_ray_angles(
    num_rays: int,
    fov_deg: float,
    *,
    yaw_offset_deg: float = 0.0,
    left_to_right: bool = False,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate ray angles without duplicating the full-circle seam.

    Partial fields of view include both edge rays. A 360-degree field is
    periodic, so it uses N equal sectors and omits the repeated endpoint.
    """
    n = int(num_rays)
    if n <= 0:
        raise ValueError("num_rays must be positive")
    fov = math.radians(float(fov_deg))
    if fov <= 0.0:
        raise ValueError("fov_deg must be positive")
    yaw0 = math.radians(float(yaw_offset_deg))
    if n == 1:
        return torch.zeros(1, device=device, dtype=dtype) + yaw0

    if math.isclose(fov, 2.0 * math.pi, rel_tol=0.0, abs_tol=1.0e-7):
        direction = -1.0 if left_to_right else 1.0
        start = 0.5 * fov if left_to_right else -0.5 * fov
        indices = torch.arange(n, device=device, dtype=dtype)
        angles = start + direction * indices * (fov / n)
    else:
        start = -0.5 * fov
        end = 0.5 * fov
        if left_to_right:
            start, end = end, start
        angles = torch.linspace(start, end, n, device=device, dtype=dtype)
    return angles + yaw0


@dataclass
class HorizontalRingPatternCfg:
    """Horizontal planar LiDAR ray pattern.

    local_offsets: all rays start from the same LiDAR origin in the attached frame.
    local_directions: rays sweep in the local XY plane.
    """

    num_rays: int = 121
    fov_deg: float = 360.0

    # LiDAR origin relative to attached frame, usually base_link.
    origin_x: float = 0.28945
    origin_y: float = 0.00
    origin_z: float = 0.25

    # Yaw offset of LiDAR frame relative to attached frame.
    yaw_offset_deg: float = 0.0

    # If true, rays sweep from robot-left to robot-right for a front-facing scan.
    left_to_right: bool = False

    def generate_rays(
        self,
        mj_model: mujoco.MjModel | None,
        device: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del mj_model

        n = int(self.num_rays)
        angles = horizontal_ray_angles(
            n,
            self.fov_deg,
            yaw_offset_deg=self.yaw_offset_deg,
            left_to_right=self.left_to_right,
            device=device,
        )

        local_offsets = torch.zeros((n, 3), device=device, dtype=torch.float32)
        local_offsets[:, 0] = float(self.origin_x)
        local_offsets[:, 1] = float(self.origin_y)
        local_offsets[:, 2] = float(self.origin_z)

        local_directions = torch.zeros((n, 3), device=device, dtype=torch.float32)
        local_directions[:, 0] = torch.cos(angles)
        local_directions[:, 1] = torch.sin(angles)
        local_directions[:, 2] = 0.0

        local_directions = local_directions / local_directions.norm(dim=-1, keepdim=True)
        return local_offsets, local_directions
