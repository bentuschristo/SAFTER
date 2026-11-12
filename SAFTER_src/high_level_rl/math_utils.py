# NOTE: this is a DUPLICATE of src/high_level_rl/math_utils.py, kept in sync manually.
# It exists here because src/high_level_rl/math_utils.py is also load-bearing for
# map-based code (high_level_sac/high_level_bcbf_sac/high_level_cbf_rl) that
# intentionally stays outside SAFTER_src. See SAFTER_src/README.md.

from __future__ import annotations

import torch


def quat_wxyz_to_rpy(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert batched [w,x,y,z] quaternions to roll, pitch, yaw."""
    w, x, y, z = quat.unbind(dim=-1)
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr, cosr)

    sinp = 2.0 * (w * y - z * x)
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny, cosy)
    return roll, pitch, yaw


def world_to_body(v_world: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    x = c * v_world[..., 0] + s * v_world[..., 1]
    y = -s * v_world[..., 0] + c * v_world[..., 1]
    return torch.stack((x, y), dim=-1)


def body_to_world(v_body: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    x = c * v_body[..., 0] - s * v_body[..., 1]
    y = s * v_body[..., 0] + c * v_body[..., 1]
    return torch.stack((x, y), dim=-1)
