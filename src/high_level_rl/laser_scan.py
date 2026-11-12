from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .raycast_patterns import horizontal_ray_angles


@dataclass
class LaserScanConfig:
    num_rays: int = 121
    fov_deg: float = 360.0
    max_range: float = 5.0
    min_range: float = 0.05
    robot_radius: float = 0.30
    obstacle_inflation: float = 0.00
    noise_std: float = 0.0
    dropout_prob: float = 0.0


def make_ray_angles(cfg: LaserScanConfig, device: torch.device) -> torch.Tensor:
    return horizontal_ray_angles(cfg.num_rays, cfg.fov_deg, device=device)


def ray_circle_intersections(
    ray_origins: torch.Tensor,
    ray_dirs: torch.Tensor,
    centers: torch.Tensor,
    radii: torch.Tensor,
    max_range: float,
) -> torch.Tensor:
    """Batched 2D ray-circle first-hit distances.

    Args:
        ray_origins: [B, 2]
        ray_dirs: [R, 2], body-frame ray directions already rotated into world if needed
        centers: [B, N, 2]
        radii: [B, N]
        max_range: scalar

    Returns:
        ranges: [B, R]
    """
    B = ray_origins.shape[0]
    R = ray_dirs.shape[0]
    N = centers.shape[1]

    o = ray_origins[:, None, None, :]          # [B,1,1,2]
    d = ray_dirs[None, :, None, :]             # [1,R,1,2]
    c = centers[:, None, :, :]                 # [B,1,N,2]
    rr = radii[:, None, :]                     # [B,1,N]

    oc = o - c
    b = 2.0 * (oc * d).sum(dim=-1)             # [B,R,N]
    cc = (oc * oc).sum(dim=-1) - rr * rr       # [B,R,N]
    disc = b * b - 4.0 * cc

    valid_disc = disc >= 0.0
    sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))

    t1 = (-b - sqrt_disc) * 0.5
    t2 = (-b + sqrt_disc) * 0.5

    inf = torch.full_like(t1, float("inf"))
    t_candidates = torch.where((t1 > 0.0) & valid_disc, t1, inf)
    t_candidates = torch.minimum(
        t_candidates,
        torch.where((t2 > 0.0) & valid_disc, t2, inf),
    )

    t_min = t_candidates.min(dim=-1).values    # [B,R]
    ranges = torch.clamp(t_min, min=0.0, max=max_range)
    ranges = torch.where(torch.isfinite(ranges), ranges, torch.full_like(ranges, max_range))
    return ranges


def ray_aabb_intersections(
    ray_origins: torch.Tensor,
    ray_dirs: torch.Tensor,
    xy_min: Tuple[float, float],
    xy_max: Tuple[float, float],
    max_range: float,
) -> torch.Tensor:
    """Distance to rectangular arena boundary from inside the rectangle.

    Args:
        ray_origins: [B,2]
        ray_dirs: [R,2]
    Returns:
        ranges: [B,R]
    """
    device = ray_origins.device
    B = ray_origins.shape[0]
    R = ray_dirs.shape[0]

    o = ray_origins[:, None, :]       # [B,1,2]
    d = ray_dirs[None, :, :]          # [1,R,2]

    xmin, ymin = xy_min
    xmax, ymax = xy_max

    eps = 1e-8
    tx1 = (torch.tensor(xmin, device=device) - o[..., 0]) / torch.where(d[..., 0].abs() > eps, d[..., 0], torch.full_like(d[..., 0], eps))
    tx2 = (torch.tensor(xmax, device=device) - o[..., 0]) / torch.where(d[..., 0].abs() > eps, d[..., 0], torch.full_like(d[..., 0], eps))
    ty1 = (torch.tensor(ymin, device=device) - o[..., 1]) / torch.where(d[..., 1].abs() > eps, d[..., 1], torch.full_like(d[..., 1], eps))
    ty2 = (torch.tensor(ymax, device=device) - o[..., 1]) / torch.where(d[..., 1].abs() > eps, d[..., 1], torch.full_like(d[..., 1], eps))

    candidates = torch.stack([tx1.expand(B, R), tx2.expand(B, R), ty1.expand(B, R), ty2.expand(B, R)], dim=-1)
    candidates = torch.where(candidates > 0.0, candidates, torch.full_like(candidates, float("inf")))

    t = candidates.min(dim=-1).values
    ranges = torch.where(torch.isfinite(t), t, torch.full_like(t, max_range))
    return torch.clamp(ranges, min=0.0, max=max_range)


def simulate_planar_laser_scan(
    robot_xy: torch.Tensor,
    robot_yaw: torch.Tensor,
    obstacle_centers: torch.Tensor,
    obstacle_radii: torch.Tensor,
    obstacle_mask: Optional[torch.Tensor],
    cfg: LaserScanConfig,
    arena_xy_min: Optional[Tuple[float, float]] = None,
    arena_xy_max: Optional[Tuple[float, float]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate a 2D planar laser scan from circular obstacles and optional rectangular arena.

    Args:
        robot_xy: [B,2]
        robot_yaw: [B]
        obstacle_centers: [B,N,2]
        obstacle_radii: [B,N]
        obstacle_mask: [B,N] bool, True means obstacle exists
        cfg: laser scan config
        arena_xy_min/max: optional rectangular wall limits

    Returns:
        ranges: [B,R]
        angles_body: [R]
    """
    device = robot_xy.device
    angles = make_ray_angles(cfg, device=device)

    ray_body = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)  # [R,2]

    c = torch.cos(robot_yaw)
    s = torch.sin(robot_yaw)

    dx = ray_body[:, 0][None, :]
    dy = ray_body[:, 1][None, :]

    ray_world = torch.stack(
        [
            c[:, None] * dx - s[:, None] * dy,
            s[:, None] * dx + c[:, None] * dy,
        ],
        dim=-1,
    )  # [B,R,2]

    B, R, _ = ray_world.shape

    # ray_circle_intersections expects one ray direction set. Handle B by flattening.
    ranges = torch.full((B, R), cfg.max_range, device=device)

    inflated_radii = obstacle_radii + cfg.robot_radius + cfg.obstacle_inflation
    if obstacle_mask is not None:
        inflated_radii = torch.where(obstacle_mask, inflated_radii, torch.zeros_like(inflated_radii))

    for b in range(B):
        circle_ranges = ray_circle_intersections(
            robot_xy[b:b+1],
            ray_world[b],
            obstacle_centers[b:b+1],
            inflated_radii[b:b+1],
            cfg.max_range,
        )[0]
        ranges[b] = torch.minimum(ranges[b], circle_ranges)

    if arena_xy_min is not None and arena_xy_max is not None:
        for b in range(B):
            wall_ranges = ray_aabb_intersections(
                robot_xy[b:b+1],
                ray_world[b],
                arena_xy_min,
                arena_xy_max,
                cfg.max_range,
            )[0]
            ranges[b] = torch.minimum(ranges[b], wall_ranges)

    ranges = torch.clamp(ranges, min=cfg.min_range, max=cfg.max_range)

    if cfg.noise_std > 0.0:
        ranges = ranges + cfg.noise_std * torch.randn_like(ranges)
        ranges = torch.clamp(ranges, min=cfg.min_range, max=cfg.max_range)

    if cfg.dropout_prob > 0.0:
        keep = torch.rand_like(ranges) > cfg.dropout_prob
        ranges = torch.where(keep, ranges, torch.full_like(ranges, cfg.max_range))

    return ranges, angles
