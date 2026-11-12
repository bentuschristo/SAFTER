# NOTE: this is a DUPLICATE of src/high_level_rl/lse_safety_filter.py, kept in sync manually.
# It exists here because src/high_level_rl/lse_safety_filter.py is also load-bearing for
# map-based code (high_level_sac/high_level_bcbf_sac/high_level_cbf_rl) that
# intentionally stays outside SAFTER_src. See SAFTER_src/README.md.

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import ArenaConfig, SafetyConfig


@dataclass
class SafetyFilterResult:
    command: torch.Tensor
    h_lse: torch.Tensor
    individual_h: torch.Tensor
    active: torch.Tensor
    correction_norm: torch.Tensor
    cbf_value_nominal: torch.Tensor


class LseCbfSafetyFilter:
    """Closed-form LSE-CBF for circular obstacles and rectangular walls.

    All primitive barriers are signed distances in meters, so their LSE
    aggregation is dimensionally consistent.
    """

    def __init__(self, arena: ArenaConfig, cfg: SafetyConfig) -> None:
        self.arena = arena
        self.cfg = cfg

    def filter(
        self,
        position_w: torch.Tensor,
        yaw: torch.Tensor,
        nominal_command: torch.Tensor,
        obstacle_centers: torch.Tensor,
        obstacle_radii: torch.Tensor,
        obstacle_mask: torch.Tensor,
    ) -> SafetyFilterResult:
        e = position_w.shape[0]
        eps = self.cfg.denominator_epsilon

        rel = position_w[:, None, :] - obstacle_centers
        dist = torch.sqrt(torch.sum(rel * rel, dim=-1) + eps)
        effective_obstacle_radius = (
            obstacle_radii + self.arena.robot_radius
        )
        h_obs = dist - effective_obstacle_radius
        grad_obs = rel / dist[..., None]

        # Inactive padded obstacles receive a very large barrier and zero gradient.
        h_obs = torch.where(obstacle_mask, h_obs, torch.full_like(h_obs, 1.0e3))
        grad_obs = torch.where(obstacle_mask[..., None], grad_obs, torch.zeros_like(grad_obs))

        x, y = position_w.unbind(dim=-1)
        h_walls = torch.stack(
            (
                x - self.arena.x_min
                - self.arena.robot_radius,
                self.arena.x_max - x
                - self.arena.robot_radius,
                y - self.arena.y_min
                - self.arena.robot_radius,
                self.arena.y_max - y
                - self.arena.robot_radius,
            ),
            dim=-1,
        )
        grad_walls = torch.tensor(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
            dtype=position_w.dtype,
            device=position_w.device,
        ).expand(e, -1, -1)

        # Obstacles and all four walls are primitive barriers in the same
        # soft-min/LSE aggregation.
        h_all = torch.cat((h_obs, h_walls), dim=-1)
        grad_all = torch.cat((grad_obs, grad_walls), dim=1)

        scaled = -self.cfg.kappa * h_all
        h_lse = -torch.logsumexp(scaled, dim=-1) / self.cfg.kappa
        weights = torch.softmax(scaled, dim=-1)
        grad = torch.sum(weights[..., None] * grad_all, dim=1)

        c = torch.cos(yaw)
        s = torch.sin(yaw)
        # World velocity = R(yaw) [vx_body, vy_body]. wz has no direct effect.
        a_vx = grad[:, 0] * c + grad[:, 1] * s
        a_vy = -grad[:, 0] * s + grad[:, 1] * c
        a = torch.stack((a_vx, a_vy, torch.zeros_like(a_vx)), dim=-1)

        cbf_nom = torch.sum(a * nominal_command, dim=-1) + self.cfg.alpha * h_lse
        violation = torch.relu(-cbf_nom)
        denom = torch.sum(a * a, dim=-1) + eps
        correction = (violation / denom)[:, None] * a
        command = nominal_command + correction

        return SafetyFilterResult(
            command=command,
            h_lse=h_lse,
            individual_h=h_all,
            active=violation > 0.0,
            correction_norm=torch.linalg.norm(correction, dim=-1),
            cbf_value_nominal=cbf_nom,
        )
