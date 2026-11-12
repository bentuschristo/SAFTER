from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.distributions import Normal

from src.high_level_rl.config import ArenaConfig, BackupSafetyConfig, HighLevelActionConfig
from src.high_level_rl.bcbf_lse_cf_filter import (
    BackupSafetyFilterResult,
    ZeroBackupLseCbfClosedForm,
)
from src.high_level_rl.cbf_cd_lse_cf_filter import LseCbfControlDynamicsClosedForm


LOG_2PI = 1.8378770664093453


def mlp(input_dim: int, hidden_dims: Iterable[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
        layers.extend((nn.Linear(previous, width), nn.ELU()))
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


@dataclass
class SafeActorOutput:
    nominal_normalized: torch.Tensor
    nominal_command: torch.Tensor
    safe_normalized: torch.Tensor
    safe_command: torch.Tensor
    log_prob: torch.Tensor
    safety: BackupSafetyFilterResult


class SafeSacActor(nn.Module):
    """Stochastic actor whose final action layer is the analytical LSE-CBF-CF map.

    The critic is evaluated at ``safe_normalized``. Therefore its gradient
    propagates through the closed-form safety correction into the nominal MLP.

    The entropy term uses the tanh-Gaussian nominal density. An exact density of
    the projected safe action is generally unavailable because the ReLU-based
    CBF projection can be non-invertible. This is the practical interpretation
    commonly used for this architecture and should be documented explicitly.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dims: tuple[int, ...],
        arena: ArenaConfig,
        action_cfg: HighLevelActionConfig,
        safety_cfg: BackupSafetyConfig,
        obstacle_centers: torch.Tensor,
        obstacle_radii: torch.Tensor,
        obstacle_mask: torch.Tensor,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        filter_kind: str = "bcbf_cf",
    ) -> None:
        super().__init__()
        self.backbone = mlp(obs_dim, hidden_dims, 2 * 3)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.filter_kind = str(filter_kind)
        if self.filter_kind == "bcbf_cf":
            self.safety_filter = ZeroBackupLseCbfClosedForm(arena, action_cfg, safety_cfg)
        elif self.filter_kind == "cbf_cd_cf":
            self.safety_filter = LseCbfControlDynamicsClosedForm(arena, action_cfg, safety_cfg)
        else:
            raise ValueError(f"Unknown filter_kind={filter_kind!r}")
        # Affine map from tanh-normalized [-1, 1] to the (possibly asymmetric,
        # e.g. vx in [min_vx, max_vx]) physical command range. vy/wz stay
        # symmetric (center 0), matching HighLevelActionConfig.
        action_low = torch.tensor(
            [action_cfg.min_vx, -action_cfg.max_vy, -action_cfg.max_wz], dtype=torch.float32
        )
        action_high = torch.tensor(
            [action_cfg.max_vx, action_cfg.max_vy, action_cfg.max_wz], dtype=torch.float32
        )
        self.register_buffer("action_center", (action_low + action_high) / 2.0)
        self.register_buffer("action_half_range", (action_high - action_low) / 2.0)
        # Fixed arena: one obstacle map shared by all vectorized environments.
        self.register_buffer("obstacle_centers", obstacle_centers.detach().clone())
        self.register_buffer("obstacle_radii", obstacle_radii.detach().clone())
        self.register_buffer("obstacle_mask", obstacle_mask.detach().clone())

    def _distribution(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, Normal]:
        mean, log_std = self.backbone(obs).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        return mean, log_std, Normal(mean, std)

    def _expand_obstacles(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        centers = self.obstacle_centers
        radii = self.obstacle_radii
        mask = self.obstacle_mask
        if centers.ndim == 2:
            centers = centers.unsqueeze(0)
            radii = radii.unsqueeze(0)
            mask = mask.unsqueeze(0)
        if centers.shape[0] == 1 and batch_size != 1:
            centers = centers.expand(batch_size, -1, -1)
            radii = radii.expand(batch_size, -1)
            mask = mask.expand(batch_size, -1)
        elif centers.shape[0] != batch_size:
            centers = centers[:1].expand(batch_size, -1, -1)
            radii = radii[:1].expand(batch_size, -1)
            mask = mask[:1].expand(batch_size, -1)
        return centers, radii, mask

    def apply_safety(
        self,
        nominal_normalized: torch.Tensor,
        position: torch.Tensor,
        yaw: torch.Tensor,
        current_command: torch.Tensor,
        previous_nominal_command: torch.Tensor,
        nominal_history_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, BackupSafetyFilterResult]:
        nominal_command = self.action_center + nominal_normalized * self.action_half_range
        centers, radii, mask = self._expand_obstacles(nominal_command.shape[0])
        safety = self.safety_filter.filter(
            position,
            yaw,
            current_command,
            nominal_command,
            previous_nominal_command,
            nominal_history_valid,
            centers,
            radii,
            mask,
        )

        # No post-filter clipping: the original command limits are enforced
        # as augmented-state CBF constraints on the command state u.
        safe_command = safety.command
        safe_normalized = (safe_command - self.action_center) / self.action_half_range
        return safe_normalized, safe_command, safety

    def sample(self, obs: torch.Tensor, position: torch.Tensor, yaw: torch.Tensor) -> SafeActorOutput:
        mean, _, dist = self._distribution(obs)
        pre_tanh = dist.rsample()
        nominal_normalized = torch.tanh(pre_tanh)

        # Standard tanh change-of-variables for the nominal stochastic policy.
        log_prob = dist.log_prob(pre_tanh) - torch.log(
            1.0 - nominal_normalized.pow(2) + 1.0e-6
        )
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        safe_normalized, safe_command, safety = self.apply_safety(
            nominal_normalized,
            position,
            yaw,
            obs[:, 5:8],
            obs[:, -4:-1],
            obs[:, -1:],
        )
        return SafeActorOutput(
            nominal_normalized=nominal_normalized,
            nominal_command=self.action_center + nominal_normalized * self.action_half_range,
            safe_normalized=safe_normalized,
            safe_command=safe_command,
            log_prob=log_prob,
            safety=safety,
        )

    @torch.no_grad()
    def deterministic(self, obs: torch.Tensor, position: torch.Tensor, yaw: torch.Tensor) -> SafeActorOutput:
        mean, _, _ = self._distribution(obs)
        nominal_normalized = torch.tanh(mean)
        safe_normalized, safe_command, safety = self.apply_safety(
            nominal_normalized,
            position,
            yaw,
            obs[:, 5:8],
            obs[:, -4:-1],
            obs[:, -1:],
        )
        return SafeActorOutput(
            nominal_normalized=nominal_normalized,
            nominal_command=self.action_center + nominal_normalized * self.action_half_range,
            safe_normalized=safe_normalized,
            safe_command=safe_command,
            log_prob=torch.zeros((obs.shape[0], 1), device=obs.device),
            safety=safety,
        )


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        self.net = mlp(obs_dim + action_dim, hidden_dims, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((obs, action), dim=-1))


class TwinQCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        self.q1 = QNetwork(obs_dim, action_dim, hidden_dims)
        self.q2 = QNetwork(obs_dim, action_dim, hidden_dims)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(obs, action), self.q2(obs, action)
