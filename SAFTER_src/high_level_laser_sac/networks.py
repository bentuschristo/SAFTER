from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.distributions import Normal

from SAFTER_src.high_level_rl.laser_cbf_filters import (
    LaserSafetyFilterResult,
    LaserScanCbfClosedForm,
    LaserScanCbfLseQp,
    LaserScanCbfQp,
    LaserScanCbfControlDynamicsClosedForm,
    LaserScanCbfLseControlDynamicsQp,
    LaserScanCbfLseControlDynamicsQpCfMatched,
    LaserZeroBackupCbfQp,
    LaserZeroBackupNonLseControlDynamicsQp,
    LaserZeroBackupLseControlDynamicsQp,
    LaserZeroBackupLseControlDynamicsQpCfMatched,
    LaserZeroBackupLseCbfClosedForm,
    LaserZeroBackupLseQp,
)

from .config import LaserSacMethod, LaserSafetyConfig


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
    safety: LaserSafetyFilterResult


class LaserSafeSacActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_dims: tuple[int, ...],
        action_limits: tuple[float, float, float],
        ray_angles: torch.Tensor,
        range_start: int,
        num_rays: int,
        max_range: float,
        safety_cfg: LaserSafetyConfig,
        method: LaserSacMethod,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        min_vx: float | None = None,
    ) -> None:
        super().__init__()
        self.backbone = mlp(obs_dim, hidden_dims, 6)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.method = method
        # Eval/deployment-only escape hatch: when True, apply_safety() still
        # runs the filter (so filter_active/correction metrics keep logging
        # normally) but executes the RAW nominal command instead of the
        # filtered one -- lets an eval script measure a trained checkpoint's
        # unfiltered collision rate without touching training, which always
        # leaves this at its default False.
        self.disable_safety_filter = False
        self.range_start = int(range_start)
        self.num_rays = int(num_rays)
        self.max_range = float(max_range)
        self.hard_clip_output = bool(safety_cfg.hard_clip_output)
        self.register_buffer("ray_angles", ray_angles.detach().clone().float())
        # This actor's own nominal/hard-clip command range, AND (as of the
        # min_vx passthrough into kwargs below) the safety filter's own
        # admissible-input box too -- the filter's formal safety proof must
        # bound what the frozen low-level locomotion policy can actually
        # track, not a wider/narrower box of the filter's own invention; a
        # "safe" command the low-level policy never learned to execute is
        # not actually safe in practice. Defaults to the archived symmetric
        # [-action_limits[0], action_limits[0]] range on vx when min_vx is
        # not given, so every existing caller/checkpoint is unaffected.
        effective_min_vx = -float(action_limits[0]) if min_vx is None else float(min_vx)
        action_low = torch.tensor(
            (effective_min_vx, -float(action_limits[1]), -float(action_limits[2])),
            dtype=torch.float32,
        )
        action_high = torch.tensor(
            (float(action_limits[0]), float(action_limits[1]), float(action_limits[2])),
            dtype=torch.float32,
        )
        self.register_buffer("action_low", action_low)
        self.register_buffer("action_high", action_high)
        self.register_buffer("action_center", (action_low + action_high) / 2.0)
        self.register_buffer("action_half_range", (action_high - action_low) / 2.0)

        # All safety filters use the same temporary local point-map geometry.
        # Raw ranges are converted to LiDAR endpoints in a scan-fixed local
        # frame; the robot pose is propagated relative to those points.
        kwargs = {
            "d_safe": safety_cfg.d_safe,
            "max_vx": action_limits[0],
            "max_vy": action_limits[1],
            "max_wz": action_limits[2],
            "min_vx": effective_min_vx,
            "robot_length_m": safety_cfg.robot_length_m,
            "robot_width_m": safety_cfg.robot_width_m,
            "lidar_x_m": safety_cfg.lidar_x_m,
            "lidar_y_m": safety_cfg.lidar_y_m,
            "footprint_clearance_m": safety_cfg.footprint_clearance_m,
            "use_elliptic_footprint": safety_cfg.use_elliptic_footprint,
            "max_range": self.max_range,
            "no_return_tolerance_m": safety_cfg.no_return_tolerance_m,
            "inactive_barrier_value": safety_cfg.inactive_barrier_value,
            "denominator_epsilon": safety_cfg.denominator_epsilon,
            "hard_clip_output": safety_cfg.hard_clip_output,
        }
        self.counterfactual_filter = LaserScanCbfClosedForm(
            alpha=safety_cfg.alpha,
            kappa=safety_cfg.kappa,
            **kwargs,
        )
        if method in ("vanilla", "cbf_cf_clip"):
            self.safety_filter = self.counterfactual_filter
        elif method == "cbf_qp":
            self.safety_filter = LaserScanCbfQp(
                alpha=safety_cfg.alpha,
                max_active_candidates=safety_cfg.qp_max_active_candidates,
                feasibility_tolerance=safety_cfg.qp_feasibility_tolerance,
                qp_regularization=safety_cfg.qp_regularization,
                straight_through=safety_cfg.qp_straight_through,
                qp_backend=safety_cfg.qp_backend,
                **kwargs,
            )
        elif method == "cbf_lse_qp":
            self.safety_filter = LaserScanCbfLseQp(
                alpha=safety_cfg.alpha,
                kappa=safety_cfg.kappa,
                max_active_candidates=safety_cfg.qp_max_active_candidates,
                feasibility_tolerance=safety_cfg.qp_feasibility_tolerance,
                qp_regularization=safety_cfg.qp_regularization,
                straight_through=safety_cfg.qp_straight_through,
                qp_backend=safety_cfg.qp_backend,
                **kwargs,
            )
        elif method == "cbf_cd_cf":
            self.safety_filter = LaserScanCbfControlDynamicsClosedForm(
                hocbf_k0=safety_cfg.hocbf_k0,
                alpha_outer=safety_cfg.alpha_outer,
                kappa=safety_cfg.kappa,
                kappa_input=safety_cfg.kappa_input,
                kappa_outer=safety_cfg.kappa_outer,
                barrier_rate_scale=safety_cfg.barrier_rate_scale,
                control_dynamics_a=safety_cfg.control_dynamics_a,
                tracking_sigma=safety_cfg.tracking_sigma,
                integration_dt=safety_cfg.integration_dt,
                **kwargs,
            )
        elif method == "cbf_lse_cd_qp":
            self.safety_filter = LaserScanCbfLseControlDynamicsQp(
                hocbf_k0=safety_cfg.hocbf_k0,
                alpha_outer=safety_cfg.alpha_outer,
                kappa=safety_cfg.kappa,
                kappa_input=safety_cfg.kappa_input,
                kappa_outer=safety_cfg.kappa_outer,
                barrier_rate_scale=safety_cfg.barrier_rate_scale,
                control_dynamics_a=safety_cfg.control_dynamics_a,
                tracking_sigma=safety_cfg.tracking_sigma,
                integration_dt=safety_cfg.integration_dt,
                max_active_candidates=safety_cfg.qp_max_active_candidates,
                feasibility_tolerance=safety_cfg.qp_feasibility_tolerance,
                qp_regularization=safety_cfg.qp_regularization,
                straight_through=safety_cfg.qp_straight_through,
                qp_backend=safety_cfg.qp_backend,
                **kwargs,
            )
        elif method == "cbf_lse_cd_qp_cfmatched":
            self.safety_filter = LaserScanCbfLseControlDynamicsQpCfMatched(
                hocbf_k0=safety_cfg.hocbf_k0,
                alpha_outer=safety_cfg.alpha_outer,
                kappa=safety_cfg.kappa,
                kappa_input=safety_cfg.kappa_input,
                kappa_outer=safety_cfg.kappa_outer,
                barrier_rate_scale=safety_cfg.barrier_rate_scale,
                control_dynamics_a=safety_cfg.control_dynamics_a,
                tracking_sigma=safety_cfg.tracking_sigma,
                integration_dt=safety_cfg.integration_dt,
                max_active_candidates=safety_cfg.qp_max_active_candidates,
                feasibility_tolerance=safety_cfg.qp_feasibility_tolerance,
                qp_regularization=safety_cfg.qp_regularization,
                straight_through=safety_cfg.qp_straight_through,
                qp_backend=safety_cfg.qp_backend,
                **kwargs,
            )
        elif method == "bcbf_cf":
            self.safety_filter = LaserZeroBackupLseCbfClosedForm(
                backup_margin=safety_cfg.backup_margin,
                backup_horizon_s=safety_cfg.backup_horizon_s,
                backup_samples=safety_cfg.backup_samples,
                backup_rollout_substeps=safety_cfg.backup_rollout_substeps,
                hocbf_k0=safety_cfg.hocbf_k0,
                barrier_rate_scale=safety_cfg.barrier_rate_scale,
                kappa_backup=safety_cfg.kappa_backup,
                kappa_input=safety_cfg.kappa_input,
                kappa_outer=safety_cfg.kappa_outer,
                alpha_outer=safety_cfg.alpha_outer,
                control_dynamics_a=safety_cfg.control_dynamics_a,
                tracking_sigma=safety_cfg.tracking_sigma,
                integration_dt=safety_cfg.integration_dt,
                terminal_backup_mode=safety_cfg.terminal_backup_mode,
                terminal_geom_scale=safety_cfg.terminal_geom_scale,
                terminal_rec_scale=safety_cfg.terminal_rec_scale,
                legacy_projection=safety_cfg.legacy_projection,
                straight_through=safety_cfg.cf_straight_through,
                **kwargs,
            )
        elif method == "bcbf_qp":
            self.safety_filter = LaserZeroBackupCbfQp(
                backup_margin=safety_cfg.backup_margin,
                backup_horizon_s=safety_cfg.backup_horizon_s,
                backup_samples=safety_cfg.backup_samples,
                backup_rollout_substeps=safety_cfg.backup_rollout_substeps,
                hocbf_k0=safety_cfg.hocbf_k0,
                barrier_rate_scale=safety_cfg.barrier_rate_scale,
                # kappa_backup aggregates only current/horizon/terminal rows;
                # six input-limit barriers remain separate QP constraints.
                kappa_backup=safety_cfg.kappa_backup,
                kappa_input=safety_cfg.kappa_input,
                kappa_outer=safety_cfg.kappa_outer,
                alpha_outer=safety_cfg.alpha_outer,
                control_dynamics_a=safety_cfg.control_dynamics_a,
                tracking_sigma=safety_cfg.tracking_sigma,
                integration_dt=safety_cfg.integration_dt,
                terminal_backup_mode=safety_cfg.terminal_backup_mode,
                terminal_geom_scale=safety_cfg.terminal_geom_scale,
                terminal_rec_scale=safety_cfg.terminal_rec_scale,
                max_active_candidates=safety_cfg.qp_max_active_candidates,
                feasibility_tolerance=safety_cfg.qp_feasibility_tolerance,
                qp_regularization=safety_cfg.qp_regularization,
                straight_through=safety_cfg.qp_straight_through,
                qp_backend=safety_cfg.qp_backend,
                **kwargs,
            )
        elif method == "bcbf_lse_cd_qp":
            self.safety_filter = LaserZeroBackupLseControlDynamicsQp(
                backup_margin=safety_cfg.backup_margin,
                backup_horizon_s=safety_cfg.backup_horizon_s,
                backup_samples=safety_cfg.backup_samples,
                backup_rollout_substeps=safety_cfg.backup_rollout_substeps,
                hocbf_k0=safety_cfg.hocbf_k0,
                barrier_rate_scale=safety_cfg.barrier_rate_scale,
                kappa_backup=safety_cfg.kappa_backup,
                kappa_input=safety_cfg.kappa_input,
                kappa_outer=safety_cfg.kappa_outer,
                alpha_outer=safety_cfg.alpha_outer,
                control_dynamics_a=safety_cfg.control_dynamics_a,
                tracking_sigma=safety_cfg.tracking_sigma,
                integration_dt=safety_cfg.integration_dt,
                terminal_backup_mode=safety_cfg.terminal_backup_mode,
                terminal_geom_scale=safety_cfg.terminal_geom_scale,
                terminal_rec_scale=safety_cfg.terminal_rec_scale,
                max_active_candidates=safety_cfg.qp_max_active_candidates,
                feasibility_tolerance=safety_cfg.qp_feasibility_tolerance,
                qp_regularization=safety_cfg.qp_regularization,
                straight_through=safety_cfg.qp_straight_through,
                qp_backend=safety_cfg.qp_backend,
                **kwargs,
            )
        elif method == "bcbf_lse_cd_qp_cfmatched":
            self.safety_filter = LaserZeroBackupLseControlDynamicsQpCfMatched(
                backup_margin=safety_cfg.backup_margin,
                backup_horizon_s=safety_cfg.backup_horizon_s,
                backup_samples=safety_cfg.backup_samples,
                backup_rollout_substeps=safety_cfg.backup_rollout_substeps,
                hocbf_k0=safety_cfg.hocbf_k0,
                barrier_rate_scale=safety_cfg.barrier_rate_scale,
                kappa_backup=safety_cfg.kappa_backup,
                kappa_input=safety_cfg.kappa_input,
                kappa_outer=safety_cfg.kappa_outer,
                alpha_outer=safety_cfg.alpha_outer,
                control_dynamics_a=safety_cfg.control_dynamics_a,
                tracking_sigma=safety_cfg.tracking_sigma,
                integration_dt=safety_cfg.integration_dt,
                terminal_backup_mode=safety_cfg.terminal_backup_mode,
                terminal_geom_scale=safety_cfg.terminal_geom_scale,
                terminal_rec_scale=safety_cfg.terminal_rec_scale,
                max_active_candidates=safety_cfg.qp_max_active_candidates,
                feasibility_tolerance=safety_cfg.qp_feasibility_tolerance,
                qp_regularization=safety_cfg.qp_regularization,
                straight_through=safety_cfg.qp_straight_through,
                qp_backend=safety_cfg.qp_backend,
                **kwargs,
            )
        elif method == "bcbf_nonlse_cd_qp":
            self.safety_filter = LaserZeroBackupNonLseControlDynamicsQp(
                backup_margin=safety_cfg.backup_margin,
                backup_horizon_s=safety_cfg.backup_horizon_s,
                backup_samples=safety_cfg.backup_samples,
                backup_rollout_substeps=safety_cfg.backup_rollout_substeps,
                hocbf_k0=safety_cfg.hocbf_k0,
                barrier_rate_scale=safety_cfg.barrier_rate_scale,
                kappa_backup=safety_cfg.kappa_backup,
                kappa_input=safety_cfg.kappa_input,
                kappa_outer=safety_cfg.kappa_outer,
                alpha_outer=safety_cfg.alpha_outer,
                control_dynamics_a=safety_cfg.control_dynamics_a,
                tracking_sigma=safety_cfg.tracking_sigma,
                integration_dt=safety_cfg.integration_dt,
                terminal_backup_mode=safety_cfg.terminal_backup_mode,
                terminal_geom_scale=safety_cfg.terminal_geom_scale,
                terminal_rec_scale=safety_cfg.terminal_rec_scale,
                max_active_candidates=safety_cfg.qp_max_active_candidates,
                feasibility_tolerance=safety_cfg.qp_feasibility_tolerance,
                qp_regularization=safety_cfg.qp_regularization,
                straight_through=safety_cfg.qp_straight_through,
                qp_backend=safety_cfg.qp_backend,
                **kwargs,
            )
        elif method == "bcbf_lse_qp":
            # No control-dynamics state at all (no control_dynamics_a/
            # tracking_sigma/integration_dt/terminal_backup_mode/hocbf_k0/
            # barrier_rate_scale/kappa_input/kappa_outer -- this filter
            # doesn't have those concepts). Reuses alpha_outer/kappa_backup/
            # backup_margin/backup_samples -- the single soft-min backup row
            # plays the same role those play in the CD backup filters.
            self.safety_filter = LaserZeroBackupLseQp(
                alpha=safety_cfg.alpha_outer,
                backup_margin=safety_cfg.backup_margin,
                backup_samples=safety_cfg.backup_samples,
                kappa_backup=safety_cfg.kappa_backup,
                max_active_candidates=safety_cfg.qp_max_active_candidates,
                feasibility_tolerance=safety_cfg.qp_feasibility_tolerance,
                qp_regularization=safety_cfg.qp_regularization,
                straight_through=safety_cfg.qp_straight_through,
                qp_backend=safety_cfg.qp_backend,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown laser SAC method {method!r}")

    def _distribution(self, obs: torch.Tensor) -> Normal:
        mean, log_std = self.backbone(obs).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return Normal(mean, log_std.exp())

    def _ranges(self, obs: torch.Tensor) -> torch.Tensor:
        normalized = obs[:, self.range_start : self.range_start + self.num_rays]
        return torch.clamp(normalized * self.max_range, min=0.0, max=self.max_range)

    def normalized_to_command(self, normalized: torch.Tensor) -> torch.Tensor:
        """Map a per-axis [-1, 1] value to this actor's own [low, high]
        command range. Reduces to normalized * action_high (the archived
        symmetric formula) when low == -high."""
        return self.action_center + normalized * self.action_half_range

    def command_to_normalized(self, command: torch.Tensor) -> torch.Tensor:
        return (command - self.action_center) / self.action_half_range

    def apply_safety(
        self,
        nominal_normalized: torch.Tensor,
        obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, LaserSafetyFilterResult]:
        nominal_command = self.normalized_to_command(nominal_normalized)
        ranges = self._ranges(obs)
        angles = self.ray_angles.to(device=obs.device, dtype=obs.dtype)

        if self.method == "vanilla":
            safety = self.counterfactual_filter.filter(ranges, angles, nominal_command)
            command = nominal_command
        elif self.method in ("cbf_cf_clip", "cbf_qp", "cbf_lse_qp"):
            safety = self.safety_filter.filter(ranges, angles, nominal_command)
            command = safety.command
        else:
            safety = self.safety_filter.filter(
                ranges,
                angles,
                obs[:, 5:8],
                nominal_command,
                obs[:, -4:-1],
                obs[:, -1:],
            )
            command = safety.command

        if self.disable_safety_filter:
            command = nominal_command

        # Do not silently reintroduce clipping when it was disabled in the
        # safety configuration; that would alter the projected CBF solution.
        safe_command = (
            torch.clamp(command, self.action_low, self.action_high)
            if self.hard_clip_output
            else command
        )
        safe_normalized = self.command_to_normalized(safe_command)
        return safe_normalized, safe_command, safety

    def sample(self, obs: torch.Tensor) -> SafeActorOutput:
        dist = self._distribution(obs)
        pre_tanh = dist.rsample()
        nominal_normalized = torch.tanh(pre_tanh)
        log_prob = dist.log_prob(pre_tanh) - torch.log(1.0 - nominal_normalized.pow(2) + 1.0e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        safe_normalized, safe_command, safety = self.apply_safety(nominal_normalized, obs)
        return SafeActorOutput(
            nominal_normalized=nominal_normalized,
            nominal_command=self.normalized_to_command(nominal_normalized),
            safe_normalized=safe_normalized,
            safe_command=safe_command,
            log_prob=log_prob,
            safety=safety,
        )

    @torch.no_grad()
    def deterministic(self, obs: torch.Tensor) -> SafeActorOutput:
        mean = self.backbone(obs).chunk(2, dim=-1)[0]
        nominal_normalized = torch.tanh(mean)
        safe_normalized, safe_command, safety = self.apply_safety(nominal_normalized, obs)
        return SafeActorOutput(
            nominal_normalized=nominal_normalized,
            nominal_command=self.normalized_to_command(nominal_normalized),
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
