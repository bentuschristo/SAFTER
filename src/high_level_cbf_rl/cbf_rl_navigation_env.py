from __future__ import annotations

from collections import deque
from typing import Any

import torch

# CbfRlNavigationEnv subclasses LaserSacNavigationEnv (LiDAR wiring, procedural
# map resets) from SAFTER_src, falling back to src/ when SAFTER_src is absent.
try:
    from SAFTER_src.high_level_laser_sac.config import LaserRewardConfig
    from SAFTER_src.high_level_laser_sac.sac_navigation_env import LaserSacNavigationEnv
except ModuleNotFoundError:
    from src.high_level_laser_sac.config import LaserRewardConfig
    from src.high_level_laser_sac.sac_navigation_env import LaserSacNavigationEnv
from src.high_level_rl.laser_cbf_filters import LaserScanCbfClosedForm

from .config import CbfRlConfig


class CbfRlNavigationEnv(LaserSacNavigationEnv):
    """PPO navigation env implementing Yang et al.'s dual CBF-RL training.

    Subclasses ``LaserSacNavigationEnv`` purely to reuse its LiDAR wiring,
    procedural-map/fixed-arena reset machinery, and laser-ray-based
    observation -- all sensor/geometry infrastructure independent of the RL
    algorithm. Unlike the SAC pipeline (whose differentiable actor decides
    the nominal/safe command pair externally and calls ``step_safe(...)``),
    this env owns the safety filter itself and applies it during ``step()``,
    exposing the standard single-argument rsl_rl VecEnv contract:
    ``step(action) -> (obs, reward, done, extras)``.

    Both the safety filter and the CBF reward terms are always computed
    every step, regardless of the ablation mode -- only whether the filtered
    command is actually executed, and whether the CBF reward terms are
    added, are gated by ``CbfRlModeConfig``. This mirrors the cbf-rl-demo
    reference implementation's ablation behavior exactly.
    """

    def __init__(
        self,
        low_env: Any,
        low_policy,
        command_interface,
        cfg: CbfRlConfig,
        **kwargs: Any,
    ) -> None:
        self.cbf_rl_cfg = cfg
        super().__init__(
            low_env,
            low_policy,
            command_interface,
            cfg.navigation,
            # Required by the parent constructor but unused: step() reads
            # cfg.reward (CbfRlRewardConfig).
            reward_cfg=LaserRewardConfig(),
            laser_cfg=cfg.laser,
            safety_cfg=cfg.safety,
            contact_collision_cfg=cfg.contact_collision,
            **kwargs,
        )
        safety = cfg.safety
        action_cfg = cfg.navigation.action
        self.cbf_filter = LaserScanCbfClosedForm(
            d_safe=safety.d_safe,
            alpha=safety.alpha,
            kappa=safety.kappa,
            max_vx=action_cfg.max_vx,
            max_vy=action_cfg.max_vy,
            max_wz=action_cfg.max_wz,
            robot_length_m=safety.robot_length_m,
            robot_width_m=safety.robot_width_m,
            lidar_x_m=safety.lidar_x_m,
            lidar_y_m=safety.lidar_y_m,
            footprint_clearance_m=safety.footprint_clearance_m,
            use_elliptic_footprint=safety.use_elliptic_footprint,
            max_range=cfg.laser.max_range,
            no_return_tolerance_m=safety.no_return_tolerance_m,
            inactive_barrier_value=safety.inactive_barrier_value,
            denominator_epsilon=safety.denominator_epsilon,
            hard_clip_output=True,
        )
        self.mode_cfg = cfg.mode
        self.cbf_reward_cfg = cfg.reward
        if self._any_square_slot and safety.collision_geometry != "circle":
            # Box-vs-ellipse Minkowski overlap is not implemented (as in step_safe()).
            raise ValueError(
                "Any square obstacle slot currently requires safety.collision_geometry='circle'"
            )

        # Asymmetric-vx affine action map [-1, 1] -> [low, high], as in
        # LaserSafeSacActor; the plain PPO actor has no such action head.
        action_low = torch.tensor(
            (action_cfg.min_vx, -action_cfg.max_vy, -action_cfg.max_wz), device=self.device
        )
        action_high = torch.tensor(
            (action_cfg.max_vx, action_cfg.max_vy, action_cfg.max_wz), device=self.device
        )
        self._action_low = action_low
        self._action_high = action_high
        self._action_center = (action_low + action_high) / 2.0
        self._action_half_range = (action_high - action_low) / 2.0

        # Rolling outcome rates over the last 100 completed episodes, with the
        # same TensorBoard tags as LaserSacTrainer so curves are comparable.
        # Only envs whose episode just ended contribute.
        self._episode_outcome_rolling: dict[str, deque[float]] = {
            name: deque(maxlen=100) for name in ("success", "collision", "timeout", "fall")
        }

    def _scale_action(self, action: torch.Tensor) -> torch.Tensor:
        action = torch.clamp(action, -1.0, 1.0)
        return self._action_center + action * self._action_half_range

    def step(self, action: torch.Tensor):
        self._maybe_refresh_laser_cache()
        nav_cfg = self.navigation_cfg
        nominal = self._scale_action(action)

        # The filter always runs; the ablation mode only gates whether its
        # command and diagnostics are used.
        safety = self.cbf_filter.filter(self.laser_ranges(), self.ray_angles, nominal)
        safe = safety.command if self.mode_cfg.use_cbf_action_filtering else nominal

        # The filter clips to a symmetric +-max_vx box; re-clamp to [min_vx, max_vx].
        safe = torch.clamp(safe, self._action_low, self._action_high)
        previous_safe = self.previous_safe_command.clone()
        previous_distance = self.previous_goal_distance.clone()

        for _ in range(nav_cfg.low_steps_per_high_step):
            low_obs = self.low_env.get_observations()
            joint_action = self.low_policy(low_obs, safe)
            self.low_env.step(joint_action)

        pos, _, _, _, roll, pitch, yaw = self._root_state()
        # Snapshot before autoreset: simulator state tensors can be views.
        p = pos[:, :2].detach().clone()
        distance = torch.linalg.norm(self.goal_positions - p, dim=-1)
        progress = previous_distance - distance

        # Collision geometry as in the laser SAC step_safe(): "ellipse"
        # (Minkowski overlap of the oriented footprint, circular slots only) or
        # "circle" (center-distance overlap, with a square-obstacle case).
        geometry = self.laser_safety_cfg
        rel_w = self.obstacle_centers - p[:, None, :]
        d_obs = torch.linalg.norm(rel_w, dim=-1)
        if geometry.collision_geometry == "circle":
            collision_radius = float(geometry.collision_circle_radius_m)
            collision_distance = self.obstacle_radii + collision_radius
            collision_h_obs = d_obs - collision_distance
            if self._any_square_slot:
                delta = torch.clamp(torch.abs(rel_w) - self.obstacle_radii[..., None], min=0.0)
                square_h_obs = torch.linalg.norm(delta, dim=-1) - collision_radius
                collision_h_obs = torch.where(
                    self._slot_is_square[None, :], square_h_obs, collision_h_obs
                )
        elif geometry.collision_geometry == "ellipse":
            half_length = 0.5 * float(geometry.robot_length_m)
            half_width = 0.5 * float(geometry.robot_width_m)
            collision_margin = float(geometry.collision_margin_m)
            collision_half_length = half_length + collision_margin
            collision_half_width = half_width + collision_margin
            c = torch.cos(yaw)[:, None]
            s = torch.sin(yaw)[:, None]
            rel_x_b = c * rel_w[..., 0] + s * rel_w[..., 1]
            rel_y_b = -s * rel_w[..., 0] + c * rel_w[..., 1]
            collision_a = collision_half_length + self.obstacle_radii
            collision_b = collision_half_width + self.obstacle_radii
            collision_h_obs = (
                (rel_x_b / collision_a).square() + (rel_y_b / collision_b).square() - 1.0
            )
        else:
            raise ValueError(f"Unknown collision_geometry={geometry.collision_geometry!r}")
        collision_h_obs = torch.where(
            self.obstacle_mask, collision_h_obs, torch.full_like(collision_h_obs, torch.inf)
        )
        obstacle_collision = torch.any(collision_h_obs <= 0.0, dim=-1)
        outside = (
            (p[:, 0] <= nav_cfg.arena.x_min)
            | (p[:, 0] >= nav_cfg.arena.x_max)
            | (p[:, 1] <= nav_cfg.arena.y_min)
            | (p[:, 1] >= nav_cfg.arena.y_max)
        )
        collision = obstacle_collision | outside
        fallen = (
            (pos[:, 2] < nav_cfg.minimum_base_height)
            | (torch.abs(roll) > nav_cfg.maximum_roll_pitch)
            | (torch.abs(pitch) > nav_cfg.maximum_roll_pitch)
        )
        # One outcome per episode: collision > fall > success > timeout.
        fallen = fallen & ~collision
        goal_candidate = distance <= nav_cfg.goal_tolerance
        goal = goal_candidate & ~collision & ~fallen
        timeout = self.episode_length_buf + 1 >= nav_cfg.max_episode_steps

        terminated = goal | collision | fallen
        truncated = timeout & ~terminated
        done = terminated | truncated
        active_mask = (~terminated).float()

        r = self.cbf_reward_cfg
        dt_scale = float(nav_cfg.high_level_dt) / float(r.reference_dt)
        progress_reward = (
            r.progress_weight
            * progress
            / max(nav_cfg.action.max_vx * nav_cfg.high_level_dt, 1.0e-8)
            * active_mask
        )
        alive_reward = r.alive_reward * dt_scale * active_mask
        goal_reward = r.goal_reward * goal.float()
        obstacle_reward = -r.obstacle_penalty * obstacle_collision.float()
        wall_reward = -r.wall_penalty * outside.float()
        timeout_reward = -r.timeout_penalty * truncated.float()

        if self.mode_cfg.use_cbf_reward_penalty:
            violation_reward = (
                r.cbf_violation_weight
                * torch.clamp(safety.cbf_value_nominal, max=0.0)
                * active_mask
            )
            match_reward = (
                r.cbf_match_weight
                * (torch.exp(-(safety.correction_norm ** 2) / (r.cbf_match_sigma ** 2)) - 1.0)
                * active_mask
            )
        else:
            violation_reward = torch.zeros_like(progress_reward)
            match_reward = torch.zeros_like(progress_reward)

        reward = (
            progress_reward
            + alive_reward
            + goal_reward
            + obstacle_reward
            + wall_reward
            + timeout_reward
            + violation_reward
            + match_reward
        )

        self.previous_safe_command = safe.detach().clone()
        self.previous_nominal_command = nominal.detach().clone()
        self.nominal_history_valid.fill_(1.0)
        self.previous_goal_distance = distance.detach().clone()
        self.episode_length_buf += 1
        self._last_safety = safety

        smoothness = torch.sum((safe - previous_safe).pow(2), dim=-1)
        log = {
            "reward_progress": progress.mean(),
            # Per-step indicator means (live signal); not comparable to the
            # per-episode rolling-100 rates below.
            "success_rate": goal.float().mean(),
            "collision_rate": collision.float().mean(),
            "fall_rate": fallen.float().mean(),
            "distance_to_goal": distance.mean(),
            "smoothness_cost": smoothness.mean(),
            "cbf_active_rate": safety.active.float().mean(),
            "cbf_correction_norm": safety.correction_norm.mean(),
            "cbf_h_lse_mean": safety.h_lse.mean(),
            "mode_use_cbf_action_filtering": float(self.mode_cfg.use_cbf_action_filtering),
            "mode_use_cbf_reward_penalty": float(self.mode_cfg.use_cbf_reward_penalty),
        }

        ids = torch.nonzero(done, as_tuple=False).flatten()
        if ids.numel() > 0:
            # The outcome precedence above makes these mutually exclusive.
            self._episode_outcome_rolling["success"].extend(goal[ids].float().tolist())
            self._episode_outcome_rolling["collision"].extend(collision[ids].float().tolist())
            self._episode_outcome_rolling["timeout"].extend(truncated[ids].float().tolist())
            self._episode_outcome_rolling["fall"].extend(fallen[ids].float().tolist())
            self._reset_selected(ids)

        for name, tag in (
            ("success", "Train/episode_success_rate_100"),
            ("collision", "Train/episode_collision_rate_100"),
            ("timeout", "Train/episode_timeout_rate_100"),
        ):
            values = self._episode_outcome_rolling[name]
            if values:
                log[tag] = sum(values) / len(values)
        fall_values = self._episode_outcome_rolling["fall"]
        if fall_values:
            log["episode_fall_rate_100"] = sum(fall_values) / len(fall_values)

        extras = {
            "observations": self._compute_observation(),
            "time_outs": truncated,
            "log": log,
            # Per-env outcome flags for this step (same names as the SAC
            # step_safe() extras); valid where done[env_id] is True.
            "completed_success": goal,
            "completed_collision": collision,
            "completed_timeouts": truncated,
            "completed_fallen": fallen,
            "completed_outside": outside,
            "completed_env_ids": ids,
            "distance_to_goal": distance.detach().clone(),
            # World (x, y) after this step, captured before resets, so terminal
            # steps report the termination position.
            "position": p.detach().clone(),
        }

        obs = self._compute_observation()
        return obs, reward, done, extras
