from __future__ import annotations

from collections import deque
from typing import Any

import torch

# NOTE: CbfRlNavigationEnv genuinely subclasses LaserSacNavigationEnv (reuses
# its LiDAR wiring/procedural-map reset machinery) and reuses several of its
# config dataclasses, so unlike the smaller shared config/math files handled
# elsewhere in this reorg (which got duplicated into SAFTER_src), these are a
# real cross-package dependency on SAFTER_src -- duplicating an 85KB,
# actively-changing environment class would just guarantee silent drift.
try:
    # Local repo state: high_level_laser_sac lives under SAFTER_src/.
    from SAFTER_src.high_level_laser_sac.config import LaserRewardConfig
    from SAFTER_src.high_level_laser_sac.sac_navigation_env import LaserSacNavigationEnv
except ModuleNotFoundError:
    # Server state: SAFTER_src/ hasn't been synced there -- fall back to the
    # original location.
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
            # Unused by this env's step() -- kept only to satisfy
            # LaserSacNavigationEnv's constructor, which stores it as
            # self.sac_reward_cfg. This env's own reward math lives in
            # step() below and reads cfg.reward (CbfRlRewardConfig) instead.
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
            # Matches step_safe()'s own restriction exactly: box-vs-ellipse
            # Minkowski overlap isn't implemented, so a square slot requires
            # the "circle" collision_geometry branch below.
            raise ValueError(
                "Any square obstacle slot currently requires safety.collision_geometry='circle'"
            )

        # Asymmetric-vx affine action map (action_low/action_high/center/
        # half_range), replicated from LaserSafeSacActor (networks.py) since
        # this env's plain PPO actor has no equivalent built-in action head.
        # min_vx is otherwise unused by HighLevelNavigationEnv's own
        # _scale_action/step(), which are symmetric-+-max_vx only.
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

        # 100-completed-episode rolling outcome rates, matching
        # LaserSacTrainer's rolling_100 mechanism and TensorBoard tag names
        # (Train/episode_{success,collision,timeout}_rate_100) exactly, so
        # this overlays as the same chart as the SAC baselines. Unlike
        # rsl_rl's own generic extras["log"] averaging (a per-iteration mean
        # of a per-step indicator diluted across every env at every step,
        # not per-episode), these deques are extended only with the outcome
        # of envs whose episode just ended.
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

        # Always compute the filter, regardless of ablation mode -- the
        # demo repo computes both the filtered action and the CBF
        # diagnostics unconditionally, gating only whether they're used.
        safety = self.cbf_filter.filter(self.laser_ranges(), self.ray_angles, nominal)
        safe = safety.command if self.mode_cfg.use_cbf_action_filtering else nominal

        # The filter's own hard_clip_output uses a symmetric +-max_vx box (see
        # LaserScanCbfClosedForm docstring precedent: narrowing the proposal/
        # output range after the filter is a subset of what it already
        # handles). Re-clamp to the actual asymmetric [min_vx, max_vx] range.
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

        # Same collision geometry used by the laser SAC pipeline's
        # step_safe(), so "same elliptical CBF, same margin" extends to how a
        # physical collision is defined too. Two branches, exactly mirroring
        # step_safe: "ellipse" (Minkowski overlap of the oriented robot
        # footprint against each obstacle, circular slots only) and "circle"
        # (simple center-distance overlap, with a square-obstacle sub-case
        # needed for maps like circle_and_square).
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
            # Per-step (not per-episode) indicator means, diluted across
            # every env at every step -- kept for a low-latency live signal,
            # but NOT comparable to the SAC trainer's rolling-100 stats
            # below.
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
            # Per-env outcome flags (this step, not aggregated), for
            # evaluation scripts scoring individual episodes -- matching the
            # SAC pipeline's step_safe()/extras naming convention
            # (completed_success/collision/timeouts/fallen/outside) so an
            # eval script can read extras["completed_X"][env_id] wherever
            # done[env_id] is True.
            "completed_success": goal,
            "completed_collision": collision,
            "completed_timeouts": truncated,
            "completed_fallen": fallen,
            "completed_outside": outside,
            "completed_env_ids": ids,
            "distance_to_goal": distance.detach().clone(),
            # Absolute (x, y) position after this step's substeps, captured
            # before any just-terminated env is reset below -- so on a
            # terminal step this is still the true termination position, not
            # a post-reset one. Lets an external script log a clean per-step
            # trajectory without duplicating step()'s internals.
            "position": p.detach().clone(),
        }

        obs = self._compute_observation()
        return obs, reward, done, extras
