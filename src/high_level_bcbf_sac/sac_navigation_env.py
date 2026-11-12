from __future__ import annotations

from typing import Any

import torch

from src.high_level_rl.navigation_env import HighLevelNavigationEnv
from src.high_level_rl.math_utils import world_to_body

from .config import OrientationAwareRewardConfig


class SacCbfNavigationEnv(HighLevelNavigationEnv):
    """High-level environment that executes the actor's already-safe command.

    Unlike the PPO wrapper, this environment does not compute the safety action.
    The SAC actor computes it as its differentiable final layer and passes both
    nominal and safe commands here for execution, reward accounting, and logs.
    """

    def __init__(self, *args: Any, reward_cfg: OrientationAwareRewardConfig, **kwargs: Any) -> None:
        self.sac_reward_cfg = reward_cfg
        self._fixed_scenario = None
        super().__init__(*args, **kwargs)
        self.episode_return_buf = torch.zeros(self.num_envs, device=self.device)

        # Nominal-command history is part of the Markov observation.
        self.previous_nominal_command = torch.zeros(
            self.num_envs,
            3,
            device=self.device,
        )
        self.nominal_history_valid = torch.zeros(
            self.num_envs,
            1,
            device=self.device,
        )

        self.episode_success_count = 0
        self.episode_collision_count = 0
        self.episode_count = 0

    @property
    def navigation_cfg(self):
        if hasattr(self, "nav_cfg"):
            return self.nav_cfg
        return self.cfg

    @staticmethod
    def actor_tensor(obs) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            return obs
        if hasattr(obs, "get"):
            actor = obs.get("actor")
            if actor is not None:
                return actor
        return obs["actor"]

    def _compute_observation(self):
        """Append previous nominal command and a validity flag."""
        base_obs = super()._compute_observation()

        # The base constructor may request an observation before these fields
        # are initialized.
        if not hasattr(self, "previous_nominal_command"):
            return base_obs

        history = torch.cat(
            (
                self.previous_nominal_command,
                self.nominal_history_valid,
            ),
            dim=-1,
        )

        if isinstance(base_obs, torch.Tensor):
            return torch.cat((base_obs, history), dim=-1)

        actor_obs = base_obs.get("actor")
        if actor_obs is None:
            actor_obs = base_obs["actor"]

        augmented = dict(base_obs)
        augmented["actor"] = torch.cat(
            (actor_obs, history),
            dim=-1,
        )
        return augmented

    def safety_context(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos, _, _, _, _, _, yaw = self._root_state()
        return pos[:, :2].clone(), yaw.clone()

    def _reset_selected(self, env_ids: torch.Tensor) -> None:
        super()._reset_selected(env_ids)
        if self._fixed_scenario is not None:
            self._apply_fixed_scenario(env_ids)
        if hasattr(self, "episode_return_buf"):
            self.episode_return_buf[env_ids] = 0.0

        if hasattr(self, "previous_nominal_command"):
            self.previous_nominal_command[env_ids] = 0.0
            self.nominal_history_valid[env_ids] = 0.0

    def set_fixed_scenario(self, scenario) -> None:
        """Use one deterministic start/goal pair for all subsequent resets."""
        self._fixed_scenario = scenario
        env_ids = torch.arange(
            self.num_envs,
            device=self.device,
            dtype=torch.long,
        )
        self._reset_selected(env_ids)

    def clear_fixed_scenario(self) -> None:
        self._fixed_scenario = None

    def _apply_fixed_scenario(self, env_ids: torch.Tensor) -> None:
        scenario = self._fixed_scenario
        if scenario is None or env_ids.numel() == 0:
            return

        robot = self.unwrapped.scene["robot"]
        pose = robot.data.root_link_pose_w[env_ids].clone()
        velocity = torch.zeros(
            env_ids.numel(),
            6,
            device=self.device,
            dtype=pose.dtype,
        )

        pose[:, 0] = float(scenario.start_xy[0])
        pose[:, 1] = float(scenario.start_xy[1])

        half_yaw = 0.5 * float(scenario.start_yaw)
        pose[:, 3] = torch.cos(pose.new_tensor(half_yaw))
        pose[:, 4] = 0.0
        pose[:, 5] = 0.0
        pose[:, 6] = torch.sin(pose.new_tensor(half_yaw))

        root_state = torch.cat((pose, velocity), dim=-1)
        robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self._refresh_state_buffers()

        goal = pose.new_tensor(scenario.goal_xy)
        self.goal_positions[env_ids] = goal
        self.previous_safe_command[env_ids] = 0.0

        if hasattr(self, "previous_nominal_command"):
            self.previous_nominal_command[env_ids] = 0.0
            self.nominal_history_valid[env_ids] = 0.0

        self.previous_goal_distance[env_ids] = torch.linalg.norm(
            goal - pose[:, :2],
            dim=-1,
        )
        self.episode_length_buf[env_ids] = 0

        if hasattr(self, "episode_return_buf"):
            self.episode_return_buf[env_ids] = 0.0

        for name in (
            "_update_goal_markers",
            "_update_goal_marker",
            "_draw_goal_markers",
            "_draw_goal_marker",
        ):
            fn = getattr(self, name, None)
            if callable(fn):
                try:
                    fn(env_ids)
                except TypeError:
                    fn()
                break

    def step_safe(
        self,
        nominal_command: torch.Tensor,
        safe_command: torch.Tensor,
        safety,
    ):
        cfg = self.navigation_cfg
        scale = torch.tensor(
            [cfg.action.max_vx, cfg.action.max_vy, cfg.action.max_wz],
            device=self.device,
            dtype=safe_command.dtype,
        )
        # No post-filter clipping. Command limits are represented by the
        # augmented-state input barriers.
        safe = safe_command
        nominal = nominal_command

        previous_safe = self.previous_safe_command.clone()
        previous_distance = self.previous_goal_distance.clone()

        for _ in range(cfg.low_steps_per_high_step):
            # Keep compatibility with both command-interface variants used in
            # the project. The frozen low-level policy also receives `safe`.
            if hasattr(self.command_interface, "write_batch"):
                self.command_interface.write_batch(safe)
            low_obs = self.low_env.get_observations()
            joint_action = self.low_policy(low_obs, safe)
            self.low_env.step(joint_action)

        pos, _, vel_b, ang_b, roll, pitch, yaw = self._root_state()
        p = pos[:, :2]
        distance = torch.linalg.norm(self.goal_positions - p, dim=-1)
        progress = previous_distance - distance
        goal = distance <= cfg.goal_tolerance

        d_obs = torch.linalg.norm(p[:, None, :] - self.obstacle_centers, dim=-1)
        robot_radius = getattr(cfg.arena, "robot_radius", 0.30)
        collision_distance = self.obstacle_radii + robot_radius
        obstacle_collision = torch.any(
            (d_obs <= collision_distance) & self.obstacle_mask,
            dim=-1,
        )
        outside = (
            (p[:, 0] <= cfg.arena.x_min + robot_radius)
            | (p[:, 0] >= cfg.arena.x_max - robot_radius)
            | (p[:, 1] <= cfg.arena.y_min + robot_radius)
            | (p[:, 1] >= cfg.arena.y_max - robot_radius)
        )
        # Treat obstacle contact and arena-wall violation as one collision event
        # for reward, termination, and headline evaluation metrics.
        collision = obstacle_collision | outside

        fallen = (
            (pos[:, 2] < cfg.minimum_base_height)
            | (torch.abs(roll) > cfg.maximum_roll_pitch)
            | (torch.abs(pitch) > cfg.maximum_roll_pitch)
        )
        timeout = self.episode_length_buf + 1 >= cfg.max_episode_steps

        goal_b = world_to_body(self.goal_positions - p, yaw)
        beta = torch.atan2(goal_b[:, 1], goal_b[:, 0])
        vx_measured = vel_b[:, 0]
        wz_measured = ang_b[:, 2]
        heading_alignment = torch.cos(beta)
        forward_alignment = (
            torch.relu(vx_measured) / max(cfg.action.max_vx, 1.0e-6)
        ) * heading_alignment
        reverse_cost = (
            torch.relu(-vx_measured) / max(cfg.action.max_vx, 1.0e-6)
        ).pow(2)
        omega_ref = torch.clamp(
            self.sac_reward_cfg.yaw_gain * beta,
            -cfg.action.max_wz,
            cfg.action.max_wz,
        )
        yaw_tracking_cost = (
            (wz_measured - omega_ref) / max(cfg.action.max_wz, 1.0e-6)
        ).pow(2)
        smoothness = torch.sum((safe - previous_safe).pow(2), dim=-1)
        normalized_correction = torch.sum(
            (
                (
                    safety.surrogate_command
                    - safety.desired_surrogate_command
                )
                / scale
            ).pow(2),
            dim=-1,
        )

        terminated = goal | collision | fallen
        truncated = timeout & ~terminated
        done = terminated | truncated

        r = self.sac_reward_cfg
        components = {
            "progress": r.progress_weight * progress,
            "goal": r.goal_bonus * goal.float(),
            "time": torch.full_like(progress, -r.time_penalty),

            "timeout": -r.timeout_penalty * truncated.float(),
            "forward_alignment": r.forward_alignment_weight * forward_alignment,
            "reverse_motion": -r.reverse_motion_weight * reverse_cost,
            "yaw_tracking": -r.yaw_tracking_weight * yaw_tracking_cost,
            "smoothness": -r.smoothness_weight * smoothness,
            "cbf_correction": -r.cbf_correction_weight * normalized_correction,
            "collision": -r.collision_penalty * collision.float(),
        }
        reward = torch.stack(tuple(components.values()), dim=0).sum(dim=0)

        self.previous_safe_command = safe.detach().clone()

        # This nominal command becomes u_d[k-1] in the next observation.
        self.previous_nominal_command = nominal.detach().clone()
        self.nominal_history_valid.fill_(1.0)

        self.previous_goal_distance = distance.detach().clone()
        self.episode_length_buf += 1
        self._last_safety = safety
        self.episode_return_buf += reward
        terminal_observation = self._compute_observation()

        completed_ids = torch.nonzero(done, as_tuple=False).flatten()
        completed_returns = self.episode_return_buf[completed_ids].detach().clone()
        completed_lengths = self.episode_length_buf[completed_ids].detach().clone()
        completed_terminal_positions = p[completed_ids].detach().clone()
        completed_timeouts = truncated[completed_ids].detach().clone()
        if completed_ids.numel() > 0:
            self.episode_count += int(completed_ids.numel())
            self.episode_success_count += int(goal[completed_ids].sum())
            self.episode_collision_count += int(collision[completed_ids].sum())
            self._reset_selected(completed_ids)

        obs = self._compute_observation()
        extras = {
            "time_outs": truncated,
            "terminal_observation": terminal_observation,
            "terminal_position": p.detach().clone(),
            "terminal_yaw": yaw.detach().clone(),
            "completed_returns": completed_returns,
            "completed_lengths": completed_lengths,
            "completed_terminal_positions": completed_terminal_positions,
            "completed_success": goal[completed_ids].detach().clone(),
            "completed_collision": collision[completed_ids].detach().clone(),
            "completed_obstacle_collision": obstacle_collision[completed_ids].detach().clone(),
            "completed_outside": outside[completed_ids].detach().clone(),
            "completed_timeouts": completed_timeouts,
            "log": {
                **{f"reward_{name}": value.mean().detach() for name, value in components.items()},
                "distance_to_goal": distance.mean().detach(),
                "success_transition_rate": goal.float().mean().detach(),
                "collision_transition_rate": collision.float().mean().detach(),
                "obstacle_collision_transition_rate": obstacle_collision.float().mean().detach(),
                "wall_collision_transition_rate": outside.float().mean().detach(),
                "fall_transition_rate": fallen.float().mean().detach(),
                "reverse_speed": torch.relu(-vx_measured).mean().detach(),
                "forward_speed": torch.relu(vx_measured).mean().detach(),
                "heading_error_abs": beta.abs().mean().detach(),
                "cbf_active_rate": safety.active.float().mean().detach(),
                "cbf_correction_norm": safety.correction_norm.mean().detach(),
                "normalized_cbf_correction": normalized_correction.mean().detach(),
            },
        }
        return obs, reward, done, extras
