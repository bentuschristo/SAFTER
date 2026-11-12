from __future__ import annotations

from typing import Any

import torch

from .arena import obstacle_tensors
from .config import HighLevelNavigationConfig
from .lse_safety_filter import LseCbfSafetyFilter
from .math_utils import quat_wxyz_to_rpy, world_to_body

try:
    from tensordict import TensorDict
except ImportError:  # pragma: no cover
    TensorDict = None


class HighLevelNavigationEnv:
    """RSL-RL-style vector environment around a frozen low-level actor.

    The underlying Unitree-Go2-Flat environment still advances at 50 Hz. One
    high-level step holds a safe velocity command and performs five low-level
    policy/environment steps by default.
    """

    def __init__(self, low_env: Any, low_policy, command_interface, cfg: HighLevelNavigationConfig) -> None:
        self.low_env = low_env
        self.unwrapped = low_env.unwrapped if hasattr(low_env, "unwrapped") else low_env
        self.low_policy = low_policy
        self.command_interface = command_interface

        # High-level navigation settings.
        self.nav_cfg = cfg

        # Keep the underlying MJLab config here so the native viewer
        # can access high_env.cfg.viewer.
        self.cfg = low_env.cfg
        self.device = self.unwrapped.device
        self.num_envs = self.unwrapped.num_envs
        self.num_actions = 3
        self.max_episode_length = cfg.max_episode_steps
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.previous_safe_command = torch.zeros(self.num_envs, 3, device=self.device)
        self.previous_goal_distance = torch.zeros(self.num_envs, device=self.device)
        self.goal_positions = torch.zeros(self.num_envs, 2, device=self.device)
        self.obstacle_centers, self.obstacle_radii, self.obstacle_mask = obstacle_tensors(
            cfg.arena, self.num_envs, self.device
        )
        self.safety_filter = LseCbfSafetyFilter(cfg.arena, cfg.safety)
        self._last_safety = None

        # Optional terminal diagnostics used during deployment.
        self.print_high_level_stats = False
        self.print_every = 10
        self.print_full_observation = False
        self._deployment_step = 0

        # Interfaces expected by RSL-RL/mjlab runners.
        try:
            import gymnasium as gym
            import numpy as np
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (cfg.observation_dim,), dtype=np.float32)
            self.action_space = gym.spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        except Exception:
            self.observation_space = None
            self.action_space = None

        # Register the goal marker with MJLab's native debug visualizer.
        self._install_goal_visualizer()

        self.reset()



    def _install_goal_visualizer(self) -> None:
        """Register the high-level goal visualization on the MJLab environment."""
        target_env = self.low_env.unwrapped

        # Preserve any visualization callback already provided by MJLab.
        previous_callback = getattr(target_env, "update_visualizers", None)

        if getattr(target_env, "_goal_visualizer_installed", False):
            return

        def combined_visualizer(visualizer) -> None:
            if previous_callback is not None:
                previous_callback(visualizer)

            self.update_visualizers(visualizer)

        target_env.update_visualizers = combined_visualizer
        target_env._goal_visualizer_installed = True

    def update_visualizers(self, visualizer) -> None:
        """Draw the current navigation goal as a flat yellow cylinder."""
        if visualizer.show_all_envs:
            env_ids = range(self.num_envs)
        else:
            env_ids = [visualizer.env_idx]

        for env_id in env_ids:
            goal_xy = self.goal_positions[env_id]

            z_bottom = goal_xy.new_tensor(0.005)
            z_top = goal_xy.new_tensor(0.025)

            start = torch.stack(
                (
                    goal_xy[0],
                    goal_xy[1],
                    z_bottom,
                )
            )
            end = torch.stack(
                (
                    goal_xy[0],
                    goal_xy[1],
                    z_top,
                )
            )

            visualizer.add_cylinder(
                start=start,
                end=end,
                radius=float(self.nav_cfg.goal_tolerance),
                color=(1.0, 0.85, 0.0, 0.90),
                label="navigation_goal",
            )

    def __getattr__(self, name):
        """Delegate unknown attributes to the underlying MJLab environment."""
        low_env = self.__dict__.get("low_env")

        if low_env is not None and hasattr(low_env, name):
            return getattr(low_env, name)

        raise AttributeError(
            f"{type(self).__name__} has no attribute {name!r}"
        )

    def _robot_data(self):
        return self.unwrapped.scene["robot"].data

    def _root_state(self):
        data = self._robot_data()

        pos = getattr(data, "root_link_pos_w", None)
        quat = getattr(data, "root_link_quat_w", None)
        vel_b = getattr(data, "root_link_lin_vel_b", None)
        ang_b = getattr(data, "root_link_ang_vel_b", None)

        if any(value is None for value in (pos, quat, vel_b, ang_b)):
            raise AttributeError(
                "Could not locate the required root-link state tensors."
            )

        roll, pitch, yaw = quat_wxyz_to_rpy(quat)

        return pos, quat, vel_b, ang_b, roll, pitch, yaw

    def _pack_obs(self, actor: torch.Tensor, critic: torch.Tensor | None = None):
        critic = actor if critic is None else critic
        if TensorDict is not None:
            return TensorDict({"actor": actor, "critic": critic}, batch_size=[self.num_envs], device=self.device)
        return {"actor": actor, "critic": critic}

    def _compute_observation(self):
        pos, _, vel_b, ang_b, _, _, yaw = self._root_state()
        p = pos[:, :2]
        goal_b = world_to_body(self.goal_positions - p, yaw) / self.nav_cfg.normalize_positions_by

        rel_obs_w = self.obstacle_centers - p[:, None, :]
        yaw_expanded = yaw[:, None].expand(-1, self.nav_cfg.arena.max_obstacles)
        rel_obs_b = world_to_body(rel_obs_w, yaw_expanded) / self.nav_cfg.normalize_positions_by
        radii = self.obstacle_radii[..., None]
        mask = self.obstacle_mask.float()[..., None]
        obstacle_features = torch.cat((rel_obs_b, radii, mask), dim=-1).flatten(start_dim=1)

        walls = torch.stack(
            (
                p[:, 0] - self.nav_cfg.arena.x_min,
                self.nav_cfg.arena.x_max - p[:, 0],
                p[:, 1] - self.nav_cfg.arena.y_min,
                self.nav_cfg.arena.y_max - p[:, 1],
            ),
            dim=-1,
        ) / self.nav_cfg.normalize_positions_by

        velocity = torch.cat((vel_b[:, :2], ang_b[:, 2:3]), dim=-1)
        obs = torch.cat((goal_b, velocity, self.previous_safe_command, obstacle_features, walls), dim=-1)
        if obs.shape[1] != self.nav_cfg.observation_dim:
            raise RuntimeError(f"Observation dimension {obs.shape[1]} != configured {self.nav_cfg.observation_dim}")
        return self._pack_obs(obs)

    def _sample_free_positions(self, env_ids: torch.Tensor) -> torch.Tensor:
        n = env_ids.numel()
        cfg = self.nav_cfg
        out = torch.empty(n, 2, device=self.device)
        valid = torch.zeros(n, dtype=torch.bool, device=self.device)
        x_lo = cfg.arena.x_min + cfg.sample_margin_from_walls
        x_hi = cfg.arena.x_max - cfg.sample_margin_from_walls
        y_lo = cfg.arena.y_min + cfg.sample_margin_from_walls
        y_hi = cfg.arena.y_max - cfg.sample_margin_from_walls
        centers = self.obstacle_centers[env_ids]
        radii = self.obstacle_radii[env_ids]
        mask = self.obstacle_mask[env_ids]
        min_dist = radii + cfg.arena.obstacle_clearance + cfg.spawn_extra_clearance

        for _ in range(cfg.max_sampling_attempts):
            cand = torch.empty(n, 2, device=self.device)
            cand[:, 0].uniform_(x_lo, x_hi)
            cand[:, 1].uniform_(y_lo, y_hi)
            d = torch.linalg.norm(cand[:, None, :] - centers, dim=-1)
            safe = torch.all((d >= min_dist) | (~mask), dim=-1)
            take = (~valid) & safe
            out[take] = cand[take]
            valid |= take
            if bool(valid.all()):
                return out
        raise RuntimeError("Failed to sample collision-free positions.")

    def _refresh_state_buffers(self) -> None:
        """Synchronize MuJoCo and MJLab state after a manual reset."""
        scene = self.unwrapped.scene
        sim = self.unwrapped.sim

        # Match MJLab's reset pipeline:
        #   reset/write state -> write scene data -> forward dynamics
        write_data = getattr(scene, "write_data_to_sim", None)
        if callable(write_data):
            write_data()

        forward = getattr(sim, "forward", None)
        if not callable(forward):
            raise RuntimeError(
                "The MJLab simulator does not expose sim.forward()."
            )

        forward()

        # Refresh MJLab entity timestamps after MuJoCo derived quantities
        # such as xpos/xquat have been recomputed.
        physics_dt = getattr(self.unwrapped, "physics_dt", None)

        if physics_dt is None:
            cfg = getattr(self.unwrapped, "cfg", None)
            sim_cfg = getattr(cfg, "sim", None)
            physics_dt = getattr(sim_cfg, "dt", None)

        if physics_dt is None:
            physics_dt = 0.0

        scene_update = getattr(scene, "update", None)
        if callable(scene_update):
            scene_update(float(physics_dt))

    def _write_root_poses(self, env_ids: torch.Tensor, xy: torch.Tensor, yaw: torch.Tensor) -> None:
        robot = self.unwrapped.scene["robot"]
        data = robot.data
        default = getattr(data, "default_root_state", None)
        if default is None:
            raise AttributeError("robot.data.default_root_state is required for randomized resets.")
        root = default[env_ids].clone()
        root[:, 0:2] = xy
        root[:, 3] = torch.cos(yaw / 2.0)
        root[:, 4] = 0.0
        root[:, 5] = 0.0
        root[:, 6] = torch.sin(yaw / 2.0)
        root[:, 7:13] = 0.0
        if hasattr(robot, "write_root_state_to_sim"):
            robot.write_root_state_to_sim(root, env_ids=env_ids)
        else:
            robot.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
            robot.write_root_velocity_to_sim(root[:, 7:13], env_ids=env_ids)

        self._refresh_state_buffers()

    def _reset_selected(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        starts = self._sample_free_positions(env_ids)
        goals = self._sample_free_positions(env_ids)
        for _ in range(self.nav_cfg.max_sampling_attempts):
            too_close = torch.linalg.norm(goals - starts, dim=-1) < self.nav_cfg.minimum_start_goal_distance
            if not bool(too_close.any()):
                break
            goals[too_close] = self._sample_free_positions(env_ids[too_close])
        yaw = torch.empty(env_ids.numel(), device=self.device).uniform_(-torch.pi, torch.pi)
        self._write_root_poses(env_ids, starts, yaw)
        self.goal_positions[env_ids] = goals

        self.previous_safe_command[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.previous_goal_distance[env_ids] = torch.linalg.norm(goals - starts, dim=-1)

    def reset(self):
        self.low_env.reset()
        ids = torch.arange(self.num_envs, device=self.device)
        self._reset_selected(ids)
        return self._compute_observation()

    def get_observations(self):
        return self._compute_observation()

    def _print_deployment_stats(
        self,
        *,
        action: torch.Tensor,
        nominal: torch.Tensor,
        safe: torch.Tensor,
        position: torch.Tensor,
        distance: torch.Tensor,
        progress: torch.Tensor,
        smoothness: torch.Tensor,
        reward: torch.Tensor,
        goal: torch.Tensor,
        collision: torch.Tensor,
        outside: torch.Tensor,
        fallen: torch.Tensor,
        timeout: torch.Tensor,
        safety,
    ) -> None:
        """Print compact high-level navigation diagnostics for environment zero."""
        if not self.print_high_level_stats:
            return

        self._deployment_step += 1

        env_id = 0
        terminal_event = bool(
            goal[env_id]
            or collision[env_id]
            or outside[env_id]
            or fallen[env_id]
            or timeout[env_id]
        )

        interval = max(int(self.print_every), 1)
        if self._deployment_step % interval != 0 and not terminal_event:
            return

        r = self.nav_cfg.reward

        progress_reward = r.progress_weight * progress[env_id]
        goal_reward = r.goal_bonus * goal[env_id].float()
        time_reward = reward.new_tensor(-float(r.time_penalty))
        smoothness_reward = -r.smoothness_weight * smoothness[env_id]
        collision_reward = -r.collision_penalty * collision[env_id].float()
        boundary_reward = -r.boundary_penalty * outside[env_id].float()

        if safety is not None:
            cbf_active = bool(safety.active[env_id])
            correction = float(safety.correction_norm[env_id])
        else:
            cbf_active = False
            correction = 0.0

        def vector_string(tensor: torch.Tensor, precision: int = 3) -> str:
            values = tensor.detach().cpu().flatten().tolist()
            return "[" + ", ".join(
                f"{float(value):.{precision}f}" for value in values
            ) + "]"

        print(
            "\n"
            f"[HIGH] global_step={self._deployment_step}  "
            f"episode_step={int(self.episode_length_buf[env_id])}\n"
            f"  pos={vector_string(position[env_id])}  "
            f"goal={vector_string(self.goal_positions[env_id])}  "
            f"distance={float(distance[env_id]):.3f}\n"
            f"  action_norm={vector_string(action[env_id])}\n"
            f"  nominal={vector_string(nominal[env_id])}\n"
            f"  safe={vector_string(safe[env_id])}\n"
            f"  cbf_active={cbf_active}  correction={correction:.4f}\n"
            f"  reward={float(reward[env_id]):+.4f}\n"
            f"    progress={float(progress_reward):+.4f}  "
            f"goal={float(goal_reward):+.4f}  "
            f"time={float(time_reward):+.4f}\n"
            f"    smoothness={float(smoothness_reward):+.4f}  "
            f"collision={float(collision_reward):+.4f}  "
            f"boundary={float(boundary_reward):+.4f}  "
            f"fall=+0.0000\n"
            f"  done: goal={bool(goal[env_id])}  "
            f"collision={bool(collision[env_id])}  "
            f"outside={bool(outside[env_id])}  "
            f"fall={bool(fallen[env_id])}  "
            f"timeout={bool(timeout[env_id])}",
            flush=True,
        )

        if self.print_full_observation:
            obs = self._compute_observation()

            if hasattr(obs, "get"):
                actor_obs = obs.get("actor")
            else:
                actor_obs = obs["actor"]

            print(
                f"  observation[48]={vector_string(actor_obs[env_id], 4)}",
                flush=True,
            )

    def _scale_action(self, action: torch.Tensor) -> torch.Tensor:
        action = torch.clamp(action, -1.0, 1.0)
        scale = torch.tensor(
            [self.nav_cfg.action.max_vx, self.nav_cfg.action.max_vy, self.nav_cfg.action.max_wz],
            device=self.device,
            dtype=action.dtype,
        )
        return action * scale

    def step(self, action: torch.Tensor):
        nominal = self._scale_action(action)
        pos, _, _, _, _, _, yaw = self._root_state()
        p = pos[:, :2]
        if self.nav_cfg.safety.enabled:
            safety = self.safety_filter.filter(
                p, yaw, nominal, self.obstacle_centers, self.obstacle_radii, self.obstacle_mask
            )
            safe = safety.command
        else:
            safe = nominal
            safety = None

        # Keep applied commands in the locomotion policy's trained range.
        safe[:, 0].clamp_(-self.nav_cfg.action.max_vx, self.nav_cfg.action.max_vx)
        safe[:, 1].clamp_(-self.nav_cfg.action.max_vy, self.nav_cfg.action.max_vy)
        safe[:, 2].clamp_(-self.nav_cfg.action.max_wz, self.nav_cfg.action.max_wz)

        for _ in range(self.nav_cfg.low_steps_per_high_step):
            # self.command_interface.write_batch(safe)
            low_obs = self.low_env.get_observations()
            joint_action = self.low_policy(low_obs, safe)
            self.low_env.step(joint_action)

        pos, _, _, _, roll, pitch, yaw = self._root_state()
        p = pos[:, :2]
        distance = torch.linalg.norm(self.goal_positions - p, dim=-1)
        progress = self.previous_goal_distance - distance
        goal = distance <= self.nav_cfg.goal_tolerance

        # Geometric collisions use physical obstacle radii, not CBF radii.
        d_obs = torch.linalg.norm(p[:, None, :] - self.obstacle_centers, dim=-1)
        contact_radius = getattr(
            self.nav_cfg.arena,
            "robot_radius",
            0.30,
        )
        collision_distance = self.obstacle_radii + contact_radius
        collision = torch.any(
            (d_obs <= collision_distance) & self.obstacle_mask,
            dim=-1,
        )
        contact_radius = getattr(
            self.nav_cfg.arena,
            "robot_radius",
            0.30,
        )
        wall_contact = (
            (p[:, 0] <= self.nav_cfg.arena.x_min + contact_radius)
            | (p[:, 0] >= self.nav_cfg.arena.x_max - contact_radius)
            | (p[:, 1] <= self.nav_cfg.arena.y_min + contact_radius)
            | (p[:, 1] >= self.nav_cfg.arena.y_max - contact_radius)
        )
        outside = wall_contact
        fallen = (
            (pos[:, 2] < self.nav_cfg.minimum_base_height)
            | (torch.abs(roll) > self.nav_cfg.maximum_roll_pitch)
            | (torch.abs(pitch) > self.nav_cfg.maximum_roll_pitch)
        )
        timeout = self.episode_length_buf + 1 >= self.nav_cfg.max_episode_steps

        terminated = goal | collision | outside | fallen
        truncated = timeout & ~terminated
        done = terminated | truncated

        smoothness = torch.sum((safe - self.previous_safe_command) ** 2, dim=-1)
        r = self.nav_cfg.reward
        reward = (
            r.progress_weight * progress
            + r.goal_bonus * goal.float()
            - r.time_penalty

            - r.timeout_penalty * truncated.float()
            - r.smoothness_weight * smoothness
            - r.collision_penalty * collision.float()
            - r.boundary_penalty * outside.float()
        )

        self._print_deployment_stats(
            action=action,
            nominal=nominal,
            safe=safe,
            position=p,
            distance=distance,
            progress=progress,
            smoothness=smoothness,
            reward=reward,
            goal=goal,
            collision=collision,
            outside=outside,
            fallen=fallen,
            timeout=timeout,
            safety=safety,
        )

        self.previous_safe_command = safe.clone()
        self.previous_goal_distance = distance
        self.episode_length_buf += 1
        self._last_safety = safety

        extras = {
            "observations": self._compute_observation(),
            "time_outs": truncated,
            "log": {
                "reward_progress": progress.mean(),
                "success_rate": goal.float().mean(),
                "collision_rate": collision.float().mean(),
                "fall_rate": fallen.float().mean(),
                "distance_to_goal": distance.mean(),
                "smoothness_cost": smoothness.mean(),
                "cbf_active_rate": (
                    safety.active.float().mean() if safety is not None else torch.tensor(0.0, device=self.device)
                ),
                "cbf_correction_norm": (
                    safety.correction_norm.mean() if safety is not None else torch.tensor(0.0, device=self.device)
                ),
            },
        }

        ids = torch.nonzero(done, as_tuple=False).flatten()
        if ids.numel() > 0:
            self._reset_selected(ids)

        obs = self._compute_observation()
        # RSL-RL VecEnv convention used by mjlab wrapper: obs, reward, done, extras.
        return obs, reward, done, extras

    def close(self):
        self.low_env.close()
