from __future__ import annotations

from collections import deque
from typing import Any

import torch

try:
    # Local repo state: this file lives under SAFTER_src/, and imports its
    # own sibling high_level_rl package from there too.
    from SAFTER_src.high_level_rl.laser_cbf_filters import build_invariant_ellipse_bcbf_filter
    from SAFTER_src.high_level_rl.math_utils import world_to_body
    from SAFTER_src.high_level_rl.navigation_env import HighLevelNavigationEnv
    from SAFTER_src.high_level_rl.raycast_patterns import horizontal_ray_angles
except ModuleNotFoundError:
    # Server state: this same file is deployed at src/high_level_laser_sac/
    # (the original, pre-rename location) -- fall back to src.high_level_rl.
    from src.high_level_rl.laser_cbf_filters import build_invariant_ellipse_bcbf_filter
    from src.high_level_rl.math_utils import world_to_body
    from src.high_level_rl.navigation_env import HighLevelNavigationEnv
    from src.high_level_rl.raycast_patterns import horizontal_ray_angles

from .config import (
    LIDAR_NOISE_BOUNDS_M,
    CollisionInitializationConfig,
    ContactCollisionConfig,
    LaserObservationConfig,
    LaserRewardConfig,
    LaserSafetyConfig,
)
from .procedural_maps import CURRICULUM_WEIGHTS, DIFFICULTIES, ProceduralMapBank, resolve_slot_shapes


def seanav_comparable_terminal_terms(
    *,
    distance: torch.Tensor,
    planar_velocity: torch.Tensor,
    yaw_rate: torch.Tensor,
    collision: torch.Tensor,
    fallen: torch.Tensor,
    truncated: torch.Tensor,
    reward_cfg: LaserRewardConfig,
    policy_dt: float,
) -> dict[str, torch.Tensor]:
    """SEA-Nav-scaled terms that have direct hierarchical counterparts.

    SEA-Nav's collision reward counts contacts by body group. This MuJoCo port
    exposes one generic non-foot physical-contact event rather than summing
    individual contacting bodies. The velocity multiplier, tight-goal
    equation, termination scale, timeout semantics, and dt scaling match the
    released equations.
    """
    tight_goal = (
        1.0 / (1.0 + 2.0 * distance.square())
    ) * (distance < reward_cfg.seanav_goal_distance_threshold_m).float()
    speed_square = planar_velocity.square().sum(dim=-1) + yaw_rate.square()
    unsafe_termination = collision | fallen
    dt = float(policy_dt)
    return {
        "goal": dt * reward_cfg.seanav_goal_scale * tight_goal,
        "timeout": dt * reward_cfg.seanav_timeout_scale * truncated.float(),
        "collision": (
            dt
            * reward_cfg.seanav_collision_scale
            * (1.0 + reward_cfg.seanav_collision_velocity_scale * speed_square)
            * collision.float()
        ),
        "termination": (
            dt
            * reward_cfg.seanav_termination_scale
            * unsafe_termination.float()
        ),
    }


class LaserSacNavigationEnv(HighLevelNavigationEnv):
    """High-level navigation env with laser-range observations only.

    The simulator still owns the arena map for collision/reset handling, but
    actor observations expose MuJoCo RayCastSensor ranges instead of obstacle
    centers, obstacle radii, and wall distances.
    """

    def __init__(
        self,
        *args: Any,
        reward_cfg: LaserRewardConfig,
        laser_cfg: LaserObservationConfig,
        safety_cfg: LaserSafetyConfig | None = None,
        contact_collision_cfg: ContactCollisionConfig | None = None,
        procedural_map_bank: ProceduralMapBank | None = None,
        procedural_map_entity_names: tuple[str, ...] | None = None,
        curriculum_recent_window: int = 10_000,
        curriculum_mode: str = "quarters",
        collision_initialization_cfg: CollisionInitializationConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self.sac_reward_cfg = reward_cfg
        self.laser_obs_cfg = laser_cfg
        self.laser_safety_cfg = safety_cfg or LaserSafetyConfig()
        self.contact_collision_cfg = contact_collision_cfg or ContactCollisionConfig()
        self.collision_initialization_cfg = (
            collision_initialization_cfg or CollisionInitializationConfig()
        )
        if curriculum_mode not in ("quarters", "seanav_adaptive"):
            raise ValueError(f"Unknown curriculum_mode={curriculum_mode!r}")
        self.curriculum_mode = curriculum_mode
        self._fixed_scenario = None
        self._forced_scenario_ids = None
        self._curriculum_progress = 0.0
        self._obstacle_shape = "circle"
        procedural_maps = procedural_map_bank.maps if procedural_map_bank is not None else None
        self.procedural_map_bank = procedural_map_bank
        self._procedural_maps = procedural_maps
        self._procedural_map_entity_names = procedural_map_entity_names
        if (procedural_maps is None) != (procedural_map_entity_names is None):
            raise ValueError("Procedural maps and entity names must be provided together")
        # Per-slot obstacle shape ("circle"/"square"), resolved unconditionally
        # regardless of whether this env uses a procedural map bank or a
        # single fixed arena (nav_cfg.arena) -- both paths share the same
        # analytical collision-clearance math further below, which is the
        # only place that needs to know per-slot shape (the LiDAR-based bCBF
        # barrier itself is a true raycast against the physical MuJoCo/arena
        # geometry and is already shape-agnostic).
        low_env = args[0]
        map_device = low_env.unwrapped.device if hasattr(low_env, "unwrapped") else low_env.device
        if procedural_maps is not None:
            max_obstacles = max(item.arena.max_obstacles for item in procedural_maps)
            self._slot_shapes = resolve_slot_shapes(procedural_maps, max_obstacles)
        else:
            fixed_arena = args[3].arena
            self._slot_shapes = tuple(
                (fixed_arena.obstacles[i].shape if i < len(fixed_arena.obstacles) else "circle")
                for i in range(fixed_arena.max_obstacles)
            )
        distinct_shapes = set(self._slot_shapes)
        self._obstacle_shape = distinct_shapes.pop() if len(distinct_shapes) == 1 else "mixed"
        self._any_square_slot = "square" in self._slot_shapes
        self._slot_is_square = torch.tensor(
            [shape == "square" for shape in self._slot_shapes], dtype=torch.bool, device=map_device
        )
        if procedural_maps is not None:
            if len(procedural_map_entity_names) != procedural_map_bank.max_obstacles:
                raise ValueError("Reusable procedural scene must contain exactly max_obstacles entities")
            max_obstacles = max(item.arena.max_obstacles for item in procedural_maps)
            self._map_centers = torch.zeros(len(procedural_maps), max_obstacles, 2, device=map_device)
            self._map_radii = torch.zeros(len(procedural_maps), max_obstacles, device=map_device)
            self._map_heights = torch.zeros(len(procedural_maps), max_obstacles, device=map_device)
            self._map_masks = torch.zeros(
                len(procedural_maps), max_obstacles, dtype=torch.bool, device=map_device
            )
            self._map_obstacle_counts = torch.empty(len(procedural_maps), device=map_device)
            self._map_id_to_index = {item.map_id: i for i, item in enumerate(procedural_maps)}
            scenarios = sorted(
                (scenario for item in procedural_maps for scenario in item.scenarios),
                key=lambda scenario: scenario.scenario_id,
            )
            self._scenarios = tuple(scenarios)
            self._scenario_map_indices = torch.tensor(
                [self._map_id_to_index[s.map_id] for s in scenarios], dtype=torch.long, device=map_device
            )
            # Keep scenario poses on the simulation device.  Constructing a
            # fresh CUDA tensor from each Python tuple during every reset is
            # both needlessly synchronous and has produced intermittent
            # ``Expected a CUDA device, but got cpu:0`` failures with recent
            # PyTorch/CUDA builds.
            self._scenario_starts = torch.tensor(
                [s.start_xy for s in scenarios], dtype=torch.float32, device=map_device
            )
            self._scenario_goals = torch.tensor(
                [s.goal_xy for s in scenarios], dtype=torch.float32, device=map_device
            )
            self._scenario_start_yaws = torch.tensor(
                [s.start_yaw for s in scenarios], dtype=torch.float32, device=map_device
            )
            self._scenario_difficulty_index = torch.tensor(
                [DIFFICULTIES.index(s.difficulty) for s in scenarios], dtype=torch.long, device=map_device
            )
            self._scenario_difficulty_scores = torch.tensor(
                [s.difficulty_score for s in scenarios], device=map_device
            )
            for map_index, item in enumerate(procedural_maps):
                self._map_obstacle_counts[map_index] = len(item.arena.obstacles)
                for obstacle_index, obstacle in enumerate(item.arena.obstacles):
                    self._map_centers[map_index, obstacle_index] = torch.tensor(
                        (obstacle.x, obstacle.y), device=map_device
                    )
                    self._map_radii[map_index, obstacle_index] = obstacle.radius
                    self._map_heights[map_index, obstacle_index] = obstacle.height
                    self._map_masks[map_index, obstacle_index] = True
            self._scenario_difficulty_pools = [
                torch.nonzero(self._scenario_difficulty_index == i, as_tuple=False).flatten()
                for i in range(len(DIFFICULTIES))
            ]
            score_order = torch.argsort(self._scenario_difficulty_scores)
            self._seanav_level_pools = tuple(torch.tensor_split(score_order, 10))
            if any(pool.numel() == 0 for pool in self._seanav_level_pools):
                raise ValueError("SEA-Nav adaptive curriculum requires at least 10 scenarios")
            self.seanav_curriculum_levels = torch.randint(
                0, 2, (low_env.num_envs,), dtype=torch.long, device=map_device
            )
            # SEA-Nav maintains a second, floating-point goal level.  It is
            # updated alongside the terrain row but does not wrap at the top;
            # collision-state initialization uses this value rather than the
            # terrain row to set its early-reset probability.
            self.seanav_goal_levels = torch.zeros(
                low_env.num_envs, dtype=torch.float32, device=map_device
            )
            self._scenario_cycle_orders = [torch.empty(0, dtype=torch.long, device=map_device) for _ in DIFFICULTIES]
            self._scenario_cycle_cursors = [0, 0, 0]
            self.map_assignment_counts = torch.zeros(len(procedural_maps), dtype=torch.long, device=map_device)
            self.scenario_assignment_counts = torch.zeros(procedural_map_bank.num_scenarios, dtype=torch.long, device=map_device)
            self.difficulty_assignment_counts = torch.zeros(3, dtype=torch.long, device=map_device)
            self.stage_difficulty_assignment_counts = torch.zeros(3, dtype=torch.long, device=map_device)
            self.quarter_difficulty_assignment_counts = torch.zeros(4, 3, dtype=torch.long, device=map_device)
            self.quarter_assignment_counts = torch.zeros(4, dtype=torch.long, device=map_device)
            self._recent_assignments = deque(maxlen=int(curriculum_recent_window))
            self.assignment_count = 0
            self._slot_geom_ids = tuple(
                int(low_env.unwrapped.scene[name].indexing.geom_ids[0])
                for name in self._procedural_map_entity_names
            )

        # --- Reset-time h_aug(z0) feasibility guard (Task 8, 2026-09 fix).
        # Built BEFORE super().__init__(), which itself calls self.reset() and
        # therefore _reset_selected() before returning control here. A
        # procedurally-sampled scenario must satisfy the SAME invariant-
        # ellipse feasibility criterion the bCBF-CF filter enforces forward in
        # time, evaluated at the actual reset state (zero command, zero
        # velocity) with the actual initial LiDAR scan. This uses the exact
        # same filter construction/computation as the live actor's own
        # bcbf_cf branch (via build_invariant_ellipse_bcbf_filter) -- never a
        # reimplementation -- so this is a scenario-selection guard, not a
        # relaxation or modification of the terminal-set theorem itself.
        self.reset_feasibility_max_attempts = 20
        self.reset_feasibility_checked_count = 0
        self.reset_feasibility_infeasible_count = 0
        self.reset_feasibility_exhausted_count = 0
        if procedural_maps is not None:
            nav_cfg = args[3]
            action_limits = (
                nav_cfg.action.max_vx,
                nav_cfg.action.max_vy,
                nav_cfg.action.max_wz,
            )
            self._reset_feasibility_filter = build_invariant_ellipse_bcbf_filter(
                self.laser_safety_cfg, action_limits, self.laser_obs_cfg.max_range,
                min_vx=nav_cfg.action.min_vx,
            )
        else:
            self._reset_feasibility_filter = None

        super().__init__(*args, **kwargs)
        if abs(
            float(self.laser_safety_cfg.integration_dt)
            - float(self.navigation_cfg.high_level_dt)
        ) > 1.0e-12:
            raise ValueError(
                "Safety command-dynamics integration_dt must equal navigation "
                "high_level_dt: "
                f"{self.laser_safety_cfg.integration_dt} != "
                f"{self.navigation_cfg.high_level_dt}"
            )
        # NOTE: the laser refresh-cache state (_cached_laser_ranges etc.) is
        # NOT created here. super().__init__() above already ran self.reset()
        # internally, which (via _reset_selected -> _enforce_reset_feasibility)
        # may already have called laser_ranges() before this line ever runs.
        # It lazily self-initializes on first use instead -- see
        # _ensure_laser_refresh_state().
        self._ensure_laser_refresh_state()

        self.episode_return_buf = torch.zeros(self.num_envs, device=self.device)
        self.episode_laser_safety_violation_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.previous_nominal_command = torch.zeros(self.num_envs, 3, device=self.device)
        self.nominal_history_valid = torch.zeros(self.num_envs, 1, device=self.device)
        self.episode_success_count = 0
        self.episode_collision_count = 0
        self.episode_count = 0
        ci = self.collision_initialization_cfg
        if not (0.0 <= ci.replay_probability <= 1.0):
            raise ValueError("collision replay_probability must be in [0, 1]")
        if not (
            0.0 <= ci.early_reset_probability_min
            <= ci.early_reset_probability_max <= 1.0
        ):
            raise ValueError("collision early-reset probabilities must satisfy 0 <= min <= max <= 1")
        self._collision_init_history_steps = max(
            2, int(round(ci.undo_time_max_s / self.navigation_cfg.high_level_dt)) + 1
        )
        robot = self.unwrapped.scene["robot"]
        self._ci_root_history = torch.zeros(
            self._collision_init_history_steps, self.num_envs, 13, device=self.device
        )
        self._ci_joint_pos_history = torch.zeros(
            self._collision_init_history_steps, *robot.data.joint_pos.shape, device=self.device
        )
        self._ci_joint_vel_history = torch.zeros_like(self._ci_joint_pos_history)
        self._ci_safe_command_history = torch.zeros(
            self._collision_init_history_steps, self.num_envs, 3, device=self.device
        )
        self._ci_nominal_command_history = torch.zeros_like(self._ci_safe_command_history)
        self._ci_history_write = 0
        self.collision_initialization_attempts = 0
        self.collision_initialization_applied = 0
        self.collision_initialization_early_resets = 0
        self._last_collision_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        if self._procedural_maps is not None:
            expected = self.procedural_map_bank.max_obstacles
            if self.obstacle_centers.shape != (self.num_envs, expected, 2):
                raise RuntimeError(f"Obstacle centers shape {self.obstacle_centers.shape} != ({self.num_envs}, {expected}, 2)")
            if self.obstacle_radii.shape != (self.num_envs, expected) or self.obstacle_mask.shape != (self.num_envs, expected):
                raise RuntimeError("Obstacle radii/mask do not have the configured 12-slot capacity")

    def set_training_progress(self, progress: float) -> bool:
        """Update curriculum progress and report whether its quarter changed."""
        old_stage = self.curriculum_stage
        self._curriculum_progress = min(max(float(progress), 0.0), 1.0)
        changed = self.curriculum_stage != old_stage
        if changed and hasattr(self, "stage_difficulty_assignment_counts"):
            self.stage_difficulty_assignment_counts.zero_()
        return changed

    @property
    def curriculum_stage(self) -> int:
        return min(int(self._curriculum_progress * 4.0), 3)

    def _requested_difficulty_weights(self) -> tuple[float, float, float]:
        mode = self.procedural_map_bank.difficulty_mode
        if mode == "mixed":
            return CURRICULUM_WEIGHTS[self.curriculum_stage]
        return tuple(1.0 if name == mode else 0.0 for name in DIFFICULTIES)

    def _sample_procedural_scenarios(
        self, count: int, env_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._forced_scenario_ids is not None:
            # Deterministic evaluation override (set_fixed_procedural_scenarios):
            # bypass curriculum/random sampling entirely and use exactly the
            # given scenario id per env slot, so a (seed, case_id) manifest
            # reproduces the identical map+start+goal every time.
            if env_ids is None:
                raise ValueError("Forced procedural scenarios require environment ids")
            selected = self._forced_scenario_ids[env_ids]
            return selected, self._scenario_difficulty_index[selected]
        if self.curriculum_mode == "seanav_adaptive":
            if env_ids is None:
                raise ValueError("SEA-Nav adaptive sampling requires environment ids")
            levels = self.seanav_curriculum_levels[env_ids]
            selected = torch.empty(count, dtype=torch.long, device=self.device)
            for level, pool in enumerate(self._seanav_level_pools):
                mask = levels == level
                take = int(mask.sum())
                if take:
                    selected[mask] = pool[torch.randint(pool.numel(), (take,), device=self.device)]
            return selected, self._scenario_difficulty_index[selected]
        weights = torch.tensor(self._requested_difficulty_weights(), device=self.device)
        difficulties = torch.multinomial(weights, count, replacement=True)
        selected = torch.empty(count, dtype=torch.long, device=self.device)
        for difficulty_index, pool in enumerate(self._scenario_difficulty_pools):
            mask = difficulties == difficulty_index
            sample_count = int(mask.sum())
            if sample_count:
                chunks = []
                remaining = sample_count
                while remaining:
                    cursor = self._scenario_cycle_cursors[difficulty_index]
                    order = self._scenario_cycle_orders[difficulty_index]
                    if cursor >= order.numel():
                        order = torch.randperm(pool.numel(), device=self.device)
                        self._scenario_cycle_orders[difficulty_index] = order
                        cursor = 0
                    take = min(remaining, order.numel() - cursor)
                    chunks.append(pool[order[cursor:cursor + take]])
                    self._scenario_cycle_cursors[difficulty_index] = cursor + take
                    remaining -= take
                selected[mask] = torch.cat(chunks)
        return selected, difficulties

    def _activate_procedural_maps(self, env_ids: torch.Tensor) -> None:
        if self._procedural_maps is None or env_ids.numel() == 0:
            return
        scenario_ids, difficulty = self._sample_procedural_scenarios(env_ids.numel(), env_ids)
        selected = self._scenario_map_indices[scenario_ids]
        self.current_procedural_map[env_ids] = selected
        self.current_procedural_scenario[env_ids] = scenario_ids
        centers, radii, heights, masks = (
            self._map_centers[selected], self._map_radii[selected], self._map_heights[selected], self._map_masks[selected]
        )
        # MuJoCo-Warp environments are separate worlds.  Their env_origins are
        # viewer/layout offsets, whereas robot state, fixed arena geometry, and
        # saved bank coordinates are map-local inside each world.
        self.obstacle_centers[env_ids] = centers
        self.obstacle_radii[env_ids] = radii
        self.obstacle_mask[env_ids] = masks

        pose = torch.zeros(env_ids.numel(), 7, device=self.device)
        pose[:, 3] = 1.0
        model = self.unwrapped.sim.model
        for slot, (entity_name, geom_id) in enumerate(zip(self._procedural_map_entity_names, self._slot_geom_ids, strict=True)):
            active = masks[:, slot]
            pose[:, :2] = centers[:, slot]
            half_height = 0.5 * heights[:, slot]
            pose[:, 2] = torch.where(active, half_height, torch.full_like(half_height, -100.0))
            self.unwrapped.scene[entity_name].write_mocap_pose_to_sim(pose, env_ids=env_ids)
            radius = torch.where(active, radii[:, slot], torch.full_like(radii[:, slot], 1.0e-4))
            safe_half_height = torch.where(active, half_height, torch.full_like(half_height, 1.0e-4))
            model.geom_size[env_ids, geom_id, 0] = radius
            if self._slot_shapes[slot] == "square":
                model.geom_size[env_ids, geom_id, 1] = radius
                model.geom_size[env_ids, geom_id, 2] = safe_half_height
                model.geom_rbound[env_ids, geom_id] = torch.sqrt(2.0 * radius.square() + safe_half_height.square())
            else:
                model.geom_size[env_ids, geom_id, 1] = safe_half_height
                model.geom_rbound[env_ids, geom_id] = torch.sqrt(radius.square() + safe_half_height.square())
            model.geom_aabb[env_ids, geom_id, 0, :] = 0.0
            model.geom_aabb[env_ids, geom_id, 1, 0] = radius
            model.geom_aabb[env_ids, geom_id, 1, 1] = radius
            model.geom_aabb[env_ids, geom_id, 1, 2] = safe_half_height
        self.map_assignment_counts += torch.bincount(selected, minlength=len(self._procedural_maps))
        self.difficulty_assignment_counts += torch.bincount(difficulty, minlength=3)
        self.stage_difficulty_assignment_counts += torch.bincount(difficulty, minlength=3)
        self.quarter_difficulty_assignment_counts[self.curriculum_stage] += torch.bincount(difficulty, minlength=3)
        self.quarter_assignment_counts[self.curriculum_stage] += env_ids.numel()
        self.scenario_assignment_counts += torch.bincount(scenario_ids, minlength=self.scenario_assignment_counts.numel())
        self._recent_assignments.extend(difficulty.cpu().tolist())
        self.assignment_count += env_ids.numel()

    def curriculum_metrics(self) -> dict[str, float]:
        total = max(self.assignment_count, 1)
        weights = self._requested_difficulty_weights()
        weighted_maps = self.map_assignment_counts.float()
        stage_total = max(int(self.stage_difficulty_assignment_counts.sum()), 1)
        recent = torch.bincount(torch.tensor(list(self._recent_assignments)), minlength=3).float() if self._recent_assignments else torch.zeros(3)
        recent_total = max(len(self._recent_assignments), 1)
        weighted_scenarios = self.scenario_assignment_counts.float()
        return {
            "progress": self._curriculum_progress, "stage": float(self.curriculum_stage + 1),
            "mode_seanav_adaptive": float(self.curriculum_mode == "seanav_adaptive"),
            "seanav_mean_level": float(self.seanav_curriculum_levels.float().mean())
            if self.curriculum_mode == "seanav_adaptive" else 0.0,
            "seanav_mean_goal_level": float(self.seanav_goal_levels.mean())
            if self.curriculum_mode == "seanav_adaptive" else 0.0,
            "requested_easy_probability": weights[0], "requested_medium_probability": weights[1],
            "requested_hard_probability": weights[2],
            **{f"current_stage_realized_{name}_fraction": float(self.stage_difficulty_assignment_counts[i] / stage_total) for i, name in enumerate(DIFFICULTIES)},
            **{f"cumulative_realized_{name}_fraction": float(self.difficulty_assignment_counts[i] / total) for i, name in enumerate(DIFFICULTIES)},
            **{f"recent_realized_{name}_fraction": float(recent[i] / recent_total) for i, name in enumerate(DIFFICULTIES)},
            "mean_obstacle_count": float((weighted_maps * self._map_obstacle_counts).sum() / total),
            "mean_scenario_difficulty_score": float((weighted_scenarios * self._scenario_difficulty_scores).sum() / total),
            "unique_maps_seen": float((self.map_assignment_counts > 0).sum()),
            "unique_scenarios_seen": float((self.scenario_assignment_counts > 0).sum()),
            "curriculum_truncations": 0.0,
            "collision_initialization_attempts": float(self.collision_initialization_attempts),
            "collision_initialization_applied": float(self.collision_initialization_applied),
            "collision_initialization_early_resets": float(
                self.collision_initialization_early_resets
            ),
            "reset_feasibility_checked_count": float(self.reset_feasibility_checked_count),
            "reset_feasibility_infeasible_count": float(self.reset_feasibility_infeasible_count),
            "reset_feasibility_infeasible_fraction": (
                float(self.reset_feasibility_infeasible_count) / self.reset_feasibility_checked_count
                if self.reset_feasibility_checked_count else 0.0
            ),
            "reset_feasibility_exhausted_count": float(self.reset_feasibility_exhausted_count),
        }

    def curriculum_state_dict(self) -> dict[str, object]:
        return {
            "progress": self._curriculum_progress,
            "map_assignment_counts": self.map_assignment_counts.cpu(),
            "scenario_assignment_counts": self.scenario_assignment_counts.cpu(),
            "difficulty_assignment_counts": self.difficulty_assignment_counts.cpu(),
            "stage_difficulty_assignment_counts": self.stage_difficulty_assignment_counts.cpu(),
            "quarter_difficulty_assignment_counts": self.quarter_difficulty_assignment_counts.cpu(),
            "quarter_assignment_counts": self.quarter_assignment_counts.cpu(),
            "recent_assignments": list(self._recent_assignments),
            "assignment_count": self.assignment_count,
            "curriculum_mode": self.curriculum_mode,
            "seanav_curriculum_levels": self.seanav_curriculum_levels.cpu()
            if self.curriculum_mode == "seanav_adaptive" else None,
            "seanav_goal_levels": self.seanav_goal_levels.cpu()
            if self.curriculum_mode == "seanav_adaptive" else None,
            "collision_initialization_attempts": self.collision_initialization_attempts,
            "collision_initialization_applied": self.collision_initialization_applied,
            "collision_initialization_early_resets": self.collision_initialization_early_resets,
            "scenario_cycle_orders": [x.cpu() for x in self._scenario_cycle_orders],
            "scenario_cycle_cursors": self._scenario_cycle_cursors,
        }

    def load_curriculum_state_dict(self, state: dict[str, object]) -> None:
        self._curriculum_progress = float(state.get("progress", 0.0))
        saved_mode = state.get("curriculum_mode", self.curriculum_mode)
        if saved_mode != self.curriculum_mode:
            raise ValueError(
                f"Curriculum mode mismatch: checkpoint={saved_mode}, current={self.curriculum_mode}"
            )
        levels = state.get("seanav_curriculum_levels")
        if levels is not None and self.curriculum_mode == "seanav_adaptive":
            self.seanav_curriculum_levels.copy_(levels.to(self.device))
        goal_levels = state.get("seanav_goal_levels")
        if goal_levels is not None and self.curriculum_mode == "seanav_adaptive":
            self.seanav_goal_levels.copy_(goal_levels.to(self.device))
        self.collision_initialization_attempts = int(
            state.get("collision_initialization_attempts", 0)
        )
        self.collision_initialization_applied = int(
            state.get("collision_initialization_applied", 0)
        )
        self.collision_initialization_early_resets = int(
            state.get("collision_initialization_early_resets", 0)
        )
        for name in ("map_assignment_counts", "scenario_assignment_counts", "difficulty_assignment_counts", "stage_difficulty_assignment_counts", "quarter_difficulty_assignment_counts", "quarter_assignment_counts"):
            if name in state:
                getattr(self, name).copy_(state[name].to(self.device))
        self.assignment_count = int(state.get("assignment_count", 0))
        self._recent_assignments.extend(state.get("recent_assignments", ()))
        for name in ("scenario_cycle_orders",):
            if name in state:
                setattr(self, name, [x.to(self.device) for x in state[name]])
        self._scenario_cycle_cursors = list(state.get("scenario_cycle_cursors", self._scenario_cycle_cursors))

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

    @property
    def ray_angles(self) -> torch.Tensor:
        # CBF rows are evaluated against body-frame commands [vx, vy, wz].
        # Therefore beta must be the configured local scan angle, not the
        # world-frame ray direction after MuJoCo yaw/base alignment.
        return horizontal_ray_angles(
            self.laser_obs_cfg.num_rays,
            self.laser_obs_cfg.fov_deg,
            yaw_offset_deg=self.laser_obs_cfg.yaw_offset_deg,
            left_to_right=self.laser_obs_cfg.left_to_right,
            device=self.device,
        )

    def safety_context(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos, _, _, _, _, _, yaw = self._root_state()
        return pos[:, :2].clone(), yaw.clone()

    def _laser_sensor(self):
        try:
            return self.unwrapped.scene[self.laser_obs_cfg.sensor_name]
        except Exception as exc:
            raise RuntimeError(
                f"Laser SAC requires MuJoCo RayCastSensor {self.laser_obs_cfg.sensor_name!r}. "
                "Use scripts/train_SAC_policies.py or add the sensor "
                "to env_cfg.scene.sensors before constructing the environment."
            ) from exc

    def laser_ranges(self) -> torch.Tensor:
        """Return this step's LiDAR ranges.

        Always reads from the cache, refreshed at most once per step_safe()
        call by _maybe_refresh_laser_cache() -- never samples the sensor
        directly, so every caller within one high-level step (pre-reset
        diagnostics, the terminal observation, the next-step observation)
        sees an identical, consistent reading, whether or not this step is
        one that actually re-samples the physical sensor.
        """
        self._ensure_laser_refresh_state()
        return self._cached_laser_ranges

    def _ensure_laser_refresh_state(self) -> None:
        """Lazily create/validate the LiDAR refresh-cache state.

        Can't run this once, eagerly, right after super().__init__(): the
        base class's __init__ already calls self.reset() internally (via
        _reset_selected -> _enforce_reset_feasibility), which may call
        laser_ranges() before this subclass's __init__ body ever resumes.
        Self-initializing on first use (which happens to be from inside
        that very first reset) sidesteps the ordering problem instead of
        fighting it. Safe to call repeatedly; a no-op once initialized.
        """
        if hasattr(self, "_cached_laser_ranges"):
            return
        laser_update_dt = float(self.laser_obs_cfg.update_dt)
        high_level_dt = float(self.navigation_cfg.high_level_dt)
        if laser_update_dt < high_level_dt - 1.0e-12:
            raise ValueError(
                "laser.update_dt must be >= navigation.high_level_dt "
                f"(cannot refresh the LiDAR faster than the policy is queried): "
                f"{laser_update_dt} < {high_level_dt}"
            )
        ratio = laser_update_dt / high_level_dt
        rounded_ratio = round(ratio)
        if abs(ratio - rounded_ratio) > 1.0e-9:
            raise ValueError(
                "laser.update_dt must be an integer multiple of "
                f"navigation.high_level_dt: {laser_update_dt} / {high_level_dt} "
                f"= {ratio}"
            )
        # Number of high-level steps between real LiDAR reads. 1 when
        # laser.update_dt == navigation.high_level_dt (the archived/default
        # 10 Hz-everywhere behavior); >1 decouples a slower sensor from a
        # faster decision loop (see _maybe_refresh_laser_cache).
        self._laser_refresh_every_n_steps = rounded_ratio
        self._laser_steps_since_refresh = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._cached_laser_ranges = torch.full(
            (self.num_envs, self.laser_obs_cfg.num_rays),
            float(self.laser_obs_cfg.max_range),
            device=self.device,
        )

    def _maybe_refresh_laser_cache(self) -> None:
        """Re-sample the LiDAR for envs whose refresh period has elapsed.

        Called once at the top of step_safe(), before the low-level
        substep loop. With laser.update_dt == navigation.high_level_dt
        (the default), every env is due every step, reproducing the
        archived always-fresh behavior exactly. With a slower update_dt,
        envs not yet due keep last step's cached reading, modeling a LiDAR
        that only physically refreshes at its own native rate while the
        policy is queried faster.
        """
        self._ensure_laser_refresh_state()
        due = self._laser_steps_since_refresh <= 0
        if due.any():
            fresh = self._sample_laser_raw()
            self._cached_laser_ranges = torch.where(due[:, None], fresh, self._cached_laser_ranges)
            self._laser_steps_since_refresh = torch.where(
                due,
                torch.full_like(self._laser_steps_since_refresh, self._laser_refresh_every_n_steps - 1),
                self._laser_steps_since_refresh,
            )
        self._laser_steps_since_refresh = torch.where(
            due, self._laser_steps_since_refresh, self._laser_steps_since_refresh - 1
        )

    def _force_laser_resample(self, env_ids: torch.Tensor) -> None:
        """Force an immediate real LiDAR read for just-(re)placed envs.

        Called after any direct root-state teleport (a normal episode reset
        or a collision-initialization rollback), once the physics/BVH state
        has already been re-sensed via sim.sense(). Without this, a
        just-reset env would keep serving its *previous* episode's cached
        scan (or the arbitrary max-range default at construction) until its
        regular refresh period happened to elapse, which is wrong regardless
        of laser.update_dt.
        """
        self._ensure_laser_refresh_state()
        if env_ids.numel() == 0:
            return
        fresh = self._sample_laser_raw()
        self._cached_laser_ranges[env_ids] = fresh[env_ids]
        self._laser_steps_since_refresh[env_ids] = self._laser_refresh_every_n_steps - 1

    def _sample_laser_raw(self) -> torch.Tensor:
        sensor = self._laser_sensor()
        raw = sensor.data.distances.detach()
        max_range = float(self.laser_obs_cfg.max_range)
        min_range = float(self.laser_obs_cfg.min_range)
        if max_range <= min_range:
            raise ValueError("Laser max_range must be greater than min_range")
        ranges = torch.where(
            torch.isfinite(raw) & (raw >= 0.0),
            raw,
            torch.full_like(raw, max_range),
        )
        ranges = torch.clamp(ranges, min=min_range, max=max_range)

        preset = self.laser_obs_cfg.lidar_noise_preset
        if preset not in LIDAR_NOISE_BOUNDS_M:
            raise ValueError(f"Unknown LiDAR noise preset {preset!r}")
        near_bound, far_bound = LIDAR_NOISE_BOUNDS_M[preset]

        # Disabled and clean modes avoid random sampling and return exact ranges.
        if not self.laser_obs_cfg.enable_lidar_noise or far_bound == 0.0:
            return ranges

        power = float(self.laser_obs_cfg.noise_distance_power)
        if power <= 0.0:
            raise ValueError("noise_distance_power must be positive")
        epsilon = 1.0e-8
        normalized_range = (
            (ranges - min_range) / max(max_range - min_range, epsilon)
        ).clamp(0.0, 1.0)
        noise_bound = near_bound + (far_bound - near_bound) * normalized_range.pow(power)

        # Noise samples are independent across rays and observation steps.
        noise = (2.0 * torch.rand_like(ranges) - 1.0) * noise_bound
        tolerance = float(self.laser_obs_cfg.no_return_tolerance_m)
        if tolerance < 0.0:
            raise ValueError("no_return_tolerance_m must be non-negative")
        valid_hit = ranges < (max_range - tolerance)
        perturbed = torch.clamp(ranges + noise, min=min_range, max=max_range)
        # Max-range rays mean "no return"; perturbing them would create false hits.
        ranges = torch.where(valid_hit, perturbed, ranges)

        return ranges

    def _compute_observation(self):
        pos, _, vel_b, ang_b, _, _, yaw = self._root_state()
        p = pos[:, :2]
        goal_b = world_to_body(self.goal_positions - p, yaw) / self.nav_cfg.normalize_positions_by
        velocity = torch.cat((vel_b[:, :2], ang_b[:, 2:3]), dim=-1)
        ranges = self.laser_ranges() / float(self.laser_obs_cfg.max_range)

        if not hasattr(self, "previous_nominal_command"):
            previous_nominal = torch.zeros(self.num_envs, 3, device=self.device)
            nominal_valid = torch.zeros(self.num_envs, 1, device=self.device)
        else:
            previous_nominal = self.previous_nominal_command
            nominal_valid = self.nominal_history_valid

        obs = torch.cat(
            (
                goal_b,
                velocity,
                self.previous_safe_command,
                ranges,
                previous_nominal,
                nominal_valid,
            ),
            dim=-1,
        )
        expected = 12 + self.laser_obs_cfg.num_rays
        if obs.shape[1] != expected:
            raise RuntimeError(f"Laser observation dimension {obs.shape[1]} != expected {expected}")
        return self._pack_obs(obs)

    def _reset_selected(self, env_ids: torch.Tensor) -> None:
        # Let the base reset finish first, then install the selected procedural
        # geometry.  This prevents later reset work from leaving sensor and
        # analytical geometry out of sync.
        super()._reset_selected(env_ids)
        if self._procedural_maps is not None:
            if not hasattr(self, "current_procedural_map"):
                self.current_procedural_map = torch.zeros(
                    self.num_envs, dtype=torch.long, device=self.device
                )
                self.current_procedural_scenario = torch.zeros(
                    self.num_envs, dtype=torch.long, device=self.device
                )
            self._activate_procedural_maps(env_ids)
        if self._procedural_maps is not None:
            self._apply_procedural_scenarios(env_ids)
        if self._fixed_scenario is not None:
            self._apply_fixed_scenario(env_ids)
        # The robot's reset pose is now committed (and, via
        # _apply_procedural_scenarios's sim.sense() call, the ray BVH refit
        # to match it). Force a real LiDAR read now, before
        # _enforce_reset_feasibility below reads laser_ranges() expecting
        # "the actual just-sensed LiDAR scan" (see _reset_feasibility_h_aug) --
        # otherwise it would see this env's stale pre-reset (or default)
        # cached scan instead.
        self._force_laser_resample(env_ids)
        if (
            self._fixed_scenario is None
            and self._forced_scenario_ids is None
            and self._procedural_maps is not None
            and self._reset_feasibility_filter is not None
        ):
            self._enforce_reset_feasibility(env_ids)
        if hasattr(self, "episode_return_buf"):
            self.episode_return_buf[env_ids] = 0.0
        if hasattr(self, "episode_laser_safety_violation_buf"):
            self.episode_laser_safety_violation_buf[env_ids] = False
        if hasattr(self, "previous_nominal_command"):
            self.previous_nominal_command[env_ids] = 0.0
            self.nominal_history_valid[env_ids] = 0.0
        if hasattr(self, "_last_collision_active"):
            self._last_collision_active[env_ids] = False

    def _apply_procedural_scenarios(self, env_ids: torch.Tensor) -> None:
        scenario_ids = self.current_procedural_scenario[env_ids]
        starts = self._scenario_starts[scenario_ids]
        goals = self._scenario_goals[scenario_ids]
        yaws = self._scenario_start_yaws[scenario_ids]
        self._write_root_poses(env_ids, starts, yaws)
        self.goal_positions[env_ids] = goals
        self.previous_safe_command[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.previous_goal_distance[env_ids] = torch.linalg.norm(goals - starts, dim=-1)

        # _write_root_poses() forwards MuJoCo after the manual state write.  A
        # fresh sense pass also refits the ray BVH after mocap poses and per-env
        # cylinder sizes changed, so the reset observation belongs to this map.
        sense = getattr(self.unwrapped.sim, "sense", None)
        if callable(sense):
            sense()

    def _reset_feasibility_h_aug(self, env_ids: torch.Tensor) -> torch.Tensor:
        """h_aug(z0) at the actual reset state (zero command/velocity) with
        the actual just-sensed LiDAR scan, for the given envs. Reuses the
        filter's own filter() so this is exactly what the theorem requires --
        not a reimplementation."""
        ranges = self.laser_ranges()[env_ids]
        angles = self._reset_feasibility_filter_angles(ranges)
        zero_command = torch.zeros(env_ids.numel(), 3, device=self.device, dtype=ranges.dtype)
        with torch.no_grad():
            result = self._reset_feasibility_filter.filter(ranges, angles, zero_command, zero_command)
        return result.h_lse

    def _reset_feasibility_filter_angles(self, ranges: torch.Tensor) -> torch.Tensor:
        return self.ray_angles.to(device=ranges.device, dtype=ranges.dtype)

    def _enforce_reset_feasibility(self, env_ids: torch.Tensor) -> None:
        """Reject/resample any candidate scenario with h_aug(z0) < 0 (Task 8).

        Does NOT modify h_aug, the terminal scales, or any theorem constant --
        only which scenario a given env slot ends up running. Exhausting the
        attempt budget raises rather than silently accepting an infeasible
        start, per spec.
        """
        remaining = env_ids
        for _ in range(self.reset_feasibility_max_attempts):
            if remaining.numel() == 0:
                return
            h_aug = self._reset_feasibility_h_aug(remaining)
            infeasible = h_aug < 0.0
            self.reset_feasibility_checked_count += remaining.numel()
            self.reset_feasibility_infeasible_count += int(infeasible.sum())
            if not bool(infeasible.any()):
                return
            remaining = remaining[infeasible]
            self._activate_procedural_maps(remaining)
            self._apply_procedural_scenarios(remaining)
            # Re-sample for real: this resample loop re-teleports `remaining`
            # to a new candidate scenario each iteration, and the next
            # _reset_feasibility_h_aug() call above must see that
            # candidate's own scan, not the previous (rejected) candidate's
            # stale cached one.
            self._force_laser_resample(remaining)

        h_aug = self._reset_feasibility_h_aug(remaining)
        still_infeasible = h_aug < 0.0
        if bool(still_infeasible.any()):
            bad_envs = remaining[still_infeasible].tolist()
            self.reset_feasibility_exhausted_count += len(bad_envs)
            raise RuntimeError(
                "Reset-feasibility guard exhausted "
                f"{self.reset_feasibility_max_attempts} resample attempts for "
                f"{len(bad_envs)} environment(s) with h_aug(z0) < 0 "
                f"(env_ids={bad_envs}). Refusing to silently accept an "
                "infeasible start -- investigate the map/scenario bank rather "
                "than raising the attempt budget."
            )

    def assert_procedural_geometry_synchronized(
        self, atol: float = 1.0e-4, *, check_selected_map: bool = True
    ) -> None:
        """Fail fast if reusable MuJoCo slots disagree with analytical tensors."""
        if self._procedural_maps is None:
            return
        if check_selected_map:
            selected = self.current_procedural_map
            expected_centers = self._map_centers[selected]
            expected_radii = self._map_radii[selected]
            expected_masks = self._map_masks[selected]
            if not torch.equal(self.obstacle_mask, expected_masks):
                bad_env = int(torch.nonzero((self.obstacle_mask != expected_masks).any(dim=1))[0])
                raise RuntimeError(
                    f"Procedural obstacle mask belongs to the wrong map in env {bad_env}: "
                    f"selected_map_index={int(selected[bad_env])}"
                )
            for name, actual, expected in (
                ("center", self.obstacle_centers, expected_centers),
                ("radius", self.obstacle_radii, expected_radii),
            ):
                if not torch.allclose(actual, expected, atol=atol, rtol=0.0):
                    per_env = (actual - expected).abs().reshape(self.num_envs, -1).amax(dim=1)
                    bad_env = int(per_env.argmax())
                    raise RuntimeError(
                        f"Procedural obstacle {name} tensor belongs to the wrong map in env "
                        f"{bad_env}: selected_map_index={int(selected[bad_env])}, "
                        f"max_error={float(per_env[bad_env]):.6g}"
                    )
        geom_xpos = self.unwrapped.sim.data.geom_xpos
        for slot, geom_id in enumerate(self._slot_geom_ids):
            active = self.obstacle_mask[:, slot]
            if not bool(active.any()):
                continue
            physical_xy = geom_xpos[active, geom_id, :2]
            analytical_xy = self.obstacle_centers[active, slot]
            if not torch.allclose(physical_xy, analytical_xy, atol=atol, rtol=0.0):
                error = torch.max(torch.abs(physical_xy - analytical_xy)).item()
                raise RuntimeError(
                    f"Procedural obstacle slot {slot} pose mismatch: max error={error:.6g} m"
                )
            physical_size = self.unwrapped.sim.model.geom_size[active, geom_id, 0]
            analytical_size = self.obstacle_radii[active, slot]
            if not torch.allclose(physical_size, analytical_size, atol=atol, rtol=0.0):
                error = torch.max(torch.abs(physical_size - analytical_size)).item()
                raise RuntimeError(
                    f"Procedural obstacle slot {slot} size mismatch: max error={error:.6g} m"
                )

    def set_fixed_scenario(self, scenario) -> None:
        self._fixed_scenario = scenario
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._reset_selected(env_ids)

    def set_fixed_scenarios(self, scenarios) -> None:
        """Assign one deterministic scenario to each vector environment."""
        scenarios = tuple(scenarios)
        if len(scenarios) != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} fixed scenarios, got {len(scenarios)}."
            )
        self._fixed_scenario = scenarios
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._reset_selected(env_ids)

    def clear_fixed_scenario(self) -> None:
        self._fixed_scenario = None

    def set_fixed_procedural_scenarios(self, scenario_ids: torch.Tensor) -> None:
        """Force every env to a specific procedural-bank scenario id (map +
        start/goal/yaw all come from that scenario, exactly reproducing the
        training map bank's own geometry) instead of curriculum-weighted
        random sampling. For deterministic (seed, case_id) evaluation on the
        TRAINING map bank -- distinct from set_fixed_scenarios, which only
        overrides start/goal/yaw within whatever map is already active and is
        for the hand-crafted structured eval maps (procedural_maps=None).
        """
        if self._procedural_maps is None:
            raise ValueError("set_fixed_procedural_scenarios requires a procedural_map_bank.")
        ids = torch.as_tensor(scenario_ids, dtype=torch.long, device=self.device)
        if ids.numel() != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} scenario ids, got {ids.numel()}.")
        self._forced_scenario_ids = ids
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._reset_selected(env_ids)

    def clear_fixed_procedural_scenarios(self) -> None:
        self._forced_scenario_ids = None

    def _apply_fixed_scenario(self, env_ids: torch.Tensor) -> None:
        fixed = self._fixed_scenario
        if fixed is None or env_ids.numel() == 0:
            return

        robot = self.unwrapped.scene["robot"]
        # Clone from the robot's DEFAULT (canonical standing) root state, not
        # its current live pose -- matching _write_root_poses's organic-reset
        # convention. root_link_pose_w reflects wherever the robot physically
        # is right now, which after a collision/fall is a broken pose (wrong
        # height, tipped over); only overwriting x/y/yaw on top of that
        # teleports the robot to the new start position while still in that
        # broken physical state, corrupting every subsequent episode in this
        # same env slot until a genuinely clean reset happens to occur.
        default = getattr(robot.data, "default_root_state", None)
        if default is None:
            raise AttributeError("robot.data.default_root_state is required for fixed-scenario resets.")
        pose = default[env_ids, :7].clone()
        velocity = torch.zeros(env_ids.numel(), 6, device=self.device, dtype=pose.dtype)
        if isinstance(fixed, (tuple, list)):
            selected = [fixed[int(env_id)] for env_id in env_ids.tolist()]
        else:
            selected = [fixed] * env_ids.numel()
        starts = pose.new_tensor([scenario.start_xy for scenario in selected])
        goals = pose.new_tensor([scenario.goal_xy for scenario in selected])
        yaws = pose.new_tensor([scenario.start_yaw for scenario in selected])
        pose[:, :2] = starts
        half_yaw = 0.5 * yaws
        pose[:, 3] = torch.cos(half_yaw)
        pose[:, 4] = 0.0
        pose[:, 5] = 0.0
        pose[:, 6] = torch.sin(half_yaw)
        robot.write_root_state_to_sim(torch.cat((pose, velocity), dim=-1), env_ids=env_ids)

        # Without this, the ray BVH stays fitted to the pre-teleport pose, so
        # the very first LiDAR read after a fixed-scenario reset (via
        # _force_laser_resample in _reset_selected) describes wherever the
        # robot USED to be, not its new teleported position -- matches
        # _apply_procedural_scenarios's own sense() call above.
        sense = getattr(self.unwrapped.sim, "sense", None)
        if callable(sense):
            sense()
        self._refresh_state_buffers()

        self.goal_positions[env_ids] = goals
        self.previous_safe_command[env_ids] = 0.0
        if hasattr(self, "previous_nominal_command"):
            self.previous_nominal_command[env_ids] = 0.0
            self.nominal_history_valid[env_ids] = 0.0
        self.previous_goal_distance[env_ids] = torch.linalg.norm(goals - pose[:, :2], dim=-1)
        self.episode_length_buf[env_ids] = 0
        if hasattr(self, "episode_return_buf"):
            self.episode_return_buf[env_ids] = 0.0

    def _record_collision_initialization_history(
        self, safe_command: torch.Tensor, nominal_command: torch.Tensor
    ) -> None:
        """Record the full simulated robot state once per high-level step.

        Also records the applied/nominal command that produced this state, so
        a later rollback can restore the command-dynamics augmented state
        consistently with the just-restored physical state, instead of
        zeroing it (see `_restore_collision_initialization`).
        """
        robot = self.unwrapped.scene["robot"]
        slot = self._ci_history_write
        self._ci_root_history[slot].copy_(torch.cat(
            (robot.data.root_link_pose_w, robot.data.root_link_vel_w), dim=-1
        ))
        self._ci_joint_pos_history[slot].copy_(robot.data.joint_pos)
        self._ci_joint_vel_history[slot].copy_(robot.data.joint_vel)
        self._ci_safe_command_history[slot].copy_(safe_command)
        self._ci_nominal_command_history[slot].copy_(nominal_command)
        self._ci_history_write = (slot + 1) % self._collision_init_history_steps

    def _update_seanav_curriculum(
        self, env_ids: torch.Tensor, terminal_distance: torch.Tensor
    ) -> None:
        """Apply SEA-Nav's outcome-adaptive level update to scenario deciles."""
        if self.curriculum_mode != "seanav_adaptive" or env_ids.numel() == 0:
            return
        levels = self.seanav_curriculum_levels[env_ids]
        goal_levels = self.seanav_goal_levels[env_ids]
        move_up = terminal_distance < 0.5
        move_down = terminal_distance > 2.0
        delta = move_up.long() - move_down.long()
        goal_levels = (goal_levels + delta).clamp(min=0.0, max=10.0)
        levels = levels + delta
        overflow = levels >= 10
        if bool(overflow.any()):
            levels[overflow] = torch.randint(0, 10, (int(overflow.sum()),), device=self.device)
        self.seanav_curriculum_levels[env_ids] = levels.clamp_min(0)
        self.seanav_goal_levels[env_ids] = goal_levels

    def _restore_collision_initialization(
        self, env_ids: torch.Tensor, undo_steps: torch.Tensor
    ) -> None:
        robot = self.unwrapped.scene["robot"]
        history_indices = (self._ci_history_write - 1 - undo_steps) % self._collision_init_history_steps
        root = self._ci_root_history[history_indices, env_ids]
        joint_pos = self._ci_joint_pos_history[history_indices, env_ids]
        joint_vel = self._ci_joint_vel_history[history_indices, env_ids]
        safe_command = self._ci_safe_command_history[history_indices, env_ids]
        nominal_command = self._ci_nominal_command_history[history_indices, env_ids]
        robot.write_root_state_to_sim(root, env_ids=env_ids)
        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._refresh_state_buffers()
        sense = getattr(self.unwrapped.sim, "sense", None)
        if callable(sense):
            sense()
        self._force_laser_resample(env_ids)

        restored_xy = root[:, :2]
        self.previous_goal_distance[env_ids] = torch.linalg.norm(
            self.goal_positions[env_ids] - restored_xy, dim=-1
        )
        # Restore the command-dynamics augmented state to what was actually
        # in effect at the rollback point, rather than zeroing it. The
        # robot's joint velocities just jumped back to a real mid-motion
        # state, so command-dynamics filters (cbf_cd_cf, bcbf_*), which treat
        # the applied command as literal augmented state, must not believe
        # "currently commanding zero" for the step immediately after rollback.
        self.previous_safe_command[env_ids] = safe_command
        self.previous_nominal_command[env_ids] = nominal_command
        self.nominal_history_valid[env_ids] = 1.0
        self.episode_length_buf[env_ids] = 0
        self.episode_return_buf[env_ids] = 0.0
        self.episode_laser_safety_violation_buf[env_ids] = False
        self._last_collision_active[env_ids] = False

    def _reset_completed_episodes(
        self,
        completed_ids: torch.Tensor,
        collision_outcome: torch.Tensor,
        terminal_distance: torch.Tensor,
    ) -> None:
        if completed_ids.numel() == 0:
            return
        ci = self.collision_initialization_cfg
        replay_mask = torch.zeros(completed_ids.numel(), dtype=torch.bool, device=self.device)
        undo_steps = torch.zeros(completed_ids.numel(), dtype=torch.long, device=self.device)
        if ci.enabled:
            candidates = collision_outcome[completed_ids]
            self.collision_initialization_attempts += int(candidates.sum())
            min_undo = max(1, int(round(ci.undo_time_min_s / self.navigation_cfg.high_level_dt)))
            max_undo = max(min_undo, int(round(ci.undo_time_max_s / self.navigation_cfg.high_level_dt)))
            undo_steps = torch.randint(min_undo, max_undo + 1, (completed_ids.numel(),), device=self.device)
            minimum_history = max(
                1, int(round(ci.minimum_history_s / self.navigation_cfg.high_level_dt))
            )
            available = self.episode_length_buf[completed_ids]
            undo_steps = torch.minimum(undo_steps, available)
            replay_mask = (
                candidates
                & (torch.rand(completed_ids.numel(), device=self.device) < ci.replay_probability)
                & (undo_steps > minimum_history)
            )

        replay_ids = completed_ids[replay_mask]
        normal_ids = completed_ids[~replay_mask]
        if replay_ids.numel():
            self._restore_collision_initialization(replay_ids, undo_steps[replay_mask])
            self.collision_initialization_applied += int(replay_ids.numel())
        if normal_ids.numel():
            normal_rows = torch.nonzero(~replay_mask, as_tuple=False).flatten()
            self._update_seanav_curriculum(normal_ids, terminal_distance[normal_rows])
            self._reset_selected(normal_ids)

    def step_safe(self, nominal_command: torch.Tensor, safe_command: torch.Tensor, safety):
        # Advance the LiDAR refresh cadence exactly once per high-level step,
        # before anything in this step reads laser_ranges(). With the
        # default laser.update_dt == navigation.high_level_dt this refreshes
        # every call (unchanged archived behavior); with a slower
        # update_dt it lets high_level_dt run faster than the sensor's own
        # native refresh rate without every intervening step re-sampling it.
        self._maybe_refresh_laser_cache()
        cfg = self.navigation_cfg
        scale = torch.tensor(
            [cfg.action.max_vx, cfg.action.max_vy, cfg.action.max_wz],
            device=self.device,
            dtype=safe_command.dtype,
        )
        safe = torch.clamp(safe_command, -scale, scale)
        nominal = torch.clamp(nominal_command, -scale, scale)
        previous_safe = self.previous_safe_command.clone()
        previous_distance = self.previous_goal_distance.clone()

        terminal_body_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        penalized_body_contact = torch.zeros_like(terminal_body_contact)
        hard_body_contact = torch.zeros_like(terminal_body_contact)
        maximum_body_contact_force = torch.zeros(
            self.num_envs, dtype=safe.dtype, device=self.device
        )
        # The contact sensors (and the seanav_contact termination path below)
        # are the only consumers of this per-substep force readout. Skipping
        # it in "circle" mode avoids querying sensors that were never added
        # to the scene, and avoids paying their contact-kernel cost on every
        # substep for a value that circle mode discards.
        use_contact_sensors = self.contact_collision_cfg.termination_mode == "seanav_contact"

        for _ in range(cfg.low_steps_per_high_step):
            if hasattr(self.command_interface, "write_batch"):
                self.command_interface.write_batch(safe)
            low_obs = self.low_env.get_observations()
            joint_action = self.low_policy(low_obs, safe)
            self.low_env.step(joint_action)
            if use_contact_sensors:
                contact_cfg = self.contact_collision_cfg
                terminal_sensor = self.unwrapped.scene[contact_cfg.terminal_sensor_name]
                penalized_sensor = self.unwrapped.scene[contact_cfg.penalized_sensor_name]
                # SEA-Nav thresholds the horizontal contact-force magnitude. The
                # sensors report global-frame forces, so x/y are directly usable.
                terminal_force = torch.linalg.vector_norm(
                    terminal_sensor.data.force[..., :2], dim=-1
                )
                penalized_force = torch.linalg.vector_norm(
                    penalized_sensor.data.force[..., :2], dim=-1
                )
                terminal_force = terminal_force.amax(dim=-1)
                penalized_force = penalized_force.amax(dim=-1)
                terminal_body_contact |= terminal_force > contact_cfg.terminal_force_threshold_n
                penalized_body_contact |= penalized_force > contact_cfg.penalized_force_threshold_n
                hard_body_contact |= penalized_force > contact_cfg.hard_force_threshold_n
                maximum_body_contact_force = torch.maximum(
                    maximum_body_contact_force, penalized_force
                )

        pos, _, vel_b, ang_b, roll, pitch, yaw = self._root_state()
        p = pos[:, :2]
        distance = torch.linalg.norm(self.goal_positions - p, dim=-1)
        self._record_collision_initialization_history(safe, nominal)
        progress = previous_distance - distance
        goal_candidate = distance <= cfg.goal_tolerance

        # Analytical overlap is retained strictly as an offline diagnostic.
        # Circular obstacles are Minkowski-expanded by the orientation-aware
        # physical Go2 ellipse plus a small configurable collision buffer.
        # The formal safety violation/reward below never uses these map tensors;
        # it is defined exclusively by the filter's local LiDAR LSE barrier.
        geometry = self.laser_safety_cfg
        half_length = 0.5 * float(geometry.robot_length_m)
        half_width = 0.5 * float(geometry.robot_width_m)
        rel_w = self.obstacle_centers - p[:, None, :]
        d_obs = torch.linalg.norm(rel_w, dim=-1)
        if self._any_square_slot and geometry.collision_geometry != "circle":
            # Box-vs-ellipse Minkowski overlap is not implemented; a square
            # slot's collision math below always assumes a circular robot
            # footprint (collision_circle_radius_m), same as legacy all-square
            # banks required.
            raise ValueError(
                "Any square obstacle slot currently requires --collision-geometry circle"
            )
        if geometry.collision_geometry == "circle":
            collision_radius = float(geometry.collision_circle_radius_m)
            if collision_radius <= 0.0:
                raise ValueError("collision_circle_radius_m must be positive")
            collision_distance = self.obstacle_radii + collision_radius
            collision_h_obs = d_obs - collision_distance
            physical_extent_x = torch.full_like(yaw, collision_radius)
            physical_extent_y = torch.full_like(yaw, collision_radius)
            if self._any_square_slot:
                delta = torch.clamp(torch.abs(rel_w) - self.obstacle_radii[..., None], min=0.0)
                square_h_obs = torch.linalg.norm(delta, dim=-1) - collision_radius
                collision_h_obs = torch.where(
                    self._slot_is_square[None, :], square_h_obs, collision_h_obs
                )
        elif geometry.collision_geometry == "ellipse":
            collision_margin = float(geometry.collision_margin_m)
            if collision_margin < 0.0:
                raise ValueError("collision_margin_m must be non-negative")
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
            # Ellipse support distances against the axis-aligned arena walls.
            physical_extent_x = torch.sqrt(
                (collision_half_length * c[:, 0]).square()
                + (collision_half_width * s[:, 0]).square()
            )
            physical_extent_y = torch.sqrt(
                (collision_half_length * s[:, 0]).square()
                + (collision_half_width * c[:, 0]).square()
            )
        else:
            raise ValueError(f"Unknown collision_geometry={geometry.collision_geometry!r}")
        collision_h_obs = torch.where(
            self.obstacle_mask, collision_h_obs, torch.full_like(collision_h_obs, torch.inf)
        )
        obstacle_collision = torch.any(collision_h_obs <= 0.0, dim=-1)
        # Physical wall contact is detected by the contact sensors. This exact
        # center-bound test is only a numerical escape guard, without the old
        # arbitrary circle/ellipse inset.
        outside = (
            (p[:, 0] <= cfg.arena.x_min)
            | (p[:, 0] >= cfg.arena.x_max)
            | (p[:, 1] <= cfg.arena.y_min)
            | (p[:, 1] >= cfg.arena.y_max)
        )
        termination_mode = self.contact_collision_cfg.termination_mode
        if termination_mode == "circle":
            # Archived SAC behavior: the configured circular robot footprint
            # terminates when it overlaps an obstacle proxy. Physical contacts
            # remain available in diagnostics but do not define reward or done.
            collision = obstacle_collision | outside
            collision_reward_event = collision
        elif termination_mode == "seanav_contact":
            # Simulator-native SEA-Nav-inspired behavior. This is intentionally
            # named "inspired": MuJoCo and Isaac Gym expose different contact
            # models and the high-level policy only resets at 10 Hz.
            collision = terminal_body_contact | hard_body_contact | outside
            collision_reward_event = penalized_body_contact | collision
        else:
            raise ValueError(f"Unknown collision termination mode {termination_mode!r}")
        # SEA-Nav's collision-state initialization also ends some episodes on
        # the onset of a non-terminal body contact. Its probability increases
        # from 0.1 at curriculum level 0 to 0.5 at level 1.5. Keeping this
        # behind the same switch makes the on/off ablation differ only by the
        # complete collision-initialization mechanism.
        ci = self.collision_initialization_cfg
        if ci.enabled:
            new_contact = (
                collision_reward_event
                & ~self._last_collision_active
                & (self.episode_length_buf > 0)
            )
            level_scale = (self.seanav_goal_levels / 1.5).clamp(max=1.0)
            early_probability = ci.early_reset_probability_min + (
                ci.early_reset_probability_max - ci.early_reset_probability_min
            ) * level_scale
            early_collision_reset = new_contact & (
                torch.rand(self.num_envs, device=self.device) < early_probability
            )
            collision |= early_collision_reset
            if bool(early_collision_reset.any()):
                self.collision_initialization_early_resets += int(early_collision_reset.sum())
        self._last_collision_active.copy_(collision_reward_event)
        # For lifted CBF/bCBF variants, h_lse additionally composes command-
        # input barriers.  The safety reward/diagnostic requested here must be
        # LiDAR-derived only, so use the ray/backup aggregate exposed as
        # h_backup_lse. Direct CBF already exposes its ray aggregate as h_lse.
        laser_h_lse = getattr(safety, "h_backup_lse", None)
        if laser_h_lse is None:
            laser_h_lse = safety.h_lse
        laser_safety_violation = laser_h_lse < 0.0
        self.episode_laser_safety_violation_buf |= laser_safety_violation

        # Legacy center-distance clearance is retained only as a diagnostic.
        diagnostic_radius = (
            float(geometry.collision_circle_radius_m)
            if geometry.collision_geometry == "circle" else half_length
        )
        diagnostic = d_obs - (self.obstacle_radii + diagnostic_radius)
        if self._any_square_slot:
            delta = torch.clamp(torch.abs(rel_w) - self.obstacle_radii[..., None], min=0.0)
            square_diagnostic = torch.linalg.norm(delta, dim=-1) - diagnostic_radius
            diagnostic = torch.where(self._slot_is_square[None, :], square_diagnostic, diagnostic)
        obstacle_surface_clearance = torch.where(
            self.obstacle_mask, diagnostic, torch.full_like(diagnostic, torch.inf)
        ).min(dim=-1).values
        # Collision has priority over goal completion when both are detected
        # after the same sampled high-level control interval. A colliding
        # transition must not receive a goal bonus or count as a success.
        goal = goal_candidate & ~collision
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
        forward_alignment = (
            torch.relu(vx_measured) / max(cfg.action.max_vx, 1.0e-6)
        ) * torch.cos(beta)
        reverse_cost = (torch.relu(-vx_measured) / max(cfg.action.max_vx, 1.0e-6)).pow(2)
        omega_ref = torch.clamp(self.sac_reward_cfg.yaw_gain * beta, -cfg.action.max_wz, cfg.action.max_wz)
        yaw_tracking_cost = ((wz_measured - omega_ref) / max(cfg.action.max_wz, 1.0e-6)).pow(2)
        smoothness = torch.sum((safe - previous_safe).pow(2), dim=-1)

        desired = getattr(safety, "desired_surrogate_command", None)
        surrogate = getattr(safety, "surrogate_command", None)
        if desired is not None and surrogate is not None:
            normalized_correction = torch.sum(((surrogate - desired) / scale).pow(2), dim=-1)
        else:
            normalized_correction = torch.sum(((safe - nominal) / scale).pow(2), dim=-1)
        applied_correction_norm = torch.linalg.norm(safe - nominal, dim=-1)
        applied_filter_active = applied_correction_norm > 1.0e-8
        # Capture pre-reset failure diagnostics. Calling the sensor after
        # _reset_selected() would incorrectly describe the next episode.
        pre_reset_laser_ranges = self.laser_ranges()
        pre_reset_laser_min_range, pre_reset_nearest_ray_index = pre_reset_laser_ranges.min(dim=-1)
        pre_reset_nearest_ray_angle = self.ray_angles[pre_reset_nearest_ray_index]
        projection_feasible = getattr(safety, "projection_feasible", None)
        if projection_feasible is None:
            projection_feasible = torch.ones_like(obstacle_collision)
        projected_residual = getattr(safety, "projected_cbf_residual", None)
        if projected_residual is None:
            projected_residual = torch.zeros_like(distance)
        input_bound_violation = getattr(safety, "input_bound_violation", None)
        if input_bound_violation is None:
            input_bound_violation = torch.zeros_like(distance)

        terminated = goal | collision | fallen
        truncated = timeout & ~terminated
        done = terminated | truncated

        r = self.sac_reward_cfg
        # See LaserRewardConfig.reference_dt: keeps each per-step shaping
        # term's *per-second* contribution approximately constant across
        # different high_level_dt settings. 1.0 at the archived default
        # (high_level_dt == reference_dt), so this is a no-op there.
        dt_scale = float(cfg.high_level_dt) / float(r.reference_dt)
        components = {
            "progress": r.progress_weight * progress,
            "time": torch.full_like(progress, -r.time_penalty * dt_scale),
            "forward_alignment": (dt_scale * r.forward_alignment_weight) * forward_alignment,
            "reverse_motion": -(dt_scale * r.reverse_motion_weight) * reverse_cost,
            "yaw_tracking": -(dt_scale * r.yaw_tracking_weight) * yaw_tracking_cost,
            "smoothness": -(dt_scale * r.smoothness_weight) * smoothness,
            "cbf_correction": -(dt_scale * r.cbf_correction_weight) * normalized_correction,
        }
        if r.profile == "original":
            # Archived v5 reward path. Keep it byte-for-byte equivalent in
            # meaning for all existing launch commands and checkpoints.
            components.update({
                "goal": r.goal_bonus * goal.float(),
                "timeout": -r.timeout_penalty * truncated.float(),
                "collision": -r.collision_penalty * collision_reward_event.float(),
            })
        elif r.profile == "seanav_comparable":
            # SEA-Nav multiplies configured scales by its policy dt before
            # summation. Port only terms with a direct counterpart in this
            # hierarchical environment: tight-goal reward, collision/failure,
            # unsafe termination, and the zero-valued timeout term.
            components.update(seanav_comparable_terminal_terms(
                distance=distance,
                planar_velocity=vel_b[:, :2],
                yaw_rate=wz_measured,
                collision=collision_reward_event,
                fallen=fallen,
                truncated=truncated,
                reward_cfg=r,
                policy_dt=cfg.high_level_dt,
            ))
        else:
            raise ValueError(f"Unknown laser reward profile {r.profile!r}")
        reward = torch.stack(tuple(components.values()), dim=0).sum(dim=0)

        self.previous_safe_command = safe.detach().clone()
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
        map_ids = getattr(
            self, "current_procedural_map",
            torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device),
        )
        scenario_ids = getattr(
            self, "current_procedural_scenario",
            torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device),
        )
        completed_map_ids = map_ids[completed_ids].detach().clone()
        completed_scenario_ids = scenario_ids[completed_ids].detach().clone()
        # Assign one outcome per completed episode, in priority order.
        outcome_success = goal
        # Public "collision" is the single unsafe-failure outcome used by the
        # trainer and evaluation plots.  It intentionally combines obstacle or
        # wall contact, a fall, and leaving the allowable workspace so readers
        # do not have to reconcile several overlapping failure series.
        outcome_collision = (collision | fallen) & ~outcome_success
        # This diagnostic can overlap any terminal outcome: it means h_lse was
        # negative at least once during the episode, not that it terminated it.
        outcome_safety_violation = self.episode_laser_safety_violation_buf.clone()
        # Wall contact is a collision outcome; keep this legacy field false so
        # pooled outcome rates remain mutually exclusive.
        outcome_outside = torch.zeros_like(outcome_collision)
        # Retain legacy detail fields in extras for old analysis/checkpoint
        # consumers, but do not expose them as separate training metrics.
        outcome_fallen = fallen & ~outcome_success
        # A time-limit flag can coincide with another terminal event on the
        # final control step.  Preserve the documented priority ordering so
        # success, collision, and timeout remain mutually exclusive.
        outcome_timeout = truncated & ~outcome_success & ~outcome_collision
        if completed_ids.numel() > 0:
            self.episode_count += int(completed_ids.numel())
            self.episode_success_count += int(goal[completed_ids].sum())
            self.episode_collision_count += int(outcome_collision[completed_ids].sum())
            self._reset_completed_episodes(
                completed_ids, outcome_collision, distance[completed_ids]
            )

        log_values = {
            **{f"reward_{name}": value.mean().detach() for name, value in components.items()},
            "distance_to_goal": distance.mean().detach(),
            "success_transition_rate": goal.float().mean().detach(),
            "collision_transition_rate": (collision | fallen).float().mean().detach(),
            "physical_contact_rate": penalized_body_contact.float().mean().detach(),
            "terminal_contact_rate": terminal_body_contact.float().mean().detach(),
            "maximum_body_contact_force_n": maximum_body_contact_force.mean().detach(),
            "reverse_speed": torch.relu(-vx_measured).mean().detach(),
            "forward_speed": torch.relu(vx_measured).mean().detach(),
            "heading_error_abs": beta.abs().mean().detach(),
            "cbf_active_rate": applied_filter_active.float().mean().detach(),
            "cbf_correction_norm": applied_correction_norm.mean().detach(),
            "normalized_cbf_correction": normalized_correction.mean().detach(),
            "minimum_safety_term": safety.individual_h.min(dim=-1).values.mean().detach(),
            "h_lse": safety.h_lse.mean().detach(),
            "laser_h_lse": laser_h_lse.mean().detach(),
            "nominal_command_norm": torch.linalg.norm(nominal, dim=-1).mean().detach(),
            "safe_command_norm": torch.linalg.norm(safe, dim=-1).mean().detach(),
            "laser_min_range": (self.laser_ranges().min(dim=-1).values).mean().detach(),
        }
        h_backup = getattr(safety, "h_backup_lse", None)
        h_input = getattr(safety, "h_input_lse", None)
        lambda_value = getattr(safety, "lambda_value", None)
        # Distinct name from the `input_bound_violation` computed earlier for
        # `extras["completed_input_bound_violation"]` below -- reusing that
        # name here previously shadowed the already-None-defaulted value with
        # a fresh, unguarded read, breaking any method (e.g. SEA-Nav) whose
        # safety result leaves this field unset.
        log_input_bound_violation = getattr(safety, "input_bound_violation", None)
        if h_backup is not None:
            log_values["h_backup_lse"] = h_backup.mean().detach()
        if h_input is not None:
            log_values["h_input_lse"] = h_input.mean().detach()
        if lambda_value is not None:
            log_values["lambda_value"] = lambda_value.mean().detach()
        if log_input_bound_violation is not None:
            log_values["input_bound_violation"] = log_input_bound_violation.mean().detach()

        obs = self._compute_observation()
        extras = {
            "time_outs": truncated,
            "terminal_observation": terminal_observation,
            "completed_returns": completed_returns,
            "completed_lengths": completed_lengths,
            "completed_terminal_positions": completed_terminal_positions,
            "completed_env_ids": completed_ids.detach().clone(),
            "completed_map_ids": completed_map_ids,
            "completed_scenario_ids": completed_scenario_ids,
            "completed_success": outcome_success[completed_ids].detach().clone(),
            "completed_collision": outcome_collision[completed_ids].detach().clone(),
            "completed_obstacle_collision": obstacle_collision[completed_ids].detach().clone(),
            "completed_physical_contact": penalized_body_contact[completed_ids].detach().clone(),
            "completed_terminal_body_contact": terminal_body_contact[completed_ids].detach().clone(),
            "completed_hard_body_contact": hard_body_contact[completed_ids].detach().clone(),
            "completed_maximum_body_contact_force_n": maximum_body_contact_force[completed_ids].detach().clone(),
            "completed_laser_safety_violation": outcome_safety_violation[completed_ids].detach().clone(),
            # Compatibility aliases for existing analysis scripts/checkpoints.
            "completed_safety_violation": outcome_safety_violation[completed_ids].detach().clone(),
            "completed_raw_safety_violation": outcome_safety_violation[completed_ids].detach().clone(),
            "completed_timeouts": outcome_timeout[completed_ids].detach().clone(),
            "completed_fallen": outcome_fallen[completed_ids].detach().clone(),
            "completed_outside": outcome_outside[completed_ids].detach().clone(),
            "completed_wall_collision": outside[completed_ids].detach().clone(),
            "completed_obstacle_surface_clearance": obstacle_surface_clearance[completed_ids].detach().clone(),
            "completed_laser_min_range": pre_reset_laser_min_range[completed_ids].detach().clone(),
            "completed_filter_active": applied_filter_active[completed_ids].detach().clone(),
            "completed_filter_correction_norm": applied_correction_norm[completed_ids].detach().clone(),
            "completed_nominal_command": nominal[completed_ids].detach().clone(),
            "completed_safe_command": safe[completed_ids].detach().clone(),
            "completed_h_lse": safety.h_lse[completed_ids].detach().clone(),
            "completed_projection_feasible": projection_feasible[completed_ids].detach().clone(),
            "completed_projected_cbf_residual": projected_residual[completed_ids].detach().clone(),
            "completed_input_bound_violation": input_bound_violation[completed_ids].detach().clone(),
            "completed_curriculum_truncations": torch.zeros(
                completed_ids.numel(), dtype=torch.bool, device=self.device
            ),
            # Full-batch, pre-reset samples feed the trainer's bounded ring
            # buffer. They are not replay observations and do not affect SAC.
            "collision_trajectory_step": {
                "position_xy": p.detach(),
                "yaw": yaw.detach(),
                "goal_xy": self.goal_positions.detach(),
                "goal_relative_body_xy": goal_b.detach(),
                "distance_to_goal": distance.detach(),
                "measured_command": torch.stack((vel_b[:, 0], vel_b[:, 1], ang_b[:, 2]), dim=-1).detach(),
                "nominal_command": nominal.detach(),
                "safe_command": safe.detach(),
                "laser_min_range": pre_reset_laser_min_range.detach(),
                "nearest_ray_index": pre_reset_nearest_ray_index.detach(),
                "nearest_ray_angle": pre_reset_nearest_ray_angle.detach(),
                "obstacle_surface_clearance": obstacle_surface_clearance.detach(),
                "laser_safety_violation": laser_safety_violation.detach(),
                "safety_violation": laser_safety_violation.detach(),
                "raw_safety_violation": laser_safety_violation.detach(),
                "minimum_collision_proxy_barrier": collision_h_obs.min(dim=-1).values.detach(),
                "minimum_geometric_safety_barrier": collision_h_obs.min(dim=-1).values.detach(),
                "filter_active": applied_filter_active.detach(),
                "filter_correction_norm": applied_correction_norm.detach(),
                "h_lse": safety.h_lse.detach(),
                "laser_h_lse": laser_h_lse.detach(),
                "minimum_safety_term": safety.individual_h.min(dim=-1).values.detach(),
                "projection_feasible": projection_feasible.detach(),
                "projected_cbf_residual": projected_residual.detach(),
                "input_bound_violation": input_bound_violation.detach(),
                "map_id": map_ids.detach(),
                "scenario_id": scenario_ids.detach(),
                # Per-environment terms used by offline evaluation diagnostics.
                # These are diagnostic copies only and do not change reward.
                "reward_components": {
                    name: value.detach() for name, value in components.items()
                },
            },
            "log": log_values,
        }
        return obs, reward, done, extras
