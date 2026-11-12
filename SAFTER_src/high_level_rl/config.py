# NOTE: this is a DUPLICATE of src/high_level_rl/config.py, kept in sync manually.
# It exists here because src/high_level_rl/config.py is also load-bearing for
# map-based code (high_level_sac/high_level_bcbf_sac/high_level_cbf_rl) that
# intentionally stays outside SAFTER_src. See SAFTER_src/README.md.

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CircularObstacle:
    x: float
    y: float
    radius: float
    height: float = 1.0
    # Per-obstacle shape, independent of ArenaConfig.obstacle_shape (which
    # remains only a legacy/default metadata field). Defaults to "circle" so
    # every existing caller that never sets this is unaffected.
    shape: str = "circle"


@dataclass(frozen=True)
class ArenaConfig:
    width: float = 15.0
    length: float = 15.0
    wall_thickness: float = 0.12
    wall_height: float = 0.8

    # Original CBF safe-set clearances.  Backup CBF adds backup_margin on top,
    # so the default backup clearance is 0.15 + 0.30 = 0.45 m.
    robot_radius: float = 0.30
    wall_clearance: float = 0.15
    obstacle_clearance: float = 0.15

    max_obstacles: int = 9
    # For squares, CircularObstacle.radius is interpreted as the half-side.
    obstacle_shape: str = "circle"
    obstacles: tuple[CircularObstacle, ...] = field(
        default_factory=lambda: (
            CircularObstacle(-3.0, -2.0, 0.55),
            CircularObstacle(-1.5,  2.2, 0.45),
            CircularObstacle( 1.2, -1.0, 0.60),
            CircularObstacle( 3.0,  2.3, 0.50),
            CircularObstacle( 0.3,  3.8, 0.40),
            CircularObstacle(-5.0,  4.5, 0.50),
            CircularObstacle(-5.0, -4.5, 0.45),
            CircularObstacle( 5.0,  4.5, 0.55),
            CircularObstacle( 5.0, -4.5, 0.50),
        )
    )

    @property
    def x_min(self) -> float:
        return -0.5 * self.width

    @property
    def x_max(self) -> float:
        return 0.5 * self.width

    @property
    def y_min(self) -> float:
        return -0.5 * self.length

    @property
    def y_max(self) -> float:
        return 0.5 * self.length

    def validate(self) -> None:
        if self.width <= 0 or self.length <= 0:
            raise ValueError("Arena dimensions must be positive.")
        if self.robot_radius <= 0:
            raise ValueError("robot_radius must be positive.")
        if self.obstacle_shape not in ("circle", "square"):
            raise ValueError(f"Unsupported obstacle_shape={self.obstacle_shape!r}.")
        if len(self.obstacles) > self.max_obstacles:
            raise ValueError("Number of obstacles exceeds max_obstacles.")
        for i, obs in enumerate(self.obstacles):
            if obs.radius <= 0 or obs.height <= 0:
                raise ValueError(f"Obstacle {i} must have positive radius and height.")
            if not (self.x_min < obs.x < self.x_max and self.y_min < obs.y < self.y_max):
                raise ValueError(f"Obstacle {i} lies outside the arena.")
            if obs.shape not in ("circle", "square"):
                raise ValueError(f"Obstacle {i} has unsupported shape={obs.shape!r}.")


@dataclass(frozen=True)
class HighLevelActionConfig:
    # Matches the frozen low-level velocity-tracking policy's own trained
    # command range (see logs/rsl_rl/go2_velocity/.../params/env.yaml's
    # lin_vel_x curriculum, final stage (-1.0, 2.0); lin_vel_y/ang_vel_z
    # final stage (-1.0, 1.0)): commanding a velocity outside that range
    # asks the locomotion policy to track something it never learned to.
    max_vx: float = 2.0
    # Lower bound for vx, independent of max_vx (asymmetric forward/reverse
    # envelope). vy/wz remain symmetric (+-max_vy/+-max_wz); only vx has
    # ever needed an asymmetric bound so only vx gets one here.
    min_vx: float = -1.0
    max_vy: float = 1.0
    max_wz: float = 1.0


@dataclass(frozen=True)
class SafetyConfig:
    enabled: bool = True
    alpha: float = 1.5
    kappa: float = 15.0
    denominator_epsilon: float = 1.0e-8


@dataclass(frozen=True)
class BackupSafetyConfig:
    """Differentiable zero-backup bCBF-LSE closed-form configuration."""

    enabled: bool = True

    # Backup-set shrinkage relative to the original safe set.
    # Obstacles: r_backup = r_physical + obstacle_clearance + backup_margin.
    # Walls:     d_backup = wall_clearance + backup_margin.
    backup_margin: float = 0.3

    backup_horizon_s: float = 4.0
    backup_samples: int = 21

    # Hierarchical soft-min gains.
    kappa_backup: float = 50.0
    kappa_input: float = 20.0
    kappa_outer: float = 20.0
    alpha_outer: float = 0.8

    # Relaxed one-constraint closed-form KKT term.
    gamma_mu: float = 1.0e4

    # u_dot = -A u + A uhat.  sigma=A makes the zero-order desired surrogate
    # input equal to the nominal command when nominal command derivative is not
    # included in replay-buffer training.
    control_dynamics_a: tuple[float, float, float] = (8.0, 8.0, 8.0)
    tracking_sigma: tuple[float, float, float] = (8.0, 8.0, 8.0)

    # One high-level command/filter update is held for 0.10 s (10 Hz).
    integration_dt: float = 0.10
    denominator_epsilon: float = 1.0e-8

    # Keep the augmented terminal set as clearance-only, matching the uploaded
    # BCBF_USE_AUGMENTED_TERMINAL_SET=0 variant.
    terminal_near_stop_enabled: bool = False
    terminal_u_eps: tuple[float, float, float] = (0.08, 0.04, 0.08)


@dataclass(frozen=True)
class RewardConfig:
    progress_weight: float = 2.0
    goal_bonus: float = 20.0
    time_penalty: float = 0.01
    timeout_penalty: float = 5.0
    smoothness_weight: float = 0.02
    collision_penalty: float = 20.0
    boundary_penalty: float = 20.0


@dataclass(frozen=True)
class HighLevelNavigationConfig:
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    action: HighLevelActionConfig = field(default_factory=HighLevelActionConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    backup_safety: BackupSafetyConfig = field(default_factory=BackupSafetyConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    low_level_dt: float = 0.02
    # The high-level policy, LiDAR observation, and safety filter run at 10 Hz.
    # The low-level locomotion controller remains at 50 Hz, so each filtered
    # command is held for five low-level simulation/control steps.
    high_level_dt: float = 0.10
    episode_duration_s: float = 90.0
    goal_tolerance: float = 0.35
    minimum_start_goal_distance: float = 2.0
    spawn_extra_clearance: float = 0.30
    sample_margin_from_walls: float = 0.80
    normalize_positions_by: float = 7.5
    max_sampling_attempts: int = 200

    minimum_base_height: float = 0.18
    maximum_roll_pitch: float = 0.90

    @property
    def low_steps_per_high_step(self) -> int:
        ratio = self.high_level_dt / self.low_level_dt
        rounded = round(ratio)
        if abs(ratio - rounded) > 1.0e-9:
            raise ValueError("high_level_dt must be an integer multiple of low_level_dt.")
        return rounded

    @property
    def max_episode_steps(self) -> int:
        return round(self.episode_duration_s / self.high_level_dt)

    @property
    def observation_dim(self) -> int:
        return 12 + 4 * self.arena.max_obstacles
