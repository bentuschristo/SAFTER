from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from SAFTER_src.high_level_rl.config import HighLevelNavigationConfig


@dataclass(frozen=True)
class SacHyperparameters:
    """Soft Actor-Critic hyperparameters for the laser-based navigation policy."""

    hidden_dims: tuple[int, ...] = (512, 256, 128)
    # gamma and tau are per-step factors calibrated at reference_dt. The trainer
    # rescales them as gamma**(dt/reference_dt) and 1-(1-tau)**(dt/reference_dt)
    # so the per-second discount and target-tracking rate are independent of
    # navigation.high_level_dt.
    reference_dt: float = 0.10
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    alpha_lr: float = 3.0e-4
    batch_size: int = 1024
    replay_size: int = 2_000_000
    learning_starts: int = 20_000
    updates_per_step: int = 1
    # Perform optimizer updates only on vector steps divisible by this value.
    update_every_steps: int = 1
    target_entropy: float = -3.0
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    gradient_clip_norm: float = 10.0
    # Ablation: the critic also conditions on the current nominal (pre-filter)
    # action in addition to (obs, safe action). False keeps the standard critic.
    critic_sees_nominal_action: bool = False


LaserSacMethod = Literal[
    "vanilla", "cbf_cf_clip", "cbf_qp", "cbf_lse_qp",
    "cbf_cd_cf", "cbf_lse_cd_qp", "bcbf_cf", "bcbf_qp", "bcbf_lse_cd_qp",
    "bcbf_nonlse_cd_qp", "bcbf_lse_cd_qp_cfmatched", "cbf_lse_cd_qp_cfmatched",
    # Adapted control-dynamics bCBF-QP (LaserZeroBackupCdLseAdapted).
    "bcbf_cd_lse_adapted",
    # Zero-backup LSE QP without control dynamics, with hard box input
    # constraints (LaserZeroBackupLseQp).
    "bcbf_lse_qp",
]
CollisionGeometry = Literal["circle", "ellipse"]
CollisionTerminationMode = Literal["circle", "direct_contact"]
LidarNoisePreset = Literal["clean", "nominal", "moderate", "severe"]
# "legacy_inflated_ellipse": velocity-inflated terminal ellipse; not generally
#   forward-invariant under the zero backup (see terminal_h_and_grad_future).
# "invariant_ellipse": fixed-geometry terminal ellipse (axes + backup_margin)
#   intersected with a residual-motion recoverability condition (see
#   terminal_invariant_h_and_grad_future).
TerminalBackupMode = Literal["legacy_inflated_ellipse", "invariant_ellipse"]
QpBackend = Literal["active_set", "qpth", "osqp"]
RewardProfile = Literal["original", "seanav_comparable"]

LIDAR_NOISE_BOUNDS_M: dict[str, tuple[float, float]] = {
    "clean": (0.0, 0.0),
    "nominal": (0.005, 0.030),
    "moderate": (0.005, 0.050),
    # 0.15 m far-range bound, for severe robustness tests.
    "severe": (0.005, 0.150),
}


@dataclass(frozen=True)
class LaserObservationConfig:
    # Nominal policy interface: 61 rays including both edges of a 180-degree
    # forward field of view (3-degree spacing).
    num_rays: int = 61
    fov_deg: float = 180.0
    max_range: float = 5.0
    min_range: float = 0.05
    enable_lidar_noise: bool = False
    lidar_noise_preset: LidarNoisePreset = "nominal"
    noise_distance_power: float = 2.0
    no_return_tolerance_m: float = 1.0e-4
    yaw_offset_deg: float = 0.0
    left_to_right: bool = True
    sensor_name: str = "navigation_laser"
    lidar_site_name: str = "lidar_site"
    debug_vis: bool = False
    # LiDAR refresh period [s], independent of navigation.high_level_dt. With a
    # faster policy rate, the cached scan is reused until this period elapses.
    update_dt: float = 0.10


@dataclass(frozen=True)
class LaserRewardConfig:
    # ``original`` is the 10 Hz high-level reward.  ``seanav_comparable``
    # changes only the analogous goal, collision/unsafe-termination, and
    # timeout terms; all other shaping terms below are shared by both profiles.
    profile: RewardProfile = "original"
    # Per-step shaping weights (time, forward_alignment, reverse_motion,
    # yaw_tracking, smoothness, cbf_correction) are scaled by
    # high_level_dt / reference_dt so their per-second effect is rate
    # independent. Progress (already a per-step distance) and one-time outcome
    # terms are not scaled.
    reference_dt: float = 0.10
    progress_weight: float = 4.0
    goal_bonus: float = 30.0
    time_penalty: float = 0.01
    timeout_penalty: float = 5.0
    forward_alignment_weight: float = 0.20
    reverse_motion_weight: float = 0.50
    yaw_tracking_weight: float = 0.05
    smoothness_weight: float = 0.01
    cbf_correction_weight: float = 0.05
    collision_penalty: float = 20.0
    yaw_gain: float = 1.5

    # Nominal SEA-Nav scales.  Its LeggedRobot implementation multiplies every
    # non-zero scale (including termination) by the policy dt before summing.
    seanav_goal_scale: float = 10.0
    seanav_goal_distance_threshold_m: float = 0.50
    seanav_collision_scale: float = -4.0
    seanav_collision_velocity_scale: float = 4.0
    seanav_termination_scale: float = -100.0
    # SEA-Nav has no timeout term; its no-progress ("stuck") scale is used as
    # the closest analog.
    seanav_timeout_scale: float = -5.0


@dataclass(frozen=True)
class ContactCollisionConfig:
    """Physical collision semantics derived from MuJoCo contact forces.

    The defaults reproduce SEA-Nav's termination thresholds: terminal body
    contacts above 1 N, or any non-foot body contact above 50 N.  The lower
    penalized-contact threshold is used by the reward only.
    """

    # ``circle`` is the SAC footprint-overlap termination.
    # ``direct_contact`` uses MuJoCo body-contact forces as a simulator-native
    # approximation of SEA-Nav's Isaac Gym contact termination.
    termination_mode: CollisionTerminationMode = "direct_contact"
    terminal_sensor_name: str = "navigation_terminal_body_contact"
    penalized_sensor_name: str = "navigation_nonfoot_body_contact"
    terminal_force_threshold_n: float = 1.0
    hard_force_threshold_n: float = 50.0
    penalized_force_threshold_n: float = 1.0


@dataclass(frozen=True)
class CollisionInitializationConfig:
    """SEA-Nav-style reset from a state preceding an unsafe collision."""

    enabled: bool = False
    replay_probability: float = 0.80
    undo_time_min_s: float = 2.0
    undo_time_max_s: float = 3.0
    minimum_history_s: float = 0.40
    early_reset_probability_min: float = 0.10
    early_reset_probability_max: float = 0.50


@dataclass(frozen=True)
class LaserSafetyConfig:
    # Versioned model identifier saved in every checkpoint.  v2 converts each
    # current LiDAR return to a point in a temporary scan-fixed local frame and
    # propagates the robot pose relative to those points; no global map is used.
    geometry_model: str = "local_scan_point_v2"

    # Front-ray threshold measured from the LiDAR origin. The LiDAR extrinsic
    # matches the lidar_site in the Go2 MJCF.
    d_safe: float = 0.30
    robot_length_m: float = 0.70
    robot_width_m: float = 0.31
    lidar_x_m: float = 0.28945
    lidar_y_m: float = 0.0
    footprint_clearance_m: float = 0.15
    use_elliptic_footprint: bool = True
    # Collision termination geometry is independent from the LiDAR filter
    # footprint. "circle" is the 0.30 m collision proxy.
    collision_geometry: CollisionGeometry = "ellipse"
    collision_circle_radius_m: float = 0.30
    # Ground-truth geometry is used only for collision termination.  Inflate
    # the physical footprint slightly to cover model/contact uncertainty while
    # keeping it well inside the formal CBF safety boundary.
    collision_margin_m: float = 0.05

    # No-return rays are excluded from the local point barrier.  Inactive terms
    # receive a large positive dimensionless value and zero Lie derivatives.
    no_return_tolerance_m: float = 1.0e-4
    inactive_barrier_value: float = 20.0

    alpha: float = 1.5
    hocbf_k0: float = 2.0
    alpha_outer: float = 0.8
    kappa: float = 30.0
    kappa_backup: float = 50.0
    kappa_input: float = 20.0
    kappa_outer: float = 20.0

    # psi=hdot+k0 h has units of inverse seconds while geometric and input
    # barriers are dimensionless.  Divide psi by this positive rate scale
    # before LSE aggregation so all composed terms are dimensionless.
    barrier_rate_scale: float = 1.0

    backup_margin: float = 0.30
    backup_horizon_s: float = 4.0
    backup_samples: int = 21
    backup_rollout_substeps: int = 8

    terminal_backup_mode: TerminalBackupMode = "invariant_ellipse"
    # Positive normalization scales applied to the two dimensionless
    # invariant-ellipse terminal constituents before they enter the
    # kappa_backup LSE. 1.0 leaves them unscaled.
    terminal_geom_scale: float = 1.0
    terminal_rec_scale: float = 1.0

    # bcbf_cf projection: True projects onto the CBF half-space alone followed
    # by a clamp; False uses the joint box + half-space projection, so the
    # hard clip cannot invalidate the CBF condition.
    legacy_projection: bool = True

    control_dynamics_a: tuple[float, float, float] = (8.0, 8.0, 8.0)
    # bcbf_cf only: "sidestep-then-stop" backup target whose intent decays at
    # backup_intent_decay_rate, so the backup still comes to rest and the
    # terminal sets remain valid. (0, 0, 0) gives the zero-backup flow.
    backup_target_command: tuple[float, float, float] = (0.0, 0.0, 0.0)
    backup_intent_decay_rate: float = 2.0
    tracking_sigma: tuple[float, float, float] = (8.0, 8.0, 8.0)
    # Matches the 10 Hz high-level policy/filter update interval.
    integration_dt: float = 0.10
    denominator_epsilon: float = 1.0e-8
    # Optional post-projection safeguard. Disable it when the formal CBF
    # residual must correspond exactly to the unmodified projected command.
    hard_clip_output: bool = True
    # Training backend for the unaggregated bCBF-QP. The batched active-set
    # implementation remains on the torch device and uses a straight-through
    # backward pass. Exact OSQP is reserved for CPU deployment/evaluation.
    qp_max_active_candidates: int = 18
    qp_feasibility_tolerance: float = 1.0e-5
    qp_regularization: float = 1.0e-5
    qp_straight_through: bool = True
    qp_backend: QpBackend = "active_set"
    # bcbf_cf ablation: replace the closed-form projection Jacobian with a
    # straight-through gradient (as used by the QP filters). False keeps the
    # true closed-form gradient.
    cf_straight_through: bool = False


@dataclass(frozen=True)
class LaserSacConfig:
    navigation: HighLevelNavigationConfig = field(default_factory=HighLevelNavigationConfig)
    reward: LaserRewardConfig = field(default_factory=LaserRewardConfig)
    laser: LaserObservationConfig = field(default_factory=LaserObservationConfig)
    safety: LaserSafetyConfig = field(default_factory=LaserSafetyConfig)
    contact_collision: ContactCollisionConfig = field(default_factory=ContactCollisionConfig)
    collision_initialization: CollisionInitializationConfig = field(
        default_factory=CollisionInitializationConfig
    )
    sac: SacHyperparameters = field(default_factory=SacHyperparameters)
    method: LaserSacMethod = "vanilla"
    # Retain only this many high-level samples in memory. A compressed history
    # is written only for episodes that end in an obstacle collision.
    collision_trajectory_history_steps: int = 100

    @property
    def actor_observation_dim(self) -> int:
        # goal (2), measured body velocity (3), previous executed command (3),
        # laser ranges, previous nominal command (3), nominal-history valid (1)
        return 12 + self.laser.num_rays
