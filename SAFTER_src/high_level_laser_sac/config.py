from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from SAFTER_src.high_level_rl.config import HighLevelNavigationConfig


@dataclass(frozen=True)
class SacHyperparameters:
    """Copied from (was previously imported cross-package from) the old
    map-based src/high_level_sac/config.py -- inlined here so SAFTER_src has
    no dependency on that package. Keep in sync manually if it changes there."""

    hidden_dims: tuple[int, ...] = (512, 256, 128)
    # gamma and tau are both *per-decision* persistence/decay factors, not
    # per-second ones -- gamma compounds once per critic-target bootstrap
    # (one high-level step) and tau's Polyak average is applied once per
    # gradient update (one high-level step at update_every_steps=1). At a
    # fixed nominal value, changing navigation.high_level_dt silently changes
    # what these factors mean in real time: e.g. gamma=0.99 compounds to an
    # effective per-second discount of 0.99**10=0.904 at high_level_dt=0.10
    # (10 Hz) but only 0.99**50=0.605 at high_level_dt=0.02 (50 Hz) -- the
    # agent's effective planning horizon shrinks from ~10s to ~2s even though
    # the reward magnitude was already correctly dt-scaled elsewhere. Both
    # values below are calibrated at reference_dt; the trainer applies
    # gamma_effective = gamma**(high_level_dt/reference_dt) and
    # tau_effective = 1-(1-tau)**(high_level_dt/reference_dt) so the same
    # *effective per-second* discounting/target-tracking rate is preserved
    # at any high_level_dt. 1.0 ratio (hence no change) at the archived
    # default high_level_dt == reference_dt.
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


LaserSacMethod = Literal[
    "vanilla", "cbf_cf_clip", "cbf_qp", "cbf_lse_qp",
    "cbf_cd_cf", "cbf_lse_cd_qp", "bcbf_cf", "bcbf_qp", "bcbf_lse_cd_qp",
    "bcbf_nonlse_cd_qp", "bcbf_lse_cd_qp_cfmatched", "cbf_lse_cd_qp_cfmatched",
    # Zero-backup LSE-aggregated QP WITHOUT control dynamics: the decision
    # variable is the directly-applied body command (no first-order command-
    # state surrogate), backup law is exactly zero velocity (kinematic flow,
    # no coasting), and the six input-limit rows are literal hard box
    # constraints rather than a softened/aggregated barrier. See
    # LaserZeroBackupLseQp in laser_cbf_filters.py.
    "bcbf_lse_qp",
]
CollisionGeometry = Literal["circle", "ellipse"]
CollisionTerminationMode = Literal["circle", "seanav_contact"]
LidarNoisePreset = Literal["clean", "nominal", "moderate", "severe"]
# "legacy_inflated_ellipse": original velocity-inflated terminal ellipse
#   (delta_b(T) = backup_margin + sqrt(vx^2+vy^2+eps^2)-eps folded directly
#   into the semi-axes). Not generally forward-invariant under the
#   zero-surrogate backup (see laser_cbf_filters.py terminal_h_and_grad_future
#   docstring) -- kept only for old-vs-new comparison.
# "invariant_ellipse": fixed-geometry terminal ellipse (R_x=r_x+backup_margin,
#   R_y=r_y+backup_margin, no velocity folded into the axes) intersected with
#   a separate residual-motion recoverability condition. See
#   terminal_invariant_h_and_grad_future.
TerminalBackupMode = Literal["legacy_inflated_ellipse", "invariant_ellipse"]
QpBackend = Literal["active_set", "qpth", "osqp"]
RewardProfile = Literal["original", "seanav_comparable"]

LIDAR_NOISE_BOUNDS_M: dict[str, tuple[float, float]] = {
    "clean": (0.0, 0.0),
    "nominal": (0.005, 0.030),
    "moderate": (0.005, 0.050),
    # Retain the old 0.15 m far-range bound only for severe robustness tests.
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
    # How often (in seconds) the LiDAR is actually re-sampled, independent of
    # how often the high-level policy is queried (navigation.high_level_dt).
    # Defaults to matching the archived 10 Hz high-level cadence exactly, so
    # any existing config with high_level_dt=0.10 gets a refresh interval of
    # 1 step -- i.e. unchanged behavior (a fresh reading every high-level
    # step). Set navigation.high_level_dt faster (e.g. 0.02, matching
    # low_level_dt) while leaving this at 0.10 to decouple decision rate from
    # sensor rate: the policy is queried every step, but reuses the same
    # cached scan across the intervening steps until this period elapses,
    # modeling a LiDAR that physically only spins/refreshes at its own rate.
    update_dt: float = 0.10


@dataclass(frozen=True)
class LaserRewardConfig:
    # ``original`` is the archived 10 Hz high-level reward.  The separate
    # ``seanav_comparable`` launcher changes only the analogous goal,
    # collision/unsafe-termination, and timeout terms; all other shaping terms
    # below remain unchanged by design.
    profile: RewardProfile = "original"
    # The high_level_dt every weight below (except progress_weight and the
    # one-time episode-outcome bonuses/penalties) was tuned at, and every
    # currently-deployed checkpoint was trained at. Per-step shaping terms
    # (time, forward_alignment, reverse_motion, yaw_tracking, smoothness,
    # cbf_correction) are auto-scaled by navigation.high_level_dt /
    # reference_dt at reward-computation time (see step_safe), so their
    # *per-second* contribution stays approximately constant however often
    # the policy is actually queried -- e.g. running at high_level_dt=0.02
    # (5x more steps/sec than this reference) automatically divides each of
    # those weights by 5, rather than silently applying them 5x more often
    # per second as raw unscaled per-step constants would. progress_weight is
    # deliberately excluded: progress is a physical per-step distance delta
    # that already shrinks proportionally with a smaller dt on its own, so
    # scaling it again would double-count. The one-time episode-outcome terms
    # (goal_bonus, timeout_penalty, collision_penalty) are also excluded --
    # they fire once per episode outcome, not once per second, so they don't
    # have the "accumulates faster at a smaller dt" problem in the first
    # place. At the default high_level_dt == reference_dt, the scale factor
    # is exactly 1.0 and every weight below behaves byte-for-byte as before
    # -- this is purely additive, not a behavior change at the archived
    # operating point.
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
    # SEA-Nav's own reward has no timeout term at all (matched here by the
    # old 0.0 default) -- but that is only safe in SEA-Nav's own reward
    # because it also has a "stuck" term (scale=-5.0: dead-end/no-progress
    # penalty) that discourages a policy from loitering near the goal to
    # dodge collision risk instead of committing to the final approach.
    # "stuck" was never ported to this reward (see the reward-recheck
    # table), so seanav_timeout_scale=0.0 here removed ALL pressure against
    # exactly that stalling failure mode -- plausibly why seaspeed
    # seanav_comparable training was observed to reach a good success rate
    # early and then erode as training continued (SAC's exploration entropy
    # decays, letting the policy settle into the now-unpunished stall).
    # 2026-09-14: set to SEA-Nav's own "stuck" scale as the closest existing
    # analog (both terms penalize "failed to make progress"), rather than an
    # arbitrary new value, restoring the missing pressure without inventing
    # an unrelated one.
    seanav_timeout_scale: float = -5.0


@dataclass(frozen=True)
class ContactCollisionConfig:
    """Physical collision semantics derived from MuJoCo contact forces.

    The defaults reproduce SEA-Nav's termination thresholds: terminal body
    contacts above 1 N, or any non-foot body contact above 50 N.  The lower
    penalized-contact threshold is used by the reward only.
    """

    # ``circle`` reproduces the archived SAC footprint-overlap termination.
    # ``seanav_contact`` uses MuJoCo body-contact forces as a simulator-native
    # approximation of SEA-Nav's Isaac Gym contact termination.
    termination_mode: CollisionTerminationMode = "seanav_contact"
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

    # Backward-compatible front-ray threshold measured from the LiDAR origin.
    # The LiDAR extrinsic matches the lidar_site in the Go2 MJCF exactly.
    d_safe: float = 0.30
    robot_length_m: float = 0.70
    robot_width_m: float = 0.31
    lidar_x_m: float = 0.28945
    lidar_y_m: float = 0.0
    footprint_clearance_m: float = 0.15
    use_elliptic_footprint: bool = True
    # Collision termination geometry is independent from the LiDAR filter
    # footprint. "circle" reproduces the archived 0.30 m collision proxy.
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

    # v6 terminal-set construction. invariant_ellipse (the new, two-condition
    # forward-invariant terminal set: h_B_geom + h_B_rec) is now the default,
    # per 2026-09-10 evaluation results -- "legacy_inflated_ellipse" remains
    # available (e.g. to reproduce pre-2026-09-07 checkpoints) but is no
    # longer the default.
    terminal_backup_mode: TerminalBackupMode = "invariant_ellipse"
    # Positive normalization scales applied to the two new dimensionless
    # invariant-ellipse terminal constituents before they enter the
    # kappa_backup LSE. Left at 1.0 (unscaled) until the unnormalized
    # distributions have been logged and inspected -- do not tune silently.
    terminal_geom_scale: float = 1.0
    terminal_rec_scale: float = 1.0

    # 2026-09-10: independent of terminal_backup_mode above -- this instead
    # controls which projection step bcbf_cf's filter uses. False (default)
    # is the 2026-09-07 joint box+half-space projection (zoh_surrogate_bounds
    # + _exact_box_halfspace_projection), which bounds the achievable
    # surrogate command given the ZOH decay BEFORE projecting, so a
    # subsequent hard clip can't invalidate the CBF guarantee. True reverts
    # to the pre-fix behavior: project onto the CBF half-space alone
    # (_exact_projection, no box baked in) and rely on a naive post-hoc
    # clamp -- the same projection cbf_cd_cf's filter has always used, and
    # the same one every bcbf_cf checkpoint trained before 2026-09-07 used.
    # Lets the terminal-set change and the projection change be isolated
    # one at a time instead of always moving together. True (the pre-2026-09-07
    # single-half-space projection) is now the default, per 2026-09-10
    # evaluation results on "clustered" -- the joint box+half-space projection
    # remains available (legacy_projection=False) for anyone who wants the
    # stricter hard-clip-consistency guarantee instead.
    legacy_projection: bool = True

    control_dynamics_a: tuple[float, float, float] = (8.0, 8.0, 8.0)
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
    # bcbf_cf-only ablation switch: LaserZeroBackupLseCbfClosedForm is natively
    # closed-form differentiable (the true CBF-projection Jacobian flows into
    # the policy), so this defaults to False -- every existing/default bcbf_cf
    # checkpoint keeps training under the true gradient. Setting True swaps in
    # the same straight-through trick qp_straight_through already uses for the
    # QP filters (output = desired + (output - desired).detach()), isolating
    # whether the CF filter's differentiability itself (vs. just executing
    # safe actions + the correction reward) is what drives bcbf_cf's benefit.
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
        # goal_b(2), measured command(3), previous safe command(3),
        # laser ranges, previous nominal command(3), nominal-history valid(1)
        return 12 + self.laser.num_rays
