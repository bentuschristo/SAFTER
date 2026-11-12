from __future__ import annotations

from dataclasses import dataclass, field

# NOTE: cross-package dependency on SAFTER_src, same rationale as
# cbf_rl_navigation_env.py's LaserSacNavigationEnv import -- CbfRlNavigationEnv
# reuses the laser pipeline's LiDAR/safety config dataclasses directly rather
# than duplicating them.
try:
    # Local repo state: high_level_laser_sac lives under SAFTER_src/.
    from SAFTER_src.high_level_laser_sac.config import (
        ContactCollisionConfig,
        LaserObservationConfig,
        LaserSafetyConfig,
    )
except ModuleNotFoundError:
    # Server state: SAFTER_src/ hasn't been synced there -- fall back to the
    # original location.
    from src.high_level_laser_sac.config import (
        ContactCollisionConfig,
        LaserObservationConfig,
        LaserSafetyConfig,
    )
from src.high_level_rl.config import HighLevelActionConfig, HighLevelNavigationConfig

# User-specified action limits (2026-10-03): symmetric vx/vy/wz envelope,
# deliberately narrower than the frozen low-level policy's own
# final-curriculum-stage trained envelope (-1.0, 2.0)/(-1.0, 1.0)/(-1.0, 1.0)
# -- see logs/rsl_rl/go2_velocity/2026-06-19_13-24-10/params/env.yaml. Still
# overrides the repo-wide HighLevelActionConfig default (+-0.5/+-0.3/+-0.5)
# for this module specifically.
CBF_RL_MAX_VX = 1.0
CBF_RL_MIN_VX = -1.0
CBF_RL_MAX_VY = 0.6
CBF_RL_MAX_WZ = 1.0

# Yang et al. run the high-level policy at 50 Hz; matching that here means the
# high-level decision/filter loop runs every low-level (locomotion) step
# instead of the laser-SAC pipeline's default 10 Hz (5 low-level steps per
# high-level step).
CBF_RL_HIGH_LEVEL_DT = 0.02

# User-specified safety margin for this module, expressed the same way the
# --safety-clearance CLI flag in scripts/train_SAC_policies.py
# does: the desired front-ray threshold measured from the LiDAR origin (not
# footprint_clearance_m directly -- the laser SAC pipeline's own
# footprint_clearance_m default is a separate 0.15 m). This is the same 0.30 m
# value used by the handcrafted2 SAC baseline runs.
CBF_RL_SAFETY_CLEARANCE_M = 0.30


def _derive_footprint_clearance_m(safety_clearance_m: float, robot_length_m: float, lidar_x_m: float) -> float:
    """Convert a front-ray safety clearance to the additional body-footprint
    margin used by the local point-geometry CBF, exactly matching the
    derivation in scripts/train_SAC_policies.py: run(),
    so a given --safety-clearance value produces the identical effective
    ellipse size as the SAC baseline it's being compared against."""
    front_body_distance = max(0.5 * robot_length_m - lidar_x_m, 1.0e-8)
    return max(0.0, safety_clearance_m - front_body_distance)


@dataclass(frozen=True)
class CbfRlModeConfig:
    """Ablation switches reproducing Yang et al. Table III.

    (use_cbf_action_filtering, use_cbf_reward_penalty):
      (False, False) -> Nominal
      (True,  False) -> Filter-only
      (False, True)  -> Reward-only
      (True,  True)  -> Dual
    """

    use_cbf_action_filtering: bool = True
    use_cbf_reward_penalty: bool = True


@dataclass(frozen=True)
class CbfRlRewardConfig:
    """Generalizable CBF-RL reward (Yang et al. Table II / cbf-rl-demo).

    Defaults match the validated cbf-rl-demo implementation (which splits the
    paper's single combined ``r_cbf`` term into a violation term and a
    separately-weighted match term), not the paper's Table II shorthand.
    Per-step shaping terms scale with ``high_level_dt / reference_dt`` so
    weights stay interpretable if the high-level rate changes, matching the
    convention already used by ``LaserRewardConfig``.
    """

    reference_dt: float = CBF_RL_HIGH_LEVEL_DT
    goal_reward: float = 1.0
    obstacle_penalty: float = 1.0
    wall_penalty: float = 1.0
    progress_weight: float = 20.0
    alive_reward: float = 0.01
    timeout_penalty: float = 10.0
    cbf_violation_weight: float = 10.0
    cbf_match_weight: float = 100.0
    cbf_match_sigma: float = 0.5


def _default_navigation() -> HighLevelNavigationConfig:
    # low_level_dt defaults to 0.02 already (the frozen Go2 velocity policy's
    # own 50 Hz control rate); setting high_level_dt to the same value makes
    # low_steps_per_high_step == 1, i.e. a fresh high-level decision every
    # low-level step, matching the paper's 50 Hz policy rate.
    action = HighLevelActionConfig(
        max_vx=CBF_RL_MAX_VX, min_vx=CBF_RL_MIN_VX, max_vy=CBF_RL_MAX_VY, max_wz=CBF_RL_MAX_WZ
    )
    return HighLevelNavigationConfig(high_level_dt=CBF_RL_HIGH_LEVEL_DT, action=action)


def _default_laser() -> LaserObservationConfig:
    return LaserObservationConfig(update_dt=CBF_RL_HIGH_LEVEL_DT)


def _default_safety() -> LaserSafetyConfig:
    defaults = LaserSafetyConfig()
    footprint_clearance_m = _derive_footprint_clearance_m(
        CBF_RL_SAFETY_CLEARANCE_M, defaults.robot_length_m, defaults.lidar_x_m
    )
    return LaserSafetyConfig(
        d_safe=CBF_RL_SAFETY_CLEARANCE_M,
        footprint_clearance_m=footprint_clearance_m,
        # LaserSacNavigationEnv requires integration_dt == navigation.high_level_dt.
        integration_dt=CBF_RL_HIGH_LEVEL_DT,
        # Matches the handcrafted2 SAC baseline's own training-time
        # collision-termination geometry exactly (its training commands
        # pass --collision-geometry circle --collision-circle-radius-m
        # 0.30). The bare LaserSafetyConfig default is "ellipse" -- left
        # unset here until 2026-09-18, so every CBF-RL checkpoint trained
        # before that date actually used ellipse-based termination during
        # training (the eval script already overrode to "circle" to match
        # SAC's eval convention, so this only affects training-time
        # collision detection, not the evaluation numbers already gathered).
        collision_geometry="circle",
        collision_circle_radius_m=0.30,
    )


@dataclass(frozen=True)
class CbfRlConfig:
    navigation: HighLevelNavigationConfig = field(default_factory=_default_navigation)
    reward: CbfRlRewardConfig = field(default_factory=CbfRlRewardConfig)
    laser: LaserObservationConfig = field(default_factory=_default_laser)
    safety: LaserSafetyConfig = field(default_factory=_default_safety)
    contact_collision: ContactCollisionConfig = field(default_factory=ContactCollisionConfig)
    mode: CbfRlModeConfig = field(default_factory=CbfRlModeConfig)
