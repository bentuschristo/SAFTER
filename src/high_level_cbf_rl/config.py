from __future__ import annotations

from dataclasses import dataclass, field

# Reuses the laser pipeline's LiDAR/safety config dataclasses from SAFTER_src
# (falling back to src/ when SAFTER_src is not present).
try:
    from SAFTER_src.high_level_laser_sac.config import (
        ContactCollisionConfig,
        LaserObservationConfig,
        LaserSafetyConfig,
    )
except ModuleNotFoundError:
    from src.high_level_laser_sac.config import (
        ContactCollisionConfig,
        LaserObservationConfig,
        LaserSafetyConfig,
    )
from src.high_level_rl.config import HighLevelActionConfig, HighLevelNavigationConfig

# Action limits for this module (m/s, m/s, rad/s): within the frozen low-level
# policy's trained envelope vx in [-1, 2], vy in [-1, 1], wz in [-1, 1], and
# overriding the HighLevelActionConfig default (+-0.5/+-0.3/+-0.5).
CBF_RL_MAX_VX = 1.0
CBF_RL_MIN_VX = -1.0
CBF_RL_MAX_VY = 0.6
CBF_RL_MAX_WZ = 1.0

# Yang et al. run the high-level policy at 50 Hz, i.e. one decision per
# low-level (locomotion) step (the laser-SAC pipeline default is 10 Hz).
CBF_RL_HIGH_LEVEL_DT = 0.02

# Safety margin, expressed like --safety-clearance in train_SAC_policies.py:
# the desired front-ray threshold from the LiDAR origin (footprint_clearance_m
# is derived from it). Same 0.30 m as the SAC baselines.
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
    # high_level_dt == low_level_dt (0.02 s) gives one high-level decision per
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
        # Same collision-termination geometry as the SAC baselines
        # (circle, radius 0.30 m); the LaserSafetyConfig default is "ellipse".
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
