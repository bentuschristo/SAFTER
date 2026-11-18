from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Literal

import yaml

# Reuses the laser pipeline's LiDAR/safety config dataclasses from SAFTER_src
# (falling back to src/ when SAFTER_src is not present).
try:
    from SAFTER_src.high_level_laser_sac.config import (
        ContactCollisionConfig,
        LaserObservationConfig,
        LaserRewardConfig,
        LaserSafetyConfig,
    )
except ModuleNotFoundError:
    from src.high_level_laser_sac.config import (
        ContactCollisionConfig,
        LaserObservationConfig,
        LaserRewardConfig,
        LaserSafetyConfig,
    )
from src.high_level_rl.config import HighLevelActionConfig, HighLevelNavigationConfig

# Action limits (m/s, m/s, rad/s): the asymmetric high-speed envelope of the
# laser-SAC methods, equal to the frozen low-level policy's trained command
# range vx in [-1, 2], vy in [-1, 1], wz in [-1, 1].
CBF_RL_MAX_VX = 2.0
CBF_RL_MIN_VX = -1.0
CBF_RL_MAX_VY = 1.0
CBF_RL_MAX_WZ = 1.0

# Yang et al. run the high-level policy at 50 Hz, i.e. one decision per
# low-level (locomotion) step.
CBF_RL_HIGH_LEVEL_DT = 0.02
# Laser-SAC high-level rate, used by default with new_reward_profile.
SAC_HIGH_LEVEL_DT = 0.10

CbfRlRewardProfile = Literal["original", "new_reward_profile"]

# Safety margin, expressed like --safety-clearance in train_SAC_policies.py:
# the desired front-ray threshold from the LiDAR origin (footprint_clearance_m
# is derived from it). Same 0.30 m as the SAC methods.
CBF_RL_SAFETY_CLEARANCE_M = 0.30


def _derive_footprint_clearance_m(safety_clearance_m: float, robot_length_m: float, lidar_x_m: float) -> float:
    """Convert a front-ray safety clearance to the additional body-footprint
    margin used by the local point-geometry CBF, with the same derivation as
    SAFTER_scripts/train_SAC_policies.py, so a given --safety-clearance value
    gives the same effective ellipse as the SAC methods."""
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

    Defaults match the cbf-rl-demo implementation (which splits the paper's
    single combined ``r_cbf`` term into a violation term and a
    separately-weighted match term), not the paper's Table II shorthand.
    Per-step shaping terms scale with ``high_level_dt / reference_dt`` so
    weights stay interpretable if the high-level rate changes, matching the
    convention used by ``LaserRewardConfig``.

    ``profile`` selects the navigation terms:
      ``original``           : Yang et al. progress/alive/goal/obstacle/wall/timeout.
      ``new_reward_profile`` : the laser-SAC navigation reward
                               (``LaserRewardConfig``, ``original`` profile:
                               progress, time, forward alignment, reverse
                               motion, yaw tracking, smoothness, goal, timeout,
                               collision) with the same weights.
    In both profiles the CBF violation/match terms are added when
    ``CbfRlModeConfig.use_cbf_reward_penalty`` is set, multiplied by
    ``cbf_reward_scale``.
    """

    profile: CbfRlRewardProfile = "original"
    # Navigation weights for new_reward_profile; its reference_dt (0.10 s) is
    # the SAC calibration rate.
    navigation_reward: LaserRewardConfig = field(default_factory=LaserRewardConfig)
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
    # Multiplier on the CBF violation/match terms. None: 1.0 for original;
    # for new_reward_profile, the ratio of the maximum per-step progress
    # reward of the two profiles,
    #   navigation_reward.progress_weight * max_vx * high_level_dt / progress_weight
    # (0.04 at 10 Hz and max_vx = 2 m/s), so the CBF terms keep the same weight
    # relative to progress as in the original profile.
    cbf_reward_scale: float | None = None


def resolve_cbf_reward_scale(reward: CbfRlRewardConfig, navigation: HighLevelNavigationConfig) -> float:
    if reward.cbf_reward_scale is not None:
        return float(reward.cbf_reward_scale)
    if reward.profile != "new_reward_profile":
        return 1.0
    max_progress_reward = (
        reward.navigation_reward.progress_weight
        * float(navigation.action.max_vx)
        * float(navigation.high_level_dt)
    )
    return max_progress_reward / float(reward.progress_weight)


def _default_navigation() -> HighLevelNavigationConfig:
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
        # Circle footprint (radius 0.30 m) as in the SAC methods; the
        # LaserSafetyConfig default is "ellipse".
        collision_geometry="circle",
        collision_circle_radius_m=0.30,
    )


@dataclass(frozen=True)
class CbfRlConfig:
    navigation: HighLevelNavigationConfig = field(default_factory=_default_navigation)
    reward: CbfRlRewardConfig = field(default_factory=CbfRlRewardConfig)
    laser: LaserObservationConfig = field(default_factory=_default_laser)
    safety: LaserSafetyConfig = field(default_factory=_default_safety)
    # Training termination; "direct_contact" (MuJoCo body-contact forces) is
    # the SAC training default. Evaluators use "circle", as the SAC evaluators do.
    contact_collision: ContactCollisionConfig = field(default_factory=ContactCollisionConfig)
    mode: CbfRlModeConfig = field(default_factory=CbfRlModeConfig)


def default_high_level_hz(reward_profile: CbfRlRewardProfile) -> float:
    """50 Hz for the original profile, 10 Hz (laser-SAC rate) for new_reward_profile."""
    dt = SAC_HIGH_LEVEL_DT if reward_profile == "new_reward_profile" else CBF_RL_HIGH_LEVEL_DT
    return 1.0 / dt


def with_high_level_rate(cfg: CbfRlConfig, high_level_hz: float) -> CbfRlConfig:
    """Set the policy, LiDAR refresh, and safety-filter integration rate together.

    The low-level controller stays at 50 Hz, so ``high_level_hz`` must divide
    50 Hz (e.g. 50, 25, 10); each command is held for 50 / high_level_hz
    low-level steps. Reward reference_dt values are calibration constants and
    are not changed.
    """
    if high_level_hz <= 0.0:
        raise ValueError(f"high_level_hz must be positive, got {high_level_hz}")
    dt = 1.0 / float(high_level_hz)
    navigation = replace(cfg.navigation, high_level_dt=dt)
    _ = navigation.low_steps_per_high_step  # validates the integer low-level ratio
    return replace(
        cfg,
        navigation=navigation,
        laser=replace(cfg.laser, update_dt=dt),
        safety=replace(cfg.safety, integration_dt=dt),
    )


def with_reward_profile(cfg: CbfRlConfig, reward_profile: CbfRlRewardProfile) -> CbfRlConfig:
    return replace(cfg, reward=replace(cfg.reward, profile=reward_profile))


class _ParamsLoader(yaml.SafeLoader):
    """SafeLoader that also reads the !!python/tuple tags written by dump_yaml."""


_ParamsLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda loader, node: tuple(loader.construct_sequence(node)),
)


@dataclass(frozen=True)
class SavedRunSettings:
    high_level_hz: float | None = None
    reward_profile: CbfRlRewardProfile | None = None
    cbf_reward_scale: float | None = None
    action: dict | None = None


def saved_run_settings(checkpoint: str) -> SavedRunSettings:
    """Rate, reward profile and action limits from the run's params/cbf_rl.yaml.

    All fields are None when the file is absent. Runs saved before the reward
    profile existed report "original".
    """
    params = Path(checkpoint).expanduser().resolve().parent / "params" / "cbf_rl.yaml"
    if not params.exists():
        return SavedRunSettings()
    with params.open() as file:
        saved = yaml.load(file, Loader=_ParamsLoader) or {}
    navigation = saved.get("navigation") or {}
    reward = saved.get("reward") or {}
    dt = navigation.get("high_level_dt")
    return SavedRunSettings(
        high_level_hz=1.0 / float(dt) if dt else None,
        reward_profile=reward.get("profile", "original"),
        cbf_reward_scale=reward.get("cbf_reward_scale"),
        action=navigation.get("action"),
    )


def resolve_checkpoint_config(
    cfg: CbfRlConfig,
    checkpoint: str,
    high_level_hz: float | None = None,
    reward_profile: CbfRlRewardProfile | None = None,
) -> CbfRlConfig:
    """Evaluation config matching the checkpoint's training run.

    Rate and reward profile: explicit arguments take precedence, then the
    values saved with the run, then the profile defaults. Action limits are
    taken from the saved run when available, since the policy's action
    scaling depends on them.
    """
    saved = saved_run_settings(checkpoint)
    if high_level_hz is not None and saved.high_level_hz is not None and abs(high_level_hz - saved.high_level_hz) > 1.0e-6:
        print(
            f"[CBF-RL] WARNING: evaluating at {high_level_hz:g} Hz, but the checkpoint "
            f"was trained at {saved.high_level_hz:g} Hz"
        )
    profile = reward_profile or saved.reward_profile or "original"
    hz = high_level_hz or saved.high_level_hz or default_high_level_hz(profile)
    resolved = with_high_level_rate(with_reward_profile(cfg, profile), hz)
    if saved.cbf_reward_scale is not None:
        resolved = replace(resolved, reward=replace(resolved.reward, cbf_reward_scale=saved.cbf_reward_scale))
    if saved.action:
        known = {f.name for f in fields(HighLevelActionConfig)}
        action = replace(resolved.navigation.action, **{k: v for k, v in saved.action.items() if k in known})
        resolved = replace(resolved, navigation=replace(resolved.navigation, action=action))
    action = resolved.navigation.action
    print(
        f"[CBF-RL] high_level_hz={hz:g} reward_profile={profile} "
        f"action vx=[{action.min_vx:g}, {action.max_vx:g}] vy=+-{action.max_vy:g} wz=+-{action.max_wz:g}"
    )
    return resolved
