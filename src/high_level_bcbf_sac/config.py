from __future__ import annotations

from dataclasses import dataclass, field

from src.high_level_rl.config import HighLevelNavigationConfig


@dataclass(frozen=True)
class OrientationAwareRewardConfig:
    """Reward used by the SAC-CBF-CF experiment.

    The values are the latest set discussed for removing the reverse-driving
    loophole and encouraging the nominal actor to cooperate with the CBF layer.
    """

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


@dataclass(frozen=True)
class SacHyperparameters:
    hidden_dims: tuple[int, ...] = (512, 256, 128)
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    alpha_lr: float = 3.0e-4
    batch_size: int = 1024
    replay_size: int = 2_000_000
    learning_starts: int = 20_000
    updates_per_step: int = 1
    target_entropy: float = -3.0
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    gradient_clip_norm: float = 10.0


@dataclass(frozen=True)
class SacBcbfCfConfig:
    navigation: HighLevelNavigationConfig = field(default_factory=HighLevelNavigationConfig)
    reward: OrientationAwareRewardConfig = field(default_factory=OrientationAwareRewardConfig)
    sac: SacHyperparameters = field(default_factory=SacHyperparameters)
    # "bcbf_cf": backup CBF-CF with command dynamics.
    # "cbf_cd_cf": normal CBF-CF with command dynamics and input barriers.
    filter_kind: str = "bcbf_cf"
