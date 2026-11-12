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


@dataclass(frozen=True)
class SacCbfCfConfig:
    navigation: HighLevelNavigationConfig = field(default_factory=HighLevelNavigationConfig)
    reward: OrientationAwareRewardConfig = field(default_factory=OrientationAwareRewardConfig)
    sac: SacHyperparameters = field(default_factory=SacHyperparameters)
    # True: actor-integrated differentiable CBF-CF.
    # False: ordinary SAC; nominal action is executed after command clipping.
    use_safety_filter: bool = True
