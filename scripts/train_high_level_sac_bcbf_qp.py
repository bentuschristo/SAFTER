"""Train SAC with Python bCBF-QP safety-layer variants.

Variants:
  cbf_qp_lim       normal direct-command CBF-QP with hard input limits; no backup, no CD, no LSE
  cbf_lse_qp_lim   normal direct-command CBF-QP with one LSE CBF row + hard input limits; no backup, no CD
  noncd_nonlse_qp  direct-command bCBF-QP, individual backup rows
  noncd_lse_qp     direct-command bCBF-QP, one LSE backup row
  cd_nonlse_qp     control-dynamics bCBF-QP, individual backup rows
  cd_lse_qp        control-dynamics bCBF-QP, one LSE backup row

The bCBF parameters are taken from HighLevelNavigationConfig.backup_safety,
matching the existing bCBF-CF configuration by default.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch
import tyro
import mjlab

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends

from src.high_level_control.command_interface import MjlabVelocityCommandInterface
from src.high_level_rl.low_level import load_frozen_low_level_policy
from src.high_level_bcbf_qp_sac.config import SacBcbfQpConfig
from src.high_level_bcbf_qp_sac.sac_navigation_env import SacCbfNavigationEnv
from src.high_level_bcbf_qp_sac.trainer import SacBcbfQpTrainer

QpVariant = Literal["cbf_qp_lim", "cbf_lse_qp_lim", "noncd_nonlse_qp", "noncd_lse_qp", "cd_nonlse_qp", "cd_lse_qp"]


@dataclass
class TrainSacBcbfQpConfig:
    low_level_checkpoint: str
    qp_variant: QpVariant = "cd_lse_qp"
    num_envs: int = 256
    device: str | None = None
    vector_steps: int = 200_000
    save_interval: int = 1_000
    log_interval: int = 10
    experiment_name: str = "go2_safe_navigation_sac_bcbf_qp"
    run_name: str = ""
    seed: int = 42
    resume_checkpoint: str | None = None
    max_active_candidates: int = 18
    straight_through: bool = True
    config: SacBcbfQpConfig = field(default_factory=SacBcbfQpConfig)


def _with_cli_overrides(cfg: TrainSacBcbfQpConfig) -> SacBcbfQpConfig:
    # tyro nested dataclass overrides still work, but these top-level flags are
    # convenient when launching sweeps.
    qp_cfg = cfg.config.qp.__class__(
        variant=cfg.qp_variant,
        max_active_candidates=cfg.max_active_candidates,
        straight_through=cfg.straight_through,
    )
    return SacBcbfQpConfig(
        navigation=cfg.config.navigation,
        reward=cfg.config.reward,
        sac=cfg.config.sac,
        qp=qp_cfg,
    )


def run(task_id: str, cfg: TrainSacBcbfQpConfig) -> None:
    configure_torch_backends()
    torch.manual_seed(cfg.seed)
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    config = _with_cli_overrides(cfg)

    env_cfg = load_env_cfg(task_id, play=False)
    low_agent_cfg = load_rl_cfg(task_id)
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.terminations = {}
    if "push_robot" in env_cfg.events:
        del env_cfg.events["push_robot"]

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    low_env = RslRlVecEnvWrapper(raw_env, clip_actions=low_agent_cfg.clip_actions)
    low_policy = load_frozen_low_level_policy(task_id, low_env, cfg.low_level_checkpoint, device)
    command_if = MjlabVelocityCommandInterface(low_env, command_name="twist")
    env = SacCbfNavigationEnv(
        low_env,
        low_policy,
        command_if,
        config.navigation,
        reward_cfg=config.reward,
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix_parts = [cfg.qp_variant]
    if cfg.run_name:
        suffix_parts.append(cfg.run_name)
    suffix = "_" + "-".join(suffix_parts)
    log_dir = Path("logs") / "sac" / cfg.experiment_name / f"{timestamp}{suffix}"

    trainer = SacBcbfQpTrainer(env, config, log_dir, device)
    if cfg.resume_checkpoint:
        checkpoint = Path(cfg.resume_checkpoint).expanduser().resolve()
        print(f"[SAC-QP] Resuming from {checkpoint}")
        trainer.load(checkpoint, load_optimizers=True)

    print(f"[SAC-QP] Logging to {log_dir}")
    trainer.train(cfg.vector_steps, cfg.save_interval, cfg.log_interval)
    env.close()


def main() -> None:
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401

    task, remaining = tyro.cli(
        tyro.extras.literal_type_from_choices(list_tasks()),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    cfg = tyro.cli(
        TrainSacBcbfQpConfig,
        args=remaining,
        prog=sys.argv[0] + f" {task}",
        config=mjlab.TYRO_FLAGS,
    )
    run(task, cfg)


if __name__ == "__main__":
    main()
