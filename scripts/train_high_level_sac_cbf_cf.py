"""Train either SAC-CBF-CF or an apple-to-apple SAC baseline without CBF."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends

from src.high_level_control.command_interface import MjlabVelocityCommandInterface
from src.high_level_rl.low_level import load_frozen_low_level_policy
from src.high_level_sac.config import SacCbfCfConfig
from src.high_level_sac.sac_navigation_env import SacCbfNavigationEnv
from src.high_level_sac.trainer import SacCbfCfTrainer


@dataclass
class TrainSacCbfCfConfig:
    low_level_checkpoint: str
    num_envs: int = 256
    device: str | None = None
    vector_steps: int = 20_000
    save_interval: int = 1_000
    log_interval: int = 10
    experiment_name: str = "go2_safe_navigation_sac_cbf_cf"
    run_name: str = "orientation_reward_v1"
    seed: int = 42
    resume_checkpoint: str | None = None
    config: SacCbfCfConfig = field(default_factory=SacCbfCfConfig)


def run(task_id: str, cfg: TrainSacCbfCfConfig) -> None:
    configure_torch_backends()
    torch.manual_seed(cfg.seed)
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

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
        cfg.config.navigation,
        reward_cfg=cfg.config.reward,
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = f"_{cfg.run_name}" if cfg.run_name else ""
    log_dir = Path("logs") / "sac" / cfg.experiment_name / f"{timestamp}{suffix}"
    trainer = SacCbfCfTrainer(env, cfg.config, log_dir, device)
    if cfg.resume_checkpoint:
        checkpoint = Path(cfg.resume_checkpoint).expanduser().resolve()
        print(f"[SAC] Resuming from {checkpoint}")
        trainer.load(checkpoint, load_optimizers=True)

    print(f"[SAC] Logging to {log_dir}")
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
        TrainSacCbfCfConfig,
        args=remaining,
        prog=sys.argv[0] + f" {task}",
        config=mjlab.TYRO_FLAGS,
    )
    run(task, cfg)


if __name__ == "__main__":
    main()
