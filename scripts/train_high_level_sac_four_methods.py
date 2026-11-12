"""Unified launcher for the four high-level SAC variants.

Methods:
  vanilla      : ordinary SAC baseline, no safety filter in training path.
  cbf_cf_clip  : normal first-order LSE-CBF-CF; output is clipped to command box.
  cbf_cd_cf    : normal LSE-CBF-CF with command dynamics and input barriers.
  bcbf_cf      : backup LSE-bCBF-CF with command dynamics and input barriers.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
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
from src.high_level_sac.config import SacCbfCfConfig
from src.high_level_sac.sac_navigation_env import SacCbfNavigationEnv as SacClipEnv
from src.high_level_sac.trainer import SacCbfCfTrainer
from src.high_level_bcbf_sac.config import SacBcbfCfConfig
from src.high_level_bcbf_sac.sac_navigation_env import SacCbfNavigationEnv as SacCdEnv
from src.high_level_bcbf_sac.trainer import SacBcbfCfTrainer


@dataclass
class TrainFourMethodSacConfig:
    low_level_checkpoint: str
    method: Literal["vanilla", "cbf_cf_clip", "cbf_cd_cf", "bcbf_cf"] = "vanilla"
    num_envs: int = 256
    device: str | None = None
    vector_steps: int = 200_000
    save_interval: int = 1_000
    log_interval: int = 10
    seed: int = 0
    run_name: str = ""
    resume_checkpoint: str | None = None
    # Used by vanilla/cbf_cf_clip.
    cbf_config: SacCbfCfConfig = field(default_factory=SacCbfCfConfig)
    # Used by cbf_cd_cf/bcbf_cf.
    cd_config: SacBcbfCfConfig = field(default_factory=SacBcbfCfConfig)


def run(task_id: str, cfg: TrainFourMethodSacConfig) -> None:
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

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = f"_{cfg.run_name}" if cfg.run_name else f"_{cfg.method}_seed{cfg.seed}"

    if cfg.method in ("vanilla", "cbf_cf_clip"):
        method_cfg = replace(
            cfg.cbf_config,
            use_safety_filter=(cfg.method == "cbf_cf_clip"),
        )
        env = SacClipEnv(
            low_env,
            low_policy,
            command_if,
            method_cfg.navigation,
            reward_cfg=method_cfg.reward,
        )
        experiment = "go2_safe_navigation_sac_cbf_cf"
        log_dir = Path("logs") / "sac" / experiment / f"{timestamp}{suffix}"
        trainer = SacCbfCfTrainer(env, method_cfg, log_dir, device)
    else:
        method_cfg = replace(cfg.cd_config, filter_kind=cfg.method)
        env = SacCdEnv(
            low_env,
            low_policy,
            command_if,
            method_cfg.navigation,
            reward_cfg=method_cfg.reward,
        )
        experiment = (
            "go2_safe_navigation_sac_bcbf_cf"
            if cfg.method == "bcbf_cf"
            else "go2_safe_navigation_sac_cbf_cd_cf"
        )
        log_dir = Path("logs") / "sac" / experiment / f"{timestamp}{suffix}"
        trainer = SacBcbfCfTrainer(env, method_cfg, log_dir, device)

    if cfg.resume_checkpoint:
        checkpoint = Path(cfg.resume_checkpoint).expanduser().resolve()
        print(f"[SAC] Resuming from {checkpoint}")
        trainer.load(checkpoint, load_optimizers=True)

    print(f"[SAC] method={cfg.method} logging to {log_dir}")
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
        TrainFourMethodSacConfig,
        args=remaining,
        prog=sys.argv[0] + f" {task}",
        config=mjlab.TYRO_FLAGS,
    )
    run(task, cfg)


if __name__ == "__main__":
    main()
