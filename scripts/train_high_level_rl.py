"""Train a PPO high-level navigation actor around a frozen Go2 locomotion actor."""
from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

from src.high_level_control.command_interface import MjlabVelocityCommandInterface
from src.high_level_rl.config import HighLevelNavigationConfig
from src.high_level_rl.low_level import load_frozen_low_level_policy
from src.high_level_rl.navigation_env import HighLevelNavigationEnv


@dataclass
class TrainHighLevelConfig:
    low_level_checkpoint: str
    num_envs: int = 64
    device: str | None = None
    max_iterations: int = 3000
    save_interval: int = 100
    experiment_name: str = "go2_safe_navigation_ppo"
    run_name: str = ""
    seed: int = 42
    resume_checkpoint: str | None = None
    navigation: HighLevelNavigationConfig = field(default_factory=HighLevelNavigationConfig)


def run(task_id: str, cfg: TrainHighLevelConfig):
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(task_id)
    low_agent_cfg = load_rl_cfg(task_id)
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.seed = cfg.seed
    # High-level code handles episode endings and reset sampling.
    env_cfg.terminations = {}
    if "push_robot" in env_cfg.events:
        del env_cfg.events["push_robot"]

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    low_env = RslRlVecEnvWrapper(raw_env, clip_actions=low_agent_cfg.clip_actions)
    low_policy = load_frozen_low_level_policy(task_id, low_env, cfg.low_level_checkpoint, device)
    command_if = MjlabVelocityCommandInterface(low_env, command_name="twist")
    high_env = HighLevelNavigationEnv(low_env, low_policy, command_if, cfg.navigation)

    # Reuse the repository's PPO defaults, but create a separate experiment.
    high_agent_cfg = load_rl_cfg(task_id)
    high_agent_cfg.experiment_name = cfg.experiment_name
    high_agent_cfg.run_name = cfg.run_name
    high_agent_cfg.seed = cfg.seed
    high_agent_cfg.max_iterations = cfg.max_iterations
    if hasattr(high_agent_cfg, "save_interval"):
        high_agent_cfg.save_interval = cfg.save_interval

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = f"_{cfg.run_name}" if cfg.run_name else ""
    log_dir = Path("logs/rsl_rl") / cfg.experiment_name / f"{stamp}{suffix}"
    log_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(log_dir / "params" / "navigation.yaml", asdict(cfg.navigation))
    dump_yaml(log_dir / "params" / "agent.yaml", asdict(high_agent_cfg))

    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(high_env, asdict(high_agent_cfg), str(log_dir), device)

    if cfg.resume_checkpoint is not None:
        checkpoint = Path(cfg.resume_checkpoint).expanduser().resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"High-level checkpoint not found: {checkpoint}"
            )

        print(f"[TRAIN] Resuming high-level policy from: {checkpoint}")
        runner.load(str(checkpoint))

    print(f"[TRAIN] High-level observation dim: {cfg.navigation.observation_dim}")
    print("[TRAIN] High-level action dim: 3")
    print(f"[TRAIN] Logging to: {log_dir}")
    runner.learn(num_learning_iterations=cfg.max_iterations, init_at_random_ep_len=True)
    high_env.close()


def main():
    import mjlab.tasks  # noqa
    import src.tasks  # noqa
    task, remaining = tyro.cli(
        tyro.extras.literal_type_from_choices(list_tasks()),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    cfg = tyro.cli(TrainHighLevelConfig, args=remaining, prog=sys.argv[0] + f" {task}", config=mjlab.TYRO_FLAGS)
    run(task, cfg)


if __name__ == "__main__":
    main()
