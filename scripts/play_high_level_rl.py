"""Deploy a trained high-level PPO policy with the frozen Go2 locomotion policy."""
from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from src.high_level_control.command_interface import MjlabVelocityCommandInterface
from src.high_level_rl.config import HighLevelNavigationConfig
from src.high_level_rl.low_level import load_frozen_low_level_policy
from src.high_level_rl.navigation_env import HighLevelNavigationEnv


@dataclass
class PlayHighLevelRlConfig:
    low_level_checkpoint: str
    high_level_checkpoint: str
    num_envs: int = 1
    device: str | None = None
    viewer: Literal["auto", "native", "viser"] = "native"

    # Periodic high-level deployment diagnostics.
    print_high_level_stats: bool = False
    print_every: int = 10
    print_full_observation: bool = False

    navigation: HighLevelNavigationConfig = field(default_factory=HighLevelNavigationConfig)


def print_high_level_spec(env: HighLevelNavigationEnv) -> None:
    """Print the high-level observation, action, and reward specification once."""
    cfg = env.nav_cfg
    num_obstacles = int(getattr(cfg.arena, "max_obstacles", 0))

    observation_terms = [
        ("goal_position_body", 2, "Relative goal [gx_B, gy_B]"),
        ("base_velocity", 3, "Measured [vx_B, vy_B, wz_B]"),
        ("previous_safe_command", 3, "Previous safe [vx, vy, wz]"),
        (
            "obstacles",
            4 * num_obstacles,
            f"{num_obstacles} × [dx_B, dy_B, radius, mask]",
        ),
        ("wall_distances", 4, "[left, right, bottom, top]"),
    ]

    print()
    print("+-----------------------------------------------------------------------+")
    print("|                  High-Level Navigation Environment                    |")
    print("+-------------------------------+---------------------------------------+")
    print(f"| Observation dimension         | {cfg.observation_dim:<37} |")
    print(f"| Action dimension              | {3:<37} |")
    print(f"| High-level control rate       | {1.0 / cfg.high_level_dt:<37.3f} |")
    print(f"| Maximum episode steps         | {cfg.max_episode_steps:<37} |")
    print(f"| Goal tolerance [m]            | {cfg.goal_tolerance:<37.3f} |")
    print(f"| CBF safety enabled            | {str(cfg.safety.enabled):<37} |")
    print("+-------------------------------+---------------------------------------+")

    print()
    print("+--------------------------------------------------------------------------------+")
    print("|                         High-Level Observations                                |")
    print("+-------+---------------------------+-----------+--------------------------------+")
    print("| Index | Name                      | Dimension | Description                    |")
    print("+-------+---------------------------+-----------+--------------------------------+")

    index = 0
    total = 0
    for name, dimension, description in observation_terms:
        clipped_description = description[:30]
        print(
            f"| {index:^5} | {name:<25} | {dimension:^9} | "
            f"{clipped_description:<30} |"
        )
        index += 1
        total += dimension

    print("+-------+---------------------------+-----------+--------------------------------+")
    print(f"| Total                             | {total:^9} |                                |")
    print("+-----------------------------------+-----------+--------------------------------+")

    print()
    print("+------------------------------------------------------------------------+")
    print("|                        High-Level Action                               |")
    print("+-------+------------------+----------------------+----------------------+")
    print("| Index | Action           | Normalized range     | Physical range       |")
    print("+-------+------------------+----------------------+----------------------+")
    print(
        f"|   0   | forward velocity | [-1, 1]              | "
        f"[-{cfg.action.max_vx:.3f}, {cfg.action.max_vx:.3f}] m/s |"
    )
    print(
        f"|   1   | lateral velocity | [-1, 1]              | "
        f"[-{cfg.action.max_vy:.3f}, {cfg.action.max_vy:.3f}] m/s |"
    )
    print(
        f"|   2   | yaw rate         | [-1, 1]              | "
        f"[-{cfg.action.max_wz:.3f}, {cfg.action.max_wz:.3f}] rad/s |"
    )
    print("+-------+------------------+----------------------+----------------------+")

    reward = cfg.reward

    reward_terms = [
        (
            "progress",
            reward.progress_weight,
            "+ weight × (previous distance - current distance)",
        ),
        (
            "goal",
            reward.goal_bonus,
            "+ bonus when distance <= goal tolerance",
        ),
        (
            "time",
            -reward.time_penalty,
            "constant penalty each high-level step",
        ),
        (
            "smoothness",
            -reward.smoothness_weight,
            "- weight × ||u_safe(t)-u_safe(t-1)||²",
        ),
        (
            "fall",
            -reward.fall_penalty,
            "penalty when base height/orientation fails",
        ),
        (
            "collision",
            -reward.collision_penalty,
            "penalty for obstacle collision",
        ),
        (
            "boundary",
            -reward.boundary_penalty,
            "penalty for wall contact",
        ),
    ]

    print()
    print("+--------------------------------------------------------------------------------+")
    print("|                           High-Level Rewards                                   |")
    print("+-------+------------------+-------------+---------------------------------------+")
    print("| Index | Name             | Weight      | Definition                            |")
    print("+-------+------------------+-------------+---------------------------------------+")

    for index, (name, weight, definition) in enumerate(reward_terms):
        print(
            f"| {index:^5} | {name:<16} | {weight:^11.4f} | "
            f"{definition[:37]:<37} |"
        )

    print("+-------+------------------+-------------+---------------------------------------+")
    print()


def run(task_id: str, cfg: PlayHighLevelRlConfig):
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(task_id, play=True)
    low_agent_cfg = load_rl_cfg(task_id)
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.terminations = {}
    if "push_robot" in env_cfg.events:
        del env_cfg.events["push_robot"]

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    low_env = RslRlVecEnvWrapper(raw_env, clip_actions=low_agent_cfg.clip_actions)
    low_policy = load_frozen_low_level_policy(task_id, low_env, cfg.low_level_checkpoint, device)
    command_if = MjlabVelocityCommandInterface(low_env, command_name="twist")
    high_env = HighLevelNavigationEnv(low_env, low_policy, command_if, cfg.navigation)

    # Print the high-level specification once at startup.
    print_high_level_spec(high_env)

    high_env.print_high_level_stats = cfg.print_high_level_stats
    high_env.print_every = max(int(cfg.print_every), 1)
    high_env.print_full_observation = cfg.print_full_observation

    high_ckpt = Path(cfg.high_level_checkpoint).expanduser().resolve()
    if not high_ckpt.exists():
        raise FileNotFoundError(high_ckpt)
    high_agent_cfg = load_rl_cfg(task_id)
    high_agent_cfg.experiment_name = "go2_safe_navigation_ppo"
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(high_env, asdict(high_agent_cfg), device=device)
    runner.load(str(high_ckpt), load_cfg={"actor": True}, strict=False, map_location=device)
    high_policy = runner.get_inference_policy(device=device)

    if cfg.viewer == "auto":
        viewer = "native" if (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) else "viser"
    else:
        viewer = cfg.viewer

    if viewer == "native":
        NativeMujocoViewer(high_env, high_policy).run()
    else:
        ViserPlayViewer(high_env, high_policy).run()
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
    cfg = tyro.cli(PlayHighLevelRlConfig, args=remaining, prog=sys.argv[0] + f" {task}", config=mjlab.TYRO_FLAGS)
    run(task, cfg)


if __name__ == "__main__":
    main()
