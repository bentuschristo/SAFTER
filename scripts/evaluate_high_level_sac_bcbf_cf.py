"""Evaluate a trained high-level SAC policy on one fixed navigation scenario.

Logs one CSV row per episode and a one-row summary CSV with:
- success rate
- collision rate
- path length
- time to goal

Path length and time-to-goal summary statistics are computed over successful
episodes only. Raw per-episode path length and duration are retained for all
outcomes.
"""
from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends

from src.high_level_control.command_interface import MjlabVelocityCommandInterface
from src.high_level_rl.low_level import load_frozen_low_level_policy
from src.high_level_bcbf_sac.config import SacBcbfCfConfig
from src.high_level_sac.evaluation_scenarios import SCENARIOS
from src.high_level_bcbf_sac.networks import SafeSacActor
from src.high_level_bcbf_sac.sac_navigation_env import SacCbfNavigationEnv


@dataclass
class EvaluateSacConfig:
    low_level_checkpoint: str
    high_level_checkpoint: str
    scenario: Literal["easy", "medium", "hard"]
    method_name: str
    episodes: int = 100
    device: str | None = None
    seed: int = 0
    output_dir: str = "logs/evaluation/go2_safe_navigation"
    print_every_episode: int = 1
    config: SacBcbfCfConfig = field(default_factory=SacBcbfCfConfig)


def _sample_std(values: list[float]) -> float:
    return stdev(values) if len(values) >= 2 else 0.0


def _optional_mean(values: list[float]) -> float:
    return mean(values) if values else math.nan


def run(task_id: str, cfg: EvaluateSacConfig) -> None:
    if cfg.episodes <= 0:
        raise ValueError("episodes must be positive")

    configure_torch_backends()
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(task_id, play=True)
    low_agent_cfg = load_rl_cfg(task_id)
    env_cfg.scene.num_envs = 1
    env_cfg.terminations = {}
    if "push_robot" in env_cfg.events:
        del env_cfg.events["push_robot"]

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    low_env = RslRlVecEnvWrapper(raw_env, clip_actions=low_agent_cfg.clip_actions)
    low_policy = load_frozen_low_level_policy(
        task_id, low_env, cfg.low_level_checkpoint, device
    )
    command_if = MjlabVelocityCommandInterface(low_env, command_name="twist")
    env = SacCbfNavigationEnv(
        low_env,
        low_policy,
        command_if,
        cfg.config.navigation,
        reward_cfg=cfg.config.reward,
    )

    scenario = SCENARIOS[cfg.scenario]
    env.set_fixed_scenario(scenario)

    obs = env.actor_tensor(env.get_observations())
    actor = SafeSacActor(
        obs_dim=obs.shape[-1],
        hidden_dims=cfg.config.sac.hidden_dims,
        arena=env.navigation_cfg.arena,
        action_cfg=env.navigation_cfg.action,
        safety_cfg=env.navigation_cfg.backup_safety,
        obstacle_centers=env.obstacle_centers[:1],
        obstacle_radii=env.obstacle_radii[:1],
        obstacle_mask=env.obstacle_mask[:1],
        log_std_min=cfg.config.sac.log_std_min,
        log_std_max=cfg.config.sac.log_std_max,
    ).to(device)

    checkpoint_path = Path(cfg.high_level_checkpoint).expanduser()
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    print(
        f"[EVAL] method={cfg.method_name} scenario={scenario.name} "
        f"episodes={cfg.episodes} "
        f"filter=integrated_bcbf_lse_cf"
    )
    print(
        f"[EVAL] start={scenario.start_xy} yaw={scenario.start_yaw:.3f} "
        f"goal={scenario.goal_xy}"
    )

    records: list[dict[str, object]] = []
    trajectory_records: list[dict[str, object]] = []

    path_length = 0.0
    episode_step = 0
    previous_endpoint = None

    with torch.no_grad():
        while len(records) < cfg.episodes:
            actor_obs = env.actor_tensor(obs)
            position, yaw = env.safety_context()
            position_before = position[0].detach().clone()

            (
                root_position,
                _,
                measured_velocity_body,
                measured_angular_velocity_body,
                roll,
                pitch,
                root_yaw,
            ) = env._root_state()

            out = actor.deterministic(actor_obs, position, yaw)

            nominal_command = out.nominal_command.detach().clone()
            safe_command = out.safe_command.detach().clone()
            safety = out.safety

            goal_position = env.goal_positions[0].detach().clone()
            goal_distance_before = torch.linalg.norm(
                goal_position - position[0]
            )

            next_obs, _, done, extras = env.step_safe(
                nominal_command, safe_command, safety
            )

            episode_step += 1

            if bool(done[0]):
                terminal_positions = extras.get("completed_terminal_positions")
                if terminal_positions is None or terminal_positions.numel() == 0:
                    raise RuntimeError(
                        "completed_terminal_positions is missing. Apply the "
                        "evaluation-metrics patch to sac_navigation_env.py."
                    )
                endpoint = terminal_positions[0]
            else:
                endpoint = env.safety_context()[0][0]

            if previous_endpoint is not None:
                path_length += float(
                    torch.linalg.norm(
                        endpoint - previous_endpoint
                    ).detach().cpu()
                )

            first_valid_sample = previous_endpoint is None

            previous_endpoint = endpoint.detach().clone()

            logged_position = (
                endpoint.detach().clone()
                if first_valid_sample
                else position_before
            )

            correction = safe_command[0] - nominal_command[0]

            cbf_active = bool(safety.active[0])
            cbf_correction_norm = float(
                safety.correction_norm[0].detach().cpu()
            )
            minimum_individual_h = float(
                safety.individual_h[0].min().detach().cpu()
            )
            h_lse = float(safety.h_lse[0].detach().cpu())
            input_bound_violation = float(
                safety.input_bound_violation[0].detach().cpu()
            )
            input_bound_active = int(
                bool(safety.input_bound_active[0])
            )

            row_outcome = ""
            row_success = 0
            row_collision = 0
            row_timeout = 0

            if bool(done[0]):
                row_success = int(bool(extras["completed_success"][0]))
                row_collision = int(bool(extras["completed_collision"][0]))
                row_timeout = int(bool(extras["completed_timeouts"][0]))

                if row_success and not row_collision:
                    row_outcome = "success"
                elif row_collision:
                    row_outcome = "collision"
                elif row_timeout:
                    row_outcome = "timeout"
                else:
                    row_outcome = "other_failure"

            trajectory_records.append(
                {
                    "method": cfg.method_name,
                    "scenario": scenario.name,
                    "episode": len(records) + 1,
                    "step": episode_step,
                    "time_s": episode_step
                    * env.navigation_cfg.high_level_dt,

                    # State at the beginning of this high-level step.
                    "x_m": float(logged_position[0].cpu()),
                    "y_m": float(logged_position[1].cpu()),
                    "z_m": float(root_position[0, 2].detach().cpu()),
                    "roll_rad": float(roll[0].detach().cpu()),
                    "pitch_rad": float(pitch[0].detach().cpu()),
                    "yaw_rad": float(root_yaw[0].detach().cpu()),

                    # Measured body-frame velocity.
                    "vx_body_mps": float(
                        measured_velocity_body[0, 0].detach().cpu()
                    ),
                    "vy_body_mps": float(
                        measured_velocity_body[0, 1].detach().cpu()
                    ),
                    "wz_body_radps": float(
                        measured_angular_velocity_body[0, 2]
                        .detach()
                        .cpu()
                    ),

                    # Goal information.
                    "goal_x_m": float(goal_position[0].detach().cpu()),
                    "goal_y_m": float(goal_position[1].detach().cpu()),
                    "goal_distance_m": float(
                        goal_distance_before.detach().cpu()
                    ),

                    # SAC nominal command.
                    "nominal_vx_mps": float(
                        nominal_command[0, 0].detach().cpu()
                    ),
                    "nominal_vy_mps": float(
                        nominal_command[0, 1].detach().cpu()
                    ),
                    "nominal_wz_radps": float(
                        nominal_command[0, 2].detach().cpu()
                    ),

                    # Applied command.
                    "safe_vx_mps": float(
                        safe_command[0, 0].detach().cpu()
                    ),
                    "safe_vy_mps": float(
                        safe_command[0, 1].detach().cpu()
                    ),
                    "safe_wz_radps": float(
                        safe_command[0, 2].detach().cpu()
                    ),

                    # Explicit command correction.
                    "correction_vx_mps": float(
                        correction[0].detach().cpu()
                    ),
                    "correction_vy_mps": float(
                        correction[1].detach().cpu()
                    ),
                    "correction_wz_radps": float(
                        correction[2].detach().cpu()
                    ),

                    # Safety diagnostics.
                    "cbf_active": int(cbf_active),
                    "cbf_correction_norm": cbf_correction_norm,
                    "minimum_individual_h": minimum_individual_h,
                    "h_lse": h_lse,
                    "input_bound_violation": input_bound_violation,
                    "input_bound_active": input_bound_active,

                    # Position after this command interval.
                    "post_x_m": float(endpoint[0].detach().cpu()),
                    "post_y_m": float(endpoint[1].detach().cpu()),
                    "cumulative_path_length_m": path_length,

                    # Non-empty only for a terminal row.
                    "done": int(bool(done[0])),
                    "outcome": row_outcome,
                    "success": row_success,
                    "collision": row_collision,
                    "timeout": row_timeout,
                }
            )

            if bool(done[0]):
                success = bool(extras["completed_success"][0])
                collision = bool(extras["completed_collision"][0])
                timeout = bool(extras["completed_timeouts"][0])
                steps = int(extras["completed_lengths"][0])
                duration_s = steps * env.navigation_cfg.high_level_dt

                if success and not collision:
                    outcome = "success"
                elif collision:
                    outcome = "collision"
                elif timeout:
                    outcome = "timeout"
                else:
                    outcome = "other_failure"

                record = {
                    "method": cfg.method_name,
                    "scenario": scenario.name,
                    "episode": len(records) + 1,
                    "outcome": outcome,
                    "success": int(success),
                    "collision": int(collision),
                    "timeout": int(timeout),
                    "path_length_m": path_length,
                    "goal_time_s": duration_s if success else "",
                    "episode_duration_s": duration_s,
                    "episode_steps": steps,
                }
                records.append(record)

                if (
                    cfg.print_every_episode > 0
                    and len(records) % cfg.print_every_episode == 0
                ):
                    print(
                        f"[EVAL] episode={len(records):4d}/{cfg.episodes} "
                        f"outcome={outcome:13s} "
                        f"path={path_length:7.3f} m "
                        f"time={duration_s:6.2f} s"
                    )

                path_length = 0.0
                episode_step = 0
                previous_endpoint = None

            obs = next_obs

    output_dir = Path(cfg.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{stamp}_{cfg.method_name}_{scenario.name}_{cfg.episodes}ep"
    episodes_csv = output_dir / f"{stem}_episodes.csv"
    trajectory_csv = output_dir / f"{stem}_trajectory.csv"
    summary_csv = output_dir / f"{stem}_summary.csv"

    episode_fields = [
        "method",
        "scenario",
        "episode",
        "outcome",
        "success",
        "collision",
        "timeout",
        "path_length_m",
        "goal_time_s",
        "episode_duration_s",
        "episode_steps",
    ]
    with episodes_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=episode_fields)
        writer.writeheader()
        writer.writerows(records)

    trajectory_fields = [
        "method",
        "scenario",
        "episode",
        "step",
        "time_s",
        "x_m",
        "y_m",
        "z_m",
        "roll_rad",
        "pitch_rad",
        "yaw_rad",
        "vx_body_mps",
        "vy_body_mps",
        "wz_body_radps",
        "goal_x_m",
        "goal_y_m",
        "goal_distance_m",
        "nominal_vx_mps",
        "nominal_vy_mps",
        "nominal_wz_radps",
        "safe_vx_mps",
        "safe_vy_mps",
        "safe_wz_radps",
        "correction_vx_mps",
        "correction_vy_mps",
        "correction_wz_radps",
        "cbf_active",
        "cbf_correction_norm",
        "minimum_individual_h",
        "h_lse",
        "input_bound_violation",
        "input_bound_active",
        "post_x_m",
        "post_y_m",
        "cumulative_path_length_m",
        "done",
        "outcome",
        "success",
        "collision",
        "timeout",
    ]

    with trajectory_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=trajectory_fields)
        writer.writeheader()
        writer.writerows(trajectory_records)

    successes = [r for r in records if int(r["success"]) == 1]
    success_paths = [float(r["path_length_m"]) for r in successes]
    success_times = [float(r["goal_time_s"]) for r in successes]
    all_paths = [float(r["path_length_m"]) for r in records]

    n = len(records)
    summary = {
        "method": cfg.method_name,
        "scenario": scenario.name,
        "episodes": n,
        "successes": sum(int(r["success"]) for r in records),
        "collisions": sum(int(r["collision"]) for r in records),
        "timeouts": sum(int(r["timeout"]) for r in records),
        "success_rate": sum(int(r["success"]) for r in records) / n,
        "collision_rate": sum(int(r["collision"]) for r in records) / n,
        "timeout_rate": sum(int(r["timeout"]) for r in records) / n,
        "mean_success_path_length_m": _optional_mean(success_paths),
        "std_success_path_length_m": _sample_std(success_paths),
        "mean_goal_time_s": _optional_mean(success_times),
        "std_goal_time_s": _sample_std(success_times),
        "mean_all_episode_path_length_m": mean(all_paths),
        "checkpoint": str(checkpoint_path),
        "safety_filter_enabled": 1,
        "seed": cfg.seed,
    }

    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print("\n[EVAL SUMMARY]")
    print(f"  success rate: {summary['success_rate']:.3f}")
    print(f"  collision rate: {summary['collision_rate']:.3f}")
    print(f"  timeout rate: {summary['timeout_rate']:.3f}")
    if success_paths:
        print(
            "  successful path length: "
            f"{summary['mean_success_path_length_m']:.3f} ± "
            f"{summary['std_success_path_length_m']:.3f} m"
        )
        print(
            "  successful goal time: "
            f"{summary['mean_goal_time_s']:.3f} ± "
            f"{summary['std_goal_time_s']:.3f} s"
        )
    else:
        print("  successful path length: N/A (no successful episodes)")
        print("  successful goal time: N/A (no successful episodes)")
    print(f"  episode CSV: {episodes_csv}")
    print(f"  trajectory CSV: {trajectory_csv}")
    print(f"  summary CSV: {summary_csv}")

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
        EvaluateSacConfig,
        args=remaining,
        prog=sys.argv[0] + f" {task}",
        config=mjlab.TYRO_FLAGS,
    )
    run(task, cfg)


if __name__ == "__main__":
    main()
