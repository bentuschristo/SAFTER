"""Log one fixed-scenario CBF-RL trajectory, in a schema compatible with
scripts/plot_seaspeed_combined_trajectories.py's trajectory loader
(load_laser_trajectory reads columns "case_id", "post_x_m", "post_y_m",
"outcome" from a CSV, filtered by case_id).

Use this to add a CBF-RL line to that figure for the exact same (start,
goal) case already selected for the SAC methods there -- run
`python scripts/plot_seaspeed_combined_trajectories.py` first, it prints
the selected start/goal per map, e.g.:

  slalom: SAC seed=1001 case_id=16 start=(-2.14, 5.03) goal=(-3.19, -5.67)

then pull start_yaw from that case's own
`<sac_run_root>/seed_<seed>/<map>/<method>/<method>_episodes.csv` (column
start_yaw) and pass all five values here.

Example:
  python scripts/log_cbf_rl_trajectory.py Unitree-Go2-Safe-Navigation \\
      --checkpoint logs/rsl_rl/go2_cbf_rl/2026-09-14_16-51-50_dual_handcrafted2_seed0/model_1666.pt \\
      --low-level-checkpoint logs/rsl_rl/go2_velocity/2026-06-19_13-24-10/model_10000.pt \\
      --map-name slalom \\
      --start-x -2.1429476737976074 --start-y 5.030910015106201 --start-yaw -1.722939590507896 \\
      --goal-x -3.194646120071411 --goal-y -5.665926933288574 \\
      --output-root logs/evaluation/cbf_rl_seaspeed_trajectory \\
      --device cuda:0
"""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

import mjlab
from mjlab.tasks.registry import list_tasks

from evaluate_high_level_cbf_rl_unseen_maps import EvalCbfRlConfig, FixedScenario, build_map_env, load_policy
from src.high_level_cbf_rl.config import CbfRlConfig


@dataclass
class LogTrajectoryConfig:
    checkpoint: str
    low_level_checkpoint: str
    map_name: str
    start_x: float
    start_y: float
    start_yaw: float
    goal_x: float
    goal_y: float
    device: str | None = None
    use_cbf_action_filtering: bool = True
    episode_duration_s: float = 60.0
    output_root: str = "logs/evaluation/cbf_rl_seaspeed_trajectory"
    method_name: str = "cbf_rl"


def run(task_id: str, cfg: LogTrajectoryConfig) -> None:
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    eval_cfg = EvalCbfRlConfig(
        checkpoint=cfg.checkpoint,
        low_level_checkpoint=cfg.low_level_checkpoint,
        num_envs=1,
        episode_duration_s=cfg.episode_duration_s,
        use_cbf_action_filtering=cfg.use_cbf_action_filtering,
        config=CbfRlConfig(),
    )
    env = build_map_env(task_id, cfg.map_name, eval_cfg, device)
    policy = load_policy(env, task_id, cfg.checkpoint, device)

    scenario = FixedScenario(
        start_xy=(cfg.start_x, cfg.start_y), goal_xy=(cfg.goal_x, cfg.goal_y), start_yaw=cfg.start_yaw
    )
    env.set_fixed_scenarios([scenario])
    obs = env.get_observations()

    rows: list[dict[str, object]] = []
    step = 0
    outcome = ""
    with torch.no_grad():
        while True:
            action = policy(obs)
            obs, reward, done, extras = env.step(action)
            step += 1
            position = extras["position"][0]
            is_done = bool(done[0])
            if is_done:
                success = bool(extras["completed_success"][0])
                collision = bool(extras["completed_collision"][0])
                timeout = bool(extras["completed_timeouts"][0])
                fallen = bool(extras["completed_fallen"][0])
                outcome = (
                    "success" if success else
                    "collision" if collision else
                    "timeout" if timeout else
                    "other_failure" if fallen else "unknown"
                )
            rows.append({
                "case_id": 0,
                "step": step,
                "post_x_m": float(position[0]),
                "post_y_m": float(position[1]),
                "outcome": outcome,
            })
            if is_done:
                break

    out_dir = Path(cfg.output_root) / cfg.map_name / cfg.method_name
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / f"{cfg.method_name}_trajectory.csv"
    with traj_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "step", "post_x_m", "post_y_m", "outcome"])
        writer.writeheader()
        writer.writerows(rows)

    episodes_path = out_dir / f"{cfg.method_name}_episodes.csv"
    with episodes_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["case_id", "start_x", "start_y", "start_yaw", "goal_x", "goal_y", "outcome", "steps"]
        )
        writer.writeheader()
        writer.writerow({
            "case_id": 0,
            "start_x": cfg.start_x, "start_y": cfg.start_y, "start_yaw": cfg.start_yaw,
            "goal_x": cfg.goal_x, "goal_y": cfg.goal_y,
            "outcome": outcome, "steps": step,
        })

    print(f"[TRAJ] map={cfg.map_name} outcome={outcome} steps={step}")
    print(f"[TRAJ] Wrote {traj_path}")
    print(f"[TRAJ] Wrote {episodes_path}")
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
    cfg = tyro.cli(LogTrajectoryConfig, args=remaining, prog=sys.argv[0] + f" {task}", config=mjlab.TYRO_FLAGS)
    run(task, cfg)


if __name__ == "__main__":
    main()
