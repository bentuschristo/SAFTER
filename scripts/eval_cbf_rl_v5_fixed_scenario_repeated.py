"""Run CBF-RL (Dual) N times on the corridor_derived_highshift_v5 start/goal
shared with the SAC-laser fixed-scenario repeats
(scripts/generate_corridor_derived_highshift_v5_fixed_scenario_manifest.py)
and the SEA-Nav fixed-scenario bank
(scripts/generate_seanav_v5_fixed_scenario_bank.py) -- part of the 4-method
(CBF-RL / SEA-Nav / bCBF-CF / CBF-CF) 8-sample trajectory comparison figure
requested 2026-09-24, where all 4 methods must share the identical initial
condition and goal.

Adapted from eval_cbf_rl_diagonal_fixed_scenario_repeated.py. The policy
samples stochastically at inference, so repeating the identical scenario
num_envs times naturally gives diverse outcomes without needing a per-repeat
reseed; --perturb-radius-m is available as a fallback if the exact repeated
point doesn't produce enough outcome diversity on its own.

Usage:
    python -u scripts/eval_cbf_rl_v5_fixed_scenario_repeated.py \\
        Unitree-Go2-Safe-Navigation \\
        --checkpoint logs/rsl_rl/go2_cbf_rl/2026-09-18_16-12-15_dual_handcrafted2_seed0_4096envs_1000iters_circlegeom/model_999.pt \\
        --low-level-checkpoint logs/rsl_rl/go2_velocity/2026-06-19_13-24-10/model_10000.pt \\
        --num-repeats 30
"""
from __future__ import annotations

import csv
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

import mjlab
from mjlab.tasks.registry import list_tasks

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_high_level_cbf_rl_unseen_maps import (  # noqa: E402
    EvalCbfRlConfig, FixedScenario, STRUCTURED_SCENARIO_REGIONS, build_map_env, load_policy,
)

# Shared with the SAC-laser and SEA-Nav fixed-scenario repeats for this same
# request (see generate_corridor_derived_highshift_v5_fixed_scenario_manifest.py).
START_XY = (0.0, 6.2)
GOAL_XY = (0.0, -6.2)
START_YAW = math.atan2(GOAL_XY[1] - START_XY[1], GOAL_XY[0] - START_XY[0])
MAP_NAME = "corridor_derived_highshift_v5"


@dataclass
class Args:
    checkpoint: str
    low_level_checkpoint: str
    num_repeats: int = 30
    perturb_radius_m: float = 0.0
    seed: int = 12345
    device: str | None = None
    output_dir: str = "logs/evaluation/corridor_derived_highshift_v5_fixed_scenario_repeated/cbf_rl"


def jittered_start_positions(n: int, radius_m: float, seed: int) -> list[tuple[float, float]]:
    if radius_m <= 0.0:
        return [START_XY] * n
    x_min, x_max, y_min, y_max = STRUCTURED_SCENARIO_REGIONS[MAP_NAME]["start"]
    rng = random.Random(seed)
    positions = []
    for _ in range(n):
        r = radius_m * rng.random() ** 0.5  # uniform over the disk area, not just the radius
        theta = rng.uniform(0.0, 2.0 * math.pi)
        x = min(max(START_XY[0] + r * math.cos(theta), x_min), x_max)
        y = min(max(START_XY[1] + r * math.sin(theta), y_min), y_max)
        positions.append((x, y))
    return positions


def run(task_id: str, args: Args) -> None:
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = EvalCbfRlConfig(
        checkpoint=args.checkpoint,
        low_level_checkpoint=args.low_level_checkpoint,
        maps=MAP_NAME,
        num_envs=args.num_repeats,
        cases_per_seed=args.num_repeats,
        device=device,
    )
    env = build_map_env(task_id, MAP_NAME, cfg, device)
    policy = load_policy(env, task_id, cfg.checkpoint, device)

    n = args.num_repeats
    start_positions = jittered_start_positions(n, args.perturb_radius_m, args.seed)
    scenarios = [
        FixedScenario(start_xy=start_positions[i], goal_xy=GOAL_XY, start_yaw=START_YAW) for i in range(n)
    ]
    env.set_fixed_scenarios(scenarios)
    obs = env.get_observations()

    finished = [False] * n
    steps = [0] * n
    trajectory_rows: list[dict[str, object]] = []
    episode_rows: list[dict[str, object]] = [
        {"case_id": i, "start_x": start_positions[i][0], "start_y": start_positions[i][1], "start_yaw": START_YAW,
         "goal_x": GOAL_XY[0], "goal_y": GOAL_XY[1], "outcome": None, "steps": None}
        for i in range(n)
    ]

    with torch.no_grad():
        while not all(finished):
            action = policy(obs)
            obs, reward, done, extras = env.step(action)
            position = extras["position"]
            completed_ids = extras["completed_env_ids"]
            completed_lookup = {int(env_id): True for env_id in completed_ids.tolist()}
            for env_id in range(n):
                if finished[env_id]:
                    continue
                steps[env_id] += 1
                success = bool(extras["completed_success"][env_id])
                collision = bool(extras["completed_collision"][env_id])
                timeout = bool(extras["completed_timeouts"][env_id])
                fallen = bool(extras["completed_fallen"][env_id])
                is_done = env_id in completed_lookup
                outcome = (
                    "success" if success else
                    "collision" if collision else
                    "timeout" if timeout else
                    "fall" if fallen else ("" if not is_done else "other")
                )
                trajectory_rows.append({
                    "case_id": env_id, "step": steps[env_id],
                    "post_x_m": float(position[env_id, 0]), "post_y_m": float(position[env_id, 1]),
                    "outcome": outcome,
                })
                if is_done:
                    finished[env_id] = True
                    episode_rows[env_id]["outcome"] = outcome
                    episode_rows[env_id]["steps"] = steps[env_id]
            if any(steps[i] % 200 == 0 for i in range(n) if not finished[i]):
                print(f"  ...still running: {sum(finished)}/{n} finished, max steps so far {max(steps)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "cbf_rl_trajectory.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trajectory_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trajectory_rows)
    with (out_dir / "cbf_rl_episodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(episode_rows[0].keys()))
        writer.writeheader()
        writer.writerows(episode_rows)

    outcome_counts: dict[str, int] = {}
    for row in episode_rows:
        outcome_counts[row["outcome"]] = outcome_counts.get(row["outcome"], 0) + 1
        print(f"  case_id={row['case_id']}: outcome={row['outcome']} steps={row['steps']}")
    print(f"\n[EVAL] outcome counts: {outcome_counts}")
    print(f"[EVAL] Wrote results to {out_dir}")


def main() -> None:
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401

    task, remaining = tyro.cli(
        tyro.extras.literal_type_from_choices(list_tasks()),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    args = tyro.cli(Args, args=remaining, prog=sys.argv[0] + f" {task}", config=mjlab.TYRO_FLAGS)
    run(task, args)


if __name__ == "__main__":
    main()
