"""Repeat CBF-RL N times each on corridor's shared case_id=10 and case_id=15
(start/goal/yaw pulled directly from logs/evaluation/laser_unseen_maps_structured_refixed/
corridor_shared_scenarios_medium_shift/seed_1000/corridor/scenarios.csv) --
their original single draws both failed via "fall" INSIDE the start region,
which looks wrong in the trajectory-comparison figure. The policy samples
stochastically at inference, so repeating each exact scenario should surface
an alternate draw that fails further along (ideally a genuine collision, not
another near-instant fall) -- mirrors eval_cbf_rl_v5_fixed_scenario_repeated.py.

Usage:
    python -u scripts/eval_cbf_rl_corridor_two_case_repeated.py \\
        Unitree-Go2-Safe-Navigation \\
        --checkpoint logs/rsl_rl/go2_cbf_rl/2026-09-18_16-12-15_dual_handcrafted2_seed0_4096envs_1000iters_circlegeom/model_999.pt \\
        --low-level-checkpoint logs/rsl_rl/go2_velocity/2026-06-19_13-24-10/model_10000.pt \\
        --num-repeats 15
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_high_level_cbf_rl_unseen_maps import (  # noqa: E402
    EvalCbfRlConfig, FixedScenario, build_map_env, load_policy,
)

MAP_NAME = "corridor"
SCENARIOS = {
    10: {"start_xy": (0.9022427201271057, 4.464519023895264), "start_yaw": -1.9572877769106176,
         "goal_xy": (-3.7856009006500244, -4.561171054840088)},
    15: {"start_xy": (1.7332384586334229, 4.699795246124268), "start_yaw": -2.033726984386224,
         "goal_xy": (-3.412853240966797, -5.367307186126709)},
}


@dataclass
class Args:
    checkpoint: str
    low_level_checkpoint: str
    num_repeats: int = 15
    device: str | None = None
    output_dir: str = "logs/evaluation/corridor_shared_scenarios_medium_shift/cbf_rl_two_case_repeated"


def run(task_id: str, args: Args) -> None:
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    n_per_case = args.num_repeats
    case_ids = sorted(SCENARIOS)
    n = n_per_case * len(case_ids)
    cfg = EvalCbfRlConfig(
        checkpoint=args.checkpoint,
        low_level_checkpoint=args.low_level_checkpoint,
        maps=MAP_NAME,
        num_envs=n,
        cases_per_seed=n,
        device=device,
    )
    env = build_map_env(task_id, MAP_NAME, cfg, device)
    policy = load_policy(env, task_id, cfg.checkpoint, device)

    env_case_ids = []
    scenarios = []
    for case_id in case_ids:
        s = SCENARIOS[case_id]
        for _ in range(n_per_case):
            env_case_ids.append(case_id)
            scenarios.append(FixedScenario(start_xy=s["start_xy"], goal_xy=s["goal_xy"], start_yaw=s["start_yaw"]))
    env.set_fixed_scenarios(scenarios)
    obs = env.get_observations()

    finished = [False] * n
    steps = [0] * n
    trajectory_rows: list[dict[str, object]] = []
    episode_rows: list[dict[str, object]] = [
        {"case_id": env_case_ids[i], "repeat": i % n_per_case, "outcome": None, "steps": None}
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
                    "case_id": env_case_ids[env_id], "repeat": env_id % n_per_case, "step": steps[env_id],
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

    for row in episode_rows:
        print(f"  case_id={row['case_id']} repeat={row['repeat']}: outcome={row['outcome']} steps={row['steps']}")
    print(f"\n[EVAL] Wrote results to {out_dir}")


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
