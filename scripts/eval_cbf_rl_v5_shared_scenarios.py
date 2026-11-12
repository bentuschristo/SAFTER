"""Run CBF-RL (Dual) once per scenario on the 20 SHARED (start, goal, yaw)
pairs already used for bcbf_cf/cbf_cd_cf (evaluation_seed=1000, case_id
0-19, from logs/evaluation/laser_unseen_maps_structured_refixed/
20260924_121722_corridor_derived_highshift_v5_seeds1000-1024_20perseed_500cases_90s/
seed_1000/corridor_derived_highshift_v5/scenarios.csv) -- part of the
4-method (CBF-RL / SEA-Nav / bCBF-CF / CBF-CF) 8-sample trajectory
comparison figure requested 2026-09-24, where every method must be
evaluated on the IDENTICAL initial-condition/goal pairing per sample (not a
single scenario repeated).

Unlike eval_cbf_rl_v5_fixed_scenario_repeated.py (one point x N repeats),
here each of the N envs gets its OWN distinct scenario, env_id i <->
case_id i.

Usage:
    python -u scripts/eval_cbf_rl_v5_shared_scenarios.py \\
        Unitree-Go2-Safe-Navigation \\
        --checkpoint logs/rsl_rl/go2_cbf_rl/2026-09-18_16-12-15_dual_handcrafted2_seed0_4096envs_1000iters_circlegeom/model_999.pt \\
        --low-level-checkpoint logs/rsl_rl/go2_velocity/2026-06-19_13-24-10/model_10000.pt
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

MAP_NAME = "corridor_derived_highshift_v5"
SHARED_MANIFEST = Path(
    "logs/evaluation/laser_unseen_maps_structured_refixed/"
    "20260924_121722_corridor_derived_highshift_v5_seeds1000-1024_20perseed_500cases_90s/"
    "seed_1000/corridor_derived_highshift_v5/scenarios.csv"
)
SHARED_EVALUATION_SEED = "1000"
NUM_SHARED_CASES = 20


def load_shared_scenarios() -> list[dict[str, float]]:
    with SHARED_MANIFEST.open() as f:
        rows = [r for r in csv.DictReader(f)]
    rows = [r for r in rows if int(r["case_id"]) < NUM_SHARED_CASES]
    rows.sort(key=lambda r: int(r["case_id"]))
    assert len(rows) == NUM_SHARED_CASES, f"expected {NUM_SHARED_CASES} rows, got {len(rows)}"
    return [
        {"case_id": int(r["case_id"]), "start_x": float(r["start_x"]), "start_y": float(r["start_y"]),
         "start_yaw": float(r["start_yaw"]), "goal_x": float(r["goal_x"]), "goal_y": float(r["goal_y"])}
        for r in rows
    ]


@dataclass
class Args:
    checkpoint: str
    low_level_checkpoint: str
    device: str | None = None
    output_dir: str = "logs/evaluation/corridor_derived_highshift_v5_shared_scenarios/cbf_rl"


def run(task_id: str, args: Args) -> None:
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    scenarios = load_shared_scenarios()
    n = len(scenarios)
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

    fixed_scenarios = [
        FixedScenario(
            start_xy=(s["start_x"], s["start_y"]),
            goal_xy=(s["goal_x"], s["goal_y"]),
            start_yaw=s["start_yaw"],
        )
        for s in scenarios
    ]
    env.set_fixed_scenarios(fixed_scenarios)
    obs = env.get_observations()

    finished = [False] * n
    steps = [0] * n
    trajectory_rows: list[dict[str, object]] = []
    episode_rows: list[dict[str, object]] = [
        {"case_id": scenarios[i]["case_id"], "start_x": scenarios[i]["start_x"], "start_y": scenarios[i]["start_y"],
         "start_yaw": scenarios[i]["start_yaw"], "goal_x": scenarios[i]["goal_x"], "goal_y": scenarios[i]["goal_y"],
         "outcome": None, "steps": None}
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
                    "case_id": scenarios[env_id]["case_id"], "step": steps[env_id],
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
