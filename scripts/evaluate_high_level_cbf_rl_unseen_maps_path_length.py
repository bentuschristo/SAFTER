"""Rerun the CBF-RL unseen-maps structured evaluation (same "500 cases = 20/
seed x 25 seeds" convention, same three maps: slalom, corridor,
circle_and_square) but ALSO accumulate true traveled path length per
episode, using extras["position"] (absolute (x, y) after each high-level
step, from CbfRlNavigationEnv.step()) -- the existing
evaluate_high_level_cbf_rl_unseen_maps.py only ever saved start/min/final
straight-line distance-to-goal, never per-step position, so path length
couldn't be reconstructed after the fact.

Filter mode and all other parameters match the existing filter-ON run at
logs/evaluation/cbf_rl_unseen_maps/2026-09-18_16-44-15_dual_handcrafted2_seed0_4096envs_1000iters_circlegeom
exactly, so results should reproduce that run's success/collision/timeout
rates (same checkpoint, same seeds, same manifest algorithm) while adding
path_length_m per episode.

Example:
  python scripts/evaluate_high_level_cbf_rl_unseen_maps_path_length.py \\
      Unitree-Go2-Safe-Navigation \\
      --checkpoint logs/rsl_rl/go2_cbf_rl/2026-09-17_11-54-21_dual_handcrafted2_seed0_4096envs_1000iters/model_999.pt \\
      --low-level-checkpoint logs/rsl_rl/go2_velocity/2026-06-19_13-24-10/model_10000.pt \\
      --maps slalom,corridor,circle_and_square \\
      --cases-per-seed 20 --num-seeds 25 --seed 1000 \\
      --num-envs 50 --device cuda:0 --use-cbf-action-filtering True
"""
from __future__ import annotations

import csv
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import torch
import tyro

import mjlab
from mjlab.tasks.registry import list_tasks

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_high_level_cbf_rl_unseen_maps import (  # noqa: E402
    EvalCbfRlConfig,
    FixedScenario,
    UNSEEN_MAPS,
    ArenaConfig,
    build_map_env,
    generate_manifest,
    load_policy,
)


def evaluate_map_with_path_length(
    *, env, policy, manifest: list[dict[str, object]], num_envs: int, print_every: int,
    stochastic_policy: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Returns (episode_records, trajectory_rows). trajectory_rows is the
    full per-step (x, y) path for every case, one row per (case, step), in
    the same schema scripts/log_cbf_rl_trajectory.py uses for its single
    example episode ("case_id", "step", "post_x_m", "post_y_m", "outcome"),
    plus "map_name" -- outcome is "" until the episode's terminal step."""
    records: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []
    if not manifest:
        return records, trajectory_rows
    batch_size = min(num_envs, env.num_envs, len(manifest))
    for batch_start in range(0, len(manifest), batch_size):
        real_cases = manifest[batch_start : batch_start + batch_size]
        padded_cases = list(real_cases)
        while len(padded_cases) < env.num_envs:
            padded_cases.append(real_cases[-1])
        torch.manual_seed(int(real_cases[0]["reset_seed"]))
        scenarios = [
            FixedScenario(start_xy=(case["start_x"], case["start_y"]), goal_xy=(case["goal_x"], case["goal_y"]),
                          start_yaw=case["start_yaw"])
            for case in padded_cases
        ]
        env.set_fixed_scenarios(scenarios)
        obs = env.get_observations()
        active_count = len(real_cases)
        finished = [False] * active_count
        returns = [0.0] * active_count
        steps = [0] * active_count
        path_length = [0.0] * active_count
        prev_position = torch.tensor(
            [[case["start_x"], case["start_y"]] for case in padded_cases], dtype=torch.float32, device=env.device,
        )
        normalize_by = float(env.navigation_cfg.normalize_positions_by)

        def distance_to_goal(obs_tensordict) -> torch.Tensor:
            goal_b = env.actor_tensor(obs_tensordict)[:, :2]
            return torch.linalg.norm(goal_b, dim=-1) * normalize_by

        start_distance = distance_to_goal(obs)
        min_distance = start_distance.clone()
        with torch.no_grad():
            while not all(finished):
                action = policy(obs, stochastic_output=True) if stochastic_policy else policy(obs)
                obs, reward, done, extras = env.step(action)
                current_distance = extras["distance_to_goal"]
                min_distance = torch.minimum(min_distance, current_distance)
                current_position = extras["position"]
                step_displacement = torch.linalg.norm(current_position - prev_position, dim=-1)
                completed_ids = extras["completed_env_ids"]
                completed_lookup = {int(env_id): row for row, env_id in enumerate(completed_ids.tolist())}
                for env_id in range(active_count):
                    if finished[env_id]:
                        continue
                    returns[env_id] += float(reward[env_id])
                    steps[env_id] += 1
                    path_length[env_id] += float(step_displacement[env_id])
                    case = real_cases[env_id]
                    row = completed_lookup.get(env_id)
                    if row is None:
                        trajectory_rows.append({
                            "map_name": case["map_name"],
                            "case_id": case["case_id"],
                            "step": steps[env_id],
                            "post_x_m": float(current_position[env_id, 0]),
                            "post_y_m": float(current_position[env_id, 1]),
                            "outcome": "",
                        })
                        continue
                    success = bool(extras["completed_success"][env_id])
                    collision = bool(extras["completed_collision"][env_id])
                    timeout = bool(extras["completed_timeouts"][env_id])
                    fallen = bool(extras["completed_fallen"][env_id])
                    outcome = (
                        "success" if success else
                        "collision" if collision else
                        "timeout" if timeout else
                        "fall" if fallen else "other"
                    )
                    trajectory_rows.append({
                        "map_name": case["map_name"],
                        "case_id": case["case_id"],
                        "step": steps[env_id],
                        "post_x_m": float(current_position[env_id, 0]),
                        "post_y_m": float(current_position[env_id, 1]),
                        "outcome": outcome,
                    })
                    records.append({
                        "map_name": case["map_name"],
                        "case_id": case["case_id"],
                        "reset_seed": case["reset_seed"],
                        "success": int(success),
                        "collision": int(collision),
                        "timeout": int(timeout),
                        "fallen": int(fallen),
                        "outcome": outcome,
                        "return": returns[env_id],
                        "episode_steps": steps[env_id],
                        "start_distance_to_goal_m": float(start_distance[env_id]),
                        "min_distance_to_goal_m": float(min_distance[env_id]),
                        "final_distance_to_goal_m": float(current_distance[env_id]),
                        "path_length_m": path_length[env_id],
                    })
                    finished[env_id] = True
                prev_position = current_position
        if print_every > 0 and (batch_start // batch_size) % max(print_every, 1) == 0:
            print(f"  ...batch {batch_start}-{batch_start + len(real_cases)}/{len(manifest)} done")
    return records, trajectory_rows


def summarize(records: list[dict[str, object]], map_name: str) -> dict[str, object]:
    n = len(records)
    succ = [r for r in records if r["success"]]
    return {
        "map_name": map_name,
        "cases": n,
        "success_rate": sum(r["success"] for r in records) / n if n else math.nan,
        "collision_rate": sum(r["collision"] for r in records) / n if n else math.nan,
        "timeout_rate": sum(r["timeout"] for r in records) / n if n else math.nan,
        "fall_rate": sum(r["fallen"] for r in records) / n if n else math.nan,
        "mean_return": sum(r["return"] for r in records) / n if n else math.nan,
        "n_success": len(succ),
        "mean_path_length_m_success": sum(r["path_length_m"] for r in succ) / len(succ) if succ else math.nan,
        "mean_time_to_goal_s_success": sum(r["episode_steps"] for r in succ) * 0.02 / len(succ) if succ else math.nan,
    }


def run(task_id: str, cfg: EvalCbfRlConfig) -> None:
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    evaluation_seeds = [cfg.seed + i * cfg.seed_stride for i in range(cfg.num_seeds)]
    requested_maps = [name.strip() for name in cfg.maps.split(",") if name.strip()]
    unknown = [name for name in requested_maps if name not in UNSEEN_MAPS]
    if unknown:
        raise ValueError(f"Unknown map(s) {unknown}; choices are {list(UNSEEN_MAPS)}")

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = f"_{cfg.run_name}" if cfg.run_name else ""
    out_dir = Path(cfg.output_dir) / f"{stamp}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    for map_index, map_name in enumerate(requested_maps):
        print(f"\n[EVAL] Building env for map={map_name!r}")
        env = build_map_env(task_id, map_name, cfg, device)
        policy = load_policy(env, task_id, cfg.checkpoint, device)

        # Pool every seed's manifest before evaluating, instead of calling
        # evaluate_map_with_path_length once per seed -- each call's batching
        # waits for its single slowest straggler (up to the full episode
        # timeout), so num_seeds separate small-batch calls pay that fixed
        # cost num_seeds times instead of once.
        pooled_manifest: list[dict[str, object]] = []
        arena = ArenaConfig(obstacles=UNSEEN_MAPS[map_name], max_obstacles=len(UNSEEN_MAPS[map_name]))
        for evaluation_seed in evaluation_seeds:
            manifest_seed = evaluation_seed + 10000 * map_index
            manifest = generate_manifest(
                env=env, arena=arena, map_name=map_name, count=cfg.cases_per_seed, seed=manifest_seed,
            )
            for case in manifest:
                case["case_id"] = len(pooled_manifest)
            pooled_manifest.extend(manifest)

        all_records, all_trajectory_rows = evaluate_map_with_path_length(
            env=env, policy=policy, manifest=pooled_manifest, num_envs=cfg.num_envs, print_every=5,
        )
        print(
            f"  {len(evaluation_seeds)} seeds pooled: {len(all_records)} cases, "
            f"success={sum(r['success'] for r in all_records)}/{len(all_records)}"
        )

        env.close()

        with (out_dir / f"{map_name}_episodes.csv").open("w", newline="") as file:
            if all_records:
                writer = csv.DictWriter(file, fieldnames=list(all_records[0].keys()))
                writer.writeheader()
                writer.writerows(all_records)

        with (out_dir / f"{map_name}_trajectories.csv").open("w", newline="") as file:
            if all_trajectory_rows:
                writer = csv.DictWriter(file, fieldnames=list(all_trajectory_rows[0].keys()))
                writer.writeheader()
                writer.writerows(all_trajectory_rows)

        summary = summarize(all_records, map_name)
        summary_rows.append(summary)
        print(f"[EVAL] {map_name}: {summary}")

    with (out_dir / "summary.csv").open("w", newline="") as file:
        if summary_rows:
            writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

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
    cfg = tyro.cli(EvalCbfRlConfig, args=remaining, prog=sys.argv[0] + f" {task}", config=mjlab.TYRO_FLAGS)
    cfg.output_dir = "logs/evaluation/cbf_rl_unseen_maps_path_length"
    run(task, cfg)


if __name__ == "__main__":
    main()
