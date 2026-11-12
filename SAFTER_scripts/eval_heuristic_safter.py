#!/usr/bin/env python3
"""Evaluate a pure go-to-goal (obstacle-blind) heuristic nominal controller
through SAFTER (LaserZeroBackupLseCbfClosedForm, the closed-form bCBF-CF
filter WITH control dynamics) on the project's structured low/medium/
high-shift evaluation maps.

Sibling of eval_heuristic_hardbox_nocd.py -- same heuristic controller, same
scenario/environment infrastructure, same CSV schema -- with only the
safety filter swapped: SAFTER models body velocity as a state with
first-order command dynamics (`control_dynamics_a`) and a genuine backup
rollout (the robot coasts to a stop under that dynamics model, not an
instantaneous-velocity kinematic flow), so (unlike the no-CD filter) this
script must supply real current_command/previous_nominal_command/
nominal_history_valid each step -- sourced directly from the env's own
tracked state (env.previous_safe_command/previous_nominal_command/
nominal_history_valid), exactly as LaserSafeSacActor.apply_safety() does
for the trained SAC methods.

Filter hyperparameters are identical to scripts/benchmark_safety_filter_latency.py's
SAFTER construction (SHARED_KWARGS/BACKUP_KWARGS, legacy_projection=True,
straight_through=False) -- the same SAFTER this project trains/evaluates
with the SAC policy, just driven by the heuristic controller instead.

Output layout and CSV schema match the reference script's on-disk
convention exactly (see eval_heuristic_hardbox_nocd.py's own docstring for
the full description; identical here except method name):
    {output_root}/seed_{seed}/{map_name}/heuristic_safter/
        heuristic_safter_summary.csv
        heuristic_safter_episodes.csv
        heuristic_safter_trajectory.csv   (only with --log-trajectories)
"""
from __future__ import annotations

import argparse
import csv
import gc
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

# Allow running as: python SAFTER_scripts/eval_heuristic_safter.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# warp.init() before anything else: see eval_heuristic_hardbox_nocd.py's own
# comment here for the full explanation -- mujoco_warp's module-level code
# calls wp.array(...) at import time, which can segfault if Warp's lazy
# auto-init hasn't already run.
import warp as wp  # noqa: E402

wp.init()

import mujoco  # noqa: F401,E402
import torch  # noqa: E402

from SAFTER_src.high_level_rl.laser_cbf_filters import LaserZeroBackupLseCbfClosedForm  # noqa: E402

# Reused, unmodified, from the reference structured-map evaluator.
from SAFTER_scripts.laser_unseen_eval.evaluate_laser_unseen_maps_structured import (  # noqa: E402
    UNSEEN_MAPS,
    STRUCTURED_SCENARIO_REGIONS,
    build_environment,
    generate_manifest,
    make_map_arena,
    optional_mean,
    physical_clearance,
    row_to_scenario,
    sample_std,
    set_seed,
    verify_compiled_physical_map,
    verify_parallel_lidar_geometry,
    write_rows,
)

METHOD_NAME = "heuristic_safter"
TASK_ID = "Unitree-Go2-Safe-Navigation"
DEFAULT_LOW_CHECKPOINT = "logs/rsl_rl/go2_velocity/2026-06-19_13-24-10/model_10000.pt"

DEFAULT_MAPS = ("low_shift", "medium_shift", "high_shift")
AVAILABLE_MAPS = tuple(name for name in UNSEEN_MAPS if name in STRUCTURED_SCENARIO_REGIONS)

# SAFTER (LaserZeroBackupLseCbfClosedForm) geometry/backup/control-dynamics
# hyperparameters. Identical to scripts/benchmark_safety_filter_latency.py's
# SHARED_KWARGS/BACKUP_KWARGS -- this project's own standard SAFTER config.
FILTER_SHARED_KWARGS: dict[str, Any] = dict(
    d_safe=0.30,
    max_vx=2.0, max_vy=1.0, max_wz=1.0,
    robot_length_m=0.70,
    robot_width_m=0.31,
    lidar_x_m=0.28945,
    lidar_y_m=0.0,
    footprint_clearance_m=0.15,
    use_elliptic_footprint=True,
    max_range=5.0,
    no_return_tolerance_m=1.0e-4,
    inactive_barrier_value=20.0,
    denominator_epsilon=1.0e-8,
    hard_clip_output=True,
)
FILTER_BACKUP_KWARGS: dict[str, Any] = dict(
    backup_horizon_s=4.0,
    backup_samples=21,
    backup_rollout_substeps=8,
    hocbf_k0=2.0,
    barrier_rate_scale=1.0,
    kappa_backup=50.0,
    kappa_input=20.0,
    kappa_outer=20.0,
    control_dynamics_a=(8.0, 8.0, 8.0),
    tracking_sigma=(8.0, 8.0, 8.0),
    integration_dt=0.10,
    terminal_backup_mode="invariant_ellipse",
    terminal_geom_scale=1.0,
    terminal_rec_scale=1.0,
    alpha_outer=0.8,
)

# ControllerConfig defaults, src/high_level_control/config.py. kp_x/kp_y/
# kp_yaw are the P-gains; min/max_v* are the heuristic's OWN output clip
# (independent of, and well inside, the filter's own box).
HEURISTIC_KP = (0.8, 0.8, 1.5)
HEURISTIC_MIN_CMD = (-0.8, -0.7, -1.0)
HEURISTIC_MAX_CMD = (1.2, 0.7, 1.0)


class BatchedGoToGoalController:
    """Batched torch port of
    src/high_level_control/controller.py:FeedbackLinearizationController.compute
    composed with src/high_level_control/trajectories.py:PointGoal and
    src/high_level_control/model.py:HighLevelKinematicModel.input_matrix.
    See eval_heuristic_hardbox_nocd.py's identical class for the full
    derivation writeup (R(theta)^T closed-form inverse, pursuit-heading
    yaw reference, etc.) -- byte-identical here, duplicated rather than
    imported so this script has no dependency on the no-CD script.
    """

    def __init__(self, device: torch.device | str, dtype: torch.dtype = torch.float32) -> None:
        self.kp = torch.tensor(HEURISTIC_KP, device=device, dtype=dtype)
        self.min_cmd = torch.tensor(HEURISTIC_MIN_CMD, device=device, dtype=dtype)
        self.max_cmd = torch.tensor(HEURISTIC_MAX_CMD, device=device, dtype=dtype)

    def compute(
        self, pose_xy: torch.Tensor, yaw: torch.Tensor, goal_xy: torch.Tensor
    ) -> torch.Tensor:
        """pose_xy: [B,2], yaw: [B], goal_xy: [B,2] -> command [B,3] (vx,vy,wz)."""
        error_xy = goal_xy - pose_xy
        yaw_ref = torch.atan2(error_xy[:, 1], error_xy[:, 0])
        raw_yaw_error = yaw_ref - yaw
        error_yaw = torch.atan2(torch.sin(raw_yaw_error), torch.cos(raw_yaw_error))

        desired_x = self.kp[0] * error_xy[:, 0]
        desired_y = self.kp[1] * error_xy[:, 1]
        desired_yaw = self.kp[2] * error_yaw

        c = torch.cos(yaw)
        s = torch.sin(yaw)
        command_x = c * desired_x + s * desired_y
        command_y = -s * desired_x + c * desired_y
        command = torch.stack((command_x, command_y, desired_yaw), dim=-1)
        return torch.clamp(command, self.min_cmd, self.max_cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the obstacle-blind go-to-goal heuristic nominal "
            "controller through SAFTER (closed-form bCBF-CF, with control "
            "dynamics) on structured maps."
        )
    )
    parser.add_argument(
        "--maps", nargs="+", default=list(DEFAULT_MAPS), choices=list(AVAILABLE_MAPS),
    )
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--seed-stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42, help="First evaluation seed.")
    parser.add_argument("--cases-per-seed", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--episode-duration-s", type=float, default=60.0)
    parser.add_argument("--laser-num-rays", type=int, default=61)
    parser.add_argument("--laser-fov-deg", type=float, default=180.0)
    parser.add_argument("--collision-geometry", choices=("circle", "ellipse"), default="ellipse")
    parser.add_argument("--collision-circle-radius-m", type=float, default=0.30)
    parser.add_argument("--collision-margin-m", type=float, default=0.05)
    parser.add_argument(
        "--collision-termination-mode", choices=("circle", "seanav_contact"), default="circle",
    )
    parser.add_argument("--low-level-checkpoint", default=DEFAULT_LOW_CHECKPOINT)
    parser.add_argument("--backup-margin", type=float, default=0.6)
    parser.add_argument("--min-endpoint-clearance", type=float, default=0.10)
    parser.add_argument("--block-margin", type=float, default=0.10)
    parser.add_argument("--blocked-t-min", type=float, default=0.15)
    parser.add_argument("--blocked-t-max", type=float, default=0.85)
    parser.add_argument("--max-sampling-attempts", type=int, default=10000)
    parser.add_argument("--yaw-jitter-rad", type=float, default=0.25)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--log-trajectories", action="store_true")
    parser.add_argument(
        "--output-root", default="logs/evaluation/heuristic_safter_structured",
        help=(
            "Point this at an EXISTING structured-eval run root to reuse "
            "its exact per-(map, seed) scenarios.csv manifests instead of "
            "generating new ones."
        ),
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def evaluate_map(
    *,
    map_name: str,
    pooled_manifest: list[dict[str, object]],
    args: argparse.Namespace,
    device: str,
) -> dict[str, object]:
    batch_size = min(int(args.num_envs), len(pooled_manifest))
    set_seed(int(pooled_manifest[0]["reset_seed"]))
    raw_env, low_env, env, cfg, dynamic_runtime = build_environment(
        task_id=TASK_ID,
        low_checkpoint=args.low_level_checkpoint,
        method="vanilla",  # only affects reward.cbf_correction_weight; irrelevant here.
        episode_duration_s=args.episode_duration_s,
        device=device,
        map_name=map_name,
        laser_num_rays=args.laser_num_rays,
        laser_fov_deg=args.laser_fov_deg,
        enable_lidar_noise=False,
        hard_clip_output=True,
        collision_geometry=args.collision_geometry,
        collision_circle_radius_m=args.collision_circle_radius_m,
        collision_margin_m=args.collision_margin_m,
        collision_termination_mode=args.collision_termination_mode,
        qp_backend="active_set",  # unused by SAFTER (closed-form), kept for build_environment's signature.
        num_envs=batch_size,
        terminal_backup_mode=FILTER_BACKUP_KWARGS["terminal_backup_mode"],
        terminal_geom_scale=FILTER_BACKUP_KWARGS["terminal_geom_scale"],
        terminal_rec_scale=FILTER_BACKUP_KWARGS["terminal_rec_scale"],
        legacy_projection=True,
        action_max_vx=FILTER_SHARED_KWARGS["max_vx"],
        action_min_vx=None,
        action_max_vy=FILTER_SHARED_KWARGS["max_vy"],
        action_max_wz=FILTER_SHARED_KWARGS["max_wz"],
        backup_margin=args.backup_margin,
    )
    records: list[dict[str, object]] = []
    trajectory_records: list[dict[str, object]] = []
    try:
        verify_compiled_physical_map(raw_env, env, map_name)

        safety_filter = LaserZeroBackupLseCbfClosedForm(
            **FILTER_SHARED_KWARGS,
            **FILTER_BACKUP_KWARGS,
            backup_margin=args.backup_margin,
            legacy_projection=True,
            straight_through=False,
        )
        controller = BatchedGoToGoalController(device=device)
        dt = float(env.navigation_cfg.high_level_dt)

        for batch_start in range(0, len(pooled_manifest), batch_size):
            real_cases = pooled_manifest[batch_start: batch_start + batch_size]
            padded_cases = list(real_cases)
            while len(padded_cases) < batch_size:
                padded_cases.append(real_cases[-1])
            env.set_fixed_scenarios([row_to_scenario(case) for case in padded_cases])
            verify_parallel_lidar_geometry(raw_env, env, map_name)

            positions, yaws = env.safety_context()
            goals = env.goal_positions
            for env_id, case in enumerate(real_cases):
                expected_start = positions.new_tensor(
                    [float(case["start_x"]), float(case["start_y"])]
                )
                expected_goal = goals.new_tensor(
                    [float(case["goal_x"]), float(case["goal_y"])]
                )
                expected_yaw = yaws.new_tensor(float(case["start_yaw"]))
                if not torch.allclose(positions[env_id], expected_start, atol=1e-4, rtol=0):
                    raise RuntimeError(f"{map_name}/case {case['case_id']}: start mismatch.")
                if not torch.allclose(goals[env_id], expected_goal, atol=1e-4, rtol=0):
                    raise RuntimeError(f"{map_name}/case {case['case_id']}: goal mismatch.")
                if not torch.allclose(yaws[env_id], expected_yaw, atol=1e-4, rtol=0):
                    raise RuntimeError(f"{map_name}/case {case['case_id']}: yaw mismatch.")

            active_count = len(real_cases)
            finished = [False] * active_count
            previous_positions = positions[:active_count].detach().clone()
            path_lengths = [0.0] * active_count
            minimum_clearances = [
                physical_clearance(env, previous_positions[i], i) for i in range(active_count)
            ]
            active_steps = [0] * active_count
            correction_sums = [0.0] * active_count
            returns = [0.0] * active_count
            step_counts = [0] * active_count

            with torch.no_grad():
                while not all(finished):
                    pre_positions, pre_yaws = env.safety_context()
                    goal_xy = env.goal_positions
                    nominal = controller.compute(pre_positions, pre_yaws, goal_xy)

                    ranges = env.laser_ranges()
                    angles = env.ray_angles
                    # SAFTER models body velocity as a state with first-
                    # order command dynamics, so these three must be the
                    # env's OWN tracked state (same calling convention
                    # LaserSafeSacActor.apply_safety() uses for the SAC
                    # methods), not None.
                    current_command = env.previous_safe_command
                    previous_nominal_command = env.previous_nominal_command
                    nominal_history_valid = env.nominal_history_valid
                    safety = safety_filter.filter(
                        ranges, angles, current_command, nominal,
                        previous_nominal_command, nominal_history_valid,
                    )
                    safe = safety.command

                    _, reward, done, extras = env.step_safe(nominal, safe, safety)

                    completed_ids = extras.get(
                        "completed_env_ids", torch.empty(0, device=env.device, dtype=torch.long)
                    )
                    completed_lookup = {
                        int(env_id): row for row, env_id in enumerate(completed_ids.tolist())
                    }
                    current_positions, current_yaws = env.safety_context()

                    for env_id, case in enumerate(real_cases):
                        if finished[env_id]:
                            continue
                        step_counts[env_id] += 1
                        step_reward = float(reward[env_id].detach().cpu())
                        returns[env_id] += step_reward
                        completed_row = completed_lookup.get(env_id)
                        is_done = completed_row is not None
                        if is_done:
                            endpoint = extras["completed_terminal_positions"][completed_row]
                        else:
                            endpoint = current_positions[env_id]

                        start_position = previous_positions[env_id].detach().clone()
                        path_lengths[env_id] += float(
                            torch.linalg.norm(endpoint - start_position).detach().cpu()
                        )
                        previous_positions[env_id] = endpoint.detach().clone()
                        clearance = physical_clearance(env, endpoint, env_id)
                        minimum_clearances[env_id] = min(minimum_clearances[env_id], clearance)
                        correction_value = float(
                            torch.linalg.norm(safe[env_id] - nominal[env_id]).detach().cpu()
                        )
                        correction_sums[env_id] += correction_value
                        filter_active = correction_value > 1e-8
                        active_steps[env_id] += int(filter_active)

                        trajectory_row = None
                        if args.log_trajectories:
                            trajectory_row = {
                                "method": METHOD_NAME, "map_name": map_name,
                                "evaluation_seed": int(case["evaluation_seed"]),
                                "case_id": int(case["case_id"]),
                                "reset_seed": int(case["reset_seed"]),
                                "env_id": env_id, "step": step_counts[env_id],
                                "time_s": step_counts[env_id] * dt,
                                "pre_x_m": float(start_position[0].detach().cpu()),
                                "pre_y_m": float(start_position[1].detach().cpu()),
                                "post_x_m": float(endpoint[0].detach().cpu()),
                                "post_y_m": float(endpoint[1].detach().cpu()),
                                "yaw_rad": float(pre_yaws[env_id].detach().cpu()),
                                "start_x_m": float(case["start_x"]),
                                "start_y_m": float(case["start_y"]),
                                "goal_x_m": float(case["goal_x"]),
                                "goal_y_m": float(case["goal_y"]),
                                "nominal_vx_mps": float(nominal[env_id, 0].detach().cpu()),
                                "nominal_vy_mps": float(nominal[env_id, 1].detach().cpu()),
                                "nominal_wz_radps": float(nominal[env_id, 2].detach().cpu()),
                                "safe_vx_mps": float(safe[env_id, 0].detach().cpu()),
                                "safe_vy_mps": float(safe[env_id, 1].detach().cpu()),
                                "safe_wz_radps": float(safe[env_id, 2].detach().cpu()),
                                "command_correction_norm": correction_value,
                                "filter_active": int(filter_active),
                                "physical_clearance_m": clearance,
                                "cumulative_path_length_m": path_lengths[env_id],
                                "step_reward": step_reward,
                                "cumulative_episode_return": returns[env_id],
                                "done": int(is_done), "outcome": "", "success": 0,
                                "collision": 0, "timeout": 0, "fallen": 0, "outside": 0,
                            }

                        if is_done:
                            success = bool(extras["completed_success"][completed_row])
                            collision = bool(extras["completed_collision"][completed_row])
                            timeout = bool(extras["completed_timeouts"][completed_row])
                            fallen = bool(extras["completed_fallen"][completed_row])
                            outside = bool(extras["completed_outside"][completed_row])
                            if success and not collision:
                                outcome = "success"
                            elif collision:
                                outcome = "collision"
                            elif timeout:
                                outcome = "timeout"
                            elif fallen:
                                outcome = "fall"
                            elif outside:
                                outcome = "outside"
                            else:
                                outcome = "other_failure"
                            returns[env_id] = float(
                                extras["completed_returns"][completed_row].detach().cpu()
                            )
                            duration = int(extras["completed_lengths"][completed_row]) * dt
                            if trajectory_row is not None:
                                trajectory_row.update({
                                    "outcome": outcome, "success": int(success),
                                    "collision": int(collision), "timeout": int(timeout),
                                    "fallen": int(fallen), "outside": int(outside),
                                })
                            cumulative_mean = (
                                sum(float(row["return"]) for row in records) + returns[env_id]
                            ) / (len(records) + 1)
                            records.append({
                                "method": METHOD_NAME, "map_name": map_name,
                                "evaluation_seed": int(case["evaluation_seed"]),
                                "case_id": int(case["case_id"]), "reset_seed": int(case["reset_seed"]),
                                "start_x": float(case["start_x"]), "start_y": float(case["start_y"]),
                                "start_yaw": float(case["start_yaw"]),
                                "goal_x": float(case["goal_x"]), "goal_y": float(case["goal_y"]),
                                "start_clearance_m": float(case.get("start_clearance_m", float("nan"))),
                                "goal_clearance_m": float(case.get("goal_clearance_m", float("nan"))),
                                "straight_line_distance_m": float(
                                    case.get("straight_line_distance_m", float("nan"))
                                ),
                                "straight_path_min_clearance_m": float(
                                    case.get("straight_path_min_clearance_m", float("nan"))
                                ),
                                "direct_path_blocked": int(case.get("direct_path_blocked", 0)),
                                "blocking_obstacle_id": int(case.get("blocking_obstacle_id", -1)),
                                "blocking_t": float(case.get("blocking_t", float("nan"))),
                                "blocking_clearance_m": float(case.get("blocking_clearance_m", float("nan"))),
                                "sampling_attempt": int(case.get("sampling_attempt", -1)),
                                "outcome": outcome, "success": int(success), "collision": int(collision),
                                "timeout": int(timeout), "fallen": int(fallen), "outside": int(outside),
                                "return": returns[env_id], "cumulative_mean_return": cumulative_mean,
                                "path_length_m": path_lengths[env_id], "episode_duration_s": duration,
                                "episode_steps": step_counts[env_id],
                                "minimum_physical_clearance_m": minimum_clearances[env_id],
                                "filter_active_fraction": active_steps[env_id] / max(step_counts[env_id], 1),
                                "mean_command_correction_norm": (
                                    correction_sums[env_id] / max(step_counts[env_id], 1)
                                ),
                            })
                            finished[env_id] = True
                            completed_total = batch_start + sum(finished)
                            if args.print_every > 0 and completed_total % args.print_every == 0:
                                print(
                                    f"[{METHOD_NAME}] map={map_name:10s} "
                                    f"completed={completed_total:3d}/{len(pooled_manifest):3d} "
                                    f"case={int(case['case_id']):3d} outcome={outcome:13s} "
                                    f"return={returns[env_id]:+8.2f} path={path_lengths[env_id]:7.3f} m "
                                    f"time={duration:6.2f} s clearance={minimum_clearances[env_id]:+.3f} m",
                                    flush=True,
                                )
                        if trajectory_row is not None:
                            trajectory_records.append(trajectory_row)

        records.sort(key=lambda row: (int(row["evaluation_seed"]), int(row["case_id"])))
        trajectory_records.sort(
            key=lambda row: (int(row["evaluation_seed"]), int(row["case_id"]), int(row["step"]))
        )
        if not records:
            raise RuntimeError(f"No completed episodes recorded for map {map_name!r}.")
        successes = [row for row in records if int(row["success"]) == 1]
        summary = {
            "method": METHOD_NAME, "map_name": map_name, "cases": len(records),
            "success_rate": sum(int(row["success"]) for row in records) / len(records),
            "collision_rate": sum(int(row["collision"]) for row in records) / len(records),
            "timeout_rate": sum(int(row["timeout"]) for row in records) / len(records),
            "fall_rate": sum(int(row["fallen"]) for row in records) / len(records),
            "outside_rate": sum(int(row["outside"]) for row in records) / len(records),
            "mean_return": mean(float(row["return"]) for row in records),
            "std_return": sample_std([float(row["return"]) for row in records]),
            "mean_success_path_length_m": optional_mean(
                [float(row["path_length_m"]) for row in successes]
            ),
            "std_success_path_length_m": sample_std([float(row["path_length_m"]) for row in successes]),
            "mean_success_goal_time_s": optional_mean(
                [float(row["episode_duration_s"]) for row in successes]
            ),
            "std_success_goal_time_s": sample_std(
                [float(row["episode_duration_s"]) for row in successes]
            ),
            "mean_minimum_clearance_m": mean(
                float(row["minimum_physical_clearance_m"]) for row in records
            ),
            "minimum_clearance_overall_m": min(
                float(row["minimum_physical_clearance_m"]) for row in records
            ),
            "mean_filter_active_fraction": mean(
                float(row["filter_active_fraction"]) for row in records
            ),
            "mean_command_correction_norm": mean(
                float(row["mean_command_correction_norm"]) for row in records
            ),
        }
        return {"records": records, "trajectory_records": trajectory_records, "summary": summary}
    finally:
        try:
            close = getattr(env, "close", None)
            if callable(close):
                close()
        except Exception as exc:
            print(f"[WARNING] Failed to close env for {map_name}: {exc}", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_or_generate_manifest(
    *, map_name: str, evaluation_seed: int, map_dir: Path, args: argparse.Namespace, device: str,
) -> list[dict[str, object]]:
    manifest_path = map_dir / "scenarios.csv"
    if manifest_path.is_file():
        with manifest_path.open(newline="") as file:
            manifest = [dict(row) for row in csv.DictReader(file)]
        if len(manifest) != args.cases_per_seed:
            raise RuntimeError(
                f"Existing manifest {manifest_path} has {len(manifest)} cases; "
                f"expected {args.cases_per_seed}."
            )
        print(f"[RESUME] reusing {manifest_path}")
        for case in manifest:
            case.setdefault("evaluation_seed", evaluation_seed)
        return manifest

    map_index = list(UNSEEN_MAPS).index(map_name)
    manifest_seed = evaluation_seed + 10000 * map_index
    arena = replace(make_map_arena(map_name), wall_clearance=0.30, obstacle_clearance=0.30)

    set_seed(manifest_seed)
    raw_env, low_env, env, _, _ = build_environment(
        task_id=TASK_ID,
        low_checkpoint=args.low_level_checkpoint,
        method="vanilla",
        episode_duration_s=args.episode_duration_s,
        device=device,
        map_name=map_name,
        laser_num_rays=args.laser_num_rays,
        laser_fov_deg=args.laser_fov_deg,
        enable_lidar_noise=False,
        hard_clip_output=True,
        collision_geometry=args.collision_geometry,
        collision_circle_radius_m=args.collision_circle_radius_m,
        collision_margin_m=args.collision_margin_m,
        collision_termination_mode=args.collision_termination_mode,
        action_max_vx=FILTER_SHARED_KWARGS["max_vx"],
        action_min_vx=None,
        action_max_vy=FILTER_SHARED_KWARGS["max_vy"],
        action_max_wz=FILTER_SHARED_KWARGS["max_wz"],
        num_envs=1,
    )
    try:
        manifest = generate_manifest(
            env=env, arena=arena, map_name=map_name, count=args.cases_per_seed,
            seed=manifest_seed, output_path=manifest_path,
            min_endpoint_clearance=args.min_endpoint_clearance,
            block_margin=args.block_margin,
            require_blocked_path=True,
            blocked_t_min=args.blocked_t_min, blocked_t_max=args.blocked_t_max,
            max_sampling_attempts=args.max_sampling_attempts,
            yaw_jitter_rad=args.yaw_jitter_rad,
        )
        for case in manifest:
            case["evaluation_seed"] = evaluation_seed
        write_rows(manifest_path, manifest)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
        del env, low_env, raw_env
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return manifest


def main() -> None:
    args = parse_args()
    if args.cases_per_seed <= 0 or args.num_seeds <= 0 or args.seed_stride <= 0:
        raise ValueError("--cases-per-seed, --num-seeds, and --seed-stride must be positive.")
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation device: {device}")

    low_path = Path(args.low_level_checkpoint).expanduser()
    if not low_path.is_file():
        raise FileNotFoundError(f"Missing low-level checkpoint: {low_path}")

    evaluation_seeds = [int(args.seed) + i * int(args.seed_stride) for i in range(args.num_seeds)]
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)

    all_summaries: list[dict[str, object]] = []
    for map_name in args.maps:
        print("\n" + "=" * 72)
        print(f"MAP: {map_name} | seeds: {evaluation_seeds}")
        print("=" * 72)
        pooled_manifest: list[dict[str, object]] = []
        for evaluation_seed in evaluation_seeds:
            seed_dir = root / f"seed_{evaluation_seed}"
            map_dir = seed_dir / map_name
            map_dir.mkdir(parents=True, exist_ok=True)
            manifest = load_or_generate_manifest(
                map_name=map_name, evaluation_seed=evaluation_seed,
                map_dir=map_dir, args=args, device=device,
            )
            pooled_manifest.extend(manifest)

        result = evaluate_map(
            map_name=map_name, pooled_manifest=pooled_manifest, args=args, device=device,
        )
        records = result["records"]
        trajectory_records = result["trajectory_records"]
        summary = result["summary"]

        by_seed: dict[int, list[dict[str, object]]] = {}
        for row in records:
            by_seed.setdefault(int(row["evaluation_seed"]), []).append(row)
        traj_by_seed: dict[int, list[dict[str, object]]] = {}
        for row in trajectory_records:
            traj_by_seed.setdefault(int(row["evaluation_seed"]), []).append(row)

        for evaluation_seed in evaluation_seeds:
            seed_rows = by_seed.get(evaluation_seed, [])
            if not seed_rows:
                continue
            method_dir = root / f"seed_{evaluation_seed}" / map_name / METHOD_NAME
            method_dir.mkdir(parents=True, exist_ok=True)
            write_rows(method_dir / f"{METHOD_NAME}_episodes.csv", seed_rows)
            seed_successes = [row for row in seed_rows if int(row["success"]) == 1]
            seed_summary = {
                "method": METHOD_NAME, "map_name": map_name, "cases": len(seed_rows),
                "success_rate": sum(int(r["success"]) for r in seed_rows) / len(seed_rows),
                "collision_rate": sum(int(r["collision"]) for r in seed_rows) / len(seed_rows),
                "timeout_rate": sum(int(r["timeout"]) for r in seed_rows) / len(seed_rows),
                "fall_rate": sum(int(r["fallen"]) for r in seed_rows) / len(seed_rows),
                "outside_rate": sum(int(r["outside"]) for r in seed_rows) / len(seed_rows),
                "mean_return": mean(float(r["return"]) for r in seed_rows),
                "std_return": sample_std([float(r["return"]) for r in seed_rows]),
                "mean_success_path_length_m": optional_mean(
                    [float(r["path_length_m"]) for r in seed_successes]
                ),
                "std_success_path_length_m": sample_std(
                    [float(r["path_length_m"]) for r in seed_successes]
                ),
                "mean_success_goal_time_s": optional_mean(
                    [float(r["episode_duration_s"]) for r in seed_successes]
                ),
                "std_success_goal_time_s": sample_std(
                    [float(r["episode_duration_s"]) for r in seed_successes]
                ),
                "mean_minimum_clearance_m": mean(
                    float(r["minimum_physical_clearance_m"]) for r in seed_rows
                ),
                "minimum_clearance_overall_m": min(
                    float(r["minimum_physical_clearance_m"]) for r in seed_rows
                ),
                "mean_filter_active_fraction": mean(
                    float(r["filter_active_fraction"]) for r in seed_rows
                ),
                "mean_command_correction_norm": mean(
                    float(r["mean_command_correction_norm"]) for r in seed_rows
                ),
            }
            write_rows(method_dir / f"{METHOD_NAME}_summary.csv", [seed_summary])
            if args.log_trajectories and evaluation_seed in traj_by_seed:
                write_rows(
                    method_dir / f"{METHOD_NAME}_trajectory.csv", traj_by_seed[evaluation_seed]
                )

        print(f"\n[{METHOD_NAME} POOLED SUMMARY] map={map_name}")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        all_summaries.append(summary)

    write_rows(root / f"{METHOD_NAME}_pooled_summary_by_map.csv", all_summaries)
    print("\n" + "=" * 72)
    print(f"{METHOD_NAME} structured evaluation completed")
    print(f"Results: {root}")
    print("=" * 72)


if __name__ == "__main__":
    main()
