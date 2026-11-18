#!/usr/bin/env python3
"""Evaluate a heuristic or SAC nominal policy through a chosen safety filter.

Runs on the procedural training-map bank (--map-source procedural) or on the
structured low/medium/high-shift maps (--map-source structured).

--nominal-source {heuristic, sac}:
    heuristic: BatchedGoToGoalController, an obstacle-blind, stateless P
               controller whose output is the applied body velocity, clipped
               to HEURISTIC_MIN_CMD/MAX_CMD. LiDAR noise is off.
    sac:       the raw (pre-filter) command of the SAC checkpoint given by
               --vanilla-checkpoint. LiDAR noise is on, matching the
               checkpoint's training configuration (load_actor checks it).
--filter-type: any filter supported by eval_vanilla_filterswap.py's
    build_retrofit_filter(), so every nominal and map source uses identical
    filter construction.
--map-source {procedural, structured} (default procedural):
    procedural: --procedural-map-bank selects the bank, --map-id one map.
    structured: --maps selects the maps; the scenario-sampling flags
        (--min-endpoint-clearance, --block-margin, --blocked-t-min,
        --blocked-t-max, --max-sampling-attempts, --yaw-jitter-rad) default to
        the values of evaluate_laser_unseen_maps_structured.py.

Per-step clip and barrier diagnostics (clip_delta, clip_changed, residual_exec,
projection_feasible) are reported per episode and pooled over steps.
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import math
import sys
from dataclasses import replace
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# warp.init() must run before mjlab.entity is imported transitively below,
# or it segfaults.
import warp as wp  # noqa: E402

wp.init()

import mujoco  # noqa: F401,E402
import torch  # noqa: E402


def _load_sibling_module(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_training_maps_module = _load_sibling_module(
    "_evaluate_training_maps_no_filter", "SAFTER_scripts/evaluate_training_maps_no_filter.py"
)
_structured_module = _load_sibling_module(
    "_evaluate_laser_unseen_maps_structured",
    "SAFTER_scripts/laser_unseen_eval/evaluate_laser_unseen_maps_structured.py",
)
_filterswap_module = _load_sibling_module(
    "_eval_vanilla_filterswap", "SAFTER_scripts/eval_vanilla_filterswap.py"
)

build_procedural_environment = _training_maps_module.build_environment
build_manifest = _training_maps_module.build_manifest

build_structured_environment = _structured_module.build_environment
UNSEEN_MAPS = _structured_module.UNSEEN_MAPS
STRUCTURED_SCENARIO_REGIONS = _structured_module.STRUCTURED_SCENARIO_REGIONS
generate_manifest = _structured_module.generate_manifest
make_map_arena = _structured_module.make_map_arena
row_to_scenario = _structured_module.row_to_scenario
verify_compiled_physical_map = _structured_module.verify_compiled_physical_map
verify_parallel_lidar_geometry = _structured_module.verify_parallel_lidar_geometry
set_seed = _structured_module.set_seed
load_actor = _structured_module.load_actor
timed_deterministic_actor = _structured_module.timed_deterministic_actor
warmup_actor_timing = _structured_module.warmup_actor_timing
write_rows = _structured_module.write_rows
sample_std = _structured_module.sample_std
optional_mean = _structured_module.optional_mean
physical_clearance = _structured_module.physical_clearance
BARRIER_VIOLATION_EPS = _structured_module.BARRIER_VIOLATION_EPS

build_retrofit_filter = _filterswap_module.build_retrofit_filter
FILTER_TYPES = _filterswap_module.FILTER_TYPES

TASK_ID = "Unitree-Go2-Safe-Navigation"
ACTION_LIMITS = (2.0, 1.0, 1.0, -1.0)  # (max_vx, max_vy, max_wz, min_vx)

DEFAULT_STRUCTURED_MAPS = ("low_shift", "medium_shift", "high_shift")
AVAILABLE_STRUCTURED_MAPS = tuple(
    name for name in UNSEEN_MAPS if name in STRUCTURED_SCENARIO_REGIONS
)

# P-gains and the heuristic's own output clip (ControllerConfig defaults),
# which lies inside the safety filter's action box.
HEURISTIC_KP = (0.8, 0.8, 1.5)
HEURISTIC_MIN_CMD = (-0.8, -0.7, -1.0)
HEURISTIC_MAX_CMD = (1.2, 0.7, 1.0)


class BatchedGoToGoalController:
    """Batched obstacle-blind go-to-goal P controller.

    With the controlled point at the base origin, the kinematic input matrix
    is a planar rotation, so the body command is

        command_xy = R(theta)^T @ (K_xy @ position_error)
        command_wz = k_yaw * yaw_error

    clipped to HEURISTIC_MIN_CMD/MAX_CMD. The goal has no target heading; the reference yaw is the current bearing
    to the goal, atan2(dy, dx), recomputed every step.
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
        error_yaw = torch.atan2(torch.sin(raw_yaw_error), torch.cos(raw_yaw_error))  # wrap_to_pi

        desired_x = self.kp[0] * error_xy[:, 0]
        desired_y = self.kp[1] * error_xy[:, 1]
        desired_yaw = self.kp[2] * error_yaw

        c = torch.cos(yaw)
        s = torch.sin(yaw)
        # R(theta)^T @ [desired_x, desired_y]
        command_x = c * desired_x + s * desired_y
        command_y = -s * desired_x + c * desired_y
        raw = torch.stack((command_x, command_y, desired_yaw), dim=-1)
        return torch.clamp(raw, self.min_cmd, self.max_cmd)


def evaluate(
    *,
    nominal_source: str,
    filter_type: str,
    map_source: str,
    manifest: list[dict],
    batch_size: int,
    raw_env,
    env,
    cfg,
    vanilla_checkpoint: str | None,
    log_trajectories: bool,
    device: str,
    structured_map_name: str | None = None,
) -> dict[str, object]:
    method_name = f"{nominal_source}_nominal_{filter_type}"
    dt = float(env.navigation_cfg.high_level_dt)

    records: list[dict[str, object]] = []
    trajectory_records: list[dict[str, object]] = []
    # Step-weighted (pooled) clip and barrier counters.
    pooled_diag_steps = 0
    pooled_clip_steps = 0
    pooled_clip_violation_steps = 0
    pooled_no_clip_violation_steps = 0
    pooled_clip_feasible_steps = 0
    pooled_clip_feasible_violation_steps = 0
    effective_min_vx = ACTION_LIMITS[3]
    box_scale = math.sqrt(
        (ACTION_LIMITS[0] - effective_min_vx) ** 2
        + (2.0 * ACTION_LIMITS[1]) ** 2
        + (2.0 * ACTION_LIMITS[2]) ** 2
    )
    pooled_clip_delta_norm_max = 0.0

    controller = None
    actor = None
    if nominal_source == "heuristic":
        controller = BatchedGoToGoalController(device=device)
    else:
        if not vanilla_checkpoint:
            raise ValueError("--vanilla-checkpoint is required for --nominal-source sac")
        actor = load_actor(env=env, cfg=cfg, checkpoint_path=vanilla_checkpoint, device=device)
        warmup_actor_timing(actor, env)

    retrofit_filter = build_retrofit_filter(filter_type, cfg, ACTION_LIMITS)

    for batch_start in range(0, len(manifest), batch_size):
        real_cases = manifest[batch_start : batch_start + batch_size]
        padded_cases = list(real_cases)
        while len(padded_cases) < batch_size:
            padded_cases.append(real_cases[-1])

        if map_source == "procedural":
            scenario_ids = torch.tensor(
                [int(case["scenario_id"]) for case in padded_cases], dtype=torch.long, device=device
            )
            env.set_fixed_procedural_scenarios(scenario_ids)
        else:
            env.set_fixed_scenarios([row_to_scenario(case) for case in padded_cases])
            verify_parallel_lidar_geometry(raw_env, env, structured_map_name)
        obs = env.get_observations()

        active_count = len(real_cases)
        finished = [False] * active_count
        positions, _ = env.safety_context()
        previous_positions = positions[:active_count].detach().clone()
        path_lengths = [0.0] * active_count
        minimum_clearances = [
            physical_clearance(env, previous_positions[i], i) for i in range(active_count)
        ]
        active_steps = [0] * active_count
        correction_sums = [0.0] * active_count
        clip_triggered_steps = [0] * active_count
        clip_delta_sums = [0.0] * active_count
        clip_delta_maxes = [0.0] * active_count
        clip_delta_norm_maxes = [0.0] * active_count
        barrier_violated_post_clip_steps = [0] * active_count
        residual_exec_mins = [math.inf] * active_count
        diag_steps = [0] * active_count
        barrier_violated_given_clip_steps = [0] * active_count
        barrier_violated_given_no_clip_steps = [0] * active_count
        clip_feasible_steps = [0] * active_count
        clip_feasible_violation_steps = [0] * active_count
        returns = [0.0] * active_count
        step_counts = [0] * active_count

        with torch.no_grad():
            while not all(finished):
                pre_positions, pre_yaws = env.safety_context()

                if nominal_source == "heuristic":
                    goal_xy = env.goal_positions
                    nominal = controller.compute(pre_positions, pre_yaws, goal_xy)
                    current_command = env.previous_safe_command
                    measured_velocity = env.measured_velocity()
                    previous_nominal_command = env.previous_nominal_command
                    nominal_history_valid = env.nominal_history_valid
                    ranges = env.laser_ranges()
                    angles = env.ray_angles
                else:
                    actor_obs = env.actor_tensor(obs)
                    nominal, _safe, _safety, _, _, _ = timed_deterministic_actor(actor, actor_obs)
                    current_command = actor_obs[:, 5:8]
                    measured_velocity = actor_obs[:, 2:5]
                    previous_nominal_command = actor_obs[:, -4:-1]
                    nominal_history_valid = actor_obs[:, -1:]
                    ranges = actor._ranges(actor_obs)
                    angles = actor.ray_angles.to(device=actor_obs.device, dtype=actor_obs.dtype)

                # The adapted filter's control-dynamics state is the measured
                # body velocity; the other filters use the previously executed
                # command (filters without control dynamics ignore it).
                command_state = (
                    measured_velocity if filter_type == "bcbf_cd_lse_adapted" else current_command
                )
                safety = retrofit_filter.filter(
                    ranges, angles, command_state, nominal,
                    previous_nominal_command, nominal_history_valid,
                )
                safe = safety.command

                next_obs, reward, done, extras = env.step_safe(nominal, safe, safety)

                completed_ids = extras.get(
                    "completed_env_ids", torch.empty(0, device=env.device, dtype=torch.long)
                )
                completed_lookup = {
                    int(env_id): row for row, env_id in enumerate(completed_ids.tolist())
                }
                current_positions, _ = env.safety_context()

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

                    feasible_flags = getattr(safety, "projection_feasible", None)
                    clip_delta_flags = getattr(safety, "clip_delta", None)
                    if clip_delta_flags is not None:
                        clip_delta_value = float(clip_delta_flags[env_id].detach().cpu())
                        clip_delta_sums[env_id] += clip_delta_value
                        clip_delta_maxes[env_id] = max(clip_delta_maxes[env_id], clip_delta_value)
                        clip_delta_norm_value = clip_delta_value / max(box_scale, 1.0e-12)
                        clip_delta_norm_maxes[env_id] = max(
                            clip_delta_norm_maxes[env_id], clip_delta_norm_value
                        )
                        pooled_clip_delta_norm_max = max(pooled_clip_delta_norm_max, clip_delta_norm_value)
                    clip_changed_flags = getattr(safety, "clip_changed", None)
                    if clip_changed_flags is not None:
                        clip_triggered_steps[env_id] += int(bool(clip_changed_flags[env_id]))
                    residual_exec_flags = getattr(safety, "residual_exec", None)
                    if residual_exec_flags is not None:
                        residual_exec_value = float(residual_exec_flags[env_id].detach().cpu())
                        is_barrier_violated = residual_exec_value < -BARRIER_VIOLATION_EPS
                        barrier_violated_post_clip_steps[env_id] += int(is_barrier_violated)
                        residual_exec_mins[env_id] = min(residual_exec_mins[env_id], residual_exec_value)
                        pooled_diag_steps += 1
                        diag_steps[env_id] += 1
                        step_clip_changed = (
                            clip_changed_flags is not None and bool(clip_changed_flags[env_id])
                        )
                        if step_clip_changed:
                            pooled_clip_steps += 1
                            pooled_clip_violation_steps += int(is_barrier_violated)
                            barrier_violated_given_clip_steps[env_id] += int(is_barrier_violated)
                            if feasible_flags is None or bool(feasible_flags[env_id]):
                                pooled_clip_feasible_steps += 1
                                pooled_clip_feasible_violation_steps += int(is_barrier_violated)
                                clip_feasible_steps[env_id] += 1
                                clip_feasible_violation_steps[env_id] += int(is_barrier_violated)
                        else:
                            pooled_no_clip_violation_steps += int(is_barrier_violated)
                            barrier_violated_given_no_clip_steps[env_id] += int(is_barrier_violated)

                    trajectory_row = None
                    if log_trajectories:
                        trajectory_row = {
                            "method": method_name,
                            "map_source": map_source,
                            "map_id": int(case["map_id"]) if "map_id" in case else -1,
                            "map_name": case.get("map_name", ""),
                            "evaluation_seed": int(case["evaluation_seed"]),
                            "case_id": int(case["case_id"]),
                            "scenario_id": int(case["scenario_id"]) if "scenario_id" in case else -1,
                            "step": step_counts[env_id],
                            "time_s": step_counts[env_id] * dt,
                            "nominal_vx_mps": float(nominal[env_id, 0].detach().cpu()),
                            "nominal_vy_mps": float(nominal[env_id, 1].detach().cpu()),
                            "nominal_wz_radps": float(nominal[env_id, 2].detach().cpu()),
                            "safe_vx_mps": float(safe[env_id, 0].detach().cpu()),
                            "safe_vy_mps": float(safe[env_id, 1].detach().cpu()),
                            "safe_wz_radps": float(safe[env_id, 2].detach().cpu()),
                            "command_correction_norm": correction_value,
                            "clip_delta_m_s": (
                                float(clip_delta_flags[env_id].detach().cpu())
                                if clip_delta_flags is not None else math.nan
                            ),
                            "clip_changed": (
                                int(bool(clip_changed_flags[env_id])) if clip_changed_flags is not None else 0
                            ),
                            "residual_exec": (
                                float(residual_exec_flags[env_id].detach().cpu())
                                if residual_exec_flags is not None else math.nan
                            ),
                            "filter_active": int(filter_active),
                            "physical_clearance_m": clearance,
                            "cumulative_path_length_m": path_lengths[env_id],
                            "step_reward": step_reward,
                            "cumulative_episode_return": returns[env_id],
                            "done": int(is_done), "outcome": "",
                            "success": 0, "collision": 0, "timeout": 0,
                            "fallen": 0, "outside": 0,
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
                        final_return = float(
                            extras["completed_returns"][completed_row].detach().cpu()
                        )
                        duration = int(extras["completed_lengths"][completed_row]) * dt
                        if trajectory_row is not None:
                            trajectory_row.update({
                                "outcome": outcome, "success": int(success),
                                "collision": int(collision), "timeout": int(timeout),
                                "fallen": int(fallen), "outside": int(outside),
                            })
                        records.append({
                            "method": method_name,
                            "map_source": map_source,
                            "map_id": int(case["map_id"]) if "map_id" in case else -1,
                            "map_name": case.get("map_name", ""),
                            "evaluation_seed": int(case["evaluation_seed"]),
                            "case_id": int(case["case_id"]),
                            "scenario_id": int(case["scenario_id"]) if "scenario_id" in case else -1,
                            "outcome": outcome, "success": int(success), "collision": int(collision),
                            "timeout": int(timeout), "fallen": int(fallen), "outside": int(outside),
                            "return": final_return,
                            "path_length_m": path_lengths[env_id], "episode_duration_s": duration,
                            "episode_steps": step_counts[env_id],
                            "minimum_physical_clearance_m": minimum_clearances[env_id],
                            "filter_active_fraction": active_steps[env_id] / max(step_counts[env_id], 1),
                            "mean_command_correction_norm": (
                                correction_sums[env_id] / max(step_counts[env_id], 1)
                            ),
                            "diag_steps": diag_steps[env_id],
                            "clip_triggered_steps": clip_triggered_steps[env_id],
                            "clip_triggered_fraction": clip_triggered_steps[env_id] / max(step_counts[env_id], 1),
                            "mean_clip_delta_m_s": clip_delta_sums[env_id] / max(step_counts[env_id], 1),
                            "max_clip_delta_m_s": clip_delta_maxes[env_id],
                            "max_clip_delta_normalized": clip_delta_norm_maxes[env_id],
                            "barrier_violated_post_clip_fraction": (
                                barrier_violated_post_clip_steps[env_id] / max(step_counts[env_id], 1)
                            ),
                            "min_residual_exec": (
                                residual_exec_mins[env_id] if math.isfinite(residual_exec_mins[env_id]) else math.nan
                            ),
                            "barrier_violated_given_clip_steps": barrier_violated_given_clip_steps[env_id],
                            "barrier_violated_given_no_clip_steps": barrier_violated_given_no_clip_steps[env_id],
                            "clip_feasible_steps": clip_feasible_steps[env_id],
                            "clip_feasible_violation_steps": clip_feasible_violation_steps[env_id],
                        })
                        finished[env_id] = True
                    if trajectory_row is not None:
                        trajectory_records.append(trajectory_row)

                obs = next_obs
        print(
            f"[{method_name}] batch {batch_start}-{batch_start + len(real_cases)}/{len(manifest)} done "
            f"({len(records)} episodes recorded so far)"
        )

    successes = [row for row in records if int(row["success"]) == 1]
    summary = {
        "method": method_name, "cases": len(records),
        "success_rate": sum(int(row["success"]) for row in records) / len(records),
        "collision_rate": sum(int(row["collision"]) for row in records) / len(records),
        "timeout_rate": sum(int(row["timeout"]) for row in records) / len(records),
        "fall_rate": sum(int(row["fallen"]) for row in records) / len(records),
        "outside_rate": sum(int(row["outside"]) for row in records) / len(records),
        "mean_return": mean(float(row["return"]) for row in records),
        "std_return": sample_std([float(row["return"]) for row in records]),
        "mean_success_path_length_m": optional_mean([float(r["path_length_m"]) for r in successes]),
        "mean_minimum_clearance_m": mean(float(r["minimum_physical_clearance_m"]) for r in records),
        "minimum_clearance_overall_m": min(float(r["minimum_physical_clearance_m"]) for r in records),
        "mean_filter_active_fraction": mean(float(r["filter_active_fraction"]) for r in records),
        "mean_command_correction_norm": mean(float(r["mean_command_correction_norm"]) for r in records),
        "mean_clip_triggered_fraction": mean(float(r["clip_triggered_fraction"]) for r in records),
        "mean_clip_delta_m_s": mean(float(r["mean_clip_delta_m_s"]) for r in records),
        "max_clip_delta_m_s": max(float(r["max_clip_delta_m_s"]) for r in records),
        "max_clip_delta_normalized": max(float(r["max_clip_delta_normalized"]) for r in records),
        "mean_barrier_violated_post_clip_fraction": mean(
            float(r["barrier_violated_post_clip_fraction"]) for r in records
        ),
        "min_residual_exec_overall": min(
            (float(r["min_residual_exec"]) for r in records if math.isfinite(float(r["min_residual_exec"]))),
            default=math.nan,
        ),
        "pooled_diag_steps": pooled_diag_steps,
        "pooled_clip_steps": pooled_clip_steps,
        "pooled_clip_triggered_rate": pooled_clip_steps / max(pooled_diag_steps, 1),
        "pooled_barrier_violated_given_clip_rate": (
            pooled_clip_violation_steps / pooled_clip_steps if pooled_clip_steps > 0 else math.nan
        ),
        "pooled_barrier_violated_given_no_clip_rate": (
            pooled_no_clip_violation_steps / max(pooled_diag_steps - pooled_clip_steps, 1)
        ),
        "pooled_clip_feasible_steps": pooled_clip_feasible_steps,
        "pooled_clip_feasible_violation_steps": pooled_clip_feasible_violation_steps,
        "pooled_barrier_violated_given_clip_feasible_rate": (
            pooled_clip_feasible_violation_steps / pooled_clip_feasible_steps
            if pooled_clip_feasible_steps > 0 else math.nan
        ),
        "pooled_clip_delta_norm_max": pooled_clip_delta_norm_max,
    }
    return {"method": method_name, "records": records, "trajectory_records": trajectory_records, "summary": summary}


def load_or_generate_structured_manifest(
    *, map_name: str, evaluation_seed: int, map_dir: Path, args: argparse.Namespace, device: str,
) -> list[dict[str, object]]:
    """Per-(map, seed) structured start/goal manifest, written/reused at
    {map_dir}/scenarios.csv using the SAME (seed + 10000*map_index) manifest
    seed evaluate_laser_unseen_maps_structured.py itself uses, so pointing
    --output-root at an existing structured run reuses its exact cases."""
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
            case.setdefault("map_name", map_name)
        return manifest

    map_index = list(UNSEEN_MAPS).index(map_name)
    manifest_seed = evaluation_seed + 10000 * map_index
    arena = replace(make_map_arena(map_name), wall_clearance=0.30, obstacle_clearance=0.30)

    set_seed(manifest_seed)
    raw_env, _low_env, env, _cfg, _ = build_structured_environment(
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
        action_max_vx=ACTION_LIMITS[0],
        action_min_vx=ACTION_LIMITS[3],
        action_max_vy=ACTION_LIMITS[1],
        action_max_wz=ACTION_LIMITS[2],
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
            case["map_name"] = map_name
        write_rows(manifest_path, manifest)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
        del env, _low_env, raw_env
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return manifest


def run_structured_map(
    *, map_name: str, args: argparse.Namespace, device: str,
) -> dict[str, object]:
    evaluation_seeds = [int(args.seed) + i * int(args.seed_stride) for i in range(args.num_seeds)]
    pooled_manifest: list[dict[str, object]] = []
    for evaluation_seed in evaluation_seeds:
        map_dir = Path(args.output_root) / f"seed_{evaluation_seed}" / map_name
        map_dir.mkdir(parents=True, exist_ok=True)
        pooled_manifest.extend(
            load_or_generate_structured_manifest(
                map_name=map_name, evaluation_seed=evaluation_seed,
                map_dir=map_dir, args=args, device=device,
            )
        )

    batch_size = min(int(args.num_envs), len(pooled_manifest))
    set_seed(int(pooled_manifest[0]["reset_seed"]))
    raw_env, _low_env, env, cfg, _ = build_structured_environment(
        task_id=TASK_ID,
        low_checkpoint=args.low_level_checkpoint,
        method="vanilla",
        episode_duration_s=args.episode_duration_s,
        device=device,
        map_name=map_name,
        laser_num_rays=args.laser_num_rays,
        laser_fov_deg=args.laser_fov_deg,
        # A SAC checkpoint's laser config (noise on) must match exactly on
        # load; the heuristic source runs without noise.
        enable_lidar_noise=(args.nominal_source == "sac"),
        hard_clip_output=True,
        collision_geometry=args.collision_geometry,
        collision_circle_radius_m=args.collision_circle_radius_m,
        collision_margin_m=args.collision_margin_m,
        collision_termination_mode=args.collision_termination_mode,
        qp_backend=args.qp_backend,
        num_envs=batch_size,
        terminal_backup_mode=args.terminal_backup_mode,
        legacy_projection=args.legacy_projection,
        action_max_vx=ACTION_LIMITS[0],
        action_min_vx=ACTION_LIMITS[3],
        action_max_vy=ACTION_LIMITS[1],
        action_max_wz=ACTION_LIMITS[2],
        backup_margin=args.backup_margin,
    )
    try:
        verify_compiled_physical_map(raw_env, env, map_name)
        result = evaluate(
            nominal_source=args.nominal_source,
            filter_type=args.filter_type,
            map_source="structured",
            manifest=pooled_manifest,
            batch_size=batch_size,
            raw_env=raw_env, env=env, cfg=cfg,
            vanilla_checkpoint=args.vanilla_checkpoint,
            log_trajectories=args.log_trajectories,
            device=device,
            structured_map_name=map_name,
        )
    finally:
        try:
            raw_env.close()
        except Exception as exc:
            print(f"[WARNING] Failed to close env for map {map_name}: {exc}", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nominal-source", choices=("heuristic", "sac"), required=True)
    parser.add_argument("--filter-type", choices=FILTER_TYPES, required=True)
    parser.add_argument("--map-source", choices=("procedural", "structured"), default="procedural")
    parser.add_argument("--vanilla-checkpoint", default=None)
    parser.add_argument(
        "--low-level-checkpoint", default="logs/rsl_rl/go2_velocity/2026-06-19_13-24-10/model_10000.pt"
    )
    # --map-source procedural only.
    parser.add_argument("--procedural-map-bank", default="generated/procedural_maps/handcrafted2")
    parser.add_argument("--expected-num-maps", type=int, default=5)
    parser.add_argument(
        "--map-id", type=int, default=None,
        help=(
            "Restrict the manifest to a single map_id (0-indexed) within "
            "--procedural-map-bank. Default (unset): pool all "
            "--expected-num-maps maps."
        ),
    )
    parser.add_argument("--expected-scenarios-per-map", type=int, default=200)
    # --map-source structured only.
    parser.add_argument(
        "--maps", nargs="+", default=list(DEFAULT_STRUCTURED_MAPS), choices=list(AVAILABLE_STRUCTURED_MAPS),
    )
    parser.add_argument("--min-endpoint-clearance", type=float, default=0.10)
    parser.add_argument("--block-margin", type=float, default=0.10)
    parser.add_argument("--blocked-t-min", type=float, default=0.15)
    parser.add_argument("--blocked-t-max", type=float, default=0.85)
    parser.add_argument("--max-sampling-attempts", type=int, default=10000)
    parser.add_argument("--yaw-jitter-rad", type=float, default=0.25)
    # Shared.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--seed-stride", type=int, default=1)
    parser.add_argument("--cases-per-seed", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=100)
    parser.add_argument("--episode-duration-s", type=float, default=60.0)
    parser.add_argument("--laser-num-rays", type=int, default=120)
    parser.add_argument("--laser-fov-deg", type=float, default=360.0)
    parser.add_argument("--collision-geometry", choices=("circle", "ellipse"), default="circle")
    parser.add_argument("--collision-circle-radius-m", type=float, default=0.30)
    parser.add_argument("--collision-margin-m", type=float, default=0.05)
    parser.add_argument("--collision-termination-mode", default="circle")
    parser.add_argument("--qp-backend", default="active_set")
    parser.add_argument("--terminal-backup-mode", default="invariant_ellipse")
    parser.add_argument("--legacy-projection", action="store_true", default=True)
    parser.add_argument("--backup-margin", type=float, default=0.6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-trajectories", action="store_true")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)

    if args.map_source == "procedural":
        seeds = list(range(args.seed, args.seed + args.num_seeds * args.seed_stride, args.seed_stride))

        from SAFTER_src.high_level_laser_sac.procedural_maps import load_procedural_map_bank
        procedural_map_bank = load_procedural_map_bank(
            args.procedural_map_bank,
            expected_num_maps=args.expected_num_maps,
            expected_scenarios_per_map=args.expected_scenarios_per_map,
            expected_max_obstacles=None,
        )
        manifest = build_manifest(procedural_map_bank, seeds, args.cases_per_seed)
        print(
            f"Manifest: {len(manifest)} cases ({len(seeds)} seeds x {args.cases_per_seed} cases/seed "
            f"x {args.expected_num_maps} maps)"
        )
        if args.map_id is not None:
            manifest = [case for case in manifest if int(case["map_id"]) == args.map_id]
            print(f"Restricted to map_id={args.map_id}: {len(manifest)} cases")
            if not manifest:
                raise ValueError(f"No cases found for --map-id {args.map_id}.")

        from_seed = int(manifest[0]["reset_seed"])
        set_seed(from_seed)
        raw_env, env, cfg, _procedural_map_bank = build_procedural_environment(
            method="vanilla",
            num_envs=min(args.num_envs, len(manifest)),
            episode_duration_s=args.episode_duration_s,
            device=args.device,
            procedural_map_bank_path=args.procedural_map_bank,
            expected_num_maps=args.expected_num_maps,
            expected_scenarios_per_map=args.expected_scenarios_per_map,
            laser_num_rays=args.laser_num_rays,
            laser_fov_deg=args.laser_fov_deg,
            collision_geometry=args.collision_geometry,
            collision_circle_radius_m=args.collision_circle_radius_m,
            collision_margin_m=args.collision_margin_m,
            collision_termination_mode=args.collision_termination_mode,
            qp_backend=args.qp_backend,
            terminal_backup_mode=args.terminal_backup_mode,
            legacy_projection=args.legacy_projection,
            action_max_vx=ACTION_LIMITS[0],
            action_min_vx=ACTION_LIMITS[3],
            action_max_vy=ACTION_LIMITS[1],
            action_max_wz=ACTION_LIMITS[2],
            backup_margin=args.backup_margin,
            low_checkpoint=args.low_level_checkpoint,
            # Same LiDAR-noise rule as the structured map source.
            enable_lidar_noise=(args.nominal_source == "sac"),
        )
        try:
            result = evaluate(
                nominal_source=args.nominal_source,
                filter_type=args.filter_type,
                map_source="procedural",
                manifest=manifest,
                batch_size=min(args.num_envs, len(manifest)),
                raw_env=raw_env, env=env, cfg=cfg,
                vanilla_checkpoint=args.vanilla_checkpoint,
                log_trajectories=args.log_trajectories,
                device=args.device,
            )
        finally:
            try:
                raw_env.close()
            except Exception as exc:
                print(f"[WARNING] Failed to close env: {exc}", flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        method_name = result["method"]
        write_rows(root / f"{method_name}_episodes.csv", result["records"])
        if args.log_trajectories:
            write_rows(root / f"{method_name}_trajectory.csv", result["trajectory_records"])
        write_rows(root / f"{method_name}_summary.csv", [result["summary"]])
        print(f"\n[{method_name} SUMMARY]")
        for key, value in result["summary"].items():
            print(f"  {key}: {value}")
        print(f"\nSaved {len(result['records'])} episode rows under {root}/")
        return

    # --map-source structured: one environment per requested map name.
    if args.cases_per_seed <= 0 or args.num_seeds <= 0 or args.seed_stride <= 0:
        raise ValueError("--cases-per-seed, --num-seeds, and --seed-stride must be positive.")
    low_path = Path(args.low_level_checkpoint).expanduser()
    if not low_path.is_file():
        raise FileNotFoundError(f"Missing low-level checkpoint: {low_path}")

    all_summaries: list[dict[str, object]] = []
    for map_name in args.maps:
        print("\n" + "=" * 72)
        print(f"MAP: {map_name}")
        print("=" * 72)
        result = run_structured_map(map_name=map_name, args=args, device=args.device)
        method_name = result["method"]
        map_root = root / map_name
        map_root.mkdir(parents=True, exist_ok=True)
        write_rows(map_root / f"{method_name}_episodes.csv", result["records"])
        if args.log_trajectories:
            write_rows(map_root / f"{method_name}_trajectory.csv", result["trajectory_records"])
        write_rows(map_root / f"{method_name}_summary.csv", [result["summary"]])
        print(f"\n[{method_name} SUMMARY] map={map_name}")
        for key, value in result["summary"].items():
            print(f"  {key}: {value}")
        all_summaries.append(result["summary"])

    if len(args.maps) > 1:
        write_rows(root / f"{args.nominal_source}_nominal_{args.filter_type}_pooled_summary_by_map.csv", all_summaries)
    print(f"\nStructured evaluation completed. Results under {root}/")


if __name__ == "__main__":
    main()
