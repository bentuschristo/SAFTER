#!/usr/bin/env python3
"""Filter-swap evaluation: take the RAW nominal command from a trained
vanilla (no-safety-filter) SAC policy -- the nominal asymmetric-limit
high-speed checkpoint (max_vx=2.0, min_vx=-1.0, max_vy=1.0, max_wz=1.0,
360 deg/120-ray LiDAR, 10Hz high-level/50Hz low-level, handcrafted2) -- and
retrofit EITHER SAFTER (LaserZeroBackupLseCbfClosedForm) or the no-CD
bCBF-LSE-QP (LaserZeroBackupLseQp) onto it at evaluation time, on the three
current structured maps (low_shift/medium_shift/high_shift).

This answers: how much does each filter actually correct a policy that
never trained with a safety filter at all, compared to the obstacle-blind
analytic heuristic controller used in eval_heuristic_safter.py /
eval_heuristic_hardbox_nocd.py? The vanilla policy at least has LiDAR in
its observation (even though nothing forced it to use that information
safely), unlike the pure P-controller.

Reuses build_environment/load_actor/timed_deterministic_actor/
row_to_scenario/manifest machinery from the structured eval driver (same
as eval_heuristic_safter.py and evaluate_ablation_filter_swap.py), and the
retrofit-filter-construction pattern from evaluate_ablation_filter_swap.py
(extended here with a "bcbf_lse_qp" branch for the no-CD filter, which
that script doesn't support).

CSV schema matches the other eval_heuristic_*.py scripts; method name is
"vanilla_nominal_<filter-type>" (e.g. "vanilla_nominal_safter",
"vanilla_nominal_bcbf_lse_qp"). With --log-trajectories, also writes a
per-step "<method_name>_trajectory.csv" (position/yaw history against the
map, plus nominal/safe control-input history) alongside the episodes and
summary CSVs, same schema as eval_heuristic_safter.py's trajectory CSV.
"""
from __future__ import annotations

import argparse
import csv
import gc
import sys
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# warp.init() before anything else: see eval_heuristic_hardbox_nocd.py's own
# comment for the full explanation of why this ordering avoids a segfault.
import warp as wp  # noqa: E402

wp.init()

import mujoco  # noqa: F401,E402
import torch  # noqa: E402

from SAFTER_src.high_level_rl.laser_cbf_filters import (  # noqa: E402
    LaserZeroBackupLseCbfClosedForm,
    LaserZeroBackupLseQp,
)

from SAFTER_scripts.laser_unseen_eval.evaluate_laser_unseen_maps_structured import (  # noqa: E402
    UNSEEN_MAPS,
    STRUCTURED_SCENARIO_REGIONS,
    build_environment,
    generate_manifest,
    load_actor,
    make_map_arena,
    optional_mean,
    physical_clearance,
    row_to_scenario,
    sample_std,
    set_seed,
    timed_deterministic_actor,
    verify_compiled_physical_map,
    verify_parallel_lidar_geometry,
    warmup_actor_timing,
    write_rows,
)

TASK_ID = "Unitree-Go2-Safe-Navigation"
DEFAULT_LOW_CHECKPOINT = "logs/rsl_rl/go2_velocity/2026-06-19_13-24-10/model_10000.pt"
# The specific asymmetric high-speed vanilla checkpoint (360 FOV/120 rays,
# 10Hz/50Hz, handcrafted2, max_vx=2.0/min_vx=-1.0/max_vy=1.0/max_wz=1.0,
# target_vector_steps=80000) identified as "the nominal asymmetrical limit
# high-speed vanilla SAC policy."
DEFAULT_VANILLA_CHECKPOINT = (
    "logs/sac_laser/go2_safe_navigation_sac_laser_vanilla/"
    "2026-09-14_00-25-56_vanilla_10hz_seaspeed_seed0/model_step_80000.pt"
)

DEFAULT_MAPS = ("low_shift", "medium_shift", "high_shift")
AVAILABLE_MAPS = tuple(name for name in UNSEEN_MAPS if name in STRUCTURED_SCENARIO_REGIONS)
FILTER_TYPES = ("safter", "bcbf_lse_qp")


def build_retrofit_filter(filter_type: str, cfg, action_limits: tuple[float, float, float, float | None]):
    """Construct a filter object matching exactly what LaserSafeSacActor.__init__
    builds for that method -- pure analytic CBF math, no learned parameters,
    so it can be retrofitted onto ANY policy's nominal command. action_limits
    is (max_vx, max_vy, max_wz, min_vx)."""
    safety_cfg = cfg.safety
    max_vx, max_vy, max_wz, min_vx = action_limits
    kwargs = dict(
        d_safe=safety_cfg.d_safe,
        max_vx=max_vx,
        max_vy=max_vy,
        max_wz=max_wz,
        min_vx=min_vx,
        robot_length_m=safety_cfg.robot_length_m,
        robot_width_m=safety_cfg.robot_width_m,
        lidar_x_m=safety_cfg.lidar_x_m,
        lidar_y_m=safety_cfg.lidar_y_m,
        footprint_clearance_m=safety_cfg.footprint_clearance_m,
        use_elliptic_footprint=safety_cfg.use_elliptic_footprint,
        max_range=cfg.laser.max_range,
        no_return_tolerance_m=safety_cfg.no_return_tolerance_m,
        inactive_barrier_value=safety_cfg.inactive_barrier_value,
        denominator_epsilon=safety_cfg.denominator_epsilon,
        hard_clip_output=safety_cfg.hard_clip_output,
    )
    if filter_type == "safter":
        return LaserZeroBackupLseCbfClosedForm(
            backup_margin=safety_cfg.backup_margin,
            backup_horizon_s=safety_cfg.backup_horizon_s,
            backup_samples=safety_cfg.backup_samples,
            backup_rollout_substeps=safety_cfg.backup_rollout_substeps,
            hocbf_k0=safety_cfg.hocbf_k0,
            barrier_rate_scale=safety_cfg.barrier_rate_scale,
            kappa_backup=safety_cfg.kappa_backup,
            kappa_input=safety_cfg.kappa_input,
            kappa_outer=safety_cfg.kappa_outer,
            alpha_outer=safety_cfg.alpha_outer,
            control_dynamics_a=safety_cfg.control_dynamics_a,
            tracking_sigma=safety_cfg.tracking_sigma,
            integration_dt=safety_cfg.integration_dt,
            terminal_backup_mode=safety_cfg.terminal_backup_mode,
            terminal_geom_scale=safety_cfg.terminal_geom_scale,
            terminal_rec_scale=safety_cfg.terminal_rec_scale,
            legacy_projection=safety_cfg.legacy_projection,
            straight_through=False,
            **kwargs,
        )
    if filter_type == "bcbf_lse_qp":
        return LaserZeroBackupLseQp(
            alpha=safety_cfg.alpha_outer,
            backup_margin=safety_cfg.backup_margin,
            backup_samples=safety_cfg.backup_samples,
            kappa_backup=safety_cfg.kappa_backup,
            max_active_candidates=safety_cfg.qp_max_active_candidates,
            feasibility_tolerance=safety_cfg.qp_feasibility_tolerance,
            qp_regularization=safety_cfg.qp_regularization,
            straight_through=safety_cfg.qp_straight_through,
            qp_backend="active_set",
            **kwargs,
        )
    raise ValueError(f"Unsupported retrofit filter_type={filter_type!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter-type", choices=FILTER_TYPES, required=True)
    parser.add_argument(
        "--nominal-label", default="vanilla",
        help=(
            "Short label for the nominal-policy checkpoint's own trained "
            "method, used only in the output method name "
            "('<label>_nominal_<filter-type>') and printed messages -- "
            "does not affect how the checkpoint is loaded or validated. "
            "Default 'vanilla' assumes --vanilla-checkpoint really is a "
            "vanilla (no-filter) checkpoint; override when pointing this "
            "script at a checkpoint trained under a different method."
        ),
    )
    parser.add_argument("--maps", nargs="+", default=list(DEFAULT_MAPS), choices=list(AVAILABLE_MAPS))
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--seed-stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cases-per-seed", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=100)
    parser.add_argument("--episode-duration-s", type=float, default=90.0)
    parser.add_argument("--vanilla-checkpoint", default=DEFAULT_VANILLA_CHECKPOINT)
    parser.add_argument("--low-level-checkpoint", default=DEFAULT_LOW_CHECKPOINT)
    # Matches the vanilla checkpoint's own trained configuration exactly
    # (see logs/sac_laser/.../vanilla/2026-09-14_00-25-56_.../model_step_80000.pt's
    # saved config) -- load_actor() strictly checks laser/safety against this.
    parser.add_argument("--laser-num-rays", type=int, default=120)
    parser.add_argument("--laser-fov-deg", type=float, default=360.0)
    parser.add_argument("--collision-geometry", choices=("circle", "ellipse"), default="circle")
    parser.add_argument("--collision-circle-radius-m", type=float, default=0.30)
    parser.add_argument("--collision-margin-m", type=float, default=0.05)
    parser.add_argument("--collision-termination-mode", choices=("circle", "seanav_contact"), default="circle")
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
        "--output-root", default="logs/evaluation/vanilla_filterswap_structured",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


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
        enable_lidar_noise=True,
        hard_clip_output=True,
        collision_geometry=args.collision_geometry,
        collision_circle_radius_m=args.collision_circle_radius_m,
        collision_margin_m=args.collision_margin_m,
        collision_termination_mode=args.collision_termination_mode,
        action_max_vx=2.0,
        action_min_vx=-1.0,
        action_max_vy=1.0,
        action_max_wz=1.0,
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


def evaluate_map(
    *, method_name: str, filter_type: str, map_name: str,
    pooled_manifest: list[dict[str, object]], args: argparse.Namespace, device: str,
) -> dict[str, object]:
    batch_size = min(int(args.num_envs), len(pooled_manifest))
    set_seed(int(pooled_manifest[0]["reset_seed"]))
    raw_env, low_env, env, cfg, dynamic_runtime = build_environment(
        task_id=TASK_ID,
        low_checkpoint=args.low_level_checkpoint,
        method="vanilla",
        episode_duration_s=args.episode_duration_s,
        device=device,
        map_name=map_name,
        laser_num_rays=args.laser_num_rays,
        laser_fov_deg=args.laser_fov_deg,
        enable_lidar_noise=True,
        hard_clip_output=True,
        collision_geometry=args.collision_geometry,
        collision_circle_radius_m=args.collision_circle_radius_m,
        collision_margin_m=args.collision_margin_m,
        collision_termination_mode=args.collision_termination_mode,
        num_envs=batch_size,
        action_max_vx=2.0,
        action_min_vx=-1.0,
        action_max_vy=1.0,
        action_max_wz=1.0,
        backup_margin=args.backup_margin,
        terminal_backup_mode="invariant_ellipse",
        terminal_geom_scale=1.0,
        terminal_rec_scale=1.0,
        legacy_projection=True,
    )
    records: list[dict[str, object]] = []
    trajectory_records: list[dict[str, object]] = []
    try:
        verify_compiled_physical_map(raw_env, env, map_name)

        actor = load_actor(env=env, cfg=cfg, checkpoint_path=args.vanilla_checkpoint, device=device)
        warmup_actor_timing(actor, env)
        retrofit_filter = build_retrofit_filter(
            filter_type, cfg, (2.0, 1.0, 1.0, -1.0),
        )
        dt = float(env.navigation_cfg.high_level_dt)

        for batch_start in range(0, len(pooled_manifest), batch_size):
            real_cases = pooled_manifest[batch_start: batch_start + batch_size]
            padded_cases = list(real_cases)
            while len(padded_cases) < batch_size:
                padded_cases.append(real_cases[-1])
            env.set_fixed_scenarios([row_to_scenario(case) for case in padded_cases])
            verify_parallel_lidar_geometry(raw_env, env, map_name)
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
            returns = [0.0] * active_count
            step_counts = [0] * active_count

            with torch.no_grad():
                while not all(finished):
                    # Only needed for the trajectory CSV's yaw_rad column; the
                    # nominal command itself comes from actor_obs, not from a
                    # manual position/yaw query.
                    _, pre_yaws = env.safety_context()
                    actor_obs = env.actor_tensor(obs)
                    nominal, _safe, safety, _, _, _ = timed_deterministic_actor(actor, actor_obs)

                    ranges = actor._ranges(actor_obs)
                    angles = actor.ray_angles.to(device=actor_obs.device, dtype=actor_obs.dtype)
                    if filter_type == "safter":
                        retrofit = retrofit_filter.filter(
                            ranges, angles, actor_obs[:, 5:8], nominal, actor_obs[:, -4:-1], actor_obs[:, -1:],
                        )
                    else:
                        retrofit = retrofit_filter.filter(ranges, angles, None, nominal, None, None)
                    applied = retrofit.command
                    if actor.hard_clip_output:
                        applied = torch.clamp(applied, actor.action_low, actor.action_high)

                    obs, reward, done, extras = env.step_safe(nominal, applied, retrofit)

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
                            torch.linalg.norm(applied[env_id] - nominal[env_id]).detach().cpu()
                        )
                        correction_sums[env_id] += correction_value
                        filter_active = correction_value > 1e-8
                        active_steps[env_id] += int(filter_active)

                        trajectory_row = None
                        if args.log_trajectories:
                            trajectory_row = {
                                "method": method_name, "map_name": map_name,
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
                                "safe_vx_mps": float(applied[env_id, 0].detach().cpu()),
                                "safe_vy_mps": float(applied[env_id, 1].detach().cpu()),
                                "safe_wz_radps": float(applied[env_id, 2].detach().cpu()),
                                "command_correction_norm": correction_value,
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
                                "method": method_name, "map_name": map_name,
                                "evaluation_seed": int(case["evaluation_seed"]),
                                "case_id": int(case["case_id"]), "reset_seed": int(case["reset_seed"]),
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
                            })
                            finished[env_id] = True
                            completed_total = batch_start + sum(finished)
                            if args.print_every > 0 and completed_total % args.print_every == 0:
                                print(
                                    f"[{method_name}] map={map_name:10s} "
                                    f"completed={completed_total:3d}/{len(pooled_manifest):3d} "
                                    f"case={int(case['case_id']):3d} outcome={outcome:13s} "
                                    f"return={final_return:+8.2f} path={path_lengths[env_id]:7.3f} m "
                                    f"time={duration:6.2f} s clearance={minimum_clearances[env_id]:+.3f} m",
                                    flush=True,
                                )
                        if trajectory_row is not None:
                            trajectory_records.append(trajectory_row)

        trajectory_records.sort(
            key=lambda row: (int(row["evaluation_seed"]), int(row["case_id"]), int(row["step"]))
        )
        if not records:
            raise RuntimeError(f"No completed episodes recorded for map {map_name!r}.")
        successes = [row for row in records if int(row["success"]) == 1]
        summary = {
            "method": method_name, "map_name": map_name, "cases": len(records),
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


def main() -> None:
    args = parse_args()
    method_name = f"{args.nominal_label}_nominal_{args.filter_type}"
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation device: {device}")

    evaluation_seeds = [int(args.seed) + i * int(args.seed_stride) for i in range(args.num_seeds)]
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)

    all_summaries: list[dict[str, object]] = []
    for map_name in args.maps:
        print("\n" + "=" * 72)
        print(f"MAP: {map_name} | filter={args.filter_type} | seeds: {evaluation_seeds}")
        print("=" * 72)
        pooled_manifest: list[dict[str, object]] = []
        for evaluation_seed in evaluation_seeds:
            map_dir = root / f"seed_{evaluation_seed}" / map_name
            map_dir.mkdir(parents=True, exist_ok=True)
            manifest = load_or_generate_manifest(
                map_name=map_name, evaluation_seed=evaluation_seed,
                map_dir=map_dir, args=args, device=device,
            )
            pooled_manifest.extend(manifest)

        result = evaluate_map(
            method_name=method_name, filter_type=args.filter_type, map_name=map_name,
            pooled_manifest=pooled_manifest, args=args, device=device,
        )
        records = result["records"]
        trajectory_records = result["trajectory_records"]
        summary = result["summary"]

        method_dir = root / f"seed_{evaluation_seeds[0]}" / map_name / method_name
        method_dir.mkdir(parents=True, exist_ok=True)
        write_rows(method_dir / f"{method_name}_episodes.csv", records)
        write_rows(method_dir / f"{method_name}_summary.csv", [summary])
        if args.log_trajectories and trajectory_records:
            write_rows(method_dir / f"{method_name}_trajectory.csv", trajectory_records)

        print(f"\n[{method_name} POOLED SUMMARY] map={map_name}")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        all_summaries.append(summary)

    write_rows(root / f"{method_name}_pooled_summary_by_map.csv", all_summaries)
    print("\n" + "=" * 72)
    print(f"{method_name} structured evaluation completed")
    print(f"Results: {root}")
    print("=" * 72)


if __name__ == "__main__":
    main()
