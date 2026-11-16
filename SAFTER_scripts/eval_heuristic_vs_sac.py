#!/usr/bin/env python3
"""Evaluate the heuristic (P-controller) nominal policy against a SAC-based
nominal policy, both passed through the same safety filter, on a bank of
procedural scenario maps.

--nominal-source {heuristic, sac} selects the nominal policy:
    heuristic: BatchedGoToGoalController, an obstacle-blind P controller.
    sac:       the raw nominal command from a trained (no-filter) SAC
               checkpoint passed via --vanilla-checkpoint.
--filter-type selects the safety filter, built via
    eval_vanilla_filterswap.py's build_retrofit_filter() so both nominal
    sources see an identical filter construction. Any filter type that
    function supports works here.
--procedural-map-bank / --map-id select the scenario map bank and
    optionally restrict evaluation to a single map within it.

Logs per-step clip-vs-barrier-certificate diagnostics (clip_delta,
clip_changed, residual_exec, projection_feasible) and reports both
per-episode and pooled (step-weighted) rates: clip_triggered_rate,
barrier_violated_given_clip_rate, barrier_violated_given_no_clip_rate,
barrier_violated_given_clip_feasible_rate, and clip_delta_norm_max.
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import math
import sys
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
_eval_module = _load_sibling_module(
    "_evaluate_laser_unseen_maps_structured",
    "SAFTER_scripts/laser_unseen_eval/evaluate_laser_unseen_maps_structured.py",
)
_filterswap_module = _load_sibling_module(
    "_eval_vanilla_filterswap", "SAFTER_scripts/eval_vanilla_filterswap.py"
)
_hardbox_module = _load_sibling_module(
    "_eval_heuristic_hardbox_nocd", "SAFTER_scripts/eval_heuristic_hardbox_nocd.py"
)

build_environment = _training_maps_module.build_environment
build_manifest = _training_maps_module.build_manifest

load_actor = _eval_module.load_actor
timed_deterministic_actor = _eval_module.timed_deterministic_actor
warmup_actor_timing = _eval_module.warmup_actor_timing
write_rows = _eval_module.write_rows
mean_stats = _eval_module.mean_stats
proportion_stats = _eval_module.proportion_stats
sample_std = _eval_module.sample_std
optional_mean = _eval_module.optional_mean
physical_clearance = _eval_module.physical_clearance
BARRIER_VIOLATION_EPS = _eval_module.BARRIER_VIOLATION_EPS

build_retrofit_filter = _filterswap_module.build_retrofit_filter
FILTER_TYPES = _filterswap_module.FILTER_TYPES

BatchedGoToGoalController = _hardbox_module.BatchedGoToGoalController

TASK_ID = "Unitree-Go2-Safe-Navigation"
ACTION_LIMITS = (2.0, 1.0, 1.0, -1.0)  # (max_vx, max_vy, max_wz, min_vx)


def evaluate(
    *,
    nominal_source: str,
    filter_type: str,
    manifest: list[dict],
    batch_size: int,
    vanilla_checkpoint: str | None,
    log_trajectories: bool,
    **env_kwargs,
) -> dict[str, object]:
    method_name = f"{nominal_source}_nominal_{filter_type}"
    from_seed = int(manifest[0]["reset_seed"])
    _eval_module.set_seed(from_seed)

    raw_env, env, cfg, _procedural_map_bank = build_environment(
        method="vanilla", num_envs=batch_size, **env_kwargs
    )
    device = env_kwargs["device"]
    dt = float(env.navigation_cfg.high_level_dt)

    records: list[dict[str, object]] = []
    trajectory_records: list[dict[str, object]] = []
    # Pooled (not per-episode-averaged) clip-vs-barrier-certificate counters
    # -- see evaluate_laser_unseen_maps_structured.evaluate_method_vectorized
    # for the full rationale.
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

    try:
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
            scenario_ids = torch.tensor(
                [int(case["scenario_id"]) for case in padded_cases], dtype=torch.long, device=device
            )
            env.set_fixed_procedural_scenarios(scenario_ids)
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
                        previous_nominal_command = env.previous_nominal_command
                        nominal_history_valid = env.nominal_history_valid
                        ranges = env.laser_ranges()
                        angles = env.ray_angles
                    else:
                        actor_obs = env.actor_tensor(obs)
                        nominal, _safe, _safety, _, _, _ = timed_deterministic_actor(actor, actor_obs)
                        current_command = actor_obs[:, 5:8]
                        previous_nominal_command = actor_obs[:, -4:-1]
                        nominal_history_valid = actor_obs[:, -1:]
                        ranges = actor._ranges(actor_obs)
                        angles = actor.ray_angles.to(device=actor_obs.device, dtype=actor_obs.dtype)

                    # Every filter built by build_retrofit_filter() accepts this same
                    # signature; filters without control dynamics simply ignore the
                    # current/previous-command arguments.
                    safety = retrofit_filter.filter(
                        ranges, angles, current_command, nominal,
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
                                "map_id": int(case["map_id"]),
                                "evaluation_seed": int(case["evaluation_seed"]),
                                "case_id": int(case["case_id"]),
                                "scenario_id": int(case["scenario_id"]),
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
                                "map_id": int(case["map_id"]),
                                "evaluation_seed": int(case["evaluation_seed"]),
                                "case_id": int(case["case_id"]),
                                "scenario_id": int(case["scenario_id"]),
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
    finally:
        try:
            raw_env.close()
        except Exception as exc:
            print(f"[WARNING] Failed to close env for {method_name}: {exc}", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nominal-source", choices=("heuristic", "sac"), required=True)
    parser.add_argument("--filter-type", choices=FILTER_TYPES, required=True)
    parser.add_argument("--vanilla-checkpoint", default=None)
    parser.add_argument(
        "--low-level-checkpoint", default="logs/rsl_rl/go2_velocity/2026-06-19_13-24-10/model_10000.pt"
    )
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

    env_kwargs = dict(
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
    )

    result = evaluate(
        nominal_source=args.nominal_source,
        filter_type=args.filter_type,
        manifest=manifest,
        batch_size=min(args.num_envs, len(manifest)),
        vanilla_checkpoint=args.vanilla_checkpoint,
        log_trajectories=args.log_trajectories,
        **env_kwargs,
    )

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    method_name = result["method"]
    write_rows(root / f"{method_name}_episodes.csv", result["records"])
    if args.log_trajectories:
        write_rows(root / f"{method_name}_trajectory.csv", result["trajectory_records"])
    write_rows(root / f"{method_name}_summary.csv", [result["summary"]])
    print(f"\n[{method_name} SUMMARY]")
    for key, value in result["summary"].items():
        print(f"  {key}: {value}")
    print(f"\nSaved {len(result['records'])} episode rows under {root}/")


if __name__ == "__main__":
    main()
