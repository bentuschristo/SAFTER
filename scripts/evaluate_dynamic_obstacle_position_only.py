#!/usr/bin/env python3
"""Position-only dynamic-obstacle evaluation for unseen Go2 navigation maps.

This is a robustness / model-mismatch experiment:
  * one obstacle center is moved according to a known trajectory;
  * the actor observation and CBF/bCBF obstacle buffers are updated with the
    current obstacle position;
  * obstacle velocity is logged but not used by the static CBF/bCBF formulas;
  * bCBF rollouts freeze the obstacle at its current position over the backup
    horizon.

Therefore this script does NOT claim dynamic-obstacle safety guarantees. It
tests how the already-trained static-obstacle filters react to a slowly moving
obstacle via repeated position updates.

Run from repo root:
  python scripts/evaluate_dynamic_obstacle_position_only.py \
    Unitree-Go2-Safe-Navigation \
    --cases-per-map 10 \
    --maps shifted corridor clustered slalom bottleneck \
    --speed 0.15 \
    --jitter 0.25 \
    --log-trajectories \
    --low-level-checkpoint "$LOW_LEVEL_CKPT" \
    --no-cbf-checkpoint "$NOCBF_CKPT" \
    --cbf-checkpoint "$CBF_CKPT" \
    --bcbf-checkpoint "$BCBF_CKPT"
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from types import SimpleNamespace
from typing import Any

import torch
import mujoco

# Reuse the static unseen-map evaluator's task registration, map definitions,
# environment construction, and actor construction utilities.
import evaluate_unseen_maps_all_methods as base


MAP_ORDER = ["shifted", "corridor", "clustered", "slalom", "bottleneck"]
METHODS = ["no_cbf", "cbf_rl", "bcbf_rl"]


@dataclass(frozen=True)
class DynamicMapSpec:
    obstacle_id: int
    axis: str
    target_value: float
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]


DYNAMIC_SPECS: dict[str, DynamicMapSpec] = {
    "bottleneck": DynamicMapSpec(
        obstacle_id=3,
        axis="y",
        target_value=0.0,
        start_xy=(2.0, 4.0),
        goal_xy=(2.0, -4.0),
    ),
    "clustered": DynamicMapSpec(
        obstacle_id=7,
        axis="y",
        target_value=2.0,
        start_xy=(6.0, 4.0),
        goal_xy=(-4.0, -6.0),
    ),
    "corridor": DynamicMapSpec(
        obstacle_id=6,
        axis="y",
        target_value=0.0,
        start_xy=(-6.0, 0.0),
        goal_xy=(0.0, 0.0),
    ),
    "shifted": DynamicMapSpec(
        obstacle_id=2,
        axis="x",
        target_value=-2.0,
        start_xy=(0.0, 0.0),
        goal_xy=(-2.0, -6.0),
    ),
    "slalom": DynamicMapSpec(
        obstacle_id=4,
        axis="y",
        target_value=0.0,
        start_xy=(-3.0, 5.0),
        goal_xy=(2.0, -6.0),
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate static SAC-CBF/bCBF policies against one moving obstacle per unseen map."
    )
    p.add_argument("task_id", nargs="?", default="Unitree-Go2-Safe-Navigation")
    p.add_argument("--cases-per-map", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episode-duration-s", type=float, default=90.0)
    p.add_argument("--print-every", type=int, default=1)
    p.add_argument("--device", default=None)

    p.add_argument("--low-level-checkpoint", default=base.DEFAULT_LOW)
    p.add_argument("--no-cbf-checkpoint", default=base.DEFAULT_NOCBF)
    p.add_argument("--cbf-checkpoint", default=base.DEFAULT_CBF)
    p.add_argument("--bcbf-checkpoint", default=base.DEFAULT_BCBF)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    p.add_argument("--maps", nargs="+", choices=MAP_ORDER, default=MAP_ORDER)

    p.add_argument("--speed", type=float, default=0.15, help="Obstacle speed in m/s along the selected segment.")
    p.add_argument("--jitter", type=float, default=0.25, help="Uniform start/goal jitter radius per x/y component.")
    p.add_argument("--random-motion-phase", action="store_true", help="Randomize the initial obstacle phase per case.")
    p.add_argument("--min-start-goal-clearance", type=float, default=0.25)
    p.add_argument("--max-sampling-attempts", type=int, default=500)
    p.add_argument("--output-root", default="logs/evaluation/dynamic_obstacle_position_only")
    p.add_argument("--log-trajectories", action="store_true")
    p.add_argument("--overwrite-manifest", action="store_true")
    return p.parse_args()


def yaw_toward(start: tuple[float, float], goal: tuple[float, float]) -> float:
    return math.atan2(goal[1] - start[1], goal[0] - start[0])


def sample_std(values: list[float]) -> float:
    return stdev(values) if len(values) >= 2 else 0.0


def optional_mean(values: list[float]) -> float:
    return mean(values) if values else math.nan


def moving_obstacle_position_velocity(
    *,
    base_xy: tuple[float, float],
    spec: DynamicMapSpec,
    time_s: float,
    speed: float,
    phase_s: float = 0.0,
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Triangular back-and-forth motion from the original center to target."""
    if speed <= 0.0:
        return base_xy, (0.0, 0.0), 0.0

    x0, y0 = base_xy
    if spec.axis == "x":
        start_value = x0
        fixed_value = y0
    elif spec.axis == "y":
        start_value = y0
        fixed_value = x0
    else:
        raise ValueError(f"Unsupported axis: {spec.axis}")

    target_value = float(spec.target_value)
    segment = target_value - start_value
    distance = abs(segment)

    if distance <= 1.0e-9:
        return base_xy, (0.0, 0.0), 0.0

    half_period = distance / speed
    period = 2.0 * half_period
    tau = (time_s + phase_s) % period

    direction_sign = 1.0 if segment >= 0.0 else -1.0

    if tau <= half_period:
        s = tau / half_period
        value = start_value + s * segment
        velocity_value = direction_sign * speed
        phase = s
    else:
        s = (tau - half_period) / half_period
        value = target_value - s * segment
        velocity_value = -direction_sign * speed
        phase = 1.0 - s

    if spec.axis == "x":
        return (float(value), float(fixed_value)), (float(velocity_value), 0.0), float(phase)
    return (float(fixed_value), float(value)), (0.0, float(velocity_value)), float(phase)


def set_dynamic_obstacle_center(env, actor, obstacle_id: int, xy: tuple[float, float]) -> None:
    """Update the analytical obstacle tensors used by observation, collision, CBF, and bCBF."""
    with torch.no_grad():
        new_xy = torch.tensor(xy, device=env.device, dtype=env.obstacle_centers.dtype)
        env.obstacle_centers[:, obstacle_id, :] = new_xy

        if actor is not None and hasattr(actor, "obstacle_centers"):
            actor.obstacle_centers[:, obstacle_id, :] = new_xy.to(actor.obstacle_centers.device)


def best_effort_move_mujoco_body(raw_env, obstacle_id: int, xy: tuple[float, float]) -> None:
    """Best-effort visual/physical body position update.

    The evaluation's authoritative obstacle geometry is env.obstacle_centers.
    This function is only a best-effort attempt to keep compiled MuJoCo body
    positions synchronized for visualizers. Depending on the MJLab backend,
    mutating mj_model.body_pos may not affect an already-compiled GPU model.
    """
    try:
        model = raw_env.sim.mj_model
        body_name = f"obstacle_{obstacle_id}"
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            return
        model.body_pos[body_id, 0] = float(xy[0])
        model.body_pos[body_id, 1] = float(xy[1])
    except Exception:
        return


def dynamic_clearance(env, position: torch.Tensor) -> float:
    cfg = env.navigation_cfg
    robot_radius = float(getattr(cfg.arena, "robot_radius", 0.30))
    p = position.reshape(1, 2)

    d_obs = torch.linalg.norm(p[:, None, :] - env.obstacle_centers[:1], dim=-1)
    obstacle_clearance = d_obs - env.obstacle_radii[:1] - robot_radius
    obstacle_clearance = torch.where(
        env.obstacle_mask[:1],
        obstacle_clearance,
        torch.full_like(obstacle_clearance, float("inf")),
    )

    x = p[0, 0]
    y = p[0, 1]
    wall_clearance = torch.stack(
        (
            x - cfg.arena.x_min - robot_radius,
            cfg.arena.x_max - x - robot_radius,
            y - cfg.arena.y_min - robot_radius,
            cfg.arena.y_max - y - robot_radius,
        )
    )
    return float(torch.minimum(obstacle_clearance.min(), wall_clearance.min()).detach().cpu())


def dynamic_obstacle_clearance_for_point(
    point: tuple[float, float],
    *,
    map_name: str,
    moving_centers: list[tuple[float, float]],
) -> float:
    arena = base.make_map_arena(map_name)
    robot_radius = float(getattr(arena, "robot_radius", 0.30))
    px, py = point

    values = []
    for i, obs in enumerate(arena.obstacles):
        centers_to_check = moving_centers if i == DYNAMIC_SPECS[map_name].obstacle_id else [(obs.x, obs.y)]
        for cx, cy in centers_to_check:
            values.append(math.hypot(px - cx, py - cy) - float(obs.radius) - robot_radius)

    values += [
        px - arena.x_min - robot_radius,
        arena.x_max - px - robot_radius,
        py - arena.y_min - robot_radius,
        arena.y_max - py - robot_radius,
    ]
    return float(min(values))


def generate_dynamic_manifest(
    *,
    map_name: str,
    count: int,
    seed: int,
    jitter: float,
    speed: float,
    random_motion_phase: bool,
    min_clearance: float,
    max_attempts: int,
    output_path: Path,
) -> list[dict[str, object]]:
    spec = DYNAMIC_SPECS[map_name]
    arena = base.make_map_arena(map_name)
    base_obs = arena.obstacles[spec.obstacle_id]
    base_xy = (float(base_obs.x), float(base_obs.y))

    # Validate start and goal against the full moving segment.
    samples = []
    distance = abs(float(spec.target_value) - (base_xy[0] if spec.axis == "x" else base_xy[1]))
    half_period = distance / max(speed, 1.0e-9) if distance > 0 else 0.0
    for t in torch.linspace(0.0, max(half_period, 1.0e-6), 31).tolist():
        xy, _, _ = moving_obstacle_position_velocity(
            base_xy=base_xy,
            spec=spec,
            time_s=float(t),
            speed=max(speed, 1.0e-9),
        )
        samples.append(xy)

    rng = torch.Generator().manual_seed(seed)
    rows: list[dict[str, object]] = []

    for case_id in range(count):
        for attempt in range(max_attempts):
            start_dxdy = (2.0 * torch.rand(2, generator=rng) - 1.0) * jitter
            goal_dxdy = (2.0 * torch.rand(2, generator=rng) - 1.0) * jitter

            start = (
                float(spec.start_xy[0] + start_dxdy[0]),
                float(spec.start_xy[1] + start_dxdy[1]),
            )
            goal = (
                float(spec.goal_xy[0] + goal_dxdy[0]),
                float(spec.goal_xy[1] + goal_dxdy[1]),
            )

            start_clearance = dynamic_obstacle_clearance_for_point(
                start, map_name=map_name, moving_centers=samples
            )
            goal_clearance = dynamic_obstacle_clearance_for_point(
                goal, map_name=map_name, moving_centers=samples
            )

            if min(start_clearance, goal_clearance) >= min_clearance:
                reset_seed = int(seed + case_id)
                phase_s = 0.0
                if random_motion_phase and half_period > 0.0:
                    phase_s = float(torch.rand((), generator=rng) * (2.0 * half_period))

                rows.append(
                    {
                        "map_name": map_name,
                        "case_id": case_id,
                        "reset_seed": reset_seed,
                        "start_x": start[0],
                        "start_y": start[1],
                        "start_yaw": yaw_toward(start, goal),
                        "goal_x": goal[0],
                        "goal_y": goal[1],
                        "start_clearance_m": start_clearance,
                        "goal_clearance_m": goal_clearance,
                        "dynamic_obstacle_id": spec.obstacle_id,
                        "dynamic_axis": spec.axis,
                        "dynamic_target_value": spec.target_value,
                        "dynamic_base_x": base_xy[0],
                        "dynamic_base_y": base_xy[1],
                        "dynamic_speed_mps": speed,
                        "dynamic_phase_s": phase_s,
                    }
                )
                break

        else:
            raise RuntimeError(
                f"Failed to sample valid dynamic manifest case {case_id} for {map_name}."
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DYNAMIC MANIFEST] map={map_name} wrote {len(rows)} cases to {output_path}")
    return rows


def row_to_scenario(row: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        name=f"dynamic_{str(row['map_name'])}_{int(row['case_id']):03d}",
        start_xy=(float(row["start_x"]), float(row["start_y"])),
        start_yaw=float(row["start_yaw"]),
        goal_xy=(float(row["goal_x"]), float(row["goal_y"])),
    )


def evaluate_dynamic_method(
    *,
    method: str,
    task_id: str,
    low_checkpoint: str,
    high_checkpoint: str,
    manifest: list[dict[str, object]],
    episode_duration_s: float,
    device: str,
    print_every: int,
    output_dir: Path,
    map_name: str,
    speed: float,
    log_trajectories: bool,
) -> tuple[Path, Path]:
    module = base.bcbf_eval if method == "bcbf_rl" else base.sac_eval
    base_seed = int(manifest[0]["reset_seed"]) if manifest else 0
    base.set_seed(base_seed)

    raw_env, low_env, env, config = base.build_environment(
        module=module,
        task_id=task_id,
        low_checkpoint=low_checkpoint,
        episode_duration_s=episode_duration_s,
        device=device,
        map_name=map_name,
    )
    actor = base.build_actor(
        method=method,
        module=module,
        env=env,
        config=config,
        checkpoint_path=high_checkpoint,
        device=device,
    )

    records: list[dict[str, object]] = []
    trajectory_records: list[dict[str, object]] = []

    try:
        for index, case in enumerate(manifest):
            case_id = int(case["case_id"])
            spec = DYNAMIC_SPECS[map_name]
            base_xy = (float(case["dynamic_base_x"]), float(case["dynamic_base_y"]))
            phase_s = float(case.get("dynamic_phase_s", 0.0))

            base.set_seed(int(case["reset_seed"]))
            scenario = row_to_scenario(case)
            env.set_fixed_scenario(scenario)

            # Reset dynamic obstacle to its phase-0 position before first observation.
            xy0, v0, phase0 = moving_obstacle_position_velocity(
                base_xy=base_xy,
                spec=spec,
                time_s=0.0,
                speed=speed,
                phase_s=phase_s,
            )
            set_dynamic_obstacle_center(env, actor, spec.obstacle_id, xy0)
            best_effort_move_mujoco_body(raw_env, spec.obstacle_id, xy0)

            obs = env.actor_tensor(env.get_observations())
            position, _ = env.safety_context()
            previous_position = position[0].detach().clone()
            path_length = 0.0
            minimum_clearance = dynamic_clearance(env, previous_position)
            active_steps = 0
            correction_sum = 0.0
            step_count = 0
            outcome = "other_failure"
            success = False
            collision = False
            timeout = False
            fallen = False
            outside = False
            duration_s = 0.0

            with torch.no_grad():
                while True:
                    t_now = step_count * env.navigation_cfg.high_level_dt
                    xy_now, v_now, motion_phase = moving_obstacle_position_velocity(
                        base_xy=base_xy,
                        spec=spec,
                        time_s=t_now,
                        speed=speed,
                        phase_s=phase_s,
                    )
                    set_dynamic_obstacle_center(env, actor, spec.obstacle_id, xy_now)
                    best_effort_move_mujoco_body(raw_env, spec.obstacle_id, xy_now)

                    actor_obs = env.actor_tensor(env.get_observations())
                    position, yaw = env.safety_context()

                    out = actor.deterministic(actor_obs, position, yaw)
                    nominal = out.nominal_command
                    safe = out.safe_command
                    safety = out.safety

                    next_obs, _, done, extras = env.step_safe(nominal, safe, safety)
                    step_count += 1
                    t_next = step_count * env.navigation_cfg.high_level_dt

                    xy_next, v_next, motion_phase_next = moving_obstacle_position_velocity(
                        base_xy=base_xy,
                        spec=spec,
                        time_s=t_next,
                        speed=speed,
                        phase_s=phase_s,
                    )
                    set_dynamic_obstacle_center(env, actor, spec.obstacle_id, xy_next)
                    best_effort_move_mujoco_body(raw_env, spec.obstacle_id, xy_next)

                    if bool(done[0]):
                        terminal_positions = extras.get("completed_terminal_positions")
                        if terminal_positions is not None and terminal_positions.numel() > 0:
                            endpoint = terminal_positions[0]
                        else:
                            endpoint = env.safety_context()[0][0]
                    else:
                        endpoint = env.safety_context()[0][0]

                    step_start_position = previous_position.detach().clone()
                    path_length += float(torch.linalg.norm(endpoint - previous_position).detach().cpu())
                    previous_position = endpoint.detach().clone()

                    clear = dynamic_clearance(env, endpoint)
                    minimum_clearance = min(minimum_clearance, clear)

                    dynamic_collision = clear <= 0.0
                    correction = torch.linalg.norm(safe[0] - nominal[0])
                    correction_sum += float(correction.detach().cpu())

                    if method == "no_cbf":
                        filter_active = False
                    elif safety is not None and hasattr(safety, "active"):
                        filter_active = bool(safety.active[0])
                    else:
                        filter_active = float(correction.detach().cpu()) > 1.0e-6
                    if filter_active:
                        active_steps += 1

                    if log_trajectories:
                        obstacle_xy = xy_next
                        obstacle_v = v_next
                        rel_x = float(endpoint[0].detach().cpu()) - obstacle_xy[0]
                        rel_y = float(endpoint[1].detach().cpu()) - obstacle_xy[1]
                        trajectory_records.append(
                            {
                                "method": method,
                                "map_name": map_name,
                                "case_id": case_id,
                                "reset_seed": int(case["reset_seed"]),
                                "step": step_count,
                                "time_s": t_next,
                                "pre_x_m": float(step_start_position[0].detach().cpu()),
                                "pre_y_m": float(step_start_position[1].detach().cpu()),
                                "post_x_m": float(endpoint[0].detach().cpu()),
                                "post_y_m": float(endpoint[1].detach().cpu()),
                                "yaw_rad": float(yaw[0].detach().cpu()),
                                "start_x_m": float(case["start_x"]),
                                "start_y_m": float(case["start_y"]),
                                "goal_x_m": float(case["goal_x"]),
                                "goal_y_m": float(case["goal_y"]),
                                "nominal_vx_mps": float(nominal[0, 0].detach().cpu()),
                                "nominal_vy_mps": float(nominal[0, 1].detach().cpu()),
                                "nominal_wz_radps": float(nominal[0, 2].detach().cpu()),
                                "safe_vx_mps": float(safe[0, 0].detach().cpu()),
                                "safe_vy_mps": float(safe[0, 1].detach().cpu()),
                                "safe_wz_radps": float(safe[0, 2].detach().cpu()),
                                "command_correction_norm": float(correction.detach().cpu()),
                                "filter_active": int(filter_active),
                                "dynamic_obstacle_id": spec.obstacle_id,
                                "obstacle_x_m": obstacle_xy[0],
                                "obstacle_y_m": obstacle_xy[1],
                                "obstacle_vx_mps": obstacle_v[0],
                                "obstacle_vy_mps": obstacle_v[1],
                                "obstacle_motion_phase": motion_phase_next,
                                "relative_x_m": rel_x,
                                "relative_y_m": rel_y,
                                "relative_distance_m": math.hypot(rel_x, rel_y),
                                "dynamic_clearance_m": clear,
                                "cumulative_path_length_m": path_length,
                                "done": int(bool(done[0]) or dynamic_collision),
                                "outcome": "",
                                "success": 0,
                                "collision": 0,
                                "timeout": 0,
                                "fallen": 0,
                                "outside": 0,
                            }
                        )

                    if bool(done[0]) or dynamic_collision:
                        success = bool(extras.get("completed_success", torch.zeros(1, dtype=torch.bool, device=env.device))[0]) if bool(done[0]) else False
                        env_collision = bool(extras.get("completed_collision", torch.zeros(1, dtype=torch.bool, device=env.device))[0]) if bool(done[0]) else False
                        timeout = bool(extras.get("completed_timeouts", torch.zeros(1, dtype=torch.bool, device=env.device))[0]) if bool(done[0]) else False
                        fallen = bool(extras.get("completed_fallen", torch.zeros(1, dtype=torch.bool, device=env.device))[0]) if "completed_fallen" in extras and bool(done[0]) else False
                        outside = bool(extras.get("completed_outside", torch.zeros(1, dtype=torch.bool, device=env.device))[0]) if "completed_outside" in extras and bool(done[0]) else False
                        collision = bool(env_collision or dynamic_collision)

                        if bool(done[0]) and "completed_lengths" in extras:
                            duration_s = int(extras["completed_lengths"][0]) * env.navigation_cfg.high_level_dt
                        else:
                            duration_s = t_next

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

                        if log_trajectories and trajectory_records:
                            trajectory_records[-1].update(
                                {
                                    "outcome": outcome,
                                    "success": int(success),
                                    "collision": int(collision),
                                    "timeout": int(timeout),
                                    "fallen": int(fallen),
                                    "outside": int(outside),
                                }
                            )
                        break

                    obs = next_obs

            record = {
                "method": method,
                "map_name": map_name,
                "case_id": case_id,
                "reset_seed": int(case["reset_seed"]),
                "start_x": float(case["start_x"]),
                "start_y": float(case["start_y"]),
                "start_yaw": float(case["start_yaw"]),
                "goal_x": float(case["goal_x"]),
                "goal_y": float(case["goal_y"]),
                "dynamic_obstacle_id": spec.obstacle_id,
                "dynamic_axis": spec.axis,
                "dynamic_target_value": spec.target_value,
                "dynamic_base_x": base_xy[0],
                "dynamic_base_y": base_xy[1],
                "dynamic_speed_mps": speed,
                "dynamic_phase_s": phase_s,
                "outcome": outcome,
                "success": int(success),
                "collision": int(collision),
                "timeout": int(timeout),
                "fallen": int(fallen),
                "outside": int(outside),
                "path_length_m": path_length,
                "episode_duration_s": duration_s,
                "episode_steps": step_count,
                "minimum_dynamic_clearance_m": minimum_clearance,
                "filter_active_fraction": active_steps / max(step_count, 1),
                "mean_command_correction_norm": correction_sum / max(step_count, 1),
            }
            records.append(record)

            if print_every > 0 and (index + 1) % print_every == 0:
                print(
                    f"[{method}] {map_name} case={index+1:3d}/{len(manifest)} "
                    f"outcome={outcome:13s} path={path_length:7.3f} m "
                    f"time={duration_s:6.2f} s dyn_clearance={minimum_clearance:+.3f} m",
                    flush=True,
                )

        output_dir.mkdir(parents=True, exist_ok=True)
        episode_csv = output_dir / f"{method}_episodes.csv"
        summary_csv = output_dir / f"{method}_summary.csv"
        trajectory_csv = output_dir / f"{method}_trajectory.csv"

        with episode_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

        if log_trajectories:
            with trajectory_csv.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(trajectory_records[0].keys()))
                writer.writeheader()
                writer.writerows(trajectory_records)
            print(f"  trajectory_csv: {trajectory_csv}")

        successes = [r for r in records if int(r["success"]) == 1]
        summary = {
            "method": method,
            "cases": len(records),
            "success_rate": sum(int(r["success"]) for r in records) / len(records),
            "collision_rate": sum(int(r["collision"]) for r in records) / len(records),
            "timeout_rate": sum(int(r["timeout"]) for r in records) / len(records),
            "fall_rate": sum(int(r["fallen"]) for r in records) / len(records),
            "outside_rate": sum(int(r["outside"]) for r in records) / len(records),
            "mean_success_path_length_m": optional_mean([float(r["path_length_m"]) for r in successes]),
            "std_success_path_length_m": sample_std([float(r["path_length_m"]) for r in successes]),
            "mean_success_goal_time_s": optional_mean([float(r["episode_duration_s"]) for r in successes]),
            "std_success_goal_time_s": sample_std([float(r["episode_duration_s"]) for r in successes]),
            "mean_minimum_dynamic_clearance_m": mean([float(r["minimum_dynamic_clearance_m"]) for r in records]),
            "minimum_dynamic_clearance_overall_m": min(float(r["minimum_dynamic_clearance_m"]) for r in records),
            "mean_filter_active_fraction": mean([float(r["filter_active_fraction"]) for r in records]),
            "mean_command_correction_norm": mean([float(r["mean_command_correction_norm"]) for r in records]),
        }

        with summary_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
            writer.writeheader()
            writer.writerow(summary)

        print(f"\n[{method} DYNAMIC SUMMARY]")
        for k, v in summary.items():
            print(f"  {k}: {v}")

        return episode_csv, summary_csv

    finally:
        close = getattr(env, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                print(f"[WARNING] Failed to close {method}/{map_name}: {exc}")

        del actor
        del env
        del low_env
        del raw_env
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(args.output_root).expanduser() / (
        f"{timestamp}_seed{args.seed}_{args.cases_per_map}permap_speed{args.speed:g}_jitter{args.jitter:g}"
    )
    root.mkdir(parents=True, exist_ok=True)

    checkpoints = {
        "no_cbf": args.no_cbf_checkpoint,
        "cbf_rl": args.cbf_checkpoint,
        "bcbf_rl": args.bcbf_checkpoint,
    }

    all_summary_rows: list[dict[str, object]] = []

    for map_index, map_name in enumerate(args.maps):
        print("\n" + "=" * 72)
        print(f"DYNAMIC POSITION-ONLY MAP: {map_name}")
        print("=" * 72)

        map_dir = root / map_name
        map_dir.mkdir(parents=True, exist_ok=True)
        manifest_seed = int(args.seed) + 10000 * map_index

        manifest_path = map_dir / "dynamic_scenarios.csv"
        if manifest_path.exists() and not args.overwrite_manifest:
            manifest = base.load_manifest(manifest_path)
        else:
            manifest = generate_dynamic_manifest(
                map_name=map_name,
                count=args.cases_per_map,
                seed=manifest_seed,
                jitter=args.jitter,
                speed=args.speed,
                random_motion_phase=args.random_motion_phase,
                min_clearance=args.min_start_goal_clearance,
                max_attempts=args.max_sampling_attempts,
                output_path=manifest_path,
            )

        arena = base.make_map_arena(map_name)
        map_csv = map_dir / "map.csv"
        with map_csv.open("w", newline="") as f:
            fields = [
                "map_name", "obstacle_id", "x", "y", "radius", "height",
                "is_dynamic", "dynamic_axis", "dynamic_target_value", "dynamic_speed_mps",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            spec = DYNAMIC_SPECS[map_name]
            for i, obs in enumerate(arena.obstacles):
                writer.writerow(
                    {
                        "map_name": map_name,
                        "obstacle_id": i,
                        "x": obs.x,
                        "y": obs.y,
                        "radius": obs.radius,
                        "height": obs.height,
                        "is_dynamic": int(i == spec.obstacle_id),
                        "dynamic_axis": spec.axis if i == spec.obstacle_id else "",
                        "dynamic_target_value": spec.target_value if i == spec.obstacle_id else "",
                        "dynamic_speed_mps": args.speed if i == spec.obstacle_id else "",
                    }
                )

        for method in args.methods:
            method_dir = map_dir / method
            method_dir.mkdir(parents=True, exist_ok=True)
            episode_csv, summary_csv = evaluate_dynamic_method(
                method=method,
                task_id=args.task_id,
                low_checkpoint=args.low_level_checkpoint,
                high_checkpoint=checkpoints[method],
                manifest=manifest,
                episode_duration_s=args.episode_duration_s,
                device=device,
                print_every=args.print_every,
                output_dir=method_dir,
                map_name=map_name,
                speed=args.speed,
                log_trajectories=args.log_trajectories,
            )

            with summary_csv.open(newline="") as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                row["map_name"] = map_name
                all_summary_rows.append(row)

    combined = root / "combined_summary_by_map.csv"
    if all_summary_rows:
        fields = ["map_name"] + [k for k in all_summary_rows[0].keys() if k != "map_name"]
        with combined.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_summary_rows)

    print("\n" + "=" * 72)
    print("Dynamic position-only evaluation completed")
    print(f"Results: {root}")
    print(f"Summary: {combined}")
    print("=" * 72)


if __name__ == "__main__":
    main()
