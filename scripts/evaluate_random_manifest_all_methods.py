#!/usr/bin/env python3
"""Paired random-scenario evaluation for SAC navigation methods.

One shared manifest of random start/goal/yaw cases is generated once and then
replayed for:

    1. no_cbf
    2. cbf_rl
    3. bcbf_rl

This ensures that all methods are evaluated on exactly the same navigation
problems rather than independently sampled random episodes.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from types import SimpleNamespace
from typing import Any

import torch

# Register built-in and project-specific tasks before calling load_env_cfg().
import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401

# Reuse the tested imports and environment construction utilities from the
# existing evaluators instead of duplicating project-specific imports.
import evaluate_high_level_sac as sac_eval
import evaluate_high_level_sac_bcbf_cf as bcbf_eval


DEFAULT_LOW = (
    "logs/rsl_rl/go2_velocity/"
    "2026-06-19_13-24-10/model_10000.pt"
)

DEFAULT_NOCBF = (
    "logs/sac/go2_safe_navigation_sac_baseline/"
    "2026-06-24_10-20-04_SAC-No-CBF-UTD025/"
    "model_step_20000.pt"
)

DEFAULT_CBF = (
    "logs/sac/go2_safe_navigation_sac_cbf_cf/"
    "2026-06-24_00-32-42_CBF-CF-RL-UTD025/"
    "model_step_20000.pt"
)

DEFAULT_BCBF = (
    "logs/sac/go2_safe_navigation_sac_bcbf_cf/"
    "2026-06-25_16-43-43_"
    "BCBF-CF-Original-Uhatd-History52-Env256-Step50000/"
    "model_step_50000.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate no-CBF, CBF-RL, and bCBF-RL on one shared "
        "random scenario manifest."
    )

    parser.add_argument(
        "task_id",
        nargs="?",
        default="Unitree-Go2-Safe-Navigation",
    )
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-duration-s", type=float, default=60.0)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--device", default=None)

    parser.add_argument("--low-level-checkpoint", default=DEFAULT_LOW)
    parser.add_argument("--no-cbf-checkpoint", default=DEFAULT_NOCBF)
    parser.add_argument("--cbf-checkpoint", default=DEFAULT_CBF)
    parser.add_argument("--bcbf-checkpoint", default=DEFAULT_BCBF)

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("no_cbf", "cbf_rl", "bcbf_rl"),
        default=("no_cbf", "cbf_rl", "bcbf_rl"),
    )

    parser.add_argument(
        "--output-root",
        default="logs/evaluation/random_manifest",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Existing manifest CSV. When omitted, a new shared manifest is generated.",
    )
    parser.add_argument(
        "--overwrite-manifest",
        action="store_true",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_std(values: list[float]) -> float:
    return stdev(values) if len(values) >= 2 else 0.0


def optional_mean(values: list[float]) -> float:
    return mean(values) if values else math.nan


def make_config(module: Any, episode_duration_s: float):
    """Create the appropriate immutable project configuration."""
    if module is sac_eval:
        config = module.SacCbfCfConfig()
    else:
        config = module.SacBcbfCfConfig()

    navigation = replace(
        config.navigation,
        episode_duration_s=float(episode_duration_s),
    )
    return replace(config, navigation=navigation)


def build_environment(
    *,
    module: Any,
    task_id: str,
    low_checkpoint: str,
    episode_duration_s: float,
    device: str,
):
    """Construct one single-environment SAC navigation stack."""
    module.configure_torch_backends()

    config = make_config(module, episode_duration_s)

    env_cfg = module.load_env_cfg(task_id, play=True)
    low_agent_cfg = module.load_rl_cfg(task_id)

    env_cfg.scene.num_envs = 1
    env_cfg.terminations = {}

    # Isolate obstacle-layout and scenario effects during evaluation.
    # Disable physical-domain randomization and disturbances.
    for event_name in (
        "push_robot",
        "randomize_terrain",
        "foot_friction",
        "encoder_bias",
        "base_com",
    ):
        if event_name in env_cfg.events:
            del env_cfg.events[event_name]

    for forbidden_event in (
        "push_robot",
        "randomize_terrain",
        "foot_friction",
        "encoder_bias",
        "base_com",
    ):
        if forbidden_event in env_cfg.events:
            raise RuntimeError(
                f"Nominal evaluation still contains event {forbidden_event!r}."
            )

    raw_env = module.ManagerBasedRlEnv(
        cfg=env_cfg,
        device=device,
        render_mode=None,
    )

    low_env = module.RslRlVecEnvWrapper(
        raw_env,
        clip_actions=low_agent_cfg.clip_actions,
    )

    low_policy = module.load_frozen_low_level_policy(
        task_id,
        low_env,
        low_checkpoint,
        device,
    )

    command_if = module.MjlabVelocityCommandInterface(
        low_env,
        command_name="twist",
    )

    env = module.SacCbfNavigationEnv(
        low_env,
        low_policy,
        command_if,
        config.navigation,
        reward_cfg=config.reward,
    )

    return raw_env, low_env, env, config


def generate_manifest(
    *,
    env,
    count: int,
    seed: int,
    output_path: Path,
) -> list[dict[str, float | int]]:
    """Generate valid random scenarios using the environment's own sampler."""
    set_seed(seed)

    env.clear_fixed_scenario()
    env_ids = torch.zeros(1, dtype=torch.long, device=env.device)

    rows: list[dict[str, float | int]] = []

    for case_id in range(count):
        # The base sampler enforces obstacle/wall clearance and minimum
        # start-goal distance.
        env._reset_selected(env_ids)

        position, yaw = env.safety_context()
        start = position[0].detach().cpu()
        goal = env.goal_positions[0].detach().cpu()

        rows.append(
            {
                "case_id": case_id,
                "reset_seed": seed + case_id,
                "start_x": float(start[0]),
                "start_y": float(start[1]),
                "start_yaw": float(yaw[0].detach().cpu()),
                "goal_x": float(goal[0]),
                "goal_y": float(goal[1]),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "case_id",
        "reset_seed",
        "start_x",
        "start_y",
        "start_yaw",
        "goal_x",
        "goal_y",
    ]

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def load_manifest(path: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []

    with path.open(newline="") as file:
        reader = csv.DictReader(file)

        required = {
            "case_id",
            "reset_seed",
            "start_x",
            "start_y",
            "start_yaw",
            "goal_x",
            "goal_y",
        }

        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Manifest is missing columns: {sorted(missing)}"
            )

        for row in reader:
            rows.append(
                {
                    "case_id": int(row["case_id"]),
                    "reset_seed": int(row["reset_seed"]),
                    "start_x": float(row["start_x"]),
                    "start_y": float(row["start_y"]),
                    "start_yaw": float(row["start_yaw"]),
                    "goal_x": float(row["goal_x"]),
                    "goal_y": float(row["goal_y"]),
                }
            )

    return rows


def row_to_scenario(row: dict[str, float | int]) -> SimpleNamespace:
    return SimpleNamespace(
        name=f"random_{int(row['case_id']):03d}",
        start_xy=(float(row["start_x"]), float(row["start_y"])),
        start_yaw=float(row["start_yaw"]),
        goal_xy=(float(row["goal_x"]), float(row["goal_y"])),
    )


def physical_clearance(env, position: torch.Tensor) -> float:
    """Minimum physical clearance to obstacles and walls."""
    cfg = env.navigation_cfg
    robot_radius = float(getattr(cfg.arena, "robot_radius", 0.30))

    p = position.reshape(1, 2)

    obstacle_distance = torch.linalg.norm(
        p[:, None, :] - env.obstacle_centers[:1],
        dim=-1,
    )

    obstacle_clearance = (
        obstacle_distance
        - env.obstacle_radii[:1]
        - robot_radius
    )

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

    value = torch.minimum(
        obstacle_clearance.min(),
        wall_clearance.min(),
    )

    return float(value.detach().cpu())


def build_actor(
    *,
    method: str,
    module: Any,
    env,
    config,
    checkpoint_path: str,
    device: str,
):
    obs = env.actor_tensor(env.get_observations())

    if method in ("no_cbf", "cbf_rl"):
        actor = module.SafeSacActor(
            obs_dim=obs.shape[-1],
            hidden_dims=config.sac.hidden_dims,
            arena=env.navigation_cfg.arena,
            action_cfg=env.navigation_cfg.action,
            safety_cfg=env.navigation_cfg.safety,
            obstacle_centers=env.obstacle_centers[:1],
            obstacle_radii=env.obstacle_radii[:1],
            obstacle_mask=env.obstacle_mask[:1],
            log_std_min=config.sac.log_std_min,
            log_std_max=config.sac.log_std_max,
            use_safety_filter=(method == "cbf_rl"),
        ).to(device)
    else:
        actor = module.SafeSacActor(
            obs_dim=obs.shape[-1],
            hidden_dims=config.sac.hidden_dims,
            arena=env.navigation_cfg.arena,
            action_cfg=env.navigation_cfg.action,
            safety_cfg=env.navigation_cfg.backup_safety,
            obstacle_centers=env.obstacle_centers[:1],
            obstacle_radii=env.obstacle_radii[:1],
            obstacle_mask=env.obstacle_mask[:1],
            log_std_min=config.sac.log_std_min,
            log_std_max=config.sac.log_std_max,
        ).to(device)

    checkpoint = torch.load(
        Path(checkpoint_path).expanduser(),
        map_location=device,
        weights_only=False,
    )

    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    return actor


def evaluate_method(
    *,
    method: str,
    task_id: str,
    low_checkpoint: str,
    high_checkpoint: str,
    manifest: list[dict[str, float | int]],
    episode_duration_s: float,
    device: str,
    print_every: int,
    output_dir: Path,
) -> tuple[Path, Path]:
    module = bcbf_eval if method == "bcbf_rl" else sac_eval

    # Reinitialize the RNG so each method receives the same initial
    # domain-randomization sequence as closely as possible.
    base_seed = int(manifest[0]["reset_seed"]) if manifest else 0
    set_seed(base_seed)

    raw_env, low_env, env, config = build_environment(
        module=module,
        task_id=task_id,
        low_checkpoint=low_checkpoint,
        episode_duration_s=episode_duration_s,
        device=device,
    )

    actor = build_actor(
        method=method,
        module=module,
        env=env,
        config=config,
        checkpoint_path=high_checkpoint,
        device=device,
    )

    records: list[dict[str, object]] = []

    try:
        for index, case in enumerate(manifest):
            case_id = int(case["case_id"])
            set_seed(int(case["reset_seed"]))

            scenario = row_to_scenario(case)
            env.set_fixed_scenario(scenario)

            obs = env.actor_tensor(env.get_observations())

            position, _ = env.safety_context()
            previous_position = position[0].detach().clone()

            path_length = 0.0
            minimum_clearance = physical_clearance(
                env,
                previous_position,
            )
            active_steps = 0
            correction_sum = 0.0
            return_value = 0.0
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
                    actor_obs = env.actor_tensor(obs)
                    position, yaw = env.safety_context()

                    out = actor.deterministic(
                        actor_obs,
                        position,
                        yaw,
                    )

                    nominal = out.nominal_command
                    safe = out.safe_command
                    safety = out.safety

                    next_obs, reward, done, extras = env.step_safe(
                        nominal,
                        safe,
                        safety,
                    )
                    return_value += float(reward[0].detach().cpu())

                    step_count += 1

                    if bool(done[0]):
                        terminal_positions = extras.get(
                            "completed_terminal_positions"
                        )
                        if (
                            terminal_positions is not None
                            and terminal_positions.numel() > 0
                        ):
                            endpoint = terminal_positions[0]
                        else:
                            endpoint = env.safety_context()[0][0]
                    else:
                        endpoint = env.safety_context()[0][0]

                    path_length += float(
                        torch.linalg.norm(
                            endpoint - previous_position
                        ).detach().cpu()
                    )
                    previous_position = endpoint.detach().clone()

                    minimum_clearance = min(
                        minimum_clearance,
                        physical_clearance(env, endpoint),
                    )

                    correction = torch.linalg.norm(
                        safe[0] - nominal[0]
                    )
                    correction_sum += float(correction.detach().cpu())

                    if method == "no_cbf":
                        filter_active = False
                    elif safety is not None and hasattr(safety, "active"):
                        filter_active = bool(safety.active[0])
                    else:
                        filter_active = (
                            float(correction.detach().cpu()) > 1.0e-6
                        )

                    if filter_active:
                        active_steps += 1

                    if bool(done[0]):
                        success = bool(
                            extras["completed_success"][0]
                        )
                        collision = bool(
                            extras["completed_collision"][0]
                        )
                        timeout = bool(
                            extras["completed_timeouts"][0]
                        )

                        if "completed_fallen" in extras:
                            fallen = bool(
                                extras["completed_fallen"][0]
                            )

                        if "completed_outside" in extras:
                            outside = bool(
                                extras["completed_outside"][0]
                            )

                        completed_steps = int(
                            extras["completed_lengths"][0]
                        )
                        duration_s = (
                            completed_steps
                            * env.navigation_cfg.high_level_dt
                        )

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

                        break

                    obs = next_obs

            cumulative_mean_return = (
                sum(float(row["return"]) for row in records) + return_value
            ) / (len(records) + 1)
            record = {
                "method": method,
                "case_id": case_id,
                "reset_seed": int(case["reset_seed"]),
                "start_x": float(case["start_x"]),
                "start_y": float(case["start_y"]),
                "start_yaw": float(case["start_yaw"]),
                "goal_x": float(case["goal_x"]),
                "goal_y": float(case["goal_y"]),
                "straight_line_distance_m": math.hypot(
                    float(case["goal_x"]) - float(case["start_x"]),
                    float(case["goal_y"]) - float(case["start_y"]),
                ),
                "outcome": outcome,
                "success": int(success),
                "collision": int(collision),
                "timeout": int(timeout),
                "fallen": int(fallen),
                "outside": int(outside),
                "return": return_value,
                "cumulative_mean_return": cumulative_mean_return,
                "path_length_m": path_length,
                "episode_duration_s": duration_s,
                "episode_steps": step_count,
                "minimum_physical_clearance_m": minimum_clearance,
                "filter_active_fraction": (
                    active_steps / max(step_count, 1)
                ),
                "mean_command_correction_norm": (
                    correction_sum / max(step_count, 1)
                ),
            }

            records.append(record)

            if (
                print_every > 0
                and (index + 1) % print_every == 0
            ):
                print(
                    f"[{method}] "
                    f"case={index + 1:3d}/{len(manifest)} "
                    f"outcome={outcome:13s} "
                    f"return={return_value:+8.2f} "
                    f"path={path_length:7.3f} m "
                    f"time={duration_s:6.2f} s "
                    f"clearance={minimum_clearance:+.3f} m",
                    flush=True,
                )

        output_dir.mkdir(parents=True, exist_ok=True)

        episode_csv = output_dir / f"{method}_episodes.csv"
        summary_csv = output_dir / f"{method}_summary.csv"

        fields = list(records[0].keys())

        with episode_csv.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)

        successes = [row for row in records if row["success"] == 1]

        summary = {
            "method": method,
            "cases": len(records),
            "success_rate": sum(
                int(row["success"]) for row in records
            ) / len(records),
            "collision_rate": sum(
                int(row["collision"]) for row in records
            ) / len(records),
            "timeout_rate": sum(
                int(row["timeout"]) for row in records
            ) / len(records),
            "fall_rate": sum(
                int(row["fallen"]) for row in records
            ) / len(records),
            "outside_rate": sum(
                int(row["outside"]) for row in records
            ) / len(records),
            "mean_return": mean(float(row["return"]) for row in records),
            "std_return": sample_std([float(row["return"]) for row in records]),
            "mean_success_path_length_m": optional_mean(
                [
                    float(row["path_length_m"])
                    for row in successes
                ]
            ),
            "std_success_path_length_m": sample_std(
                [
                    float(row["path_length_m"])
                    for row in successes
                ]
            ),
            "mean_success_goal_time_s": optional_mean(
                [
                    float(row["episode_duration_s"])
                    for row in successes
                ]
            ),
            "std_success_goal_time_s": sample_std(
                [
                    float(row["episode_duration_s"])
                    for row in successes
                ]
            ),
            "mean_minimum_clearance_m": mean(
                [
                    float(row["minimum_physical_clearance_m"])
                    for row in records
                ]
            ),
            "minimum_clearance_overall_m": min(
                float(row["minimum_physical_clearance_m"])
                for row in records
            ),
            "mean_filter_active_fraction": mean(
                [
                    float(row["filter_active_fraction"])
                    for row in records
                ]
            ),
            "mean_command_correction_norm": mean(
                [
                    float(row["mean_command_correction_norm"])
                    for row in records
                ]
            ),
        }

        with summary_csv.open("w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(summary.keys()),
            )
            writer.writeheader()
            writer.writerow(summary)

        print(f"\n[{method} SUMMARY]")
        for key, value in summary.items():
            print(f"  {key}: {value}")

        return episode_csv, summary_csv

    finally:
        # Different wrappers expose close() at different levels.
        for candidate in (env, low_env, raw_env):
            close = getattr(candidate, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

        del actor
        del env
        del low_env
        del raw_env

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def write_combined_summary(
    summary_paths: list[Path],
    output_path: Path,
) -> None:
    rows: list[dict[str, str]] = []

    for path in summary_paths:
        with path.open(newline="") as file:
            rows.extend(csv.DictReader(file))

    if not rows:
        return

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    if args.cases <= 0:
        raise ValueError("--cases must be positive")

    device = args.device or (
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = (
        Path(args.output_root).expanduser()
        / f"{stamp}_seed{args.seed}_{args.cases}cases"
    )
    batch_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest is not None:
        manifest_path = Path(args.manifest).expanduser()
        manifest = load_manifest(manifest_path)

        if len(manifest) < args.cases:
            raise ValueError(
                f"Manifest contains {len(manifest)} cases, "
                f"but --cases={args.cases}."
            )

        manifest = manifest[: args.cases]

        copied_manifest = batch_dir / "scenarios.csv"
        with copied_manifest.open("w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(manifest[0].keys()),
            )
            writer.writeheader()
            writer.writerows(manifest)

        manifest_path = copied_manifest
    else:
        manifest_path = batch_dir / "scenarios.csv"

        # Build one temporary standard SAC environment solely to use the
        # project's tested collision-free random sampler.
        raw_env, low_env, env, _ = build_environment(
            module=sac_eval,
            task_id=args.task_id,
            low_checkpoint=args.low_level_checkpoint,
            episode_duration_s=args.episode_duration_s,
            device=device,
        )

        try:
            manifest = generate_manifest(
                env=env,
                count=args.cases,
                seed=args.seed,
                output_path=manifest_path,
            )
        finally:
            for candidate in (env, low_env, raw_env):
                close = getattr(candidate, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass

            del env
            del low_env
            del raw_env
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("=" * 72)
    print("Shared random-scenario evaluation")
    print(f"Cases:       {len(manifest)}")
    print(f"Seed:        {args.seed}")
    print(f"Duration:    {args.episode_duration_s:.1f} s")
    print(f"Manifest:    {manifest_path}")
    print(f"Output:      {batch_dir}")
    print(f"Methods:     {', '.join(args.methods)}")
    print("=" * 72)

    checkpoint_for_method = {
        "no_cbf": args.no_cbf_checkpoint,
        "cbf_rl": args.cbf_checkpoint,
        "bcbf_rl": args.bcbf_checkpoint,
    }

    summary_paths: list[Path] = []

    for method in args.methods:
        print("\n" + "=" * 72)
        print(f"Evaluating {method}")
        print("=" * 72)

        method_dir = batch_dir / method

        _, summary_path = evaluate_method(
            method=method,
            task_id=args.task_id,
            low_checkpoint=args.low_level_checkpoint,
            high_checkpoint=checkpoint_for_method[method],
            manifest=manifest,
            episode_duration_s=args.episode_duration_s,
            device=device,
            print_every=args.print_every,
            output_dir=method_dir,
        )

        summary_paths.append(summary_path)

    combined_path = batch_dir / "combined_summary.csv"
    write_combined_summary(summary_paths, combined_path)

    print("\n" + "=" * 72)
    print("All evaluations completed")
    print(f"Manifest: {manifest_path}")
    print(f"Summary:  {combined_path}")
    print(f"Results:  {batch_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
