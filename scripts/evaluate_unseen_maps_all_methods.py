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
import mujoco

from src.high_level_rl.config import ArenaConfig, CircularObstacle
from src.tasks.velocity.config.go2 import obstacle_cfg as physical_arena_cfg

# Register built-in and project-specific tasks before calling load_env_cfg().
import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401

# Reuse the tested imports and environment construction utilities from the
# existing evaluators instead of duplicating project-specific imports.
import evaluate_high_level_sac as sac_eval
import evaluate_high_level_sac_bcbf_cf as bcbf_eval



# All evaluation maps preserve the training interface:
#   - 15 x 15 m arena
#   - 9 cylindrical obstacles
#   - 4 fixed outer walls
#
# Radii remain within the approximate training-map range of 0.40--0.60 m.
UNSEEN_MAPS: dict[str, tuple[CircularObstacle, ...]] = {
    "shifted": (
        CircularObstacle(-4.2, -1.0, 0.55),
        CircularObstacle(-2.0,  3.1, 0.45),
        CircularObstacle( 0.4, -2.8, 0.60),
        CircularObstacle( 2.8,  3.7, 0.50),
        CircularObstacle( 4.4,  0.2, 0.40),
        CircularObstacle(-5.4,  5.2, 0.50),
        CircularObstacle(-5.2, -4.8, 0.45),
        CircularObstacle( 5.3,  4.7, 0.55),
        CircularObstacle( 4.8, -5.1, 0.50),
    ),
    "corridor": (
        CircularObstacle(-3.8, -2.2, 0.50),
        CircularObstacle(-1.9, -2.2, 0.50),
        CircularObstacle( 0.0, -2.2, 0.50),
        CircularObstacle( 1.9, -2.2, 0.50),
        CircularObstacle( 3.8, -2.2, 0.50),
        CircularObstacle(-3.8,  2.2, 0.50),
        CircularObstacle(-1.9,  2.2, 0.50),
        CircularObstacle( 1.9,  2.2, 0.50),
        CircularObstacle( 3.8,  2.2, 0.50),
    ),
    "clustered": (
        CircularObstacle(-2.6,  1.7, 0.55),
        CircularObstacle(-1.3,  2.7, 0.45),
        CircularObstacle(-0.2,  1.5, 0.60),
        CircularObstacle(-2.1,  3.8, 0.50),
        CircularObstacle( 0.1,  3.6, 0.40),
        CircularObstacle( 3.8, -3.6, 0.50),
        CircularObstacle( 5.0, -2.4, 0.45),
        CircularObstacle( 3.1, -1.9, 0.55),
        CircularObstacle( 5.2, -4.8, 0.50),
    ),
    "slalom": (
        CircularObstacle(-5.2, -2.3, 0.50),
        CircularObstacle(-3.9,  2.1, 0.50),
        CircularObstacle(-2.6, -2.2, 0.50),
        CircularObstacle(-1.3,  2.2, 0.50),
        CircularObstacle( 0.0, -2.2, 0.50),
        CircularObstacle( 1.3,  2.2, 0.50),
        CircularObstacle( 2.6, -2.2, 0.50),
        CircularObstacle( 3.9,  2.1, 0.50),
        CircularObstacle( 5.2, -2.3, 0.50),
    ),
    "bottleneck": (
        CircularObstacle(-4.8,  4.5, 0.50),
        CircularObstacle(-4.8, -4.5, 0.50),
        CircularObstacle( 4.8,  4.5, 0.50),
        CircularObstacle( 4.8, -4.5, 0.50),
        CircularObstacle(-1.8,  1.15, 0.55),
        CircularObstacle( 0.0,  1.15, 0.55),
        CircularObstacle( 1.8,  1.15, 0.55),
        CircularObstacle(-0.9, -1.15, 0.55),
        CircularObstacle( 0.9, -1.15, 0.55),
    ),
}


def make_map_arena(map_name: str) -> ArenaConfig:
    obstacles = UNSEEN_MAPS[map_name]
    arena = ArenaConfig(obstacles=obstacles)
    arena.validate()
    return arena


def install_physical_map(map_name: str) -> None:
    """Replace obstacle_cfg's physical MJCF obstacle source."""
    source = UNSEEN_MAPS[map_name]

    physical_arena_cfg.OBSTACLES = tuple(
        physical_arena_cfg.CircularObstacle(
            x=float(obs.x),
            y=float(obs.y),
            radius=float(obs.radius),
            height=float(obs.height),
        )
        for obs in source
    )


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

    parser.add_argument(
        "--cases-per-map",
        type=int,
        default=20,
        help="Number of shared start-goal cases evaluated on each unseen map.",
    )
    parser.add_argument(
        "--maps",
        nargs="+",
        default=list(UNSEEN_MAPS),
        choices=list(UNSEEN_MAPS),
        help="Unseen maps to evaluate.",
    )
    parser.add_argument(
        "--case-ids",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional manifest case IDs to evaluate. "
            "Example: --case-ids 26 78"
        ),
    )
    parser.add_argument(
        "--log-trajectories",
        action="store_true",
        help="Write one per-step trajectory CSV for each evaluated method.",
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


def make_config(
    module: Any,
    episode_duration_s: float,
    arena: ArenaConfig | None = None,
):
    """Create the appropriate immutable project configuration."""
    if module is sac_eval:
        config = module.SacCbfCfConfig()
    else:
        config = module.SacBcbfCfConfig()

    navigation_changes = {
        "episode_duration_s": float(episode_duration_s),
    }
    if arena is not None:
        navigation_changes["arena"] = arena

    navigation = replace(
        config.navigation,
        **navigation_changes,
    )
    return replace(config, navigation=navigation)


def build_environment(
    *,
    module: Any,
    task_id: str,
    low_checkpoint: str,
    episode_duration_s: float,
    device: str,
    map_name: str | None = None,
):
    """Construct one single-environment SAC navigation stack."""
    module.configure_torch_backends()

    arena = None
    if map_name is not None:
        install_physical_map(map_name)
        arena = make_map_arena(map_name)

    config = make_config(
        module,
        episode_duration_s,
        arena=arena,
    )

    env_cfg = module.load_env_cfg(task_id, play=True)
    low_agent_cfg = module.load_rl_cfg(task_id)

    # Rebuild the physical MuJoCo arena after installing the unseen-map
    # obstacle list. The task config otherwise keeps the EntityCfg created
    # earlier from the original training map.
    if map_name is not None:
        # Ensure the physical navigation arena entity exists.
        # Some task configs do not include it by default, so add/overwrite it here.
        env_cfg.scene.entities["navigation_arena"] = (
            physical_arena_cfg.get_go2_navigation_arena_cfg()
        )

    env_cfg.scene.num_envs = 1
    env_cfg.terminations = {}

    # Evaluation A isolates unseen obstacle geometry.
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
                f"Evaluation A still contains event {forbidden_event!r}."
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
    arena: ArenaConfig,
    map_name: str,
    count: int,
    seed: int,
    output_path: Path,
) -> list[dict[str, object]]:
    """Generate reproducible scenarios and record geometric difficulty."""
    env.clear_fixed_scenario()
    env_ids = torch.zeros(
        1,
        dtype=torch.long,
        device=env.device,
    )

    rows: list[dict[str, object]] = []

    def obstacle_clearance(point: torch.Tensor) -> float:
        centers = env.obstacle_centers[0]
        radii = env.obstacle_radii[0]
        mask = env.obstacle_mask[0]

        point_device = point.to(
            device=centers.device,
            dtype=centers.dtype,
        )

        distances = torch.linalg.norm(
            centers - point_device.unsqueeze(0),
            dim=-1,
        )

        clearances = (
            distances
            - radii
            - float(arena.obstacle_clearance)
        )

        clearances = clearances[mask]

        if clearances.numel() == 0:
            return math.inf

        return float(clearances.min().detach().cpu())

    def wall_clearance(point: torch.Tensor) -> float:
        x = float(point[0])
        y = float(point[1])
        margin = float(arena.wall_clearance)

        clearances = (
            x - (arena.x_min + margin),
            (arena.x_max - margin) - x,
            y - (arena.y_min + margin),
            (arena.y_max - margin) - y,
        )

        return float(min(clearances))

    def total_clearance(point: torch.Tensor) -> float:
        return min(
            obstacle_clearance(point),
            wall_clearance(point),
        )

    def straight_path_min_clearance(
        start_point: torch.Tensor,
        goal_point: torch.Tensor,
        samples: int = 201,
    ) -> float:
        fractions = torch.linspace(
            0.0,
            1.0,
            samples,
            dtype=start_point.dtype,
        )

        minimum = math.inf

        for fraction in fractions:
            point = (
                start_point
                + fraction * (goal_point - start_point)
            )
            minimum = min(minimum, total_clearance(point))

        return float(minimum)

    for case_id in range(count):
        # Make every case reproducible independently of preceding resets.
        case_seed = int(seed + case_id)
        set_seed(case_seed)

        env._reset_selected(env_ids)

        position, yaw = env.safety_context()
        start_point = position[0].detach().cpu()
        goal_point = env.goal_positions[0].detach().cpu()

        start_obstacle_clearance = obstacle_clearance(start_point)
        goal_obstacle_clearance = obstacle_clearance(goal_point)

        start_wall_clearance = wall_clearance(start_point)
        goal_wall_clearance = wall_clearance(goal_point)

        start_clearance = min(
            start_obstacle_clearance,
            start_wall_clearance,
        )
        goal_clearance = min(
            goal_obstacle_clearance,
            goal_wall_clearance,
        )

        straight_line_distance = float(
            torch.linalg.norm(
                goal_point - start_point
            )
        )

        direct_path_clearance = straight_path_min_clearance(
            start_point,
            goal_point,
        )

        if start_clearance <= 0.0:
            raise RuntimeError(
                f"Map {map_name!r}, case {case_id}: "
                f"invalid start clearance {start_clearance:.6f} m."
            )

        if goal_clearance <= 0.0:
            raise RuntimeError(
                f"Map {map_name!r}, case {case_id}: "
                f"invalid goal clearance {goal_clearance:.6f} m."
            )

        rows.append(
            {
                "map_name": map_name,
                "case_id": case_id,
                "reset_seed": case_seed,
                "start_x": float(start_point[0]),
                "start_y": float(start_point[1]),
                "start_yaw": float(yaw[0].detach().cpu()),
                "goal_x": float(goal_point[0]),
                "goal_y": float(goal_point[1]),
                "start_obstacle_clearance_m":
                    start_obstacle_clearance,
                "start_wall_clearance_m":
                    start_wall_clearance,
                "start_clearance_m":
                    start_clearance,
                "goal_obstacle_clearance_m":
                    goal_obstacle_clearance,
                "goal_wall_clearance_m":
                    goal_wall_clearance,
                "goal_clearance_m":
                    goal_clearance,
                "straight_line_distance_m":
                    straight_line_distance,
                "straight_path_min_clearance_m":
                    direct_path_clearance,
                "direct_path_blocked":
                    int(direct_path_clearance <= 0.0),
            }
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "map_name",
        "case_id",
        "reset_seed",
        "start_x",
        "start_y",
        "start_yaw",
        "goal_x",
        "goal_y",
        "start_obstacle_clearance_m",
        "start_wall_clearance_m",
        "start_clearance_m",
        "goal_obstacle_clearance_m",
        "goal_wall_clearance_m",
        "goal_clearance_m",
        "straight_line_distance_m",
        "straight_path_min_clearance_m",
        "direct_path_blocked",
    ]

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[MANIFEST DIAGNOSTICS] map={map_name}")

    for row in rows:
        print(
            f"  case={int(row['case_id']):2d} "
            f"start_clearance="
            f"{float(row['start_clearance_m']):+.3f} m "
            f"goal_clearance="
            f"{float(row['goal_clearance_m']):+.3f} m "
            f"direct_path_clearance="
            f"{float(row['straight_path_min_clearance_m']):+.3f} m "
            f"blocked={int(row['direct_path_blocked'])}"
        )

    return rows

def load_manifest(
    path: Path,
) -> list[dict[str, object]]:
    """Load scenarios while preserving all diagnostic columns."""
    integer_fields = {
        "case_id",
        "reset_seed",
        "direct_path_blocked",
    }

    string_fields = {
        "map_name",
    }

    required_fields = {
        "case_id",
        "reset_seed",
        "start_x",
        "start_y",
        "start_yaw",
        "goal_x",
        "goal_y",
    }

    rows: list[dict[str, object]] = []

    with path.open(newline="") as file:
        reader = csv.DictReader(file)

        missing = required_fields.difference(
            reader.fieldnames or []
        )
        if missing:
            raise ValueError(
                f"Manifest is missing columns: {sorted(missing)}"
            )

        for raw_row in reader:
            row: dict[str, object] = {}

            for key, value in raw_row.items():
                if key is None:
                    continue

                if value is None or value == "":
                    row[key] = math.nan
                elif key in string_fields:
                    row[key] = value
                elif key in integer_fields:
                    row[key] = int(value)
                else:
                    row[key] = float(value)

            rows.append(row)

    return rows

def row_to_scenario(row: dict[str, float | int]) -> SimpleNamespace:
    return SimpleNamespace(
        name=f"random_{int(row['case_id']):03d}",
        start_xy=(float(row["start_x"]), float(row["start_y"])),
        start_yaw=float(row["start_yaw"]),
        goal_xy=(float(row["goal_x"]), float(row["goal_y"])),
    )



def verify_compiled_physical_map(
    raw_env,
    env,
    map_name: str,
) -> None:
    """Verify MuJoCo body positions/radii against analytical obstacle tensors."""
    model = raw_env.sim.mj_model

    expected_centers = env.obstacle_centers[0].detach().cpu()
    expected_radii = env.obstacle_radii[0].detach().cpu()

    compiled_centers = []
    compiled_radii = []

    def find_named_id(
        object_type,
        target_name: str,
        count: int,
    ) -> int:
        exact_id = mujoco.mj_name2id(
            model,
            object_type,
            target_name,
        )
        if exact_id >= 0:
            return exact_id

        matches = []

        for object_id in range(count):
            compiled_name = mujoco.mj_id2name(
                model,
                object_type,
                object_id,
            )

            if compiled_name is None:
                continue

            if (
                compiled_name == target_name
                or compiled_name.endswith("/" + target_name)
            ):
                matches.append(object_id)

        if len(matches) == 1:
            return matches[0]

        if not matches:
            return -1

        raise RuntimeError(
            f"Multiple compiled objects match "
            f"{target_name!r}: {matches}"
        )

    for index in range(expected_centers.shape[0]):
        body_name = f"obstacle_{index}"
        geom_name = f"obstacle_{index}_collision"

        body_id = find_named_id(
            mujoco.mjtObj.mjOBJ_BODY,
            body_name,
            model.nbody,
        )
        geom_id = find_named_id(
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_name,
            model.ngeom,
        )

        if body_id < 0:
            raise RuntimeError(
                f"Missing MuJoCo body {body_name!r} for map {map_name!r}."
            )

        if geom_id < 0:
            raise RuntimeError(
                f"Missing MuJoCo geom {geom_name!r} for map {map_name!r}."
            )

        compiled_centers.append(
            [
                float(model.body_pos[body_id, 0]),
                float(model.body_pos[body_id, 1]),
            ]
        )
        compiled_radii.append(float(model.geom_size[geom_id, 0]))

    compiled_centers = torch.tensor(compiled_centers)
    compiled_radii = torch.tensor(compiled_radii)

    print(f"[PHYSICAL GEOMETRY CHECK] map={map_name}")
    print("Compiled MuJoCo centers:")
    print(compiled_centers)
    print("Expected analytical centers:")
    print(expected_centers)

    if not torch.allclose(
        compiled_centers,
        expected_centers,
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise RuntimeError(
            f"Compiled MuJoCo obstacle centers do not match map {map_name!r}."
        )

    if not torch.allclose(
        compiled_radii,
        expected_radii,
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise RuntimeError(
            f"Compiled MuJoCo obstacle radii do not match map {map_name!r}."
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

    # The checkpoint contains obstacle buffers from the training map.
    # Load all learned weights first, then restore the current unseen map.
    actor.load_state_dict(checkpoint["actor"])

    with torch.no_grad():
        actor.obstacle_centers.copy_(
            env.obstacle_centers[: actor.obstacle_centers.shape[0]]
        )
        actor.obstacle_radii.copy_(
            env.obstacle_radii[: actor.obstacle_radii.shape[0]]
        )
        actor.obstacle_mask.copy_(
            env.obstacle_mask[: actor.obstacle_mask.shape[0]]
        )

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
    map_name: str | None = None,
    log_trajectories: bool = False,
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
        map_name=map_name,
    )

    if map_name is not None:
        verify_compiled_physical_map(
            raw_env,
            env,
            map_name,
        )

    actor = build_actor(
        method=method,
        module=module,
        env=env,
        config=config,
        checkpoint_path=high_checkpoint,
        device=device,
    )

    env_centers = env.obstacle_centers[:1]
    env_radii = env.obstacle_radii[:1]
    env_mask = env.obstacle_mask[:1]

    print(f"[GEOMETRY CHECK] method={method}, map={map_name}")
    print("Environment centers:")
    print(env_centers[0].detach().cpu())
    print("Actor centers:")
    print(actor.obstacle_centers[0].detach().cpu())

    if not torch.allclose(actor.obstacle_centers, env_centers):
        raise RuntimeError(
            "Actor obstacle centers do not match environment centers."
        )

    if not torch.allclose(actor.obstacle_radii, env_radii):
        raise RuntimeError(
            "Actor obstacle radii do not match environment radii."
        )

    if not torch.equal(actor.obstacle_mask, env_mask):
        raise RuntimeError(
            "Actor obstacle mask does not match environment mask."
        )

    records: list[dict[str, object]] = []
    trajectory_records: list[dict[str, object]] = []

    try:
        for index, case in enumerate(manifest):
            case_id = int(case["case_id"])
            set_seed(int(case["reset_seed"]))

            scenario = row_to_scenario(case)
            env.set_fixed_scenario(scenario)

            obs = env.actor_tensor(env.get_observations())

            # Verify that every method receives the exact shared scenario.
            actual_position, actual_yaw = env.safety_context()
            actual_goal = env.goal_positions[0]

            expected_start = torch.tensor(
                [
                    float(case["start_x"]),
                    float(case["start_y"]),
                ],
                device=actual_position.device,
                dtype=actual_position.dtype,
            )

            expected_goal = torch.tensor(
                [
                    float(case["goal_x"]),
                    float(case["goal_y"]),
                ],
                device=actual_goal.device,
                dtype=actual_goal.dtype,
            )

            expected_yaw = torch.tensor(
                float(case["start_yaw"]),
                device=actual_yaw.device,
                dtype=actual_yaw.dtype,
            )

            if not torch.allclose(
                actual_position[0],
                expected_start,
                atol=1.0e-4,
                rtol=0.0,
            ):
                raise RuntimeError(
                    f"{method}/{map_name}/case {case_id}: "
                    f"start mismatch. Expected "
                    f"{expected_start.detach().cpu().tolist()}, got "
                    f"{actual_position[0].detach().cpu().tolist()}."
                )

            if not torch.allclose(
                actual_goal,
                expected_goal,
                atol=1.0e-4,
                rtol=0.0,
            ):
                raise RuntimeError(
                    f"{method}/{map_name}/case {case_id}: "
                    f"goal mismatch. Expected "
                    f"{expected_goal.detach().cpu().tolist()}, got "
                    f"{actual_goal.detach().cpu().tolist()}."
                )

            if not torch.allclose(
                actual_yaw[0],
                expected_yaw,
                atol=1.0e-4,
                rtol=0.0,
            ):
                raise RuntimeError(
                    f"{method}/{map_name}/case {case_id}: "
                    f"yaw mismatch. Expected "
                    f"{float(expected_yaw.detach().cpu()):.6f}, got "
                    f"{float(actual_yaw[0].detach().cpu()):.6f}."
                )

            position = actual_position
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

                    step_start_position = (
                        previous_position.detach().clone()
                    )

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

                    if log_trajectories:
                        trajectory_records.append(
                            {
                                "method": method,
                                "map_name": map_name,
                                "case_id": case_id,
                                "reset_seed": int(case["reset_seed"]),
                                "step": step_count,
                                "time_s": (
                                    step_count
                                    * env.navigation_cfg.high_level_dt
                                ),
                                "pre_x_m": float(
                                    step_start_position[0]
                                    .detach()
                                    .cpu()
                                ),
                                "pre_y_m": float(
                                    step_start_position[1]
                                    .detach()
                                    .cpu()
                                ),
                                "post_x_m": float(
                                    endpoint[0].detach().cpu()
                                ),
                                "post_y_m": float(
                                    endpoint[1].detach().cpu()
                                ),

                                # Body heading at the beginning of this
                                # high-level control step.
                                "yaw_rad": float(
                                    yaw[0].detach().cpu()
                                ),

                                "start_x_m": float(case["start_x"]),
                                "start_y_m": float(case["start_y"]),
                                "goal_x_m": float(case["goal_x"]),
                                "goal_y_m": float(case["goal_y"]),
                                "nominal_vx_mps": float(
                                    nominal[0, 0].detach().cpu()
                                ),
                                "nominal_vy_mps": float(
                                    nominal[0, 1].detach().cpu()
                                ),
                                "nominal_wz_radps": float(
                                    nominal[0, 2].detach().cpu()
                                ),
                                "safe_vx_mps": float(
                                    safe[0, 0].detach().cpu()
                                ),
                                "safe_vy_mps": float(
                                    safe[0, 1].detach().cpu()
                                ),
                                "safe_wz_radps": float(
                                    safe[0, 2].detach().cpu()
                                ),
                                "command_correction_norm": float(
                                    correction.detach().cpu()
                                ),
                                "filter_active": int(filter_active),
                                "physical_clearance_m": float(
                                    physical_clearance(env, endpoint)
                                ),
                                "cumulative_path_length_m": path_length,
                                "done": int(bool(done[0])),
                                "outcome": "",
                                "success": 0,
                                "collision": 0,
                                "timeout": 0,
                                "fallen": 0,
                                "outside": 0,
                            }
                        )

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

                        if log_trajectories:
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

            cumulative_mean_return = (
                sum(float(row["return"]) for row in records) + return_value
            ) / (len(records) + 1)
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
                "start_clearance_m": float(
                    case.get("start_clearance_m", math.nan)
                ),
                "goal_clearance_m": float(
                    case.get("goal_clearance_m", math.nan)
                ),
                "straight_line_distance_m": float(
                    case.get(
                        "straight_line_distance_m",
                        math.nan,
                    )
                ),
                "straight_path_min_clearance_m": float(
                    case.get(
                        "straight_path_min_clearance_m",
                        math.nan,
                    )
                ),
                "direct_path_blocked": int(
                    case.get("direct_path_blocked", 0)
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
        trajectory_csv = output_dir / f"{method}_trajectory.csv"
        summary_csv = output_dir / f"{method}_summary.csv"

        fields = list(records[0].keys())

        with episode_csv.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)

        if log_trajectories:
            if not trajectory_records:
                raise RuntimeError(
                    f"No trajectory records were generated for "
                    f"{method}/{map_name}."
                )

            trajectory_fields = list(trajectory_records[0].keys())
            with trajectory_csv.open("w", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=trajectory_fields,
                )
                writer.writeheader()
                writer.writerows(trajectory_records)

            print(f"  trajectory_csv: {trajectory_csv}")

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
        # Close only the outermost wrapper. Its close() owns and closes the
        # wrapped low-level and raw environments.
        close = getattr(env, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                print(
                    f"[WARNING] Failed to close {method}/{map_name}: {exc}",
                    flush=True,
                )

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

    device = (
        args.device
        if args.device is not None
        else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    print(f"Evaluation device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(
        "logs/evaluation/unseen_maps/"
        f"{timestamp}_seed{args.seed}_{args.cases_per_map}permap"
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
        print(f"UNSEEN MAP: {map_name}")
        print("=" * 72)

        map_dir = root / map_name
        map_dir.mkdir(parents=True, exist_ok=True)

        # Build one temporary environment using this map's geometry solely
        # to sample valid shared start-goal cases.
        module = sac_eval
        manifest_seed = int(args.seed) + 10000 * map_index
        arena = make_map_arena(map_name)

        set_seed(manifest_seed)
        raw_env, low_env, env, _ = build_environment(
            module=module,
            task_id=args.task_id,
            low_checkpoint=args.low_level_checkpoint,
            episode_duration_s=args.episode_duration_s,
            device=device,
            map_name=map_name,
        )

        try:
            manifest_path = map_dir / "scenarios.csv"
            manifest = generate_manifest(
                arena=arena,
                map_name=map_name,
                env=env,
                count=args.cases_per_map,
                seed=manifest_seed,
                output_path=manifest_path,
            )
        finally:
            # Close the outermost wrapper once; it owns the wrapped stack.
            close = getattr(env, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    print(
                        f"[WARNING] Failed to close manifest environment "
                        f"for {map_name}: {exc}",
                        flush=True,
                    )

            del env
            del low_env
            del raw_env
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Add map identity and obstacle geometry to a separate metadata file.
        metadata_path = map_dir / "map.csv"
        with metadata_path.open("w", newline="") as file:
            fields = ["map_name", "obstacle_id", "x", "y", "radius", "height"]
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for obstacle_id, obs in enumerate(arena.obstacles):
                writer.writerow(
                    {
                        "map_name": map_name,
                        "obstacle_id": obstacle_id,
                        "x": obs.x,
                        "y": obs.y,
                        "radius": obs.radius,
                        "height": obs.height,
                    }
                )

        evaluation_manifest = manifest

        if args.case_ids is not None:
            requested_case_ids = set(args.case_ids)
            evaluation_manifest = [
                row
                for row in manifest
                if int(row["case_id"]) in requested_case_ids
            ]

            found_case_ids = {
                int(row["case_id"])
                for row in evaluation_manifest
            }
            missing_case_ids = requested_case_ids - found_case_ids

            if missing_case_ids:
                raise ValueError(
                    "Requested case IDs were not found in the generated "
                    f"manifest: {sorted(missing_case_ids)}"
                )

            print(
                "Selected manifest cases: "
                f"{sorted(found_case_ids)}"
            )

        for method, checkpoint in checkpoints.items():
            method_dir = map_dir / method
            method_dir.mkdir(parents=True, exist_ok=True)

            episode_csv, summary_csv = evaluate_method(
                method=method,
                task_id=args.task_id,
                low_checkpoint=args.low_level_checkpoint,
                high_checkpoint=checkpoint,
                manifest=evaluation_manifest,
                episode_duration_s=args.episode_duration_s,
                device=device,
                print_every=args.print_every,
                output_dir=method_dir,
                map_name=map_name,
                log_trajectories=args.log_trajectories,
            )

            with summary_csv.open(newline="") as file:
                summary_rows = list(csv.DictReader(file))

            for row in summary_rows:
                row["map_name"] = map_name
                all_summary_rows.append(row)

    combined_path = root / "combined_summary_by_map.csv"

    if all_summary_rows:
        ordered_fields = ["map_name"] + [
            field
            for field in all_summary_rows[0].keys()
            if field != "map_name"
        ]

        with combined_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=ordered_fields)
            writer.writeheader()
            writer.writerows(all_summary_rows)

    print("\n" + "=" * 72)
    print("Unseen-map evaluation completed")
    print(f"Results: {root}")
    print(f"Summary: {combined_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
