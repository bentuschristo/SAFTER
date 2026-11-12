"""Evaluate a CBF-RL PPO checkpoint on fixed unseen-map layouts.

Mirrors scripts/laser_unseen_eval/evaluate_laser_unseen_maps_structured.py's
"N cases per seed x M seeds" structured-scenario evaluation convention
(same UNSEEN_MAPS obstacle layouts and STRUCTURED_SCENARIO_REGIONS
start/goal boxes, same rejection-sampling manifest algorithm), but drives a
plain rsl_rl/PPO CbfRlNavigationEnv checkpoint -- a deterministic policy
call plus a single env.step(action) -- instead of the SAC pipeline's
differentiable actor + step_safe(nominal, safe, safety) interface.

Example (matches the SAC baselines' "500 cases, 25 seeds" convention:
cases_per_seed=20 x num_seeds=25 = 500):

  python scripts/evaluate_high_level_cbf_rl_unseen_maps.py \\
      Unitree-Go2-Safe-Navigation \\
      --checkpoint logs/rsl_rl/go2_cbf_rl/2026-09-14_16-51-50_dual_handcrafted2_seed0/model_1666.pt \\
      --low-level-checkpoint logs/rsl_rl/go2_velocity/2026-06-19_13-24-10/model_10000.pt \\
      --maps slalom corridor clustered_v2 circle_and_square \\
      --cases-per-seed 20 --num-seeds 25 --seed 1000 \\
      --num-envs 50 --device cuda:0
"""
from __future__ import annotations

import csv
import math
import random
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path

import mujoco
import torch
import tyro

import mjlab
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends

from src.high_level_cbf_rl.cbf_rl_navigation_env import CbfRlNavigationEnv
from src.high_level_cbf_rl.config import CbfRlConfig, CbfRlModeConfig
from src.high_level_control.command_interface import MjlabVelocityCommandInterface
from src.high_level_rl.config import ArenaConfig, CircularObstacle
from src.high_level_rl.low_level import load_frozen_low_level_policy

from train_high_level_cbf_rl import add_navigation_laser_sensor, make_walls_only_arena_cfg

# --- The unseen-map obstacle layouts + structured start/goal regions, copied
# verbatim from scripts/laser_unseen_eval/evaluate_laser_unseen_maps_structured.py
# (UNSEEN_MAPS / STRUCTURED_SCENARIO_REGIONS) so this evaluates the exact
# same scenarios the SAC baselines were scored on. Duplicated rather than
# imported since scripts/ isn't a package and that file is a large,
# SAC-actor-specific evaluator not meant to be imported from.

UNSEEN_MAPS: dict[str, tuple[CircularObstacle, ...]] = {
    "corridor": (
        CircularObstacle(-3.8, -2.2, 0.50),
        CircularObstacle(-1.9, -2.2, 0.50),
        CircularObstacle(0.0, -2.2, 0.50),
        CircularObstacle(1.9, -2.2, 0.50),
        CircularObstacle(3.8, -2.2, 0.50),
        CircularObstacle(-3.8, 2.2, 0.50),
        CircularObstacle(-1.9, 2.2, 0.50),
        CircularObstacle(1.9, 2.2, 0.50),
        CircularObstacle(3.8, 2.2, 0.50),
    ),
    # Corridor-derived High-Shift v5. Upper row preserves the v4 larger
    # circles and 0.70 m gates. The lower row forms a symmetric V: outer
    # circles are raised farther than inner circles, with 0.80 m Euclidean
    # edge-to-edge gaps between every adjacent lower-row obstacle.
    "corridor_derived_highshift_v5": (
        CircularObstacle(-4.00,  2.2, 0.65),
        CircularObstacle(-2.00,  2.2, 0.65),
        CircularObstacle( 2.00,  2.2, 0.65),
        CircularObstacle( 4.00,  2.2, 0.65),
        CircularObstacle(-4.08, -0.4, 0.50),
        CircularObstacle(-2.35, -1.3, 0.65),
        CircularObstacle( 0.00, -2.2, 0.90, shape="square"),
        CircularObstacle( 2.35, -1.3, 0.65),
        CircularObstacle( 4.08, -0.4, 0.50),
    ),
    # "Eased" circle_and_square (2026-09-24): center circle radius reduced
    # 1.00 -> 0.70, and the two small goal-side squares at (+-1.0, -2.6,
    # r=0.40) removed entirely -- matches SAFTER_scripts/laser_unseen_eval/
    # evaluate_laser_unseen_maps_structured.py's own updated layout exactly,
    # now used as the "High-Shift" tier in place of the diagonal map.
    "circle_and_square": (
        CircularObstacle(-1.6, 3.4, 0.35, shape="circle"),
        CircularObstacle(1.6, 3.4, 0.35, shape="circle"),
        CircularObstacle(0.0, 1.9, 0.70, shape="circle"),
        CircularObstacle(-1.5, -0.2, 0.35, shape="circle"),
        CircularObstacle(1.5, -0.2, 0.35, shape="circle"),
        CircularObstacle(-4.6, -2.3, 0.65, shape="circle"),
        CircularObstacle(4.6, -2.3, 0.65, shape="circle"),
        CircularObstacle(-3.6, 0.4, 0.70, shape="square"),
        CircularObstacle(3.6, 0.4, 0.70, shape="square"),
    ),
    "clustered_v2": (
        CircularObstacle(-3.0, 1.4, 0.55),
        CircularObstacle(-1.3, 2.7, 0.45),
        CircularObstacle(-0.08, 0.55, 0.60),
        CircularObstacle(-2.4, 4.1, 0.50),
        CircularObstacle(0.5, 3.9, 0.40),
        CircularObstacle(1.98, -2.71, 0.50),
        CircularObstacle(3.35, -1.44, 0.45),
        CircularObstacle(-0.42, -1.55, 0.55),
        CircularObstacle(3.47, -3.86, 0.50),
    ),
    "slalom": (
        CircularObstacle(-5.2, -2.3, 0.50),
        CircularObstacle(-3.9, 2.1, 0.50),
        CircularObstacle(-2.6, -2.2, 0.50),
        CircularObstacle(-1.3, 2.2, 0.50),
        CircularObstacle(0.0, -2.2, 0.50),
        CircularObstacle(1.3, 2.2, 0.50),
        CircularObstacle(2.6, -2.2, 0.50),
        CircularObstacle(3.9, 2.1, 0.50),
        CircularObstacle(5.2, -2.3, 0.50),
    ),
    # diagonal (2026-09-23, v4): copied verbatim from
    # evaluate_laser_unseen_maps_structured.py's UNSEEN_MAPS["diagonal"] --
    # see that file's comment and scripts/plot_diagonal_shift_map_preview.py
    # for the full design writeup (bottom-left -> top-right traversal,
    # point-symmetric ring of large circles around a dead-center r=1.0
    # blocker, one symmetric 0.88 m tight gate).
    "diagonal": (
        CircularObstacle(0.0, 0.0, 1.00, shape="circle"),
        CircularObstacle(-4.5, -0.75, 0.65, shape="circle"),
        CircularObstacle(4.5, 0.75, 0.65, shape="circle"),
        CircularObstacle(-0.7, -3.6, 0.60, shape="circle"),
        CircularObstacle(0.7, 3.6, 0.60, shape="circle"),
        CircularObstacle(-2.569, -2.759, 0.55, shape="circle"),
        CircularObstacle(2.569, 2.759, 0.55, shape="circle"),
        CircularObstacle(-4.7, 1.7, 0.55, shape="circle"),
        CircularObstacle(4.7, -1.7, 0.55, shape="circle"),
        CircularObstacle(-1.7, 4.7, 0.50, shape="circle"),
        CircularObstacle(1.7, -4.7, 0.50, shape="circle"),
    ),
}

STRUCTURED_SCENARIO_REGIONS: dict[str, dict[str, tuple[float, float, float, float]]] = {
    "slalom": {"start": (-4.0, 4.0, 5.0, 7.5), "goal": (-4.0, 4.0, -7.5, -4.0)},
    "clustered_v2": {"start": (-7.5, -4.0, 4.0, 7.5), "goal": (5.5, 7.5, -7.5, -5.5)},
    "corridor": {"start": (-2.0, 2.0, 4.0, 7.5), "goal": (-4.0, 4.0, -7.5, -4.0)},
    "corridor_derived_highshift_v5": {
        "start": (-2.0, 2.0, 4.0, 7.5), "goal": (-4.0, 4.0, -7.5, -4.0),
    },
    "circle_and_square": {"start": (-1.0, 1.0, 5.0, 6.8), "goal": (-5.5, 5.5, -7.0, -4.0)},
    "diagonal": {"start": (-7.5, -4.6, -7.5, -4.6), "goal": (4.6, 7.5, 4.6, 7.5)},
}


@dataclass
class FixedScenario:
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    start_yaw: float


def clip_box_to_arena(
    box: tuple[float, float, float, float], arena: ArenaConfig
) -> tuple[float, float, float, float]:
    x0, x1, y0, y1 = map(float, box)
    margin = float(arena.wall_clearance)
    x0 = max(x0, float(arena.x_min) + margin)
    x1 = min(x1, float(arena.x_max) - margin)
    y0 = max(y0, float(arena.y_min) + margin)
    y1 = min(y1, float(arena.y_max) - margin)
    if x0 >= x1 or y0 >= y1:
        raise ValueError(f"Sampling box {box} is empty after clipping to arena {arena}.")
    return x0, x1, y0, y1


def sample_point_from_box(rng: random.Random, box: tuple[float, float, float, float]) -> torch.Tensor:
    x0, x1, y0, y1 = box
    return torch.tensor([rng.uniform(x0, x1), rng.uniform(y0, y1)], dtype=torch.float32)


def generate_manifest(
    *,
    env: CbfRlNavigationEnv,
    arena: ArenaConfig,
    map_name: str,
    count: int,
    seed: int,
    min_endpoint_clearance: float = 0.10,
    block_margin: float = 0.10,
    require_blocked_path: bool = True,
    blocked_t_min: float = 0.15,
    blocked_t_max: float = 0.85,
    max_sampling_attempts: int = 10000,
    yaw_jitter_rad: float = 0.25,
) -> list[dict[str, object]]:
    """Structured start/goal cases for one map -- ported verbatim (algorithm
    unchanged) from evaluate_laser_unseen_maps_structured.py::generate_manifest,
    minus its CSV-writing side effect."""
    regions = STRUCTURED_SCENARIO_REGIONS[map_name]
    start_box = clip_box_to_arena(regions["start"], arena)
    goal_box = clip_box_to_arena(regions["goal"], arena)
    rng = random.Random(int(seed))

    centers_cpu = env.obstacle_centers[0].detach().cpu()
    radii_cpu = env.obstacle_radii[0].detach().cpu()
    mask_cpu = env.obstacle_mask[0].detach().cpu().bool()
    obstacle_margin = float(arena.obstacle_clearance)

    def obstacle_clearance(point: torch.Tensor) -> float:
        distances = torch.linalg.norm(centers_cpu - point.unsqueeze(0), dim=-1)
        clearances = (distances - radii_cpu - obstacle_margin)[mask_cpu]
        return float(clearances.min().item()) if clearances.numel() else math.inf

    def wall_clearance(point: torch.Tensor) -> float:
        x, y = float(point[0]), float(point[1])
        margin = float(arena.wall_clearance)
        return min(
            x - (arena.x_min + margin), (arena.x_max - margin) - x,
            y - (arena.y_min + margin), (arena.y_max - margin) - y,
        )

    def total_clearance(point: torch.Tensor) -> float:
        return min(obstacle_clearance(point), wall_clearance(point))

    def blocking_obstacle_details(start: torch.Tensor, goal: torch.Tensor) -> bool:
        seg = goal - start
        seg_len_sq = float(torch.dot(seg, seg).item())
        if seg_len_sq <= 1.0e-12:
            return False
        blocked = False
        for center, radius, is_valid in zip(centers_cpu, radii_cpu, mask_cpu):
            if not bool(is_valid):
                continue
            t = float(torch.dot(center - start, seg).item() / seg_len_sq)
            if t < blocked_t_min or t > blocked_t_max:
                continue
            closest = start + t * seg
            distance = float(torch.linalg.norm(center - closest).item())
            threshold = float(radius.item()) + obstacle_margin + block_margin
            if distance <= threshold:
                blocked = True
        return blocked

    rows: list[dict[str, object]] = []
    attempts = 0
    while len(rows) < count and attempts < max_sampling_attempts:
        attempts += 1
        start_point = sample_point_from_box(rng, start_box)
        goal_point = sample_point_from_box(rng, goal_box)
        if total_clearance(start_point) < min_endpoint_clearance:
            continue
        if total_clearance(goal_point) < min_endpoint_clearance:
            continue
        if float(torch.linalg.norm(goal_point - start_point).item()) <= 1.0:
            continue
        if require_blocked_path and not blocking_obstacle_details(start_point, goal_point):
            continue
        yaw_nominal = math.atan2(float(goal_point[1] - start_point[1]), float(goal_point[0] - start_point[0]))
        yaw = yaw_nominal + rng.uniform(-yaw_jitter_rad, yaw_jitter_rad)
        case_id = len(rows)
        rows.append({
            "map_name": map_name,
            "case_id": case_id,
            "reset_seed": int(seed + case_id),
            "start_x": float(start_point[0]),
            "start_y": float(start_point[1]),
            "start_yaw": float(yaw),
            "goal_x": float(goal_point[0]),
            "goal_y": float(goal_point[1]),
        })
    if len(rows) < count:
        raise RuntimeError(
            f"Could only sample {len(rows)}/{count} valid structured cases for map {map_name!r} "
            f"after {attempts} attempts."
        )
    return rows


def make_fixed_map_obstacle_entities(obstacles: tuple[CircularObstacle, ...]) -> dict[str, EntityCfg]:
    """One static body per obstacle, placed once (no per-episode movement
    needed -- unlike training's reusable procedural mocap slots, an eval run
    only ever uses one fixed map)."""
    entities: dict[str, EntityCfg] = {}
    for index, obstacle in enumerate(obstacles):
        half_height = 0.5 * float(obstacle.height)
        is_square = obstacle.shape == "square"
        geom_type = "box" if is_square else "cylinder"
        size = (
            f"{float(obstacle.radius)} {float(obstacle.radius)} {half_height}"
            if is_square else f"{float(obstacle.radius)} {half_height}"
        )
        xml = f"""
        <mujoco model="fixed_eval_obstacle_{index}">
          <worldbody>
            <body name="fixed_eval_obstacle_{index}" mocap="true" pos="0 0 0">
              <geom name="fixed_eval_obstacle_{index}_collision" type="{geom_type}"
                    size="{size}" rgba="0.85 0.15 0.08 1" contype="1" conaffinity="1"
                    condim="3" friction="0.8 0.05 0.001" group="0"/>
            </body>
          </worldbody>
        </mujoco>
        """
        entities[f"fixed_eval_obstacle_{index}"] = EntityCfg(
            spec_fn=lambda xml=xml: mujoco.MjSpec.from_string(xml),
            init_state=EntityCfg.InitialStateCfg(pos=(float(obstacle.x), float(obstacle.y), half_height)),
        )
    return entities


@dataclass
class EvalCbfRlConfig:
    checkpoint: str
    low_level_checkpoint: str
    # Comma-separated because the task launcher performs two-stage CLI
    # parsing (tyro.cli for the task id, then tyro.cli(EvalCbfRlConfig,
    # args=remaining)); a variable-length tuple loses all but its first
    # value across that boundary -- same reason
    # train_SAC_policies.py's --hidden-dims is a
    # comma-separated string instead of a tuple field.
    maps: str = "slalom,corridor,clustered_v2,circle_and_square"
    cases_per_seed: int = 20
    num_seeds: int = 25
    seed_stride: int = 1
    seed: int = 1000
    num_envs: int = 50
    # None inherits the navigation config (90 s, matching what training
    # actually used) -- the old 60.0 default here never matched training and
    # inflated the apparent timeout rate.
    episode_duration_s: float | None = None
    device: str | None = None
    use_cbf_action_filtering: bool = True
    laser_num_rays: int = 120
    laser_fov_deg: float = 360.0
    enable_lidar_noise: bool = True
    output_dir: str = "logs/evaluation/cbf_rl_unseen_maps"
    run_name: str = ""
    config: CbfRlConfig = field(default_factory=CbfRlConfig)


def build_map_env(task_id: str, map_name: str, cfg: EvalCbfRlConfig, device: str) -> CbfRlNavigationEnv:
    obstacles = UNSEEN_MAPS[map_name]
    arena = ArenaConfig(obstacles=obstacles, max_obstacles=len(obstacles))
    navigation_cfg = replace(cfg.config.navigation, arena=arena)
    if cfg.episode_duration_s is not None:
        navigation_cfg = replace(navigation_cfg, episode_duration_s=cfg.episode_duration_s)
    laser_cfg = replace(
        cfg.config.laser,
        num_rays=int(cfg.laser_num_rays),
        fov_deg=float(cfg.laser_fov_deg),
        enable_lidar_noise=bool(cfg.enable_lidar_noise),
    )
    # circle_and_square has square slots -- CbfRlNavigationEnv requires
    # collision_geometry="circle" whenever any slot is square, so use it
    # uniformly across all maps for a consistent, comparable eval
    # (matches the SAC baselines' own --collision-geometry circle convention
    # for this exact map set, see run_slalom_corridor_clustered_v2_*.sh).
    safety_cfg = replace(cfg.config.safety, collision_geometry="circle", collision_circle_radius_m=0.30)
    method_cfg = replace(
        cfg.config,
        navigation=navigation_cfg,
        laser=laser_cfg,
        safety=safety_cfg,
        mode=CbfRlModeConfig(use_cbf_action_filtering=cfg.use_cbf_action_filtering, use_cbf_reward_penalty=False),
    )

    env_cfg = load_env_cfg(task_id, play=True)
    low_agent_cfg = load_rl_cfg(task_id)
    # Not capped to cases_per_seed: run() now pools every seed's manifest for
    # a map into one list before calling evaluate_map, so a bigger env here
    # means fewer, larger batches -- each batch pays its "wait for the
    # single slowest straggler" cost once, so more/smaller batches multiply
    # that fixed cost for no benefit (matches the same fix applied to
    # evaluate_high_level_cbf_rl_training_maps.py's build_bank_env).
    env_cfg.scene.num_envs = int(cfg.num_envs)
    env_cfg.terminations = {}
    if "push_robot" in env_cfg.events:
        del env_cfg.events["push_robot"]
    # unitree_go2_safe_navigation_env_cfg() bakes the *default* 9-obstacle
    # arena (src/tasks/velocity/config/go2/obstacle_cfg.py's module-level
    # OBSTACLES) into "navigation_arena" unconditionally. Left alone, that
    # default set stays physically present -- LiDAR-visible and collidable
    # -- on top of the UNSEEN_MAPS obstacles added below, giving the robot
    # two overlapping, conflicting obstacle layouts (the analytical
    # obstacle_centers/collision bookkeeping only ever knows about the
    # UNSEEN_MAPS set). Replace it with a walls-only arena matching this
    # map's own wall geometry before adding this map's actual obstacles.
    env_cfg.scene.entities["navigation_arena"] = make_walls_only_arena_cfg(arena)
    env_cfg.scene.entities.update(make_fixed_map_obstacle_entities(obstacles))
    add_navigation_laser_sensor(env_cfg, method_cfg)

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    low_env = RslRlVecEnvWrapper(raw_env, clip_actions=low_agent_cfg.clip_actions)
    low_policy = load_frozen_low_level_policy(task_id, low_env, cfg.low_level_checkpoint, device)
    command_if = MjlabVelocityCommandInterface(low_env, command_name="twist")
    return CbfRlNavigationEnv(low_env, low_policy, command_if, method_cfg)


def load_policy(env: CbfRlNavigationEnv, task_id: str, checkpoint: str, device: str):
    agent_cfg = load_rl_cfg(task_id)
    agent_cfg.actor.hidden_dims = (128, 128)
    agent_cfg.critic.hidden_dims = (128, 128)
    runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    runner.load(str(checkpoint_path), load_cfg={"actor": True}, strict=False, map_location=device)
    return runner.get_inference_policy(device=device)


def evaluate_map(
    *, env: CbfRlNavigationEnv, policy, manifest: list[dict[str, object]], num_envs: int, print_every: int,
    stochastic_policy: bool = False,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if not manifest:
        return records
    batch_size = min(num_envs, env.num_envs, len(manifest))
    for batch_start in range(0, len(manifest), batch_size):
        real_cases = manifest[batch_start : batch_start + batch_size]
        padded_cases = list(real_cases)
        while len(padded_cases) < env.num_envs:
            padded_cases.append(real_cases[-1])
        # Reproducible initial RNG state for matching ablation batches.
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
        # goal_b (body-frame, normalized goal offset) is the first 2 columns
        # of the actor observation for every LaserSacNavigationEnv subclass
        # (config.py: "goal_b(2), measured command(3), ..."). Recovering the
        # absolute distance from it needs no env changes -- just undo the
        # normalize_positions_by scaling already applied when it was built.
        normalize_by = float(env.navigation_cfg.normalize_positions_by)

        def distance_to_goal(obs_tensordict) -> torch.Tensor:
            goal_b = env.actor_tensor(obs_tensordict)[:, :2]
            return torch.linalg.norm(goal_b, dim=-1) * normalize_by

        start_distance = distance_to_goal(obs)
        min_distance = start_distance.clone()
        with torch.no_grad():
            while not all(finished):
                # get_inference_policy()'s returned callable indexes obs by
                # obs_groups (obs["actor"]/obs["critic"]) itself -- pass the
                # full TensorDict, not env.actor_tensor(obs)'s flat tensor.
                action = policy(obs, stochastic_output=True) if stochastic_policy else policy(obs)
                obs, reward, done, extras = env.step(action)
                # Returned obs already contains the next episode for done envs.
                current_distance = extras["distance_to_goal"]
                min_distance = torch.minimum(min_distance, current_distance)
                completed_ids = extras["completed_env_ids"]
                completed_lookup = {int(env_id): row for row, env_id in enumerate(completed_ids.tolist())}
                for env_id in range(active_count):
                    if finished[env_id]:
                        continue
                    returns[env_id] += float(reward[env_id])
                    steps[env_id] += 1
                    row = completed_lookup.get(env_id)
                    if row is None:
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
                    case = real_cases[env_id]
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
                    })
                    finished[env_id] = True
        completed = sum(1 for c in real_cases)
        if print_every > 0 and (batch_start // batch_size) % max(print_every, 1) == 0:
            print(f"  ...batch {batch_start}-{batch_start + len(real_cases)}/{len(manifest)} done")
    return records


def summarize(records: list[dict[str, object]], map_name: str) -> dict[str, object]:
    n = len(records)
    return {
        "map_name": map_name,
        "cases": n,
        "success_rate": sum(r["success"] for r in records) / n if n else math.nan,
        "collision_rate": sum(r["collision"] for r in records) / n if n else math.nan,
        "timeout_rate": sum(r["timeout"] for r in records) / n if n else math.nan,
        "fall_rate": sum(r["fallen"] for r in records) / n if n else math.nan,
        "mean_return": sum(r["return"] for r in records) / n if n else math.nan,
    }


def run(task_id: str, cfg: EvalCbfRlConfig) -> None:
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
        # evaluate_map once per seed -- each call's batching waits for its
        # single slowest straggler (up to the full episode timeout), so
        # num_seeds separate small-batch calls pay that fixed cost
        # num_seeds times instead of once.
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

        all_records = evaluate_map(env=env, policy=policy, manifest=pooled_manifest, num_envs=cfg.num_envs, print_every=5)
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
    run(task, cfg)


if __name__ == "__main__":
    main()
