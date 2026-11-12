"""Evaluate CBF-RL on handcrafted2's training maps using the SAME
construction and reset mechanism training itself uses: one environment
built with the REAL procedural map bank (all 5 maps, full 12-slot obstacle
padding), pinned per-batch to specific scenario ids via
env.set_fixed_procedural_scenarios() -- exactly the method
scripts/evaluate_training_maps_no_filter.py already uses for the SAC-based
methods.

This replaces an earlier version of this script that built one environment
PER MAP with procedural_map_bank=None (only that map's own obstacle count as
slots, e.g. 7 instead of the trained 12) and pinned cases via
env.set_fixed_scenarios() with freshly-sampled region-box coordinates. That
construction reproducibly capped pooled success around ~35-42% regardless of
checkpoint, collision geometry, episode duration, or reset-function
correctness (all individually verified and/or fixed during this
investigation) -- while this construction (real bank + max_obstacles=12 +
set_fixed_procedural_scenarios) reproduces training's own ~80-85% success
rate almost exactly, confirmed on all 5 maps individually before this
rewrite. The obstacle-slot-count/environment-construction mismatch, not any
of the reset-function bugs found and fixed along the way, was the actual
root cause of the eval-vs-training gap this whole session.

Scenario selection mirrors evaluate_training_maps_no_filter.py's own
convention exactly: for each of the 5 maps and each of num_seeds evaluation
seeds, cases_per_seed scenario ids are sampled (without replacement within
that seed) from the map's own 200 curated scenarios via a seeded RNG, giving
the same "N cases per seed x M seeds" structured evaluation used throughout
this project.

Example (500 cases = 20/seed x 25 seeds, all 5 handcrafted2 maps):

  python scripts/evaluate_high_level_cbf_rl_training_maps.py \\
      Unitree-Go2-Safe-Navigation \\
      --checkpoint logs/rsl_rl/go2_cbf_rl/<run>/model_999.pt \\
      --low-level-checkpoint logs/rsl_rl/go2_velocity/2026-06-19_13-24-10/model_10000.pt \\
      --procedural-map-bank generated/procedural_maps/handcrafted2 \\
      --expected-num-maps 5 --expected-scenarios-per-map 200 --expected-max-obstacles 12 \\
      --cases-per-seed 20 --num-seeds 25 --seed 1000 \\
      --num-envs 50 --device cuda:0
"""
from __future__ import annotations

import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path

import torch
import tyro

# SAFTER_src isn't registered by the repo's editable install (only `src` is,
# via __editable__.unitree_rl_mjlab-*.pth) -- resolvable from a cwd-relative
# `python -c` snippet but not from `python scripts/foo.py` (script-relative
# sys.path[0]). Same fix already used elsewhere in this repo for ad-hoc dirs.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends

from src.high_level_cbf_rl.cbf_rl_navigation_env import CbfRlNavigationEnv
from src.high_level_cbf_rl.config import CbfRlConfig, CbfRlModeConfig
from src.high_level_control.command_interface import MjlabVelocityCommandInterface
try:
    # Local repo state: high_level_laser_sac lives under SAFTER_src/ (moved
    # there for the paper's supplementary code package).
    from SAFTER_src.high_level_laser_sac.procedural_maps import (
        ProceduralMapBank,
        load_procedural_map_bank,
        resolve_slot_shapes,
    )
except ModuleNotFoundError:
    # Server state: SAFTER_src/ hasn't been synced there -- fall back to the
    # original location.
    from src.high_level_laser_sac.procedural_maps import (
        ProceduralMapBank,
        load_procedural_map_bank,
        resolve_slot_shapes,
    )
from src.high_level_rl.low_level import load_frozen_low_level_policy

from evaluate_high_level_cbf_rl_unseen_maps import load_policy
from train_high_level_cbf_rl import (
    add_navigation_laser_sensor,
    make_procedural_obstacle_slot_entity,
    make_walls_only_arena_cfg,
)


@dataclass
class EvalTrainingMapsConfig:
    checkpoint: str
    low_level_checkpoint: str
    procedural_map_bank: str = "generated/procedural_maps/handcrafted2"
    expected_num_maps: int | None = 5
    expected_scenarios_per_map: int | None = 200
    expected_max_obstacles: int | None = 12
    cases_per_seed: int = 20
    num_seeds: int = 25
    seed_stride: int = 1
    seed: int = 1000
    num_envs: int = 50
    # None inherits the navigation config (90 s for the handcrafted2 runs).
    episode_duration_s: float | None = None
    device: str | None = None
    use_cbf_action_filtering: bool = True
    stochastic_policy: bool = False
    # Training uses play=False, including low-level observation noise.
    training_low_level: bool = True
    laser_num_rays: int = 120
    laser_fov_deg: float = 360.0
    enable_lidar_noise: bool = True
    output_dir: str = "logs/evaluation/cbf_rl_training_maps"
    run_name: str = ""
    config: CbfRlConfig = field(default_factory=CbfRlConfig)


def build_bank_env(
    task_id: str, bank: ProceduralMapBank, cfg: EvalTrainingMapsConfig, device: str
) -> CbfRlNavigationEnv:
    """Builds ONE environment with the real procedural map bank (all 5 maps,
    the bank's own full max_obstacles slot count) -- matching
    train_high_level_cbf_rl.py's own construction exactly, so scenario
    selection and obstacle-slot padding are identical to training."""
    slot_shapes = resolve_slot_shapes(bank.maps, bank.max_obstacles)
    entity_names = tuple(f"procedural_obstacle_slot_{slot:02d}" for slot in range(bank.max_obstacles))

    navigation_cfg = cfg.config.navigation
    bank_arena = bank.maps[0].arena
    if any(
        (item.arena.width, item.arena.length, item.arena.wall_thickness, item.arena.wall_height)
        != (bank_arena.width, bank_arena.length, bank_arena.wall_thickness, bank_arena.wall_height)
        for item in bank.maps
    ):
        raise ValueError("All maps in one procedural bank must share arena wall geometry")
    empty_arena_template = replace(bank_arena, obstacles=())
    for item in bank.maps[1:]:
        if replace(item.arena, obstacles=()) != empty_arena_template:
            raise ValueError("All procedural maps must use matching arena configuration")
    navigation_cfg = replace(navigation_cfg, arena=empty_arena_template)
    if cfg.episode_duration_s is not None:
        navigation_cfg = replace(navigation_cfg, episode_duration_s=cfg.episode_duration_s)

    laser_cfg = replace(
        cfg.config.laser,
        num_rays=int(cfg.laser_num_rays),
        fov_deg=float(cfg.laser_fov_deg),
        enable_lidar_noise=bool(cfg.enable_lidar_noise),
    )
    method_cfg = replace(
        cfg.config,
        navigation=navigation_cfg,
        laser=laser_cfg,
        mode=CbfRlModeConfig(use_cbf_action_filtering=cfg.use_cbf_action_filtering, use_cbf_reward_penalty=False),
    )

    torch.manual_seed(cfg.seed)
    env_cfg = load_env_cfg(task_id, play=not cfg.training_low_level)
    env_cfg.seed = cfg.seed
    low_agent_cfg = load_rl_cfg(task_id)
    # Not capped to cases_per_seed (a leftover from the old per-seed-manifest
    # design, where one batch == one seed's cases): the eval loop below
    # already chunks the full per-map manifest (all seeds pooled) into
    # batches of min(num_envs, len(manifest)), so a bigger env here means
    # fewer, larger batches -- each batch pays its "wait for the single
    # slowest straggler" cost once, so more/smaller batches multiply that
    # fixed cost for no benefit.
    env_cfg.scene.num_envs = int(cfg.num_envs)
    env_cfg.terminations = {}
    if "push_robot" in env_cfg.events:
        del env_cfg.events["push_robot"]
    env_cfg.scene.entities["navigation_arena"] = make_walls_only_arena_cfg(bank_arena)
    for slot, entity_name in enumerate(entity_names):
        env_cfg.scene.entities[entity_name] = make_procedural_obstacle_slot_entity(slot, slot_shapes[slot])
    add_navigation_laser_sensor(env_cfg, method_cfg)

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    raw_env.sim.expand_model_fields(("geom_size", "geom_rbound", "geom_aabb"))
    low_env = RslRlVecEnvWrapper(raw_env, clip_actions=low_agent_cfg.clip_actions)
    low_policy = load_frozen_low_level_policy(task_id, low_env, cfg.low_level_checkpoint, device)
    command_if = MjlabVelocityCommandInterface(low_env, command_name="twist")

    env = CbfRlNavigationEnv(
        low_env, low_policy, command_if, method_cfg,
        procedural_map_bank=bank, procedural_map_entity_names=entity_names,
    )
    env.assert_procedural_geometry_synchronized()
    return env


def build_manifest(bank: ProceduralMapBank, evaluation_seeds: list[int], cases_per_seed: int) -> list[dict[str, object]]:
    """Mirrors evaluate_training_maps_no_filter.py::build_manifest exactly:
    for each map and each seed, sample cases_per_seed scenario ids (without
    replacement within that seed) from the map's own 200 scenarios."""
    manifest: list[dict[str, object]] = []
    for map_item in bank.maps:
        scenarios_by_id = {s.scenario_id: s for s in map_item.scenarios}
        pool = list(scenarios_by_id)
        for seed in evaluation_seeds:
            rng = random.Random(seed * 1000 + map_item.map_id)
            chosen = rng.sample(pool, cases_per_seed)
            for case_id, scenario_id in enumerate(chosen):
                scenario = scenarios_by_id[scenario_id]
                manifest.append({
                    "map_name": f"train_map_{map_item.map_id}",
                    "map_id": map_item.map_id,
                    "evaluation_seed": seed,
                    "case_id": case_id,
                    "reset_seed": seed,
                    "scenario_id": scenario_id,
                    "start_x": scenario.start_xy[0],
                    "start_y": scenario.start_xy[1],
                })
    return manifest


def evaluate_procedural_scenarios_with_path_length(
    *, env: CbfRlNavigationEnv, policy, manifest: list[dict[str, object]], num_envs: int, print_every: int,
    stochastic_policy: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Same convention as evaluate_map_with_path_length, but pins cases via
    env.set_fixed_procedural_scenarios(scenario_ids) (the correct,
    training-matching reset path) instead of env.set_fixed_scenarios(...)."""
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
        scenario_ids = torch.tensor(
            [int(case["scenario_id"]) for case in padded_cases], dtype=torch.long, device=env.device,
        )
        env.set_fixed_procedural_scenarios(scenario_ids)
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
                current_distance = distance_to_goal(obs)
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
                        "scenario_id": case["scenario_id"],
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


def run(task_id: str, cfg: EvalTrainingMapsConfig) -> None:
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    evaluation_seeds = [cfg.seed + i * cfg.seed_stride for i in range(cfg.num_seeds)]

    bank = load_procedural_map_bank(
        cfg.procedural_map_bank,
        expected_num_maps=cfg.expected_num_maps,
        expected_scenarios_per_map=cfg.expected_scenarios_per_map,
        expected_max_obstacles=cfg.expected_max_obstacles,
    )

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = f"_{cfg.run_name}" if cfg.run_name else ""
    out_dir = Path(cfg.output_dir) / f"{stamp}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")

    print(f"\n[EVAL] Building one environment for the full bank ({len(bank.maps)} maps, "
          f"{bank.max_obstacles} obstacle slots)")
    env = build_bank_env(task_id, bank, cfg, device)
    policy = load_policy(env, task_id, cfg.checkpoint, device)

    manifest = build_manifest(bank, evaluation_seeds, cfg.cases_per_seed)
    print(f"[EVAL] Manifest: {len(manifest)} cases ({len(evaluation_seeds)} seeds x "
          f"{cfg.cases_per_seed} cases/seed x {len(bank.maps)} maps)")

    summary_rows: list[dict[str, object]] = []
    all_trajectory_rows_by_map: dict[str, list[dict[str, object]]] = {}
    all_records_by_map: dict[str, list[dict[str, object]]] = {}
    for map_item in bank.maps:
        map_name = f"train_map_{map_item.map_id}"
        map_manifest = [case for case in manifest if case["map_id"] == map_item.map_id]
        records, trajectory_rows = evaluate_procedural_scenarios_with_path_length(
            env=env, policy=policy, manifest=map_manifest, num_envs=cfg.num_envs, print_every=5,
            stochastic_policy=cfg.stochastic_policy,
        )
        all_records_by_map[map_name] = records
        all_trajectory_rows_by_map[map_name] = trajectory_rows
        print(f"  {map_name}: {len(records)} cases, success={sum(r['success'] for r in records)}/{len(records)}")

        with (out_dir / f"{map_name}_episodes.csv").open("w", newline="") as file:
            if records:
                writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
                writer.writeheader()
                writer.writerows(records)

        with (out_dir / f"{map_name}_trajectories.csv").open("w", newline="") as file:
            if trajectory_rows:
                writer = csv.DictWriter(file, fieldnames=list(trajectory_rows[0].keys()))
                writer.writeheader()
                writer.writerows(trajectory_rows)

        n = len(records)
        succ = [r for r in records if r["success"]]
        summary = {
            "map_name": map_name,
            "cases": n,
            "success_rate": sum(r["success"] for r in records) / n if n else math.nan,
            "collision_rate": sum(r["collision"] for r in records) / n if n else math.nan,
            "timeout_rate": sum(r["timeout"] for r in records) / n if n else math.nan,
            "fall_rate": sum(r["fallen"] for r in records) / n if n else math.nan,
            "mean_return": sum(r["return"] for r in records) / n if n else math.nan,
            "n_success": len(succ),
            "mean_path_length_m_success": sum(r["path_length_m"] for r in succ) / len(succ) if succ else math.nan,
            "mean_time_to_goal_s_success": (
                sum(r["episode_steps"] for r in succ) * 0.02 / len(succ) if succ else math.nan
            ),
        }
        summary_rows.append(summary)
        print(f"[EVAL] {map_name}: {summary}")

    env.close()

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
    cfg = tyro.cli(EvalTrainingMapsConfig, args=remaining, prog=sys.argv[0] + f" {task}", config=mjlab.TYRO_FLAGS)
    run(task, cfg)


if __name__ == "__main__":
    main()
