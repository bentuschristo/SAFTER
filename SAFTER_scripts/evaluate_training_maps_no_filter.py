#!/usr/bin/env python3
"""Evaluate the latest SAC policies (vanilla/cbf_cd_cf/bcbf_cf) on the actual
TRAINING procedural map bank (handcrafted2: 5 maps x 200 scenarios each),
with the safety filter DISABLED at runtime.

"Disabled" means: the actor's `safety` object (filter diagnostics/correction)
is still computed every step (so BcbfDiag-style stats remain available if
ever needed), but the command that actually reaches the low-level policy is
always the RAW/NOMINAL actor output -- `env.step_safe(nominal, nominal,
safety)` -- never the filtered/safe command. For method="vanilla" this is a
no-op (nominal == safe already, by construction). For cbf_cd_cf/bcbf_cf this
answers "how much does the CBF/bCBF math itself matter" versus what the
learned policy does on its own, in-distribution.

Scenario selection is fully deterministic: for each of the 5 training maps
and each of 25 evaluation seeds, 20 scenario ids are sampled (without
replacement) from that map's own 200 scenarios via a seeded RNG, giving 500
cases/map (2500 total) -- the same 25-seed x 20-case convention used
throughout this project's unseen-map evaluations. Each case is pinned with
env.set_fixed_procedural_scenarios(...) (bypassing curriculum/random
sampling and the reset-time feasibility resampler), so re-running this
script reproduces the identical map+start+goal every time.

No per-step trajectory logging -- only one row per episode (ids + outcome +
path_length_m/episode_duration_s), per the "just the id should be enough"
scope of this evaluation. path_length_m is the true integrated path length
(summed step-to-step displacement of the robot's root position, via the same
env.safety_context() positions used by evaluate_laser_unseen_maps_structured.py's
own path-length tracking), not a straight-line distance.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import random
import sys
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends

from SAFTER_src.high_level_control.command_interface import MjlabVelocityCommandInterface
from SAFTER_src.high_level_laser_sac.procedural_maps import load_procedural_map_bank, resolve_slot_shapes
from SAFTER_src.high_level_laser_sac.sac_navigation_env import LaserSacNavigationEnv
from SAFTER_src.high_level_rl.low_level import load_frozen_low_level_policy


def _load_sibling_module(name: str, relative_path: str):
    # A regular (has __init__.py) "scripts" package from a ROS install
    # shadows this repo's scripts/ directory on sys.path -- `import
    # scripts.foo` would resolve to THAT package and fail. Load these
    # sibling script files directly by path instead.
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_train_module = _load_sibling_module(
    "_train_SAC_policies", "SAFTER_scripts/train_SAC_policies.py"
)
_eval_module = _load_sibling_module(
    "_evaluate_laser_unseen_maps_structured",
    "SAFTER_scripts/laser_unseen_eval/evaluate_laser_unseen_maps_structured.py",
)

add_navigation_laser_sensor = _train_module.add_navigation_laser_sensor
make_procedural_obstacle_slot_entity = _train_module.make_procedural_obstacle_slot_entity
make_walls_only_arena_cfg = _train_module.make_walls_only_arena_cfg

load_actor = _eval_module.load_actor
make_laser_config = _eval_module.make_laser_config
mean_stats = _eval_module.mean_stats
proportion_stats = _eval_module.proportion_stats
set_seed = _eval_module.set_seed
timed_deterministic_actor = _eval_module.timed_deterministic_actor
warmup_actor_timing = _eval_module.warmup_actor_timing
write_rows = _eval_module.write_rows

TASK_ID = "Unitree-Go2-Safe-Navigation"


def build_manifest(procedural_map_bank, seeds: list[int], cases_per_seed: int) -> list[dict]:
    manifest = []
    for map_item in procedural_map_bank.maps:
        scenarios_by_id = {s.scenario_id: s for s in map_item.scenarios}
        pool = list(scenarios_by_id)
        for seed in seeds:
            rng = random.Random(seed * 1000 + map_item.map_id)
            chosen = rng.sample(pool, cases_per_seed)
            for case_id, scenario_id in enumerate(chosen):
                scenario = scenarios_by_id[scenario_id]
                manifest.append({
                    "map_id": map_item.map_id,
                    "evaluation_seed": seed,
                    "case_id": case_id,
                    "reset_seed": seed,
                    "scenario_id": scenario_id,
                    "start_x": scenario.start_xy[0],
                    "start_y": scenario.start_xy[1],
                    "goal_x": scenario.goal_xy[0],
                    "goal_y": scenario.goal_xy[1],
                    "start_yaw": scenario.start_yaw,
                    "difficulty": scenario.difficulty,
                })
    return manifest


def build_environment(
    *,
    method: str,
    low_checkpoint: str,
    episode_duration_s: float,
    device: str,
    procedural_map_bank_path: str,
    expected_num_maps: int,
    expected_scenarios_per_map: int,
    laser_num_rays: int,
    laser_fov_deg: float,
    collision_geometry: str,
    collision_circle_radius_m: float,
    collision_margin_m: float,
    collision_termination_mode: str,
    qp_backend: str,
    num_envs: int,
    terminal_backup_mode: str,
    legacy_projection: bool,
    action_max_vx: float,
    action_min_vx: float | None,
    action_max_vy: float,
    action_max_wz: float,
    backup_margin: float | None,
    laser_debug_vis: bool = False,
):
    configure_torch_backends()

    cfg = make_laser_config(
        method,
        episode_duration_s,
        arena=None,
        laser_num_rays=laser_num_rays,
        laser_fov_deg=laser_fov_deg,
        enable_lidar_noise=True,  # must match training exactly (checkpoint's saved laser dict is compared for full equality)
        hard_clip_output=True,
        collision_geometry=collision_geometry,
        collision_circle_radius_m=collision_circle_radius_m,
        collision_margin_m=collision_margin_m,
        collision_termination_mode=collision_termination_mode,
        qp_backend=qp_backend,
        terminal_backup_mode=terminal_backup_mode,
        terminal_geom_scale=1.0,
        terminal_rec_scale=1.0,
        legacy_projection=legacy_projection,
        action_max_vx=action_max_vx,
        action_min_vx=action_min_vx,
        action_max_vy=action_max_vy,
        action_max_wz=action_max_wz,
        backup_margin=backup_margin,
        laser_debug_vis=laser_debug_vis,
    )

    procedural_map_bank = load_procedural_map_bank(
        procedural_map_bank_path,
        expected_num_maps=expected_num_maps,
        expected_scenarios_per_map=expected_scenarios_per_map,
        expected_max_obstacles=None,
    )
    procedural_maps = procedural_map_bank.maps
    slot_shapes = resolve_slot_shapes(procedural_maps, procedural_map_bank.max_obstacles)

    env_cfg = load_env_cfg(TASK_ID, play=True)
    low_agent_cfg = load_rl_cfg(TASK_ID)
    env_cfg.scene.num_envs = int(num_envs)
    bank_arena = procedural_maps[0].arena
    env_cfg.scene.entities["navigation_arena"] = make_walls_only_arena_cfg(bank_arena)
    procedural_entity_names = tuple(
        f"procedural_obstacle_slot_{slot:02d}" for slot in range(procedural_map_bank.max_obstacles)
    )
    for slot, entity_name in enumerate(procedural_entity_names):
        env_cfg.scene.entities[entity_name] = make_procedural_obstacle_slot_entity(slot, slot_shapes[slot])
    env_cfg.terminations = {}
    for event_name in ("push_robot", "randomize_terrain", "foot_friction", "encoder_bias", "base_com"):
        if event_name in env_cfg.events:
            del env_cfg.events[event_name]

    # Match the training entry point's arena override exactly: only
    # wall/obstacle clearance and obstacles=() are overridden; width/length/
    # wall_thickness/wall_height/robot_radius/max_obstacles/obstacle_shape
    # come from the bank's own arena (verified identical to LaserSacConfig()
    # defaults for handcrafted2 -- checked directly against the saved
    # checkpoint's own navigation.arena dict).
    safety_arena_template = replace(
        bank_arena, obstacles=(), wall_clearance=float(cfg.safety.d_safe), obstacle_clearance=float(cfg.safety.d_safe)
    )
    cfg = replace(cfg, navigation=replace(cfg.navigation, arena=safety_arena_template))

    add_navigation_laser_sensor(env_cfg, cfg)

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    raw_env.sim.expand_model_fields(("geom_size", "geom_rbound", "geom_aabb"))
    low_env = RslRlVecEnvWrapper(raw_env, clip_actions=low_agent_cfg.clip_actions)
    low_policy = load_frozen_low_level_policy(TASK_ID, low_env, low_checkpoint, device)
    command_if = MjlabVelocityCommandInterface(low_env, command_name="twist")

    env = LaserSacNavigationEnv(
        low_env,
        low_policy,
        command_if,
        cfg.navigation,
        reward_cfg=cfg.reward,
        laser_cfg=cfg.laser,
        safety_cfg=cfg.safety,
        contact_collision_cfg=cfg.contact_collision,
        procedural_map_bank=procedural_map_bank,
        procedural_map_entity_names=procedural_entity_names,
    )
    env.assert_procedural_geometry_synchronized()
    return raw_env, env, cfg, procedural_map_bank


def evaluate_method(
    *,
    method: str,
    checkpoint_path: str,
    manifest: list[dict],
    batch_size: int,
    episodes: list[dict],
    filter_active: bool = False,
    **env_kwargs,
) -> None:
    set_seed(int(manifest[0]["reset_seed"]))
    raw_env, env, cfg, _ = build_environment(method=method, num_envs=batch_size, **env_kwargs)
    device = env_kwargs["device"]
    try:
        actor = load_actor(env=env, cfg=cfg, checkpoint_path=checkpoint_path, device=device)
        warmup_actor_timing(actor, env)
        dt = float(env.navigation_cfg.high_level_dt)

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
            returns = [0.0] * active_count
            step_counts = [0] * active_count
            positions, _ = env.safety_context()
            previous_positions = positions[:active_count].detach().clone()
            path_lengths = [0.0] * active_count

            with torch.no_grad():
                while not all(finished):
                    actor_obs = env.actor_tensor(obs)
                    nominal, safe, safety, _, _, _ = timed_deterministic_actor(actor, actor_obs)
                    applied = safe if filter_active else nominal
                    obs, reward, done, extras = env.step_safe(nominal, applied, safety)

                    completed_ids = extras.get(
                        "completed_env_ids", torch.empty(0, device=env.device, dtype=torch.long)
                    )
                    completed_lookup = {int(env_id): row for row, env_id in enumerate(completed_ids.tolist())}
                    current_positions, _ = env.safety_context()

                    for env_id, case in enumerate(real_cases):
                        if finished[env_id]:
                            continue
                        step_counts[env_id] += 1
                        step_reward = float(reward[env_id].detach().cpu())
                        returns[env_id] += step_reward
                        completed_row = completed_lookup.get(env_id)
                        is_done = completed_row is not None
                        endpoint = (
                            extras["completed_terminal_positions"][completed_row]
                            if is_done else current_positions[env_id]
                        )
                        path_lengths[env_id] += float(
                            torch.linalg.norm(endpoint - previous_positions[env_id]).detach().cpu()
                        )
                        previous_positions[env_id] = endpoint.detach().clone()
                        if not is_done:
                            continue
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
                        final_return = float(extras["completed_returns"][completed_row].detach().cpu())
                        duration_s = int(extras["completed_lengths"][completed_row]) * dt
                        episodes.append({
                            "method": method,
                            "map_id": int(case["map_id"]),
                            "evaluation_seed": int(case["evaluation_seed"]),
                            "case_id": int(case["case_id"]),
                            "scenario_id": int(case["scenario_id"]),
                            "outcome": outcome,
                            "success": int(success),
                            "collision": int(collision),
                            "timeout": int(timeout),
                            "fallen": int(fallen),
                            "outside": int(outside),
                            "return": final_return,
                            "episode_duration_s": duration_s,
                            "episode_steps": step_counts[env_id],
                            "path_length_m": path_lengths[env_id],
                        })
                        finished[env_id] = True
            print(f"[{method}] batch {batch_start}-{batch_start + len(real_cases)}/{len(manifest)} done "
                  f"({sum(1 for e in episodes if e['method'] == method)} episodes recorded so far)")
    finally:
        raw_env.close()


def summarize(episodes: list[dict]) -> list[dict]:
    rows = []
    methods = sorted({e["method"] for e in episodes})
    map_ids = sorted({e["map_id"] for e in episodes})
    for method in methods:
        for map_id in map_ids:
            group = [e for e in episodes if e["method"] == method and e["map_id"] == map_id]
            if not group:
                continue
            row = {"method": method, "map_id": map_id, "cases": len(group)}
            for field in ("success", "collision", "timeout", "fallen", "outside"):
                row.update(proportion_stats(sum(int(e[field]) for e in group), len(group), f"{field}_rate"))
            row.update(mean_stats([float(e["return"]) for e in group], "return"))
            successes = [e for e in group if e["success"]]
            row.update(mean_stats([float(e["path_length_m"]) for e in successes], "success_path_length_m"))
            row.update(mean_stats([float(e["episode_duration_s"]) for e in successes], "success_episode_duration_s"))
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-level-checkpoint", required=True)
    parser.add_argument("--vanilla-checkpoint", default=None)
    parser.add_argument("--cbf-cd-cf-checkpoint", default=None)
    parser.add_argument("--bcbf-cf-checkpoint", default=None)
    parser.add_argument("--methods", nargs="+", default=["vanilla", "cbf_cd_cf", "bcbf_cf"])
    parser.add_argument("--procedural-map-bank", default="Maps/Training Maps/handcrafted2")
    parser.add_argument("--expected-num-maps", type=int, default=5)
    parser.add_argument("--expected-scenarios-per-map", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--num-seeds", type=int, default=25)
    parser.add_argument("--seed-stride", type=int, default=1)
    parser.add_argument("--cases-per-seed", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=50)
    parser.add_argument("--episode-duration-s", type=float, default=60.0)
    parser.add_argument("--laser-num-rays", type=int, default=120)
    parser.add_argument("--laser-fov-deg", type=float, default=360.0)
    parser.add_argument("--collision-geometry", choices=("circle", "ellipse"), default="circle")
    parser.add_argument("--collision-circle-radius-m", type=float, default=0.30)
    parser.add_argument("--collision-margin-m", type=float, default=0.05)
    parser.add_argument("--collision-termination-mode", default="circle")
    parser.add_argument("--qp-backend", default="osqp")
    parser.add_argument("--action-max-vx", type=float, default=2.0)
    parser.add_argument("--action-min-vx", type=float, default=-1.0)
    parser.add_argument("--action-max-vy", type=float, default=1.0)
    parser.add_argument("--action-max-wz", type=float, default=1.0)
    parser.add_argument("--backup-margin", type=float, default=0.6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--filter-active", action="store_true",
                         help="Apply the trained safety filter's command (normal deployment) instead of "
                              "the raw nominal command. Default off (matches evaluate_training_maps_no_filter's "
                              "original no-filter behavior); pass this to evaluate cbf_cd_cf/bcbf_cf WITH "
                              "their filter active on the training maps (vanilla is unaffected either way).")
    args = parser.parse_args()

    checkpoints = {
        "vanilla": args.vanilla_checkpoint,
        "cbf_cd_cf": args.cbf_cd_cf_checkpoint,
        "bcbf_cf": args.bcbf_cf_checkpoint,
    }
    for method in args.methods:
        if checkpoints.get(method) is None:
            raise SystemExit(f"--{method.replace('_', '-')}-checkpoint is required for --methods {method}")

    procedural_map_bank = load_procedural_map_bank(
        args.procedural_map_bank,
        expected_num_maps=args.expected_num_maps,
        expected_scenarios_per_map=args.expected_scenarios_per_map,
        expected_max_obstacles=None,
    )
    seeds = list(range(args.seed, args.seed + args.num_seeds * args.seed_stride, args.seed_stride))
    manifest = build_manifest(procedural_map_bank, seeds, args.cases_per_seed)
    print(f"Manifest: {len(manifest)} cases ({len(seeds)} seeds x {args.cases_per_seed} cases/seed "
          f"x {args.expected_num_maps} maps)")

    common_env_kwargs = dict(
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
        action_max_vx=args.action_max_vx,
        action_min_vx=args.action_min_vx,
        action_max_vy=args.action_max_vy,
        action_max_wz=args.action_max_wz,
        backup_margin=args.backup_margin,
        low_checkpoint=args.low_level_checkpoint,
    )

    episodes: list[dict] = []
    for method in args.methods:
        method_env_kwargs = dict(common_env_kwargs)
        # All three seeds0-4-batch checkpoints (vanilla/cbf_cd_cf/bcbf_cf)
        # were trained with these terminal-set settings, confirmed via
        # load_actor's own config-mismatch check (matches the vxm1 3-map
        # eval scripts, which needed the same override for every method).
        method_env_kwargs["terminal_backup_mode"] = "invariant_ellipse"
        method_env_kwargs["legacy_projection"] = True
        evaluate_method(
            method=method,
            checkpoint_path=checkpoints[method],
            manifest=manifest,
            batch_size=min(args.num_envs, len(manifest)),
            episodes=episodes,
            filter_active=args.filter_active,
            **method_env_kwargs,
        )

    root = Path(args.run_root)
    root.mkdir(parents=True, exist_ok=True)
    write_rows(root / "episodes.csv", episodes)
    summary_rows = summarize(episodes)
    write_rows(root / "summary_by_map.csv", summary_rows)
    print(f"Saved {len(episodes)} episode rows to {root / 'episodes.csv'}")
    print(f"Saved summary to {root / 'summary_by_map.csv'}")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
