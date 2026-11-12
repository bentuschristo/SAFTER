#!/usr/bin/env python3
"""Filter-swap ablation on the three structured evaluation maps (Easy=slalom,
Medium=corridor, Hard=circle_and_square), 25 seeds x 20 cases/seed = 500
cases/map, matching the corrected [-1, 2] vx-bound checkpoint generation:

  - vanilla:   FILTER DEPLOYED. The vanilla-trained policy's nominal command
               is passed through its own actor's `counterfactual_filter`
               (the plain, non-backup direct-laser LSE-CBF-CF -- the same
               filter object every actor builds for diagnostics regardless
               of method) at DEPLOYMENT time, retrofitting a safety filter
               onto a policy that never trained with one.
  - cbf_cd_cf: FILTER INACTIVE. The physically-applied command is the raw
               nominal actor output, bypassing the trained CBF-CF filter
               entirely (same swap as evaluate_training_maps_no_filter.py).
  - bcbf_cf:   FILTER INACTIVE, same as cbf_cd_cf.

Reuses the existing scenario manifests from the prior vxm1 3-map eval runs
(scenario sampling is method-independent -- generated once with
method="vanilla" regardless of which method is later evaluated -- so those
CSVs are the exact same cases already used for the filter-on numbers this
ablates against). No trajectory logging -- ids + outcome only.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_sibling_module(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_eval_module = _load_sibling_module(
    "_evaluate_laser_unseen_maps_structured",
    "SAFTER_scripts/laser_unseen_eval/evaluate_laser_unseen_maps_structured.py",
)

# 2026-11 maps restructure: the structured evaluator's UNSEEN_MAPS/
# STRUCTURED_SCENARIO_REGIONS now only carry the 3 current evaluation maps
# ("low_shift"/"medium_shift"/"high_shift", formerly "slalom"/"corridor"/
# "corridor_derived_highshift_v5"). This ablation script still keys its
# on-disk scenario manifests (see load_manifest below) by the OLD names
# ("slalom", "corridor", "circle_and_square") from the prior vxm1 3-map eval
# runs, so those old map keys are restored here -- "slalom"/"corridor" as
# aliases of the (byte-identical, just renamed) "low_shift"/"medium_shift"
# data, and "circle_and_square" (which has no equivalent among the new 3
# maps -- it was never renamed, just retired) from legacy_unseen_maps.py --
# so build_environment(map_name=...) still resolves exactly as it always did.
from SAFTER_scripts.laser_unseen_eval.legacy_unseen_maps import (  # noqa: E402
    LEGACY_STRUCTURED_SCENARIO_REGIONS,
    LEGACY_UNSEEN_MAPS,
)

_eval_module.UNSEEN_MAPS["slalom"] = _eval_module.UNSEEN_MAPS["low_shift"]
_eval_module.STRUCTURED_SCENARIO_REGIONS["slalom"] = _eval_module.STRUCTURED_SCENARIO_REGIONS["low_shift"]
_eval_module.UNSEEN_MAPS["corridor"] = _eval_module.UNSEEN_MAPS["medium_shift"]
_eval_module.STRUCTURED_SCENARIO_REGIONS["corridor"] = _eval_module.STRUCTURED_SCENARIO_REGIONS["medium_shift"]
_eval_module.UNSEEN_MAPS["circle_and_square"] = LEGACY_UNSEEN_MAPS["circle_and_square"]
_eval_module.STRUCTURED_SCENARIO_REGIONS["circle_and_square"] = LEGACY_STRUCTURED_SCENARIO_REGIONS["circle_and_square"]

build_environment = _eval_module.build_environment
load_actor = _eval_module.load_actor
row_to_scenario = _eval_module.row_to_scenario
timed_deterministic_actor = _eval_module.timed_deterministic_actor
warmup_actor_timing = _eval_module.warmup_actor_timing
set_seed = _eval_module.set_seed
mean_stats = _eval_module.mean_stats
proportion_stats = _eval_module.proportion_stats
write_rows = _eval_module.write_rows

from SAFTER_src.high_level_rl.laser_cbf_filters import (  # noqa: E402
    LaserScanCbfControlDynamicsClosedForm,
    LaserZeroBackupLseCbfClosedForm,
)

MAPS = ("slalom", "corridor", "circle_and_square")
ABLATION_MODE = {"vanilla": "filter_deployed", "cbf_cd_cf": "filter_inactive", "bcbf_cf": "filter_inactive"}


def build_retrofit_filter(filter_type: str, cfg, action_limits: tuple[float, float, float]):
    """Construct a filter object matching exactly what LaserSafeSacActor.__init__
    builds for method=filter_type -- these filters have no learned/trained
    parameters (pure analytic CBF math), so they can be applied to ANY
    policy's nominal command, not only the one that was actually trained
    with that filter attached."""
    safety_cfg = cfg.safety
    kwargs = dict(
        d_safe=safety_cfg.d_safe,
        max_vx=action_limits[0],
        max_vy=action_limits[1],
        max_wz=action_limits[2],
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
    if filter_type == "cbf_cd_cf":
        return LaserScanCbfControlDynamicsClosedForm(
            hocbf_k0=safety_cfg.hocbf_k0,
            alpha_outer=safety_cfg.alpha_outer,
            kappa=safety_cfg.kappa,
            kappa_input=safety_cfg.kappa_input,
            kappa_outer=safety_cfg.kappa_outer,
            barrier_rate_scale=safety_cfg.barrier_rate_scale,
            control_dynamics_a=safety_cfg.control_dynamics_a,
            tracking_sigma=safety_cfg.tracking_sigma,
            integration_dt=safety_cfg.integration_dt,
            **kwargs,
        )
    if filter_type == "bcbf_cf":
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
            **kwargs,
        )
    raise ValueError(f"Unsupported retrofit filter_type={filter_type!r}")


def load_manifest(manifest_root: Path, map_name: str, seed: int) -> list[dict]:
    path = manifest_root / f"seed_{seed}" / map_name / "scenarios.csv"
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def evaluate(
    *,
    method: str,
    checkpoint_path: str,
    map_name: str,
    manifests_by_seed: dict[int, list[dict]],
    batch_size: int,
    episodes: list[dict],
    vanilla_filter_type: str = "cbf_cf",
    **env_kwargs,
) -> None:
    mode = ABLATION_MODE[method]
    all_cases = []
    for seed, rows in manifests_by_seed.items():
        for row in rows:
            case = dict(row)
            case["evaluation_seed"] = seed
            all_cases.append(case)

    set_seed(int(all_cases[0]["evaluation_seed"]))
    raw_env, low_env, env, cfg, _ = build_environment(
        method=method, map_name=map_name, num_envs=batch_size, **env_kwargs
    )
    device = env_kwargs["device"]
    try:
        actor = load_actor(env=env, cfg=cfg, checkpoint_path=checkpoint_path, device=device)
        warmup_actor_timing(actor, env)
        dt = float(env.navigation_cfg.high_level_dt)

        retrofit_filter = None
        if mode == "filter_deployed" and vanilla_filter_type != "cbf_cf":
            action_limits = (cfg.navigation.action.max_vx, cfg.navigation.action.max_vy, cfg.navigation.action.max_wz)
            retrofit_filter = build_retrofit_filter(vanilla_filter_type, cfg, action_limits)

        for batch_start in range(0, len(all_cases), batch_size):
            real_cases = all_cases[batch_start : batch_start + batch_size]
            padded_cases = list(real_cases)
            while len(padded_cases) < batch_size:
                padded_cases.append(real_cases[-1])
            env.set_fixed_scenarios([row_to_scenario(case) for case in padded_cases])
            obs = env.get_observations()

            active_count = len(real_cases)
            finished = [False] * active_count

            with torch.no_grad():
                while not all(finished):
                    actor_obs = env.actor_tensor(obs)
                    nominal, safe, safety, _, _, _ = timed_deterministic_actor(actor, actor_obs)
                    if mode == "filter_deployed":
                        ranges = actor._ranges(actor_obs)
                        angles = actor.ray_angles.to(device=actor_obs.device, dtype=actor_obs.dtype)
                        if vanilla_filter_type == "cbf_cf":
                            retrofit = actor.counterfactual_filter.filter(ranges, angles, nominal)
                        else:
                            retrofit = retrofit_filter.filter(
                                ranges, angles, actor_obs[:, 5:8], nominal, actor_obs[:, -4:-1], actor_obs[:, -1:]
                            )
                        applied = retrofit.command
                        if actor.hard_clip_output:
                            applied = torch.clamp(applied, actor.action_low, actor.action_high)
                    else:
                        applied = nominal
                    obs, reward, done, extras = env.step_safe(nominal, applied, safety)

                    completed_ids = extras.get(
                        "completed_env_ids", torch.empty(0, device=env.device, dtype=torch.long)
                    )
                    completed_lookup = {int(env_id): row for row, env_id in enumerate(completed_ids.tolist())}

                    for env_id, case in enumerate(real_cases):
                        if finished[env_id]:
                            continue
                        completed_row = completed_lookup.get(env_id)
                        if completed_row is None:
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
                            "map_name": map_name,
                            "evaluation_seed": int(case["evaluation_seed"]),
                            "case_id": int(case["case_id"]),
                            "outcome": outcome,
                            "success": int(success),
                            "collision": int(collision),
                            "timeout": int(timeout),
                            "fallen": int(fallen),
                            "outside": int(outside),
                            "return": final_return,
                            "episode_duration_s": duration_s,
                        })
                        finished[env_id] = True
            print(f"[{method}/{map_name}] batch {batch_start}-{batch_start + len(real_cases)}/{len(all_cases)} done")
    finally:
        raw_env.close()


def summarize(episodes: list[dict]) -> list[dict]:
    import statistics

    rows = []
    methods = sorted({e["method"] for e in episodes})
    maps = sorted({e["map_name"] for e in episodes}, key=MAPS.index)
    for method in methods:
        for map_name in maps:
            group = [e for e in episodes if e["method"] == method and e["map_name"] == map_name]
            if not group:
                continue
            row = {"method": method, "map_name": map_name, "cases": len(group)}
            for field in ("success", "collision", "timeout", "fallen"):
                row.update(proportion_stats(sum(int(e[field]) for e in group), len(group), f"{field}_rate"))
            row.update(mean_stats([float(e["return"]) for e in group], "return"))
            seeds = sorted({e["evaluation_seed"] for e in group})
            for field in ("success", "collision", "timeout"):
                per_seed_rates = []
                for seed in seeds:
                    seed_group = [e for e in group if e["evaluation_seed"] == seed]
                    per_seed_rates.append(sum(int(e[field]) for e in seed_group) / len(seed_group))
                row[f"{field}_rate_seed_mean_pct"] = 100.0 * statistics.mean(per_seed_rates)
                row[f"{field}_rate_seed_std_pct"] = (
                    100.0 * statistics.stdev(per_seed_rates) if len(per_seed_rates) > 1 else 0.0
                )
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", default="Unitree-Go2-Safe-Navigation")
    parser.add_argument("--low-level-checkpoint", required=True)
    parser.add_argument("--vanilla-checkpoint", required=True)
    parser.add_argument("--cbf-cd-cf-checkpoint", required=True)
    parser.add_argument("--bcbf-cf-checkpoint", required=True)
    parser.add_argument("--manifest-root", required=True,
                         help="Existing run root whose seed_*/​<map>/scenarios.csv files to reuse.")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--num-seeds", type=int, default=25)
    parser.add_argument("--seed-stride", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=50)
    parser.add_argument("--episode-duration-s", type=float, default=60.0)
    parser.add_argument("--laser-num-rays", type=int, default=120)
    parser.add_argument("--laser-fov-deg", type=float, default=360.0)
    parser.add_argument("--collision-geometry", default="circle")
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
    parser.add_argument("--methods", nargs="+", default=["vanilla", "cbf_cd_cf", "bcbf_cf"],
                         choices=["vanilla", "cbf_cd_cf", "bcbf_cf"])
    parser.add_argument("--vanilla-filter-type", default="cbf_cf", choices=["cbf_cf", "cbf_cd_cf", "bcbf_cf"],
                         help="Which filter class to retrofit onto vanilla's nominal command "
                              "(only relevant when 'vanilla' is in --methods).")
    args = parser.parse_args()

    checkpoints = {
        "vanilla": args.vanilla_checkpoint,
        "cbf_cd_cf": args.cbf_cd_cf_checkpoint,
        "bcbf_cf": args.bcbf_cf_checkpoint,
    }
    seeds = list(range(args.seed, args.seed + args.num_seeds * args.seed_stride, args.seed_stride))
    manifest_root = Path(args.manifest_root)

    common_env_kwargs = dict(
        task_id=args.task_id,
        low_checkpoint=args.low_level_checkpoint,
        episode_duration_s=args.episode_duration_s,
        device=args.device,
        laser_num_rays=args.laser_num_rays,
        laser_fov_deg=args.laser_fov_deg,
        enable_lidar_noise=True,
        hard_clip_output=True,
        collision_geometry=args.collision_geometry,
        collision_circle_radius_m=args.collision_circle_radius_m,
        collision_margin_m=args.collision_margin_m,
        collision_termination_mode=args.collision_termination_mode,
        qp_backend=args.qp_backend,
        terminal_backup_mode="invariant_ellipse",
        terminal_geom_scale=1.0,
        terminal_rec_scale=1.0,
        legacy_projection=True,
        action_max_vx=args.action_max_vx,
        action_min_vx=args.action_min_vx,
        action_max_vy=args.action_max_vy,
        action_max_wz=args.action_max_wz,
        backup_margin=args.backup_margin,
    )

    episodes: list[dict] = []
    for method in args.methods:
        for map_name in MAPS:
            manifests_by_seed = {seed: load_manifest(manifest_root, map_name, seed) for seed in seeds}
            evaluate(
                method=method,
                checkpoint_path=checkpoints[method],
                map_name=map_name,
                manifests_by_seed=manifests_by_seed,
                batch_size=args.num_envs,
                episodes=episodes,
                vanilla_filter_type=args.vanilla_filter_type,
                **common_env_kwargs,
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
