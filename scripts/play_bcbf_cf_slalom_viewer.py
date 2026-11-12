#!/usr/bin/env python3
"""Live MuJoCo-viewer playback of the bCBF-CF policy (safety filter ACTIVE,
normal deployment -- env.step_safe applies the filtered/safe command, exactly
as in every real bcbf_cf evaluation this session) on the actual slalom
(Easy) structured evaluation map.

Cycles through every SUCCESSFUL (evaluation_seed, case_id) recorded in the
prior vxm1 bcbf_cf eval run for the chosen map/seed -- each episode ends,
the viewer immediately teleports to the next successful case's start/goal,
looping back to the first once the list is exhausted -- so this plays a
back-to-back demo reel of real recorded successes, not a single case.

Reuses build_environment/load_actor from the structured unseen-map eval
script (same hand-crafted slalom obstacle layout, same [-1, 2] vx / +-1 vy /
+-1 wz / backup_margin=0.6 / invariant-ellipse terminal-set config as every
other bcbf_cf evaluation in this session) and DeterministicLaserPolicy from
the interactive play script, so the viewer sees exactly the same physical
policy behavior already measured numerically.
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
_play_module = _load_sibling_module(
    "_play_high_level_sac_laser_four_methods",
    "SAFTER_scripts/play_high_level_sac_laser_four_methods.py",
)

build_environment = _eval_module.build_environment
load_actor = _eval_module.load_actor
row_to_scenario = _eval_module.row_to_scenario

DeterministicLaserPolicy = _play_module.DeterministicLaserPolicy

from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer  # noqa: E402


class CyclingViewerLaserSacEnv:
    """Like play_high_level_sac_laser_four_methods.ViewerLaserSacEnv, but
    advances env.set_fixed_scenario() to the next case in a list every time
    the single env's episode ends, looping back to the start once
    exhausted."""

    def __init__(self, env, policy, scenarios, labels) -> None:
        self.env = env
        self.policy = policy
        self.unwrapped = env.unwrapped
        self.device = env.device
        self.num_envs = env.num_envs
        self.scenarios = scenarios
        self.labels = labels
        self.index = 0
        print(f"[CYCLE] case 1/{len(scenarios)}: {labels[0]}")

    def __getattr__(self, name):
        return getattr(self.env, name)

    def get_observations(self):
        return self.env.get_observations()

    def step(self, packed_action):
        nominal, safe = packed_action[:, :3], packed_action[:, 3:]
        safety = self.policy.last_safety
        if safety is None:
            raise RuntimeError("Viewer called env.step() before policy produced safety data.")
        obs, reward, done, extras = self.env.step_safe(nominal, safe, safety)
        if bool(done[0]):
            completed_ids = extras.get("completed_env_ids")
            outcome = "unknown"
            if completed_ids is not None and completed_ids.numel() > 0:
                success = bool(extras["completed_success"][0])
                collision = bool(extras["completed_collision"][0])
                timeout = bool(extras["completed_timeouts"][0])
                outcome = "success" if (success and not collision) else (
                    "collision" if collision else ("timeout" if timeout else "other")
                )
            print(f"[CYCLE] case {self.index + 1}/{len(self.scenarios)} "
                  f"({self.labels[self.index]}) ended: {outcome}")
            self.index = (self.index + 1) % len(self.scenarios)
            print(f"[CYCLE] advancing to case {self.index + 1}/{len(self.scenarios)}: {self.labels[self.index]}")
            self.env.set_fixed_scenario(self.scenarios[self.index])
        return obs, reward, done, extras


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", default="Unitree-Go2-Safe-Navigation")
    parser.add_argument("--low-level-checkpoint",
                         default="logs/rsl_rl/go2_velocity/2026-06-19_13-24-10/model_10000.pt")
    parser.add_argument("--bcbf-cf-checkpoint",
                         default="logs/sac_laser/go2_safe_navigation_sac_laser_bcbf_cf/"
                                  "2026-09-14_00-26-17_bcbf_cf_10hz_seaspeed_margin06_seed0/model_step_80000.pt")
    parser.add_argument("--map-name", default="slalom", choices=("slalom", "corridor", "circle_and_square"))
    parser.add_argument("--manifest-root", default=None,
                         help="Existing run root to pull a seed_*/​<map>/scenarios.csv case from. "
                              "Defaults to the vxm1 3-map eval run used throughout this session.")
    parser.add_argument("--bcbf-eval-root", default=None,
                         help="Existing bcbf_cf eval run root whose all_episodes.csv records which "
                              "cases succeeded. Defaults to the vxm1 bcbf_cf 3-map eval run.")
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=[1000],
                         help="Which evaluation seed(s) to pull successful cases from.")
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
    parser.add_argument("--laser-debug-vis", action=argparse.BooleanOptionalAction, default=True,
                         help="Draw the LiDAR rays in the viewer (default on).")
    parser.add_argument("--device", default=None)
    parser.add_argument("--viewer", choices=("auto", "native", "viser"), default="auto")
    parser.add_argument("--print-every", type=int, default=20)
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    manifest_root = Path(
        args.manifest_root
        or "logs/evaluation/laser_unseen_maps_structured_slalom_corridor_circle_and_square_seaspeed_vxm1/"
           "20260915_023930_seeds1000-1024_20perseed_500permap"
    )
    bcbf_eval_root = Path(
        args.bcbf_eval_root
        or "logs/evaluation/laser_unseen_maps_structured_slalom_corridor_circle_and_square_seaspeed_bcbf_vxm1/"
           "20260915_023933_seeds1000-1024_20perseed_500permap"
    )
    with (bcbf_eval_root / "all_episodes.csv").open(newline="") as f:
        outcome_rows = list(csv.DictReader(f))
    successful = {
        (int(r["evaluation_seed"]), int(r["case_id"]))
        for r in outcome_rows
        if r["method"] == "bcbf_cf" and r["map_name"] == args.map_name and r["outcome"] == "success"
    }

    cases: list[dict] = []
    for seed in args.evaluation_seeds:
        manifest_path = manifest_root / f"seed_{seed}" / args.map_name / "scenarios.csv"
        with manifest_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            if (seed, int(row["case_id"])) in successful:
                case = dict(row)
                case["evaluation_seed"] = seed
                cases.append(case)
    if not cases:
        raise SystemExit(
            f"No successful bcbf_cf cases found for map={args.map_name} seeds={args.evaluation_seeds}"
        )
    labels = [f"seed{c['evaluation_seed']}/case{c['case_id']}" for c in cases]
    print(f"[SCENARIOS] {len(cases)} successful bcbf_cf cases on map={args.map_name}, "
          f"seeds={args.evaluation_seeds}: {labels}")
    case = cases[0]

    raw_env, low_env, env, cfg, _ = build_environment(
        task_id=args.task_id,
        low_checkpoint=args.low_level_checkpoint,
        method="bcbf_cf",
        episode_duration_s=args.episode_duration_s,
        device=device,
        map_name=args.map_name,
        num_envs=1,
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
        laser_debug_vis=args.laser_debug_vis,
    )
    scenarios = [row_to_scenario(c) for c in cases]
    env.set_fixed_scenario(scenarios[0])

    actor = load_actor(env=env, cfg=cfg, checkpoint_path=args.bcbf_cf_checkpoint, device=device)
    actor.eval()
    print(f"[LASER ACTOR] method=bcbf_cf (filter ACTIVE) checkpoint={args.bcbf_cf_checkpoint} "
          f"map={args.map_name}")

    policy = DeterministicLaserPolicy(env, actor, args.print_every)
    viewer_env = CyclingViewerLaserSacEnv(env, policy, scenarios, labels)
    viewer = args.viewer
    if viewer == "auto":
        import os
        viewer = "native" if (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) else "viser"
    print(f"[VIEWER] mode={viewer}")
    if viewer == "native":
        NativeMujocoViewer(viewer_env, policy).run()
    else:
        ViserPlayViewer(viewer_env, policy).run()
    raw_env.close()


if __name__ == "__main__":
    main()
