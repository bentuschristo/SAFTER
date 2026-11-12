#!/usr/bin/env python3
"""Live MuJoCo-viewer playback of the bCBF-CF policy (safety filter ACTIVE,
normal deployment) on one of the 5 TRAINING procedural maps
(generated/procedural_maps/handcrafted2), with LiDAR rays visualized.

Cycles through every SUCCESSFUL (evaluation_seed, case_id, scenario_id)
recorded in the prior training_maps_filter_active eval run for the chosen
map_id -- each episode ends, the viewer immediately teleports to the next
successful case, looping back to the first once the list is exhausted.

Reuses build_environment/load_actor from evaluate_training_maps_no_filter.py
(same procedural map bank machinery, same [-1, 2] vx / +-1 vy / +-1 wz /
backup_margin=0.6 / invariant-ellipse terminal-set config as every other
bcbf_cf evaluation this session) and DeterministicLaserPolicy from the
interactive play script.
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


_train_no_filter_module = _load_sibling_module(
    "_evaluate_training_maps_no_filter", "SAFTER_scripts/evaluate_training_maps_no_filter.py"
)
_play_module = _load_sibling_module(
    "_play_high_level_sac_laser_four_methods", "SAFTER_scripts/play_high_level_sac_laser_four_methods.py"
)

build_environment = _train_no_filter_module.build_environment
load_actor = _train_no_filter_module.load_actor

DeterministicLaserPolicy = _play_module.DeterministicLaserPolicy

from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer  # noqa: E402


class CyclingProceduralViewerEnv:
    """Like play_high_level_sac_laser_four_methods.ViewerLaserSacEnv, but
    advances env.set_fixed_procedural_scenarios() to the next scenario_id in
    a list every time the single env's episode ends, looping back to the
    start once exhausted."""

    def __init__(self, env, policy, scenario_ids: list[int], labels: list[str]) -> None:
        self.env = env
        self.policy = policy
        self.unwrapped = env.unwrapped
        self.device = env.device
        self.num_envs = env.num_envs
        self.scenario_ids = scenario_ids
        self.labels = labels
        self.index = 0
        print(f"[CYCLE] case 1/{len(scenario_ids)}: {labels[0]}")

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
            print(f"[CYCLE] case {self.index + 1}/{len(self.scenario_ids)} "
                  f"({self.labels[self.index]}) ended: {outcome}")
            self.index = (self.index + 1) % len(self.scenario_ids)
            print(f"[CYCLE] advancing to case {self.index + 1}/{len(self.scenario_ids)}: "
                  f"{self.labels[self.index]}")
            next_scenario = torch.tensor([self.scenario_ids[self.index]], device=self.env.device, dtype=torch.long)
            self.env.set_fixed_procedural_scenarios(next_scenario)
        return obs, reward, done, extras


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-level-checkpoint",
                         default="logs/rsl_rl/go2_velocity/2026-06-19_13-24-10/model_10000.pt")
    parser.add_argument("--bcbf-cf-checkpoint",
                         default="logs/sac_laser/go2_safe_navigation_sac_laser_bcbf_cf/"
                                  "2026-09-14_00-26-17_bcbf_cf_10hz_seaspeed_margin06_seed0/model_step_80000.pt")
    parser.add_argument("--map-id", type=int, default=3, help="Which training map (0-4).")
    parser.add_argument("--procedural-map-bank", default="generated/procedural_maps/handcrafted2")
    parser.add_argument("--expected-num-maps", type=int, default=5)
    parser.add_argument("--expected-scenarios-per-map", type=int, default=200)
    parser.add_argument("--eval-root", default=None,
                         help="Existing training_maps_filter_active run whose episodes.csv records "
                              "which (map_id, scenario_id) cases succeeded. Defaults to the most "
                              "recent such run used this session.")
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=None,
                         help="Restrict to specific evaluation seeds (default: all seeds present).")
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
    eval_root = Path(
        args.eval_root
        or "logs/evaluation/training_maps_filter_active/20260916_131126_seeds1000-1024_20perseed_500permap"
    )
    with (eval_root / "episodes.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    successful = [
        r for r in rows
        if r["method"] == "bcbf_cf" and int(r["map_id"]) == args.map_id and r["outcome"] == "success"
    ]
    if args.evaluation_seeds is not None:
        wanted = set(args.evaluation_seeds)
        successful = [r for r in successful if int(r["evaluation_seed"]) in wanted]
    if not successful:
        raise SystemExit(f"No successful bcbf_cf cases found for map_id={args.map_id} in {eval_root}")
    scenario_ids = [int(r["scenario_id"]) for r in successful]
    labels = [f"seed{r['evaluation_seed']}/case{r['case_id']}(scenario{r['scenario_id']})" for r in successful]
    print(f"[SCENARIOS] {len(scenario_ids)} successful bcbf_cf cases on map_id={args.map_id}: {labels}")

    raw_env, env, cfg, _ = build_environment(
        method="bcbf_cf",
        low_checkpoint=args.low_level_checkpoint,
        episode_duration_s=args.episode_duration_s,
        device=device,
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
        num_envs=1,
        terminal_backup_mode="invariant_ellipse",
        legacy_projection=True,
        action_max_vx=args.action_max_vx,
        action_min_vx=args.action_min_vx,
        action_max_vy=args.action_max_vy,
        action_max_wz=args.action_max_wz,
        backup_margin=args.backup_margin,
        laser_debug_vis=args.laser_debug_vis,
    )
    first_scenario = torch.tensor([scenario_ids[0]], device=device, dtype=torch.long)
    env.set_fixed_procedural_scenarios(first_scenario)

    actor = load_actor(env=env, cfg=cfg, checkpoint_path=args.bcbf_cf_checkpoint, device=device)
    actor.eval()
    print(f"[LASER ACTOR] method=bcbf_cf (filter ACTIVE) checkpoint={args.bcbf_cf_checkpoint} "
          f"map_id={args.map_id}")

    policy = DeterministicLaserPolicy(env, actor, args.print_every)
    viewer_env = CyclingProceduralViewerEnv(env, policy, scenario_ids, labels)
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
