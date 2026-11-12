#!/usr/bin/env python3
"""Play one saved unseen-map case with optional local obstacle sensing.

This is a deployment/robustness variant of play_selected_unseen_case_v2.py.
It keeps the trained SAC checkpoints unchanged, but at test time it can mask
obstacles outside a finite sensing radius from both:

  1. the actor observation obstacle slots, and
  2. the CBF/bCBF safety-filter obstacle mask.

Collision/reward/termination still use the true full obstacle map. Therefore,
unknown obstacles can still cause collisions; they are simply not available to
policy/filter until they enter the sensing area.

Example:
    python scripts/play_selected_unseen_case_local_sensing.py \
      --method bcbf_rl \
      --map slalom \
      --case-id 13 \
      --manifest-root logs/evaluation/unseen_maps/20260703_122438_seed42_20permap \
      --low-level-checkpoint logs/rsl_rl/go2_velocity/2026-06-19_13-24-10/model_10000.pt \
      --high-level-checkpoint logs/sac/v4_policies/bcbf_v4.pt \
      --sensing-mode local \
      --sensing-radius 3.0 \
      --viewer native
"""
from __future__ import annotations

import argparse
import csv
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import torch

# Register tasks before building environments.
import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401

from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

import evaluate_unseen_maps_all_methods as unseen_eval


METHODS = ("no_cbf", "cbf_rl", "bcbf_rl")
SENSING_MODES = ("full", "local")


def read_case(manifest_root: Path, map_name: str, case_id: int) -> dict[str, object]:
    path = manifest_root / map_name / "scenarios.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find scenarios.csv for map={map_name!r}: {path}"
        )

    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    for row in rows:
        if int(row["case_id"]) == int(case_id):
            return row

    available = [int(row["case_id"]) for row in rows]
    raise ValueError(
        f"case_id={case_id} was not found in {path}. "
        f"Available case IDs: {available}"
    )


class LocalObstacleSensing:
    """Switch env/actor obstacle masks between true map and local perception.

    The environment uses ``env.obstacle_mask`` in two places:
      * observation construction, and
      * collision/reward/termination checks.

    For local sensing, we temporarily replace the env mask only while computing
    observations. Before stepping the environment, we restore the true mask so
    collision checks remain physically correct.

    The actor has its own registered obstacle-mask buffer for the CBF/bCBF
    safety constraints. That buffer is set to the same local visibility mask
    before calling actor.deterministic(), so safety constraints are only applied
    to visible obstacles.
    """

    def __init__(
        self,
        env,
        actor,
        *,
        mode: str = "full",
        radius: float = 3.0,
        use_surface_distance: bool = True,
        print_every: int = 0,
    ) -> None:
        if mode not in SENSING_MODES:
            raise ValueError(f"Unknown sensing mode {mode!r}; expected {SENSING_MODES}")

        self.env = env
        self.actor = actor
        self.mode = mode
        self.radius = float(radius)
        self.use_surface_distance = bool(use_surface_distance)
        self.print_every = max(int(print_every), 0)
        self.steps = 0

        self.true_env_mask = env.obstacle_mask.detach().clone()
        self.true_actor_mask = actor.obstacle_mask.detach().clone()
        self.last_local_mask = self.true_env_mask.detach().clone()

    def compute_local_mask(self) -> torch.Tensor:
        """Return env-shaped visible obstacle mask, shape [num_envs, max_obstacles]."""
        if self.mode == "full":
            return self.true_env_mask.detach().clone()

        position, _ = self.env.safety_context()
        centers = self.env.obstacle_centers
        radii = self.env.obstacle_radii

        center_distance = torch.linalg.norm(
            centers - position[:, None, :],
            dim=-1,
        )

        if self.use_surface_distance:
            # Obstacle becomes visible when its nearest surface enters the
            # sensing ball: ||p - c_i|| - r_i <= R.
            visible = (center_distance - radii) <= self.radius
        else:
            # Stricter/simple model: obstacle center must enter sensing radius.
            visible = center_distance <= self.radius

        return self.true_env_mask & visible

    def set_actor_mask(self, local_mask: torch.Tensor) -> None:
        """Set the actor's CBF/bCBF obstacle mask to the locally visible mask."""
        actor_mask = local_mask[: self.actor.obstacle_mask.shape[0]]
        if actor_mask.shape != self.actor.obstacle_mask.shape:
            actor_mask = actor_mask[:1].expand_as(self.actor.obstacle_mask)
        self.actor.obstacle_mask.copy_(actor_mask)

    def restore_true_env_mask(self) -> None:
        self.env.obstacle_mask.copy_(self.true_env_mask)

    def restore_true_actor_mask(self) -> None:
        self.actor.obstacle_mask.copy_(self.true_actor_mask)

    @contextmanager
    def observation_context(self):
        """Temporarily use local mask for observation construction."""
        local_mask = self.compute_local_mask()
        self.last_local_mask = local_mask.detach().clone()
        old_env_mask = self.env.obstacle_mask.detach().clone()
        self.env.obstacle_mask.copy_(local_mask)
        try:
            yield local_mask
        finally:
            self.env.obstacle_mask.copy_(old_env_mask)

    def get_local_observations(self):
        """Compute actor observations with only locally visible obstacles."""
        with self.observation_context():
            return self.env.get_observations()

    def prepare_actor_for_current_step(self) -> torch.Tensor:
        """Set actor CBF/bCBF constraints to currently visible obstacles."""
        local_mask = self.compute_local_mask()
        self.last_local_mask = local_mask.detach().clone()
        self.set_actor_mask(local_mask)
        return local_mask

    def post_step_local_observation(self):
        """Return a local observation after env.step_safe() used true collision mask."""
        return self.get_local_observations()

    def maybe_print(self) -> None:
        self.steps += 1
        if not self.print_every or self.steps % self.print_every != 0:
            return
        local = self.last_local_mask[0]
        true = self.true_env_mask[0]
        n_visible = int((local & true).sum().detach().cpu())
        n_total = int(true.sum().detach().cpu())
        print(
            f"[local sensing] step={self.steps:05d} "
            f"mode={self.mode} radius={self.radius:.2f} "
            f"visible={n_visible}/{n_total}",
            flush=True,
        )


class DeterministicViewerPolicy:
    def __init__(
        self,
        env,
        actor,
        method: str,
        sensing: LocalObstacleSensing,
        print_every: int = 0,
    ) -> None:
        self.env = env
        self.actor = actor
        self.method = method
        self.sensing = sensing
        self.print_every = max(int(print_every), 0)
        self.steps = 0
        self.last_safety = None

    @torch.no_grad()
    def __call__(self, obs_dict):
        # The viewer may pass the observation returned by the previous step.
        # For robustness, always build a fresh local observation at the current
        # robot state, so the actor input and CBF/bCBF mask are synchronized.
        obs_dict = self.sensing.get_local_observations()
        obs = self.env.actor_tensor(obs_dict)
        position, yaw = self.env.safety_context()

        # Critical: CBF and bCBF constraints use only visible obstacles.
        self.sensing.prepare_actor_for_current_step()
        out = self.actor.deterministic(obs, position, yaw)
        self.last_safety = out.safety

        self.steps += 1
        self.sensing.maybe_print()
        if self.print_every and self.steps % self.print_every == 0:
            correction = torch.linalg.norm(out.safe_command[0] - out.nominal_command[0])
            active = False
            if out.safety is not None and hasattr(out.safety, "active"):
                active = bool(out.safety.active[0])
            else:
                active = float(correction.detach().cpu()) > 1.0e-6

            local = self.sensing.last_local_mask[0]
            true = self.sensing.true_env_mask[0]
            n_visible = int((local & true).sum().detach().cpu())
            n_total = int(true.sum().detach().cpu())

            print(
                f"[{self.method}] step={self.steps:05d} "
                f"visible={n_visible}/{n_total} "
                f"nominal={out.nominal_command[0].detach().cpu().numpy()} "
                f"safe={out.safe_command[0].detach().cpu().numpy()} "
                f"active={active} "
                f"correction={float(correction.detach().cpu()):.4f}",
                flush=True,
            )

        # Viewer calls env.step(action). Pack nominal and safe commands so the
        # wrapper below can preserve the exact shielded command path.
        return torch.cat((out.nominal_command, out.safe_command), dim=-1)


class ViewerSafeCommandLocalSensingEnv:
    """Adapter exposing the usual viewer step(action) interface.

    Safety metadata is computed in the policy using local obstacle constraints.
    During the physical step, the environment obstacle mask is restored to the
    true full map so collision, wall, reward, and termination metrics are not
    fooled by hidden obstacles.
    """

    def __init__(
        self,
        env,
        policy: DeterministicViewerPolicy,
        sensing: LocalObstacleSensing,
    ) -> None:
        self.env = env
        self.policy = policy
        self.sensing = sensing
        self.unwrapped = env.unwrapped
        self.device = env.device
        self.num_envs = env.num_envs

    def __getattr__(self, name):
        return getattr(self.env, name)

    def get_observations(self):
        return self.sensing.get_local_observations()

    def step(self, packed_action):
        nominal = packed_action[:, :3]
        safe = packed_action[:, 3:]

        safety = self.policy.last_safety
        if safety is None:
            raise RuntimeError(
                "Viewer called env.step() before the policy produced safety data."
            )

        # True map for collision/reward/termination.
        self.sensing.restore_true_env_mask()
        result = self.env.step_safe(nominal, safe, safety)

        # Return a local observation for callers that reuse step()'s obs.
        if isinstance(result, tuple) and len(result) == 4:
            _obs, reward, done, extras = result
            return self.sensing.post_step_local_observation(), reward, done, extras

        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play a saved unseen-map case with optional local obstacle sensing."
    )
    parser.add_argument(
        "task_id",
        nargs="?",
        default="Unitree-Go2-Safe-Navigation",
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=METHODS,
        help="Policy/filter mode to play.",
    )
    parser.add_argument(
        "--map",
        dest="map_name",
        required=True,
        choices=list(unseen_eval.UNSEEN_MAPS),
        help="Unseen map name.",
    )
    parser.add_argument(
        "--case-id",
        required=True,
        type=int,
        help="Case ID from the saved scenarios.csv manifest.",
    )
    parser.add_argument(
        "--manifest-root",
        default="logs/evaluation/unseen_maps_100ep_v4",
        help="Folder containing map/scenarios.csv files.",
    )
    parser.add_argument(
        "--low-level-checkpoint",
        default=unseen_eval.DEFAULT_LOW,
    )
    parser.add_argument(
        "--high-level-checkpoint",
        required=True,
        help="Path to nocbf_v4.pt, cbf_v4.pt, or bcbf_v4.pt.",
    )
    parser.add_argument(
        "--episode-duration-s",
        type=float,
        default=60.0,
    )
    parser.add_argument(
        "--device",
        default=None,
    )
    parser.add_argument(
        "--viewer",
        choices=("auto", "native", "viser"),
        default="native",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=0,
        help="Print policy/safety diagnostics every N high-level steps.",
    )
    parser.add_argument(
        "--sensing-mode",
        choices=SENSING_MODES,
        default="local",
        help="full uses the full obstacle map; local masks out obstacles outside --sensing-radius.",
    )
    parser.add_argument(
        "--sensing-radius",
        type=float,
        default=3.0,
        help="Local sensing radius in meters.",
    )
    parser.add_argument(
        "--center-distance-only",
        action="store_true",
        help="Use center distance <= R instead of surface distance <= R for visibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    method = args.method
    map_name = args.map_name
    case_id = int(args.case_id)

    case = read_case(Path(args.manifest_root), map_name, case_id)

    print("=" * 72)
    print(f"method:       {method}")
    print(f"map:          {map_name}")
    print(f"case:         {case_id}")
    print(f"seed:         {case['reset_seed']}")
    print(
        "start:       "
        f"({float(case['start_x']):+.3f}, {float(case['start_y']):+.3f}), "
        f"yaw={float(case['start_yaw']):+.3f}"
    )
    print(
        "goal:        "
        f"({float(case['goal_x']):+.3f}, {float(case['goal_y']):+.3f})"
    )
    print(f"sensing:      {args.sensing_mode}")
    if args.sensing_mode == "local":
        visibility = "center distance" if args.center_distance_only else "surface distance"
        print(f"radius:       {args.sensing_radius:.3f} m ({visibility})")
    print(f"device:       {device}")
    print("=" * 72)

    module = unseen_eval.bcbf_eval if method == "bcbf_rl" else unseen_eval.sac_eval

    unseen_eval.set_seed(int(case["reset_seed"]))

    raw_env, low_env, env, config = unseen_eval.build_environment(
        module=module,
        task_id=args.task_id,
        low_checkpoint=args.low_level_checkpoint,
        episode_duration_s=args.episode_duration_s,
        device=device,
        map_name=map_name,
    )

    try:
        unseen_eval.verify_compiled_physical_map(raw_env, env, map_name)

        scenario = unseen_eval.row_to_scenario(case)
        env.set_fixed_scenario(scenario)

        actor = unseen_eval.build_actor(
            method=method,
            module=module,
            env=env,
            config=config,
            checkpoint_path=args.high_level_checkpoint,
            device=device,
        )

        sensing = LocalObstacleSensing(
            env,
            actor,
            mode=args.sensing_mode,
            radius=args.sensing_radius,
            use_surface_distance=not args.center_distance_only,
            print_every=args.print_every,
        )

        # Force one local observation after the fixed scenario is installed.
        _ = env.actor_tensor(sensing.get_local_observations())

        policy = DeterministicViewerPolicy(
            env,
            actor,
            method=method,
            sensing=sensing,
            print_every=args.print_every,
        )
        viewer_env = ViewerSafeCommandLocalSensingEnv(env, policy, sensing)

        viewer: Literal["native", "viser"]
        if args.viewer == "auto":
            viewer = (
                "native"
                if (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
                else "viser"
            )
        else:
            viewer = args.viewer

        if viewer == "native":
            NativeMujocoViewer(viewer_env, policy).run()
        else:
            ViserPlayViewer(viewer_env, policy).run()

    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
