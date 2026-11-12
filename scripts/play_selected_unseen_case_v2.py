#!/usr/bin/env python3
"""Play one fixed unseen-map case (v2, viewer-safe metadata reuse) in the MuJoCo viewer.

This reuses the same environment/actor construction utilities as
evaluate_unseen_maps_all_methods.py, but instead of regenerating random
cases, it reads the saved scenarios.csv from a previous unseen-map evaluation.

Example:
    python scripts/play_selected_unseen_case.py \
      --method bcbf_rl \
      --map slalom \
      --case-id 13 \
      --manifest-root logs/evaluation/unseen_maps_100ep_v4 \
      --low-level-checkpoint logs/rsl_rl/go2_velocity/2026-06-19_13-24-10/model_10000.pt \
      --high-level-checkpoint logs/sac/v4_policies/bcbf_v4.pt \
      --viewer native
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Literal

import torch

# Register tasks before building environments.
import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401

from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

import evaluate_unseen_maps_all_methods as unseen_eval


METHODS = ("no_cbf", "cbf_rl", "bcbf_rl")


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


class DeterministicViewerPolicy:
    def __init__(self, env, actor, method: str, print_every: int = 0) -> None:
        self.env = env
        self.actor = actor
        self.method = method
        self.print_every = max(int(print_every), 0)
        self.steps = 0
        self.last_safety = None

    @torch.no_grad()
    def __call__(self, obs_dict):
        obs = self.env.actor_tensor(obs_dict)
        position, yaw = self.env.safety_context()
        out = self.actor.deterministic(obs, position, yaw)
        self.last_safety = out.safety

        self.steps += 1
        if self.print_every and self.steps % self.print_every == 0:
            correction = torch.linalg.norm(out.safe_command[0] - out.nominal_command[0])
            active = False
            if out.safety is not None and hasattr(out.safety, "active"):
                active = bool(out.safety.active[0])
            else:
                active = float(correction.detach().cpu()) > 1.0e-6

            print(
                f"[{self.method}] step={self.steps:05d} "
                f"nominal={out.nominal_command[0].detach().cpu().numpy()} "
                f"safe={out.safe_command[0].detach().cpu().numpy()} "
                f"active={active} "
                f"correction={float(correction.detach().cpu()):.4f}",
                flush=True,
            )

        # Viewer calls env.step(action). Pack nominal and safe commands so the
        # wrapper below can preserve the exact shielded command path.
        return torch.cat((out.nominal_command, out.safe_command), dim=-1)


class ViewerSafeCommandEnv:
    """Adapter exposing the usual viewer step(action) interface.

    The policy already computed the safety metadata while producing the packed
    action. Reusing that exact metadata avoids API mismatches between the CBF
    actor and the bCBF actor: the bCBF version needs current_command and
    nominal-command history, while the ordinary CBF version does not.
    """

    def __init__(self, env, policy: DeterministicViewerPolicy) -> None:
        self.env = env
        self.policy = policy
        self.unwrapped = env.unwrapped
        self.device = env.device
        self.num_envs = env.num_envs

    def __getattr__(self, name):
        return getattr(self.env, name)

    def get_observations(self):
        return self.env.get_observations()

    def step(self, packed_action):
        nominal = packed_action[:, :3]
        safe = packed_action[:, 3:]

        safety = self.policy.last_safety
        if safety is None:
            raise RuntimeError(
                "Viewer called env.step() before the policy produced safety data."
            )

        return self.env.step_safe(nominal, safe, safety)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play a saved unseen-map case with one selected SAC policy."
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
    print(f"method: {method}")
    print(f"map:    {map_name}")
    print(f"case:   {case_id}")
    print(f"seed:   {case['reset_seed']}")
    print(
        "start: "
        f"({float(case['start_x']):+.3f}, {float(case['start_y']):+.3f}), "
        f"yaw={float(case['start_yaw']):+.3f}"
    )
    print(
        "goal:  "
        f"({float(case['goal_x']):+.3f}, {float(case['goal_y']):+.3f})"
    )
    print(f"device: {device}")
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

        # Force one observation after the fixed scenario is installed.
        _ = env.actor_tensor(env.get_observations())

        policy = DeterministicViewerPolicy(
            env,
            actor,
            method=method,
            print_every=args.print_every,
        )
        viewer_env = ViewerSafeCommandEnv(env, policy)

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
