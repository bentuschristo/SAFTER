"""Deploy the actor-integrated SAC + analytical LSE-CBF-CF policy."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from src.high_level_control.command_interface import MjlabVelocityCommandInterface
from src.high_level_rl.low_level import load_frozen_low_level_policy
from src.high_level_sac.config import SacCbfCfConfig
from src.high_level_sac.networks import SafeSacActor
from src.high_level_sac.sac_navigation_env import SacCbfNavigationEnv
from src.high_level_sac.evaluation_scenarios import SCENARIOS


def print_reward_table(cfg) -> None:
    """Print the reward weights used during deployment."""
    r = cfg.config.reward
    print("\nReward table")
    rows = [
        ("progress", r.progress_weight),
        ("goal", r.goal_bonus),
        ("time", -r.time_penalty),
        ("timeout", -r.timeout_penalty),
        ("forward_alignment", r.forward_alignment_weight),
        ("reverse_motion", -r.reverse_motion_weight),
        ("yaw_tracking", -r.yaw_tracking_weight),
        ("smoothness", -r.smoothness_weight),
        ("cbf_correction", -r.cbf_correction_weight),
        ("fall", -r.fall_penalty),
        ("collision", -r.collision_penalty),
    ]
    print("  Deployment method: SAC / CBF-CF")
    for name, weight in rows:
        print(f"  {name:<24} {weight:+.4f}")
    print()


@dataclass
class PlaySacCbfCfConfig:
    low_level_checkpoint: str
    high_level_checkpoint: str
    num_envs: int = 1
    device: str | None = None
    viewer: Literal["auto", "native", "viser"] = "native"
    print_every: int = 0
    scenario: Literal["random", "easy", "medium", "hard"] = "random"
    # None uses config.use_safety_filter. Set explicitly for paired deployment:
    # False = raw SAC, True = the same actor shielded by CBF-CF.
    deployment_safety_filter: bool | None = None
    config: SacCbfCfConfig = field(default_factory=SacCbfCfConfig)


class DeterministicSafePolicy:
    def __init__(self, env: SacCbfNavigationEnv, actor: SafeSacActor, print_every: int = 0) -> None:
        self.env = env
        self.actor = actor
        self.print_every = max(int(print_every), 0)
        self.steps = 0

    @torch.no_grad()
    def __call__(self, obs_dict):
        obs = self.env.actor_tensor(obs_dict)
        position, yaw = self.env.safety_context()
        out = self.actor.deterministic(obs, position, yaw)
        self.steps += 1
        if self.print_every and self.steps % self.print_every == 0:
            print(
                f"[SAC {'CBF-CF' if self.actor.use_safety_filter else 'NO-CBF'}] "
                f"nominal={out.nominal_command[0].cpu().numpy()} "
                f"safe={out.safe_command[0].cpu().numpy()} "
                f"active={bool(out.safety.active[0])} "
                f"correction={float(out.safety.correction_norm[0]):.4f}"
            )
        # Viewer calls env.step(action). We return a packed 6-vector and adapt
        # the environment below so nominal and safe remain available.
        return torch.cat((out.nominal_command, out.safe_command), dim=-1)


class ViewerSacEnv:
    """Small adapter exposing the usual viewer step(action) interface."""
    def __init__(self, env: SacCbfNavigationEnv, actor: SafeSacActor) -> None:
        self.env = env
        self.actor = actor
        self.unwrapped = env.unwrapped
        self.device = env.device
        self.num_envs = env.num_envs

    def __getattr__(self, name):
        return getattr(self.env, name)

    def get_observations(self):
        return self.env.get_observations()

    def step(self, packed_action):
        nominal, safe = packed_action[:, :3], packed_action[:, 3:]
        position, yaw = self.env.safety_context()
        nominal_norm = nominal / self.actor.action_scale
        _, _, safety = self.actor.apply_safety(nominal_norm, position, yaw)
        return self.env.step_safe(nominal, safe, safety)


def run(task_id: str, cfg: PlaySacCbfCfConfig) -> None:
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(task_id, play=True)
    low_agent_cfg = load_rl_cfg(task_id)
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.terminations = {}
    if "push_robot" in env_cfg.events:
        del env_cfg.events["push_robot"]

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    low_env = RslRlVecEnvWrapper(raw_env, clip_actions=low_agent_cfg.clip_actions)
    low_policy = load_frozen_low_level_policy(task_id, low_env, cfg.low_level_checkpoint, device)
    command_if = MjlabVelocityCommandInterface(low_env, command_name="twist")
    env = SacCbfNavigationEnv(
        low_env, low_policy, command_if, cfg.config.navigation, reward_cfg=cfg.config.reward
    )

    print_reward_table(cfg)

    if cfg.scenario != "random":
        scenario = SCENARIOS[cfg.scenario]
        env.set_fixed_scenario(scenario)
        print(
            f"[SAC] Scenario: {scenario.name} | "
            f"start={scenario.start_xy}, yaw={scenario.start_yaw:.3f}, "
            f"goal={scenario.goal_xy}"
        )

    obs = env.actor_tensor(env.get_observations())
    actor = SafeSacActor(
        obs_dim=obs.shape[-1],
        hidden_dims=cfg.config.sac.hidden_dims,
        arena=env.navigation_cfg.arena,
        action_cfg=env.navigation_cfg.action,
        safety_cfg=env.navigation_cfg.safety,
        obstacle_centers=env.obstacle_centers[:1],
        obstacle_radii=env.obstacle_radii[:1],
        obstacle_mask=env.obstacle_mask[:1],
        log_std_min=cfg.config.sac.log_std_min,
        log_std_max=cfg.config.sac.log_std_max,
        use_safety_filter=(
            cfg.config.use_safety_filter
            if cfg.deployment_safety_filter is None
            else cfg.deployment_safety_filter
        ),
    ).to(device)
    checkpoint = torch.load(Path(cfg.high_level_checkpoint).expanduser(), map_location=device, weights_only=False)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    mode = "CBF-CF enabled" if actor.use_safety_filter else "raw SAC, no CBF"
    print(f"[SAC] Deployment mode: {mode}")

    policy = DeterministicSafePolicy(env, actor, cfg.print_every)
    viewer_env = ViewerSacEnv(env, actor)
    viewer = cfg.viewer
    if viewer == "auto":
        viewer = "native" if (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) else "viser"
    if viewer == "native":
        NativeMujocoViewer(viewer_env, policy).run()
    else:
        ViserPlayViewer(viewer_env, policy).run()
    env.close()


def main() -> None:
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401

    task, remaining = tyro.cli(
        tyro.extras.literal_type_from_choices(list_tasks()),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    cfg = tyro.cli(
        PlaySacCbfCfConfig,
        args=remaining,
        prog=sys.argv[0] + f" {task}",
        config=mjlab.TYRO_FLAGS,
    )
    run(task, cfg)


if __name__ == "__main__":
    main()
