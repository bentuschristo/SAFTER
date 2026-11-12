"""Deploy a high-level SAC checkpoint trained with laser-range observations."""
from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal

import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import ObjRef, RayCastSensorCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from SAFTER_src.high_level_control.command_interface import MjlabVelocityCommandInterface
from SAFTER_src.high_level_laser_sac.config import LaserSacConfig, LaserSacMethod
from SAFTER_src.high_level_laser_sac.networks import LaserSafeSacActor
from SAFTER_src.high_level_laser_sac.sac_navigation_env import LaserSacNavigationEnv
from SAFTER_src.high_level_rl.low_level import load_frozen_low_level_policy
from SAFTER_src.high_level_rl.raycast_patterns import HorizontalRingPatternCfg
from SAFTER_src.high_level_rl.evaluation_scenarios import SCENARIOS


@dataclass
class PlayLaserSacConfig:
    low_level_checkpoint: str
    high_level_checkpoint: str
    method: LaserSacMethod = "vanilla"
    num_envs: int = 1
    device: str | None = None
    viewer: Literal["auto", "native", "viser"] = "native"
    scenario: Literal["random", "easy", "medium", "hard"] = "random"
    print_every: int = 0
    config: LaserSacConfig = field(default_factory=LaserSacConfig)


def add_navigation_laser_sensor(env_cfg, cfg: LaserSacConfig) -> None:
    laser = cfg.laser
    existing = tuple(s for s in (env_cfg.scene.sensors or ()) if s.name != laser.sensor_name)
    navigation_laser = RayCastSensorCfg(
        name=laser.sensor_name,
        frame=ObjRef(type="site", name=laser.lidar_site_name, entity="robot"),
        ray_alignment="yaw",
        pattern=HorizontalRingPatternCfg(
            num_rays=laser.num_rays,
            fov_deg=laser.fov_deg,
            origin_x=0.0,
            origin_y=0.0,
            origin_z=0.0,
            yaw_offset_deg=laser.yaw_offset_deg,
            left_to_right=laser.left_to_right,
        ),
        max_distance=laser.max_range,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=laser.debug_vis,
        viz=RayCastSensorCfg.VizCfg(
            show_rays=laser.debug_vis,
            show_normals=False,
            hit_sphere_radius=0.08,
        ),
    )
    env_cfg.scene.sensors = existing + (navigation_laser,)


def check_checkpoint_compatibility(checkpoint: dict, cfg: LaserSacConfig, path: Path) -> None:
    saved_cfg = checkpoint.get("config")
    if not isinstance(saved_cfg, dict):
        return
    saved_laser = saved_cfg.get("laser")
    if not isinstance(saved_laser, dict):
        return

    current_laser = asdict(cfg)["laser"]
    mismatches: list[str] = []
    fields = ("num_rays", "fov_deg", "max_range", "min_range", "yaw_offset_deg", "left_to_right")
    for field in fields:
        saved_value = saved_laser.get(field)
        if saved_value is None and field == "left_to_right":
            saved_value = False
        elif saved_value is None:
            mismatches.append(f"{field}: checkpoint=<missing>, current={current_laser[field]}")
            continue
        current_value = current_laser[field]
        if isinstance(current_value, float):
            if abs(float(saved_value) - current_value) > 1.0e-6:
                mismatches.append(f"{field}: checkpoint={saved_value}, current={current_value}")
        elif saved_value != current_value:
            mismatches.append(f"{field}: checkpoint={saved_value}, current={current_value}")

    saved_method = saved_cfg.get("method")
    if saved_method is not None and saved_method != cfg.method:
        mismatches.append(f"method: checkpoint={saved_method}, current={cfg.method}")
    saved_safety = saved_cfg.get("safety")
    current_safety = asdict(cfg)["safety"]
    if saved_safety != current_safety:
        mismatches.append(
            f"safety configuration differs: checkpoint={saved_safety}, current={current_safety}"
        )

    if mismatches:
        details = "\n  - ".join(mismatches)
        raise ValueError(
            f"Laser SAC checkpoint {path} was created with a different observation/safety interface:\n"
            f"  - {details}\n"
            "Use matching --config.laser.* and --method values, or train a new checkpoint."
        )


class DeterministicLaserPolicy:
    def __init__(self, env: LaserSacNavigationEnv, actor: LaserSafeSacActor, print_every: int = 0) -> None:
        self.env = env
        self.actor = actor
        self.print_every = max(int(print_every), 0)
        self.steps = 0
        self.last_safety = None

    @torch.no_grad()
    def __call__(self, obs_dict):
        obs = self.env.actor_tensor(obs_dict)
        out = self.actor.deterministic(obs)
        self.last_safety = out.safety
        self.steps += 1
        if self.print_every and self.steps % self.print_every == 0:
            print(
                f"[LASER SAC {self.actor.method}] "
                f"nominal={out.nominal_command[0].detach().cpu().numpy()} "
                f"safe={out.safe_command[0].detach().cpu().numpy()} "
                f"active={bool(out.safety.active[0])} "
                f"correction={float(out.safety.correction_norm[0]):.4f}"
            )
        return torch.cat((out.nominal_command, out.safe_command), dim=-1)


class ViewerLaserSacEnv:
    def __init__(self, env: LaserSacNavigationEnv, policy: DeterministicLaserPolicy) -> None:
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
        nominal, safe = packed_action[:, :3], packed_action[:, 3:]
        safety = self.policy.last_safety
        if safety is None:
            raise RuntimeError("Viewer called env.step() before policy produced safety data.")
        return self.env.step_safe(nominal, safe, safety)


def run(task_id: str, cfg: PlayLaserSacConfig) -> None:
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(task_id, play=True)
    low_agent_cfg = load_rl_cfg(task_id)
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.terminations = {}
    if "push_robot" in env_cfg.events:
        del env_cfg.events["push_robot"]

    evaluation_arena = replace(
        cfg.config.navigation.arena,
        wall_clearance=0.30,
        obstacle_clearance=0.30,
    )
    method_cfg = replace(
        cfg.config,
        method=cfg.method,
        navigation=replace(cfg.config.navigation, arena=evaluation_arena),
    )
    add_navigation_laser_sensor(env_cfg, method_cfg)

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    low_env = RslRlVecEnvWrapper(raw_env, clip_actions=low_agent_cfg.clip_actions)
    low_policy = load_frozen_low_level_policy(task_id, low_env, cfg.low_level_checkpoint, device)
    command_if = MjlabVelocityCommandInterface(low_env, command_name="twist")

    env = LaserSacNavigationEnv(
        low_env,
        low_policy,
        command_if,
        method_cfg.navigation,
        reward_cfg=method_cfg.reward,
        laser_cfg=method_cfg.laser,
    )

    if cfg.scenario != "random":
        scenario = SCENARIOS[cfg.scenario]
        env.set_fixed_scenario(scenario)
        print(
            f"[LASER SAC] Scenario: {scenario.name} | "
            f"start={scenario.start_xy}, yaw={scenario.start_yaw:.3f}, goal={scenario.goal_xy}"
        )

    obs = env.actor_tensor(env.get_observations())
    nav = env.navigation_cfg
    actor = LaserSafeSacActor(
        obs_dim=obs.shape[-1],
        hidden_dims=method_cfg.sac.hidden_dims,
        action_limits=(nav.action.max_vx, nav.action.max_vy, nav.action.max_wz),
        ray_angles=env.ray_angles,
        range_start=8,
        num_rays=method_cfg.laser.num_rays,
        max_range=method_cfg.laser.max_range,
        safety_cfg=method_cfg.safety,
        method=method_cfg.method,
        log_std_min=method_cfg.sac.log_std_min,
        log_std_max=method_cfg.sac.log_std_max,
        min_vx=nav.action.min_vx,
    ).to(device)

    checkpoint_path = Path(cfg.high_level_checkpoint).expanduser()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    check_checkpoint_compatibility(checkpoint, method_cfg, checkpoint_path)
    # action_scale (old symmetric-only buffer) was replaced by
    # action_low/action_high/action_center/action_half_range; both are
    # derived purely from config, never learned, so a legacy checkpoint's
    # saved value is dropped rather than loaded.
    actor_state_dict = dict(checkpoint["actor"])
    actor_state_dict.pop("action_scale", None)
    for buffer_name in ("action_low", "action_high", "action_center", "action_half_range"):
        actor_state_dict.setdefault(buffer_name, actor.state_dict()[buffer_name])
    actor.load_state_dict(actor_state_dict)
    actor.eval()

    order = "left-to-right" if method_cfg.laser.left_to_right else "right-to-left"
    print(
        f"[LASER SAC] method={cfg.method} checkpoint={checkpoint_path} "
        f"rays={method_cfg.laser.num_rays} fov={method_cfg.laser.fov_deg:.1f}deg order={order} "
        f"safety_model={method_cfg.safety.geometry_model} "
        f"body_clearance={method_cfg.safety.footprint_clearance_m:.3f}m"
    )
    policy = DeterministicLaserPolicy(env, actor, cfg.print_every)
    viewer_env = ViewerLaserSacEnv(env, policy)
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
        PlayLaserSacConfig,
        args=remaining,
        prog=sys.argv[0] + f" {task}",
        config=mjlab.TYRO_FLAGS,
    )
    run(task, cfg)


if __name__ == "__main__":
    main()
