"""Play a frozen Unitree Go2 locomotion policy under a modular high-level controller."""

from __future__ import annotations

import math
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import torch
import tyro
import mujoco

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.sensor import ObjRef, RayCastSensorCfg

from src.high_level_control.command_interface import MjlabVelocityCommandInterface
from src.high_level_control.config import HighLevelConfig
from src.high_level_control.controller import FeedbackLinearizationController
from src.high_level_control.logger import ExperimentLogger
from src.high_level_control.keyboard_controller import (
    KeyboardControllerConfig,
    TerminalKeyboardController,
)
from src.high_level_control.model import HighLevelKinematicModel
from src.high_level_control.policy_wrapper import HighLevelPolicyWrapper
from src.high_level_control.safety_filter import ClosedFormCircleCbf, IdentitySafetyFilter
from src.high_level_control.state_interface import MjlabHighLevelStateInterface
from src.high_level_control.trajectories import make_trajectory
from src.tasks.velocity.config.go2 import obstacle_cfg as physical_arena_cfg
from src.high_level_rl.laser_scan import LaserScanConfig, simulate_planar_laser_scan
# NOTE: raycast_patterns.py moved to SAFTER_src (it's laser-pipeline code);
# this keyboard-teleop demo is excluded from that move (it uses the separate
# generic high_level_control feedback-linearization framework, not SAC), so
# it now reaches across into SAFTER_src just for this one small utility.
from SAFTER_src.high_level_rl.raycast_patterns import HorizontalRingPatternCfg, horizontal_ray_angles



@dataclass
class LaserDebugConfig:
    enabled: bool = True
    source: Literal["virtual", "mujoco"] = "mujoco"
    map: Literal["single", "slalom"] = "slalom"
    num_rays: int = 61
    fov_deg: float = 180.0
    max_range: float = 5.0
    min_range: float = 0.05
    print_every: int = 10
    noise_std: float = 0.0
    dropout_prob: float = 0.0

    # Planar LiDAR extrinsic relative to base_link.
    # x forward, y left, yaw positive CCW.
    origin_x: float = 0.28945
    origin_y: float = 0.0
    origin_z: float = 0.25
    yaw_offset_deg: float = 0.0
    left_to_right: bool = True

    # For source="mujoco", add the physical navigation arena/cylinders.
    add_navigation_arena: bool = True
    # Arena boundary walls. Off by default: at the ~7.5m half-size arena
    # extent, rays bouncing off the far walls clutter the LiDAR view.
    add_walls: bool = False
    # Draw the LiDAR ray lines/hit markers in the viewer. The sensor itself
    # (and the console readings from --laser.print-every) keep running
    # either way; this only toggles the drawn ray arrows.
    show_rays: bool = True
    # Flat, visual-only yellow square marking a goal near the edge of the
    # currently-installed physical map.
    add_goal_marker: bool = True
    goal_marker_edge_margin: float = 0.7
    goal_marker_half_size: float = 0.5
    # Physical obstacle layout rendered by MuJoCo. Supported values are
    # ``default``, ``eval_slalom``, ``eval_corridor``, ``eval_clustered``, and
    # ``handcrafted2_0`` through ``handcrafted2_4``, and ``seanav_0`` through
    # ``seanav_99``.
    physical_map: str = "default"


@dataclass
class PlayHighLevelConfig:
    checkpoint_file: str
    device: str | None = None
    viewer: Literal["auto", "native", "viser"] = "auto"
    num_envs: int = 1
    no_terminations: bool = True
    high_level: HighLevelConfig = field(default_factory=HighLevelConfig)
    laser: LaserDebugConfig = field(default_factory=LaserDebugConfig)


EVALUATION_PHYSICAL_MAPS: dict[str, tuple[tuple[float, float, float], ...]] = {
    "eval_slalom": (
        (-5.2, -2.3, 0.50), (-3.9, 2.1, 0.50), (-2.6, -2.2, 0.50),
        (-1.3, 2.2, 0.50), (0.0, -2.2, 0.50), (1.3, 2.2, 0.50),
        (2.6, -2.2, 0.50), (3.9, 2.1, 0.50), (5.2, -2.3, 0.50),
    ),
    "eval_corridor": (
        (-3.8, -2.2, 0.50), (-1.9, -2.2, 0.50), (0.0, -2.2, 0.50),
        (1.9, -2.2, 0.50), (3.8, -2.2, 0.50), (-3.8, 2.2, 0.50),
        (-1.9, 2.2, 0.50), (1.9, 2.2, 0.50), (3.8, 2.2, 0.50),
    ),
    "eval_clustered": (
        (-2.6, 1.7, 0.55), (-1.3, 2.7, 0.45), (-0.2, 1.5, 0.60),
        (-2.1, 3.8, 0.50), (0.1, 3.6, 0.40), (3.8, -3.6, 0.50),
        (5.0, -2.4, 0.45), (3.1, -1.9, 0.55), (5.2, -4.8, 0.50),
    ),
}


class LaserNativeViewer(NativeMujocoViewer):
    """Native viewer without the world-origin XYZ frame marker."""

    def _setup_camera(self) -> None:
        super()._setup_camera()
        assert self.viewer is not None
        self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE.value


def install_selected_physical_map(selection: str) -> None:
    """Install an evaluation, handcrafted2, or SEA-Nav layout before XML creation."""
    if selection == "default":
        return

    if selection in EVALUATION_PHYSICAL_MAPS:
        obstacles = EVALUATION_PHYSICAL_MAPS[selection]
    elif selection.startswith("handcrafted2_"):
        suffix = selection.removeprefix("handcrafted2_")
        if not suffix.isdigit() or int(suffix) not in range(5):
            raise ValueError("handcrafted2 physical map index must be in 0..4")
        map_id = int(suffix)
        repo_root = Path(__file__).resolve().parents[2]
        arena_path = (
            repo_root / "generated/procedural_maps/handcrafted2/arenas"
            / f"map_{map_id:03d}_arena.json"
        )
        import json

        arena_data = json.loads(arena_path.read_text(encoding="utf-8"))
        obstacles = tuple(
            (float(item["x"]), float(item["y"]), float(item["radius"]))
            for item in arena_data["obstacles"]
        )
    elif selection.startswith("seanav_"):
        suffix = selection.removeprefix("seanav_")
        if not suffix.isdigit() or int(suffix) not in range(100):
            raise ValueError("SEA-Nav physical map index must be in 0..99")
        map_id = int(suffix)
        repo_root = Path(__file__).resolve().parents[2]
        arena_path = (
            repo_root / "generated/procedural_maps/seanav_100_seed42/arenas"
            / f"map_{map_id:03d}_arena.json"
        )
        import json

        arena_data = json.loads(arena_path.read_text(encoding="utf-8"))
        physical_arena_cfg.ARENA_HALF_SIZE = 0.5 * min(
            float(arena_data["width"]), float(arena_data["length"])
        )
        physical_arena_cfg.WALL_THICKNESS = float(arena_data["wall_thickness"])
        physical_arena_cfg.WALL_HEIGHT = float(arena_data["wall_height"])
        physical_arena_cfg.OBSTACLES = tuple(
            physical_arena_cfg.CircularObstacle(
                x=float(item["x"]),
                y=float(item["y"]),
                radius=float(item["radius"]),
                height=float(item.get("height", 1.0)),
            )
            for item in arena_data["obstacles"]
        )
        print(
            f"[LASER] Installed physical map {selection}: "
            f"{len(physical_arena_cfg.OBSTACLES)} obstacles, "
            f"arena_half_size={physical_arena_cfg.ARENA_HALF_SIZE:g}m"
        )
        return
    else:
        choices = ["default", *EVALUATION_PHYSICAL_MAPS]
        choices.extend(f"handcrafted2_{index}" for index in range(5))
        choices.append("seanav_0..seanav_99")
        raise ValueError(
            f"Unknown --laser.physical-map {selection!r}; choose one of {choices}"
        )

    physical_arena_cfg.OBSTACLES = tuple(
        physical_arena_cfg.CircularObstacle(x=x, y=y, radius=radius, height=1.0)
        for x, y, radius in obstacles
    )
    print(f"[LASER] Installed physical map {selection}: {len(obstacles)} obstacles")


# Flat, visual-only (contype=0/conaffinity=0) yellow square marking a goal
# near the edge of the currently-installed physical map, styled like
# scripts/capture_map_topdown.py's GOAL_COLOR overlay (translucent fill +
# solid-color border).
_GOAL_RGB = (1.000, 0.839, 0.039)
_GOAL_FILL_ALPHA = 0.43
_GOAL_BORDER_WIDTH = 0.08


def _goal_region_at_edge(edge_margin: float, half_size: float) -> tuple[float, float, float, float]:
    """Return (x_min, x_max, y_min, y_max) for a square centered `edge_margin`
    meters inside the +x boundary of the current ARENA_HALF_SIZE."""
    edge_x = physical_arena_cfg.ARENA_HALF_SIZE - edge_margin
    return (edge_x - half_size, edge_x + half_size, -half_size, half_size)


def _goal_marker_body_xml(region: tuple[float, float, float, float]) -> str:
    x_min, x_max, y_min, y_max = region
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    hx, hy = (x_max - x_min) / 2.0, (y_max - y_min) / 2.0
    r, g, b = _GOAL_RGB
    fill_rgba = f"{r} {g} {b} {_GOAL_FILL_ALPHA}"
    border_rgba = f"{r} {g} {b} 1.0"
    bw = _GOAL_BORDER_WIDTH
    return f"""
      <body name="goal_marker_body" pos="0 0 0">
        <geom name="goal_marker_fill" type="box" pos="{cx} {cy} 0.002" size="{hx} {hy} 0.002"
              rgba="{fill_rgba}" contype="0" conaffinity="0" group="2"/>
        <geom name="goal_marker_border_top" type="box" pos="{cx} {cy + hy - bw / 2} 0.003" size="{hx} {bw / 2} 0.002"
              rgba="{border_rgba}" contype="0" conaffinity="0" group="2"/>
        <geom name="goal_marker_border_bottom" type="box" pos="{cx} {cy - hy + bw / 2} 0.003" size="{hx} {bw / 2} 0.002"
              rgba="{border_rgba}" contype="0" conaffinity="0" group="2"/>
        <geom name="goal_marker_border_left" type="box" pos="{cx - hx + bw / 2} {cy} 0.003" size="{bw / 2} {hy} 0.002"
              rgba="{border_rgba}" contype="0" conaffinity="0" group="2"/>
        <geom name="goal_marker_border_right" type="box" pos="{cx + hx - bw / 2} {cy} 0.003" size="{bw / 2} {hy} 0.002"
              rgba="{border_rgba}" contype="0" conaffinity="0" group="2"/>
      </body>
    """


def get_goal_marker_cfg(region: tuple[float, float, float, float]) -> EntityCfg:
    xml = f"""
    <mujoco model="goal_marker">
      <worldbody>
        {_goal_marker_body_xml(region)}
      </worldbody>
    </mujoco>
    """
    return EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_string(xml),
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )


def _make_virtual_obstacles(cfg: PlayHighLevelConfig, device: str):
    """Return virtual obstacle circles used only for laser diagnostics."""
    if cfg.laser.map == "single":
        centers = torch.tensor(
            [[[cfg.high_level.obstacle.x, cfg.high_level.obstacle.y]]],
            dtype=torch.float32,
            device=device,
        )
        radii = torch.tensor(
            [[cfg.high_level.obstacle.radius]],
            dtype=torch.float32,
            device=device,
        )
    elif cfg.laser.map == "slalom":
        # Simple slalom corridor in world frame.
        # These are virtual diagnostic obstacles first, not physical collision geoms.
        centers = torch.tensor(
            [[
                [1.0,  0.85],
                [2.0, -0.85],
                [3.0,  0.85],
                [4.0, -0.85],
                [5.0,  0.85],
            ]],
            dtype=torch.float32,
            device=device,
        )
        radii = torch.tensor(
            [[0.35, 0.35, 0.35, 0.35, 0.35]],
            dtype=torch.float32,
            device=device,
        )
    else:
        raise ValueError(f"Unknown laser map: {cfg.laser.map}")

    mask = torch.ones_like(radii, dtype=torch.bool)
    return centers, radii, mask




def get_manager_env(env):
    """Unwrap common RL/viewer wrappers until we find the env with .scene."""
    cur = env
    seen = set()
    for _ in range(10):
        if hasattr(cur, "scene"):
            return cur
        obj_id = id(cur)
        if obj_id in seen:
            break
        seen.add(obj_id)
        for attr in ("env", "_env", "unwrapped"):
            if hasattr(cur, attr):
                nxt = getattr(cur, attr)
                if nxt is not cur:
                    cur = nxt
                    break
        else:
            break
    raise AttributeError(f"Could not find ManagerBasedRlEnv with .scene from {type(env)}")


def make_laser_angles(
    num_rays: int,
    fov_deg: float,
    yaw_offset_deg: float,
    device: str,
    left_to_right: bool = False,
):
    return horizontal_ray_angles(
        num_rays,
        fov_deg,
        yaw_offset_deg=yaw_offset_deg,
        left_to_right=left_to_right,
        device=device,
    )


class LaserDiagnosticPolicy:
    """Wrap a viewer policy and print simulated laser scan readings."""

    def __init__(self, policy, state_interface, cfg: PlayHighLevelConfig, device: str):
        self.policy = policy
        self.env = policy.env
        self.manager_env = get_manager_env(policy.env)
        self.state_interface = state_interface
        self.cfg = cfg
        self.device = device
        self.step_count = 0

        self.obstacle_centers, self.obstacle_radii, self.obstacle_mask = (
            _make_virtual_obstacles(cfg, device)
        )
        self.laser_cfg = LaserScanConfig(
            num_rays=cfg.laser.num_rays,
            fov_deg=cfg.laser.fov_deg,
            max_range=cfg.laser.max_range,
            min_range=cfg.laser.min_range,
            robot_radius=0.0,
            obstacle_inflation=0.0,
            noise_std=cfg.laser.noise_std,
            dropout_prob=cfg.laser.dropout_prob,
        )

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def __call__(self, obs):
        action = self.policy(obs)
        self.step_count += 1

        if self.cfg.laser.enabled and self.step_count % self.cfg.laser.print_every == 0:
            self._print_laser()

        return action

    def _print_laser(self):
        state = self.state_interface.read()
        pose = state.base_pose_world

        if isinstance(pose, torch.Tensor):
            pose_t = pose.detach()
        else:
            pose_t = torch.as_tensor(pose, dtype=torch.float32, device=self.device)

        pose_t = pose_t.reshape(-1)
        base_x = pose_t[0]
        base_y = pose_t[1]
        base_yaw = pose_t[2]

        # Transform planar LiDAR origin from base_link to world.
        lx_b = torch.tensor(self.cfg.laser.origin_x, dtype=pose_t.dtype, device=self.device)
        ly_b = torch.tensor(self.cfg.laser.origin_y, dtype=pose_t.dtype, device=self.device)
        cy = torch.cos(base_yaw)
        sy = torch.sin(base_yaw)

        x = base_x + cy * lx_b - sy * ly_b
        y = base_y + sy * lx_b + cy * ly_b
        yaw = base_yaw + torch.tensor(
            self.cfg.laser.yaw_offset_deg * math.pi / 180.0,
            dtype=pose_t.dtype,
            device=self.device,
        )

        robot_xy = torch.stack([x, y]).reshape(1, 2).to(self.device)
        robot_yaw = yaw.reshape(1).to(self.device)

        angles = make_laser_angles(
            self.cfg.laser.num_rays,
            self.cfg.laser.fov_deg,
            self.cfg.laser.yaw_offset_deg,
            device=self.device,
            left_to_right=self.cfg.laser.left_to_right,
        )

        if self.cfg.laser.source == "mujoco":
            raw = self.manager_env.scene["navigation_laser"].data.distances.detach()
            ranges = torch.where(
                raw >= 0.0,
                raw,
                torch.full_like(raw, self.cfg.laser.max_range),
            )
        else:
            ranges, angles = simulate_planar_laser_scan(
                robot_xy=robot_xy,
                robot_yaw=robot_yaw,
                obstacle_centers=self.obstacle_centers,
                obstacle_radii=self.obstacle_radii,
                obstacle_mask=self.obstacle_mask,
                cfg=self.laser_cfg,
                arena_xy_min=(-6.0, -6.0),
                arena_xy_max=(6.0, 6.0),
            )

        r = ranges[0]
        nearest_idx = int(torch.argmin(r).item())
        nearest = float(r[nearest_idx].item())
        nearest_angle = float(angles[nearest_idx].item() * 180.0 / math.pi)

        def at_angle(deg: float) -> float:
            target = torch.tensor(deg * math.pi / 180.0, device=r.device)
            idx = int(torch.argmin(torch.abs(angles - target)).item())
            return float(r[idx].item())

        valid = int((r < self.laser_cfg.max_range).sum().item())

        print(
            f"[LASER:{self.cfg.laser.source}:{self.cfg.laser.map}] "
            f"pose=({float(x):+.2f},{float(y):+.2f},{float(yaw):+.2f}) "
            f"min={nearest:.3f}m angle={nearest_angle:+.1f}deg "
            f"front={at_angle(0.0):.3f} "
            f"left={at_angle(90.0):.3f} "
            f"right={at_angle(-90.0):.3f} "
            f"valid={valid}/{self.laser_cfg.num_rays}"
        )


def run(task_id: str, cfg: PlayHighLevelConfig) -> None:
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)
    env_cfg.scene.num_envs = cfg.num_envs
    # Keep the LiDAR sensor visualization, but omit the command/velocity arrows
    # drawn over the robot body.
    if "twist" in env_cfg.commands:
        env_cfg.commands["twist"].debug_vis = False
    if cfg.no_terminations:
        env_cfg.terminations = {}
    if "push_robot" in env_cfg.events:
        del env_cfg.events["push_robot"]

    if cfg.laser.enabled and cfg.laser.source == "mujoco":
        if cfg.laser.add_navigation_arena:
            install_selected_physical_map(cfg.laser.physical_map)
            env_cfg.scene.entities["navigation_arena"] = (
                physical_arena_cfg.get_go2_navigation_arena_cfg(include_walls=cfg.laser.add_walls)
            )
            if cfg.laser.add_goal_marker:
                goal_region = _goal_region_at_edge(
                    cfg.laser.goal_marker_edge_margin, cfg.laser.goal_marker_half_size
                )
                env_cfg.scene.entities["goal_marker"] = get_goal_marker_cfg(goal_region)
                print(f"[GOAL] marker region (x_min,x_max,y_min,y_max)={goal_region}")

        navigation_laser = RayCastSensorCfg(
            name="navigation_laser",
            frame=ObjRef(type="site", name="lidar_site", entity="robot"),
            ray_alignment="yaw",
            pattern=HorizontalRingPatternCfg(
                num_rays=cfg.laser.num_rays,
                fov_deg=cfg.laser.fov_deg,
                # The site is already the LiDAR origin, so local ray offsets are zero.
                origin_x=0.0,
                origin_y=0.0,
                origin_z=0.0,
                yaw_offset_deg=cfg.laser.yaw_offset_deg,
                left_to_right=cfg.laser.left_to_right,
            ),
            max_distance=cfg.laser.max_range,
            exclude_parent_body=True,
            # Only raycast against world/navigation geometry.
            # Robot visual geoms are group 2; robot collision geoms are group 3.
            # Navigation arena geoms default to group 0.
            include_geom_groups=(0,),
            debug_vis=cfg.laser.show_rays,
            viz=RayCastSensorCfg.VizCfg(
                show_rays=cfg.laser.show_rays,
                show_normals=False,
                hit_sphere_radius=0.08,
            ),
        )
        env_cfg.scene.sensors = (env_cfg.scene.sensors or ()) + (navigation_laser,)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env.extras.setdefault("log", {})
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    checkpoint = Path(cfg.checkpoint_file).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
    locomotion_policy = runner.get_inference_policy(device=device)

    state_if = MjlabHighLevelStateInterface(env)
    command_if = MjlabVelocityCommandInterface(env, command_name="twist")
    model = HighLevelKinematicModel(cfg.high_level.model)
    initial_pose = model.controlled_pose(state_if.read().base_pose_world)
    trajectory = make_trajectory(cfg.high_level, initial_pose)
    controller = FeedbackLinearizationController(model, cfg.high_level.controller)

    if cfg.high_level.safety.enabled:
        safety_filter = ClosedFormCircleCbf(
            model=model,
            obstacle=cfg.high_level.obstacle,
            cfg=cfg.high_level.safety,
        )
    else:
        safety_filter = IdentitySafetyFilter()

    keyboard_controller = None
    if cfg.high_level.control_mode == "keyboard":
        c = cfg.high_level.controller
        k = cfg.high_level.keyboard
        keyboard_controller = TerminalKeyboardController(
            KeyboardControllerConfig(
                vx_speed=k.vx_speed,
                vy_speed=k.vy_speed,
                wz_speed=k.wz_speed,
                release_timeout=k.release_timeout,
                min_vx=c.min_vx,
                max_vx=c.max_vx,
                min_vy=c.min_vy,
                max_vy=c.max_vy,
                min_wz=c.min_wz,
                max_wz=c.max_wz,
            )
        )

    logger = None
    if cfg.high_level.logging.enabled:
        logger = ExperimentLogger(
            root_dir=cfg.high_level.logging.root_dir,
            run_name=cfg.high_level.logging.run_name or cfg.high_level.trajectory,
            flush_every=cfg.high_level.logging.flush_every,
            metadata={
                "task_id": task_id,
                "checkpoint": str(checkpoint),
                "device": device,
                "configuration": asdict(cfg),
            },
        )

    policy = HighLevelPolicyWrapper(
        env=env,
        locomotion_policy=locomotion_policy,
        state_interface=state_if,
        command_interface=command_if,
        model=model,
        controller=controller,
        trajectory=trajectory,
        safety_filter=safety_filter,
        keyboard_controller=keyboard_controller,
        high_level_rate_hz=cfg.high_level.high_level_rate_hz,
        command_obs_start=cfg.high_level.command_obs_start,
        logger=logger,
        trajectory_name=cfg.high_level.trajectory,
    )

    if cfg.laser.enabled:
        policy = LaserDiagnosticPolicy(
            policy=policy,
            state_interface=state_if,
            cfg=cfg,
            device=device,
        )
        print(
            f"[LASER] enabled source={cfg.laser.source} map={cfg.laser.map} "
            f"rays={cfg.laser.num_rays} fov={cfg.laser.fov_deg}deg "
            f"max_range={cfg.laser.max_range}m "
            f"physical_map={cfg.laser.physical_map} "
            f"site=lidar_site "
            f"yaw_offset={cfg.laser.yaw_offset_deg:+.1f}deg"
        )

    if cfg.viewer == "auto":
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        viewer = "native" if has_display else "viser"
    else:
        viewer = cfg.viewer

    try:
        if viewer == "native":
            LaserNativeViewer(env, policy).run()
        else:
            ViserPlayViewer(env, policy).run()
    finally:
        if keyboard_controller is not None:
            keyboard_controller.close()
        if logger is not None:
            logger.close()
            if cfg.high_level.logging.plot_on_exit:
                try:
                    from src.high_level_control.plotter import plot_log
                    paths = plot_log(logger.csv_path)
                    print(f"[PLOTTER] Saved {len(paths)} plots to: {logger.run_dir / 'plots'}")
                except Exception as exc:
                    print(f"[PLOTTER] Failed: {exc}")
        env.close()


def main() -> None:
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401

    all_tasks = list_tasks()
    chosen_task, remaining = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    cfg = tyro.cli(
        PlayHighLevelConfig,
        args=remaining,
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )
    run(chosen_task, cfg)


if __name__ == "__main__":
    main()
