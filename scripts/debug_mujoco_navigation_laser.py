from __future__ import annotations

import os
import sys
from dataclasses import dataclass
import math
from typing import Literal

import torch
import tyro

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401

from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ObjRef, RayCastSensorCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from src.high_level_rl.raycast_patterns import HorizontalRingPatternCfg
from src.high_level_rl.raycast_patterns import horizontal_ray_angles

try:
    from src.tasks.velocity.config.go2 import obstacle_cfg as physical_arena_cfg
except Exception:
    physical_arena_cfg = None


@dataclass
class DebugLaserCfg:
    device: str | None = None
    viewer: Literal["auto", "native", "viser"] = "auto"
    num_envs: int = 1
    num_rays: int = 61
    fov_deg: float = 180.0
    max_distance: float = 5.0
    yaw_offset_deg: float = 0.0
    left_to_right: bool = True
    add_navigation_arena: bool = True
    debug_vis: bool = True
    print_every: int = 10
    print_angles_deg: str = "-90,-45,0,45,90"
    print_all_ranges: bool = False


class ZeroActionPolicy:
    """Keep the robot stationary while the ray geometry is inspected."""

    def __init__(self, action_shape: tuple[int, ...], device: str) -> None:
        self.action_shape = action_shape
        self.device = device

    def __call__(self, obs) -> torch.Tensor:
        del obs
        return torch.zeros(self.action_shape, device=self.device)


class DiagnosticViewerEnv:
    """Expose the observation accessor expected by MJLab's play viewers."""

    def __init__(self, env: ManagerBasedRlEnv, cfg: DebugLaserCfg) -> None:
        self.env = env
        self.unwrapped = env.unwrapped
        # Do not call this attribute ``cfg``: BaseViewer expects env.cfg to be
        # the underlying MJLab environment configuration with cfg.viewer.
        self.debug_cfg = cfg
        self.step_count = 0
        self.angles = horizontal_ray_angles(
            cfg.num_rays,
            cfg.fov_deg,
            yaw_offset_deg=cfg.yaw_offset_deg,
            left_to_right=cfg.left_to_right,
            device=env.device,
        )
        try:
            self.requested_angles_deg = tuple(
                float(value.strip()) for value in cfg.print_angles_deg.split(",") if value.strip()
            )
        except ValueError as exc:
            raise ValueError("--print-angles-deg must be a comma-separated list of degrees") from exc

    def __getattr__(self, name):
        return getattr(self.env, name)

    def get_observations(self):
        return self.env.obs_buf

    def step(self, action: torch.Tensor):
        result = self.env.step(action)
        self.step_count += 1
        if self.debug_cfg.print_every > 0 and self.step_count % self.debug_cfg.print_every == 0:
            self._print_laser()
        return result

    def _nearest_index(self, requested_deg: float) -> int:
        requested = math.radians(requested_deg)
        delta = torch.atan2(torch.sin(self.angles - requested), torch.cos(self.angles - requested))
        return int(torch.argmin(torch.abs(delta)).item())

    def _print_laser(self) -> None:
        sensor = self.env.scene["navigation_laser"]
        raw = sensor.data.distances[0].detach()
        ranges = torch.where(
            torch.isfinite(raw) & (raw >= 0.0),
            raw,
            torch.full_like(raw, self.debug_cfg.max_distance),
        ).clamp(0.0, self.debug_cfg.max_distance)
        nearest = int(torch.argmin(ranges).item())
        valid = int((ranges < self.debug_cfg.max_distance - 1.0e-4).sum().item())
        samples = []
        for requested in self.requested_angles_deg:
            index = self._nearest_index(requested)
            actual = math.degrees(float(self.angles[index]))
            samples.append(f"{requested:+g}deg[i={index},actual={actual:+.1f}]={float(ranges[index]):.3f}m")
        print(
            f"[LASER READ] step={self.step_count} valid={valid}/{ranges.numel()} "
            f"min={float(ranges[nearest]):.3f}m index={nearest} "
            f"angle={math.degrees(float(self.angles[nearest])):+.1f}deg | "
            + " ".join(samples),
            flush=True,
        )
        if self.debug_cfg.print_all_ranges:
            values = ", ".join(f"{float(value):.3f}" for value in ranges)
            print(f"[LASER RANGES] [{values}]", flush=True)


def main() -> None:
    all_tasks = list_tasks()
    chosen_task, remaining = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    cfg = tyro.cli(
        DebugLaserCfg,
        args=remaining,
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )

    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(chosen_task, play=True)
    env_cfg.scene.num_envs = cfg.num_envs

    if cfg.add_navigation_arena:
        if physical_arena_cfg is None:
            print("[WARN] Could not import physical navigation arena config.")
        else:
            env_cfg.scene.entities["navigation_arena"] = (
                physical_arena_cfg.get_go2_navigation_arena_cfg()
            )
            print("[INFO] Added physical navigation_arena entity.")

    laser_sensor = RayCastSensorCfg(
        name="navigation_laser",
        # Match training/evaluation exactly: lidar_site is at x=0.28945 m in the
        # Go2 base frame and the ray pattern has no additional origin offset.
        frame=ObjRef(type="site", name="lidar_site", entity="robot"),
        ray_alignment="yaw",
        pattern=HorizontalRingPatternCfg(
            num_rays=cfg.num_rays,
            fov_deg=cfg.fov_deg,
            origin_x=0.0,
            origin_y=0.0,
            origin_z=0.0,
            yaw_offset_deg=cfg.yaw_offset_deg,
            left_to_right=cfg.left_to_right,
        ),
        max_distance=cfg.max_distance,
        exclude_parent_body=True,
        # Match the production training sensor exactly.
        include_geom_groups=(0,),
        debug_vis=cfg.debug_vis,
        viz=RayCastSensorCfg.VizCfg(
            show_rays=True,
            show_normals=False,
            hit_sphere_radius=0.08,
        ),
    )

    env_cfg.scene.sensors = (env_cfg.scene.sensors or ()) + (laser_sensor,)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env.extras.setdefault("log", {})

    print("=" * 80)
    print("Scene sensors:")
    for name in env.scene.sensors.keys():
        print("  ", name)

    sensor = env.scene["navigation_laser"]
    print("=" * 80)
    print("navigation_laser object:", type(sensor))
    print("navigation_laser data:", type(sensor.data))
    print("data fields/attrs:")
    for k in dir(sensor.data):
        if k.startswith("_"):
            continue
        try:
            v = getattr(sensor.data, k)
        except Exception:
            continue
        if torch.is_tensor(v):
            print(f"  {k}: Tensor shape={tuple(v.shape)} dtype={v.dtype} device={v.device}")
        elif not callable(v):
            print(f"  {k}: {type(v)}")

    print(
        "[LASER VIS] production geometry: site=lidar_site "
        "base_offset=(0.28945, 0.00, -0.075)m pattern_offset=(0, 0, 0)m "
        f"rays={cfg.num_rays} fov={cfg.fov_deg:.1f}deg max={cfg.max_distance:.2f}m"
    )
    policy = ZeroActionPolicy(tuple(env.action_space.shape), device)
    if cfg.print_every < 0:
        raise ValueError("--print-every must be non-negative")
    viewer_env = DiagnosticViewerEnv(env, cfg)
    viewer = cfg.viewer
    if viewer == "auto":
        viewer = "native" if (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) else "viser"
    try:
        if viewer == "native":
            NativeMujocoViewer(viewer_env, policy).run()
        else:
            ViserPlayViewer(viewer_env, policy).run()
    finally:
        env.close()


if __name__ == "__main__":
    main()
