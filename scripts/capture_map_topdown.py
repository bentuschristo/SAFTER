#!/usr/bin/env python3
"""Top-down MuJoCo screenshot of an evaluation/training map, with the
initial (start) and goal regions overlaid as translucent rectangles.

Builds a bare ManagerBasedRlEnv (no low-level policy, no RL wrapper --
we only need one physics reset to get the arena/obstacles/robot in their
initial spawn state) with render_mode="rgb_array" and a straight-down
ViewerConfig camera, then composites the start/goal region rectangles in
pixel space afterward using PIL.

Usage:
    python -u scripts/capture_map_topdown.py --map corridor
    python -u scripts/capture_map_topdown.py --map slalom
    python -u scripts/capture_map_topdown.py --map corridor_derived_highshift_v5
    python -u scripts/capture_map_topdown.py --map handcrafted2 --map-id 0
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
import tyro
from PIL import Image, ImageDraw

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.viewer.viewer_config import ViewerConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_high_level_cbf_rl_unseen_maps import (  # noqa: E402
    UNSEEN_MAPS,
    STRUCTURED_SCENARIO_REGIONS,
    make_fixed_map_obstacle_entities,
)
from train_high_level_cbf_rl import add_navigation_laser_sensor, make_walls_only_arena_cfg  # noqa: E402
from src.high_level_rl.config import ArenaConfig  # noqa: E402
from src.high_level_cbf_rl.config import CbfRlConfig  # noqa: E402

TASK_ID = "Unitree-Go2-Safe-Navigation"
ARENA_HALF = 7.5  # matches width=length=15.0 used throughout this project

# Camera params below (distance=20, fovy=45, straight-down, world-origin
# lookat) determine the visible half-span in world meters: half_span =
# distance * tan(fovy/2). Kept as constants here so the pixel<->world
# mapping used for the region overlay always matches the render call.
CAM_DISTANCE = 20.0
CAM_FOVY_DEG = 45.0
START_COLOR = (102, 255, 102, 110)  # light green, translucent -- contrasts with the blue MuJoCo floor
GOAL_COLOR = (255, 214, 10, 110)   # yellow, translucent

def handcrafted2_regions(map_id: int) -> dict[str, tuple[float, float, float, float]]:
    """Per-map fixed start/goal regions used to generate this map's 200
    scenarios -- authoritative source: generated/procedural_maps/handcrafted2/
    generation_config.json's per_map_endpoint_regions (NOT arbitrary
    whole-arena sampling; each of the 5 maps has its own fixed corner/edge
    pair, e.g. map_id=0 is bottom-right start -> top-left goal)."""
    import json
    cfg = json.loads(Path("generated/procedural_maps/handcrafted2/generation_config.json").read_text())
    region = cfg["per_map_endpoint_regions"][str(map_id)]
    x0, x1, y0, y1 = region["start_region"]
    gx0, gx1, gy0, gy1 = region["goal_region"]
    return {"start": (x0, x1, y0, y1), "goal": (gx0, gx1, gy0, gy1)}


def world_half_span() -> float:
    import math
    return CAM_DISTANCE * math.tan(math.radians(CAM_FOVY_DEG) / 2.0)


def world_to_pixel(x: float, y: float, image_size: int) -> tuple[float, float]:
    """world +x -> image right, world +y -> image up (verified empirically
    against corridor's known asymmetric obstacle row)."""
    half_span = world_half_span()
    px = (x + half_span) / (2 * half_span) * image_size
    py = (half_span - y) / (2 * half_span) * image_size
    return px, py


def overlay_region(base: Image.Image, region: tuple[float, float, float, float], image_size: int, color) -> Image.Image:
    x0, x1, y0, y1 = region
    px0, py0 = world_to_pixel(x0, y1, image_size)  # top-left: min-x, max-y
    px1, py1 = world_to_pixel(x1, y0, image_size)  # bottom-right: max-x, min-y
    overlay = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    odraw.rectangle([px0, py0, px1, py1], fill=color, outline=color[:3] + (255,), width=4)
    return Image.alpha_composite(base.convert("RGBA"), overlay)


@dataclass
class Args:
    map: str = "corridor"
    map_id: int | None = None  # for handcrafted2: which of the 5 maps (0-4)
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    image_size: int = 1024
    output_dir: str = "logs/evaluation/figures/map_captures"


def build_env(map_name: str, map_id: int | None, device: str, image_size: int) -> ManagerBasedRlEnv:
    if map_name == "handcrafted2":
        import json
        bank_root = Path("generated/procedural_maps/handcrafted2")
        bank = json.loads((bank_root / "bank.json").read_text())
        mid = 0 if map_id is None else map_id
        map_entry = next(m for m in bank["maps"] if m["map_id"] == mid)
        map_json = json.loads((bank_root / map_entry["file"]).read_text())
        from src.high_level_rl.config import CircularObstacle as CO
        obstacles = tuple(
            CO(float(o["x"]), float(o["y"]), float(o["radius"]), shape=o.get("shape", "circle"))
            for o in map_json["obstacles"]
        )
        arena_width = float(map_json["arena"]["width"])
    else:
        obstacles = UNSEEN_MAPS[map_name]
        arena_width = 15.0

    arena = ArenaConfig(obstacles=obstacles, max_obstacles=len(obstacles))
    method_cfg = CbfRlConfig(navigation=__import__("dataclasses").replace(CbfRlConfig().navigation, arena=arena))

    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.terminations = {}
    if "push_robot" in env_cfg.events:
        del env_cfg.events["push_robot"]
    env_cfg.scene.entities["navigation_arena"] = make_walls_only_arena_cfg(arena)
    env_cfg.scene.entities.update(make_fixed_map_obstacle_entities(obstacles))
    add_navigation_laser_sensor(env_cfg, method_cfg)

    env_cfg.viewer = ViewerConfig(
        origin_type=ViewerConfig.OriginType.WORLD,
        lookat=(0.0, 0.0, 0.0),
        elevation=-90.0,
        azimuth=90.0,
        distance=20.0,
        fovy=45.0,
        height=image_size,
        width=image_size,
        enable_shadows=True,
        enable_reflections=False,
    )

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    return env


def hide_robot(env: ManagerBasedRlEnv) -> None:
    """Make every geom belonging to the robot's kinematic tree fully
    transparent so the top-down capture shows only the arena/obstacles."""
    robot = env.scene["robot"]
    root_body_id = robot.indexing.root_body_id
    model = env.sim.mj_model
    for geom_id in range(model.ngeom):
        body_id = model.geom_bodyid[geom_id]
        if model.body_rootid[body_id] == root_body_id:
            model.geom_rgba[geom_id, 3] = 0.0


def main() -> None:
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401

    args = tyro.cli(Args)
    env = build_env(args.map, args.map_id, args.device, args.image_size)
    env.reset()
    hide_robot(env)
    frame = env.render()
    print("frame shape:", None if frame is None else np.asarray(frame).shape)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_map{args.map_id}" if args.map == "handcrafted2" and args.map_id is not None else ""

    base = Image.fromarray(np.asarray(frame)).convert("RGBA")
    raw_path = out_dir / f"{args.map}{suffix}_topdown_raw.png"
    base.convert("RGB").save(raw_path)
    print(f"Saved (no region overlay, for compositing): {raw_path}")
    if args.map == "handcrafted2":
        regions = handcrafted2_regions(0 if args.map_id is None else args.map_id)
    else:
        regions = STRUCTURED_SCENARIO_REGIONS.get(args.map)
    if regions is not None:
        base = overlay_region(base, regions["start"], args.image_size, START_COLOR)
        base = overlay_region(base, regions["goal"], args.image_size, GOAL_COLOR)
    out_path = out_dir / f"{args.map}{suffix}_topdown.png"

    base.convert("RGB").save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
