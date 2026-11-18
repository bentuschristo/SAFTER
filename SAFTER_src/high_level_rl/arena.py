from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from xml.etree import ElementTree as ET

import torch

from .config import ArenaConfig, CircularObstacle


def arena_to_json(cfg: ArenaConfig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    return path


def arena_from_json(path: str | Path) -> ArenaConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    raw["obstacles"] = tuple(CircularObstacle(**item) for item in raw["obstacles"])
    cfg = ArenaConfig(**raw)
    cfg.validate()
    return cfg


def write_arena_fragment(cfg: ArenaConfig, output_path: str | Path) -> Path:
    """Write an MJCF include containing walls and cylinders."""
    cfg.validate()
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    root = ET.Element("mujoco", {"model": "safe_navigation_arena"})
    world = ET.SubElement(root, "worldbody")
    half_w = cfg.width / 2.0
    half_l = cfg.length / 2.0
    half_t = cfg.wall_thickness / 2.0
    half_h = cfg.wall_height / 2.0

    def add_box(name: str, pos: tuple[float, float, float], size: tuple[float, float, float]) -> None:
        body = ET.SubElement(world, "body", {"name": name, "pos": " ".join(map(str, pos))})
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{name}_geom",
                "type": "box",
                "size": " ".join(map(str, size)),
                "rgba": "0.25 0.30 0.38 1",
                "contype": "1",
                "conaffinity": "1",
                "friction": "0.8 0.05 0.001",
            },
        )

    # Inner wall faces remain at +/- width/2 and +/- length/2.
    add_box("arena_wall_left", (-half_w - half_t, 0.0, half_h), (half_t, half_l + cfg.wall_thickness, half_h))
    add_box("arena_wall_right", (half_w + half_t, 0.0, half_h), (half_t, half_l + cfg.wall_thickness, half_h))
    add_box("arena_wall_bottom", (0.0, -half_l - half_t, half_h), (half_w + cfg.wall_thickness, half_t, half_h))
    add_box("arena_wall_top", (0.0, half_l + half_t, half_h), (half_w + cfg.wall_thickness, half_t, half_h))

    for i, obs in enumerate(cfg.obstacles):
        body = ET.SubElement(
            world,
            "body",
            {"name": f"arena_obstacle_{i}", "pos": f"{obs.x} {obs.y} {obs.height / 2.0}"},
        )
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"arena_obstacle_{i}_geom",
                "type": "cylinder",
                "size": f"{obs.radius} {obs.height / 2.0}",
                "rgba": "0.82 0.20 0.12 1",
                "contype": "1",
                "conaffinity": "1",
                "friction": "0.8 0.05 0.001",
            },
        )

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def patch_scene_with_arena(
    base_scene_xml: str | Path,
    arena_fragment_xml: str | Path,
    output_scene_xml: str | Path,
) -> Path:
    """Copy a base MJCF scene and add a relative <include> for the arena."""
    base = Path(base_scene_xml).resolve()
    fragment = Path(arena_fragment_xml).resolve()
    output = Path(output_scene_xml).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(base)
    root = tree.getroot()
    if root.tag != "mujoco":
        raise ValueError(f"Expected <mujoco> root in {base}")

    # Copy fragment next to output so the include is portable.
    local_fragment = output.parent / fragment.name
    if fragment != local_fragment:
        local_fragment.write_bytes(fragment.read_bytes())

    for inc in root.findall("include"):
        if inc.get("file") == local_fragment.name:
            break
    else:
        root.insert(0, ET.Element("include", {"file": local_fragment.name}))

    ET.indent(root, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output


def obstacle_tensors(cfg: ArenaConfig, num_envs: int, device: torch.device | str):
    centers = torch.zeros(num_envs, cfg.max_obstacles, 2, device=device)
    radii = torch.zeros(num_envs, cfg.max_obstacles, device=device)
    mask = torch.zeros(num_envs, cfg.max_obstacles, dtype=torch.bool, device=device)
    for i, obs in enumerate(cfg.obstacles):
        centers[:, i, 0] = obs.x
        centers[:, i, 1] = obs.y
        radii[:, i] = obs.radius
        mask[:, i] = True
    return centers, radii, mask
