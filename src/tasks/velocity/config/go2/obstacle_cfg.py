"""Fixed 15 x 15 m navigation arena for the Unitree Go2."""

from __future__ import annotations

from dataclasses import dataclass

import os

import mujoco

from mjlab.entity import EntityCfg


@dataclass(frozen=True)
class CircularObstacle:
  """Physical circular (or, with shape="square", axis-aligned box) obstacle."""

  x: float
  y: float
  radius: float
  height: float = 1.0
  # "circle" (radius = radius) or "square" (radius = half-side, axis-aligned;
  # matches the same convention used by the procedural map bank/generator).
  shape: str = "circle"


ARENA_HALF_SIZE = 7.5
WALL_THICKNESS = 0.10
WALL_HEIGHT = 0.75


# Visual-only safety set overlays for native/viser playback.
# Enable with: GO2_SHOW_SAFETY_SETS=1 python ...
# These geoms have contype=0 and conaffinity=0, so they do NOT affect physics.
SHOW_SAFETY_SET_OVERLAYS = os.environ.get("GO2_SHOW_SAFETY_SETS", "0") == "1"
ROBOT_CLEARANCE_RADIUS = float(os.environ.get("GO2_VIS_ROBOT_CLEARANCE", "0.30"))
BACKUP_MARGIN_RADIUS = float(os.environ.get("GO2_VIS_BACKUP_MARGIN", "0.30"))
VISUAL_OVERLAY_HEIGHT = float(os.environ.get("GO2_VIS_OVERLAY_HEIGHT", "0.75"))
VISUAL_OVERLAY_Z = float(os.environ.get("GO2_VIS_OVERLAY_Z", "0.0"))

OBSTACLES: tuple[CircularObstacle, ...] = (
  CircularObstacle(x=-3.0, y=-2.0, radius=0.55),
  CircularObstacle(x=-1.5, y=2.2, radius=0.45),
  CircularObstacle(x=1.2, y=-1.0, radius=0.60),
  CircularObstacle(x=3.0, y=2.3, radius=0.50),
  CircularObstacle(x=0.3, y=3.8, radius=0.40),

  # Additional cylinders in the outer open regions.
  CircularObstacle(x=-5.0, y=4.5, radius=0.50),
  CircularObstacle(x=-5.0, y=-4.5, radius=0.45),
  CircularObstacle(x=5.0, y=4.5, radius=0.55),
  CircularObstacle(x=5.0, y=-4.5, radius=0.50),
)

def _make_arena_xml(include_walls: bool = True) -> str:
  """Construct an MJCF model containing walls and cylindrical obstacles."""

  half_wall_thickness = 0.5 * WALL_THICKNESS
  half_wall_height = 0.5 * WALL_HEIGHT

  # Shift wall centers outward so their inner surfaces are at +/- 7.5 m.
  wall_center = ARENA_HALF_SIZE + half_wall_thickness

  obstacle_bodies: list[str] = []

  def visual_cylinder_xml(name: str, radius: float, rgba: str) -> str:
    return f"""
        <geom
          name="{name}"
          type="cylinder"
          size="{radius} {0.5 * obstacle.height}"
          pos="0 0 0"
          rgba="{rgba}"
          contype="0"
          conaffinity="0"
          group="0"
        />
    """

  for index, obstacle in enumerate(OBSTACLES):
    half_height = 0.5 * obstacle.height
    if obstacle.shape not in ("circle", "square"):
      raise ValueError(f"Unsupported obstacle shape={obstacle.shape!r} at index {index}")
    geom_type = "box" if obstacle.shape == "square" else "cylinder"
    geom_size = (
      f"{obstacle.radius} {obstacle.radius} {half_height}"
      if obstacle.shape == "square"
      else f"{obstacle.radius} {half_height}"
    )

    clearance_radius = obstacle.radius + ROBOT_CLEARANCE_RADIUS
    backup_radius = clearance_radius + BACKUP_MARGIN_RADIUS
    overlay_xml = ""
    if SHOW_SAFETY_SET_OVERLAYS:
      # Draw largest first. The physical obstacle remains opaque on top.
      overlay_xml = (
        visual_cylinder_xml(
          f"obstacle_{index}_backup_set_visual",
          backup_radius,
          "0.10 0.40 1.00 0.13",
        )
        + visual_cylinder_xml(
          f"obstacle_{index}_safe_set_visual",
          backup_radius,
          "0.00 0.85 0.25 0.15",
        )
        + visual_cylinder_xml(
          f"obstacle_{index}_clearance_visual",
          clearance_radius,
          "1.00 0.05 0.05 0.22",
        )
      )

    obstacle_bodies.append(
      f"""
      <body
        name="obstacle_{index}"
        pos="{obstacle.x} {obstacle.y} {half_height}"
      >
        {overlay_xml}
        <geom
          name="obstacle_{index}_collision"
          type="{geom_type}"
          size="{geom_size}"
          rgba="0.85 0.15 0.08 1"
          contype="1"
          conaffinity="1"
          condim="3"
          friction="0.8 0.05 0.001"
        />
      </body>
      """
    )

  obstacles_xml = "\n".join(obstacle_bodies)

  wall_visual_xml = ""
  if SHOW_SAFETY_SET_OVERLAYS:
    z = half_wall_height
    h = half_wall_height
    safe = ROBOT_CLEARANCE_RADIUS
    backup = ROBOT_CLEARANCE_RADIUS + BACKUP_MARGIN_RADIUS
    # Horizontal/vertical visual strips are placed inside the physical wall faces.
    # Red = clearance region, green = original safe-set boundary band,
    # blue = backup-set boundary band. Visual only: no contacts.
    wall_visual_xml = f"""
      <!-- Visual-only safety overlays for wall clearance/safe/backup regions. -->
      <body name="wall_visual_left" pos="0 0 0">
        <geom name="wall_left_clearance_visual" type="box" pos="{-ARENA_HALF_SIZE + 0.5 * safe} 0 {z}" size="{0.5 * safe} {ARENA_HALF_SIZE} {h}" rgba="1.00 0.05 0.05 0.20" contype="0" conaffinity="0" group="0"/>
        <geom name="wall_left_safe_visual" type="box" pos="{-ARENA_HALF_SIZE + safe + 0.5 * BACKUP_MARGIN_RADIUS} 0 {z}" size="{0.5 * BACKUP_MARGIN_RADIUS} {ARENA_HALF_SIZE} {h}" rgba="0.00 0.85 0.25 0.16" contype="0" conaffinity="0" group="0"/>
        <geom name="wall_left_backup_visual" type="box" pos="{-ARENA_HALF_SIZE + backup} 0 {z}" size="0.018 {ARENA_HALF_SIZE} {h}" rgba="0.10 0.40 1.00 0.35" contype="0" conaffinity="0" group="0"/>
      </body>
      <body name="wall_visual_right" pos="0 0 0">
        <geom name="wall_right_clearance_visual" type="box" pos="{ARENA_HALF_SIZE - 0.5 * safe} 0 {z}" size="{0.5 * safe} {ARENA_HALF_SIZE} {h}" rgba="1.00 0.05 0.05 0.20" contype="0" conaffinity="0" group="0"/>
        <geom name="wall_right_safe_visual" type="box" pos="{ARENA_HALF_SIZE - safe - 0.5 * BACKUP_MARGIN_RADIUS} 0 {z}" size="{0.5 * BACKUP_MARGIN_RADIUS} {ARENA_HALF_SIZE} {h}" rgba="0.00 0.85 0.25 0.16" contype="0" conaffinity="0" group="0"/>
        <geom name="wall_right_backup_visual" type="box" pos="{ARENA_HALF_SIZE - backup} 0 {z}" size="0.018 {ARENA_HALF_SIZE} {h}" rgba="0.10 0.40 1.00 0.35" contype="0" conaffinity="0" group="0"/>
      </body>
      <body name="wall_visual_bottom" pos="0 0 0">
        <geom name="wall_bottom_clearance_visual" type="box" pos="0 {-ARENA_HALF_SIZE + 0.5 * safe} {z}" size="{ARENA_HALF_SIZE} {0.5 * safe} {h}" rgba="1.00 0.05 0.05 0.20" contype="0" conaffinity="0" group="0"/>
        <geom name="wall_bottom_safe_visual" type="box" pos="0 {-ARENA_HALF_SIZE + safe + 0.5 * BACKUP_MARGIN_RADIUS} {z}" size="{ARENA_HALF_SIZE} {0.5 * BACKUP_MARGIN_RADIUS} {h}" rgba="0.00 0.85 0.25 0.16" contype="0" conaffinity="0" group="0"/>
        <geom name="wall_bottom_backup_visual" type="box" pos="0 {-ARENA_HALF_SIZE + backup} {z}" size="{ARENA_HALF_SIZE} 0.018 {h}" rgba="0.10 0.40 1.00 0.35" contype="0" conaffinity="0" group="0"/>
      </body>
      <body name="wall_visual_top" pos="0 0 0">
        <geom name="wall_top_clearance_visual" type="box" pos="0 {ARENA_HALF_SIZE - 0.5 * safe} {z}" size="{ARENA_HALF_SIZE} {0.5 * safe} {h}" rgba="1.00 0.05 0.05 0.20" contype="0" conaffinity="0" group="0"/>
        <geom name="wall_top_safe_visual" type="box" pos="0 {ARENA_HALF_SIZE - safe - 0.5 * BACKUP_MARGIN_RADIUS} {z}" size="{ARENA_HALF_SIZE} {0.5 * BACKUP_MARGIN_RADIUS} {h}" rgba="0.00 0.85 0.25 0.16" contype="0" conaffinity="0" group="0"/>
        <geom name="wall_top_backup_visual" type="box" pos="0 {ARENA_HALF_SIZE - backup} {z}" size="{ARENA_HALF_SIZE} 0.018 {h}" rgba="0.10 0.40 1.00 0.35" contype="0" conaffinity="0" group="0"/>
      </body>
    """

  walls_xml = ""
  if include_walls:
    walls_xml = f"""
      <!-- Left wall -->
      <body
        name="wall_left"
        pos="{-wall_center} 0 {half_wall_height}"
      >
        <geom
          name="wall_left_collision"
          type="box"
          size="{half_wall_thickness}
                {ARENA_HALF_SIZE + WALL_THICKNESS}
                {half_wall_height}"
          rgba="0.25 0.28 0.32 1"
          contype="1"
          conaffinity="1"
          condim="3"
          friction="0.8 0.05 0.001"
        />
      </body>

      <!-- Right wall -->
      <body
        name="wall_right"
        pos="{wall_center} 0 {half_wall_height}"
      >
        <geom
          name="wall_right_collision"
          type="box"
          size="{half_wall_thickness}
                {ARENA_HALF_SIZE + WALL_THICKNESS}
                {half_wall_height}"
          rgba="0.25 0.28 0.32 1"
          contype="1"
          conaffinity="1"
          condim="3"
          friction="0.8 0.05 0.001"
        />
      </body>

      <!-- Bottom wall -->
      <body
        name="wall_bottom"
        pos="0 {-wall_center} {half_wall_height}"
      >
        <geom
          name="wall_bottom_collision"
          type="box"
          size="{ARENA_HALF_SIZE + WALL_THICKNESS}
                {half_wall_thickness}
                {half_wall_height}"
          rgba="0.25 0.28 0.32 1"
          contype="1"
          conaffinity="1"
          condim="3"
          friction="0.8 0.05 0.001"
        />
      </body>

      <!-- Top wall -->
      <body
        name="wall_top"
        pos="0 {wall_center} {half_wall_height}"
      >
        <geom
          name="wall_top_collision"
          type="box"
          size="{ARENA_HALF_SIZE + WALL_THICKNESS}
                {half_wall_thickness}
                {half_wall_height}"
          rgba="0.25 0.28 0.32 1"
          contype="1"
          conaffinity="1"
          condim="3"
          friction="0.8 0.05 0.001"
        />
      </body>

      {wall_visual_xml}
    """

  return f"""
  <mujoco model="go2_navigation_arena">
    <worldbody>

      {walls_xml}

      {obstacles_xml}
</worldbody>
  </mujoco>
  """


def get_go2_navigation_arena_cfg(include_walls: bool = True) -> EntityCfg:
  """Return the fixed navigation-arena entity configuration."""

  xml = _make_arena_xml(include_walls=include_walls)
  if SHOW_SAFETY_SET_OVERLAYS:
    print(
      f"[GO2 visual safety sets] enabled: clearance={ROBOT_CLEARANCE_RADIUS:.3f}, "
      f"backup_margin={BACKUP_MARGIN_RADIUS:.3f}, barrier_height={VISUAL_OVERLAY_HEIGHT:.3f}"
    )

  return EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(xml),
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.0),
    ),
  )


# Backward-compatible name, in case existing code imports it.
def get_go2_obstacle_cfg() -> EntityCfg:
  return get_go2_navigation_arena_cfg()


if __name__ == "__main__":
  import mujoco.viewer

  spec = mujoco.MjSpec.from_string(_make_arena_xml())
  model = spec.compile()
  data = mujoco.MjData(model)

  mujoco.viewer.launch(model, data)
