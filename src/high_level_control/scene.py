from __future__ import annotations

from pathlib import Path

from .config import ObstacleConfig


def write_obstacle_mjcf(path: str | Path, obstacle: ObstacleConfig) -> Path:
    """Write a tiny MJCF body that can be included in a scene before compilation.

    The obstacle is a fixed cylinder whose vertical axis is z. MuJoCo cylinder
    ``size`` is ``radius half-height``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    half_height = 0.5 * obstacle.height
    xml = f'''<mujoco model="high_level_obstacle">
  <worldbody>
    <body name="hl_obstacle" pos="{obstacle.x} {obstacle.y} {half_height}">
      <geom name="hl_obstacle_geom"
            type="cylinder"
            size="{obstacle.radius} {half_height}"
            rgba="0.8 0.2 0.2 1"
            contype="1" conaffinity="1"
            friction="0.8 0.05 0.001"/>
    </body>
  </worldbody>
</mujoco>
'''
    path.write_text(xml, encoding="utf-8")
    return path


def obstacle_include_line(relative_path: str = "high_level_obstacle.xml") -> str:
    return f'<include file="{relative_path}"/>'
