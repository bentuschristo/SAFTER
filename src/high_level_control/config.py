from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

TrajectoryType = Literal["point", "line", "circle", "lemniscate"]
ControlMode = Literal["trajectory", "keyboard"]


@dataclass
class ModelConfig:
    controlled_point_x: float = 0.0
    controlled_point_y: float = 0.0


@dataclass
class ControllerConfig:
    kp_x: float = 0.8
    kp_y: float = 0.8
    kp_yaw: float = 1.5
    min_vx: float = -0.8
    max_vx: float = 1.2
    min_vy: float = -0.7
    max_vy: float = 0.7
    min_wz: float = -1.0
    max_wz: float = 1.0


@dataclass
class KeyboardConfig:
    # Command applied to an axis while its key is held down (video-game
    # style), not an incremental step.
    vx_speed: float = 0.8
    vy_speed: float = 0.5
    wz_speed: float = 0.8
    # A key counts as "still held" as long as the terminal's OS key-repeat
    # stream keeps delivering it faster than this; once repeats stop for
    # longer than this, the axis snaps to zero.
    release_timeout: float = 0.35


@dataclass
class ObstacleConfig:
    x: float = 0.0
    y: float = 1.5
    radius: float = 0.25
    robot_radius: float = 0.35
    margin: float = 0.15
    height: float = 1.0

    @property
    def safe_radius(self) -> float:
        return self.radius + self.robot_radius + self.margin


@dataclass
class SafetyConfig:
    enabled: bool = False
    alpha: float = 1.5
    epsilon: float = 1.0e-8


@dataclass
class LoggingConfig:
    enabled: bool = True
    root_dir: str = "logs/high_level"
    run_name: str | None = None
    flush_every: int = 10
    plot_on_exit: bool = False


@dataclass
class HighLevelConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    keyboard: KeyboardConfig = field(default_factory=KeyboardConfig)
    obstacle: ObstacleConfig = field(default_factory=ObstacleConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    control_mode: ControlMode = "trajectory"
    trajectory: TrajectoryType = "point"
    high_level_rate_hz: float = 10.0

    # Actor observation layout for Unitree-Go2-Flat:
    # base_ang_vel[0:3], projected_gravity[3:6], command[6:9], ...
    command_obs_start: int = 6

    goal_x: float = 6.0
    goal_y: float = 0.0
    goal_yaw: float = 0.0

    line_speed: float = 0.5

    circle_center_x: float = 2.5
    circle_center_y: float = 0.0
    circle_radius: float = 2.0
    circle_angular_rate: float = 0.20

    lemniscate_center_x: float = 3.0
    lemniscate_center_y: float = 0.0
    lemniscate_scale_x: float = 2.5
    lemniscate_scale_y: float = 1.3
    lemniscate_angular_rate: float = 0.18
