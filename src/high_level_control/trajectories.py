from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import math
import numpy as np

from .config import HighLevelConfig


@dataclass(frozen=True)
class ReferenceState:
    pose: np.ndarray      # [x_ref, y_ref, yaw_ref]
    velocity: np.ndarray  # [xdot_ref, ydot_ref, yawdot_ref]


class ReferenceTrajectory(Protocol):
    def evaluate(self, time_s: float) -> ReferenceState: ...


@dataclass
class PointGoal:
    x: float
    y: float
    yaw: float

    def evaluate(self, time_s: float) -> ReferenceState:
        del time_s
        return ReferenceState(
            np.array([self.x, self.y, self.yaw], dtype=np.float64),
            np.zeros(3, dtype=np.float64),
        )


@dataclass
class StraightLine:
    x0: float
    y0: float
    yaw: float
    speed: float

    def evaluate(self, time_s: float) -> ReferenceState:
        vx = self.speed * math.cos(self.yaw)
        vy = self.speed * math.sin(self.yaw)
        return ReferenceState(
            np.array([self.x0 + vx * time_s, self.y0 + vy * time_s, self.yaw], dtype=np.float64),
            np.array([vx, vy, 0.0], dtype=np.float64),
        )


@dataclass
class Circle:
    cx: float
    cy: float
    radius: float
    angular_rate: float

    def evaluate(self, time_s: float) -> ReferenceState:
        q = self.angular_rate * time_s
        cq, sq = math.cos(q), math.sin(q)
        x = self.cx + self.radius * cq
        y = self.cy + self.radius * sq
        xd = -self.radius * self.angular_rate * sq
        yd =  self.radius * self.angular_rate * cq
        yaw = math.atan2(yd, xd)
        return ReferenceState(
            np.array([x, y, yaw], dtype=np.float64),
            np.array([xd, yd, self.angular_rate], dtype=np.float64),
        )


@dataclass
class Lemniscate:
    cx: float
    cy: float
    ax: float
    ay: float
    angular_rate: float

    def evaluate(self, time_s: float) -> ReferenceState:
        q = self.angular_rate * time_s
        x = self.cx + self.ax * math.sin(q)
        y = self.cy + self.ay * math.sin(q) * math.cos(q)
        xd = self.ax * self.angular_rate * math.cos(q)
        yd = self.ay * self.angular_rate * math.cos(2.0 * q)
        yaw = math.atan2(yd, xd)
        denom = max(xd * xd + yd * yd, 1.0e-8)
        xdd = -self.ax * self.angular_rate**2 * math.sin(q)
        ydd = -2.0 * self.ay * self.angular_rate**2 * math.sin(2.0 * q)
        yawd = (xd * ydd - yd * xdd) / denom
        return ReferenceState(
            np.array([x, y, yaw], dtype=np.float64),
            np.array([xd, yd, yawd], dtype=np.float64),
        )


def make_trajectory(cfg: HighLevelConfig, initial_pose: np.ndarray) -> ReferenceTrajectory:
    if cfg.trajectory == "point":
        return PointGoal(cfg.goal_x, cfg.goal_y, cfg.goal_yaw)
    if cfg.trajectory == "line":
        return StraightLine(float(initial_pose[0]), float(initial_pose[1]), cfg.goal_yaw, cfg.line_speed)
    if cfg.trajectory == "circle":
        return Circle(cfg.circle_center_x, cfg.circle_center_y, cfg.circle_radius, cfg.circle_angular_rate)
    if cfg.trajectory == "lemniscate":
        return Lemniscate(
            cfg.lemniscate_center_x,
            cfg.lemniscate_center_y,
            cfg.lemniscate_scale_x,
            cfg.lemniscate_scale_y,
            cfg.lemniscate_angular_rate,
        )
    raise ValueError(f"Unsupported trajectory: {cfg.trajectory}")
