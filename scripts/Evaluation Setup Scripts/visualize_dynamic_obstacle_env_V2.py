#!/usr/bin/env python3
"""
Visualize and sanity-check a 2-D navigation environment with walls, static circular
obstacles, dynamic circular obstacles, dynamic-obstacle trajectories, and optional
safe sets.

This script is intentionally independent from the Unitree/MuJoCo code so that you
can quickly design/evaluate candidate dynamic-obstacle test scenarios before
turning them into simulation assets.

Coordinate convention used here:
    The world is centered at (0, 0).
    x in [-WORLD_SIZE[0]/2, WORLD_SIZE[0]/2],
    y in [-WORLD_SIZE[1]/2, WORLD_SIZE[1]/2].
    For a 15 m x 15 m world, the lower-left corner is (-7.5, -7.5) and
    the upper-right corner is (7.5, 7.5).

Safe-set convention:
    Safe radius around each obstacle = obstacle_radius + ROBOT_RADIUS + SAFETY_MARGIN.
    The wall safe set is the inner rectangle obtained by shrinking the wall by
    ROBOT_RADIUS + SAFETY_MARGIN.

Author: generated for laser/bCBF evaluation prototyping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


# =============================================================================
# User-adjustable configuration
# =============================================================================

WORLD_SIZE = (15.0, 15.0)  # meters: width, height


def world_bounds(
    world_size: tuple[float, float] = WORLD_SIZE,
) -> tuple[float, float, float, float]:
    """Return centered arena bounds as (xmin, xmax, ymin, ymax)."""
    width, height = world_size
    return (-0.5 * width, 0.5 * width, -0.5 * height, 0.5 * height)


ROBOT_RADIUS = 0.30        # Unitree body footprint approximation used for 2-D safety plots
SAFETY_MARGIN = 0.15       # additional clearance around obstacles/walls
SHOW_SAFE_SET = True
SHOW_DYNAMIC_SWEEP = True
SHOW_TIME_SNAPSHOTS = True
SAVE_FIGURE = True
OUTPUT_DIR = Path("dynamic_obstacle_env_preview")

T_FINAL = 60.0             # seconds for trajectory preview/evaluation
DT = 0.05                  # seconds for sampling dynamic trajectories

# Optional reference path to visualize possible robot route. This is not used
# for policy evaluation here; it is only plotted for scenario design.
START = np.array([-6.0, -6.0], dtype=float)
GOAL = np.array([6.0, 6.0], dtype=float)


@dataclass
class StaticObstacle:
    center: tuple[float, float]
    radius: float
    name: str = "static"


@dataclass
class DynamicObstacle:
    """Circular moving obstacle.

    Supported trajectory types:
      - line:     p(t) = p0 + speed*[cos(heading), sin(heading)]*t, with optional wall bounce.
      - line_oscillate:  moves sinusoidally back and forth along a straight line.
      - circle:   p(t) = center + path_radius*[cos(omega*t+phase), sin(omega*t+phase)].
      - sine:     moves along heading direction with speed, plus sinusoidal lateral offset.
      - waypoints: piecewise-linear path through waypoints at constant speed.
    """

    radius: float
    trajectory_type: Literal["line", "line_oscillate", "circle", "sine", "waypoints"]
    name: str = "dynamic"

    # Common / line parameters
    p0: tuple[float, float] = (0.0, 0.0)
    speed: float = 0.20
    heading_deg: float = 0.0
    bounce: bool = False

    # Circle parameters
    center: tuple[float, float] = (0.0, 0.0)
    path_radius: float = 2.0
    omega: float | None = None       # rad/s. If None, derived from speed / path_radius.
    phase_deg: float = 0.0

    # Sine parameters
    amplitude: float = 1.0
    frequency_hz: float = 0.10

    # Waypoint parameters
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    loop: bool = True

    def position_at(self, t: float, world_size: tuple[float, float] = WORLD_SIZE) -> np.ndarray:
        if self.trajectory_type == "line":
            heading = math.radians(self.heading_deg)
            v = self.speed * np.array([math.cos(heading), math.sin(heading)], dtype=float)
            p = np.array(self.p0, dtype=float) + v * t
            if self.bounce:
                # Reflect the center trajectory within the valid center range [r, L-r].
                p = _bounce_point(p, self.radius, world_size)
            return p

        if self.trajectory_type == "line_oscillate":
            heading = math.radians(self.heading_deg)

            direction = np.array(
                [
                    math.cos(heading),
                    math.sin(heading),
                ],
                dtype=float,
            )

            center = np.array(self.p0, dtype=float)

            # Interpret speed as the maximum linear speed.
            omega = self.speed / max(self.amplitude, 1e-9)

            phase = math.radians(self.phase_deg)

            displacement = self.amplitude * math.sin(
                omega * t + phase
            )

            return center + direction * displacement

        if self.trajectory_type == "circle":
            omega = self.omega if self.omega is not None else self.speed / max(self.path_radius, 1e-9)
            phase = math.radians(self.phase_deg)
            c = np.array(self.center, dtype=float)
            return c + self.path_radius * np.array(
                [math.cos(omega * t + phase), math.sin(omega * t + phase)], dtype=float
            )

        if self.trajectory_type == "sine":
            heading = math.radians(self.heading_deg)
            forward = np.array([math.cos(heading), math.sin(heading)], dtype=float)
            lateral = np.array([-math.sin(heading), math.cos(heading)], dtype=float)
            p = np.array(self.p0, dtype=float)
            return p + forward * self.speed * t + lateral * self.amplitude * math.sin(
                2.0 * math.pi * self.frequency_hz * t
            )

        if self.trajectory_type == "waypoints":
            return _position_on_waypoints(self.waypoints, self.speed, t, loop=self.loop)

        raise ValueError(f"Unsupported trajectory_type: {self.trajectory_type}")


# =============================================================================
# Dynamic evaluation scenarios. Edit SCENARIO_NAME to preview each map.
# =============================================================================

# SCENARIO_NAME = "D1_crossing_orbit"
# SCENARIO_NAME = "D2_moving_gate"
# SCENARIO_NAME = "D3_oncoming_corridor"
SCENARIO_NAME = "D4_slow_gate_brakezone"
# SCENARIO_NAME = "D5_Corridor"

def make_scenario(name: str):
    if name == "D1_crossing_orbit":
        start = (-6.0, -6.0)
        goal = (6.0, 6.0)

        static_obstacles = [
            StaticObstacle(center=(-2.5, -2.5), radius=0.70, name="static_1"),
            StaticObstacle(center=(-2.0, -5.5), radius=0.70, name="static_2"),
            StaticObstacle(center=(-5.5, -2.0), radius=0.70, name="static_3"),
            StaticObstacle(center=(-2.0, 4.0), radius=0.70, name="static_4"),
            StaticObstacle(center=(4.0, -2.0), radius=0.70, name="static_5"),
            StaticObstacle(center=(4.5, 4.5), radius=0.70, name="static_6"),
            StaticObstacle(center=(2.0, 6.0), radius=0.50, name="static_7"),
            StaticObstacle(center=(6.0, 2.0), radius=0.50, name="static_8"),
        ]

        dynamic_obstacles = [
            DynamicObstacle(
                name="dyn_crossing",
                radius=0.40,
                trajectory_type="line_oscillate",
                p0=(-1.0, -1.0),
                speed=0.30,
                heading_deg=-45.0,
                amplitude=2.0,
                phase_deg=0.0,
                bounce=False,
            ),
            DynamicObstacle(
                name="dyn_orbit",
                radius=0.40,
                trajectory_type="circle",
                center=(2.0, 2.0),
                path_radius=1.5,
                speed=0.25,
                phase_deg=90.0,
            ),
        ]

        return start, goal, static_obstacles, dynamic_obstacles

    if name == "D2_moving_gate":
        start = (-6.0, -6.0)
        goal = (6.0, 6.0)

        # This map creates a recoverability trap near the center.
        # Two static obstacles form a narrow diagonal gate, while the dynamic
        # obstacle moves along the gate direction and periodically blocks entry.
        static_obstacles = [
            StaticObstacle(center=(-5, -3.5), radius=0.70, name="static_1"),
            StaticObstacle(center=(-2, -3), radius=0.70, name="static_2"),

            # Central gate obstacles. These are intentionally close to the
            # nominal diagonal path, but still leave a feasible passage.
            StaticObstacle(center=(-1.2, 1.2), radius=0.65, name="gate_left"),
            StaticObstacle(center=(1.2, -1.2), radius=0.65, name="gate_right"),

            # Exit-side clutter.
            StaticObstacle(center=(2.8, 1.4), radius=0.70, name="static_5"),
            StaticObstacle(center=(1.4, 2.8), radius=0.70, name="static_6"),
            StaticObstacle(center=(4.9, 4.7), radius=0.60, name="static_7"),
            StaticObstacle(center=(-4.5, 3.2), radius=0.60, name="static_8"),
        ]

        dynamic_obstacles = [
            DynamicObstacle(
                name="dyn_gate_blocker",
                radius=0.35,
                trajectory_type="line_oscillate",
                p0=(0.0, 0.0),
                speed=0.20,
                heading_deg=45.0,
                amplitude=1.45,
                phase_deg=-60.0,
                bounce=False,
            ),
            DynamicObstacle(
                name="dyn_exit_cross",
                radius=0.35,
                trajectory_type="line_oscillate",
                p0=(3.5, 3.5),
                speed=0.20,
                heading_deg=-45.0,
                amplitude=1.15,
                phase_deg=30.0,
                bounce=False,
            ),
        ]

        return start, goal, static_obstacles, dynamic_obstacles

    if name == "D3_oncoming_corridor":
        start = (-6.0, -6.0)
        goal = (6.0, 6.0)

        # This map tests whether the robot slows down early when a moving
        # obstacle approaches along the same general route.
        static_obstacles = [
            StaticObstacle(center=(-5.2, -2.5), radius=0.70, name="static_1"),
            StaticObstacle(center=(-2.5, -5.2), radius=0.70, name="static_2"),

            # Mild corridor-shaping obstacles near the diagonal route.
            StaticObstacle(center=(-1.2, 0.9), radius=0.60, name="corridor_left_1"),
            StaticObstacle(center=(0.9, -1.2), radius=0.60, name="corridor_right_1"),
            StaticObstacle(center=(1.9, 4.1), radius=0.65, name="corridor_left_2"),
            StaticObstacle(center=(4.1, 1.9), radius=0.65, name="corridor_right_2"),

            # Extra off-path clutter.
            StaticObstacle(center=(-4.2, 3.8), radius=0.65, name="static_7"),
            StaticObstacle(center=(5.5, -2.2), radius=0.60, name="static_8"),
        ]

        dynamic_obstacles = [
            DynamicObstacle(
                name="dyn_oncoming",
                radius=0.40,
                trajectory_type="line_oscillate",
                p0=(1.7, 1.7),
                speed=0.28,
                heading_deg=45.0,
                amplitude=3.5,
                phase_deg=90.0,
                bounce=False,
            ),
        ]

        return start, goal, static_obstacles, dynamic_obstacles

    if name == "D4_slow_gate_brakezone":
        start = (-6.0, -6.0)
        goal = (6.0, 6.0)

        # This map is intended to test braking/recoverability rather than
        # high-speed dynamic-obstacle prediction. The robot approaches a
        # slow-varying bottleneck near the center. A good bCBF behavior is
        # to slow down or hesitate before entering if stopping is not recoverable.
        static_obstacles = [
            # Entry-side obstacles: encourage a diagonal approach.
            StaticObstacle(center=(-3.2, -1.4), radius=0.60, name="static_1"),
            StaticObstacle(center=(-1.4, -3.2), radius=0.60, name="static_2"),

            # Central bottleneck. Widened from 0.60 to 0.50 so that avoiding
            # the dynamic obstacle does not immediately cause a static collision.
            StaticObstacle(center=(-0.9, 1.1), radius=0.50, name="static_3"),
            StaticObstacle(center=(1.1, -0.9), radius=0.50, name="static_4"),

            # Exit-side obstacles. Also widened to reduce static/wall collisions.
            StaticObstacle(center=(2.8, 4.0), radius=0.50, name="static_5"),
            StaticObstacle(center=(4.0, 2.8), radius=0.50, name="static_6"),

            # Mild side clutter / detour shaping.
            StaticObstacle(center=(-5.2, 1.4), radius=0.55, name="static_7"),
            StaticObstacle(center=(1.4, -5.2), radius=0.55, name="static_8"),
        ]

        dynamic_obstacles = [
            DynamicObstacle(
                name="dyn_slow_gate",
                radius=0.30,
                trajectory_type="line_oscillate",
                p0=(1.2, 1.2),
                speed=0.10,
                heading_deg=-45.0,
                amplitude=0.90,
                phase_deg=90.0,
                bounce=False,
            ),
        ]

        return start, goal, static_obstacles, dynamic_obstacles

    if name == "D5_Corridor":
        start = (-6.0, -6.0)
        goal = (6.0, 6.0)

        static_obstacles = [
            StaticObstacle(center=(-2.5, -2.5), radius=0.70, name="static_1"),
            StaticObstacle(center=(-2.0, -5.5), radius=0.70, name="static_2"),
            StaticObstacle(center=(-5.5, -2.0), radius=0.70, name="static_3"),
            StaticObstacle(center=(-2.0, 4.0), radius=0.70, name="static_4"),
            StaticObstacle(center=(4.0, -2.0), radius=0.70, name="static_5"),
            StaticObstacle(center=(4.5, 4.5), radius=0.70, name="static_6"),
            StaticObstacle(center=(2.0, 6.0), radius=0.50, name="static_7"),
            StaticObstacle(center=(6.0, 2.0), radius=0.50, name="static_8"),
        ]

        dynamic_obstacles = [
            DynamicObstacle(
                name="dyn_crossing",
                radius=0.40,
                trajectory_type="line_oscillate",
                p0=(-1.0, -1.0),
                speed=0.30,
                heading_deg=-45.0,
                amplitude=2.0,
                phase_deg=0.0,
                bounce=False,
            ),
            DynamicObstacle(
                name="dyn_orbit",
                radius=0.40,
                trajectory_type="circle",
                center=(2.0, 2.0),
                path_radius=1.5,
                speed=0.25,
                phase_deg=90.0,
            ),
        ]

        return start, goal, static_obstacles, dynamic_obstacles
    raise ValueError(f"Unknown scenario name: {name}")


_start, _goal, STATIC_OBSTACLES, DYNAMIC_OBSTACLES = make_scenario(SCENARIO_NAME)

START = np.array(_start, dtype=float)
GOAL = np.array(_goal, dtype=float)

OUTPUT_DIR = Path("dynamic_obstacle_env_preview") / SCENARIO_NAME


# =============================================================================
# Geometry / evaluation utilities
# =============================================================================


def _bounce_scalar(x: float, lo: float, hi: float) -> float:
    """Reflect x into [lo, hi] using a triangular-wave reflection."""
    if hi <= lo:
        return 0.5 * (lo + hi)
    span = hi - lo
    z = (x - lo) % (2.0 * span)
    if z <= span:
        return lo + z
    return hi - (z - span)


def _bounce_point(p: np.ndarray, radius: float, world_size: tuple[float, float]) -> np.ndarray:
    xmin, xmax, ymin, ymax = world_bounds(world_size)
    return np.array(
        [
            _bounce_scalar(float(p[0]), xmin + radius, xmax - radius),
            _bounce_scalar(float(p[1]), ymin + radius, ymax - radius),
        ],
        dtype=float,
    )


def _position_on_waypoints(
    waypoints: Iterable[tuple[float, float]], speed: float, t: float, loop: bool = True
) -> np.ndarray:
    pts = np.asarray(list(waypoints), dtype=float)
    if len(pts) == 0:
        raise ValueError("waypoints trajectory needs at least one waypoint")
    if len(pts) == 1 or speed <= 0:
        return pts[0].copy()

    if loop:
        seg_start = pts
        seg_end = np.vstack([pts[1:], pts[:1]])
    else:
        seg_start = pts[:-1]
        seg_end = pts[1:]

    seg_vec = seg_end - seg_start
    seg_len = np.linalg.norm(seg_vec, axis=1)
    total_len = float(np.sum(seg_len))
    if total_len <= 1e-9:
        return pts[0].copy()

    dist = speed * t
    if loop:
        dist = dist % total_len
    else:
        dist = min(dist, total_len)

    accum = 0.0
    for a, b, L in zip(seg_start, seg_end, seg_len):
        if dist <= accum + L or L <= 1e-9:
            s = 0.0 if L <= 1e-9 else (dist - accum) / L
            return a + s * (b - a)
        accum += L
    return seg_end[-1].copy()


def sample_dynamic_obstacle(obs: DynamicObstacle, times: np.ndarray) -> np.ndarray:
    return np.vstack([obs.position_at(float(t), WORLD_SIZE) for t in times])


def wall_clearance(center: np.ndarray, radius: float, world_size: tuple[float, float]) -> float:
    """Positive means the circular obstacle is inside the centered walls."""
    xmin, xmax, ymin, ymax = world_bounds(world_size)
    x, y = center
    return min(
        x - xmin - radius,
        xmax - x - radius,
        y - ymin - radius,
        ymax - y - radius,
    )


def circle_clearance(c1: np.ndarray, r1: float, c2: np.ndarray, r2: float) -> float:
    """Positive means two disks do not overlap."""
    return float(np.linalg.norm(c1 - c2) - (r1 + r2))


def robot_safe_radius(obstacle_radius: float) -> float:
    return obstacle_radius + ROBOT_RADIUS + SAFETY_MARGIN


def evaluate_scenario() -> dict[str, float | str]:
    times = np.arange(0.0, T_FINAL + 0.5 * DT, DT)

    min_dyn_wall_clearance = math.inf
    min_dyn_static_clearance = math.inf
    min_dyn_dyn_clearance = math.inf
    min_dyn_safe_wall_clearance = math.inf
    min_static_wall_clearance = math.inf
    min_static_static_clearance = math.inf

    # Static sanity checks.
    for i, obs in enumerate(STATIC_OBSTACLES):
        c = np.asarray(obs.center, dtype=float)
        min_static_wall_clearance = min(min_static_wall_clearance, wall_clearance(c, obs.radius, WORLD_SIZE))
        for other in STATIC_OBSTACLES[i + 1 :]:
            min_static_static_clearance = min(
                min_static_static_clearance,
                circle_clearance(c, obs.radius, np.asarray(other.center, dtype=float), other.radius),
            )

    # Dynamic sanity checks over sampled horizon.
    dyn_paths = [sample_dynamic_obstacle(obs, times) for obs in DYNAMIC_OBSTACLES]

    for obs, path in zip(DYNAMIC_OBSTACLES, dyn_paths):
        for p in path:
            min_dyn_wall_clearance = min(min_dyn_wall_clearance, wall_clearance(p, obs.radius, WORLD_SIZE))
            min_dyn_safe_wall_clearance = min(
                min_dyn_safe_wall_clearance,
                wall_clearance(p, robot_safe_radius(obs.radius), WORLD_SIZE),
            )
            for stat in STATIC_OBSTACLES:
                min_dyn_static_clearance = min(
                    min_dyn_static_clearance,
                    circle_clearance(p, obs.radius, np.asarray(stat.center, dtype=float), stat.radius),
                )

    for k in range(len(times)):
        for i in range(len(DYNAMIC_OBSTACLES)):
            for j in range(i + 1, len(DYNAMIC_OBSTACLES)):
                min_dyn_dyn_clearance = min(
                    min_dyn_dyn_clearance,
                    circle_clearance(
                        dyn_paths[i][k],
                        DYNAMIC_OBSTACLES[i].radius,
                        dyn_paths[j][k],
                        DYNAMIC_OBSTACLES[j].radius,
                    ),
                )

    def finite_or_nan(x: float) -> float:
        return float(x) if math.isfinite(x) else float("nan")

    return {
        "world_size_m": f"{WORLD_SIZE[0]:.2f} x {WORLD_SIZE[1]:.2f}",
        "num_static_obstacles": len(STATIC_OBSTACLES),
        "num_dynamic_obstacles": len(DYNAMIC_OBSTACLES),
        "t_final_s": T_FINAL,
        "dt_s": DT,
        "robot_radius_m": ROBOT_RADIUS,
        "safety_margin_m": SAFETY_MARGIN,
        "min_static_wall_clearance_m": finite_or_nan(min_static_wall_clearance),
        "min_static_static_clearance_m": finite_or_nan(min_static_static_clearance),
        "min_dynamic_wall_clearance_m": finite_or_nan(min_dyn_wall_clearance),
        "min_dynamic_safe_wall_clearance_m": finite_or_nan(min_dyn_safe_wall_clearance),
        "min_dynamic_static_clearance_m": finite_or_nan(min_dyn_static_clearance),
        "min_dynamic_dynamic_clearance_m": finite_or_nan(min_dyn_dyn_clearance),
    }


# =============================================================================
# Plotting
# =============================================================================


def draw_circle( ax, center, radius, *, fill=False, linestyle="-", linewidth=1.5, alpha=1.0, label=None, color=None,):
    patch = Circle(
        center,
        radius,
        fill=fill,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        label=label,
        color=color,
    )
    ax.add_patch(patch)
    return patch


def plot_environment() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    times = np.arange(0.0, T_FINAL + 0.5 * DT, DT)

    fig, ax = plt.subplots(figsize=(8, 8))
    xmin, xmax, ymin, ymax = world_bounds(WORLD_SIZE)

    # Walls / outer boundary.
    ax.add_patch(Rectangle((xmin, ymin), WORLD_SIZE[0], WORLD_SIZE[1], fill=False, linewidth=2.5, label="walls"))

    # Wall safe set: inner rectangle.
    if SHOW_SAFE_SET:
        inset = ROBOT_RADIUS + SAFETY_MARGIN
        ax.add_patch(
            Rectangle(
                (xmin + inset, ymin + inset),
                WORLD_SIZE[0] - 2.0 * inset,
                WORLD_SIZE[1] - 2.0 * inset,
                fill=False,
                linestyle="--",
                linewidth=1.5,
                label="wall safe boundary",
            )
        )

    # Start/goal and nominal straight path.
    ax.plot([START[0]], [START[1]], marker="o", markersize=8, label="start")
    ax.plot([GOAL[0]], [GOAL[1]], marker="*", markersize=12, label="goal")
    ax.plot([START[0], GOAL[0]], [START[1], GOAL[1]], linestyle=":", linewidth=1.5, label="reference path")

    # Static obstacles.
    for obs in STATIC_OBSTACLES:
        c = np.asarray(obs.center, dtype=float)
        draw_circle(ax, c, obs.radius, fill=True, alpha=0.45, label="static obstacle")
        ax.text(c[0], c[1], obs.name, ha="center", va="center", fontsize=8)
        if SHOW_SAFE_SET:
            draw_circle(ax, c, robot_safe_radius(obs.radius), fill=False, linestyle="--", linewidth=1.2, alpha=0.9)

    # Dynamic obstacles and trajectories.
    snapshot_times = np.linspace(0.0, T_FINAL, 7) if SHOW_TIME_SNAPSHOTS else np.array([0.0, T_FINAL])
    for obs in DYNAMIC_OBSTACLES:
        path = sample_dynamic_obstacle(obs, times)
        ax.plot(path[:, 0], path[:, 1], linewidth=2.0, label=f"{obs.name} center path")

        # Initial and final disk.
        p0 = path[0]
        pf = path[-1]
        draw_circle(
            ax,
            p0,
            obs.radius,
            fill=True,
            alpha=0.35,
            label="dynamic obstacle start",
            color="lightcoral",
        )

        draw_circle(
            ax,
            pf,
            obs.radius,
            fill=False,
            linestyle="-",
            linewidth=2.0,
            label="dynamic obstacle final",
            color="lightcoral",
        )
        ax.plot([p0[0]], [p0[1]], marker="s", markersize=6)
        ax.plot([pf[0]], [pf[1]], marker="x", markersize=7)

        if SHOW_SAFE_SET:
            draw_circle(ax, p0, robot_safe_radius(obs.radius), fill=False, linestyle="--", linewidth=1.2)
            draw_circle(ax, pf, robot_safe_radius(obs.radius), fill=False, linestyle="--", linewidth=1.2)

        # Swept safe set shown by translucent snapshots, not a true exact union.
        if SHOW_DYNAMIC_SWEEP:
            for t in snapshot_times:
                p = obs.position_at(float(t), WORLD_SIZE)
                draw_circle(ax, p, obs.radius, fill=False, linestyle=":", linewidth=0.8, alpha=0.7)
                if SHOW_SAFE_SET:
                    draw_circle(ax, p, robot_safe_radius(obs.radius), fill=False, linestyle=":", linewidth=0.6, alpha=0.45)
                ax.text(p[0], p[1], f"{t:.0f}s", fontsize=7)

    # Clean duplicated legend labels.
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    new_handles, new_labels = [], []
    for h, lab in zip(handles, labels):
        if lab and lab not in seen:
            new_handles.append(h)
            new_labels.append(lab)
            seen.add(lab)
    ax.legend(new_handles, new_labels, loc="upper left", fontsize=8)

    ax.set_aspect("equal", adjustable="box")
    plot_padding = 0.5
    ax.set_xlim(xmin - plot_padding, xmax + plot_padding)
    ax.set_ylim(ymin - plot_padding, ymax + plot_padding)
    ax.axhline(0.0, linewidth=0.8, linestyle=":")
    ax.axvline(0.0, linewidth=0.8, linestyle=":")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Dynamic obstacle scenario preview")
    ax.grid(True, linewidth=0.3)

    fig.tight_layout()
    if SAVE_FIGURE:
        out = OUTPUT_DIR / "dynamic_obstacle_environment_preview.png"
        fig.savefig(out, dpi=200)
        print(f"Saved figure: {out}")
    plt.show()


def main() -> None:
    summary = evaluate_scenario()
    print("\n=== Scenario sanity/evaluation summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:+.4f}" if "clearance" in k else f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    print("\nInterpretation:")
    print("  Positive clearances mean no overlap for the sampled trajectory.")
    print("  Negative dynamic-safe-wall clearance means the obstacle's robot-inflated safe set crosses the wall boundary.")
    print("  This is only a scenario-design tool; MuJoCo evaluation should still be run afterward.")

    plot_environment()


if __name__ == "__main__":
    main()
