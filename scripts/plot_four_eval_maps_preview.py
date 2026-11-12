#!/usr/bin/env python3
"""Preview of the 4 structured evaluation maps (obstacles + initial/goal
regions only, no trajectories): slalom, corridor, clustered_v2,
circle_and_square -- titled Slalom/Corridor/Clustered/Combination.
Obstacle/region coordinates copied verbatim from
evaluate_laser_unseen_maps_structured.py's UNSEEN_MAPS/
STRUCTURED_SCENARIO_REGIONS (also used by plot_seaspeed_combined_trajectories.py).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch, Rectangle

START_COLOR = "#7b2cbf"
GOAL_COLOR = "#ffd60a"
OBSTACLE_COLOR = "#6f747a"
ARENA_MIN, ARENA_MAX = -7.5, 7.5

# (x, y, radius, shape) -- shape omitted (defaults to circle) where every
# obstacle in that map is circular.
MAPS = {
    "slalom": (
        (-5.2, -2.3, 0.50), (-3.9, 2.1, 0.50), (-2.6, -2.2, 0.50),
        (-1.3, 2.2, 0.50), (0.0, -2.2, 0.50), (1.3, 2.2, 0.50),
        (2.6, -2.2, 0.50), (3.9, 2.1, 0.50), (5.2, -2.3, 0.50),
    ),
    "corridor": (
        (-3.8, -2.2, 0.50), (-1.9, -2.2, 0.50), (0.0, -2.2, 0.50),
        (1.9, -2.2, 0.50), (3.8, -2.2, 0.50), (-3.8, 2.2, 0.50),
        (-1.9, 2.2, 0.50), (1.9, 2.2, 0.50), (3.8, 2.2, 0.50),
    ),
    "clustered_v2": (
        (-3.0, 1.4, 0.55), (-1.3, 2.7, 0.45), (-0.08, 0.55, 0.60),
        (-2.4, 4.1, 0.50), (0.5, 3.9, 0.40), (1.98, -2.71, 0.50),
        (3.35, -1.44, 0.45), (-0.42, -1.55, 0.55), (3.47, -3.86, 0.50),
    ),
    "circle_and_square": (
        (-1.6, 3.4, 0.35, "circle"), (1.6, 3.4, 0.35, "circle"),
        (0.0, 1.9, 1.00, "circle"),
        (-1.5, -0.2, 0.35, "circle"), (1.5, -0.2, 0.35, "circle"),
        (-4.6, -2.3, 0.65, "circle"),
        (-1.0, -2.6, 0.40, "square"), (1.0, -2.6, 0.40, "square"),
        (4.6, -2.3, 0.65, "circle"),
        (-3.6, 0.4, 0.70, "square"), (3.6, 0.4, 0.70, "square"),
    ),
}
REGIONS = {
    "slalom": {"start": (-4.0, 4.0, 5.0, 7.5), "goal": (-4.0, 4.0, -7.5, -4.0)},
    "corridor": {"start": (-2.0, 2.0, 4.0, 7.5), "goal": (-4.0, 4.0, -7.5, -4.0)},
    "clustered_v2": {"start": (-7.5, -4.0, 4.0, 7.5), "goal": (5.5, 7.5, -7.5, -5.5)},
    "circle_and_square": {"start": (-1.0, 1.0, 5.0, 6.8), "goal": (-5.5, 5.5, -7.0, -4.0)},
}
MAP_ORDER = ("slalom", "corridor", "clustered_v2", "circle_and_square")
COLUMN_TITLES = ("Slalom", "Corridor", "Clustered", "Combination")


def add_region(ax, bounds, color) -> None:
    x_min, x_max, y_min, y_max = bounds
    ax.add_patch(Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        facecolor=color, edgecolor=color, linewidth=1.0, alpha=0.28, zorder=1,
    ))


def draw_panel(ax, map_name: str, title: str) -> None:
    add_region(ax, REGIONS[map_name]["start"], START_COLOR)
    add_region(ax, REGIONS[map_name]["goal"], GOAL_COLOR)
    for obstacle in MAPS[map_name]:
        x, y, radius = obstacle[0], obstacle[1], obstacle[2]
        shape = obstacle[3] if len(obstacle) > 3 else "circle"
        if shape == "square":
            ax.add_patch(Rectangle(
                (x - radius, y - radius), 2 * radius, 2 * radius,
                facecolor=OBSTACLE_COLOR, edgecolor="black", linewidth=0.8, zorder=2, alpha=0.9,
            ))
        else:
            ax.add_patch(Circle(
                (x, y), radius, facecolor=OBSTACLE_COLOR, edgecolor="black",
                linewidth=0.8, zorder=2, alpha=0.9,
            ))
    ax.add_patch(Rectangle(
        (ARENA_MIN, ARENA_MIN), ARENA_MAX - ARENA_MIN, ARENA_MAX - ARENA_MIN,
        fill=False, edgecolor="black", linewidth=1.6, zorder=4,
    ))

    ax.set_xlim(ARENA_MIN - 0.15, ARENA_MAX + 0.15)
    ax.set_ylim(ARENA_MIN - 0.15, ARENA_MAX + 0.15)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks((-5, 0, 5))
    ax.set_yticks((-5, 0, 5))
    ax.tick_params(labelsize=9, length=3)
    ax.grid(False)
    ax.set_xlabel("x [m]", fontsize=10)
    ax.set_title(title, fontsize=13, pad=8)


def main() -> None:
    fig, axes = plt.subplots(1, len(MAP_ORDER), figsize=(4.0 * len(MAP_ORDER), 4.4))
    for idx, (ax, map_name) in enumerate(zip(axes, MAP_ORDER)):
        draw_panel(ax, map_name, COLUMN_TITLES[idx])
        if idx == 0:
            ax.set_ylabel("y [m]", fontsize=10)
        else:
            ax.tick_params(labelleft=False)

    legend_handles = [
        Patch(facecolor=START_COLOR, edgecolor=START_COLOR, alpha=0.28, label="Initial region"),
        Patch(facecolor=GOAL_COLOR, edgecolor=GOAL_COLOR, alpha=0.28, label="Goal region"),
        Patch(facecolor=OBSTACLE_COLOR, edgecolor="black", label="Obstacle (circle or square)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.06), fontsize=10, columnspacing=1.4)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.90, bottom=0.18, wspace=0.08)

    output_base = Path("logs/evaluation/figures/four_eval_maps_preview")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out_path = output_base.with_suffix(f".{ext}")
        fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
