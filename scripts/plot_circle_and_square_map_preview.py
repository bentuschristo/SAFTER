#!/usr/bin/env python3
"""Single-panel preview of the circle_and_square map (obstacles + initial/
goal regions only, no trajectories) -- reflects the 2026-09-24 eased layout:
center circle radius 1.00 -> 0.70, and the two small squares at (-1.0/1.0,
-2.6) removed. Obstacle/region coordinates copied verbatim from
evaluate_laser_unseen_maps_structured.py's UNSEEN_MAPS["circle_and_square"] /
STRUCTURED regions.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch, Rectangle

START_COLOR = "#7b2cbf"
GOAL_COLOR = "#ffd60a"
OBSTACLE_COLOR = "#6f747a"
ARENA_MIN, ARENA_MAX = -7.5, 7.5

# (x, y, radius, shape)
OBSTACLES = (
    (-1.6, 3.4, 0.35, "circle"),
    (1.6, 3.4, 0.35, "circle"),
    (0.0, 1.9, 0.70, "circle"),
    (-1.5, -0.2, 0.35, "circle"),
    (1.5, -0.2, 0.35, "circle"),
    (-4.6, -2.3, 0.65, "circle"),
    (4.6, -2.3, 0.65, "circle"),
    (-3.6, 0.4, 0.70, "square"),
    (3.6, 0.4, 0.70, "square"),
)
START_REGION = (-1.0, 1.0, 5.0, 6.8)
GOAL_REGION = (-5.5, 5.5, -7.0, -4.0)

FS_TITLE = 33
FS_LABEL = 28
FS_TICK = 24
FS_LEGEND = 23


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.5, 7.2))

    x_min, x_max, y_min, y_max = START_REGION
    ax.add_patch(Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        facecolor=START_COLOR, edgecolor=START_COLOR, linewidth=1.0, alpha=0.28, zorder=1,
    ))
    x_min, x_max, y_min, y_max = GOAL_REGION
    ax.add_patch(Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        facecolor=GOAL_COLOR, edgecolor=GOAL_COLOR, linewidth=1.0, alpha=0.28, zorder=1,
    ))

    for x, y, radius, shape in OBSTACLES:
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
    ax.tick_params(labelsize=FS_TICK, length=5)
    ax.grid(False)
    ax.set_xlabel("x [m]", fontsize=FS_LABEL)
    ax.set_ylabel("y [m]", fontsize=FS_LABEL)
    ax.set_title("Circle & Square (Eased)", fontsize=FS_TITLE, fontweight="bold", pad=16)

    legend_handles = [
        Patch(facecolor=START_COLOR, edgecolor=START_COLOR, alpha=0.28, label="Initial region"),
        Patch(facecolor=GOAL_COLOR, edgecolor=GOAL_COLOR, alpha=0.28, label="Goal region"),
        Patch(facecolor=OBSTACLE_COLOR, edgecolor="black", label="Obstacle"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=3, frameon=False, fontsize=FS_LEGEND,
        columnspacing=1.8, bbox_to_anchor=(0.5, 0.0),
    )
    fig.subplots_adjust(left=0.22, right=0.95, top=0.93, bottom=0.14)

    output_base = Path("logs/evaluation/figures/circle_and_square_map_preview")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out_path = output_base.with_suffix(f".{ext}")
        fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
