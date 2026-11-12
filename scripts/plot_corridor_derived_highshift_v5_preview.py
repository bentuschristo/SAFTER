#!/usr/bin/env python3
"""Preview a V-shaped v5 variant of the corridor-derived High-Shift map.

The upper obstacle row and start/goal regions match v4. In the lower row,
the four circles rise toward the outer sides while preserving 0.80 m
surface-to-surface clearance between neighboring obstacles, including the
central square.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch, Rectangle

START_COLOR = "#7b2cbf"
GOAL_COLOR = "#ffd60a"
OBSTACLE_COLOR = "#6f747a"
ARENA_MIN, ARENA_MAX = -7.5, 7.5

# Preserve the v4 upper row.
TOP_R = 0.65
TOP_GAP = 0.70
TOP_SPACING = 2 * TOP_R + TOP_GAP
TOP_X = [-2 * TOP_SPACING, -TOP_SPACING, TOP_SPACING, 2 * TOP_SPACING]
OBSTACLES = [(x, 2.2, TOP_R, "circle") for x in TOP_X]

# Lower row: the center square stays at the v4 location. The inner circles
# are raised 0.9 m and the outer circles 1.8 m, forming a symmetric V.
# Horizontal offsets are solved from Euclidean edge-to-edge gaps of 0.80 m.
GAP = 0.80
SQUARE_HALF_SIDE = 0.90
SQUARE_Y = -2.2
INNER_R, OUTER_R = 0.65, 0.50
INNER_Y, OUTER_Y = -1.3, -0.4

# Distance from the square's vertical side to the inner circle center.
square_to_inner_dx = SQUARE_HALF_SIDE + INNER_R + GAP
# Circle-to-circle center distance needed for GAP at the vertical offset.
inner_outer_dy = abs(INNER_Y - OUTER_Y)
inner_outer_center_distance = INNER_R + OUTER_R + GAP
if inner_outer_dy >= inner_outer_center_distance:
    raise ValueError("Vertical offset leaves no horizontal room for the requested gap")
inner_outer_dx = math.sqrt(inner_outer_center_distance**2 - inner_outer_dy**2)
inner_x = square_to_inner_dx
outer_x = inner_x + inner_outer_dx

OBSTACLES.extend([
    (-outer_x, OUTER_Y, OUTER_R, "circle"),
    (-inner_x, INNER_Y, INNER_R, "circle"),
    (0.0, SQUARE_Y, SQUARE_HALF_SIDE, "square"),
    (inner_x, INNER_Y, INNER_R, "circle"),
    (outer_x, OUTER_Y, OUTER_R, "circle"),
])

START_REGION = (-2.0, 2.0, 4.0, 7.5)
GOAL_REGION = (-4.0, 4.0, -7.5, -4.0)

FS_TITLE = 30
FS_LABEL = 28
FS_TICK = 24
FS_LEGEND = 23


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 7.2))

    for region, color in ((START_REGION, START_COLOR), (GOAL_REGION, GOAL_COLOR)):
        x0, x1, y0, y1 = region
        ax.add_patch(Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            facecolor=color, edgecolor=color, linewidth=1.0, alpha=0.28, zorder=1,
        ))

    for x, y, radius, shape in OBSTACLES:
        if shape == "square":
            ax.add_patch(Rectangle(
                (x - radius, y - radius), 2 * radius, 2 * radius,
                facecolor=OBSTACLE_COLOR, edgecolor="black", linewidth=0.8,
                zorder=2, alpha=0.9,
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
    ax.set_title("High-Shift (corridor-derived v5)", fontsize=FS_TITLE, fontweight="bold", pad=16)

    legend_handles = [
        Patch(facecolor=START_COLOR, edgecolor=START_COLOR, alpha=0.28, label="Initial region"),
        Patch(facecolor=GOAL_COLOR, edgecolor=GOAL_COLOR, alpha=0.28, label="Goal region"),
        Patch(facecolor=OBSTACLE_COLOR, edgecolor="black", label="Obstacle"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=3, frameon=False,
        fontsize=FS_LEGEND, columnspacing=1.8, bbox_to_anchor=(0.5, -0.06),
    )
    fig.subplots_adjust(left=0.20, right=0.96, top=0.93, bottom=0.20)

    output_base = Path("logs/evaluation/figures/corridor_derived_highshift_v5_preview")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out_path = output_base.with_suffix(f".{ext}")
        fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out_path}")
    plt.close(fig)

    print("\nObstacle list (x, y, radius, shape):")
    for x, y, radius, shape in OBSTACLES:
        print(f"  ({x:6.3f}, {y:5.2f}, {radius:.2f}, {shape})")
    print(f"\nLower-row adjacent surface gaps: {GAP:.2f} m")
    print(f"Inner circle y={INNER_Y:.2f}; outer circle y={OUTER_Y:.2f}")


if __name__ == "__main__":
    main()
