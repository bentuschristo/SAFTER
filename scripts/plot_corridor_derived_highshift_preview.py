#!/usr/bin/env python3
"""Preview of a candidate NEW High-Shift map, derived directly from
corridor's (Medium-Shift's) proven two-row pillar layout rather than
tuning circle_and_square further.

Design: same two-row structure and start/goal regions as corridor, and
EVERY adjacent-obstacle gap within a row is kept at exactly corridor's own
0.9m (1.9m center spacing - 2*0.50m radius), the gap width that already
produces corridor's balanced difficulty split (bcbf_cf ~88.6%, SEA-Nav
~50%). Row obstacles are no longer uniform 0.50m circles though: mirrored
pattern per row (outer-to-center) is
    circle r=0.50 -> square r=0.65 (half-side) -> circle r=0.75 (center)
Center-to-center spacing between each adjacent pair is solved so the gap
stays 0.9m even as radii change: dist(i,i+1) = r_i + r_(i+1) + 0.9.
Both rows use the identical x-positions (like corridor), one near the
start region (y=+2.2) and one near the goal region (y=-2.2).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch, Rectangle

START_COLOR = "#7b2cbf"
GOAL_COLOR = "#ffd60a"
OBSTACLE_COLOR = "#6f747a"
ARENA_MIN, ARENA_MAX = -7.5, 7.5

# Row pattern (outer -> center), mirrored: (x_offset_from_prev, radius, shape)
# solved so every adjacent gap = 0.9m (corridor's own gap width).
ROW_X = [-4.35, -2.30, 0.0, 2.30, 4.35]
ROW_R = [0.50, 0.65, 0.75, 0.65, 0.50]
ROW_SHAPE = ["circle", "square", "circle", "square", "circle"]

OBSTACLES = []
for y in (-2.2, 2.2):
    for x, r, shape in zip(ROW_X, ROW_R, ROW_SHAPE):
        OBSTACLES.append((x, y, r, shape))

# Reused verbatim from corridor (evaluate_laser_unseen_maps_structured.py's
# STRUCTURED regions for "corridor").
START_REGION = (-2.0, 2.0, 4.0, 7.5)
GOAL_REGION = (-4.0, 4.0, -7.5, -4.0)

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
    ax.set_title("High-Shift (corridor-derived)", fontsize=FS_TITLE, fontweight="bold", pad=16)

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

    output_base = Path("logs/evaluation/figures/corridor_derived_highshift_preview")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out_path = output_base.with_suffix(f".{ext}")
        fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out_path}")
    plt.close(fig)

    print("\nObstacle list (x, y, radius, shape):")
    for x, y, r, shape in OBSTACLES:
        print(f"  ({x:6.2f}, {y:5.2f}, {r:.2f}, {shape})")


if __name__ == "__main__":
    main()
