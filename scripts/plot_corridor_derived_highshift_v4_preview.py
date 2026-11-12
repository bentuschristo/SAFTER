#!/usr/bin/env python3
"""Preview of a v2 revision of the corridor-derived High-Shift candidate.

Top row (near start, y=+2.2): reverted to uniform circles (like corridor
itself), but bigger (r=0.65 vs corridor's 0.50) and the gate deliberately
TIGHTENED to 0.70m (below corridor's own 0.90m) -- a genuinely harder
circles-only row, not relying on the square-corner effect at all here.

Bottom row (near goal, y=-2.2): keeps the circle/square/CENTER/square/circle
pattern, but the center obstacle is now a LARGER square (0.90 half-side, up
from 0.75) instead of a circle. Gap kept at corridor's own 0.90m throughout
this row (unchanged from v1) -- only the center obstacle's size/shape and
the resulting spacing needed to preserve that gap changed.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch, Rectangle

START_COLOR = "#7b2cbf"
GOAL_COLOR = "#ffd60a"
OBSTACLE_COLOR = "#6f747a"
ARENA_MIN, ARENA_MAX = -7.5, 7.5

# Top row: uniform bigger circles, tighter 0.70m gate -- center circle
# removed (2026-09-24 revision), leaving a wide-open center passage while
# the four remaining circles keep their original 0.70m-gate positions.
TOP_R = 0.65
TOP_GAP = 0.70
TOP_SPACING = 2 * TOP_R + TOP_GAP  # center-to-center
TOP_X = [-2 * TOP_SPACING, -TOP_SPACING, TOP_SPACING, 2 * TOP_SPACING]
# 2026-09-24 v4: reverted back to plain circles -- v2/v3's top-row squares
# (flanking the gate) proved catastrophic for SEA-Nav (0.0% success,
# reproduced on two separate configurations). Same positions/radius/gate
# as v2/v3, shape only.
TOP_SHAPE = ["circle", "circle", "circle", "circle"]
TOP_OBS = [(x, 2.2, TOP_R, shape) for x, shape in zip(TOP_X, TOP_SHAPE)]

# Bottom row (2026-09-24 revision): only the CENTER obstacle stays a
# square -- the two flanking "squares" are reverted to circles at the same
# radius (0.65), isolating the square-corner effect to just the one
# center obstacle. Every gap still solved to stay exactly 0.90m (corridor's
# own gap width).
BOTTOM_R = [0.50, 0.65, 0.90, 0.65, 0.50]
BOTTOM_SHAPE = ["circle", "circle", "square", "circle", "circle"]
GAP = 0.90
xs = [0.0, 0.0, 0.0, 0.0, 0.0]
xs[2] = 0.0
xs[3] = xs[2] + BOTTOM_R[2] + BOTTOM_R[3] + GAP
xs[4] = xs[3] + BOTTOM_R[3] + BOTTOM_R[4] + GAP
xs[1] = -xs[3]
xs[0] = -xs[4]
BOTTOM_OBS = [(xs[i], -2.2, BOTTOM_R[i], BOTTOM_SHAPE[i]) for i in range(5)]

OBSTACLES = TOP_OBS + BOTTOM_OBS

START_REGION = (-2.0, 2.0, 4.0, 7.5)
GOAL_REGION = (-4.0, 4.0, -7.5, -4.0)

FS_TITLE = 30
FS_LABEL = 28
FS_TICK = 24
FS_LEGEND = 23


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 7.2))

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
    ax.set_title("High-Shift (corridor-derived v4)", fontsize=FS_TITLE, fontweight="bold", pad=16)

    legend_handles = [
        Patch(facecolor=START_COLOR, edgecolor=START_COLOR, alpha=0.28, label="Initial region"),
        Patch(facecolor=GOAL_COLOR, edgecolor=GOAL_COLOR, alpha=0.28, label="Goal region"),
        Patch(facecolor=OBSTACLE_COLOR, edgecolor="black", label="Obstacle"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=3, frameon=False, fontsize=FS_LEGEND,
        columnspacing=1.8, bbox_to_anchor=(0.5, -0.06),
    )
    fig.subplots_adjust(left=0.20, right=0.96, top=0.93, bottom=0.20)

    output_base = Path("logs/evaluation/figures/corridor_derived_highshift_v4_preview")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out_path = output_base.with_suffix(f".{ext}")
        fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out_path}")
    plt.close(fig)

    print("\nObstacle list (x, y, radius, shape):")
    for x, y, r, shape in OBSTACLES:
        print(f"  ({x:6.2f}, {y:5.2f}, {r:.2f}, {shape})")
    print(f"\nTop row: uniform circles r={TOP_R}, gate={TOP_GAP}m (tighter than corridor's 0.90m)")
    print(f"Bottom row: gate={GAP}m throughout (same as corridor), center obstacle now square r={BOTTOM_R[2]}")


if __name__ == "__main__":
    main()
