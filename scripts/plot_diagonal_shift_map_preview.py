#!/usr/bin/env python3
"""Preview of a new candidate structured eval map, "diagonal" -- intended
difficulty between corridor (Medium-Shift) and circle_and_square
(High-Shift). Same visual convention as plot_three_eval_maps_preview.py
(colors/style copied verbatim) so it can be compared side by side, or folded
into that script later once approved.

v2 (2026-09-23): the first attempt (single-file zigzag chain, start
top-left/goal bottom-right) was too close in overall shape to the existing
clustered_v2 map. Flipped the traversal to the other diagonal and replaced
the chain with a rejection-sampled scattered field -- still felt "chaotic"
per feedback.

v3 (2026-09-23): rebuilt around explicit 180-degree point symmetry about the
arena center (for every obstacle at (x,y) there is one at (-x,-y)) so the
field reads as deliberately symmetric between the start and goal corners,
not scattered. Deliberately loose (min gap 1.04 m) -- but a 100-case pilot
eval showed this made the map too EASY for SEA-Nav (89% success, all real
attempts succeeded) despite being meaningfully hard for SAFTER (68%) --
the opposite of corridor/circle_and_square, where SEA-Nav fails much more
than SAFTER (corridor: SEA-Nav 48.6% vs SAFTER 88%; circle_and_square:
SEA-Nav 25.2% vs SAFTER 47.2%).

v4 (2026-09-23): root-caused via corridor's own SEA-Nav trajectories (SEA-Nav
seed 1000, medium/env4/ep0: approaches the gate normally, gets to within
~0.44 m of an obstacle edge right at the gate, then physically STALLS --
x,y barely move for 170+ steps -- until the episode ends as "other_failure",
not "collision". Corridor's dominant SEA-Nav failure mode is this stall
(51.2% other_failure, only 0.2% collision), not crashing -- its CBF-shield
policy gets stuck in a local minimum right at a genuinely tight (~0.9 m)
gate. v3's loosest gap (1.04 m) never triggers this. Reintroduced ONE
corridor-tightness gate, symmetric (both copies tightened together to
preserve the (x,y)->(-x,-y) symmetry): the P2-P3 gap tightened from 1.04 m
to 0.88 m by moving P3/P3' 0.32 m toward P2/P2' along their connecting
line. Nothing else changed. New minimum gap 0.88 m (was corridor's own
~0.9 m the reference point); next-tightest gap is now 1.21 m so this is a
single deliberate pinch, not a general tightening.

Verified before rendering: no obstacle-obstacle overlap, minimum
obstacle-region surface clearance 2.20 m, minimum obstacle-wall clearance
2.25 m, the new gate gap is 0.880 m (both symmetric copies).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch, Rectangle

START_COLOR = "#7b2cbf"
GOAL_COLOR = "#ffd60a"
OBSTACLE_COLOR = "#6f747a"
ARENA_MIN, ARENA_MAX = -7.5, 7.5

# (x, y, radius) -- all circles. Explicit 180-degree point symmetry about
# the origin: the center circle is self-symmetric, and every other circle
# has its (-x, -y) mirror also in this list.
DIAGONAL_OBSTACLES = (
    (0.0, 0.0, 1.0),        # center blocker
    (-4.5, -0.75, 0.65), (4.5, 0.75, 0.65),    # P1 / P1' -- widened away from P3 (was an accidental 1.15m pinch)
    (-0.7, -3.6, 0.60), (0.7, 3.6, 0.60),      # P2 / P2'
    (-2.569, -2.759, 0.55), (2.569, 2.759, 0.55),  # P3 / P3' -- gate retargeted to 0.899 m (corridor's own value)
    (-4.7, 1.7, 0.55), (4.7, -1.7, 0.55),      # P4 / P4'
    (-1.7, 4.7, 0.50), (1.7, -4.7, 0.50),      # P5 / P5'
)
DIAGONAL_REGIONS = {
    "start": (-7.5, -4.6, -7.5, -4.6),  # bottom-left corner
    "goal": (4.6, 7.5, 4.6, 7.5),       # top-right corner
}
TITLE = "Diagonal (candidate v5)"

FS_TITLE = 44
FS_LABEL = 28
FS_TICK = 24
FS_LEGEND = 23


def add_region(ax, bounds, color) -> None:
    x_min, x_max, y_min, y_max = bounds
    ax.add_patch(Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        facecolor=color, edgecolor=color, linewidth=1.0, alpha=0.28, zorder=1,
    ))


def draw_panel(ax) -> None:
    add_region(ax, DIAGONAL_REGIONS["start"], START_COLOR)
    add_region(ax, DIAGONAL_REGIONS["goal"], GOAL_COLOR)
    for x, y, radius in DIAGONAL_OBSTACLES:
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
    ax.set_title(TITLE, fontsize=FS_TITLE, fontweight="bold", pad=16)


PANEL_IN = 6.5
LEFT_MARGIN_IN = 1.15
RIGHT_MARGIN_IN = 0.2
TOP_MARGIN_IN = 1.35
BOTTOM_GAP_IN = 1.15
LEGEND_H_IN = 0.7


def main() -> None:
    fig_w = LEFT_MARGIN_IN + PANEL_IN + RIGHT_MARGIN_IN
    fig_h = TOP_MARGIN_IN + PANEL_IN + BOTTOM_GAP_IN + LEGEND_H_IN
    fig = plt.figure(figsize=(fig_w, fig_h))

    axes_bottom_in = LEGEND_H_IN + BOTTOM_GAP_IN
    ax = fig.add_axes([
        LEFT_MARGIN_IN / fig_w, axes_bottom_in / fig_h, PANEL_IN / fig_w, PANEL_IN / fig_h,
    ])
    draw_panel(ax)

    legend_handles = [
        Patch(facecolor=START_COLOR, edgecolor=START_COLOR, alpha=0.28, label="Initial region"),
        Patch(facecolor=GOAL_COLOR, edgecolor=GOAL_COLOR, alpha=0.28, label="Goal region"),
        Patch(facecolor=OBSTACLE_COLOR, edgecolor="black", label="Obstacle"),
    ]
    fig.legend(
        handles=legend_handles, loc="center", ncol=3, frameon=False, fontsize=FS_LEGEND, columnspacing=1.8,
        bbox_to_anchor=(0.5, (LEGEND_H_IN / 2) / fig_h), bbox_transform=fig.transFigure,
    )

    output_base = Path("logs/evaluation/figures/diagonal_shift_map_preview_v5")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out_path = output_base.with_suffix(f".{ext}")
        fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
