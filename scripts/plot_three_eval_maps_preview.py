#!/usr/bin/env python3
"""Preview of the 3 structured evaluation maps actually used in the
seaspeed comparisons (obstacles + initial/goal regions only, no
trajectories): slalom, corridor, diagonal -- titled
Low-Shift/Medium-Shift/High-Shift. Obstacle/region coordinates copied
verbatim from evaluate_laser_unseen_maps_structured.py's UNSEEN_MAPS/
STRUCTURED_SCENARIO_REGIONS.

2026-09-24: High-Shift slot uses corridor-derived v5, matching the map used
for the latest high-shift evaluation.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch, Rectangle
from plot_corridor_derived_highshift_v5_preview import (
    OBSTACLES as HIGH_SHIFT_V5_OBSTACLES,
    START_REGION as HIGH_SHIFT_V5_START_REGION,
    GOAL_REGION as HIGH_SHIFT_V5_GOAL_REGION,
)

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
    "corridor_derived_highshift_v5": HIGH_SHIFT_V5_OBSTACLES,
}
REGIONS = {
    "slalom": {"start": (-4.0, 4.0, 5.0, 7.5), "goal": (-4.0, 4.0, -7.5, -4.0)},
    "corridor": {"start": (-2.0, 2.0, 4.0, 7.5), "goal": (-4.0, 4.0, -7.5, -4.0)},
    "corridor_derived_highshift_v5": {
        "start": HIGH_SHIFT_V5_START_REGION,
        "goal": HIGH_SHIFT_V5_GOAL_REGION,
    },
}
MAP_ORDER = ("slalom", "corridor", "corridor_derived_highshift_v5")
COLUMN_TITLES = ("Low-Shift", "Medium-Shift", "High-Shift")

# Font sizes matched to scripts/plot_seaspeed_combined_trajectories_with_
# cbfrl_regenerated.py's titles and scripts/plot_handcrafted2_maps_preview_
# 2row.py's labels/ticks/legend (2026-09-22, per request).
FS_TITLE = 24.75
FS_LABEL = 28
FS_TICK = 24
FS_LEGEND = 23


def add_region(ax, bounds, color) -> None:
    x_min, x_max, y_min, y_max = bounds
    ax.add_patch(Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        facecolor=color, edgecolor=color, linewidth=1.0, alpha=0.28, zorder=1,
    ))


def draw_panel(ax, map_name: str, title: str, is_leftmost: bool) -> None:
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
    ax.tick_params(labelsize=FS_TICK, length=5)
    ax.grid(False)
    ax.set_xlabel("x [m]", fontsize=FS_LABEL)
    if is_leftmost:
        ax.set_ylabel("y [m]", fontsize=FS_LABEL)
    else:
        ax.tick_params(labelleft=False)
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", pad=16)


# Explicit inch-based axes placement (not plt.subplots/GridSpec wspace):
# each panel is equal-aspect on IDENTICAL data limits for all 3 maps, so its
# rendered size is driven by whichever of (available slot width, available
# slot height) is smaller -- here that's height, so a GridSpec's wspace only
# shrinks the empty slot around an already-height-limited square axes and
# barely changes the real visible gap. Placing axes at exact inch positions
# sidesteps this entirely.
PANEL_IN = 4.6
COL_GAP_IN = 0.12
LEFT_MARGIN_IN = 1.05
RIGHT_MARGIN_IN = 0.15
TOP_MARGIN_IN = 1.35
# Gap below each panel's data area that its own x-tick numbers + "x [m]"
# label render into (not empty space -- just room for that text).
BOTTOM_GAP_IN = 1.15
LEGEND_H_IN = 0.7


def main() -> None:
    n = len(MAP_ORDER)
    fig_w = LEFT_MARGIN_IN + n * PANEL_IN + (n - 1) * COL_GAP_IN + RIGHT_MARGIN_IN
    fig_h = TOP_MARGIN_IN + PANEL_IN + BOTTOM_GAP_IN + LEGEND_H_IN
    fig = plt.figure(figsize=(fig_w, fig_h))

    axes_bottom_in = LEGEND_H_IN + BOTTOM_GAP_IN
    for idx, map_name in enumerate(MAP_ORDER):
        x_in = LEFT_MARGIN_IN + idx * (PANEL_IN + COL_GAP_IN)
        ax = fig.add_axes([x_in / fig_w, axes_bottom_in / fig_h, PANEL_IN / fig_w, PANEL_IN / fig_h])
        draw_panel(ax, map_name, COLUMN_TITLES[idx], is_leftmost=(idx == 0))

    legend_handles = [
        Patch(facecolor=START_COLOR, edgecolor=START_COLOR, alpha=0.28, label="Initial region"),
        Patch(facecolor=GOAL_COLOR, edgecolor=GOAL_COLOR, alpha=0.28, label="Goal region"),
        Patch(facecolor=OBSTACLE_COLOR, edgecolor="black", label="Obstacle"),
    ]
    fig.legend(
        handles=legend_handles, loc="center", ncol=3, frameon=False, fontsize=FS_LEGEND, columnspacing=1.8,
        bbox_to_anchor=(0.5, (LEGEND_H_IN / 2) / fig_h), bbox_transform=fig.transFigure,
    )

    output_base = Path("logs/evaluation/figures/three_eval_maps_preview")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out_path = output_base.with_suffix(f".{ext}")
        fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
