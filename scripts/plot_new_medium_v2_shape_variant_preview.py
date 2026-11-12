#!/usr/bin/env python3
"""Preview: current new_medium_v2 (all circles, symmetric start/goal boxes)
vs. a proposed shape-variant (small obstacles as squares, smaller start
box, larger goal box), side by side. PREVIEW ONLY -- not yet wired into
any eval script; obstacle radii for the square candidates use the
already-supported CircularObstacle.shape="square" field (radius = half-side).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch, Rectangle

START_COLOR = "#7b2cbf"
GOAL_COLOR = "#ffd60a"
OBSTACLE_COLOR = "#6f747a"
ARENA_MIN, ARENA_MAX = -7.5, 7.5

# (x, y, size, shape) -- size = radius for circle, half-side for square
CURRENT = {
    "obstacles": [
        (0.0, 2.4, 1.00, "circle"), (-1.6, -0.1, 1.00, "circle"), (1.6, -0.1, 1.00, "circle"),
        (-3.7, -1.9, 0.45, "circle"), (-4.9, -3.4, 0.50, "circle"),
        (3.7, -1.9, 0.45, "circle"), (4.9, -3.4, 0.50, "circle"),
    ],
    "start": (-2.0, 2.0, 4.0, 7.5),
    "goal": (-4.0, 4.0, -7.5, -4.0),
}
PROPOSED = {
    # Sketch-faithful, mirror-symmetric about x=0: small pair -> big center
    # circle -> small pair -> (big circle / square / square / big circle)
    # row, matching the hand-drawn layout exactly (same 9-obstacle count as
    # slalom/corridor/clustered_v2).
    "obstacles": [
        (-1.6, 3.4, 0.35, "circle"),   # row1 left (small)
        (1.6, 3.4, 0.35, "circle"),    # row1 right (small)
        (0.0, 1.9, 1.00, "circle"),    # center (big)
        (-1.5, -0.2, 0.35, "circle"),  # row2 left (small)
        (1.5, -0.2, 0.35, "circle"),   # row2 right (small)
        (-4.6, -2.3, 0.65, "circle"),  # bottom-left (big)
        (-1.0, -2.6, 0.40, "square"),  # bottom square left
        (1.0, -2.6, 0.40, "square"),   # bottom square right
        (4.6, -2.3, 0.65, "circle"),   # bottom-right (big)
        (-3.6, 0.4, 0.70, "square"),   # large square, left of cluster
        (3.6, 0.4, 0.70, "square"),    # large square, right of cluster
    ],
    "start": (-1.0, 1.0, 5.0, 6.8),
    "goal": (-5.5, 5.5, -7.0, -4.0),
}
TITLES = ("new_medium_v2 (current)", "new_medium_v2 (sketch-based, proposed)")


def add_region(ax, bounds, color) -> None:
    x_min, x_max, y_min, y_max = bounds
    ax.add_patch(Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        facecolor=color, edgecolor=color, linewidth=1.2, alpha=0.32, zorder=1,
    ))


def draw_map(ax, spec: dict, title: str, show_ylabel: bool) -> None:
    add_region(ax, spec["start"], START_COLOR)
    add_region(ax, spec["goal"], GOAL_COLOR)
    for x, y, size, shape in spec["obstacles"]:
        if shape == "square":
            ax.add_patch(Rectangle(
                (x - size, y - size), 2 * size, 2 * size,
                facecolor=OBSTACLE_COLOR, edgecolor="black", linewidth=1.0, zorder=3,
            ))
        else:
            ax.add_patch(Circle(
                (x, y), size, facecolor=OBSTACLE_COLOR, edgecolor="black",
                linewidth=1.0, zorder=3,
            ))
    ax.add_patch(Rectangle(
        (ARENA_MIN, ARENA_MIN), ARENA_MAX - ARENA_MIN, ARENA_MAX - ARENA_MIN,
        fill=False, edgecolor="black", linewidth=2.0, zorder=4,
    ))
    ax.set_xlim(ARENA_MIN - 0.15, ARENA_MAX + 0.15)
    ax.set_ylim(ARENA_MIN - 0.15, ARENA_MAX + 0.15)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks((-5, 0, 5))
    ax.set_yticks((-5, 0, 5))
    ax.tick_params(labelsize=10, length=3)
    ax.grid(False)
    ax.set_xlabel("x [m]", fontsize=11)
    if show_ylabel:
        ax.set_ylabel("y [m]", fontsize=11)
    else:
        ax.tick_params(labelleft=False)
    ax.set_title(title, fontsize=13, pad=8)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.2))
    draw_map(axes[0], CURRENT, TITLES[0], show_ylabel=True)
    draw_map(axes[1], PROPOSED, TITLES[1], show_ylabel=False)

    fig.legend(
        handles=(
            Patch(facecolor=START_COLOR, edgecolor=START_COLOR, alpha=0.32, label="Initial region"),
            Patch(facecolor=GOAL_COLOR, edgecolor=GOAL_COLOR, alpha=0.32, label="Goal region"),
            Patch(facecolor=OBSTACLE_COLOR, edgecolor="black", label="Obstacle (circle or square)"),
        ),
        loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=10,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.20, wspace=0.1)

    output_base = Path("logs/evaluation/figures/new_medium_v2_shape_variant_preview")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out_path = output_base.with_suffix(f".{ext}")
        fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
