#!/usr/bin/env python3
"""Preview clustered vs clustered_v2 side by side.

clustered_v2 (2026-09-11): eases the two obstacle clusters that made
"clustered" the hardest of the five structured unseen maps. Each cluster is
scaled 1.30x outward from its own centroid (every in-cluster gap now
>=0.83 m, vs. the original's tightest 0.640 m), the bottom-right cluster is
additionally shifted 1.8 m toward the arena center with its leftmost
obstacle nudged a further 0.3 m left, and the goal region is enlarged from
a 1.5x1.5 m box to 2.0x2.0 m. The overall tightest gap in clustered_v2 is
the cross-cluster one at 0.686 m, still looser than the original's worst
gap. Definitions live in
scripts/laser_unseen_eval/evaluate_laser_unseen_maps_structured.py's
UNSEEN_MAPS / STRUCTURED_SCENARIO_REGIONS ("clustered_v2"); this script
just visualizes them without importing the (heavy, mjlab-dependent) eval
module -- the numbers below are copied from there.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch, Rectangle

START_COLOR = "#7b2cbf"
GOAL_COLOR = "#ffd60a"
OBSTACLE_COLOR = "#6f747a"
ARENA_MIN, ARENA_MAX = -7.5, 7.5

MAPS = {
    "clustered": (
        (-2.6, 1.7, 0.55), (-1.3, 2.7, 0.45), (-0.2, 1.5, 0.60),
        (-2.1, 3.8, 0.50), (0.1, 3.6, 0.40), (3.8, -3.6, 0.50),
        (5.0, -2.4, 0.45), (3.1, -1.9, 0.55), (5.2, -4.8, 0.50),
    ),
    "clustered_v2": (
        (-3.0, 1.4, 0.55), (-1.3, 2.7, 0.45), (-0.08, 0.55, 0.60),
        (-2.4, 4.1, 0.50), (0.5, 3.9, 0.40), (1.98, -2.71, 0.50),
        (3.35, -1.44, 0.45), (-0.42, -1.55, 0.55), (3.47, -3.86, 0.50),
    ),
}
REGIONS = {
    "clustered": {"start": (-7.5, -4.0, 4.0, 7.5), "goal": (6.0, 7.5, -7.5, -6.0)},
    "clustered_v2": {"start": (-7.5, -4.0, 4.0, 7.5), "goal": (5.5, 7.5, -7.5, -5.5)},
}
TITLES = {"clustered": "Clustered (original)", "clustered_v2": "Clustered v2 (proposed)"}


def add_region(ax, bounds, color) -> None:
    x_min, x_max, y_min, y_max = bounds
    ax.add_patch(Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        facecolor=color, edgecolor=color, linewidth=1.2, alpha=0.32, zorder=1,
    ))


def draw_map(ax, map_name: str, show_ylabel: bool) -> None:
    add_region(ax, REGIONS[map_name]["start"], START_COLOR)
    add_region(ax, REGIONS[map_name]["goal"], GOAL_COLOR)
    for x, y, radius in MAPS[map_name]:
        ax.add_patch(Circle(
            (x, y), radius, facecolor=OBSTACLE_COLOR, edgecolor="black",
            linewidth=1.0, zorder=3,
        ))
    ax.add_patch(Rectangle(
        (ARENA_MIN, ARENA_MIN), ARENA_MAX - ARENA_MIN, ARENA_MAX - ARENA_MIN,
        fill=False, edgecolor="black", linewidth=2.0, zorder=4,
    ))
    ax.set_xlim(ARENA_MIN - 0.15, ARENA_MAX + 0.15)
    ax.set_ylim(ARENA_MIN - 0.15, ARENA_MAX + 0.15)
    ax.set_aspect("equal", adjustable="box")
    ticks = (-5, 0, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(labelsize=9, length=3)
    ax.grid(False)
    ax.set_xlabel("x [m]", fontsize=11)
    if show_ylabel:
        ax.set_ylabel("y [m]", fontsize=11)
    else:
        ax.tick_params(labelleft=False)
    ax.set_title(TITLES[map_name], fontsize=13, pad=8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="logs/evaluation/figures/clustered_v2_preview",
        help="Output path without an extension.",
    )
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"), default=("png", "pdf"))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    figure, axes = plt.subplots(1, 2, figsize=(8.4, 4.6), constrained_layout=False)
    draw_map(axes[0], "clustered", show_ylabel=True)
    draw_map(axes[1], "clustered_v2", show_ylabel=False)

    figure.legend(
        handles=(
            Patch(facecolor=START_COLOR, edgecolor=START_COLOR, alpha=0.32, label="Initial region"),
            Patch(facecolor=GOAL_COLOR, edgecolor=GOAL_COLOR, alpha=0.32, label="Goal region"),
            Patch(facecolor=OBSTACLE_COLOR, edgecolor="black", label="Obstacle"),
        ),
        loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.01), fontsize=10,
    )
    figure.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.22, wspace=0.12)

    output_base = Path(args.output).expanduser()
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for extension in args.formats:
        output_path = output_base.with_suffix(f".{extension}")
        figure.savefig(output_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved: {output_path}")
    plt.close(figure)


if __name__ == "__main__":
    main()
